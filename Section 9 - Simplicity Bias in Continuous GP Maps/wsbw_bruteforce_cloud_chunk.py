from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_nnse import get_spec, reset_model, set_vector, setup_rr
from wsbw_pipeline import DIVERGENCE_CAP_FACTOR, RESULTS, SPECS, clz, encode_signal, prepare_models


OUT_ROOT = RESULTS / "bruteforce_cloud"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CHUNK_SEED_STRIDE = 10_000_019
_WORKER: dict[str, Any] = {}


def chunk_seed_offset(chunk_id: str | None) -> int:
    if chunk_id is None:
        return 0
    try:
        return int(chunk_id) * CHUNK_SEED_STRIDE
    except ValueError:
        return sum((idx + 1) * ord(char) for idx, char in enumerate(chunk_id)) * CHUNK_SEED_STRIDE


@contextlib.contextmanager
def suppress_solver_stderr():
    saved_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)


def bits_to_uint64(bits: str) -> int:
    return int(bits, 2) if bits else 0


def uint64_to_bits(code: int, length: int = 49) -> str:
    return format(int(code), f"0{length}b")


def _init_worker(model_key: str, audit_item: dict) -> None:
    spec = get_spec(model_key)
    params = audit_item["free_parameters"]
    rr, defaults, initials = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=float)

    reset_model(rr, spec, defaults, initials)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    result = rr.simulate(0, spec.t_end, spec.npoints)
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    if not np.any(mask):
        raise RuntimeError(f"Wildtype time mask is empty for {spec.label}")
    divergence_cap = DIVERGENCE_CAP_FACTOR * max(float(np.nanmax(np.abs(y))), 1e-12)
    wt_bits = encode_signal(t[mask], y[mask], 50)

    _WORKER.clear()
    _WORKER.update(
        {
            "spec": spec,
            "params": params,
            "rr": rr,
            "defaults": defaults,
            "initials": initials,
            "p0": p0,
            "ref_time": t[mask],
            "ref_signal": y[mask],
            "divergence_cap": divergence_cap,
            "wt_bits": wt_bits,
        }
    )


def _simulate_vector(vector: np.ndarray) -> tuple[int, float, float] | None:
    spec = _WORKER["spec"]
    rr = _WORKER["rr"]
    reset_model(rr, spec, _WORKER["defaults"], _WORKER["initials"])
    set_vector(rr, _WORKER["params"], vector)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    result = rr.simulate(0, spec.t_end, spec.npoints)
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    if np.any(np.abs(y) > _WORKER["divergence_cap"]):
        return None
    if np.any(np.abs(y) > 1e9):
        return None
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    if not np.any(mask):
        return None
    tm = t[mask]
    ym = y[mask]
    bits = encode_signal(tm, ym, 50)
    ref = np.interp(tm, _WORKER["ref_time"], _WORKER["ref_signal"])
    objective = float(np.trapz((ym - ref) ** 2, tm))
    return bits_to_uint64(bits), float(clz(bits)), objective


def _evaluate_batch(task: tuple[int, int, int]) -> dict:
    batch_id, count, seed = task
    rng = np.random.default_rng(seed)
    p0 = _WORKER["p0"]
    points: list[np.ndarray] = []
    codes: list[int] = []
    complexities: list[float] = []
    objectives: list[float] = []
    failures = 0

    with suppress_solver_stderr(), contextlib.redirect_stderr(io.StringIO()):
        for _ in range(count):
            vector = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
            try:
                result = _simulate_vector(vector)
            except Exception:
                result = None
            if result is None:
                failures += 1
                continue
            code, complexity, objective = result
            points.append(vector.astype(np.float32))
            codes.append(code)
            complexities.append(complexity)
            objectives.append(objective)

    n_params = len(p0)
    return {
        "batch_id": batch_id,
        "attempted": count,
        "failures": failures,
        "points": np.vstack(points).astype(np.float32) if points else np.empty((0, n_params), dtype=np.float32),
        "phenotype_codes": np.asarray(codes, dtype=np.uint64),
        "complexities": np.asarray(complexities, dtype=np.float32),
        "objectives": np.asarray(objectives, dtype=np.float32),
    }


def chunk_sizes(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, rem = divmod(total, chunks)
    return [base + (idx < rem) for idx in range(chunks)]


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} hr"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} sec"


def sample_label(samples: int) -> str:
    exponent = round(math.log10(samples)) if samples > 0 else 0
    if samples > 0 and 10**exponent == samples:
        return f"1e{exponent}"
    return str(samples)


def run_chunk(args: argparse.Namespace) -> Path:
    start = time.time()
    audit = prepare_models()
    spec = get_spec(args.model)
    audit_item = audit[spec.key]
    params = audit_item["free_parameters"]
    workers = max(1, args.workers)
    task_count = max(workers, math.ceil(args.samples / args.batch_size))
    sizes = chunk_sizes(args.samples, task_count)
    effective_seed = args.seed + chunk_seed_offset(args.chunk_id)
    tasks = [(idx, size, effective_seed + idx * 100_003) for idx, size in enumerate(sizes)]

    print(
        f"Sampling {args.samples:,} random points for {spec.label} with {workers} workers "
        f"across {len(tasks)} batches (seed {effective_seed})",
        flush=True,
    )

    points: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    complexities: list[np.ndarray] = []
    objectives: list[np.ndarray] = []
    failures = 0
    successes = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(spec.key, audit_item)) as executor:
        futures = [executor.submit(_evaluate_batch, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            failures += int(result["failures"])
            successes += len(result["objectives"])
            completed += int(result["attempted"])
            points.append(result["points"])
            codes.append(result["phenotype_codes"])
            complexities.append(result["complexities"])
            objectives.append(result["objectives"])
            if done == 1 or done == len(futures) or done % max(1, len(futures) // 20) == 0:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"  batches {done}/{len(futures)}; elapsed={format_duration(elapsed)}; "
                    f"rate={rate:.1f} points/s; successes={successes:,}; failures={failures:,}",
                    flush=True,
                )

    all_points = np.vstack(points).astype(np.float32) if points else np.empty((0, len(params)), dtype=np.float32)
    all_codes = np.concatenate(codes).astype(np.uint64) if codes else np.empty(0, dtype=np.uint64)
    all_complexities = np.concatenate(complexities).astype(np.float32) if complexities else np.empty(0, dtype=np.float32)
    all_objectives = np.concatenate(objectives).astype(np.float32) if objectives else np.empty(0, dtype=np.float32)

    tag = args.tag or f"{spec.key}_bruteforce"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_suffix = f"_chunk-{args.chunk_id}" if args.chunk_id is not None else ""
    out = out_dir / f"{spec.key}_bruteforce_cloud_N={sample_label(args.samples)}{chunk_suffix}.npz"

    # Repeat a lightweight WT setup in the parent only for metadata.
    rr, defaults, _ = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=np.float32)
    wt_bits = _WORKER.get("wt_bits", None)
    if wt_bits is None:
        # Worker state is process-local; compute WT encoding cheaply in a temporary worker-style setup.
        _init_worker(spec.key, audit_item)
        wt_bits = _WORKER["wt_bits"]

    np.savez_compressed(
        out,
        points=all_points,
        phenotype_codes=all_codes,
        complexities=all_complexities,
        objectives=all_objectives,
        p0=p0,
        parameter_names=np.asarray(params, dtype=object),
        wildtype_code=np.asarray([bits_to_uint64(wt_bits)], dtype=np.uint64),
        wildtype_complexity=np.asarray([clz(wt_bits)], dtype=np.float32),
        samples_attempted=np.asarray([args.samples], dtype=np.int64),
        successes=np.asarray([len(all_objectives)], dtype=np.int64),
        failures=np.asarray([failures], dtype=np.int64),
    )

    elapsed = time.time() - start
    summary = {
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "tag": tag,
        "chunk_id": args.chunk_id,
        "seed": effective_seed,
        "base_seed": args.seed,
        "chunk_seed_offset": chunk_seed_offset(args.chunk_id),
        "samples_attempted": args.samples,
        "successes": int(len(all_objectives)),
        "failures": int(failures),
        "success_fraction": int(len(all_objectives)) / max(1, args.samples),
        "parameter_count": len(params),
        "wildtype_code": int(bits_to_uint64(wt_bits)),
        "wildtype_complexity": float(clz(wt_bits)),
        "elapsed_seconds": elapsed,
        "points_per_second": args.samples / elapsed if elapsed else None,
        "npz": str(out),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(
        f"Done in {format_duration(elapsed)}: successes={len(all_objectives):,}/{args.samples:,}, "
        f"failures={failures:,}",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydra chunk for brute-force point/complexity/objective clouds")
    parser.add_argument("--model", default="tyson1991", choices=[spec.key for spec in SPECS])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    run_chunk(parser.parse_args())


if __name__ == "__main__":
    main()
