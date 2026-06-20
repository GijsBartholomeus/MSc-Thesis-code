from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import roadrunner

from wsbw_pipeline import (
    DIVERGENCE_CAP_FACTOR,
    PLOTS,
    RESULTS,
    SPECS,
    clz,
    plot_complexity_frequency,
    prepare_models,
    sample_size_label,
    simulate_encoding,
)


def get_spec(key: str):
    for spec in SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")


def auto_workers() -> int:
    count = os.cpu_count() or 1
    return max(1, count - 1)


def chunk_sizes(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, rem = divmod(total, chunks)
    return [base + (1 if idx < rem else 0) for idx in range(chunks)]


def load_worker_state(spec_key: str, audit_item: dict):
    spec = get_spec(spec_key)
    rr = roadrunner.RoadRunner(audit_item["promoted_sbml"])
    if spec.setup:
        spec.setup(rr)
    defaults = {pid: float(rr.getValue(pid)) for pid in audit_item["free_parameters"]}
    base_initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            base_initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            base_initials[sid] = float(rr.getValue(sid))
    return spec, rr, defaults, base_initials


def wildtype_metadata(spec_key: str, audit_item: dict, seed: int) -> dict:
    spec, rr, defaults, base_initials = load_worker_state(spec_key, audit_item)
    rng = random.Random(seed)
    wt_bits = simulate_encoding(rr, spec, defaults, base_initials, wildtype=True, rng=rng)

    for sid, val in base_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    wt_result = rr.simulate(0, spec.t_end, spec.npoints)
    wt_signal = np.asarray(wt_result[:, 1], dtype=float)
    wildtype_max_abs = float(np.max(np.abs(wt_signal)))
    divergence_cap = DIVERGENCE_CAP_FACTOR * max(wildtype_max_abs, 1e-12)
    return {
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": clz(wt_bits) if wt_bits else None,
        "wildtype_max_abs": wildtype_max_abs,
        "divergence_cap": divergence_cap,
    }


def process_chunk(args: tuple[str, dict, int, int, float]) -> dict:
    spec_key, audit_item, samples, seed, divergence_cap = args
    spec, rr, defaults, base_initials = load_worker_state(spec_key, audit_item)
    rng = random.Random(seed)
    counts = Counter()
    failures = 0

    for _ in range(samples):
        try:
            bits = simulate_encoding(
                rr,
                spec,
                defaults,
                base_initials,
                wildtype=False,
                rng=rng,
                divergence_cap=divergence_cap,
            )
            if bits is None:
                failures += 1
            else:
                counts[bits] += 1
        except Exception:
            failures += 1

    return {"counts": dict(counts), "failures": failures}


def run_model_parallel(spec_key: str, audit: dict, samples: int, seed: int, workers: int, chunks_per_worker: int) -> dict:
    spec = get_spec(spec_key)
    audit_item = audit[spec.key]
    chunks = chunk_sizes(samples, max(1, workers * chunks_per_worker))
    meta = wildtype_metadata(spec.key, audit_item, seed)
    merged = Counter()
    failures = 0
    tasks = []
    offset = 0
    for idx, size in enumerate(chunks):
        tasks.append((spec.key, audit_item, size, seed + idx + offset, meta["divergence_cap"]))
        offset += 10_000

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_chunk, task) for task in tasks if task[2] > 0]
        done = 0
        for future in as_completed(futures):
            result = future.result()
            merged.update(result["counts"])
            failures += result["failures"]
            done += 1
            print(f"  {spec.label}: finished chunk {done}/{len(futures)}", flush=True)

    wt_bits = meta["wildtype_encoding"]
    data = {
        "model": spec.key,
        "label": spec.label,
        "samples": samples,
        "successes": sum(merged.values()),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": meta["wildtype_complexity"],
        "wildtype_count": merged.get(wt_bits, 0) if wt_bits else 0,
        "wildtype_max_abs": meta["wildtype_max_abs"],
        "divergence_cap_factor": DIVERGENCE_CAP_FACTOR,
        "divergence_cap": meta["divergence_cap"],
        "time_window": {
            "t_end": spec.t_end,
            "coarse_start": spec.coarse_start,
            "coarse_duration": spec.coarse_duration,
        },
        "parallel": {
            "workers": workers,
            "chunks": len(chunks),
            "chunks_per_worker": chunks_per_worker,
        },
        "phenotypes": [{"encoding": enc, "count": n, "complexity": clz(enc)} for enc, n in merged.items()],
    }
    (RESULTS / f"{spec.key}_complexity_frequency.json").write_text(json.dumps(data))
    return data


def main(
    samples: int,
    seed: int,
    workers: int | None,
    chunks_per_worker: int,
    models: list[str] | None,
    show_wildtype: bool = True,
    auto_hide_low_wildtype: bool = True,
    min_complexity: float | None = None,
    max_complexity: float | None = None,
) -> Path:
    audit = prepare_models()
    selected = [spec for spec in SPECS if models is None or spec.key in models]
    if not selected:
        raise ValueError("No models selected")
    worker_count = workers if workers is not None else auto_workers()
    print(f"Using {worker_count} worker processes", flush=True)

    all_data = []
    for idx, spec in enumerate(selected):
        print(f"Running {spec.label} with {samples} samples", flush=True)
        all_data.append(
            run_model_parallel(
                spec.key,
                audit,
                samples=samples,
                seed=seed + idx * 1_000_000,
                workers=worker_count,
                chunks_per_worker=chunks_per_worker,
            )
        )

    # Keep the publication-style combined figure only when all models are present.
    if len(all_data) == len(SPECS):
        out = plot_complexity_frequency(
            all_data,
            PLOTS / f"CompFreq{sample_size_label(samples)}.png",
            show_wildtype=show_wildtype,
            auto_hide_low_wildtype=auto_hide_low_wildtype,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )
        legacy_out = PLOTS / "oscillatory_subset_complexity_frequency_trough_windows_parallel.png"
        if out != legacy_out:
            import matplotlib.pyplot as plt

            plt.gcf().savefig(legacy_out, dpi=220)
        print(out)
        return out
    out = RESULTS / "parallel_subset_complete.txt"
    out.write_text("\n".join(data["model"] for data in all_data))
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Chico-style complexity-frequency pipeline")
    parser.add_argument("--samples", type=int, default=10000, help="Samples per model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="Worker processes. Default: cpu_count - 1")
    parser.add_argument("--chunks-per-worker", type=int, default=2)
    parser.add_argument("--hide-wildtype", action="store_true")
    parser.add_argument("--show-low-wildtype", action="store_true")
    parser.add_argument("--min-complexity", type=float, default=None)
    parser.add_argument("--max-complexity", type=float, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys, e.g. chen2004 vilar2002. Default: all models",
    )
    args = parser.parse_args()
    main(
        samples=args.samples,
        seed=args.seed,
        workers=args.workers,
        chunks_per_worker=args.chunks_per_worker,
        models=args.models,
        show_wildtype=not args.hide_wildtype,
        auto_hide_low_wildtype=not args.show_low_wildtype,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
    )
