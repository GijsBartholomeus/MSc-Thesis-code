from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import roadrunner

from wsbw_pipeline import SPECS, ModelSpec, prepare_models


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parent
NNSE_RESULTS = ROOT / "results" / "nnse"
NNSE_RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class NNSEConfig:
    model: str
    steps: int = 6000
    n_bins: int = 50
    sigma: float = 0.01
    k_initial: int = 1
    target_empty: int = 1
    extra_steps: int = 1000
    bin_min: float = 1e-2
    bin_max: float = 250.0
    bin_top: float = 1000.0
    spacing: str = "linear"
    seed: int = 42
    neutral_threshold: float | None = None


def get_spec(key: str) -> ModelSpec:
    for spec in SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} hr"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} sec"


def make_thresholds(config: NNSEConfig) -> np.ndarray:
    if config.spacing == "log":
        base = np.logspace(np.log10(config.bin_min), np.log10(config.bin_max), config.n_bins)
    elif config.spacing == "linear":
        base = np.linspace(config.bin_min, config.bin_max, config.n_bins)
    else:
        raise ValueError(f"Unknown spacing: {config.spacing}")
    return np.append(base, config.bin_top)


def setup_rr(spec: ModelSpec, sbml: str, params: list[str]) -> tuple[roadrunner.RoadRunner, dict[str, float], dict[str, float]]:
    rr = roadrunner.RoadRunner(sbml)
    if spec.setup:
        spec.setup(rr)
    defaults = {pid: float(rr.getValue(pid)) for pid in params}
    initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            initials[sid] = float(rr.getValue(sid))
    return rr, defaults, initials


def reset_model(rr: roadrunner.RoadRunner, spec: ModelSpec, defaults: dict[str, float], initials: dict[str, float]) -> None:
    for sid, val in initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)


def set_vector(rr: roadrunner.RoadRunner, params: list[str], vector: np.ndarray) -> None:
    for pid, val in zip(params, vector):
        rr.setValue(pid, float(val))


def simulate_output(
    rr: roadrunner.RoadRunner,
    spec: ModelSpec,
    defaults: dict[str, float],
    initials: dict[str, float],
    params: list[str],
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        reset_model(rr, spec, defaults, initials)
        set_vector(rr, params, vector)
        if spec.warmup:
            spec.warmup(rr)
        rr.selections = ["time", spec.output]
        result = rr.simulate(0, spec.t_end, spec.npoints)
    except Exception:
        return None
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    if not np.any(mask):
        return None
    return t[mask], y[mask]


def objective_factory(spec: ModelSpec, audit_item: dict):
    params = audit_item["free_parameters"]
    rr, defaults, initials = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.array([defaults[pid] for pid in params], dtype=float)
    ref = simulate_output(rr, spec, defaults, initials, params, p0)
    if ref is None:
        raise RuntimeError(f"Wildtype reference simulation failed for {spec.key}")
    t_ref, y_ref = ref

    def objective(vector: np.ndarray) -> float:
        out = simulate_output(rr, spec, defaults, initials, params, vector)
        if out is None:
            return float("inf")
        t, y = out
        y0 = np.interp(t, t_ref, y_ref)
        return float(np.trapz((y - y0) ** 2, t))

    return objective, p0, params, {"time": t_ref, "signal": y_ref}


def place_in_bin(value: float, thresholds: np.ndarray) -> int | None:
    if not math.isfinite(value):
        return None
    for idx, threshold in enumerate(thresholds):
        if value <= threshold:
            return idx
    return None


def fill_initial_population(
    objective,
    p0: np.ndarray,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    k_initial: int,
    max_attempts: int = 10000,
) -> tuple[list[np.ndarray | None], list[float | None]]:
    n = len(thresholds)
    xs: list[np.ndarray | None] = [None] * n
    fs: list[float | None] = [None] * n
    filled = 0
    attempts = 0
    target = max(1, min(k_initial, n))

    while filled < target and attempts < max_attempts:
        attempts += 1
        candidate = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
        value = objective(candidate)
        bin_idx = place_in_bin(value, thresholds)
        if bin_idx is None:
            continue
        for pos in range(bin_idx, n):
            if xs[pos] is None:
                xs[pos] = candidate
                fs[pos] = value
                filled += 1
                break
    if filled == 0:
        raise RuntimeError("NNSE initialization failed to place any samples")
    return xs, fs


def nnse_step(
    xs: list[np.ndarray | None],
    fs: list[float | None],
    objective,
    p0: np.ndarray,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> tuple[list[np.ndarray | None], list[float | None], list[tuple[int, int]]]:
    n = len(xs)
    proposed_xs: list[np.ndarray | None] = [None] * n
    proposed_fs: list[float | None] = [None] * n

    for idx, (x, fx) in enumerate(zip(xs, fs)):
        if x is None or fx is None:
            continue
        u = x / (2.0 * p0)
        u_mut = (u + rng.normal(0.0, sigma, size=len(p0))) % 1.0
        x_mut = 2.0 * p0 * u_mut
        fx_mut = objective(x_mut)
        if fx_mut <= thresholds[idx]:
            proposed_xs[idx] = x_mut
            proposed_fs[idx] = fx_mut
        else:
            proposed_xs[idx] = x.copy()
            proposed_fs[idx] = fx

    xs_out = proposed_xs.copy()
    fs_out = proposed_fs.copy()
    empty_before = {idx for idx in range(n) if xs_out[idx] is None or fs_out[idx] is None}
    swaps: list[tuple[int, int]] = []

    for idx in range(n - 1, 0, -1):
        if fs_out[idx] is None:
            continue
        if fs_out[idx] <= thresholds[idx - 1]:
            xs_out[idx], xs_out[idx - 1] = xs_out[idx - 1], xs_out[idx]
            fs_out[idx], fs_out[idx - 1] = fs_out[idx - 1], fs_out[idx]
            swaps.append((idx, idx - 1))

    empty_after = {idx for idx in range(n) if xs_out[idx] is None or fs_out[idx] is None}
    for empty_pos in sorted(empty_after - empty_before):
        for _ in range(1000):
            candidate = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
            value = objective(candidate)
            bin_idx = place_in_bin(value, thresholds)
            if bin_idx is None:
                continue
            for pos in range(bin_idx, n):
                if xs_out[pos] is None or fs_out[pos] is None or value < fs_out[pos]:
                    xs_out[pos] = candidate
                    fs_out[pos] = value
                    break
            break

    return xs_out, fs_out, swaps


def run_nnse(config: NNSEConfig) -> Path:
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    audit = prepare_models()
    spec = get_spec(config.model)
    objective, p0, params, reference = objective_factory(spec, audit[spec.key])
    thresholds = make_thresholds(config)
    if config.neutral_threshold is None:
        config.neutral_threshold = float(thresholds[0])

    xs, fs = fill_initial_population(objective, p0, thresholds, rng, config.k_initial)
    all_best = []
    neutral_points = []
    neutral_values = []
    swap_count = np.zeros(len(thresholds), dtype=int)
    opportunity_count = np.zeros(len(thresholds), dtype=int)
    reached_target_at = None
    start = time.time()
    iterator = range(config.steps)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"NNSE {spec.key}", unit="step")

    for step in iterator:
        xs, fs, swaps = nnse_step(xs, fs, objective, p0, thresholds, rng, config.sigma)
        filled = sum(x is not None for x in xs)
        empty = len(xs) - filled
        finite = [fx for fx in fs if fx is not None and math.isfinite(fx)]
        best = min(finite) if finite else float("inf")
        all_best.append(best)

        swap_set = set(swaps)
        for idx in range(1, len(fs)):
            if fs[idx] is not None and fs[idx - 1] is not None:
                opportunity_count[idx] += 1
                if (idx, idx - 1) in swap_set:
                    swap_count[idx] += 1

        for x, fx in zip(xs, fs):
            if x is not None and fx is not None and fx <= config.neutral_threshold:
                neutral_points.append(x.copy())
                neutral_values.append(float(fx))

        if reached_target_at is None and empty <= config.target_empty:
            reached_target_at = step
        if reached_target_at is not None and step >= reached_target_at + config.extra_steps:
            break

        if step == 0 or (step + 1) % max(1, config.steps // 20) == 0:
            elapsed = time.time() - start
            rate = (step + 1) / elapsed if elapsed else 0.0
            print(
                f"{spec.key} step {step + 1:,}: filled={filled}/{len(xs)}, "
                f"best={best:.3e}, neutral_records={len(neutral_points):,}, "
                f"elapsed={format_duration(elapsed)}, rate={rate:.1f} step/s",
                flush=True,
            )

    elapsed = time.time() - start
    final_x = np.array([x if x is not None else np.full_like(p0, np.nan) for x in xs])
    final_f = np.array([fx if fx is not None else np.nan for fx in fs], dtype=float)
    neutral = np.unique(np.asarray(neutral_points, dtype=float), axis=0) if neutral_points else np.empty((0, len(p0)))
    neutral_f = np.asarray(neutral_values, dtype=float)
    volume_ratios = np.divide(
        swap_count,
        opportunity_count,
        out=np.full_like(swap_count, np.nan, dtype=float),
        where=opportunity_count > 0,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = NNSE_RESULTS / f"{spec.key}_nnse_{stamp}.npz"
    np.savez_compressed(
        out,
        neutral_points=neutral,
        neutral_objective_values=neutral_f,
        final_population=final_x,
        final_objective_values=final_f,
        p0=p0,
        parameter_names=np.array(params, dtype=object),
        bin_thresholds=thresholds,
        volume_ratios=volume_ratios,
        swap_count=swap_count,
        opportunity_count=opportunity_count,
        best_history=np.asarray(all_best, dtype=float),
        reference_time=reference["time"],
        reference_signal=reference["signal"],
    )
    summary = {
        "config": asdict(config),
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "parameter_count": len(params),
        "steps_completed": len(all_best),
        "reached_target_at": reached_target_at,
        "elapsed_seconds": elapsed,
        "neutral_points": int(len(neutral)),
        "final_filled": int(np.sum(~np.isnan(final_f))),
        "best_objective": float(np.nanmin(final_f)),
        "npz": str(out),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    print(f"Saved {summary_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NNSE neutral-set sampler for WSBW models")
    parser.add_argument("--model", default="chen2004", choices=[spec.key for spec in SPECS] + ["all"])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--k-initial", type=int, default=1)
    parser.add_argument("--target-empty", type=int, default=1)
    parser.add_argument("--extra-steps", type=int, default=1000)
    parser.add_argument("--bin-min", type=float, default=1e-2)
    parser.add_argument("--bin-max", type=float, default=250.0)
    parser.add_argument("--bin-top", type=float, default=1000.0)
    parser.add_argument("--spacing", choices=["linear", "log"], default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neutral-threshold", type=float, default=None)
    args = parser.parse_args()
    if args.model == "all":
        for spec in SPECS:
            model_args = vars(args).copy()
            model_args["model"] = spec.key
            run_nnse(NNSEConfig(**model_args))
    else:
        run_nnse(NNSEConfig(**vars(args)))


if __name__ == "__main__":
    main()
