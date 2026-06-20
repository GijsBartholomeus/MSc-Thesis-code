from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner

from wsbw_pipeline import DIVERGENCE_CAP_FACTOR, PLOTS, RESULTS, SPECS, clz, encode_signal, prepare_models


def restore_initials(rr: roadrunner.RoadRunner, base_initials: dict[str, float]) -> None:
    for sid, val in base_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()


def simulate_wildtype(spec, sbml: Path) -> dict:
    rr = roadrunner.RoadRunner(str(sbml))
    if spec.setup:
        spec.setup(rr)

    base_initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            base_initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            base_initials[sid] = float(rr.getValue(sid))

    restore_initials(rr, base_initials)
    if spec.setup:
        spec.setup(rr)
    if spec.warmup:
        spec.warmup(rr)

    rr.selections = ["time", spec.output]
    result = rr.simulate(0, spec.t_end, spec.npoints)
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    bits = encode_signal(t[mask], y[mask], 50)
    return {
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "time": t,
        "signal": y,
        "encoding": bits,
        "complexity": clz(bits),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
    }


def main() -> Path:
    audit = prepare_models()
    traces = [simulate_wildtype(spec, Path(audit[spec.key]["promoted_sbml"])) for spec in SPECS]
    existing = {}
    for trace in traces:
        path = RESULTS / f"{trace['model']}_complexity_frequency.json"
        if path.exists():
            existing[trace["model"]] = json.loads(path.read_text())

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9C755F"]
    ncols = 3
    nrows = int(math.ceil(len(traces) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    report = []
    for ax, trace, color in zip(axes, traces, colors):
        spec = next(item for item in SPECS if item.key == trace["model"])
        t = trace["time"]
        y = trace["signal"]
        mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
        t = t[mask] - spec.coarse_start
        y = y[mask]
        if trace["model"] == "kholodenko2000":
            t = t / 60.0
            xlabel = "time in phenotype window (min)"
        else:
            xlabel = "time in phenotype window"
        ax.plot(t, y, color=color, lw=1.8)
        count = existing.get(trace["model"], {}).get("wildtype_count")
        count_text = f", sampled count {count}" if count is not None else ""
        window_start = spec.coarse_start
        window_end = spec.coarse_start + spec.coarse_duration
        if trace["model"] == "kholodenko2000":
            window_text = f"window {window_start / 60.0:.1f}-{window_end / 60.0:.1f} min"
        else:
            window_text = f"window {window_start:.1f}-{window_end:.1f}"
        ax.set_title(f"{trace['label']} wildtype\n{window_text}, K={trace['complexity']:.2f}{count_text}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(trace["output"])
        ax.grid(alpha=0.25)
        report.append(
            {
                "model": trace["model"],
                "label": trace["label"],
                "output": trace["output"],
                "complexity": trace["complexity"],
                "min": trace["min"],
                "max": trace["max"],
                "window_start": float(spec.coarse_start),
                "window_end": float(spec.coarse_start + spec.coarse_duration),
                "window_duration": float(spec.coarse_duration),
                "divergence_cap_factor": DIVERGENCE_CAP_FACTOR,
                "divergence_cap": DIVERGENCE_CAP_FACTOR * max(abs(trace["min"]), abs(trace["max"]), 1e-12),
                "sampled_count_in_last_run": count,
            }
        )

    for ax in axes[len(traces) :]:
        ax.axis("off")

    out = PLOTS / "oscillatory_subset_wildtype_traces.png"
    fig.savefig(out, dpi=220)
    window_out = PLOTS / "oscillatory_subset_wildtype_trough_windows.png"
    fig.savefig(window_out, dpi=220)
    (RESULTS / "wildtype_trace_report.json").write_text(json.dumps(report, indent=2))
    print(out)
    print(window_out)
    return out


if __name__ == "__main__":
    main()
