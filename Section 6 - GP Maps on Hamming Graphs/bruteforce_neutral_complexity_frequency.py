#!/usr/bin/env python3
"""Plot phenotype complexity frequencies within a WT-distance cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wsbw_pipeline import RESULTS, SPECS


STATS_ROOT = RESULTS / "bruteforce_cloud_stats"


def model_label(model: str) -> str:
    for spec in SPECS:
        if spec.key == model:
            return spec.label
    return model


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def main(args: argparse.Namespace) -> None:
    stats_dir = STATS_ROOT / args.tag
    sample_npz = stats_dir / f"{args.model}_bruteforce_samples_{args.tag}.npz"
    if not sample_npz.exists():
        raise FileNotFoundError(sample_npz)

    data = np.load(sample_npz, allow_pickle=True)
    if "neutral_complexities" in data and len(data["neutral_complexities"]):
        neutral_complexities = np.asarray(data["neutral_complexities"], dtype=float)
        source = "saved neutral_complexities"
        points_in_sample = int(len(neutral_complexities))
    elif "all_complexities" in data and "all_objectives" in data:
        complexities = np.asarray(data["all_complexities"], dtype=float)
        objectives = np.asarray(data["all_objectives"], dtype=float)
        mask = np.isfinite(complexities) & np.isfinite(objectives) & (objectives <= args.neutral_cutoff)
        neutral_complexities = complexities[mask]
        source = "filtered all_complexities by all_objectives"
        points_in_sample = int(len(complexities))
    else:
        raise KeyError("Expected neutral_complexities or all_complexities/all_objectives in sample NPZ")

    if len(neutral_complexities) == 0:
        counts = np.empty(0, dtype=int)
        bins = np.empty(0, dtype=int)
    else:
        bins, counts = np.unique(np.rint(neutral_complexities).astype(int), return_counts=True)

    out_dir = stats_dir / "neutral_complexity_frequency"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"f{args.neutral_cutoff:g}".replace(".", "p")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if len(bins):
        ax.bar(bins, counts, width=0.8, color="#4c78a8", edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No sampled points passed cutoff", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("phenotype complexity K(x), rounded to nearest integer")
    ax.set_ylabel(f"sampled count with f <= {args.neutral_cutoff:g}")
    ax.set_title(f"{model_label(args.model)} WT-distance filtered complexity frequency")

    plot_path = out_dir / f"{args.model}_neutral_complexity_frequency_{args.tag}_{suffix}.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary = {
        "model": args.model,
        "label": model_label(args.model),
        "tag": args.tag,
        "sample_npz": str(sample_npz),
        "neutral_cutoff": float(args.neutral_cutoff),
        "complexity_source": source,
        "points_in_sample": points_in_sample,
        "points_passing_cutoff": int(len(neutral_complexities)),
        "complexity_stats": summarize(neutral_complexities),
        "counts_by_rounded_complexity": {str(int(k)): int(v) for k, v in zip(bins, counts)},
        "plot": str(plot_path),
    }
    json_path = out_dir / f"{args.model}_neutral_complexity_frequency_{args.tag}_{suffix}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {plot_path}")
    print(f"Saved {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--neutral-cutoff", type=float, required=True)
    main(parser.parse_args())
