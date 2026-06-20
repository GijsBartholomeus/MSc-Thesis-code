#!/usr/bin/env python3
"""Plot Tyson brute-force cloud in the WT Hessian eigenvector basis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bruteforce_cloud_locality import PALETTE, quartile_labels
from nnse_sloppy_subspace import compute_tyson_hessian
from wsbw_nnse import get_spec
from wsbw_pipeline import prepare_models


STATS_ROOT = ROOT / "results" / "bruteforce_cloud_stats"
FIGURE_ROOT = ROOT / "figures" / "sloppy_geometry"
SUMMARY_ROOT = ROOT / "results_summaries" / "sloppy_geometry"


def tyson_parameter_names() -> list[str]:
    audit = prepare_models()
    return list(audit[get_spec("tyson1991").key]["free_parameters"])


def load_tyson_sample(tag: str, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    path = STATS_ROOT / tag / f"tyson1991_bruteforce_samples_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["all_points"], dtype=float)
        complexities = np.asarray(data["all_complexities"], dtype=float)
        p0 = np.asarray(data["p0"], dtype=float)
        parameter_names = [str(x) for x in data["parameter_names"]] if "parameter_names" in data.files else tyson_parameter_names()

    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(complexities) & np.all(points > 0, axis=1)
    points = points[finite]
    complexities = complexities[finite]
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        complexities = complexities[idx]
    meta = {"sample_npz": str(path), "finite_positive_points_loaded": int(np.sum(finite)), "points_plotted": int(len(points))}
    return points, complexities, p0, parameter_names, meta


def orient_eigenvectors(eigvecs: np.ndarray) -> np.ndarray:
    out = np.array(eigvecs, dtype=float, copy=True)
    for col in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, col])))
        if out[pivot, col] < 0:
            out[:, col] *= -1.0
    return out


def top_loadings(eigvecs: np.ndarray, names: list[str], axis: int, n: int = 3) -> str:
    vec = eigvecs[:, axis]
    order = np.argsort(np.abs(vec))[::-1][:n]
    return ", ".join(f"{names[i]} {vec[i]:+.2f}" for i in order)


def transform_to_hessian_basis(points: np.ndarray, p0: np.ndarray, parameter_names: list[str]) -> tuple[np.ndarray, dict]:
    hess = compute_tyson_hessian(p0, parameter_names)
    supported_idx = hess["supported_indices"]
    mapped_names = hess["mapped_names"]
    eigvals = np.asarray(hess["eigvals"], dtype=float)
    eigvecs = orient_eigenvectors(np.asarray(hess["eigvecs"], dtype=float))
    log_delta = np.log(points[:, supported_idx]) - np.log(p0[supported_idx])[None, :]
    coords = log_delta @ eigvecs
    meta = {
        "supported_parameter_names": mapped_names,
        "unsupported_parameter_names": hess["unsupported_names"],
        "hessian_eigenvalues_descending_stiff_to_sloppy": eigvals.tolist(),
        "axis_loadings_top3": {
            f"hessian_axis_{idx + 1}": top_loadings(eigvecs, mapped_names, idx)
            for idx in range(eigvecs.shape[1])
        },
        "axis_order": "descending Hessian eigenvalue: axis 1 stiffest, final axis sloppiest",
    }
    return coords, meta


def plot_pairgrid(
    coords: np.ndarray,
    labels: np.ndarray,
    eigvals: list[float],
    loadings: dict[str, str],
    out_path: Path,
    tag: str,
) -> None:
    dim = coords.shape[1]
    fig, axes = plt.subplots(dim, dim, figsize=(2.05 * dim, 2.05 * dim), constrained_layout=True)
    axes = np.asarray(axes)
    axis_labels = [f"h{i + 1}" for i in range(dim)]
    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]
            if row == col:
                for lab in range(4):
                    vals = coords[labels == lab, col]
                    ax.hist(vals, bins=45, density=True, color=PALETTE[lab], alpha=0.35, linewidth=0)
                ax.set_yticks([])
            elif row > col:
                ax.scatter(coords[:, col], coords[:, row], c=PALETTE[labels], s=2, alpha=0.24, linewidths=0)
            else:
                ax.axis("off")
                continue

            if row == dim - 1:
                ax.set_xlabel(axis_labels[col], fontsize=8)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(axis_labels[row], fontsize=8)
            else:
                ax.set_yticks([])
            if row == 0 and col == 0:
                ax.set_title("axis histograms", fontsize=8)
            elif row == col:
                eig = eigvals[col]
                ax.set_title(f"{axis_labels[col]}\nλ={eig:.2g}", fontsize=7)

    fig.suptitle(
        f"Tyson brute-force cloud in WT Hessian eigenbasis\n"
        f"{tag}; colors = phenotype-complexity quartiles; h1 stiffest -> h{dim} sloppiest",
        fontsize=14,
    )
    # Add a compact loadings note outside the grid.
    note = "\n".join(f"h{idx + 1}: {loadings[f'hessian_axis_{idx + 1}']}" for idx in range(dim))
    fig.text(0.995, 0.01, note, ha="right", va="bottom", fontsize=7, family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    points, complexities, p0, parameter_names, load_meta = load_tyson_sample(args.tag, args.max_points, args.seed)
    labels, edges = quartile_labels(complexities)
    coords, hessian_meta = transform_to_hessian_basis(points, p0, parameter_names)

    result = {
        "model": "tyson1991",
        "tag": args.tag,
        "seed": int(args.seed),
        "sample": load_meta,
        "complexity_quartile_edges": [float(x) for x in edges],
        "coordinate_system": "log(theta)-log(theta_WT), rotated into WT Hessian eigenvectors",
        **hessian_meta,
    }
    figure_dir = FIGURE_ROOT / "tyson_hessian_eigenpair_grid"
    summary_dir = SUMMARY_ROOT / "tyson_hessian_eigenpair_grid"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figure_dir / f"tyson1991_hessian_eigenpair_grid_{args.tag}.png"
    json_path = summary_dir / f"tyson1991_hessian_eigenpair_grid_{args.tag}.json"
    plot_pairgrid(
        coords,
        labels,
        result["hessian_eigenvalues_descending_stiff_to_sloppy"],
        result["axis_loadings_top3"],
        fig_path,
        args.tag,
    )
    json_path.write_text(json.dumps(result, indent=2))
    return json_path, fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="tyson_bfc_1e8")
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    json_path, fig_path = run(args)
    print(f"Saved {json_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
