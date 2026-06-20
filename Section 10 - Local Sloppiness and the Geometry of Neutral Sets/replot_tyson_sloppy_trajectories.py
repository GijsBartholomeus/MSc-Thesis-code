#!/usr/bin/env python3
"""Restyle saved Tyson sloppy walks without recomputing Hessians."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from make_tyson_sloppy_multiwalk_figures import PARAM_LABELS, PARAM_NAMES


WALK_CMAP = plt.get_cmap("coolwarm")


def padded_limits(values: np.ndarray, fraction: float = 0.07) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = max(fraction * (hi - lo), 0.025)
    return lo - pad, hi + pad


def style_3d_axis(ax) -> None:
    ax.grid(True, color="#c8ccd1", alpha=0.34, linewidth=0.45)
    ax.tick_params(labelsize=7, pad=-1)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor("#d4d7db")
        pane.set_alpha(0.08)


def draw_progressive_path(ax, coords: np.ndarray, norm: Normalize) -> None:
    steps = np.arange(len(coords))
    for j in range(len(coords) - 1):
        ax.plot(
            coords[j : j + 2, 0],
            coords[j : j + 2, 1],
            coords[j : j + 2, 2],
            color=WALK_CMAP(norm(j + 0.5)),
            linewidth=1.85,
            alpha=0.98,
            solid_capstyle="round",
        )
    sample = np.unique(np.r_[np.arange(0, len(coords), 8), len(coords) - 1])
    ax.scatter(
        coords[sample, 0], coords[sample, 1], coords[sample, 2],
        c=steps[sample], cmap=WALK_CMAP, norm=norm,
        s=8, linewidths=0, alpha=0.9, depthshade=False,
    )
    ax.scatter(*coords[0], color="#171717", s=39, edgecolor="white", linewidth=0.65, depthshade=False, zorder=10)
    ax.scatter(*coords[-1], color=WALK_CMAP(1.0), s=43, edgecolor="#171717", linewidth=0.55, depthshade=False, zorder=10)


def add_step_colorbar(fig, norm: Normalize, rect: list[float]) -> None:
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=WALK_CMAP)
    scalar.set_array([])
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    cbar.set_label("walk step", fontsize=8, labelpad=2)
    cbar.set_ticks([0, 50, 100, 150, 200, 250])
    cbar.ax.tick_params(labelsize=7, length=2.5, pad=2)
    cbar.outline.set_linewidth(0.55)


def plot_global_walks(log_walks: list[np.ndarray], outdir: Path) -> tuple[int, int, int]:
    """Plot all walks in one common, globally selected 3D coordinate system."""
    stacked = np.concatenate(log_walks, axis=0)
    top3 = tuple(np.argsort(np.ptp(stacked, axis=0))[-3:][::-1])
    all_coords = stacked[:, top3]
    norm = Normalize(0, max(len(walk) - 1 for walk in log_walks))

    fig = plt.figure(figsize=(7.15, 5.35))
    ax = fig.add_subplot(111, projection="3d")
    for mode, walk in enumerate(log_walks, start=1):
        coords = walk[:, top3]
        draw_progressive_path(ax, coords, norm)
        endpoint = coords[-1]
        ax.text(*endpoint, f"  {mode}", color="#9f2928", fontsize=8.5, weight="bold")

    wt = log_walks[0][0, top3]
    ax.text(*wt, "  WT", color="#171717", fontsize=8, weight="bold")
    names = [PARAM_NAMES[i] for i in top3]
    ax.set_xlabel(PARAM_LABELS[names[0]], fontsize=9, labelpad=5)
    ax.set_ylabel(PARAM_LABELS[names[1]], fontsize=9, labelpad=5)
    ax.set_zlabel(PARAM_LABELS[names[2]], fontsize=9, labelpad=4)
    ax.set_xlim(*padded_limits(all_coords[:, 0]))
    ax.set_ylim(*padded_limits(all_coords[:, 1]))
    ax.set_zlim(*padded_limits(all_coords[:, 2]))
    ax.set_box_aspect((1.28, 1.0, 0.92))
    ax.view_init(elev=23, azim=-57)
    style_3d_axis(ax)

    add_step_colorbar(fig, norm, [0.30, 0.055, 0.40, 0.024])
    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.13, top=0.99)
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"tyson_sloppy_walks_global_3d.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return top3


def plot_four_views(log_walks: list[np.ndarray], summary: dict, outdir: Path) -> None:
    """Detailed per-walk projections for the appendix, with one shared colorbar."""
    norm = Normalize(0, max(len(walk) - 1 for walk in log_walks))
    fig = plt.figure(figsize=(7.35, 6.65))
    for i, coords_full in enumerate(log_walks):
        displacement = np.abs(coords_full[-1] - coords_full[0])
        top3 = np.argsort(displacement)[-3:][::-1]
        coords = coords_full[:, top3]
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        draw_progressive_path(ax, coords, norm)
        names = [PARAM_NAMES[j] for j in top3]
        ax.set_xlabel(PARAM_LABELS[names[0]], fontsize=7.5, labelpad=0)
        ax.set_ylabel(PARAM_LABELS[names[1]], fontsize=7.5, labelpad=0)
        ax.set_zlabel(PARAM_LABELS[names[2]], fontsize=7.5, labelpad=0)
        ax.set_xlim(*padded_limits(coords[:, 0]))
        ax.set_ylim(*padded_limits(coords[:, 1]))
        ax.set_zlim(*padded_limits(coords[:, 2]))
        ax.set_box_aspect((1.08, 1.0, 0.92))
        ax.view_init(elev=23, azim=-56)
        style_3d_axis(ax)
        eigval = float(summary["walks"][i]["initial_eigenvalue"])
        ax.set_title(f"{chr(97 + i)}  Sloppy mode {i + 1}", fontsize=8.5, pad=1, loc="left", weight="semibold")
        ax.text2D(0.97, 0.96, rf"$\lambda_0={eigval:.2e}$", transform=ax.transAxes, fontsize=6.8, ha="right", va="top", color="#555555")

    add_step_colorbar(fig, norm, [0.31, 0.035, 0.38, 0.018])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.10, top=0.98, wspace=0.01, hspace=0.16)
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"tyson_sloppy_walks_four_views.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    log_walks = [np.asarray(data[f"walk_{i}_log_parameters"]) for i in range(len(summary["walks"]))]
    top3 = plot_global_walks(log_walks, args.outdir)
    plot_four_views(log_walks, summary, args.outdir)
    print("Global axes:", ", ".join(PARAM_NAMES[i] for i in top3))


if __name__ == "__main__":
    main()
