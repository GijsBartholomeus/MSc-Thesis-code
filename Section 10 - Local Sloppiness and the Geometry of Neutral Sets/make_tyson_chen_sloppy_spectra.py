from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nnse_sloppy_subspace import TYSON_CANONICAL, compute_tyson_hessian
from plot_chen_hessian_direction_phenotypes import numerical_eigenvalue_cutoff


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "figures" / "sloppy_geometry"
CHEN_HESSIAN = ROOT / "results_summaries" / "chen_hessian" / "chen2004_wt_sensitivity_hessian.npz"


def positive_modes(eigvals: np.ndarray) -> tuple[np.ndarray, int]:
    eigvals = np.asarray(eigvals, dtype=float)
    cutoff = numerical_eigenvalue_cutoff(eigvals)
    keep = eigvals > cutoff
    return eigvals[keep], int(np.sum(~keep))


def plot_sloppy_spectrum(ax, eigvals, label=None, color="steelblue"):
    """Plot Hessian/FIM eigenvalues as a vertical semilog-y spectrum ladder."""
    vals, omitted = positive_modes(np.asarray(eigvals, dtype=float))
    if vals.size == 0:
        raise ValueError("No positive eigenvalues above numerical cutoff")

    vals = vals / np.max(vals)
    position = getattr(ax, "_sloppy_spectrum_next_position", 1)
    setattr(ax, "_sloppy_spectrum_next_position", position + 1)

    half_width = 0.13 if vals.size < 20 else 0.10
    ax.vlines(position, np.min(vals), 1.0, color="#b8b8b8", linewidth=0.8, zorder=1)
    ax.hlines(
        vals,
        position - half_width,
        position + half_width,
        color=color,
        linewidth=1.05,
        alpha=0.95,
        zorder=2,
    )

    ax.set_yscale("log")
    ax.set_ylabel("normalized eigenvalue", fontsize=9)
    ax.grid(True, which="major", axis="y", color="#d2d6da", linewidth=0.65, alpha=0.75)
    ax.grid(True, which="minor", axis="y", color="#e6e8eb", linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, width=0.8)
    ax.set_xticks(list(range(1, position + 1)))
    labels = [tick.get_text() for tick in ax.get_xticklabels()[:-1]] + ([label] if label else [""])
    ax.set_xticklabels(labels, fontsize=8)
    return vals, omitted


def tyson_eigenvalues() -> np.ndarray:
    names = [
        "k6",
        "k8_minusP",
        "k9",
        "k3_CT",
        "k1_aa_over_CT",
        "k7",
        "k4",
        "k4prime",
    ]
    vector = np.array([TYSON_CANONICAL[name] for name in names], dtype=float)
    return np.asarray(compute_tyson_hessian(vector, names, t_end=100.0, n_time=501)["eigvals"], dtype=float)


def chen_eigenvalues() -> np.ndarray:
    data = np.load(CHEN_HESSIAN, allow_pickle=True)
    return np.asarray(data["eigvals_desc"], dtype=float)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tyson = tyson_eigenvalues()
    chen = chen_eigenvalues()

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    plotted_tyson, omitted_tyson = plot_sloppy_spectrum(
        ax,
        tyson,
        label="Tyson\n1991",
        color="#4C78A8",
    )
    plotted_chen, omitted_chen = plot_sloppy_spectrum(ax, chen, label="Chen\n2004", color="#9C6BC7")

    ymin = min(np.min(plotted_tyson), np.min(plotted_chen))
    ax.set_ylim(min(ymin / 2.0, 7e-9), 1.5)
    ax.set_xlim(0.45, 2.55)

    fig.tight_layout()
    png = OUTDIR / "fig_sloppy_spectra_tyson_chen.png"
    pdf = OUTDIR / "fig_sloppy_spectra_tyson_chen.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
