from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "figures" / "concept"


def smooth_signal(t: np.ndarray) -> np.ndarray:
    baseline = 0.18 * np.sin(0.65 * t - 0.5)
    oscillation = 0.88 * np.sin(1.55 * t - 0.85)
    harmonic = 0.18 * np.sin(3.3 * t + 0.45)
    envelope = 1.0 + 0.10 * np.sin(0.38 * t + 1.2)
    return 1.55 + envelope * oscillation + baseline + harmonic


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    t_dense = np.linspace(0, 14.0, 800)
    y_dense = smooth_signal(t_dense)

    q = 22
    t_sample = np.linspace(0.55, 13.45, q + 1)
    y_sample = smooth_signal(t_sample)
    bits = (np.diff(y_sample) > 0).astype(int)

    fig = plt.figure(figsize=(9.4, 4.8), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.25], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    ax_bits = fig.add_subplot(gs[1])

    green = "#2ca25f"
    red = "#de2d26"
    curve = "#252a30"
    sample = "#f4f7fb"
    edge = "#52626d"

    ax.plot(t_dense, y_dense, color=curve, lw=2.4, zorder=2)
    ax.scatter(t_sample, y_sample, s=42, facecolor=sample, edgecolor=edge, linewidth=1.2, zorder=4)

    for j, bit in enumerate(bits):
        color = green if bit else red
        ax.plot(
            [t_sample[j], t_sample[j + 1]],
            [y_sample[j], y_sample[j + 1]],
            color=color,
            lw=4.0,
            alpha=0.56,
            solid_capstyle="round",
            zorder=3,
        )
        mid_x = 0.5 * (t_sample[j] + t_sample[j + 1])
        mid_y = 0.5 * (y_sample[j] + y_sample[j + 1])
        dy = y_sample[j + 1] - y_sample[j]
        ax.annotate(
            "",
            xy=(mid_x + 0.17, mid_y + 0.17 * dy),
            xytext=(mid_x - 0.17, mid_y - 0.17 * dy),
            arrowprops={
                "arrowstyle": "-|>",
                "mutation_scale": 10,
                "lw": 1.2,
                "color": color,
                "alpha": 0.88,
            },
            zorder=5,
        )

    ax.set_xlim(0.1, 13.9)
    ax.set_ylim(y_dense.min() - 0.36, y_dense.max() + 0.36)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.linspace(0, 14, 8))
    ax.set_yticks(np.linspace(0.5, 3.0, 6))
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    ax.grid(True, color="#d6d9dd", lw=0.8, alpha=0.62)

    ax_bits.set_ylim(0, 1)
    ax_bits.set_xlim(ax.get_xlim())
    ax_bits.set_yticks([])
    ax_bits.set_xticks([])
    for spine in ax_bits.spines.values():
        spine.set_visible(False)
    ax_bits.grid(False)

    for j, bit in enumerate(bits):
        color = green if bit else red
        x0 = t_sample[j]
        width = t_sample[j + 1] - t_sample[j]
        rect = FancyBboxPatch(
            (x0 + 0.025, 0.23),
            width - 0.05,
            0.54,
            boxstyle="round,pad=0.015,rounding_size=0.045",
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.88,
        )
        ax_bits.add_patch(rect)
        ax_bits.text(
            x0 + 0.5 * width,
            0.50,
            str(bit),
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            weight="bold",
        )

    png = OUTDIR / "fig_updown_example.png"
    pdf = OUTDIR / "fig_updown_example.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
