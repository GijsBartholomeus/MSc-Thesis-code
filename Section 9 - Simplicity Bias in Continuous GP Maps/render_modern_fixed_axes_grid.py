from __future__ import annotations

import argparse
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import numpy as np


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "pipeline"

MODERN_MODELS = ["rodenfels2019", "almeida2020", "novak2022"]
ORIGINAL_MODELS = ["kholodenko2000", "vilar2002", "tyson1991", "leloup1999", "locke2005", "ueda2001"]
GRID_MODELS = MODERN_MODELS + ORIGINAL_MODELS
ROMANS = ["ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
AUTO_AXIS_MODELS = {"leloup1999", "locke2005", "ueda2001"}

COLORS = {
    "rodenfels2019": "#4C78A8",
    "almeida2020": "#F58518",
    "novak2022": "#54A24B",
    "kholodenko2000": "#4C78A8",
    "leloup1999": "#F58518",
    "locke2005": "#54A24B",
    "ueda2001": "#E45756",
    "vilar2002": "#B279A2",
    "tyson1991": "#9C755F",
}
PHENOTYPE_OBJECT_RE = re.compile(r"\{[^{}]*\}")


@dataclass
class PanelData:
    model: str
    label: str
    samples: int
    successes: int
    x_min: float = math.inf
    x_max: float = -math.inf
    y_min: float = math.inf
    y_max: float = -math.inf
    bins: dict[float, list[float]] = field(default_factory=dict)
    sample_x: list[float] = field(default_factory=list)
    sample_y: list[float] = field(default_factory=list)
    seen: int = 0
    wildtype_complexity: float | None = None
    wildtype_count: int | None = None


def original_path(model: str) -> Path:
    return RESULTS / f"{model}_complexity_frequency_hydra_1e8_no_chen_tyson_v3_merged.json"


def modern_path(model: str, sample_label: str) -> Path:
    tag = f"modern_oscillators_{sample_label}"
    return RESULTS / "freqcomp_chunks" / tag / f"{model}_freqcomp_N={sample_label}_chunk-0.json"


def meta_value(pattern: str, text: str, default: str | None = None) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def update_panel(panel: PanelData, complexity: float, count: int, rng: random.Random, sample_cap: int) -> None:
    if count <= 0 or panel.successes <= 0:
        return
    y = count / panel.successes
    x = complexity
    panel.x_min = min(panel.x_min, x)
    panel.x_max = max(panel.x_max, x)
    panel.y_min = min(panel.y_min, y)
    panel.y_max = max(panel.y_max, y)

    bx = round(x, 1)
    if bx not in panel.bins:
        panel.bins[bx] = [y, y]
    else:
        panel.bins[bx][0] = min(panel.bins[bx][0], y)
        panel.bins[bx][1] = max(panel.bins[bx][1], y)

    panel.seen += 1
    if len(panel.sample_x) < sample_cap:
        panel.sample_x.append(x)
        panel.sample_y.append(y)
    else:
        j = rng.randrange(panel.seen)
        if j < sample_cap:
            panel.sample_x[j] = x
            panel.sample_y[j] = y


def parse_phenotype_object(obj: str) -> tuple[float, int] | None:
    count_match = re.search(r'"count"\s*:\s*(\d+)', obj)
    complexity_match = re.search(r'"complexity"\s*:\s*([0-9.eE+-]+)', obj)
    if not count_match or not complexity_match:
        return None
    return float(complexity_match.group(1)), int(count_match.group(1))


def load_panel_stream(path: Path, model: str, *, sample_cap: int) -> PanelData:
    if not path.exists():
        raise FileNotFoundError(path)

    marker = '"phenotypes": ['
    header_parts: list[str] = []
    rng = random.Random(12345)
    in_array = False
    buffer = ""
    panel: PanelData | None = None

    with path.open("r", encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), ""):
            if not in_array:
                header_parts.append(chunk)
                header = "".join(header_parts)
                pos = header.find(marker)
                if pos < 0:
                    continue
                meta = header[:pos]
                label = meta_value(r'"label"\s*:\s*"([^"]+)"', meta, model) or model
                samples = int(meta_value(r'"samples"\s*:\s*(\d+)', meta, "0") or 0)
                successes = int(meta_value(r'"successes"\s*:\s*(\d+)', meta, str(samples)) or samples)
                wt_c = meta_value(r'"wildtype_complexity"\s*:\s*([0-9.eE+-]+)', meta)
                wt_n = meta_value(r'"wildtype_count"\s*:\s*(\d+)', meta)
                panel = PanelData(
                    model=model,
                    label=label,
                    samples=samples,
                    successes=successes,
                    wildtype_complexity=float(wt_c) if wt_c else None,
                    wildtype_count=int(wt_n) if wt_n else None,
                )
                chunk = header[pos + len(marker) :]
                header_parts.clear()
                in_array = True

            buffer += chunk
            last_end = 0
            for match in PHENOTYPE_OBJECT_RE.finditer(buffer):
                parsed = parse_phenotype_object(match.group(0))
                if parsed is not None and panel is not None:
                    complexity, count = parsed
                    update_panel(panel, complexity, count, rng, sample_cap)
                last_end = match.end()
            if last_end:
                buffer = buffer[last_end:]
            if "]" in buffer and "{" not in buffer:
                return panel if panel is not None else PanelData(model, model, 0, 0)

    if panel is None:
        raise ValueError(f"Could not find phenotypes array in {path}")
    return panel


def draw_panel(
    ax,
    panel: PanelData,
    color: str,
    roman: str,
    *,
    xlim: tuple[float, float],
    fixed_axes: bool,
) -> None:
    ax.scatter(panel.sample_x, panel.sample_y, s=3.0, color="black", alpha=0.45, linewidths=0, zorder=3)
    if len(panel.bins) >= 2:
        xs = np.array(sorted(panel.bins), dtype=float)
        lower = np.array([panel.bins[x][0] for x in xs], dtype=float)
        upper = np.array([panel.bins[x][1] for x in xs], dtype=float)
        ax.fill_between(xs, lower, upper, color=color, alpha=0.28, zorder=1)
        ax.plot(xs, upper, color=color, lw=1.8, zorder=2)

    if panel.wildtype_complexity is not None and panel.wildtype_count is not None and panel.successes > 0:
        wt_y = max(panel.wildtype_count, 0.5) / panel.successes
        ax.scatter(
            [panel.wildtype_complexity],
            [wt_y],
            color="red",
            edgecolor="black",
            linewidth=0.4,
            s=34,
            zorder=5,
        )

    ax.set_yscale("log")
    if fixed_axes:
        ax.set_ylim(1e-8, 1)
        ax.set_xlim(*xlim)
    else:
        ax.set_ylim(max(panel.y_min / 1.5, 1e-12), min(panel.y_max * 1.6, 1.0))
        ax.set_xlim(max(0.0, panel.x_min - 2.0), panel.x_max + 2.0)
    ax.grid(True, color="#D0D0D0", linewidth=0.35, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.03, 0.95, roman, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 3x3 modern+original FreqComp grid with fixed axes.")
    parser.add_argument("--modern-sample-label", default="1e7", choices=["1e7", "1e8"])
    parser.add_argument("--scatter-cap", type=int, default=90_000)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=None)
    args = parser.parse_args()

    paths = {
        **{model: modern_path(model, args.modern_sample_label) for model in MODERN_MODELS},
        **{model: original_path(model) for model in ORIGINAL_MODELS},
    }

    panels: list[PanelData] = []
    for model in GRID_MODELS:
        print(f"loading {model}: {paths[model]}", flush=True)
        panels.append(load_panel_stream(paths[model], model, sample_cap=args.scatter_cap))
        print(f"  phenotypes={panels[-1].seen:,} x=[{panels[-1].x_min:.2f}, {panels[-1].x_max:.2f}]", flush=True)

    x_max = args.x_max
    if x_max is None:
        x_max = float(math.ceil(max(panel.x_max for panel in panels) / 5.0) * 5.0)
    xlim = (args.x_min, x_max)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"FreqCompModern{args.modern_sample_label}_Other6_1e8_grid3x3_fixedaxes.png"
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 8.4), sharex=False, sharey=False, constrained_layout=True)
    for idx, (ax, panel, roman) in enumerate(zip(axes.flat, panels, ROMANS)):
        draw_panel(
            ax,
            panel,
            COLORS.get(panel.model, "#4C78A8"),
            roman,
            xlim=xlim,
            fixed_axes=panel.model not in AUTO_AXIS_MODELS,
        )
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        if idx % 3 == 0:
            ax.set_ylabel(r"$P(x)$")
        else:
            ax.tick_params(labelleft=False)
        if idx // 3 == 2:
            ax.set_xlabel(r"$K(x)$")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
