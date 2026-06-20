#!/usr/bin/env python3
"""Create the two thesis frequency--complexity figures from rendered panels.

The underlying frequency estimates are expensive cluster outputs.  This script
does not recompute them: it extracts the genuine Chen N=10^10 panel from the
legacy composite and copies the independently rendered 3x3 grid containing the
other nine models at N=10^8 each.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_DIR = (
    ROOT
    / "Yeast"
    / "Chen"
    / "WhySystemsBiologyWorks"
    / "figures"
    / "pipeline"
)


def save_chen_panel(source: Path, destination: Path) -> None:
    """Extract Chen from the old 1+6 composite while removing its panel letter."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        # Source is 4050 x 1740.  The crop retains the full Chen axes and labels,
        # but excludes the old composite's A/B panel headings and right-hand grid.
        sx = image.width / 4050.0
        sy = image.height / 1740.0
        box = tuple(
            round(value)
            for value in (0 * sx, 110 * sy, 1970 * sx, 1725 * sy)
        )
        panel = image.crop(box)
        destination.parent.mkdir(parents=True, exist_ok=True)
        panel.save(destination, dpi=(300, 300), optimize=True)


def save_other_grid(source: Path, destination: Path) -> None:
    """Copy the lossless nine-model 3x3 render under its thesis-facing name."""
    with Image.open(source) as image:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, dpi=(300, 300), optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chen-source",
        type=Path,
        default=DEFAULT_PIPELINE_DIR / "FreqCompChen1e10_Tyson1e8_Other5_1e8_grid.png",
    )
    parser.add_argument(
        "--grid-source",
        type=Path,
        default=DEFAULT_PIPELINE_DIR / "FreqCompModern1e8_Other6_1e8_grid3x3_fixedaxes.png",
    )
    parser.add_argument("--outdir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    chen_out = args.outdir / "FreqCompChen1e10.png"
    grid_out = args.outdir / "FreqCompOther9_1e8_grid3x3.png"
    save_chen_panel(args.chen_source, chen_out)
    save_other_grid(args.grid_source, grid_out)
    print(chen_out)
    print(grid_out)


if __name__ == "__main__":
    main()
