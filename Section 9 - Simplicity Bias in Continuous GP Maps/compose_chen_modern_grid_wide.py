from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "pipeline"

CHEN_SOURCE = FIGURES / "FreqCompChen1e10_Tyson1e8_Other5_1e8_grid.png"
GRID_SOURCE = FIGURES / "FreqCompModern1e8_Other6_1e8_grid3x3_fixedaxes.png"
OUT = FIGURES / "FreqCompChen1e10_Modern1e8_Other6_1e8_grid3x3_widechen.png"


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    chen_full = Image.open(CHEN_SOURCE).convert("RGB")
    grid = Image.open(GRID_SOURCE).convert("RGB")

    # Crop the well-proportioned Chen panel from the older 1+6 figure, stopping
    # before the old right-side panel block starts.
    chen = chen_full.crop((0, 0, 1935, chen_full.height))

    draw = ImageDraw.Draw(chen)
    # Remove the old "A" label and replace it with the roman label used in the
    # 3x3 paper figure.
    draw.rectangle((0, 0, 120, 105), fill="white")
    draw.text((42, 34), "i", fill="black", font=font(42))

    target_h = 1580
    chen_w = round(chen.width * target_h / chen.height)
    grid_w = round(grid.width * target_h / grid.height)
    chen = chen.resize((chen_w, target_h), Image.Resampling.LANCZOS)
    grid = grid.resize((grid_w, target_h), Image.Resampling.LANCZOS)

    gap = 92
    pad_left = 0
    pad_right = 0
    canvas = Image.new("RGB", (pad_left + chen.width + gap + grid.width + pad_right, target_h), "white")
    canvas.paste(chen, (pad_left, 0))
    canvas.paste(grid, (pad_left + chen.width + gap, 0))
    canvas.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
