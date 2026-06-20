#!/usr/bin/env python3
"""Merge Chen 1e10 FreqComp chunks and render the pipeline plot."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import PLOTS, RESULTS, SPECS, plot_complexity_frequency


TAG = "chen_freqcomp_1e10_iid"
OTHER_TAG = "hydra_1e8_no_chen_tyson_v3"
PLOT_NAME = "FreqCompChen1e10_Tyson1e8_Other5_1e8.png"


def main() -> None:
    root = RESULTS / "freqcomp_chunks" / TAG
    paths = sorted(root.glob("chen2004_freqcomp_N=20000000_chunk-*.json"))
    if len(paths) != 500:
        raise RuntimeError(f"Need 500 Chen chunks, found {len(paths)} in {root}")

    counts: Counter[tuple[str, float]] = Counter()
    samples = successes = failures = 0
    first: dict | None = None
    for idx, path in enumerate(paths, start=1):
        data = json.loads(path.read_text())
        if first is None:
            first = data
        samples += int(data["samples"])
        successes += int(data["successes"])
        failures += int(data["failures"])
        for phenotype in data["phenotypes"]:
            counts[(phenotype["encoding"], float(phenotype["complexity"]))] += int(phenotype["count"])
        if idx == 1 or idx == len(paths) or idx % 25 == 0:
            print(f"merged {idx}/{len(paths)} chunks; unique phenotypes={len(counts):,}", flush=True)

    assert first is not None
    wt = first["wildtype_encoding"]
    merged = {
        "model": "chen2004",
        "label": "Chen 2004",
        "samples": samples,
        "successes": successes,
        "failures": failures,
        "wildtype_encoding": wt,
        "wildtype_complexity": first.get("wildtype_complexity"),
        "wildtype_count": sum(n for (encoding, _), n in counts.items() if encoding == wt),
        "hydra_merged_chunks": len(paths),
        "phenotypes": [
            {"encoding": encoding, "complexity": complexity, "count": count}
            for (encoding, complexity), count in counts.items()
        ],
    }

    out_json = RESULTS / f"chen2004_complexity_frequency_{TAG}_merged.json"
    out_json.write_text(json.dumps(merged))

    all_data = [merged]
    missing: list[str] = []
    for spec in SPECS:
        if spec.key == "chen2004":
            continue
        path = RESULTS / f"{spec.key}_complexity_frequency_{OTHER_TAG}_merged.json"
        if path.exists():
            all_data.append(json.loads(path.read_text()))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing comparison JSONs:\n" + "\n".join(missing))

    plot = PLOTS / PLOT_NAME
    plot_complexity_frequency(all_data, out=plot)
    figdir = Path("figures/pipeline")
    figdir.mkdir(parents=True, exist_ok=True)
    for src in [plot, plot.with_name(f"{plot.stem}_grid{plot.suffix}")]:
        if src.exists():
            shutil.copy2(src, figdir / src.name)

    print(f"merged_json={out_json}", flush=True)
    print(f"plot={plot}", flush=True)
    print(f"grid={plot.with_name(f'{plot.stem}_grid{plot.suffix}')}", flush=True)
    print(
        f"samples={samples:,} successes={successes:,} failures={failures:,} "
        f"phenotypes={len(counts):,} wt_count={merged['wildtype_count']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
