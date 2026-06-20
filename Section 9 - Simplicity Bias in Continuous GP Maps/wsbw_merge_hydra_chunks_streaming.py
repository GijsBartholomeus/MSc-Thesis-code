from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS, SPECS, clz


CHUNK_DIR = RESULTS / "hydra_chunks"


def merge_one_model(tag: str, model: str) -> Path:
    in_dir = CHUNK_DIR / tag
    paths = sorted(in_dir.glob(f"{model}_chunk-*.json"))
    if not paths:
        # Historical chunk naming sometimes starts with the model label embedded
        # in a longer file stem.  Fall back to scanning the model field.
        paths = []
        for path in sorted(in_dir.glob("*_chunk-*.json")):
            with path.open() as handle:
                data = json.load(handle)
            if data["model"] == model:
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No chunks for {model} in {in_dir}")

    merged: Counter[str] = Counter()
    samples = 0
    failures = 0
    first = None
    for pos, path in enumerate(paths, start=1):
        with path.open() as handle:
            chunk = json.load(handle)
        if first is None:
            first = chunk
        samples += int(chunk["samples"])
        failures += int(chunk["failures"])
        for phenotype in chunk["phenotypes"]:
            merged[phenotype["encoding"]] += int(phenotype["count"])
        if pos % 10 == 0 or pos == len(paths):
            print(f"{model}: merged {pos}/{len(paths)} chunks; phenotypes={len(merged):,}", flush=True)

    assert first is not None
    wt_bits = first.get("wildtype_encoding")
    out_data = {
        "model": first["model"],
        "label": first["label"],
        "samples": samples,
        "successes": int(sum(merged.values())),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": first.get("wildtype_complexity"),
        "wildtype_count": int(merged.get(wt_bits, 0)) if wt_bits else 0,
        "wildtype_max_abs": first.get("wildtype_max_abs"),
        "divergence_cap_factor": first.get("divergence_cap_factor"),
        "divergence_cap": first.get("divergence_cap"),
        "time_window": first.get("time_window"),
        "hydra_merged_chunks": len(paths),
        "phenotypes": [
            {"encoding": enc, "count": int(n), "complexity": clz(enc)}
            for enc, n in merged.items()
        ],
    }
    out = RESULTS / f"{model}_complexity_frequency_{tag}_merged.json"
    out.write_text(json.dumps(out_data))
    print(out, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming merge for large Hydra chunk JSONs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    known = {spec.key for spec in SPECS}
    for model in args.models:
        if model not in known:
            raise ValueError(f"Unknown model: {model}")
        merge_one_model(args.tag, model)


if __name__ == "__main__":
    main()
