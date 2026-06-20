from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import PLOTS, RESULTS, SPECS, draw_complexity_panel, plot_complexity_frequency


IN_ROOT = RESULTS / "bruteforce_cloud"
OUT_ROOT = RESULTS / "bruteforce_cloud_stats"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "chen2004": "#9467bd",
    "tyson1991": "#8c564b",
}


def sample_label(samples: int) -> str:
    if samples <= 0:
        return str(samples)
    exponent = round(math.log10(samples))
    if 10**exponent == samples:
        return f"1e{exponent}"
    return str(samples)


def uint64_to_bits(code: int, length: int = 49) -> str:
    return format(int(code), f"0{length}b")


def reservoir_update(
    sample: dict[str, np.ndarray | int | None],
    points: np.ndarray,
    complexities: np.ndarray,
    objectives: np.ndarray,
    codes: np.ndarray | None,
    max_size: int,
    rng: np.random.Generator,
) -> None:
    if max_size <= 0 or len(points) == 0:
        return
    seen = int(sample.get("seen", 0) or 0)
    current_points = sample.get("points")
    current_complexities = sample.get("complexities")
    current_objectives = sample.get("objectives")
    current_codes = sample.get("codes")
    if current_points is None:
        current_points = np.empty((0, points.shape[1]), dtype=np.float32)
        current_complexities = np.empty(0, dtype=np.float32)
        current_objectives = np.empty(0, dtype=np.float32)
        current_codes = np.empty(0, dtype=np.uint64)

    current_points = np.asarray(current_points)
    current_complexities = np.asarray(current_complexities)
    current_objectives = np.asarray(current_objectives)
    current_codes = np.asarray(current_codes, dtype=np.uint64)
    codes = np.asarray(codes, dtype=np.uint64) if codes is not None else None

    start_idx = 0
    if len(current_points) < max_size:
        n_fill = min(max_size - len(current_points), len(points))
        if n_fill:
            current_points = np.concatenate([current_points, points[:n_fill].astype(np.float32)], axis=0)
            current_complexities = np.concatenate(
                [current_complexities, complexities[:n_fill].astype(np.float32)],
                axis=0,
            )
            current_objectives = np.concatenate(
                [current_objectives, objectives[:n_fill].astype(np.float32)],
                axis=0,
            )
            if codes is not None:
                current_codes = np.concatenate([current_codes, codes[:n_fill].astype(np.uint64)], axis=0)
            seen += n_fill
            start_idx = n_fill

    remaining = len(points) - start_idx
    if remaining > 0:
        replace = rng.integers(0, seen + np.arange(1, remaining + 1), size=remaining)
        keep = replace < max_size
        if np.any(keep):
            src = np.nonzero(keep)[0] + start_idx
            dst = replace[keep].astype(np.int64)
            current_points[dst] = points[src]
            current_complexities[dst] = complexities[src]
            current_objectives[dst] = objectives[src]
            if codes is not None:
                current_codes[dst] = codes[src]
        seen += remaining

    sample["seen"] = seen
    sample["points"] = current_points
    sample["complexities"] = current_complexities
    sample["objectives"] = current_objectives
    sample["codes"] = current_codes


def normalize_points(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return points / np.maximum(2.0 * p0[None, :], np.finfo(np.float32).tiny)


def sample_array(sample: dict, key: str, n_cols: int | None = None, dtype=np.float32) -> np.ndarray:
    value = sample.get(key)
    if value is None:
        if n_cols is None:
            return np.empty(0, dtype=dtype)
        return np.empty((0, n_cols), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def pairwise_stats(points: np.ndarray, rng: np.random.Generator, max_points: int) -> dict:
    if len(points) < 2:
        return {"n": int(len(points)), "mean": None, "max": None, "q95": None}
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    distances = pdist(points, metric="euclidean")
    return {
        "n": int(len(points)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "q95": float(np.quantile(distances, 0.95)),
        "max": float(np.max(distances)),
    }


def pca_scores(points: np.ndarray, n_components: int = 4) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    scores = centered @ components
    variance = singular**2
    explained = variance / np.sum(variance) if np.sum(variance) > 0 else np.full_like(variance, np.nan)
    return scores, explained[:n_components]


def complexity_quartile_labels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(len(values), dtype=int), np.array([])
    edges = np.quantile(finite, [0.25, 0.5, 0.75])
    labels = np.digitize(values, edges, right=True)
    return labels, edges


def make_data_for_panel(model: str, label: str, attempted: int, successes: int, phenotype_counts: Counter, wildtype_code: int, wildtype_complexity: float) -> dict:
    phenotypes = [
        {"encoding": uint64_to_bits(code), "count": int(count), "complexity": float(complexity)}
        for (code, complexity), count in phenotype_counts.items()
    ]
    wt_bits = uint64_to_bits(wildtype_code)
    wt_count = sum(count for (code, _), count in phenotype_counts.items() if int(code) == int(wildtype_code))
    return {
        "model": model,
        "label": label,
        "samples": attempted,
        "successes": successes,
        "failures": attempted - successes,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": wildtype_complexity,
        "wildtype_count": int(wt_count),
        "phenotypes": phenotypes,
    }


def save_complexity_frequency_json(data: dict, out_dir: Path, tag: str) -> Path:
    out = out_dir / f"{data['model']}_complexity_frequency_{tag}.json"
    out.write_text(json.dumps(data, indent=2))
    return out


def plot_stats(
    model: str,
    data: dict,
    point_sample: dict,
    p0: np.ndarray,
    out_dir: Path,
    tag: str,
    rng: np.random.Generator,
    max_plot_points: int,
) -> Path:
    color = MODEL_COLORS.get(model, "#4c78a8")
    points = np.asarray(point_sample.get("points"), dtype=np.float32)
    complexities = np.asarray(point_sample.get("complexities"), dtype=np.float32)
    objectives = np.asarray(point_sample.get("objectives"), dtype=np.float32)
    if len(points) > max_plot_points:
        idx = rng.choice(len(points), size=max_plot_points, replace=False)
        points = points[idx]
        complexities = complexities[idx]
        objectives = objectives[idx]

    norm_points = normalize_points(points, p0)
    scores, explained = pca_scores(norm_points, 4) if len(norm_points) >= 4 else (np.empty((0, 4)), np.empty(4))
    quartiles, edges = complexity_quartile_labels(complexities)
    palette = np.array(["#3b4cc0", "#78b7ff", "#f6c85f", "#c44e52"])

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_pca12 = fig.add_subplot(gs[1, 0])
    ax_pca34 = fig.add_subplot(gs[1, 1])

    phenos = data["phenotypes"]
    unique_complexities = np.asarray([p["complexity"] for p in phenos], dtype=float)
    unique_counts = np.asarray([p["count"] for p in phenos], dtype=float)
    bins = np.arange(np.floor(np.nanmin(unique_complexities)), np.ceil(np.nanmax(unique_complexities)) + 1.5, 1.0)
    ax_hist.hist(unique_complexities, bins=bins, color=color, alpha=0.35, label="unique phenotypes")
    ax_hist.hist(unique_complexities, bins=bins, weights=unique_counts, histtype="step", lw=2.0, color="black", label="sample count")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("phenotype complexity K(x)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Complexity distribution")
    ax_hist.legend(frameon=False, fontsize=8)

    draw_complexity_panel(
        ax_panel,
        data,
        color,
        show_wildtype=True,
        auto_hide_low_wildtype=False,
        min_complexity=None,
        max_complexity=None,
        xlabel="Phenotype complexity K(x)",
        ylabel="Phenotype frequency P(x)",
        grid=False,
        scatter_size=10,
    )
    ax_panel.set_title("Complexity-frequency GP map")

    if len(scores):
        ax_pca12.scatter(scores[:, 0], scores[:, 1], c=palette[quartiles], s=4, alpha=0.45, linewidths=0)
        ax_pca12.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax_pca12.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax_pca12.set_title("Parameter PCA, colored by K quartile")

        if scores.shape[1] >= 4:
            ax_pca34.scatter(scores[:, 2], scores[:, 3], c=palette[quartiles], s=4, alpha=0.45, linewidths=0)
            ax_pca34.set_xlabel(f"PC3 ({explained[2]:.1%})")
            ax_pca34.set_ylabel(f"PC4 ({explained[3]:.1%})")
            ax_pca34.set_title("Parameter PCA")

    info = [
        f"model: {data['label']}",
        f"attempted: {data['samples']:,}",
        f"successes: {data['successes']:,}",
        f"unique phenotypes: {len(phenos):,}",
        f"plot sample: {len(points):,}",
        f"K quartile edges: {', '.join(f'{x:.2f}' for x in edges)}",
    ]
    ax_hist.text(
        0.98,
        0.98,
        "\n".join(info),
        ha="right",
        va="top",
        fontsize=8,
        transform=ax_hist.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    out = out_dir / f"{model}_bruteforce_stats_{tag}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    start = time.time()
    in_dir = IN_ROOT / args.tag
    out_tag = args.out_tag or args.tag
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)
    all_paths = sorted(in_dir.glob(f"{args.model}_bruteforce_cloud_N=*chunk-*.npz"))
    if not all_paths:
        raise FileNotFoundError(f"No chunk files for {args.model} in {in_dir}")

    rng = np.random.default_rng(args.seed)
    paths = all_paths
    if args.max_chunks > 0 and len(paths) > args.max_chunks:
        chosen = np.sort(rng.choice(len(paths), size=args.max_chunks, replace=False))
        paths = [paths[int(i)] for i in chosen]

    phenotype_counts: Counter = Counter()
    attempted = 0
    successes = 0
    failures = 0
    p0 = None
    parameter_names = None
    wildtype_code = None
    wildtype_complexity = None
    label = args.model

    all_sample: dict[str, np.ndarray | int | None] = {"seen": 0, "points": None, "complexities": None, "objectives": None, "codes": None}
    neutral_sample: dict[str, np.ndarray | int | None] = {"seen": 0, "points": None, "complexities": None, "objectives": None, "codes": None}
    wt_phenotype_sample: dict[str, np.ndarray | int | None] = {"seen": 0, "points": None, "complexities": None, "objectives": None, "codes": None}

    out_dir = OUT_ROOT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(paths, start=1):
        data = np.load(path, allow_pickle=True)
        points = np.asarray(data["points"], dtype=np.float32)
        codes = np.asarray(data["phenotype_codes"], dtype=np.uint64)
        complexities = np.asarray(data["complexities"], dtype=np.float32)
        objectives = np.asarray(data["objectives"], dtype=np.float32)
        attempted += int(data["samples_attempted"][0])
        successes += int(data["successes"][0])
        failures += int(data["failures"][0])
        if p0 is None:
            p0 = np.asarray(data["p0"], dtype=np.float32)
            parameter_names = [str(x) for x in data["parameter_names"]]
            wildtype_code = int(data["wildtype_code"][0])
            wildtype_complexity = float(data["wildtype_complexity"][0])
            summary_path = path.with_suffix(".json")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                label = summary.get("label", args.model)

        if not args.sampled_phenotype_counts:
            unique_codes, first_indices, counts = np.unique(codes, return_index=True, return_counts=True)
            for code, first_idx, count in zip(unique_codes, first_indices, counts):
                code_int = int(code)
                complexity = float(complexities[int(first_idx)])
                phenotype_counts[(code_int, complexity)] += int(count)

        reservoir_update(all_sample, points, complexities, objectives, codes, args.max_point_sample, rng)
        neutral_mask = np.isfinite(objectives) & (objectives <= args.neutral_cutoff)
        reservoir_update(
            neutral_sample,
            points[neutral_mask],
            complexities[neutral_mask],
            objectives[neutral_mask],
            codes[neutral_mask],
            args.max_neutral_sample,
            rng,
        )
        wt_mask = codes == np.asarray(wildtype_code, dtype=codes.dtype)
        reservoir_update(
            wt_phenotype_sample,
            points[wt_mask],
            complexities[wt_mask],
            objectives[wt_mask],
            codes[wt_mask],
            args.max_wt_phenotype_sample,
            rng,
        )
        print(f"  merged {idx}/{len(paths)} chunks; attempted={attempted:,}; successes={successes:,}", flush=True)

    assert p0 is not None
    assert parameter_names is not None
    assert wildtype_code is not None
    assert wildtype_complexity is not None

    processed_attempted = attempted
    processed_successes = successes
    processed_failures = failures
    if args.scale_sampled_counts_to_full_run:
        attempted = 0
        successes = 0
        failures = 0
        for path in all_paths:
            summary_path = path.with_suffix(".json")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                attempted += int(summary.get("samples_attempted", 0))
                successes += int(summary.get("successes", 0))
                failures += int(summary.get("failures", 0))
            else:
                with np.load(path, allow_pickle=True) as data:
                    attempted += int(data["samples_attempted"][0])
                    successes += int(data["successes"][0])
                    failures += int(data["failures"][0])

    if args.sampled_phenotype_counts:
        sample_codes = sample_array(all_sample, "codes", None, np.uint64)
        sample_complexities = sample_array(all_sample, "complexities", None, np.float32)
        scale = successes / max(1, len(sample_codes))
        sampled_counts: Counter = Counter()
        if len(sample_codes):
            unique_codes, first_indices, counts = np.unique(sample_codes, return_index=True, return_counts=True)
            for code, first_idx, count in zip(unique_codes, first_indices, counts):
                code_int = int(code)
                complexity = float(sample_complexities[int(first_idx)])
                sampled_counts[(code_int, complexity)] += max(1, int(round(float(count) * scale)))
        phenotype_counts = sampled_counts

    panel_data = make_data_for_panel(
        args.model,
        label,
        attempted,
        successes,
        phenotype_counts,
        wildtype_code,
        wildtype_complexity,
    )
    freq_json = None if args.skip_frequency_json else save_complexity_frequency_json(panel_data, out_dir, out_tag)

    all_points_arr = sample_array(all_sample, "points", len(p0), np.float32)
    neutral_points_arr = sample_array(neutral_sample, "points", len(p0), np.float32)
    wt_phenotype_points_arr = sample_array(wt_phenotype_sample, "points", len(p0), np.float32)
    all_points_norm = normalize_points(all_points_arr, p0)
    neutral_points_norm = normalize_points(neutral_points_arr, p0)
    wt_phenotype_points_norm = normalize_points(wt_phenotype_points_arr, p0)
    all_pairwise = pairwise_stats(all_points_norm, rng, args.max_pairwise_points)
    neutral_pairwise = pairwise_stats(neutral_points_norm, rng, args.max_pairwise_points)
    wt_phenotype_pairwise = pairwise_stats(wt_phenotype_points_norm, rng, args.max_pairwise_points)

    plot_path = plot_stats(
        args.model,
        panel_data,
        all_sample,
        p0,
        out_dir,
        out_tag,
        rng,
        args.max_plot_points,
    )
    pipeline_freqcomp_path = None
    if args.pipeline_freqcomp:
        all_data = [panel_data]
        missing = []
        for spec in SPECS:
            if spec.key == args.model:
                continue
            path = RESULTS / f"{spec.key}_complexity_frequency_{args.pipeline_other_tag}_merged.json"
            if path.exists():
                all_data.append(json.loads(path.read_text()))
            else:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError("Missing pipeline comparison JSONs:\n" + "\n".join(missing))
        pipeline_freqcomp_path = plot_complexity_frequency(all_data, out=PLOTS / args.pipeline_freqcomp)

    sample_npz = None
    if not args.skip_sample_npz:
        sample_npz = out_dir / f"{args.model}_bruteforce_samples_{out_tag}.npz"
        np.savez_compressed(
            sample_npz,
            all_points=all_points_arr,
            all_complexities=sample_array(all_sample, "complexities", None, np.float32),
            all_objectives=sample_array(all_sample, "objectives", None, np.float32),
            all_phenotype_codes=sample_array(all_sample, "codes", None, np.uint64),
            neutral_points=neutral_points_arr,
            neutral_complexities=sample_array(neutral_sample, "complexities", None, np.float32),
            neutral_objectives=sample_array(neutral_sample, "objectives", None, np.float32),
            neutral_phenotype_codes=sample_array(neutral_sample, "codes", None, np.uint64),
            wt_phenotype_points=wt_phenotype_points_arr,
            wt_phenotype_complexities=sample_array(wt_phenotype_sample, "complexities", None, np.float32),
            wt_phenotype_objectives=sample_array(wt_phenotype_sample, "objectives", None, np.float32),
            wt_phenotype_codes=sample_array(wt_phenotype_sample, "codes", None, np.uint64),
            p0=p0,
            parameter_names=np.asarray(parameter_names, dtype=object),
        )

    elapsed = time.time() - start
    summary = {
        "model": args.model,
        "label": label,
        "tag": out_tag,
        "input_tag": args.tag,
        "chunks_merged": len(paths),
        "chunks_available": len(all_paths),
        "samples_attempted": attempted,
        "successes": successes,
        "failures": failures,
        "processed_samples_attempted": processed_attempted,
        "processed_successes": processed_successes,
        "processed_failures": processed_failures,
        "success_fraction": successes / max(1, attempted),
        "unique_phenotypes": len(phenotype_counts),
        "phenotype_count_mode": "sampled_reservoir_scaled" if args.sampled_phenotype_counts else "exact_full_cloud",
        "scaled_sampled_counts_to_full_run": bool(args.scale_sampled_counts_to_full_run),
        "neutral_cutoff": args.neutral_cutoff,
        "all_point_sample_seen": int(all_sample["seen"]),
        "all_point_sample_saved": int(len(all_points_arr)),
        "neutral_point_sample_seen": int(neutral_sample["seen"]),
        "neutral_point_sample_saved": int(len(neutral_points_arr)),
        "wt_phenotype_sample_seen": int(wt_phenotype_sample["seen"]),
        "wt_phenotype_sample_saved": int(len(wt_phenotype_points_arr)),
        "pairwise_distance_normalized_cube_all_sample": all_pairwise,
        "pairwise_distance_normalized_cube_neutral_sample": neutral_pairwise,
        "pairwise_distance_normalized_cube_wt_phenotype_sample": wt_phenotype_pairwise,
        "wildtype_code": wildtype_code,
        "wildtype_complexity": wildtype_complexity,
        "parameter_names": parameter_names,
        "elapsed_seconds": elapsed,
        "complexity_frequency_json": None if freq_json is None else str(freq_json),
        "sample_npz": None if sample_npz is None else str(sample_npz),
        "plot": str(plot_path),
        "pipeline_freqcomp_plot": None if pipeline_freqcomp_path is None else str(pipeline_freqcomp_path),
    }
    summary_path = out_dir / f"{args.model}_bruteforce_summary_{out_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    print(f"Saved {plot_path}")
    if sample_npz is not None:
        print(f"Saved {sample_npz}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge/analyze brute-force point/complexity/objective cloud chunks")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-tag", default=None)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--neutral-cutoff", type=float, required=True)
    parser.add_argument("--max-point-sample", type=int, default=100000)
    parser.add_argument("--max-neutral-sample", type=int, default=100000)
    parser.add_argument("--max-wt-phenotype-sample", type=int, default=100000)
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--max-pairwise-points", type=int, default=3000)
    parser.add_argument("--skip-frequency-json", action="store_true")
    parser.add_argument("--skip-sample-npz", action="store_true")
    parser.add_argument("--sampled-phenotype-counts", action="store_true")
    parser.add_argument("--pipeline-freqcomp", default=None)
    parser.add_argument("--pipeline-other-tag", default="hydra_1e7")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--scale-sampled-counts-to-full-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
