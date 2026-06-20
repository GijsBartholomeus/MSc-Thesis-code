from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import libsbml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCES = ROOT / "sources" / "Dataset1"
PROMOTED = ROOT / "models_promoted"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
for path in (PROMOTED, RESULTS, PLOTS):
    path.mkdir(exist_ok=True)

MULTIPLIERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DIVERGENCE_CAP_FACTOR = 100.0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    source_dir: str
    sbml: Path
    output: str
    t_end: float
    npoints: int
    coarse_start: float
    coarse_duration: float
    promoted_sbml: Path | None = None
    setup: Callable[[roadrunner.RoadRunner], None] | None = None
    warmup: Callable[[roadrunner.RoadRunner], None] | None = None


def sample_label(samples: int | None) -> str:
    if samples is None:
        return "N=unknown"
    if samples <= 0:
        return f"N={samples}"
    exponent = round(math.log10(samples))
    if 10**exponent == samples:
        return f"N=1e{exponent}"
    return f"N={samples}"


def sample_size_label(samples: int | None) -> str:
    return sample_label(samples).replace("N=", "")


def sample_label_from_data(all_data: list[dict]) -> str:
    sample_values = {int(data["samples"]) for data in all_data if "samples" in data}
    if len(sample_values) == 1:
        return sample_label(sample_values.pop())
    if sample_values:
        return "N=mixed"
    return "N=unknown"


def sample_size_label_from_data(all_data: list[dict]) -> str:
    return sample_label_from_data(all_data).replace("N=", "")


def parse_sample_label(text: str) -> int | None:
    match = re.search(r"N=(\d+(?:e\d+)?)", text) or re.search(r"CompFreq(\d+(?:e\d+)?)", text)
    if not match:
        return None
    value = match.group(1)
    if "e" in value:
        mantissa, exponent = value.split("e", 1)
        return int(float(mantissa) * (10 ** int(exponent)))
    return int(value)


def compfreq_plot_path(all_data: list[dict]) -> Path:
    return PLOTS / f"CompFreq{sample_size_label_from_data(all_data)}.png"


def paper_figure_dirs() -> list[Path]:
    dirs: list[Path] = []
    configured = os.environ.get("WSBW_PAPER_DIR")
    if configured:
        dirs.append(Path(configured).expanduser() / "Figures")

    papers_root = Path("/Users/gijsbartholomeus/Documents/STUDIE/Papers/WhySystemsBiologyWorks")
    candidates = [
        papers_root / "PNAS-WhySystemsBiologyWorks-git" / "Figures",
        papers_root / "PNAS_WhySystemsBiologyWorks" / "Figures",
        papers_root / "Figures",
    ]
    if papers_root.exists():
        candidates.extend(path / "Figures" for path in papers_root.glob("PNAS*") if path.is_dir())
    dirs.extend(candidates)

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = directory.expanduser()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def max_sample_in_plot_dir() -> int | None:
    values = [parse_sample_label(path.name) for path in PLOTS.glob("*.png")]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def sync_freqcomp_to_paper(plot_path: Path, all_data: list[dict]) -> None:
    samples = parse_sample_label(plot_path.name) or parse_sample_label(sample_label_from_data(all_data))
    largest = max_sample_in_plot_dir()
    is_largest = samples is not None and largest is not None and samples >= largest
    for figures_dir in paper_figure_dirs():
        target = figures_dir / "FreqComp.png"
        if not figures_dir.exists():
            continue
        if is_largest or not target.exists():
            shutil.copy2(plot_path, target)
            reason = "largest N in plots" if is_largest else "FreqComp.png was missing"
            print(f"Synced {plot_path.name} to {target} ({reason})")


def set_if_exists(rr: roadrunner.RoadRunner, key: str, value: float) -> None:
    try:
        rr.setValue(key, value)
    except Exception:
        pass


def setup_chen(rr):
    set_if_exists(rr, "PE", 0.698687)
    set_if_exists(rr, "CDC15", 0.6565)
    set_if_exists(rr, "CDC15i", 0.3435)


def warmup_leloup(rr):
    rr.simulate(0, 72, 500)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


def warmup_locke(rr):
    rr.simulate(0, 24 * 10, 1200)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


def warmup_ueda(rr):
    rr.simulate(0, 20 * 24, 2000)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


SPECS = [
    ModelSpec("chen2004", "Chen 2004", "Chen_2004", MODELS / "BIOMD0000000056.xml", "CLB2", 418.948, 1001, 14.140, 404.808, setup=setup_chen),
    ModelSpec("kholodenko2000", "Kholodenko 2000", "Kholodenko_2000", MODELS / "BIOMD0000000010.xml", "MKK_PP", 6526.107, 1201, 1301.750, 5224.357),
    ModelSpec("leloup1999", "Leloup 1999", "Leloup_1999", MODELS / "BIOMD0000000021.xml", "Cn", 131.371, 1001, 34.795, 96.576, warmup=warmup_leloup),
    ModelSpec("locke2005", "Locke 2005", "Locke_2005", MODELS / "BIOMD0000000055.xml", "cXn", 106.272, 1001, 10.272, 96.000, warmup=warmup_locke),
    ModelSpec("ueda2001", "Ueda 2001", "Ueda_2001", MODELS / "BIOMD0000000022.xml", "CCc", 88.713, 1001, 2.269, 86.444, warmup=warmup_ueda),
    ModelSpec("vilar2002", "Vilar 2002", "Vilar_2002", MODELS / "BIOMD0000000035.xml", "C", 116.960, 1201, 14.705, 102.255),
    ModelSpec("tyson1991", "Tyson 1991", "Tyson_1991", MODELS / "BIOMD0000000005.xml", "M", 160.0, 1001, 0.0, 160.0),
    ModelSpec("almeida2020", "Almeida 2020", "", MODELS / "BIOMD0000000839.xml", "PERCRY", 240.0, 1001, 40.0, 200.0),
    ModelSpec("rodenfels2019", "Rodenfels 2019", "", MODELS / "BIOMD0000000952.xml", "Cyclin_B1_Cdk1_complex_phosphorylated", 500.0, 1001, 50.0, 450.0),
    ModelSpec("novak2022", "Novak 2022", "", MODELS / "BIOMD0000001058.xml", "Cdh1", 1000.0, 1001, 100.0, 900.0),
]


def hessian_keys(spec: ModelSpec) -> list[str]:
    path = SOURCES / spec.source_dir / "hessian_keys.dat"
    if not spec.source_dir or not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_key(key: str) -> str:
    return re.sub(r"_\d+$", "", key)


def local_parameter_values(sbml: Path) -> dict[str, list[tuple[str, float]]]:
    doc = libsbml.readSBML(str(sbml))
    model = doc.getModel()
    values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for rxn in model.getListOfReactions():
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        locals_list = []
        if hasattr(kl, "getListOfLocalParameters"):
            locals_list.extend(list(kl.getListOfLocalParameters()))
        if hasattr(kl, "getListOfParameters"):
            locals_list.extend(list(kl.getListOfParameters()))
        for par in locals_list:
            values[par.getId()].append((rxn.getId(), par.getValue()))
    return values


def promote_local_parameters(spec: ModelSpec) -> Path:
    keys = {canonical_key(k) for k in hessian_keys(spec)}
    out = PROMOTED / f"{spec.key}.xml"
    if out.exists() and out.stat().st_size > 0:
        return out
    doc = libsbml.readSBML(str(spec.sbml))
    model = doc.getModel()

    existing_globals = {p.getId() for p in model.getListOfParameters()}
    local_values = local_parameter_values(spec.sbml)
    for pid in sorted(keys):
        if pid in existing_globals or pid not in local_values:
            continue
        vals = [v for _, v in local_values[pid]]
        if max(vals) - min(vals) > 1e-12:
            # Duplicate local IDs with different values cannot be safely promoted to one global.
            continue
        par = model.createParameter()
        par.setId(pid)
        par.setConstant(True)
        par.setValue(vals[0])

    for rxn in model.getListOfReactions():
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        for pid in sorted(keys):
            if pid in existing_globals or pid in {p.getId() for p in model.getListOfParameters()}:
                if hasattr(kl, "getLocalParameter") and kl.getLocalParameter(pid) is not None:
                    kl.removeLocalParameter(pid)
                if hasattr(kl, "getParameter") and kl.getParameter(pid) is not None:
                    kl.removeParameter(pid)

    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    libsbml.writeSBMLToFile(doc, str(tmp))
    os.replace(tmp, out)
    return out


def candidate_parameter_ids(spec: ModelSpec, sbml: Path) -> tuple[list[str], dict[str, list[str]]]:
    rr = roadrunner.RoadRunner(str(sbml))
    if spec.setup:
        spec.setup(rr)
    if spec.key == "chen2004":
        resolved = []
        rejected: dict[str, list[str]] = defaultdict(list)
        for pid in rr.model.getGlobalParameterIds():
            try:
                val = float(rr.getValue(pid))
                rr.setValue(pid, val)
            except Exception:
                rejected["not_settable"].append(pid)
                continue
            low = pid.lower()
            if (low.endswith("t") and val in (0.0, 1.0)) or (
                low.startswith("d") and low.endswith("n")
            ) or ("flag" in low) or ("switch" in low) or val == 0.0 or pid in {"cell"} or (
                "total" in low and val in (0.0, 1.0)
            ):
                rejected["nonkinetic_or_switch_like"].append(pid)
            else:
                resolved.append(pid)
        return resolved, dict(rejected)

    global_ids = set(rr.model.getGlobalParameterIds())
    boundary_ids = set(rr.model.getBoundarySpeciesIds())
    keys = hessian_keys(spec)
    resolved = []
    rejected: dict[str, list[str]] = defaultdict(list)
    seen = set()

    if not keys:
        keys = list(rr.model.getGlobalParameterIds()) + list(rr.model.getBoundarySpeciesIds())

    for raw in keys:
        options = [raw, canonical_key(raw)]
        pid = next((opt for opt in options if opt in global_ids or opt in boundary_ids), None)
        if pid is None:
            rejected["not_settable"].append(raw)
            continue
        if pid in seen:
            continue
        seen.add(pid)
        try:
            val = float(rr.getValue(pid))
            rr.setValue(pid, val)
        except Exception:
            rejected["not_settable"].append(raw)
            continue
        low = pid.lower()
        if not math.isfinite(val):
            rejected["nonfinite"].append(pid)
        elif val == 0.0:
            rejected["zero"].append(pid)
        elif val in (0.0, 1.0) and (low in {"light", "emptyset"} or low.startswith("switch") or "flag" in low):
            rejected["binary_switch"].append(pid)
        elif low in {"n", "emptyset", "light", "time", "turntime"}:
            rejected["manual_exclude"].append(pid)
        else:
            resolved.append(pid)
    return resolved, dict(rejected)


def prepare_models() -> dict[str, dict]:
    audit = {}
    for spec in SPECS:
        sbml = promote_local_parameters(spec)
        params, rejected = candidate_parameter_ids(spec, sbml)
        audit[spec.key] = {
            "label": spec.label,
            "source_sbml": str(spec.sbml),
            "promoted_sbml": str(sbml),
            "output": spec.output,
            "free_parameter_count": len(params),
            "free_parameters": params,
            "rejected": rejected,
        }
    (RESULTS / "parameter_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def lz76_phrase_count(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    i = 0
    c = 1
    k = 1
    while i + k <= n:
        if s[i : i + k] in s[: i + k - 1]:
            k += 1
            if i + k - 1 > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
    return c


def clz(bits: str) -> float:
    if not bits:
        return 0.0
    if bits.count("0") == len(bits) or bits.count("1") == len(bits):
        return math.log2(len(bits))
    return math.log2(len(bits)) / 2.0 * (lz76_phrase_count(bits) + lz76_phrase_count(bits[::-1]))


def encode_signal(time: np.ndarray, signal: np.ndarray, nbins: int = 50) -> str:
    coarse_time = np.linspace(time[0], time[-1], nbins)
    coarse_signal = np.interp(coarse_time, time, signal)
    slopes = np.diff(coarse_signal) / np.diff(coarse_time)
    return "".join("1" if slope > 0 else "0" for slope in slopes)


def simulate_encoding(
    rr: roadrunner.RoadRunner,
    spec: ModelSpec,
    defaults: dict[str, float],
    base_initials: dict[str, float],
    wildtype: bool,
    rng: random.Random,
    divergence_cap: float | None = None,
):
    for sid, val in base_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if not wildtype:
        for pid, val in defaults.items():
            rr.setValue(pid, val * rng.choice(MULTIPLIERS))
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    result = rr.simulate(0, spec.t_end, spec.npoints)
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    if divergence_cap is not None and np.any(np.abs(y) > divergence_cap):
        return None
    if np.any(np.abs(y) > 1e9):
        return None
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    if not np.any(mask):
        return None
    bits = encode_signal(t[mask], y[mask], 50)
    return bits


def run_model(spec: ModelSpec, audit: dict, samples: int = 5000, seed: int = 1):
    sbml = Path(audit[spec.key]["promoted_sbml"])
    params = audit[spec.key]["free_parameters"]
    rng = random.Random(seed)
    counts = Counter()
    failures = 0
    rr = roadrunner.RoadRunner(str(sbml))
    if spec.setup:
        spec.setup(rr)
    defaults = {pid: float(rr.getValue(pid)) for pid in params}
    base_initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            base_initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            base_initials[sid] = float(rr.getValue(sid))
    wt_bits = simulate_encoding(rr, spec, defaults, base_initials, wildtype=True, rng=rng)
    restore_initials = dict(base_initials)
    for sid, val in restore_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    wt_result = rr.simulate(0, spec.t_end, spec.npoints)
    wt_signal = np.asarray(wt_result[:, 1], dtype=float)
    wildtype_max_abs = float(np.max(np.abs(wt_signal)))
    divergence_cap = DIVERGENCE_CAP_FACTOR * max(wildtype_max_abs, 1e-12)
    for _ in range(samples):
        try:
            bits = simulate_encoding(
                rr,
                spec,
                defaults,
                base_initials,
                wildtype=False,
                rng=rng,
                divergence_cap=divergence_cap,
            )
            if bits is None:
                failures += 1
            else:
                counts[bits] += 1
        except Exception:
            failures += 1
    data = {
        "model": spec.key,
        "label": spec.label,
        "samples": samples,
        "successes": sum(counts.values()),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": clz(wt_bits) if wt_bits else None,
        "wildtype_count": counts.get(wt_bits, 0) if wt_bits else 0,
        "wildtype_max_abs": wildtype_max_abs,
        "divergence_cap_factor": DIVERGENCE_CAP_FACTOR,
        "divergence_cap": divergence_cap,
        "time_window": {
            "t_end": spec.t_end,
            "coarse_start": spec.coarse_start,
            "coarse_duration": spec.coarse_duration,
        },
        "phenotypes": [{"encoding": enc, "count": n, "complexity": clz(enc)} for enc, n in counts.items()],
    }
    (RESULTS / f"{spec.key}_complexity_frequency.json").write_text(json.dumps(data))
    return data


def plot_complexity_frequency(
    all_data: list[dict],
    out: Path | None = None,
    show_wildtype: bool = True,
    auto_hide_low_wildtype: bool = True,
    min_complexity: float | None = None,
    max_complexity: float | None = None,
):
    data_by_model = {data["model"]: data for data in all_data}
    if "chen2004" in data_by_model and len(all_data) == 7:
        out = plot_complexity_frequency_chico_layout(
            all_data,
            out=out,
            show_wildtype=show_wildtype,
            auto_hide_low_wildtype=auto_hide_low_wildtype,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            grid=False,
        )
        grid_out = out.with_name(f"{out.stem}_grid{out.suffix}")
        plot_complexity_frequency_chico_layout(
            all_data,
            out=grid_out,
            show_wildtype=show_wildtype,
            auto_hide_low_wildtype=auto_hide_low_wildtype,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            grid=True,
        )
        legacy_out = PLOTS / "oscillatory_subset_complexity_frequency.png"
        if out != legacy_out:
            legacy_out.write_bytes(out.read_bytes())
        trough_legacy_out = PLOTS / "oscillatory_subset_complexity_frequency_trough_windows.png"
        if out != trough_legacy_out:
            trough_legacy_out.write_bytes(out.read_bytes())
        sync_freqcomp_to_paper(out, all_data)
        return out

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9C755F"]
    ncols = min(3, max(1, len(all_data)))
    nrows = int(math.ceil(len(all_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, data, color in zip(axes, all_data, colors):
        draw_complexity_panel(
            ax,
            data,
            color,
            show_wildtype=show_wildtype,
            auto_hide_low_wildtype=False,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            title=data["label"],
            xlabel="K(x)",
            ylabel="P(x)",
            grid=False,
        )
    for ax in axes[len(all_data) :]:
        ax.axis("off")
    if out is None:
        out = compfreq_plot_path(all_data)
    fig.savefig(out, dpi=220)
    legacy_out = PLOTS / "oscillatory_subset_complexity_frequency.png"
    if out != legacy_out:
        fig.savefig(legacy_out, dpi=220)
    trough_legacy_out = PLOTS / "oscillatory_subset_complexity_frequency_trough_windows.png"
    if out != trough_legacy_out:
        fig.savefig(trough_legacy_out, dpi=220)
    sync_freqcomp_to_paper(out, all_data)
    return out


def panel_points(data: dict) -> tuple[np.ndarray, np.ndarray]:
    phenos = data["phenotypes"]
    successes = max(data["successes"], 1)
    xs = np.array([p["complexity"] for p in phenos], dtype=float)
    ys = np.array([p["count"] / successes for p in phenos], dtype=float)
    return xs, ys


def draw_complexity_panel(
    ax,
    data: dict,
    color: str,
    *,
    show_wildtype: bool,
    auto_hide_low_wildtype: bool,
    min_complexity: float | None,
    max_complexity: float | None,
    title: str | None = None,
    roman: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = False,
    scatter_size: float = 10,
    hull_alpha: float = 0.35,
    trim_ylim_to_points: bool = True,
):
    xs, ys = panel_points(data)
    ax.scatter(xs, ys, s=scatter_size, color="black", alpha=0.58, linewidths=0, zorder=3)
    bins = defaultdict(list)
    for x, y in zip(xs, ys):
        if min_complexity is not None and x < min_complexity:
            continue
        if max_complexity is not None and x > max_complexity:
            continue
        bins[round(float(x), 1)].append(float(y))
    if len(bins) >= 2:
        bx = np.array(sorted(bins))
        upper = np.array([max(bins[x]) for x in bx])
        lower = np.array([min(bins[x]) for x in bx])
        ax.fill_between(bx, lower, upper, color=color, alpha=hull_alpha, zorder=1)
        ax.plot(bx, upper, color=color, lw=1.7, zorder=2)
    wt_y = None
    if show_wildtype and data.get("wildtype_encoding"):
        wt_x = data["wildtype_complexity"]
        wt_y = max(data["wildtype_count"], 0.5) / max(data["successes"], 1)
        positive = ys[ys > 0]
        should_hide_wt = auto_hide_low_wildtype and len(positive) and wt_y < float(np.nanmin(positive))
        if not should_hide_wt:
            ax.scatter([wt_x], [wt_y], color="red", edgecolor="black", linewidth=0.4, s=34, zorder=5)
        else:
            wt_y = None
    if min_complexity is not None or max_complexity is not None:
        left = min_complexity if min_complexity is not None else float(np.nanmin(xs))
        right = max_complexity if max_complexity is not None else float(np.nanmax(xs))
        ax.set_xlim(left, right)
    ax.set_yscale("log")
    if trim_ylim_to_points and len(ys):
        positive = ys[ys > 0]
        if len(positive):
            ymin = float(np.nanmin(positive))
            ymax = float(np.nanmax(positive))
            if wt_y is not None and wt_y >= ymin:
                ymin = min(ymin, float(wt_y))
            if wt_y is not None:
                ymax = max(ymax, float(wt_y))
            ax.set_ylim(ymin / 1.4, ymax * 1.5)
    if title:
        ax.set_title(title, fontsize=10)
    if roman:
        ax.text(0.95, 0.92, roman, transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.0 if not grid else 0.25)
    if not grid:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_complexity_frequency_chico_layout(
    all_data: list[dict],
    out: Path | None,
    show_wildtype: bool,
    auto_hide_low_wildtype: bool,
    min_complexity: float | None,
    max_complexity: float | None,
    grid: bool,
) -> Path:
    data_by_model = {data["model"]: data for data in all_data}
    chen = data_by_model["chen2004"]
    right_order = ["kholodenko2000", "leloup1999", "locke2005", "ueda2001", "vilar2002", "tyson1991"]
    right_data = [data_by_model[key] for key in right_order if key in data_by_model]
    colors = {
        "chen2004": "#9C6BC7",
        "kholodenko2000": "#4C78A8",
        "leloup1999": "#F58518",
        "locke2005": "#54A24B",
        "ueda2001": "#E45756",
        "vilar2002": "#B279A2",
        "tyson1991": "#9C755F",
    }
    all_x = np.concatenate([panel_points(data)[0] for data in all_data])
    xmin = min_complexity if min_complexity is not None else float(np.floor(np.nanmin(all_x)))
    xmax = max_complexity if max_complexity is not None else float(np.ceil(np.nanmax(all_x)))

    fig = plt.figure(figsize=(13.5, 5.8))
    ax_a = fig.add_axes([0.065, 0.14, 0.40, 0.78])
    draw_complexity_panel(
        ax_a,
        chen,
        colors["chen2004"],
        show_wildtype=show_wildtype,
        auto_hide_low_wildtype=auto_hide_low_wildtype,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
        xlabel=r"Phenotype complexity $K(x)$",
        ylabel=r"Phenotype frequency $P(x)$",
        grid=grid,
        scatter_size=13,
        hull_alpha=0.45,
    )
    ax_a.text(-0.13, 1.03, "A", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")

    left0 = 0.535
    bottom0 = 0.12
    panel_w = 0.165
    panel_h = 0.22
    xgap = 0.085
    ygap = 0.105
    romans = ["i", "ii", "iii", "iv", "v", "vi"]
    for idx, (data, roman) in enumerate(zip(right_data, romans)):
        row = idx // 2
        col = idx % 2
        x0 = left0 + col * (panel_w + xgap)
        y0 = bottom0 + (2 - row) * (panel_h + ygap)
        ax = fig.add_axes([x0, y0, panel_w, panel_h])
        draw_complexity_panel(
            ax,
            data,
            colors.get(data["model"], "#4C78A8"),
            show_wildtype=show_wildtype,
            auto_hide_low_wildtype=auto_hide_low_wildtype,
            min_complexity=xmin,
            max_complexity=xmax,
            roman=roman,
            xlabel=r"$K(x)$" if idx >= 4 else None,
            ylabel=r"$P(x)$" if idx % 2 == 0 else None,
            grid=grid,
            scatter_size=8,
        )
        ax.set_xlim(xmin, xmax)
        ax.tick_params(labelsize=8)
        if idx % 2 != 0:
            ax.set_yticklabels([])
        if idx < 4:
            ax.set_xticklabels([])
    fig.text(0.49, 0.96, "B", ha="left", va="top", fontsize=15, fontweight="bold")

    if out is None:
        out = compfreq_plot_path(all_data)
    fig.savefig(out, dpi=300)
    return out


def main(
    samples: int = 1000,
    seed: int = 1,
    show_wildtype: bool = True,
    auto_hide_low_wildtype: bool = True,
    min_complexity: float | None = None,
    max_complexity: float | None = None,
):
    audit = prepare_models()
    all_data = []
    for idx, spec in enumerate(SPECS):
        print(f"Running {spec.label} with {audit[spec.key]['free_parameter_count']} free parameters")
        all_data.append(run_model(spec, audit, samples=samples, seed=seed + idx))
    out = plot_complexity_frequency(
        all_data,
        show_wildtype=show_wildtype,
        auto_hide_low_wildtype=auto_hide_low_wildtype,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
    )
    print(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hide-wildtype", action="store_true")
    parser.add_argument("--show-low-wildtype", action="store_true")
    parser.add_argument("--min-complexity", type=float, default=None)
    parser.add_argument("--max-complexity", type=float, default=None)
    args = parser.parse_args()
    main(
        samples=args.samples,
        seed=args.seed,
        show_wildtype=not args.hide_wildtype,
        auto_hide_low_wildtype=not args.show_low_wildtype,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
    )
