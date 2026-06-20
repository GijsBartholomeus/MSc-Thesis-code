#!/usr/bin/env python3
"""Create thesis-quality phenotype panels for Tyson sloppy walks.

This is a script version of the TysonSloppyMultiWalk notebook idea: at each step
we recompute the local log-parameter sensitivity Hessian, choose one of the
current sloppy eigenvectors, align its sign with the previous direction, and step
in log-parameter space.  The output figure shows how the MPF/M phenotype changes
along those walks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
from scipy.integrate import solve_ivp


BASE_PARAMS = {
    "k1_aa_over_CT": 0.015,
    "k2": 0.0,
    "k3_CT": 200.0,
    "k4": 180.0,
    "k4prime": 0.018,
    "k5_minusP": 0.0,
    "k6": 1.0,
    "k7": 0.6,
    "k8_minusP": 100.0,
    "k9": 50.0,
    "CT": 1.0,
}

# This matches TysonSloppyMultiWalk.ipynb.
PARAM_NAMES = ["k1_aa_over_CT", "k3_CT", "k4", "k4prime", "k6", "k7"]
P = len(PARAM_NAMES)
Y0 = np.array([0.9, 0.05, 0.0, 0.005, 0.3, 0.0])
T_SPAN = (0.0, 100.0)


def f_m(m, params):
    return params["k4prime"] + params["k4"] * (m / params["CT"]) ** 2


def df_m_dm(m, params):
    return params["k4"] * 2.0 * (m / params["CT"]) / params["CT"]


def rhs(_t, x, params):
    c2, cp, pm, m, y, yp = x
    k3 = params["k3_CT"] / params["CT"]
    k1 = params["k1_aa_over_CT"] * params["CT"]
    return np.array(
        [
            params["k6"] * m - params["k8_minusP"] * c2 + params["k9"] * cp,
            -k3 * cp * y + params["k8_minusP"] * c2 - params["k9"] * cp,
            k3 * cp * y - pm * f_m(m, params) + params["k5_minusP"] * m,
            pm * f_m(m, params) - params["k5_minusP"] * m - params["k6"] * m,
            k1 - params["k2"] * y - k3 * cp * y,
            params["k6"] * m - params["k7"] * yp,
        ]
    )


def jacobian_x(x, params):
    c2, cp, pm, m, y, yp = x
    k3 = params["k3_CT"] / params["CT"]
    df = df_m_dm(m, params)
    j = np.zeros((6, 6))
    j[0, 0] = -params["k8_minusP"]
    j[0, 1] = params["k9"]
    j[0, 3] = params["k6"]
    j[1, 0] = params["k8_minusP"]
    j[1, 1] = -k3 * y - params["k9"]
    j[1, 4] = -k3 * cp
    j[2, 1] = k3 * y
    j[2, 2] = -f_m(m, params)
    j[2, 3] = -pm * df + params["k5_minusP"]
    j[2, 4] = k3 * cp
    j[3, 2] = f_m(m, params)
    j[3, 3] = pm * df - params["k5_minusP"] - params["k6"]
    j[4, 1] = -k3 * y
    j[4, 4] = -params["k2"] - k3 * cp
    j[5, 3] = params["k6"]
    j[5, 5] = -params["k7"]
    return j


def df_dparam(x, params):
    c2, cp, pm, m, y, yp = x
    out = np.zeros((6, P))
    for j, name in enumerate(PARAM_NAMES):
        df = np.zeros(6)
        if name == "k1_aa_over_CT":
            df[4] = params["CT"]
        elif name == "k3_CT":
            coeff = 1.0 / params["CT"]
            df[1] = -coeff * cp * y
            df[2] = coeff * cp * y
            df[4] = -coeff * cp * y
        elif name == "k4":
            val = (m / params["CT"]) ** 2
            df[2] = -pm * val
            df[3] = pm * val
        elif name == "k4prime":
            df[2] = -pm
            df[3] = pm
        elif name == "k6":
            df[0] = m
            df[3] = -m
            df[5] = m
        elif name == "k7":
            df[5] = -yp
        else:
            raise KeyError(name)
        out[:, j] = df
    return out


def aug_rhs(t, z, params):
    x = z[:6]
    s = z[6:].reshape((6, P), order="F")
    xdot = rhs(t, x, params)
    a = jacobian_x(x, params)
    dfdh = df_dparam(x, params)
    dfdlogh = np.zeros_like(dfdh)
    for j, name in enumerate(PARAM_NAMES):
        dfdlogh[:, j] = params[name] * dfdh[:, j]
    sdot = a @ s + dfdlogh
    return np.concatenate([xdot, sdot.ravel(order="F")])


def quadrature_weights(t):
    dt = np.diff(t)
    q = np.zeros(len(t))
    q[0] = dt[0] / 2.0
    q[-1] = dt[-1] / 2.0
    q[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return q


def compute_hessian(params, t_eval, rtol=1e-6, atol=1e-8):
    z0 = np.concatenate([Y0, np.zeros(6 * P)])
    sol = solve_ivp(
        lambda t, z: aug_rhs(t, z, params),
        T_SPAN,
        z0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    weights = quadrature_weights(sol.t)
    total_time = T_SPAN[1] - T_SPAN[0]
    w = np.diag(np.ones(6) / (6.0 * total_time))
    hessian = np.zeros((P, P))
    for i in range(sol.t.size):
        st = sol.y[6:, i].reshape((6, P), order="F")
        hessian += weights[i] * st.T @ w @ st
    eigvals, eigvecs = eigh(hessian)
    return eigvals[::-1].copy(), eigvecs[:, ::-1].copy()


def simulate(params, t_eval, rtol=1e-6, atol=1e-8):
    sol = solve_ivp(
        lambda t, x: rhs(t, x, params),
        T_SPAN,
        Y0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y


def observables(x):
    c2, cp, pm, m, y, yp = x
    yt = y + yp + pm + m
    return {"YT": yt / BASE_PARAMS["CT"], "M": m / BASE_PARAMS["CT"]}


def step_size_for_lambda(lam, eigvals0, sloppy_indices0, base_step_size):
    lam0 = max(abs(float(eigvals0[sloppy_indices0[0]])), 1e-12)
    lam = max(abs(float(lam)), 1e-12)
    ratio = lam0 / lam
    return float(base_step_size * ratio ** 0.5)


def run_sloppy_walks(
    walk_steps=32,
    n_walks=4,
    base_step_size=1.0,
    sloppy_threshold=1e-3,
    n_time=801,
    noise=0.0,
    seed=1,
    initial_sign=1.0,
):
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], n_time)
    base = copy.deepcopy(BASE_PARAMS)
    base_x = simulate(base, t_eval)
    base_obs = observables(base_x)

    eigvals0, eigvecs0 = compute_hessian(base, t_eval)
    sloppy_indices0 = np.where(eigvals0 < sloppy_threshold)[0][::-1]
    if len(sloppy_indices0) == 0:
        raise RuntimeError(f"No sloppy modes found below {sloppy_threshold:g}")

    walks = []
    for walk_idx in range(min(n_walks, len(sloppy_indices0))):
        init_idx = int(sloppy_indices0[walk_idx])
        init_lambda = float(eigvals0[init_idx])
        direction_ref = float(initial_sign) * eigvecs0[:, init_idx].copy()
        current = copy.deepcopy(base)
        records = [
            {
                "step": 0,
                "params": copy.deepcopy(current),
                "obs": base_obs,
                "lambda": init_lambda,
                "distance": 0.0,
            }
        ]
        for step in range(1, walk_steps + 1):
            try:
                eigvals, eigvecs = compute_hessian(current, t_eval)
            except Exception as exc:
                records[-1]["stop_reason"] = f"hessian failed at step {step}: {exc}"
                break
            sloppy_local = np.where(eigvals < sloppy_threshold)[0][::-1]
            if len(sloppy_local) == 0:
                records[-1]["stop_reason"] = f"no sloppy mode below threshold at step {step}"
                break
            dir_idx = int(sloppy_local[min(walk_idx, len(sloppy_local) - 1)])
            direction = eigvecs[:, dir_idx].copy()
            if np.dot(direction, direction_ref) < 0:
                direction = -direction
            direction_ref = direction.copy()

            lam = float(eigvals[dir_idx])
            step_size = step_size_for_lambda(lam, eigvals0, sloppy_indices0, base_step_size)
            dlog = step_size * direction
            if noise > 0:
                dlog += rng.normal(0.0, np.sqrt(noise), size=P)

            nxt = copy.deepcopy(current)
            for j, name in enumerate(PARAM_NAMES):
                nxt[name] *= float(np.exp(dlog[j]))
            try:
                x = simulate(nxt, t_eval)
            except Exception as exc:
                records[-1]["stop_reason"] = f"simulation failed at step {step}: {exc}"
                break
            obs = observables(x)
            distance = float(np.sqrt(np.mean((obs["M"] - base_obs["M"]) ** 2)))
            records.append({"step": step, "params": copy.deepcopy(nxt), "obs": obs, "lambda": lam, "distance": distance})
            current = nxt
        walks.append({"walk_idx": walk_idx, "eig_idx": init_idx, "lambda0": init_lambda, "records": records})
    return t_eval, walks, eigvals0



PARAM_LABELS = {
    "k1_aa_over_CT": r"$\log k_1$",
    "k3_CT": r"$\log(k_3 C_T)$",
    "k4": r"$\log k_4$",
    "k4prime": r"$\log k_4^{\prime}$",
    "k6": r"$\log k_6$",
    "k7": r"$\log k_7$",
}


def save_walk_data(t_eval, walks, eigvals0, outdir):
    """Save compact numerical results so figures need not rerun the walks."""
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {"t": np.asarray(t_eval), "initial_eigenvalues": np.asarray(eigvals0)}
    summary = {"parameter_names": PARAM_NAMES, "walks": []}
    for i, walk in enumerate(walks):
        records = walk["records"]
        arrays[f"walk_{i}_log_parameters"] = np.array(
            [[np.log(record["params"][name]) for name in PARAM_NAMES] for record in records]
        )
        arrays[f"walk_{i}_M"] = np.array([record["obs"]["M"] for record in records])
        arrays[f"walk_{i}_YT"] = np.array([record["obs"]["YT"] for record in records])
        arrays[f"walk_{i}_distance"] = np.array([record["distance"] for record in records])
        arrays[f"walk_{i}_lambda"] = np.array([record["lambda"] for record in records])
        summary["walks"].append(
            {
                "walk": i + 1,
                "initial_eigenvalue": float(walk["lambda0"]),
                "completed_steps": len(records) - 1,
                "stop_reason": records[-1].get("stop_reason"),
            }
        )
    np.savez_compressed(outdir / "tyson_sloppy_multiwalk_data.npz", **arrays)
    (outdir / "tyson_sloppy_multiwalk_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _set_3d_limits(ax, coords):
    """Use stable, mildly padded limits without forcing a cubic data range."""
    for values, setter in zip(coords.T, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = float(values.min()), float(values.max())
        pad = max(0.08 * (hi - lo), 0.025)
        setter(lo - pad, hi + pad)


def plot_walk_parameter_trajectories(walks, outdir):
    """Plot four recomputed-Hessian walks in their most-varying coordinates."""
    outdir.mkdir(parents=True, exist_ok=True)
    n_walks = len(walks)
    ncols = 2
    nrows = int(np.ceil(n_walks / ncols))
    fig = plt.figure(figsize=(8.0, 6.25))
    max_step = max(len(walk["records"]) - 1 for walk in walks)
    norm = plt.Normalize(0, max_step)
    cmap = plt.get_cmap("viridis")

    for i, walk in enumerate(walks):
        records = walk["records"]
        log_params = np.array([[np.log(rec["params"][name]) for name in PARAM_NAMES] for rec in records])
        delta = np.abs(log_params[-1] - log_params[0])
        top3 = np.argsort(delta)[-3:][::-1]
        coords = log_params[:, top3]
        names = [PARAM_NAMES[j] for j in top3]
        steps = np.arange(len(records))

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        for j in range(len(records) - 1):
            ax.plot(
                coords[j : j + 2, 0],
                coords[j : j + 2, 1],
                coords[j : j + 2, 2],
                color=cmap(norm(j + 0.5)),
                lw=1.45,
                alpha=0.95,
            )
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c=steps, cmap=cmap,
            norm=norm, s=7, linewidths=0, depthshade=False
        )
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="#171717", s=31, marker="o", depthshade=False)
        ax.scatter(
            coords[-1, 0], coords[-1, 1], coords[-1, 2], color=cmap(norm(steps[-1])),
            edgecolor="#171717", linewidth=0.45, s=34, marker="o", depthshade=False
        )

        ax.text2D(0.01, 0.97, f"{chr(97 + i)}", transform=ax.transAxes, fontsize=10, weight="bold", va="top")
        ax.set_title(f"Sloppy mode {i + 1}", fontsize=9, pad=2)
        ax.text2D(
            0.99, 0.96, rf"$\lambda_0={walk['lambda0']:.2e}$",
            transform=ax.transAxes, fontsize=6.8, ha="right", va="top", color="#555555"
        )
        ax.set_xlabel(PARAM_LABELS[names[0]], fontsize=7.5, labelpad=-1)
        ax.set_ylabel(PARAM_LABELS[names[1]], fontsize=7.5, labelpad=-1)
        ax.set_zlabel(PARAM_LABELS[names[2]], fontsize=7.5, labelpad=-1)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.grid(True, alpha=0.25, linewidth=0.45)
        ax.xaxis.pane.set_alpha(0.035)
        ax.yaxis.pane.set_alpha(0.035)
        ax.zaxis.pane.set_alpha(0.035)
        ax.view_init(elev=23, azim=-56)
        _set_3d_limits(ax, coords)

        label_steps = np.unique(np.round(np.linspace(0, len(records) - 1, 6)).astype(int))
        for step in label_steps:
            ax.text(*coords[step], str(step), fontsize=5.5, color=cmap(norm(step)), weight="semibold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.935, 0.18, 0.016, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("walk step", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.01, right=0.89, bottom=0.03, top=0.96, wspace=0.03, hspace=0.07)
    stem = "tyson_sloppy_walk_parameter_trajectories"
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_walk_phenotypes(t_eval, walks, outdir, observable="M", snapshot_count=7):
    outdir.mkdir(parents=True, exist_ok=True)
    label = "active cdc2-cyclin / CT" if observable == "M" else "total cyclin / CT"
    colors = ["#4C78A8", "#9C6BC7", "#59A14F", "#E15759"]
    n_walks = len(walks)
    ncols = 2
    nrows = int(np.ceil(n_walks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 5.0), sharex=True, sharey=True, squeeze=False)

    for ax, walk, color in zip(axes.ravel(), walks, colors):
        records = walk["records"]
        n = len(records)
        snapshot_idx = np.unique(np.round(np.linspace(0, n - 1, min(snapshot_count, n))).astype(int))
        middle_idx = [i for i in snapshot_idx if i not in (0, n - 1)]
        for rank, idx in enumerate(middle_idx):
            shade = 0.78 - 0.35 * (rank / max(len(middle_idx) - 1, 1))
            ax.plot(t_eval, records[idx]["obs"][observable], color=str(shade), lw=1.0, alpha=0.95, zorder=1)
        ax.plot(t_eval, records[0]["obs"][observable], color="#111111", lw=1.7, zorder=3)
        ax.plot(t_eval, records[-1]["obs"][observable], color=color, lw=1.9, zorder=4)
        ax.text(0.03, 0.93, f"{chr(97 + walk['walk_idx'])}", transform=ax.transAxes, fontsize=10, weight="bold")
        ax.text(
            0.97,
            0.93,
            f"dir {walk['walk_idx'] + 1}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            color="#444444",
        )
        ax.grid(True, color="#d8dadd", alpha=0.65, linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(label, fontsize=9)
    for ax in axes.ravel()[n_walks:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="#111111", lw=1.7, label="start"),
        plt.Line2D([0], [0], color="#777777", lw=1.0, label="intermediate"),
        plt.Line2D([0], [0], color=colors[0], lw=1.9, label="end"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = f"tyson_sloppy_walk_{observable.lower()}_phenotypes"
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-steps", type=int, default=250)
    parser.add_argument("--n-walks", type=int, default=4)
    parser.add_argument("--base-step-size", type=float, default=1.0)
    parser.add_argument("--sloppy-threshold", type=float, default=1e-3)
    parser.add_argument("--n-time", type=int, default=801)
    parser.add_argument("--snapshot-count", type=int, default=7)
    parser.add_argument("--observable", choices=["M", "YT"], default="M")
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--initial-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "TysonSloppy_figures")
    args = parser.parse_args()

    t_eval, walks, eigvals0 = run_sloppy_walks(
        walk_steps=args.walk_steps,
        n_walks=args.n_walks,
        base_step_size=args.base_step_size,
        sloppy_threshold=args.sloppy_threshold,
        n_time=args.n_time,
        noise=args.noise,
        initial_sign=args.initial_sign,
    )
    plot_walk_phenotypes(t_eval, walks, args.outdir, observable=args.observable, snapshot_count=args.snapshot_count)
    plot_walk_parameter_trajectories(walks, args.outdir)
    save_walk_data(t_eval, walks, eigvals0, args.outdir)
    print("Initial eigenvalues:", " ".join(f"{x:.3e}" for x in eigvals0))
    print("Saved to", args.outdir)


if __name__ == "__main__":
    main()
