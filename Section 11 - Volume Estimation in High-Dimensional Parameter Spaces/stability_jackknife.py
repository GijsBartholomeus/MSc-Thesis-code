"""
Jackknife stability / convergence diagnostics over *independent full runs*.

Design goals:
- Treat the simulation as a black box: each run returns final scalar(s).
- Jackknife over runs (observations), NOT over internal time steps.
- Minimal, modular, dependency-light (numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike1D = Union[Sequence[float], np.ndarray]
RunResult = Union[ArrayLike1D, Dict[str, float]]


@dataclass(frozen=True)
class JackknifeResult:
    names: List[str]
    theta_hat: np.ndarray  # (p,)
    jackknife_replicates: np.ndarray  # (n, p)
    bias: np.ndarray  # (p,)
    variance: np.ndarray  # (p,)
    se: np.ndarray  # (p,)
    rel_bias: np.ndarray  # (p,)
    rel_se: np.ndarray  # (p,)
    rel_bias_threshold: float
    rel_se_threshold: float
    ok_rel_bias: np.ndarray  # (p,) bool
    ok_rel_se: np.ndarray  # (p,) bool


def _coerce_runs_to_matrix(
    runs: Sequence[RunResult],
    *,
    names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if len(runs) == 0:
        raise ValueError("Need at least 1 run to compute diagnostics.")

    first = runs[0]
    if isinstance(first, dict):
        if names is None:
            names_list = sorted(first.keys())
        else:
            names_list = list(names)

        X = np.empty((len(runs), len(names_list)), dtype=float)
        for i, r in enumerate(runs):
            if not isinstance(r, dict):
                raise TypeError("Mixed run result types: expected dict for all runs.")
            for j, k in enumerate(names_list):
                if k not in r:
                    raise KeyError(f"Run {i} missing key '{k}'.")
                X[i, j] = float(r[k])
        return X, names_list

    # array-like path
    X = np.asarray(runs, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected runs to form a 2D array, got shape {X.shape}.")
    p = X.shape[1]
    if names is None:
        names_list = [f"stat_{j}" for j in range(p)]
    else:
        names_list = list(names)
        if len(names_list) != p:
            raise ValueError(f"names has length {len(names_list)} but runs have p={p} stats.")
    return X, names_list


def jackknife_over_runs(
    runs_matrix: np.ndarray,
    *,
    statistic_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute jackknife replicates and (bias, variance) estimates.

    runs_matrix: shape (n, p) where each row is one independent full run's final stats.
    statistic_fn: maps (m, p) -> (p,). Default is column-wise mean.

    Returns: (theta_hat, theta_j, bias, variance)
    - theta_hat: statistic on full data, shape (p,)
    - theta_j: leave-one-out replicates, shape (n, p)
    - bias: jackknife bias estimate, shape (p,)
    - variance: jackknife variance estimate, shape (p,)
    """
    X = np.asarray(runs_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"runs_matrix must be 2D (n,p), got shape {X.shape}.")
    n, p = X.shape
    if n < 2:
        raise ValueError("Jackknife needs at least 2 independent runs (n >= 2).")

    if statistic_fn is None:
        statistic_fn = lambda A: np.mean(A, axis=0)

    theta_hat = np.asarray(statistic_fn(X), dtype=float).reshape((p,))
    theta_j = np.empty((n, p), dtype=float)
    for i in range(n):
        X_loo = np.delete(X, i, axis=0)
        theta_j[i, :] = np.asarray(statistic_fn(X_loo), dtype=float).reshape((p,))

    theta_bar = np.mean(theta_j, axis=0)
    bias = (n - 1) * (theta_bar - theta_hat)
    variance = (n - 1) / n * np.sum((theta_j - theta_bar) ** 2, axis=0)
    return theta_hat, theta_j, bias, variance


def jackknife_diagnostic(
    runs: Sequence[RunResult],
    *,
    names: Optional[Sequence[str]] = None,
    statistic_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rel_bias_threshold: float = 0.05,
    rel_se_threshold: float = 0.05,
    eps: float = 1e-12,
) -> JackknifeResult:
    """
    Convenience wrapper:
    - Coerces per-run results into (n,p).
    - Computes jackknife bias/variance for statistic_fn (default: mean).
    - Computes relative bias and relative standard error vs |theta_hat|.
    """
    X, names_list = _coerce_runs_to_matrix(runs, names=names)
    theta_hat, theta_j, bias, variance = jackknife_over_runs(X, statistic_fn=statistic_fn)
    se = np.sqrt(np.maximum(variance, 0.0))

    denom = np.maximum(np.abs(theta_hat), eps)
    rel_bias = np.abs(bias) / denom
    rel_se = se / denom
    ok_rel_bias = rel_bias <= rel_bias_threshold
    ok_rel_se = rel_se <= rel_se_threshold

    return JackknifeResult(
        names=names_list,
        theta_hat=theta_hat,
        jackknife_replicates=theta_j,
        bias=bias,
        variance=variance,
        se=se,
        rel_bias=rel_bias,
        rel_se=rel_se,
        rel_bias_threshold=rel_bias_threshold,
        rel_se_threshold=rel_se_threshold,
        ok_rel_bias=ok_rel_bias,
        ok_rel_se=ok_rel_se,
    )


def run_jackknife(
    run_once: Callable[[int], RunResult],
    *,
    n_runs: int,
    seed0: int = 0,
    seeds: Optional[Sequence[int]] = None,
    names: Optional[Sequence[str]] = None,
    statistic_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rel_bias_threshold: float = 0.05,
    rel_se_threshold: float = 0.05,
    eps: float = 1e-12,
) -> Tuple[List[int], List[RunResult], JackknifeResult]:
    """
    Run the black-box simulation multiple times with different seeds, then jackknife.
    """
    if seeds is None:
        if n_runs <= 0:
            raise ValueError("n_runs must be positive.")
        seeds_list = list(range(seed0, seed0 + n_runs))
    else:
        seeds_list = list(seeds)
        if len(seeds_list) < 2:
            raise ValueError("Jackknife needs at least 2 seeds/runs.")
        n_runs = len(seeds_list)

    runs: List[RunResult] = []
    for s in seeds_list:
        runs.append(run_once(int(s)))

    result = jackknife_diagnostic(
        runs,
        names=names,
        statistic_fn=statistic_fn,
        rel_bias_threshold=rel_bias_threshold,
        rel_se_threshold=rel_se_threshold,
        eps=eps,
    )
    return seeds_list, runs, result


