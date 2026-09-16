"""Conformal risk control for one-directional Actor-fall vetoes.

Risk of threshold λ is the induced missed-fall rate among vetoed windows:
    R(λ) = P(y = fall | s(w) ≥ λ)
We certify an upper bound U(λ) ≥ R(λ) with probability ≥ 1-δ and pick
    λ* = min { λ : U(λ) ≤ α }.
If no such λ exists, vetoing is infeasible (threshold = +∞).
"""
from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

import numpy as np

BoundMethod = Literal["bernstein", "hoeffding"]


def alpha_break_even(cost_fn: float = 10.0, cost_fp: float = 1.0) -> float:
    """Largest induced miss rate at which vetoing still lowers  cost_fn·FN + cost_fp·FP."""
    return float(cost_fp / (cost_fn + cost_fp))


def hoeffding_upper(r_hat: float, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    slack = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return float(min(1.0, r_hat + slack))


def empirical_bernstein_upper(losses: Sequence[float], delta: float) -> float:
    """Maurer–Pontil empirical Bernstein bound on the mean of [0, 1] losses."""
    arr = np.asarray(list(losses), dtype=float)
    n = int(arr.size)
    if n <= 1:
        return 1.0
    r_hat = float(arr.mean())
    var = float(arr.var(ddof=1))
    log_term = math.log(2.0 / delta)
    slack = math.sqrt(2.0 * var * log_term / n) + (7.0 * log_term) / (3.0 * (n - 1))
    return float(min(1.0, r_hat + slack))


def _eligible(scores: np.ndarray, labels: np.ndarray, lam: float) -> np.ndarray:
    return scores >= lam


def risk_at_lambda(
    scores: Sequence[float],
    labels: Sequence[int],
    lam: float,
    *,
    delta: float = 0.1,
    method: BoundMethod = "bernstein",
) -> dict[str, float]:
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    mask = _eligible(s, y, lam)
    n = int(mask.sum())
    if n == 0:
        return {
            "lambda": float(lam),
            "n": 0,
            "r_hat": 0.0,
            "U": 1.0,
            "feasible": False,
        }
    losses = y[mask].astype(float)  # 1 iff veto would miss a fall
    r_hat = float(losses.mean())
    if method == "hoeffding":
        u = hoeffding_upper(r_hat, n, delta)
    else:
        u = empirical_bernstein_upper(losses, delta)
    return {
        "lambda": float(lam),
        "n": float(n),
        "r_hat": r_hat,
        "U": u,
        "feasible": bool(u <= 1.0),
    }


def select_lambda_star(
    scores: Sequence[float],
    labels: Sequence[int],
    alpha: float,
    *,
    delta: float = 0.1,
    method: BoundMethod = "bernstein",
    min_n: int = 20,
    grid: Iterable[float] | None = None,
) -> dict[str, float | bool | None]:
    """Most permissive (smallest) λ with U(λ) ≤ α and |E_λ| ≥ min_n.

    Returns ``lambda_star=None`` and ``feasible=False`` when nothing certifies.
    """
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if s.size == 0:
        return {
            "lambda_star": None,
            "feasible": False,
            "alpha": float(alpha),
            "delta": float(delta),
            "method": method,
            "n_eligible": 0,
            "r_hat": None,
            "U": None,
        }
    if grid is None:
        qs = np.quantile(s, np.linspace(0.0, 1.0, 51))
        grid_vals = np.unique(np.concatenate([qs, np.linspace(0.0, 1.0, 21), s]))
    else:
        grid_vals = np.unique(np.asarray(list(grid), dtype=float))
    grid_vals = np.sort(grid_vals)

    chosen: dict[str, float | bool | None] | None = None
    for lam in grid_vals:
        stats = risk_at_lambda(s, y, float(lam), delta=delta, method=method)
        if int(stats["n"]) < min_n:
            continue
        if stats["U"] <= alpha:
            chosen = {
                "lambda_star": float(lam),
                "feasible": True,
                "alpha": float(alpha),
                "delta": float(delta),
                "method": method,
                "n_eligible": int(stats["n"]),
                "r_hat": float(stats["r_hat"]),
                "U": float(stats["U"]),
            }
            break
    if chosen is None:
        return {
            "lambda_star": None,
            "feasible": False,
            "alpha": float(alpha),
            "delta": float(delta),
            "method": method,
            "n_eligible": int(s.size),
            "r_hat": None,
            "U": None,
        }
    return chosen


def risk_coverage_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    delta: float = 0.1,
    method: BoundMethod = "bernstein",
    n_grid: int = 51,
) -> list[dict[str, float]]:
    """Coverage = share of eligible (Actor-fall) windows *kept* as fall."""
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if s.size == 0:
        return []
    grid = np.unique(np.concatenate([np.quantile(s, np.linspace(0.0, 1.0, n_grid)), [0.0, 1.0]]))
    n = float(s.size)
    rows = []
    for lam in grid:
        stats = risk_at_lambda(s, y, float(lam), delta=delta, method=method)
        veto = s >= lam
        coverage = float((~veto).mean()) if n else 1.0
        rows.append(
            {
                "lambda": float(lam),
                "coverage": coverage,
                "veto_rate": float(veto.mean()) if n else 0.0,
                "r_hat": float(stats["r_hat"]),
                "U": float(stats["U"]),
                "n_veto": float(stats["n"]),
            }
        )
    return rows


def area_under_risk_coverage(curve: Sequence[dict[str, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    xs = np.asarray([r["coverage"] for r in curve], dtype=float)
    ys = np.asarray([r["r_hat"] for r in curve], dtype=float)
    order = np.argsort(xs)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ys[order], xs[order]))


def apply_veto(
    actor_pred: Sequence[str],
    scores: Sequence[float],
    lam: float | None,
) -> list[str]:
    """One-directional: fall→adl iff score ≥ λ. None/inf λ means no veto."""
    if lam is None or not math.isfinite(float(lam)):
        return [str(p) for p in actor_pred]
    out = []
    for p, s in zip(actor_pred, scores):
        if p == "fall" and float(s) >= float(lam):
            out.append("adl")
        else:
            out.append(str(p))
    return out
