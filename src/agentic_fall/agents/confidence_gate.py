from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Route = Literal["adl", "fall", "ambiguous"]


@dataclass
class GateDecision:
    route: Route
    p_fall: float
    tau_low: float
    tau_high: float


class ConfidenceGate:
    """Dual-threshold uncertainty gate (τ_low / τ_high)."""

    def __init__(self, tau_low: float = 0.30, tau_high: float = 0.85):
        if not (0.0 <= tau_low < tau_high <= 1.0):
            raise ValueError("Require 0 <= tau_low < tau_high <= 1")
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)

    def decide(self, p_fall: float) -> GateDecision:
        p = float(p_fall)
        if p >= self.tau_high:
            route: Route = "fall"
        elif p <= self.tau_low:
            route = "adl"
        else:
            route = "ambiguous"
        return GateDecision(route=route, p_fall=p, tau_low=self.tau_low, tau_high=self.tau_high)

    def calibrate(
        self,
        p_falls: list[float],
        y_true: list[int],
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        grid: int = 20,
    ) -> tuple[float, float]:
        """Grid-search thresholds minimizing expected misclassification cost.

        For calibration we treat mid-band as 'defer' (no cost here) and only
        score confident decisions; prefer wider ambiguous band when uncertain.
        """
        import numpy as np

        ps = np.asarray(p_falls, dtype=np.float64)
        ys = np.asarray(y_true, dtype=np.int64)
        best = (self.tau_low, self.tau_high)
        best_cost = float("inf")
        lows = np.linspace(0.05, 0.45, grid)
        highs = np.linspace(0.55, 0.95, grid)
        for lo in lows:
            for hi in highs:
                if lo >= hi:
                    continue
                pred = np.full_like(ys, fill_value=-1)  # -1 = defer
                pred[ps >= hi] = 1
                pred[ps <= lo] = 0
                mask = pred >= 0
                if mask.sum() < max(10, int(0.2 * len(ys))):
                    continue
                fp = ((pred == 1) & (ys == 0) & mask).sum()
                fn = ((pred == 0) & (ys == 1) & mask).sum()
                # mild penalty for too much deferral
                defer = (~mask).mean()
                cost = cost_fp * fp + cost_fn * fn + 0.5 * defer * len(ys)
                if cost < best_cost:
                    best_cost = float(cost)
                    best = (float(lo), float(hi))
        self.tau_low, self.tau_high = best
        return best
