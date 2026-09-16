"""Split conformal prediction sets for binary fall/ADL gating."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..agents.confidence_gate import GateDecision

Route = Literal["adl", "fall", "ambiguous"]


@dataclass
class ConformalAudit:
    alpha: float
    q_hat: float
    n_cal: int
    edge_fn_rate: float
    edge_n: int
    escalation_rate: float
    set_size_hist: dict[int, int]


class ConformalGate:
    """Binary split-conformal gate: |C(x)|=1 → edge; |C(x)|=2 → escalate."""

    def __init__(self, q_hat: float = 0.5, alpha: float = 0.05):
        self.q_hat = float(q_hat)
        self.alpha = float(alpha)
        # Compatibility fields for GateDecision / logging
        self.tau_low = float(q_hat)
        self.tau_high = float(1.0 - q_hat)

    @staticmethod
    def _score(p_fall: float, y: int) -> float:
        p = float(p_fall)
        return 1.0 - (p if int(y) == 1 else 1.0 - p)

    @classmethod
    def calibrate(
        cls,
        p_falls: list[float],
        y_true: list[int],
        alpha: float = 0.05,
    ) -> ConformalGate:
        scores = np.asarray(
            [cls._score(p, y) for p, y in zip(p_falls, y_true, strict=True)],
            dtype=np.float64,
        )
        n = int(scores.size)
        if n == 0:
            return cls(q_hat=1.0, alpha=alpha)
        # Split conformal quantile: ceil((n+1)(1-alpha))-th smallest score
        k = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
        q_hat = float(np.sort(scores)[k - 1])
        return cls(q_hat=q_hat, alpha=alpha)

    def prediction_set(self, p_fall: float) -> set[int]:
        p = float(p_fall)
        out: set[int] = set()
        if p >= 1.0 - self.q_hat:
            out.add(1)
        if p <= self.q_hat:
            out.add(0)
        return out

    def decide(self, p_fall: float) -> GateDecision:
        s = self.prediction_set(p_fall)
        if s == {0}:
            route: Route = "adl"
        elif s == {1}:
            route = "fall"
        else:
            route = "ambiguous"
        return GateDecision(
            route=route,
            p_fall=float(p_fall),
            tau_low=self.tau_low,
            tau_high=self.tau_high,
        )

    def audit_edge_fn(
        self,
        p_falls: list[float],
        y_true: list[int],
    ) -> ConformalAudit:
        edge_fn = 0
        edge_n = 0
        esc = 0
        hist: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for p, y in zip(p_falls, y_true, strict=True):
            s = self.prediction_set(p)
            hist[len(s)] = hist.get(len(s), 0) + 1
            if len(s) != 1:
                esc += 1
                continue
            edge_n += 1
            pred = 1 if 1 in s else 0
            if int(y) == 1 and pred == 0:
                edge_fn += 1
        return ConformalAudit(
            alpha=self.alpha,
            q_hat=self.q_hat,
            n_cal=0,
            edge_fn_rate=float(edge_fn / max(1, edge_n)),
            edge_n=edge_n,
            escalation_rate=float(esc / max(1, len(y_true))),
            set_size_hist=hist,
        )
