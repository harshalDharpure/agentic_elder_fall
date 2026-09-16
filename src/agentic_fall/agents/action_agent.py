from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..features.biomechanics import BiomechanicalFeatures

ActionName = Literal["emergency", "notify_caregiver", "monitor", "log"]


@dataclass
class ActionDecision:
    action: ActionName
    severity: str
    reason: str
    simulated: bool = True


class ActionAgent:
    """Cost-sensitive graded response policy (simulated side-effects)."""

    def __init__(
        self,
        impact_severe_g: float = 5.0,
        impact_moderate_g: float = 3.0,
        stillness_severe_s: float = 10.0,
        stillness_moderate_s: float = 3.0,
        simulate: bool = True,
    ):
        self.impact_severe_g = impact_severe_g
        self.impact_moderate_g = impact_moderate_g
        self.stillness_severe_s = stillness_severe_s
        self.stillness_moderate_s = stillness_moderate_s
        self.simulate = simulate

    def decide(
        self,
        prediction: str,
        severity: str | None = None,
        feats: BiomechanicalFeatures | None = None,
        suggested_action: str | None = None,
        allow_downgrade: bool = False,
    ) -> ActionDecision:
        pred = prediction.lower()
        if pred == "adl":
            # near-fall monitoring if evidence suggests risk
            if feats and (
                feats.impact_magnitude_g >= 2.0 or feats.freefall_duration_s >= 0.12
            ):
                return ActionDecision(
                    action="monitor",
                    severity="none",
                    reason="ADL/near-fall — enhanced monitoring",
                    simulated=self.simulate,
                )
            return ActionDecision(
                action="log",
                severity="none",
                reason="ADL — log only",
                simulated=self.simulate,
            )

        # Fall path
        sev = (severity or "moderate").lower()
        if feats is not None:
            if (
                feats.impact_magnitude_g >= self.impact_severe_g
                or feats.post_impact_stillness_s >= self.stillness_severe_s
            ):
                sev = "severe"
            elif (
                feats.impact_magnitude_g >= self.impact_moderate_g
                or feats.post_impact_stillness_s >= self.stillness_moderate_s
            ):
                sev = "moderate"
            else:
                sev = "mild"

        mapping = {
            "severe": ("emergency", "Severe fall — emergency call (simulated)"),
            "moderate": ("notify_caregiver", "Moderate fall — notify caregiver (simulated)"),
            "mild": ("monitor", "Mild fall — monitor + check-in (simulated)"),
        }
        action, reason = mapping.get(sev, mapping["moderate"])

        # Prefer LLM suggestion when it is more conservative (higher severity action)
        order = ["log", "monitor", "notify_caregiver", "emergency"]
        if suggested_action and suggested_action in order:
            sug_i = order.index(suggested_action)
            cur_i = order.index(action)
            if sug_i > cur_i:
                action = suggested_action  # type: ignore
                reason = f"Escalated via LLM suggestion to {action}"
            elif allow_downgrade and sug_i < cur_i:
                action = suggested_action  # type: ignore
                reason = f"Downgraded via Critic/Judge suggestion to {action}"

        return ActionDecision(action=action, severity=sev, reason=reason, simulated=self.simulate)  # type: ignore