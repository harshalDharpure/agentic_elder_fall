from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from ..features.biomechanics import BiomechanicalFeatures
from .constraints import ConstraintPack, rationale_cites_sigma
from .llm_reasoner import _extract_json, _heuristic_reason

ACTION_ORDER = ["log", "monitor", "notify_caregiver", "emergency"]


@dataclass
class ActorHypothesis:
    hypothesis: str  # fall | near_fall | adl
    prediction: str  # fall | adl
    severity: str
    confidence: float
    preconditions: list[str] = field(default_factory=list)
    rationale: str = ""
    suggested_action: str = "log"
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriticVerdict:
    verdict: str  # confirm | reject | revise
    revised_prediction: str  # fall | adl
    revised_severity: str
    revised_action: str
    confidence: float
    violations: list[str] = field(default_factory=list)
    critique: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACTOR_SYSTEM = """You are the ACTOR in a clinical fall-detection adjudication team.
Given biomechanical evidence and similar historical cases, propose whether the event is a FALL or ADL (including near-fall/stumble).
Respond with ONLY valid JSON:
{
  "hypothesis": "fall" | "near_fall" | "adl",
  "prediction": "fall" or "adl",
  "severity": "mild" | "moderate" | "severe" | "none",
  "confidence": 0.0-1.0,
  "preconditions": ["list of biomechanical conditions supporting a fall"],
  "rationale": "step-by-step reasoning in one string",
  "action": "emergency" | "notify_caregiver" | "monitor" | "log"
}
Be conservative on missed falls: when evidence is mixed but stillness and tilt are high, prefer fall.
Near-falls with short free-fall, moderate impact, brief stillness, partial tilt are usually
hypothesis=near_fall with prediction=adl and action=monitor.
"""

CRITIC_SYSTEM = """You are the CRITIC in a clinical fall-detection team.
Do NOT change the Actor fall/ADL label. Only critique action/severity and whether the rationale cites Σ(w).
You will see POSITIVE past falls and HARD-NEGATIVE ADLs that looked similar but were not falls.
Downgrade emergency/notify when impact is not severe, tilt is not horizontal, stillness is brief,
or hard-negative ADLs outvote similar falls.
Respond with ONLY valid JSON:
{
  "verdict": "confirm" | "revise",
  "revised_prediction": "<copy Actor prediction exactly>",
  "revised_severity": "mild" | "moderate" | "severe" | "none",
  "revised_action": "emergency" | "notify_caregiver" | "monitor" | "log",
  "confidence": 0.0-1.0,
  "violations": ["action or grounding issues only"],
  "critique": "short justification citing checklist and hard negatives"
}
"""


def _heuristic_actor(
    feats: BiomechanicalFeatures,
    p_fall: float,
    constraints: ConstraintPack,
    cases: list | None = None,
) -> ActorHypothesis:
    """Recall-oriented proposer (lower decision threshold than Critic)."""
    from .knn_memory import MemoryCase

    case_list: list[MemoryCase] = list(cases or [])
    base = _heuristic_reason(feats, p_fall, case_list)
    # Slightly more aggressive than base heuristic: boost ambiguous band toward fall
    if base.prediction == "adl" and constraints.fall_support_score >= 0.05 and p_fall >= 0.4:
        base = _heuristic_reason(feats, min(1.0, p_fall + 0.15), case_list)
        base = type(base)(
            prediction=base.prediction,
            severity=base.severity if base.prediction == "fall" else "none",
            confidence=base.confidence,
            rationale="Actor recall bias: " + base.rationale,
            action=base.action if base.prediction == "fall" else "monitor",
            source="heuristic",
        )
    if constraints.near_fall_likely and base.prediction == "adl":
        hyp = "near_fall"
    elif base.prediction == "fall":
        hyp = "fall"
    else:
        hyp = "adl"
    preconds = [
        f"freefall_band={constraints.freefall_band}",
        f"impact_band={constraints.impact_band}",
        f"stillness_band={constraints.stillness_band}",
        f"tilt_band={constraints.tilt_band}",
        f"fall_support_score>={constraints.fall_support_score:.2f}",
    ]
    return ActorHypothesis(
        hypothesis=hyp,
        prediction=base.prediction,
        severity=base.severity,
        confidence=base.confidence,
        preconditions=preconds,
        rationale=base.rationale,
        suggested_action=base.action,
        source="heuristic",
    )


def _clamp_action(action: str, max_action: str) -> str:
    if action not in ACTION_ORDER:
        action = "monitor"
    if ACTION_ORDER.index(action) > ACTION_ORDER.index(max_action):
        return max_action
    return action


def _heuristic_critic(
    actor: ActorHypothesis,
    constraints: ConstraintPack,
) -> CriticVerdict:
    """Action/rationale skeptic. Never changes fall/ADL."""
    violations: list[str] = []
    pred = actor.prediction if actor.prediction in ("fall", "adl") else "adl"
    severity = actor.severity
    action = actor.suggested_action if actor.suggested_action in ACTION_ORDER else "log"

    if not rationale_cites_sigma(actor.rationale):
        violations.append("rationale missing Σ(w) citations")

    weak_biomechanics = (
        constraints.impact_band in ("low", "moderate")
        and constraints.stillness_band == "brief"
        and constraints.tilt_band != "horizontal"
    )
    hard_neg_dominate = constraints.knn_adl_votes > constraints.knn_fall_votes
    soft_support = float(constraints.fall_support_score) < 0.20

    if pred == "adl":
        if action in ("emergency", "notify_caregiver"):
            violations.append(f"{action} is too severe for an ADL/near-fall label")
            action = "monitor" if (
                constraints.near_fall_likely
                or constraints.impact_band in ("moderate", "high", "severe")
                or constraints.freefall_band != "short"
            ) else "log"
            severity = "none"
    else:
        emergency_supported = (
            constraints.impact_band == "severe"
            or constraints.tilt_band == "horizontal"
            or constraints.stillness_band == "prolonged"
        )
        if action == "emergency" and not emergency_supported:
            violations.append("emergency not supported by window-scale impact/tilt/stillness")
            action = "notify_caregiver"
            if severity == "severe":
                severity = "moderate"
        # Visible Critic impact: downgrade notify→monitor when Σ(w) / hard negatives
        # argue the event is stumble-like even if the Actor kept the fall label.
        if action in ("emergency", "notify_caregiver") and (
            constraints.near_fall_likely or weak_biomechanics or hard_neg_dominate or soft_support
        ):
            reasons = []
            if constraints.near_fall_likely:
                reasons.append("near-fall-like Σ")
            if weak_biomechanics:
                reasons.append("weak impact/stillness/tilt")
            if hard_neg_dominate:
                reasons.append(
                    f"hard-neg kNN ({constraints.knn_adl_votes} ADL > {constraints.knn_fall_votes} fall)"
                )
            if soft_support:
                reasons.append(f"low fall-support ({constraints.fall_support_score:.2f})")
            violations.append("fall action over-severe: " + "; ".join(reasons))
            action = "monitor"
            if severity in ("severe", "moderate"):
                severity = "mild"

    if not violations:
        return CriticVerdict(
            verdict="confirm",
            revised_prediction=pred,
            revised_severity=severity,
            revised_action=action,
            confidence=min(0.95, actor.confidence + 0.05),
            violations=[],
            critique="Actor label kept. Action and rationale consistent with Σ(w).",
            source="heuristic",
        )

    return CriticVerdict(
        verdict="revise",
        revised_prediction=pred,
        revised_severity=severity,
        revised_action=action,
        confidence=0.8,
        violations=violations,
        critique="Label frozen. " + "; ".join(violations) + f" → action={action}.",
        source="heuristic",
    )


class CriticAgent:
    """Constraint-oriented skeptic that validates Actor hypotheses."""

    def __init__(
        self,
        backend: str = "ollama",
        base_url: str = "http://127.0.0.1:11434",
        model: str = "mistral:latest",
        temperature: float = 0.1,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        fallback_heuristic: bool = True,
    ):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.fallback_heuristic = fallback_heuristic

    def _ollama(self, system: str, user: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "prompt": f"{system}\n\n{user}\n\nJSON:",
                    "stream": False,
                    "options": {"temperature": self.temperature},
                }
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_s,
                )
                r.raise_for_status()
                raw = r.json().get("response", "")
                return _extract_json(raw)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Critic LLM call failed: {last_err}")

    def propose_actor(
        self,
        evidence_text: str,
        feats: BiomechanicalFeatures,
        p_fall: float,
        constraints: ConstraintPack,
        cases_text: str,
        cases: list | None = None,
    ) -> ActorHypothesis:
        if self.backend == "heuristic":
            return _heuristic_actor(feats, p_fall, constraints, cases=cases)

        user = (
            f"Evidence:\n{evidence_text}\n\n"
            f"Constraint checklist Σ(w):\n{constraints.checklist_text()}\n\n"
            f"Similar cases:\n{cases_text}\n\n"
            "Propose hypothesis now."
        )
        try:
            obj = self._ollama(ACTOR_SYSTEM, user)
            pred = str(obj.get("prediction", "adl")).lower()
            if pred not in ("fall", "adl"):
                pred = "fall" if p_fall >= 0.5 else "adl"
            hyp = str(obj.get("hypothesis", pred)).lower()
            if hyp not in ("fall", "near_fall", "adl"):
                hyp = "near_fall" if pred == "adl" else "fall"
            preconds = obj.get("preconditions", [])
            if not isinstance(preconds, list):
                preconds = [str(preconds)]
            return ActorHypothesis(
                hypothesis=hyp,
                prediction=pred,
                severity=str(obj.get("severity", "none")).lower(),
                confidence=float(obj.get("confidence", 0.7)),
                preconditions=[str(p) for p in preconds],
                rationale=str(obj.get("rationale", "")),
                suggested_action=str(obj.get("action", "log")).lower(),
                source="ollama",
            )
        except Exception:
            if self.fallback_heuristic:
                return _heuristic_actor(feats, p_fall, constraints, cases=cases)
            raise

    def critique(
        self,
        actor: ActorHypothesis,
        evidence_text: str,
        constraints: ConstraintPack,
    ) -> CriticVerdict:
        if self.backend == "heuristic":
            return _heuristic_critic(actor, constraints)

        user = (
            f"Actor hypothesis JSON:\n{json.dumps(actor.to_dict())}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Constraint checklist Σ(w):\n{constraints.checklist_text()}\n\n"
            "Validate the hypothesis against constraints only."
        )
        try:
            obj = self._ollama(CRITIC_SYSTEM, user)
            verdict = str(obj.get("verdict", "confirm")).lower()
            if verdict not in ("confirm", "reject", "revise"):
                verdict = "confirm"
            pred = actor.prediction
            violations = obj.get("violations", [])
            if not isinstance(violations, list):
                violations = [str(violations)]
            return CriticVerdict(
                verdict="revise" if verdict == "reject" else verdict,
                revised_prediction=pred,
                revised_severity=str(obj.get("revised_severity", actor.severity)).lower(),
                revised_action=str(obj.get("revised_action", actor.suggested_action)).lower(),
                confidence=float(obj.get("confidence", 0.7)),
                violations=[str(v) for v in violations],
                critique=str(obj.get("critique", "")),
                source="ollama",
            )
        except Exception:
            if self.fallback_heuristic:
                return _heuristic_critic(actor, constraints)
            raise
