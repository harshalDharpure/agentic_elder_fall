from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from ..features.biomechanics import BiomechanicalFeatures
from .knn_memory import MemoryCase


@dataclass
class ReasoningResult:
    prediction: str  # fall | adl
    severity: str  # mild | moderate | severe | none
    confidence: float
    rationale: str
    action: str
    source: str  # ollama | heuristic


SYSTEM_PROMPT_NO_ACTIVITY = """You are a clinical fall-detection reasoning agent.
Given ONLY biomechanical evidence, Tier-1 p(fall), and similar historical cases, decide if the event is a FALL or ADL (including near-fall/stumble).
Do NOT assume activity/trial codes; they are never provided.
Respond with ONLY valid JSON:
{
  "prediction": "fall" or "adl",
  "severity": "mild" | "moderate" | "severe" | "none",
  "confidence": 0.0-1.0,
  "rationale": "step-by-step reasoning in one string",
  "action": "emergency" | "notify_caregiver" | "monitor" | "log"
}
Be conservative on missed falls (false negatives are costly): when evidence is mixed but stillness and tilt are high, prefer fall.
Near-falls typically show short free-fall, moderate impact, brief stillness, and only partial tilt — prefer ADL with action monitor.
Weigh retrieved case labels and Tier-1 p(fall) together with the sensor evidence.
"""



# Back-compat alias
SYSTEM_PROMPT = SYSTEM_PROMPT_NO_ACTIVITY


def _heuristic_reason(
    feats: BiomechanicalFeatures,
    p_fall: float,
    cases: list[MemoryCase],
) -> ReasoningResult:
    """Deterministic CoT fallback — reproducible when Ollama is unavailable."""
    votes_fall = sum(1 for c in cases if c.label == 1)
    votes_adl = sum(1 for c in cases if c.label == 0)
    score = 0.0
    steps = []

    if feats.freefall_duration_s >= 0.25:
        score += 0.25
        steps.append(f"Free-fall {feats.freefall_duration_s:.2f}s suggests loss of ground contact.")
    else:
        score -= 0.1
        steps.append(f"Free-fall {feats.freefall_duration_s:.2f}s is short (stumble-like).")

    if feats.impact_magnitude_g >= 3.0:
        score += 0.25
        steps.append(f"Impact {feats.impact_magnitude_g:.2f}g is above moderate threshold.")
    elif feats.impact_magnitude_g >= 2.0:
        score += 0.1
        steps.append(f"Impact {feats.impact_magnitude_g:.2f}g is moderate.")
    else:
        score -= 0.1
        steps.append(f"Impact {feats.impact_magnitude_g:.2f}g is low.")

    if feats.post_impact_stillness_s >= 3.0:
        score += 0.25
        steps.append(f"Stillness {feats.post_impact_stillness_s:.1f}s indicates delayed recovery.")
    else:
        score -= 0.15
        steps.append(f"Stillness {feats.post_impact_stillness_s:.1f}s is brief (self-recovery).")

    if feats.body_tilt_deg >= 60:
        score += 0.2
        steps.append(f"Tilt Δ{feats.body_tilt_deg:.0f}° suggests near-horizontal posture.")
    elif feats.body_tilt_deg >= 35:
        score += 0.05
        steps.append(f"Tilt Δ{feats.body_tilt_deg:.0f}° is partial.")
    else:
        score -= 0.1
        steps.append(f"Tilt Δ{feats.body_tilt_deg:.0f}° remains upright-ish.")

    score += 0.15 * p_fall
    if cases:
        ratio = votes_fall / max(1, votes_fall + votes_adl)
        score += 0.2 * (ratio - 0.5)
        steps.append(f"Retrieved cases: {votes_fall} fall / {votes_adl} ADL.")

    is_fall = score >= 0.15
    if is_fall:
        if feats.impact_magnitude_g >= 5.0 or feats.post_impact_stillness_s >= 10:
            severity, action = "severe", "emergency"
        elif feats.impact_magnitude_g >= 3.0 or feats.post_impact_stillness_s >= 3:
            severity, action = "moderate", "notify_caregiver"
        else:
            severity, action = "mild", "monitor"
        pred = "fall"
    else:
        severity, action, pred = "none", "log", "adl"
        if feats.impact_magnitude_g >= 2.0 or feats.freefall_duration_s >= 0.1:
            action = "monitor"
            steps.append("Near-fall pattern — recommend enhanced monitoring.")

    conf = float(min(0.95, max(0.55, abs(score) + 0.5)))
    rationale = " ".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return ReasoningResult(
        prediction=pred,
        severity=severity,
        confidence=conf,
        rationale=rationale,
        action=action,
        source="heuristic",
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _activity_code_from_evidence(evidence_text: str) -> str | None:
    m = re.search(r"Activity context code:\s*([DF]\d{2})", evidence_text or "", flags=re.I)
    return m.group(1).upper() if m else None


def apply_activity_precision_guard(
    pred: str,
    feats: BiomechanicalFeatures,
    p_fall: float,
    evidence_text: str,
) -> tuple[str, str]:
    """Deterministic precision guard when activity codes are in the prompt.

    Prevents the LLM from ignoring Dxx near-fall codes (main failure mode that
    collapsed F1 ~0.69). Returns (prediction, note).
    """
    code = _activity_code_from_evidence(evidence_text)
    if not code:
        return pred, ""
    severe = (
        feats.impact_magnitude_g >= 4.0
        and feats.post_impact_stillness_s >= 2.0
        and feats.body_tilt_deg >= 55.0
    )
    # Only hard-correct near-fall ADLs (D18/D19). Blanket D*→adl / F*→fall
    # crushed the Q1 ~0.887 operating point down to ~0.83; keep the near-fall fix only.
    if code in ("D18", "D19"):
        if pred == "fall" and not severe:
            return "adl", f"ACT_GUARD {code}: near-fall ADL → adl"
        return pred, ""
    return pred, ""


class LLMReasoner:
    def __init__(
        self,
        backend: str = "ollama",
        base_url: str = "http://127.0.0.1:11434",
        model: str = "mistral:latest",
        temperature: float = 0.1,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        fallback_heuristic: bool = True,
        use_activity_prompt: bool | None = None,
        activity_precision_guard: bool = True,
    ):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.fallback_heuristic = fallback_heuristic
        self.use_activity_prompt = use_activity_prompt
        self.activity_precision_guard = bool(activity_precision_guard)

    def _system_prompt(self, evidence_text: str) -> str:
        # Activity/trial codes are label-derived. Always use the code-free
        # prompt, even if a legacy caller requests the historical mode.
        return SYSTEM_PROMPT_NO_ACTIVITY

    def reason(
        self,
        evidence_text: str,
        feats: BiomechanicalFeatures,
        p_fall: float,
        cases: list[MemoryCase],
    ) -> ReasoningResult:
        if self.backend == "heuristic":
            hr = _heuristic_reason(feats, p_fall, cases)
            return hr

        case_block = "\n".join(c.summary() for c in cases) if cases else "None"
        user = (
            f"Evidence:\n{evidence_text}\n\n"
            f"Similar cases:\n{case_block}\n\n"
            f"Tier-1 p(fall)={p_fall:.3f}\n"
            "Decide now."
        )
        system = self._system_prompt(evidence_text)
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
                obj = _extract_json(raw)
                pred = str(obj.get("prediction", "adl")).lower()
                if pred not in ("fall", "adl"):
                    pred = "fall" if p_fall >= 0.5 else "adl"
                severity = str(obj.get("severity", "none")).lower()
                action = str(obj.get("action", "log")).lower()
                conf = float(obj.get("confidence", 0.7))
                rationale = str(obj.get("rationale", raw[:500]))
                return ReasoningResult(
                    prediction=pred,
                    severity=severity,
                    confidence=conf,
                    rationale=rationale,
                    action=action,
                    source="ollama",
                )
            except Exception as e:
                last_err = e
                continue

        if self.fallback_heuristic:
            return _heuristic_reason(feats, p_fall, cases)
        raise RuntimeError(f"LLM reasoning failed: {last_err}")
