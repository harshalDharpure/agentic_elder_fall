from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..features.biomechanics import extract_biomechanics
from .action_agent import ActionAgent, ActionDecision
from .confidence_gate import ConfidenceGate, GateDecision
from .evidence import serialize_evidence
from .knn_memory import KNNMemory, MemoryCase
from .llm_reasoner import LLMReasoner, ReasoningResult


@dataclass
class PipelineResult:
    p_fall: float
    gate: GateDecision
    prediction: str
    severity: str
    confidence: float
    rationale: str
    action: ActionDecision
    retrieved: list[str]
    latency_ms: float
    escalated: bool
    evidence_text: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate"] = asdict(self.gate)
        d["action"] = asdict(self.action)
        return d


class AgenticPipeline:
    """End-to-end: Tier-1 → Confidence Gate → (Evidence → kNN → LLM) → Action."""

    def __init__(
        self,
        model: torch.nn.Module,
        gate: ConfidenceGate,
        memory: KNNMemory,
        reasoner: LLMReasoner,
        action_agent: ActionAgent,
        device: str | torch.device = "cpu",
        sample_rate_hz: float = 200.0,
        freefall_g_threshold: float = 0.5,
        stillness_var_threshold: float = 0.05,
        text_embed_fn=None,
    ):
        self.model = model.to(device).eval()
        self.gate = gate
        self.memory = memory
        self.reasoner = reasoner
        self.action_agent = action_agent
        self.device = device
        self.sample_rate_hz = sample_rate_hz
        self.freefall_g_threshold = freefall_g_threshold
        self.stillness_var_threshold = stillness_var_threshold
        self.text_embed_fn = text_embed_fn

    @torch.no_grad()
    def _tier1(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        """x: (1, C, T) or (C, T). Returns p_fall and embedding."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        logits, emb = self.model(x)
        probs = F.softmax(logits, dim=-1)[0]
        if probs.numel() == 2:
            p_fall = float(probs[1].item())
        else:
            # multi-class: sum fall-class probabilities if model has fall_mask
            fall_mask = getattr(self.model, "fall_class_mask", None)
            if fall_mask is not None:
                p_fall = float(probs[fall_mask].sum().item())
            else:
                p_fall = float(probs.max().item())
        return p_fall, emb[0].detach().cpu().numpy()

    def run(
        self,
        x: torch.Tensor,
        window_tc: np.ndarray,
        activity: str | None = None,
        force_escalate: bool = False,
    ) -> PipelineResult:
        """
        x: model input (C, T)
        window_tc: same window as (T, C) for biomechanics
        """
        t0 = time.perf_counter()
        p_fall, emb = self._tier1(x)
        gate = self.gate.decide(p_fall)
        feats = extract_biomechanics(
            window_tc,
            sample_rate_hz=self.sample_rate_hz,
            freefall_g_threshold=self.freefall_g_threshold,
            stillness_var_threshold=self.stillness_var_threshold,
        )
        evidence = serialize_evidence(feats, activity=activity, p_fall=p_fall)
        escalated = force_escalate or gate.route == "ambiguous"
        retrieved_summaries: list[str] = []
        source = "tier1"

        if not escalated:
            if gate.route == "fall":
                pred, sev, conf = "fall", "moderate", p_fall
                rationale = f"High-confidence Tier-1 fall (p={p_fall:.3f} >= {gate.tau_high})."
                suggested = None
            else:
                pred, sev, conf = "adl", "none", 1.0 - p_fall
                rationale = f"High-confidence Tier-1 ADL (p={p_fall:.3f} <= {gate.tau_low})."
                suggested = None
            reasoning = ReasoningResult(
                prediction=pred,
                severity=sev,
                confidence=conf,
                rationale=rationale,
                action="notify_caregiver" if pred == "fall" else "log",
                source="tier1",
            )
        else:
            # retrieval embedding: prefer text embedder if provided
            if self.text_embed_fn is not None:
                q = self.text_embed_fn(evidence)
            else:
                q = emb
            cases = self.memory.retrieve(np.asarray(q, dtype=np.float32))
            retrieved_summaries = [c.summary() for c in cases]
            reasoning = self.reasoner.reason(evidence, feats, p_fall, cases)
            source = reasoning.source

        action = self.action_agent.decide(
            prediction=reasoning.prediction,
            severity=reasoning.severity,
            feats=feats,
            suggested_action=reasoning.action,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            p_fall=p_fall,
            gate=gate,
            prediction=reasoning.prediction,
            severity=action.severity,
            confidence=reasoning.confidence,
            rationale=reasoning.rationale,
            action=action,
            retrieved=retrieved_summaries,
            latency_ms=latency,
            escalated=escalated,
            evidence_text=evidence,
            source=source,
        )
