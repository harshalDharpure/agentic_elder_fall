from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..features.biomechanics import extract_biomechanics
from .action_agent import ActionAgent, ActionDecision
from .adjudicator import Adjudicator
from .confidence_gate import ConfidenceGate, GateDecision
from .constraints import build_constraint_pack, rationale_cites_sigma, veto_score
from .critic_agent import ActorHypothesis
from .evidence import serialize_evidence
from .knn_memory import KNNMemory
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
    actor_prediction: str | None = None
    critic_verdict: str | None = None
    adjudicated: bool = False
    num_llm_calls: int = 0
    adjudication: dict[str, Any] = field(default_factory=dict)
    veto_score: float | None = None
    vetoed: bool = False
    knn_fall_frac: float | None = None
    fusion_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate"] = asdict(self.gate)
        d["action"] = asdict(self.action)
        return d


class AgenticPipeline:
    """End-to-end: Tier-1 → Confidence Gate → (Evidence → kNN → Actor/Critic) → Action."""

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
        adjudicator: Adjudicator | None = None,
        allow_action_downgrade: bool = True,
        include_activity: bool = False,
        require_text_embedder: bool = False,
        fuse_system2: bool = False,
        fusion_threshold: float | None = None,
        fusion_w_p: float = 0.45,
        fusion_w_knn: float = 0.25,
        fusion_w_llm: float = 0.30,
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
        self.adjudicator = adjudicator
        self.allow_action_downgrade = allow_action_downgrade
        self.include_activity = bool(include_activity)
        if self.include_activity:
            raise ValueError(
                "Activity/trial codes are label-derived and prohibited in the "
                "submission pipeline; use a separate audit-only implementation."
            )
        self.require_text_embedder = bool(require_text_embedder)
        self.fuse_system2 = bool(fuse_system2)
        self.fusion_threshold = fusion_threshold
        self.fusion_w_p = float(fusion_w_p)
        self.fusion_w_knn = float(fusion_w_knn)
        self.fusion_w_llm = float(fusion_w_llm)

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
        evidence = serialize_evidence(
            feats,
            activity=activity if self.include_activity else None,
            p_fall=p_fall,
            include_activity=self.include_activity,
        )
        escalated = force_escalate or gate.route == "ambiguous"
        retrieved_summaries: list[str] = []
        source = "tier1"
        actor_pred = None
        critic_verdict = None
        adjudicated = False
        num_llm_calls = 0
        adj_dict: dict[str, Any] = {}
        v_score: float | None = None
        vetoed = False
        knn_fall_frac: float | None = None
        fusion_score: float | None = None
        cases_for_frac: list = []

        crc_on = (
            self.adjudicator is not None
            and self.adjudicator.enabled
            and self.adjudicator.mode == "crc_veto"
        )
        action_crit_on = (
            self.adjudicator is not None
            and self.adjudicator.enabled
            and self.adjudicator.mode in ("action_critique", "contrastive_action_critique")
        )
        screen_conf = (
            self.adjudicator is not None
            and self.adjudicator.enabled
            and bool(self.adjudicator.screen_confident_fall)
            and (crc_on or action_crit_on)
        )

        def _query_embedding(evidence_text: str, model_emb: np.ndarray) -> np.ndarray:
            if self.text_embed_fn is not None:
                return self.text_embed_fn(evidence_text)
            if self.require_text_embedder:
                raise RuntimeError(
                    "Paper/no-leak mode requires a real text embedder; "
                    "refusing silent model-embedding fallback."
                )
            return model_emb

        def _fetch_cases(query_emb):
            if (
                self.adjudicator is not None
                and self.adjudicator.enabled
                and self.adjudicator.use_contrastive
            ):
                pos, neg = self.memory.contrastive_retrieve(
                    np.asarray(query_emb, dtype=np.float32),
                    k_pos=self.adjudicator.k_pos,
                    k_neg=self.adjudicator.k_neg,
                )
                return list(pos) + list(neg)
            return self.memory.retrieve(np.asarray(query_emb, dtype=np.float32))

        if not escalated:
            if gate.route == "fall":
                pred, sev, conf = "fall", "moderate", p_fall
                rationale = f"High-confidence Tier-1 fall (p={p_fall:.3f} >= {gate.tau_high})."
                suggested = "notify_caregiver"
            else:
                pred, sev, conf = "adl", "none", 1.0 - p_fall
                rationale = f"High-confidence Tier-1 ADL (p={p_fall:.3f} <= {gate.tau_low})."
                suggested = "log"
            actor_pred = pred
            if screen_conf and pred == "fall":
                q = _query_embedding(evidence, emb)
                cases = _fetch_cases(q)
                cases_for_frac = cases
                retrieved_summaries = [c.summary() for c in cases]
                pack = build_constraint_pack(feats, p_fall, cases)
                v_score = veto_score(pack)
                if crc_on:
                    thr = self.adjudicator.veto_threshold
                    if (
                        self.adjudicator.crc_feasible
                        and thr is not None
                        and v_score >= float(thr)
                    ):
                        pred, sev = "adl", "none"
                        rationale = (
                            f"{rationale} | CRC_VETO confident-path s={v_score:.3f} "
                            f"≥ λ*={float(thr):.3f}"
                        )
                        vetoed = True
                        critic_verdict = "reject"
                        source = "crc_veto"
                        suggested = "monitor"
                elif action_crit_on:
                    # Label frozen: Critic only grounds rationale / downgrades action.
                    actor = ActorHypothesis(
                        hypothesis="fall",
                        prediction="fall",
                        severity=sev,
                        confidence=conf,
                        rationale=rationale,
                        suggested_action=suggested,
                        source="tier1",
                    )
                    critic_v = self.adjudicator._critic_agent.critique(actor, evidence, pack)
                    critic_v.revised_prediction = "fall"
                    critic_verdict = critic_v.verdict
                    adjudicated = True
                    adj_dict = {
                        "mode": self.adjudicator.mode,
                        "confident_path_action_critique": True,
                        "veto_score": v_score,
                        "actor": actor.to_dict(),
                        "critic": critic_v.to_dict(),
                    }
                    if critic_v.verdict != "confirm":
                        sev = critic_v.revised_severity or sev
                        suggested = critic_v.revised_action or suggested
                        if not rationale_cites_sigma(rationale):
                            rationale = (
                                f"{rationale} | GROUNDED Σ(w): "
                                + pack.checklist_text().replace("\n", "; ")
                            )
                        rationale = f"{rationale} | CRITIC({critic_v.verdict}): {critic_v.critique}"
                        source = "action_critique_confident"
            reasoning = ReasoningResult(
                prediction=pred,
                severity=sev,
                confidence=conf,
                rationale=rationale,
                action=suggested,
                source=source if source not in ("tier1",) else "tier1",
            )
        else:
            q = _query_embedding(evidence, emb)
            cases = _fetch_cases(q)
            cases_for_frac = cases
            retrieved_summaries = [c.summary() for c in cases]

            if self.adjudicator is not None and self.adjudicator.enabled:
                reasoning, trace = self.adjudicator.reason(evidence, feats, p_fall, cases)
                actor_pred = trace.actor.prediction if trace.actor else None
                critic_verdict = trace.critic.verdict if trace.critic else None
                adjudicated = True
                num_llm_calls = trace.num_llm_calls
                adj_dict = trace.to_dict()
                v_score = trace.veto_score
                vetoed = bool(trace.vetoed)
            else:
                reasoning = self.reasoner.reason(evidence, feats, p_fall, cases)
                num_llm_calls = 1 if reasoning.source == "ollama" else 0
                actor_pred = reasoning.prediction
                pack = build_constraint_pack(feats, p_fall, cases)
                v_score = veto_score(pack)
            source = reasoning.source

            # Honest System-2 fusion (no activity codes): blend Tier-1, kNN vote, LLM.
            if cases_for_frac:
                knn_fall_frac = float(np.mean([int(c.label) for c in cases_for_frac]))
            else:
                knn_fall_frac = 0.5
            llm_bin = 1.0 if reasoning.prediction == "fall" else 0.0
            fusion_score = (
                self.fusion_w_p * float(p_fall)
                + self.fusion_w_knn * knn_fall_frac
                + self.fusion_w_llm * llm_bin
            )
            if self.fuse_system2:
                thr = 0.45 if self.fusion_threshold is None else float(self.fusion_threshold)
                fused_pred = "fall" if fusion_score >= thr else "adl"
                if fused_pred != reasoning.prediction:
                    rationale = (
                        f"{reasoning.rationale} | FUSE score={fusion_score:.3f} "
                        f"(p={p_fall:.3f}, knn={knn_fall_frac:.2f}, llm={llm_bin:.0f}) "
                        f" thr={thr:.3f} → {fused_pred}"
                    )
                    reasoning = ReasoningResult(
                        prediction=fused_pred,
                        severity=reasoning.severity if fused_pred == "fall" else "none",
                        confidence=float(max(reasoning.confidence, abs(fusion_score - thr) + 0.5)),
                        rationale=rationale,
                        action=reasoning.action if fused_pred == "fall" else "monitor",
                        source=f"{reasoning.source}+fuse",
                    )
                    source = reasoning.source

        allow_down = self.allow_action_downgrade and critic_verdict in ("reject", "revise")
        action = self.action_agent.decide(
            prediction=reasoning.prediction,
            severity=reasoning.severity,
            feats=feats,
            suggested_action=reasoning.action,
            allow_downgrade=allow_down,
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
            actor_prediction=actor_pred,
            critic_verdict=critic_verdict,
            adjudicated=adjudicated,
            num_llm_calls=num_llm_calls,
            adjudication=adj_dict,
            veto_score=v_score,
            vetoed=vetoed,
            knn_fall_frac=knn_fall_frac,
            fusion_score=fusion_score,
        )

def build_adjudicator_from_config(acfg: dict, reasoner: LLMReasoner) -> Adjudicator | None:
    """Factory used by eval scripts."""
    adj = acfg.get("adjudication") or {}
    enabled = bool(adj.get("enabled", False))
    mode = str(adj.get("mode", "actor_critic"))
    actor_backend = adj.get("actor_backend") or reasoner.backend
    critic_backend = adj.get("critic_backend") or reasoner.backend
    veto_thr = adj.get("veto_threshold")
    if veto_thr is not None:
        veto_thr = float(veto_thr)
    return Adjudicator(
        reasoner=reasoner,
        enabled=enabled,
        mode=mode,
        role_swap=bool(adj.get("role_swap", False)),
        multi_prompt_vote=bool(adj.get("multi_prompt_vote", False)),
        tie_break=str(adj.get("tie_break", "recall_preserving")),
        freeze_label=bool(adj.get("freeze_label", True)),
        actor_backend=str(actor_backend),
        critic_backend=str(critic_backend),
        veto_threshold=veto_thr,
        screen_confident_fall=bool(adj.get("screen_confident_fall", True)),
        crc_alpha=float(adj["crc_alpha"]) if adj.get("crc_alpha") is not None else None,
        crc_feasible=bool(adj.get("crc_feasible", False)),
        use_contrastive=bool(adj.get("use_contrastive", False)),
        k_pos=int(adj.get("k_pos", 3)),
        k_neg=int(adj.get("k_neg", 3)),
    )
