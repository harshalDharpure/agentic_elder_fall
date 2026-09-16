from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..features.biomechanics import BiomechanicalFeatures
from .constraints import ConstraintPack, build_constraint_pack, rationale_cites_sigma, veto_score
from .critic_agent import ActorHypothesis, CriticAgent, CriticVerdict
from .knn_memory import MemoryCase
from .llm_reasoner import LLMReasoner, ReasoningResult, _heuristic_reason


@dataclass
class AdjudicationTrace:
    actor: ActorHypothesis | None = None
    critic: CriticVerdict | None = None
    actor_swapped: ActorHypothesis | None = None
    critic_swapped: CriticVerdict | None = None
    num_llm_calls: int = 0
    mode: str = "single_shot"  # single_shot | actor_only | actor_critic | actor_critic_roleswap | crc_veto
    final_source: str = "tier1"
    veto_score: float | None = None
    vetoed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "num_llm_calls": self.num_llm_calls,
            "final_source": self.final_source,
            "veto_score": self.veto_score,
            "vetoed": self.vetoed,
            "actor": self.actor.to_dict() if self.actor else None,
            "critic": self.critic.to_dict() if self.critic else None,
            "actor_swapped": self.actor_swapped.to_dict() if self.actor_swapped else None,
            "critic_swapped": self.critic_swapped.to_dict() if self.critic_swapped else None,
        }


def _judge_merge(
    actor: ActorHypothesis,
    critic: CriticVerdict,
    constraints: ConstraintPack,
    tie_break: str = "recall_preserving",
    freeze_label: bool = True,
) -> ReasoningResult:
    """Reconcile Actor and Critic. Default: freeze fall/ADL, allow action change."""
    pred = actor.prediction
    if freeze_label:
        severity = critic.revised_severity if critic.revised_severity else actor.severity
        action = critic.revised_action or actor.suggested_action
        conf = max(actor.confidence, critic.confidence)
        rationale = actor.rationale or ""
        if not rationale_cites_sigma(rationale):
            rationale = (
                f"{rationale} | GROUNDED Σ(w): {constraints.checklist_text().replace(chr(10), '; ')}"
            ).strip(" |")
        if critic.verdict != "confirm":
            rationale = f"{rationale} | CRITIC({critic.verdict}): {critic.critique}"
        src = "action_critique"
        if actor.source == "heuristic" and critic.source == "heuristic":
            src = "heuristic_action_critique"
        elif "ollama" in (actor.source, critic.source):
            src = "ollama_action_critique"
        return ReasoningResult(
            prediction=pred,
            severity=severity,
            confidence=float(min(0.99, conf)),
            rationale=rationale,
            action=action,
            source=src,
        )

    # Legacy label-merge path (kept for ablations; freeze_label should stay True in paper runs)
    if critic.verdict == "confirm":
        pred = actor.prediction
        severity = actor.severity
        action = actor.suggested_action
        conf = max(actor.confidence, critic.confidence)
        rationale = (
            f"JUDGE:confirm. Actor={actor.prediction}/{actor.hypothesis}. "
            f"Critic confirmed. {critic.critique} | Actor: {actor.rationale}"
        )
    elif (
        critic.verdict in ("reject", "revise")
        and critic.revised_prediction == "fall"
        and actor.prediction == "adl"
    ):
        pred = critic.revised_prediction
        severity = critic.revised_severity
        action = critic.revised_action
        conf = critic.confidence
        rationale = (
            f"JUDGE:upgrade. Actor={actor.prediction} -> fall. "
            f"Violations={critic.violations}. {critic.critique}"
        )
    elif (
        critic.verdict in ("reject", "revise")
        and critic.revised_prediction == "adl"
        and actor.prediction == "fall"
    ):
        if tie_break == "precision":
            pred = "adl"
            severity = "none"
            action = critic.revised_action if critic.revised_action in ("monitor", "log") else "monitor"
            conf = critic.confidence
            rationale = f"JUDGE:downgrade (precision). {critic.critique}"
        else:
            pred = "fall"
            severity = actor.severity if actor.severity != "none" else "moderate"
            action = actor.suggested_action if actor.suggested_action != "log" else "notify_caregiver"
            conf = max(actor.confidence, 0.7)
            rationale = f"JUDGE:keep_fall. Critic: {critic.critique}"
    else:
        pred = critic.revised_prediction
        severity = critic.revised_severity
        action = critic.revised_action
        conf = critic.confidence
        rationale = f"JUDGE:{critic.verdict}. Actor={actor.prediction} -> {pred}. {critic.critique}"

    src = "adjudicated"
    if actor.source == "heuristic" and critic.source == "heuristic":
        src = "heuristic_adjudicated"
    elif "ollama" in (actor.source, critic.source):
        src = "ollama_adjudicated"

    return ReasoningResult(
        prediction=pred,
        severity=severity,
        confidence=float(min(0.99, conf)),
        rationale=rationale,
        action=action,
        source=src,
    )


def _vote_predictions(preds: list[str], tie_break: str) -> str:
    falls = sum(1 for p in preds if p == "fall")
    adls = len(preds) - falls
    if falls > adls:
        return "fall"
    if adls > falls:
        return "adl"
    return "fall" if tie_break == "recall_preserving" else "adl"


class Adjudicator:
    """Actor–Critic–Judge orchestration for ambiguous windows."""

    def __init__(
        self,
        reasoner: LLMReasoner,
        critic: CriticAgent | None = None,
        enabled: bool = False,
        mode: str = "actor_critic",
        role_swap: bool = False,
        multi_prompt_vote: bool = False,
        tie_break: str = "recall_preserving",
        freeze_label: bool = True,
        actor_backend: str | None = None,
        critic_backend: str | None = None,
        veto_threshold: float | None = None,
        screen_confident_fall: bool = True,
        crc_alpha: float | None = None,
        crc_feasible: bool = False,
        use_contrastive: bool = False,
        k_pos: int = 3,
        k_neg: int = 3,
    ):
        self.reasoner = reasoner
        self.critic = critic or CriticAgent(backend=reasoner.backend)
        self.enabled = enabled
        self.mode = mode
        self.role_swap = role_swap or mode == "actor_critic_roleswap"
        self.multi_prompt_vote = multi_prompt_vote
        self.tie_break = tie_break
        self.freeze_label = True if mode in ("action_critique", "contrastive_action_critique") else bool(freeze_label)
        self.actor_backend = actor_backend or reasoner.backend
        self.critic_backend = critic_backend or reasoner.backend
        self.veto_threshold = veto_threshold
        self.screen_confident_fall = bool(screen_confident_fall)
        self.crc_alpha = crc_alpha
        self.crc_feasible = bool(crc_feasible)
        self.use_contrastive = bool(use_contrastive) or mode == "contrastive_action_critique"
        self.k_pos = int(k_pos)
        self.k_neg = int(k_neg)
        common = dict(
            base_url=reasoner.base_url,
            model=reasoner.model,
            temperature=reasoner.temperature,
            timeout_s=reasoner.timeout_s,
            max_retries=reasoner.max_retries,
            fallback_heuristic=reasoner.fallback_heuristic,
        )
        self._actor_agent = CriticAgent(backend=self.actor_backend, **common)
        self._critic_agent = CriticAgent(backend=self.critic_backend, **common)

    def reason(
        self,
        evidence_text: str,
        feats: BiomechanicalFeatures,
        p_fall: float,
        cases: list[MemoryCase],
    ) -> tuple[ReasoningResult, AdjudicationTrace]:
        if not self.enabled or self.mode in ("single_shot", "off", ""):
            res = self.reasoner.reason(evidence_text, feats, p_fall, cases)
            return res, AdjudicationTrace(
                mode="single_shot",
                num_llm_calls=1 if res.source == "ollama" else 0,
                final_source=res.source,
            )

        constraints = build_constraint_pack(feats, p_fall, cases)
        if self.use_contrastive and cases:
            pos = [c for c in cases if int(c.label) == 1]
            neg = [c for c in cases if int(c.label) == 0]
            cases_text = (
                "POSITIVE past falls (similar):\n"
                + ("\n".join(c.summary() for c in pos) if pos else "None")
                + "\nHARD-NEGATIVE ADLs that looked similar but were NOT falls:\n"
                + ("\n".join(c.summary() for c in neg) if neg else "None")
            )
        else:
            cases_text = "\n".join(c.summary() for c in cases) if cases else "None"
        trace = AdjudicationTrace(mode=self.mode)
        llm_calls = 0

        # Q1 path: same LLM Actor as gate_knn_llm, then heuristic action/rationale Critic.
        if self.mode in ("action_critique", "contrastive_action_critique", "crc_veto"):
            base = self.reasoner.reason(evidence_text, feats, p_fall, cases)
            if base.source == "ollama":
                llm_calls += 1
            actor = ActorHypothesis(
                hypothesis="near_fall" if constraints.near_fall_likely and base.prediction == "adl" else base.prediction,
                prediction=base.prediction,
                severity=base.severity,
                confidence=base.confidence,
                preconditions=[],
                rationale=base.rationale,
                suggested_action=base.action,
                source=base.source,
            )
            trace.actor = actor
            critic_v = self._critic_agent.critique(actor, evidence_text, constraints)
            if critic_v.source == "ollama":
                llm_calls += 1
            s = veto_score(constraints)
            trace.veto_score = s
            freeze = True
            if self.mode == "crc_veto":
                thr = self.veto_threshold
                can_veto = (
                    self.crc_feasible
                    and thr is not None
                    and actor.prediction == "fall"
                    and s >= float(thr)
                )
                if can_veto:
                    freeze = False
                    critic_v.verdict = "reject"
                    critic_v.revised_prediction = "adl"
                    critic_v.revised_severity = "none"
                    if critic_v.revised_action in ("emergency", "notify_caregiver"):
                        critic_v.revised_action = "monitor"
                    critic_v.violations = list(critic_v.violations) + [
                        f"CRC veto s={s:.3f} ≥ λ*={float(thr):.3f}"
                    ]
                    critic_v.critique = (
                        f"CRC veto: Σ(w) argues against fall (s={s:.3f} ≥ {float(thr):.3f}). "
                        + critic_v.critique
                    )
                    trace.vetoed = True
                else:
                    critic_v.revised_prediction = actor.prediction
            else:
                critic_v.revised_prediction = actor.prediction
            trace.critic = critic_v
            final = _judge_merge(
                actor, critic_v, constraints, self.tie_break, freeze_label=freeze
            )
            if trace.vetoed:
                final = ReasoningResult(
                    prediction="adl",
                    severity="none",
                    confidence=final.confidence,
                    rationale=final.rationale + f" | CRC_VETO s={s:.3f}",
                    action=critic_v.revised_action if critic_v.revised_action in ("monitor", "log") else "monitor",
                    source="crc_veto",
                )
            trace.num_llm_calls = llm_calls
            trace.final_source = final.source
            return final, trace

        actor = self._actor_agent.propose_actor(
            evidence_text, feats, p_fall, constraints, cases_text, cases=cases
        )
        if actor.source == "ollama":
            llm_calls += 1
        trace.actor = actor

        if self.mode == "actor_only":
            res = ReasoningResult(
                prediction=actor.prediction,
                severity=actor.severity,
                confidence=actor.confidence,
                rationale=f"ACTOR-only: {actor.rationale}",
                action=actor.suggested_action,
                source=f"actor_{actor.source}",
            )
            trace.num_llm_calls = llm_calls
            trace.final_source = res.source
            return res, trace

        critic_v = self._critic_agent.critique(actor, evidence_text, constraints)
        if critic_v.source == "ollama":
            llm_calls += 1
        if self.freeze_label:
            critic_v.revised_prediction = actor.prediction
        trace.critic = critic_v
        final = _judge_merge(
            actor, critic_v, constraints, self.tie_break, freeze_label=self.freeze_label
        )

        if self.role_swap or self.mode == "actor_critic_roleswap":
            caution = ActorHypothesis(
                hypothesis="near_fall" if constraints.near_fall_likely else actor.hypothesis,
                prediction="adl" if constraints.fall_support_score < 0.2 else actor.prediction,
                severity="none" if constraints.fall_support_score < 0.2 else actor.severity,
                confidence=0.55,
                preconditions=actor.preconditions,
                rationale="Role-swapped cautious prior based on Σ(w).",
                suggested_action="monitor" if constraints.fall_support_score < 0.35 else actor.suggested_action,
                source="role_swap_prior",
            )
            swapped_critic = self._critic_agent.critique(caution, evidence_text, constraints)
            if swapped_critic.source == "ollama":
                llm_calls += 1
            rebuttal = self._actor_agent.propose_actor(
                evidence_text + f"\n\nCautious critique: {swapped_critic.critique}",
                feats,
                p_fall,
                constraints,
                cases_text,
                cases=cases,
            )
            if rebuttal.source == "ollama":
                llm_calls += 1
            swapped_final = _judge_merge(
                rebuttal, swapped_critic, constraints, self.tie_break, freeze_label=self.freeze_label
            )
            trace.actor_swapped = rebuttal
            trace.critic_swapped = swapped_critic
            if self.freeze_label:
                vote_pred = actor.prediction
            else:
                vote_pred = _vote_predictions(
                    [final.prediction, swapped_final.prediction], self.tie_break
                )
            chosen = final if final.prediction == vote_pred else swapped_final
            final = ReasoningResult(
                prediction=vote_pred,
                severity=chosen.severity if chosen.prediction == vote_pred else (
                    "moderate" if vote_pred == "fall" else "none"
                ),
                confidence=0.5 * (final.confidence + swapped_final.confidence),
                rationale=(
                    f"ROLE-SWAP vote={vote_pred}. Forward: {final.rationale} || "
                    f"Swapped: {swapped_final.rationale}"
                ),
                action=chosen.action if chosen.prediction == vote_pred else (
                    "notify_caregiver" if vote_pred == "fall" else "monitor"
                ),
                source=f"roleswap_{chosen.source}",
            )

        if self.multi_prompt_vote and not self.freeze_label:
            clinical = _heuristic_reason(feats, p_fall, cases)
            vote_pred = _vote_predictions([final.prediction, clinical.prediction], self.tie_break)
            if vote_pred != final.prediction:
                final = ReasoningResult(
                    prediction=vote_pred,
                    severity=clinical.severity if vote_pred == clinical.prediction else final.severity,
                    confidence=0.5 * (final.confidence + clinical.confidence),
                    rationale=final.rationale + f" | Multi-prompt vote with clinical heuristic -> {vote_pred}",
                    action=clinical.action if vote_pred == clinical.prediction else final.action,
                    source=f"voted_{final.source}",
                )

        trace.num_llm_calls = llm_calls
        trace.final_source = final.source
        return final, trace
