# Complete Analysis: With vs Without Critic / Critique Component

**Date:** 2026-08-22  
**Protocol:** SisFall 5-fold, homogeneous n=1000 test windows/fold (stratified, D18/D19 forced)  
**Primary backbone:** `cnn_lstm_attn` (fixed per fold)  
**Main stack (no Critic):** `gate_knn_llm` = Tier-1 → confidence gate → kNN memory → LLM Actor → Action  

---

## Executive summary

| Question | Answer |
|----------|--------|
| Does Critic improve **fall detection** (F1/recall/FN)? | **No meaningful gain.** Q1-safe critique modes freeze the fall/ADL label; F1/recall stay within noise of `gate_knn_llm`. |
| Does Critic improve **response quality**? | **Yes, modestly.** Graded action cost drops with contrastive critique; rationale grounding rises sharply (0.44 → 0.84). |
| Does Critic reduce **near-fall FAR**? | **No.** FAR is essentially unchanged (~0.156 → ~0.152–0.164). |
| Should Critic be the **main paper claim**? | **No.** Keep `gate_knn_llm` as primary. Critic is optional appendix for action/explanation only. |
| Did the K-fold ambiguous bench use Critic? | **No.** Those runs (`results/kfold_ambiguous*`) are `gate_knn_llm` only (Ollama or heuristic Actor). |

**Bottom line:** The big win comes from the **agentic stack itself** (gate + kNN + LLM Actor), not from adding a second Critic LLM. Critic adds explanation/action refinement value but not detection lift.

---

## Table 1 — Combined 5-fold results (detection + response)

| Mode | Critic? | F1 (mean±std) | Recall | FN | FAR (D18/D19) | Cost/1k | Graded cost | Grounding | Critic usage |
|------|---------|---------------|--------|-----|---------------|---------|-------------|-----------|--------------|
| **tier1_only** (old baseline) | No | 0.759±0.014 | 0.759 | 105 | 0.192 | 1151 | — | — | — |
| **gate_knn_llm** (current primary) | No | **0.887±0.026** | **0.977** | 10 | 0.156 | **200** | — | — | 0% |
| gate_knn_llm_action_critique | Yes (action-only) | 0.891±0.021 | 0.971 | 12 | 0.152 | 215 | 586 | 0.448 | 100% escalated |
| gate_knn_llm_contrastive_action_critique | Yes (contrastive) | 0.878±0.019 | 0.971 | 12 | 0.164 | 228 | **324** | **0.843** | 100% escalated |
| gate_knn_llm_crc_veto | Yes (+ CRC) | 0.891±0.021 | 0.971 | 12 | 0.152 | 215 | 579 | 0.448 | 100%; **veto=0** |
| gate_knn_llm_actor_critic ❌ | Yes (label flip) | 0.761±0.010 | 0.732 | **116** | 0.150 | 1245 | — | — | harmful |

### Δ vs `gate_knn_llm` (no Critic)

| Mode | ΔF1 | ΔRecall | ΔFN | ΔFAR | ΔCost | ΔGraded cost | ΔGrounding |
|------|-----|---------|-----|------|-------|--------------|------------|
| action_critique | +0.004 | −0.006 | +2.2 | −0.004 | +15 | +586 | +0.448 |
| contrastive_action_critique | −0.009 | −0.006 | +2.2 | +0.008 | +29 | +324 | **+0.843** |
| crc_veto | +0.004 | −0.006 | +2.2 | −0.004 | +15 | +579 | +0.448 |
| actor_critic (broken) | **−0.126** | **−0.245** | **+106** | −0.006 | +1045 | — | — |

---

## Table 2 — Per-fold paired comparison (F1 / Recall / FN)

| Fold | gate_knn_llm | action_critique | contrastive_critique | crc_veto |
|------|--------------|-----------------|----------------------|----------|
| 0 F1 / Rec / FN | 0.892 / 0.986 / 6 | 0.892 / 0.986 / 6 | 0.878 / 0.986 / 6 | 0.891 / 0.986 / 6 |
| 1 F1 / Rec / FN | 0.851 / 0.911 / 39 | 0.852 / 0.911 / 39 | 0.846 / 0.911 / 39 | 0.852 / 0.911 / 39 |
| 2 F1 / Rec / FN | 0.874 / 0.998 / 1 | 0.896 / 0.986 / 6 | 0.888 / 0.986 / 6 | 0.896 / 0.986 / 6 |
| 3 F1 / Rec / FN | 0.930 / 0.995 / 2 | 0.917 / 0.993 / 3 | 0.903 / 0.993 / 3 | 0.918 / 0.993 / 3 |
| 4 F1 / Rec / FN | 0.887 / 0.993 / 3 | 0.897 / 0.981 / 8 | 0.876 / 0.981 / 8 | 0.896 / 0.981 / 8 |

**Observation:** FN counts are identical or within ±1–5 across folds when labels are frozen. F1 wiggles come from FP changes and LLM stochasticity, not Critic “fixing” detection.

---

## Table 3 — K-fold ambiguous benchmark (300/fold, real D18/D19)

**Important:** These runs do **not** include Critic. They evaluate `gate_knn_llm` on a curated subset (100 ambiguous + 100 fall + 100 clear ADL per fold).

| Backend | Acc | F1 | Recall | FAR | Ambiguous Acc | Cost |
|---------|-----|-----|--------|-----|---------------|------|
| Ollama (Mistral) | 0.924 | 0.896 | 0.976 | 0.102 | 0.966 | 44 |
| Heuristic Actor | 0.852 | 0.766 | 0.726 | 0.085 | 0.990 | 291 |

**vs main 5-fold stack:** Ambiguous-case accuracy is high even without Critic when Mistral is the Actor. Heuristic Actor is weak on ambiguous cases (F1 0.766 vs 0.896 Ollama).

---

## Table 4 — Where the real improvement comes from (old → new, no Critic)

| Comparison | ΔF1 | ΔRecall | Cost reduction |
|------------|-----|---------|----------------|
| tier1_only → gate_knn_llm | **+0.128** | **+0.218** | **~83%** |
| gate_knn → gate_knn_llm | large on ambiguous | large | large |
| gate_llm alone | hurts FAR | — | — |
| gate_knn_llm → + Critic | +0.004 | ~0 | negligible |

The **agentic wrapper** (gate + kNN + LLM) is the contribution. Critic is incremental on top.

---

## Critic behavior (when enabled)

| Mode | Adjudicated | Confirm | Revise | Reject | Veto rate |
|------|-------------|---------|--------|--------|-----------|
| action_critique | ~448/fold | ~437 | ~10 | 0 | 0 |
| contrastive_action_critique | ~843/fold | ~320 | **~522** | 0 | 0 |
| crc_veto | ~448/fold | ~417 | ~30 | 0 | **0 (infeasible)** |

- **Contrastive critique** is active (522 revises/fold) and drives grounding to 0.84.
- **CRC veto** never fires on SisFall: confident-path FPs cannot be certified at α≤0.20 without inducing misses.
- **Actor–critic with label changes** causes FN explosion (6 → 105 on fold 0) — excluded from claims.

---

## Do we really improve using Critique?

### What improves ✅
1. **Rationale grounding** (+0.40 to +0.84) — Critic ties explanations to biomechanical evidence.
2. **Graded action cost** (contrastive: 324 vs baseline ungraded 0; vs action_critique 586) — more `monitor` vs `notify` on weak evidence.
3. **Emergency rate on ADL** — tiny reduction (0.0018 vs 0 on some folds).
4. **Visible Critic usage** — 100% on escalated ambiguous windows when forced.

### What does NOT improve ❌
1. **Fall detection F1/recall/FN** — frozen labels by design; no stable lift.
2. **Near-fall FAR** — unchanged (~15–16%).
3. **End-to-end cost** — slightly **higher** (+15–29 per 1k) due to extra Critic pass.
4. **CRC certification** — infeasible; zero vetoes.
5. **K-fold ambiguous bench accuracy** — not tested with Critic yet.

### Verdict for Q1 paper
> **Primary claim:** cost-sensitive agentic fall **response** under uncertainty (`gate_knn_llm`).  
> **Secondary (optional):** contrastive action critique improves **grounding and action grading**, not detection.  
> **Do not claim:** “Actor–Critic improves fall detection F1.”

---

## What to add next (ranked)

| Priority | Addition | Why |
|----------|----------|-----|
| 1 | **Contrastive hard-negative RAG inside Actor** (not 2nd LLM) | Pull D18/D19 negatives into Actor prompt; targets FAR without label-flip risk |
| 2 | **Recall-locked FAR / cost ablation** | Show Pareto: fix recall ≥0.97, minimize cost & near-fall FP |
| 3 | **Run Critic on K-fold ambiguous bench** | Fair stress test: 100 forced ambiguous cases WITH vs WITHOUT critique |
| 4 | **KFall external validation** | Already started under `results/kfall/`; strengthens generalization |
| 5 | **Latency / deployment table** | Tier-1 ~7ms vs escalated ~3–7s; honest for clinical framing |
| 6 | Skip | LoRA debate, multi-round buffer, post-fall recovery on 90-frame windows |

---

## Recommended paper tables

**Main Table (keep):** tier1_only vs gate_knn_llm (5-fold mean) — Table B in PROFESSOR_MEETING_BRIEF.md

**Appendix Table (optional):** gate_knn_llm vs contrastive_action_critique on action metrics only:
- graded_action_cost, rationale_grounding_rate, emergency_rate_adl
- explicitly note: F1/recall unchanged (label freeze)

**Do not put in main table:** actor_critic (label flip), crc_veto (zero impact), synthetic critique_bench (removed)

---

## File references

| Artifact | Path |
|----------|------|
| No-Critic 5-fold report | `results/PROFESSOR_MEETING_BRIEF.md` |
| Ollama ambiguous bench | `results/kfold_ambiguous/ALL_FOLDS_REPORT.txt` |
| Heuristic ambiguous bench | `results/kfold_ambiguous_heuristic/ALL_FOLDS_REPORT.txt` |
| Critic impact snapshot | `results/critic_action_impact.json` |
| Actor–critic postmortem | `results/ACTOR_CRITIC_STATUS.md` |
| Per-fold critique ablations | `results/fold{N}/ablation_fold{N}_*.json` |
