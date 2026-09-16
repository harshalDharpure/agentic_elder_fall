# Professor Meeting Brief — Agentic Fall Detection (Q1)

**Use this file tomorrow.** It has every main table + exactly what to say.

---

## 1. Opening pitch (30–45 seconds)

> Sir, Bhatti et al. (IEEE Access 2025) built a strong **CNN–LSTM–Attention fall detector**. We keep that detector fixed and add an **agentic safety layer**: confidence gate → k-NN memory → local LLM reasoning → cost-sensitive action. Under subject-independent **5-fold SisFall**, our full stack improves F1 and recall and cuts normalized response cost by ~**83%** versus the **same** Tier-1 checkpoint on every fold.

**One sentence difference:** Their paper is about *detection*; ours is about *what to do when the detector is uncertain*.

---

## 2. How we differ from the base paper (say this clearly)

| Aspect | Bhatti et al. 2025 (base) | Our work |
|---|---|---|
| Contribution | Multi-phase fall / ADL **detection** | Detection **+** uncertainty **+** explanation **+** response |
| Backbone | CNN–LSTM–Attention | Same primary backbone (+ zoo for context) |
| Uncertainty | Softmax only | Dual-threshold confidence gate |
| Explainability | Attention weights | Biomechanical evidence + LLM CoT JSON |
| Memory | None | Retrieval-augmented k-NN |
| Response | None | Cost-sensitive Action Agent |
| Fair baseline in *our* tables | — | `tier1_only` = same ckpt, same folds |
| Original paper dataset focus | KFall multi-class | SisFall binary + near-fall D18/D19 |

**Important honesty line:**
> We do **not** claim we beat Bhatti’s published KFall table under their protocol. We claim we beat the **same Bhatti-style detector** under **our locked SisFall 5-fold protocol**.

---

## 3. Architecture (30 seconds + point to figure)

```
IMU window (6×90)
   → Tier-1 Detector (cnn_lstm_attn) → p(fall)
   → Confidence Gate (τ_low / τ_high)
        ├─ confident ADL / Fall  → decide directly
        └─ ambiguous → Evidence → k-NN memory → LLM CoT → Action Agent
```

Cost we optimize: **Cost = 10×FN + 1×FP** (missed falls are 10× more expensive).

Paper figure: `paper/fig/architecture.tex` / PDF `paper/main.pdf`.

---

## Table A — Per-fold: Tier-1 vs full stack (`gate_knn_llm`)

Protocol: homogeneous **n = 1000** windows/fold (stratified, forced D18/D19). Cost = cost per 1000 windows.

| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 | Cost ↓ |
|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|--------|
| 0 | 1000 | 0.774 | 0.781 | 1042 | 0.892 | 0.986 | 156 | +0.119 | 85% |
| 1 | 1000 | 0.737 | 0.748 | 1223 | 0.851 | 0.911 | 490 | +0.114 | 60% |
| 2 | 1000 | 0.755 | 0.783 | 1066 | 0.874 | 0.998 | 134 | +0.119 | 87% |
| 3 | 1000 | 0.766 | 0.732 | 1248 | 0.930 | 0.995 | 83 | +0.164 | 93% |
| 4 | 1000 | 0.760 | 0.749 | 1175 | 0.887 | 0.993 | 136 | +0.126 | 88% |

**What to say:**
> Sir, on **every fold** the agentic stack wins: ΔF1 about +0.11 to +0.16, recall goes near 0.99 on most folds, and cost drops 60–93%.

---

## Table B — 5-fold mean ± std (main claim)

| Mode | F1 | Recall | Cost / 1000 windows |
|------|----|--------|---------------------|
| tier1_only (Bhatti-style baseline) | 0.759±0.014 | 0.759±0.022 | 1151±93 |
| **gate_knn_llm (our full stack)** | **0.887±0.029** | **0.977±0.037** | **200±164** |

- Mean ΔF1 = **+0.128**
- Mean ΔRecall = **+0.218**
- Mean cost reduction ≈ **83%**

### Paired statistics (stack − Tier-1)

| Metric | Mean Δ | Bootstrap 95% CI | Wilcoxon p |
|--------|--------|------------------|------------|
| cost_per_1000 | -951.0000 | [-1084.000, -824.800] | 0.0625 |
| f1 | +0.1284 | [0.117, 0.147] | 0.0625 |
| recall | +0.2179 | [0.187, 0.248] | 0.0625 |

**What to say about p-values:**
> With only 5 folds, Wilcoxon p bottoms at 0.0625 when all folds agree in sign. The bootstrap CIs for F1/recall/cost **exclude zero**, so the direction of improvement is stable.

---

## Table C — Ablation ladder (why each module matters)

Fold 0, archived full budget n=4000 (shows full ladder including `gate_llm`).

| Mode | F1 | Recall | Cost | Near-fall FAR | Escalated F1 |
|------|----|--------|------|---------------|--------------|
| tier1_only | 0.788 | 0.787 | 4596 | 0.219 | 0.000 |
| gate_only | 0.788 | 0.787 | 4596 | 0.219 | 0.175 |
| gate_knn | 0.789 | 0.764 | 4972 | 0.188 | 0.026 |
| gate_llm | 0.760 | 0.924 | 2491 | 0.625 | 0.445 |
| gate_knn_llm | 0.907 | 0.993 | 516 | 0.212 | 0.952 |

**What to say:**
> Gate alone does almost nothing. k-NN alone is weak. LLM alone can hurt near-fall FAR. **Only k-NN + LLM together** gives the strong result (F1 0.907, escalated F1 0.952). So the paper story is modular and ablation-backed.

---

## Table D — LLM backends on ambiguous cases

| Fold | Heuristic F1 | Mistral F1 | Qwen2.5 F1 |
|------|--------------|------------|------------|
| 0 | 0.042 | 0.959 | 0.807 |
| 1 | 0.105 | 0.977 | 0.874 |
| 2 | 0.000 | 0.975 | 0.819 |
| 3 | 0.021 | 0.984 | 0.802 |
| 4 | 0.000 | 0.966 | 0.813 |
| **Mean** | 0.034 | **0.972** | 0.823 |

**What to say:**
> Escalation alone is not enough — the **reasoner** matters. Heuristic fails on ambiguous cases (F1 ~0.03). **Mistral ~0.97** recovers them. That is strong evidence for LLM reasoning in the loop.

---

## Table E — Backbone zoo (context only)

Fold 0 detector-only (we fix `cnn_lstm_attn` as primary for agentic tables).

| Model | F1 | Recall | Spec. |
|-------|----|--------|-------|
| cnn1d | 0.721 | 0.985 | 0.277 |
| lstm | 0.700 | 0.986 | 0.197 |
| cnn_lstm | 0.694 | 0.995 | 0.155 |
| cnn_lstm_attn ← **primary** | 0.710 | 0.996 | 0.216 |
| tcn | 0.727 | 0.988 | 0.295 |
| modern_tcn | 0.737 | 0.990 | 0.326 |
| tsmixer | 0.695 | 0.984 | 0.181 |
| patchtst | 0.669 | 0.996 | 0.051 |
| transformer | 0.678 | 0.994 | 0.092 |

**What to say:**
> We also benchmark modern backbones, but the Q1 claim is **not** “new SOTA detector.” It is **agentic wrapper on a strong fixed detector**.

---

## 4. 5–7 minute talking script

1. **Problem:** Fall detectors miss costly FNs; near-falls (D18/D19) are ambiguous; no response policy.
2. **Base paper:** Bhatti = strong detector; still force-classifies ambiguous windows; no gate/LLM/cost policy.
3. **Our method:** escalate only uncertain windows → retrieve similar cases → local LLM → action under 10×FN cost.
4. **Show Table A + B:** every-fold win; mean F1 0.76→0.89; recall 0.76→0.98; cost ↓ ~83%.
5. **Show Table C:** ablation proves kNN+LLM are both needed.
6. **Show Table D:** Mistral ≫ heuristic on paired ambiguous IDs.
7. **Limitations (builds trust):** local LLM latency; KFall not run yet; FAR not the main claim; 5-fold Wilcoxon coarse but bootstrap CIs clean.
8. **Ask:** Approve this as the Q1 paper story (agentic safety layer) and proceed to writing/submission.

---

## 5. What NOT to say

| Avoid | Why |
|-------|-----|
| “We beat Bhatti’s published numbers” | Different protocol/dataset focus |
| “Detector F1 ≥ 0.90” | Our Tier-1 alone is ~0.76 (cost-sensitive OP) |
| “We solved near-fall FAR” | FAR gain is modest; lead with cost + recall |
| “Heuristic agentic is the main result” | Full stack needs Ollama/LLM |

---

## 6. Files to open in the meeting

1. **This file** — `results/PROFESSOR_MEETING_BRIEF.md`
2. Paper PDF — `paper/main.pdf`
3. Optional — `results/GUIDE_BRIEF.md`, `results/Q1_SUBMISSION_PACK.md`

---

## 7. Closing line

> Sir, this is Q1-ready evidence for an **agentic wrapper around a strong fall detector**: five-fold SisFall, fair same-backbone baseline, ablations, and LLM comparisons — with honest limitations noted.

## KFall vs Bhatti (say this)

> Sir, Bhatti’s paper is multi-class detection (~98% F1). Ours is an agentic safety layer on the same CNN–LSTM–Attention idea. On KFall binary 5-fold we compare against the *same* Tier-1 checkpoint—not their published 36-class table.

- KFall mean: F1 0.987→0.990; recall 0.991→0.996; cost/1k 50→26
- Their D18/D19 confusion → our Confidence Gate + LLM escalation.
- Their no response policy → our Action Agent with 10×FN cost.
