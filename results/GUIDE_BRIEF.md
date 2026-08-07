# Guide Brief — Agentic Fall Detection (Q1-style evidence)

**Dataset:** SisFall · **Splits:** subject-independent 5-fold · **Primary detector:** `cnn_lstm_attn`

## One-line claim

> Confidence-gated retrieval + local LLM (`gate_knn_llm`) improves F1/recall and reduces **normalized response cost** vs the same Tier-1 detector on every fold; paired ambiguous-case compare shows **Mistral ≫ heuristic**.

## Table 1 — Per-fold: Tier-1 vs full agentic stack

| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 | Cost ↓ % |
|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|----------|
| 0 | 1000 | 0.774 | 0.781 | 1042 | 0.892 | 0.986 | 156 | +0.119 | 85% |
| 1 | 1000 | 0.737 | 0.748 | 1223 | 0.851 | 0.911 | 490 | +0.114 | 60% |
| 2 | 1000 | 0.755 | 0.783 | 1066 | 0.874 | 0.998 | 134 | +0.119 | 87% |
| 3 | 1000 | 0.766 | 0.732 | 1248 | 0.930 | 0.995 | 83 | +0.164 | 93% |
| 4 | 1000 | 0.760 | 0.749 | 1175 | 0.887 | 0.993 | 136 | +0.126 | 88% |

## Table 2 — 5-fold mean ± std (normalized)

| Mode | F1 | Recall | Cost / 1000 windows |
|------|----|--------|---------------------|
| tier1_only | 0.759±0.014 | 0.759±0.022 | 1151±93 |
| **gate_knn_llm** | **0.887±0.029** | **0.977±0.037** | **200±164** |

- Mean ΔF1 = **+0.128** · Mean ΔRecall = **+0.218** · Mean cost reduction = **83%** (on cost/1000).

## Table 3 — Ablation ladder (Fold 0, full protocol n=4000)

| Mode | F1 | Recall | Cost | Near-fall FAR | Escalated F1 |
|------|----|--------|------|---------------|--------------|
| tier1_only | 0.774 | 0.781 | 1042 | 0.194 | 0.000 |
| gate_only | 0.774 | 0.781 | 1042 | 0.194 | 0.200 |
| gate_knn | 0.771 | 0.755 | 1138 | 0.169 | 0.038 |
| gate_knn_llm | 0.892 | 0.986 | 156 | 0.175 | 0.953 |

Takeaway: **gate or kNN alone is not enough** — the win needs **kNN + LLM**.

## Table 4 — LLM backends on shared ambiguous cases (5 folds)

| Fold | Heuristic F1 | Mistral F1 | Qwen F1 |
|------|--------------|------------|---------|
| 0 | 0.042 | 0.959 | 0.807 |
| 1 | 0.105 | 0.977 | 0.874 |
| 2 | 0.000 | 0.975 | 0.819 |
| 3 | 0.021 | 0.984 | 0.802 |
| 4 | 0.000 | 0.966 | 0.813 |

**Mean:** heuristic 0.034 · **Mistral 0.972** · Qwen 0.823

## Methods footnote (say this if asked)

- Folds **0–1** ablations: `max_test=4000` (full ladder incl. `gate_llm`). Folds **2–4**: `max_test=1000`, skip `gate_llm` (compute budget). Main comparisons use **cost per 1000 windows** so budgets are comparable.
- End-to-end `agentic_fold*.json` with **heuristic** backend is **not** the full stack; primary paper row is ablation **`gate_knn_llm`** (Ollama). Folds 2–4 agentic already use ollama.
- Near-fall FAR is reported; it is **not** the primary differentiator in these runs.
- KFall external validation: code ready, raw data not run yet.

## What to say / not say

| Say | Do not say |
|-----|------------|
| Agentic wrapper on a fixed Tier-1 detector | We beat all SOTA fall detectors |
| Primary gains: cost↓ + recall/F1↑ via escalation | Detector F1 ≥ 0.90 |
| Mistral recovers ambiguous cases heuristics miss | FAR problem is solved |
| 5-fold subject-independent SisFall | Results from a single fold only |

## Artifacts

- Per-fold: `results/fold{0-4}/ablation_fold*.json`, `llm_compare_fold*.json`
- Aggregate: `results/aggregate_ablation_mean_std.*`, `aggregate_ablation_tests.json`
- Footnotes: `results/aggregate_paper_footnotes.json`

