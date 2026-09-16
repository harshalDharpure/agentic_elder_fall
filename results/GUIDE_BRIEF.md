# Guide Brief — Agentic Fall Detection (JBHI Track A)

**Dataset:** SisFall · **Splits:** subject-independent 5-fold · **Primary detector:** `cnn_lstm_attn` · **Framing:** RCDP (Risk-Calibrated Dual-Process)

## One-line claim

> On the **ambiguous stress benchmark** (D18/D19 near-falls), Tier-1 F1 collapses to **0.713** while the full stack reaches **0.896** (ΔF1 **+0.183**). On the main protocol, `gate_knn_llm` improves F1/recall and reduces normalized response cost vs the same Tier-1 detector on every fold.

## Ground Truth Data Lock (LOCKED — do not change without re-run)

| Metric | Value |
|--------|-------|
| **Ambiguous bench F1** | Tier-1 **0.713** → Stack **0.896** (Δ **+0.183**) |
| **Ambiguous bench Recall** | **0.768 → 0.976** (Δ **+0.208**) |
| **Ambiguous cost/win** | **0.90 → 0.15** (−84%) |
| **Main SisFall F1** | **0.759±0.014 → 0.887±0.029** |
| **BCa CI on ΔF1** | **[0.119, 0.144]** (5/5 fold sign consistency) |
| **BCa CI on ΔCost/1000** | **[-1064, -829]** |
| **KFall external F1** | **0.987 → 0.990** (Δ **+0.003**) — stability, not breakthrough |
| **Escalation rate** | Main ~**45%**; Ambiguous bench ~**62%** |
| **G_faith (contrastive)** | **0.44 → 0.84** |

## Table 1 — Ambiguous stress benchmark (LEAD RESULT, n=300/fold)

| Fold | Tier1 F1 | Stack F1 | ΔF1 | Tier1 Cost/win | Stack Cost/win | Esc. rate |
|------|----------|----------|-----|----------------|----------------|-----------|
| 0 | 0.696 | 0.868 | +0.172 | 0.86 | 0.13 | 0.59 |
| 1 | 0.738 | 0.894 | +0.156 | 0.82 | 0.28 | 0.52 |
| 2 | 0.682 | 0.892 | +0.210 | 0.98 | 0.11 | 0.63 |
| 3 | 0.714 | 0.902 | +0.189 | 0.92 | 0.16 | 0.61 |
| 4 | 0.735 | 0.922 | +0.186 | 0.93 | 0.06 | 0.62 |
| **Mean** | **0.713** | **0.896** | **+0.183** | **0.90** | **0.15** | **0.59** |

Source: `results/kfold_ambiguous/COMPARISON_NO_CRITIC.txt`

## Table 2 — Per-fold: Tier-1 vs full agentic stack (main n=1000)

| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 | Cost ↓ % |
|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|----------|
| 0 | 1000 | 0.774 | 0.781 | 1042 | 0.892 | 0.986 | 156 | +0.119 | 85% |
| 1 | 1000 | 0.737 | 0.748 | 1223 | 0.851 | 0.911 | 490 | +0.114 | 60% |
| 2 | 1000 | 0.755 | 0.783 | 1066 | 0.874 | 0.998 | 134 | +0.119 | 87% |
| 3 | 1000 | 0.766 | 0.732 | 1248 | 0.930 | 0.995 | 83 | +0.164 | 93% |
| 4 | 1000 | 0.760 | 0.749 | 1175 | 0.887 | 0.993 | 136 | +0.126 | 88% |

## Table 3 — 5-fold mean ± std (main protocol)

| Mode | F1 | Recall | Cost / 1000 windows |
|------|----|--------|---------------------|
| tier1_only | 0.759±0.014 | 0.759±0.022 | 1151±93 |
| **gate_knn_llm** | **0.887±0.029** | **0.977±0.037** | **200±164** |

## Table 4 — KFall external (confirmatory)

| Mode | F1 | Recall | Cost/1000 |
|------|-----|--------|-----------|
| tier1_only | 0.987±0.005 | 0.991±0.007 | 50±35 |
| gate_knn_llm | 0.990±0.004 | 0.996±0.002 | 26±12 |

**Do not claim large F1 gains on KFall.** Frame as cross-domain stability.

## Table 5 — LLM backends on shared ambiguous cases (5 folds)

| Fold | Heuristic F1 | Mistral F1 | Qwen F1 |
|------|--------------|------------|---------|
| 0 | 0.042 | 0.959 | 0.807 |
| 1 | 0.105 | 0.977 | 0.874 |
| 2 | 0.000 | 0.975 | 0.819 |
| 3 | 0.021 | 0.984 | 0.802 |
| 4 | 0.000 | 0.966 | 0.813 |

**Mean:** heuristic 0.034 · **Mistral 0.972** · Qwen 0.823

## What to say / not say

| Say | Do not say |
|-----|------------|
| Ambiguous bench: Tier-1 F1 0.713 → stack 0.896 | Generic "LLM improves F1" without ambiguous context |
| KFall confirms cross-domain stability (ΔF1 +0.003) | KFall zero-shot F1 0.62 → 0.81 |
| Escalation ~45% main, ~62% ambiguous (honest) | System 2 invoked <5% of windows |
| RCDP dual-process: 7ms edge + 10s deliberative triage | Real-time millisecond LLM fall detection |
| Agentic wrapper on fixed Tier-1 detector | We beat all SOTA fall detectors |

## Artifacts

- Ambiguous bench: `results/kfold_ambiguous/comparison_no_critic.json`
- Per-fold: `results/fold{0-4}/ablation_fold*.json`
- KFall: `results/kfall/kfall_external_summary.json`
- Paper table: `paper/table/tab_ambiguous_sisfall.tex`

Updated: Track A Week 1 (2026-09-01)


## Week 3 — Clinical framing (added)

| Item | Value |
|------|-------|
| G_faith (base → contrastive) | **0.436 → 0.843** |
| Graded action cost reduction | **~45%** (589 → 324) |
| Escalation: main / ambiguous / KFall | **47% / 59% / 1.1%** |
| β-cost table | β ∈ {5, 10, 20} with τ* = 1/(β+1) |

Artifacts: `paper/fig/deployment_architecture.pdf`, `tab_beta_cost.tex`, `tab_grounding.tex`, `tab_escalation.tex`
