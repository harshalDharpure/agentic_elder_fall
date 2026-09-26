# Q1 Submission Pack — Agentic Fall Detection (JBHI Track A)

Generated for IEEE JBHI submission. See also `GUIDE_BRIEF.md` and Ground Truth Data Lock below.

## Ground Truth Data Lock (LOCKED)

| Metric | Value | Do NOT claim |
|--------|-------|--------------|
| Ambiguous bench F1 | 0.713 → 0.896 (Δ +0.183) | Obscure or minimize |
| Ambiguous recall | 0.768 → 0.976 (Δ +0.208) | — |
| Ambiguous cost/win | 0.90 → 0.15 (−84%) | — |
| Main SisFall F1 | 0.759±0.014 → 0.887±0.029 | Detector SOTA |
| KFall external F1 | 0.987 → 0.990 (Δ +0.003) | 0.62 → 0.81 zero-shot |
| Escalation rate | ~45% main; ~62% ambiguous | <5% duty cycle |
| G_faith | 0.44 → 0.84 (contrastive) | Unverified grounding |

## Positioning vs Bhatti et al. (IEEE Access 2025)

| | Bhatti (base) | This work |
|---|---|---|
| Role | Tier-1 CNN–LSTM–Attention detector | Same detector + RCDP agentic safety layer |
| Fair baseline | `tier1_only` (same ckpt, same folds) | `gate_knn_llm` |
| Lead result | — | Ambiguous stress bench (D18/D19): F1 0.713→0.896 |
| Claim | Detection accuracy | Recall↑ + clinical cost↓ under selective escalation |

## Table — Ambiguous stress benchmark (LEAD, n=300/fold)

| Fold | Tier1 F1 | Stack F1 | ΔF1 | Cost/win (T1→Stack) | Esc. |
|------|----------|----------|-----|----------------------|------|
| 0 | 0.696 | 0.868 | +0.172 | 0.86→0.13 | 0.59 |
| 1 | 0.738 | 0.894 | +0.156 | 0.82→0.28 | 0.52 |
| 2 | 0.682 | 0.892 | +0.210 | 0.98→0.11 | 0.63 |
| 3 | 0.714 | 0.902 | +0.189 | 0.92→0.16 | 0.61 |
| 4 | 0.735 | 0.922 | +0.186 | 0.93→0.06 | 0.62 |
| **Mean** | **0.713** | **0.896** | **+0.183** | **0.90→0.15** | **0.59** |

Source: `results/kfold_ambiguous/comparison_no_critic.json`

## Table — Per-fold Tier-1 vs gate_knn_llm (main n=1000)

| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 |
|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|
| 0 | 1000 | 0.774 | 0.781 | 1042 | 0.892 | 0.986 | 156 | +0.119 |
| 1 | 1000 | 0.737 | 0.748 | 1223 | 0.851 | 0.911 | 490 | +0.114 |
| 2 | 1000 | 0.755 | 0.783 | 1066 | 0.874 | 0.998 | 134 | +0.119 |
| 3 | 1000 | 0.766 | 0.732 | 1248 | 0.930 | 0.995 | 83 | +0.164 |
| 4 | 1000 | 0.760 | 0.749 | 1175 | 0.887 | 0.993 | 136 | +0.126 |

## 5-fold mean ± std (main)

| tier1_only | F1 0.759±0.014 | Rec 0.759±0.022 | cost/1k 1151±93 |
| **gate_knn_llm** | **F1 0.887±0.029** | **Rec 0.977±0.037** | **cost/1k 200±164** |

## Paired tests (gate_knn_llm − tier1_only) — BCa bootstrap

| Metric | Mean Δ | BCa 95% CI | Sign | Wilcoxon p |
|--------|--------|------------|------|------------|
| F1 | +0.132 | [0.119, 0.144] | **5/5** | 0.0625 |
| Recall | +0.213 | [0.182, 0.238] | **5/5** | 0.0625 |
| Cost/1000 | -936 | [-1064, -829] | **5/5** | 0.0625 |

Source: `paper/table/tab_stats.tex`, `results/stats_sign_consistency.json`

## Paired tests (legacy percentile, superseded by BCa above)

## KFall external validation (COMPLETE)

- Tier-1 F1 **0.987±0.005** → stack **0.990±0.004** (ΔF1 **+0.003**)
- Tier-1 recall 0.991 → stack 0.996
- Cost/1000 50±35 → 26±12
- **Frame as cross-domain stability**, not headline F1 breakthrough
- Source: `results/kfall/kfall_external_summary.json`

## LLM compare (ambiguous cases)

- Heuristic mean F1=0.034 · **Mistral=0.972** · Qwen=0.823

## JBHI Track A checklist (Week 1–4)

- [x] Ground Truth Data Lock documented
- [x] Ambiguous stress table in paper (`tab_ambiguous_sisfall.tex`)
- [x] KFall stale "access pending" text fixed in `main.tex`
- [x] RCDP framing in abstract/intro
- [x] GUIDE_BRIEF + Q1 pack synced
- [x] BCa bootstrap CIs in `tab_stats.tex` — Week 2 (F1 Δ [0.119, 0.144], cost Δ [-1064, -829])
- [x] 5/5 fold sign consistency documented (`results/stats_sign_consistency.json`)
- [ ] Ollama seed sweep (`run_seed_sweep.py --with-ollama --seeds 42,43,44`) — **RUNNING** in background
- [x] Deployment architecture figure (`paper/fig/deployment_architecture.pdf`)
- [x] β-cost sensitivity table (`tab_beta_cost.tex`)
- [x] Actor-critic negative ablation (`tab_negative_ablation.tex`, FN 12→116 mean)
- [x] KFall 36-class fold 0: macro-F1 **0.541**, weighted-F1 **0.583** (`tab_kfall_multiclass.tex`)
- [ ] Submit IEEE JBHI — ready after PDF compile + seed sweep

## Reproducibility

```bash
python scripts/aggregate_folds.py --results-dir results
python scripts/generate_paper_tables.py
python scripts/generate_kfall_paper_tables.py
python scripts/plot_paper_figures.py
```

Updated: Track A Week 1 (2026-09-01)
