# Q1 Submission Pack — Agentic Fall Detection

Generated after overnight homogenize. See also `GUIDE_BRIEF.md`.

## Positioning vs Bhatti et al. (IEEE Access 2025)

| | Bhatti (base) | This work |
|---|---|---|
| Role | Tier-1 CNN–LSTM–Attention detector | Same detector + agentic safety layer |
| Fair baseline in our tables | `tier1_only` / `cnn_lstm_attn` (same ckpt, same folds) | `gate_knn_llm` |
| Claim | Detection accuracy | Recall↑ + normalized response cost↓ under escalation |

**Do not** claim beating Bhatti’s published table under their exact protocol; claim improvement vs **our** locked Tier-1 under subject-independent 5-fold SisFall.

## Table — Per-fold Tier-1 vs gate_knn_llm (homogeneous main protocol)

| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 |
|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|
| 0 | 1000 | 0.774 | 0.781 | 1042 | 0.892 | 0.986 | 156 | +0.119 |
| 1 | 1000 | 0.737 | 0.748 | 1223 | 0.851 | 0.911 | 490 | +0.114 |
| 2 | 1000 | 0.755 | 0.783 | 1066 | 0.874 | 0.998 | 134 | +0.119 |
| 3 | 1000 | 0.766 | 0.732 | 1248 | 0.930 | 0.995 | 83 | +0.164 |
| 4 | 1000 | 0.760 | 0.749 | 1175 | 0.887 | 0.993 | 136 | +0.126 |

## 5-fold mean ± std

| tier1_only | F1 0.759±0.014 | Rec 0.759±0.022 | cost/1k 1151±93 |
| **gate_knn_llm** | **F1 0.887±0.029** | **Rec 0.977±0.037** | **cost/1k 200±164** |

## Paired tests (gate_knn_llm − tier1_only)

- **cost_per_1000**: mean Δ=-951.0000, bootstrap 95% CI=[-1084.0, -824.8], Wilcoxon p=0.0625
- **f1**: mean Δ=0.1284, bootstrap 95% CI=[0.11679407001942303, 0.14742453815156034], Wilcoxon p=0.0625
- **recall**: mean Δ=0.2179, bootstrap 95% CI=[0.1873456878565707, 0.2477575871700719], Wilcoxon p=0.0625

## LLM compare (ambiguous cases)

- Heuristic mean F1=0.034 · **Mistral=0.972** · Qwen=0.823

## Q1 draft checklist

- [x] Subject-independent 5-fold SisFall
- [x] Same Tier-1 ckpt for fair baseline
- [x] Ablation ladder + LLM backend compare
- [x] Normalized cost_per_1000
- [x] Fold0/1 homogenized to max_test=1000 (full4000 archived)
- [x] End-to-end agentic uses ollama for full stack
- [ ] Paper writing (Methods/Results from this pack)
- [ ] KFall (optional, when raw data available)

## Reproducibility

```bash
python scripts/aggregate_folds.py --results-dir results
python scripts/export_tables.py
python scripts/build_guide_brief.py
```

