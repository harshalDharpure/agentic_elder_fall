# HARMamba Results (available now)

**Status:** Fold 0 Tier-1 + partial ablations (n=4000) complete. Full 5-fold `gate_knn_llm` restarted with faster n=1000 protocol.

## Train (fold 0)
- HARMamba best val F1: **0.8046**

## Fold 0 ablation comparison (HARMamba n≈4000 vs CNN-LSTM baseline)

| Mode | CNN-LSTM F1 | HARMamba F1 | Δ F1 | CNN Rec | HARMamba Rec |
|------|------------:|------------:|-----:|--------:|-------------:|
| `tier1_only` | 0.774 | **0.786** | +0.012 | 0.781 | 0.793 |
| `gate_only` | 0.774 | **0.786** | +0.012 | 0.781 | 0.793 |
| `gate_knn` | 0.771 | **0.782** | +0.011 | 0.755 | 0.763 |
| `gate_llm` | 0.717 | **0.760** | +0.044 | 0.911 | 0.910 |
| `gate_knn_llm` | 0.892 | *pending* | — | 0.986 | — |

## Verdict so far
- **Tier-1 detector:** HARMamba **beats** CNN-LSTM (0.786 vs 0.774 F1).
- **gate_llm:** HARMamba **beats** CNN-LSTM (0.760 vs 0.717 F1).
- **Primary stack `gate_knn_llm`:** baseline is **0.892** — HARMamba number still pending (restarted with n=1000).

Artifacts: `results/harmamba/PARTIAL_FOLD0_COMPARISON.json`
