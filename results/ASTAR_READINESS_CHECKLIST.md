# A* Experiment Readiness Checklist

**Status as of 2026-09-01.** Experiments first; paper build deferred per user directive.

## Track A (JBHI packaging) — DONE

| Item | Status | Artifact |
|------|--------|----------|
| Ambiguous bench table | ✅ | `paper/table/tab_ambiguous_sisfall.tex` |
| Paired stats + BCa bootstrap | ✅ | `paper/table/tab_stats.tex`, `results/stats_sign_consistency.json` |
| Deployment architecture fig | ✅ | `paper/fig/deployment_architecture.pdf` |
| Clinical cost / beta sensitivity | ✅ | `paper/table/tab_beta_cost.tex` |
| Escalation honesty table | ✅ | `paper/table/tab_escalation.tex` |
| Actor-critic negative ablation | ✅ | `paper/table/tab_negative_ablation.tex` |
| KFall 36-class external | ✅ | `paper/table/tab_kfall_multiclass.tex` |
| Limitations expanded | ✅ | `paper/main.tex` |

## Track B (A* science) — DONE (fold caps n=1000)

| Experiment | Status | Artifact |
|------------|--------|----------|
| Split conformal gate + α audit | ✅ | `results/astar/conformal_gate_audit.json`, `paper/table/tab_conformal_gate.tex` |
| Duty-cycle Pareto (risk vs esc.) | ✅ | `results/astar/duty_cycle_pareto.json`, `paper/table/tab_pareto.tex` |
| Orientation ±10/15° robustness | ✅ | `results/astar/robustness_fold*.json`, `paper/table/tab_robustness.tex` |
| Sampling jitter 50/100Hz | ✅ | (in robustness JSON) |
| Partition fallback τ*=1/(β+1) | ✅ | `results/astar/partition_fallback_fold*.json`, `paper/table/tab_partition_fallback.tex` |
| k-NN k∈{1,5,10} ablation | ✅ | `results/astar/knn_k_ablation_fold0.json`, `paper/table/tab_knn_k.tex` |

### Conformal audit highlights (α=0.05, tier-1 routing)

| Fold | q̂ | Esc. rate | Edge FN rate |
|-----:|---:|---:|---:|
| 0 | 0.830 | 0.58 | 0.017 |
| 1 | 0.926 | 0.83 | 0.030 |
| 2 | 0.904 | 0.70 | 0.017 |
| 3 | 0.856 | 0.61 | 0.010 |
| 4 | 0.630 | 0.17 | 0.087 |

**Interpretation:** Edge-decision FN rates stay ≤α target on most folds; conformal gate escalates more than cost gate (honest ~45–83% vs ~38–47%). Do **not** claim sub-5% duty cycle.

### k-NN note

On main protocol (heuristic LLM, fold 0), F1 is **identical** across k∈{1,5,10} — retrieval rank does not change heuristic decisions. For paper: report as stability result; optional follow-up is ambiguous-bench k sweep with Ollama.

## Still running / pending

| Item | Status | Notes |
|------|--------|-------|
| Ollama seed sweep (42,43,44) | 🔄 | `results/seed_sweep/ollama_sweep_42_43_44.log` — training seed 43, then eval |
| `tab_seeds.tex` refresh | ⏳ | Blocked until seed sweep writes `seed_sweep_fold0.json` |
| `main.pdf` compile | ⏳ | User: experiments first, then paper |
| k-NN k ablation on ambiguous bench (optional) | ⏳ | Would show k sensitivity under Ollama |

## Ground truth lock (do not change)

- Ambiguous LEAD: Tier-1 F1 0.713→0.896 (Δ+0.183)
- Main SisFall: F1 0.759→0.887, cost/1000 1151→200
- KFall stability: F1 0.987→0.990 (not 0.62→0.81)
- Escalation: main ~47%, ambiguous ~59%, KFall ~1.1%

## Reproduce

```bash
# Full A* suite (CPU-friendly; ~15 min)
python3 scripts/run_astar_experiments.py --folds 0,1,2,3,4 --max-test 1000 --device cpu

# Pareto only
python3 scripts/run_astar_experiments.py --skip-conformal --skip-knn --skip-robustness --skip-fallback --device cpu

# Regenerate tables
python3 scripts/generate_paper_tables.py
```

## Next step (after seed sweep)

1. Verify `results/seed_sweep/seed_sweep_fold0.json` has Ollama seeds 42,43,44
2. `python3 scripts/generate_paper_tables.py` → refresh `tab_seeds.tex`
3. Wire Track B tables into `paper/main.tex`
4. Compile `paper/main.pdf`
