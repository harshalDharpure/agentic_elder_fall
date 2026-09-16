# Background Q1 finish jobs

## Running now
- `scripts/run_gate_llm_5fold.sh` — adds `gate_llm` to all 5 folds @ n=1000, then full4000 appendix for folds 2–4
- `scripts/wait_then_finish_q1.sh` — after gate_llm, trains backbone zoo folds 2–4, regenerates tables/figures/PDF

## Already done
- Expanded journal manuscript + figures (`paper/main.pdf`)
- Multi-seed Fold-0 sweep → `results/seed_sweep/seed_sweep_fold0.json`
- KFall harness + `results/kfall/ACCESS_REQUIRED.json` (raw data gated)

## Watch
```bash
tail -f results/gate_llm_5fold.nohup.out
tail -f results/q1_finish_waiter.log
ls results/*DONE*.flag
```

## ETA (rough)
- gate_llm fold0 ~40–60 min remaining at current rate, then folds 1–4
- full pipeline may take several hours (LLM escalation)
