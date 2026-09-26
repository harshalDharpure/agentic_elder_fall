# KFall Q1 — background jobs

## Running now (do not kill)
1. `scripts/run_kfall_q1_pipeline.sh` — Fold 0 full ladder, then folds 1–4 (`tier1_only` vs `gate_knn_llm`)
2. `scripts/wait_kfall_then_finish.sh` — waits for pipeline DONE, then:
   - `aggregate_kfall.py`
   - LLM compare folds 0–4
   - 36-class multiclass context (fold 0)
   - paper tables + `main.tex` refresh + PDF

## Watch
```bash
tail -f results/kfall/kfall_q1_pipeline.nohup.out
tail -f results/kfall/kfall_post_finish.nohup.out
ls results/kfall/*DONE* results/kfall/*COMPLETE*
```

## Done when
- `results/kfall/KFALL_EXTERNAL_DONE.flag`
- `results/kfall/KFALL_Q1_COMPLETE.flag`
