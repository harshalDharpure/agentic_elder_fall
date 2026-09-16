#!/usr/bin/env bash
# After binary KFall pipeline: LLM compare, multiclass context, paper regen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/kfall/kfall_post_finish.log"
echo "=== KFall post-finish waiter START $(date) ===" | tee "$LOG"

while [[ ! -f results/kfall/KFALL_EXTERNAL_DONE.flag ]]; do
  if ! pgrep -f 'run_kfall_q1_pipeline.sh|run_kfall_external.py' >/dev/null; then
    echo "pipeline processes gone; checking partial results" | tee -a "$LOG"
    break
  fi
  sleep 120
done

python -u scripts/aggregate_kfall.py 2>&1 | tee -a "$LOG" || true

# LLM compare on folds that have checkpoints (at least fold 0)
for F in 0 1 2 3 4; do
  CKPT="checkpoints/kfall/cnn_lstm_attn_binary_fold${F}.pt"
  [[ -f "$CKPT" ]] || continue
  OUT="results/kfall/fold${F}/kfall_llm_compare_fold${F}.json"
  if [[ -f "$OUT" ]]; then
    echo "LLM compare fold $F exists — skip" | tee -a "$LOG"
    continue
  fi
  echo "=== LLM compare fold $F $(date) ===" | tee -a "$LOG"
  python -u scripts/run_kfall_llm_compare.py --fold "$F" --device cuda --max-cases 100 2>&1 | tee -a "$LOG" || true
done

# Multiclass context fold 0
if [[ ! -f results/kfall/kfall_multiclass_fold0.json ]]; then
  echo "=== Multiclass context fold 0 $(date) ===" | tee -a "$LOG"
  python -u scripts/run_kfall_multiclass.py --fold 0 --device cuda --epochs 40 --max-test 4000 2>&1 | tee -a "$LOG" || true
fi

python -u scripts/aggregate_kfall.py 2>&1 | tee -a "$LOG" || true
python -u scripts/generate_kfall_paper_tables.py 2>&1 | tee -a "$LOG" || true
python -u scripts/update_paper_for_kfall.py 2>&1 | tee -a "$LOG" || true

(
  cd paper
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  bibtex main >/dev/null || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
)

echo "=== KFall post-finish DONE $(date) ===" | tee -a "$LOG"
touch results/kfall/KFALL_Q1_COMPLETE.flag
