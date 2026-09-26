#!/usr/bin/env bash
# Complete backbone zoo on folds 2–4 (folds 0–1 already have full zoo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda}"
LOG="$ROOT/results/backbone_zoo_folds_2_4.log"
mkdir -p "$ROOT/results"
echo "=== backbone zoo folds 2–4 START $(date) ===" | tee -a "$LOG"

for FOLD in 2 3 4; do
  # Detect whether non-primary zoo models already exist for this fold
  N_OTHER=$(ls checkpoints/backbones/*_fold${FOLD}.pt 2>/dev/null | grep -vc 'cnn_lstm_attn' || true)
  if [[ "${N_OTHER}" -ge 8 ]]; then
    echo "Fold ${FOLD}: zoo checkpoints present (${N_OTHER}) — skip train" | tee -a "$LOG"
    continue
  fi
  echo "Fold ${FOLD}: training full backbone zoo" | tee -a "$LOG"
  python -u scripts/run_backbone_benchmark.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --device "$DEVICE" \
    --out-dir "results/fold${FOLD}" \
    --skip-agentic \
    2>&1 | tee -a "$LOG"
done

python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee -a "$LOG"
python -u scripts/generate_paper_tables.py 2>&1 | tee -a "$LOG"
echo "=== backbone zoo folds 2–4 DONE $(date) ===" | tee -a "$LOG"
touch "$ROOT/results/BACKBONE_ZOO_2_4_DONE.flag"
