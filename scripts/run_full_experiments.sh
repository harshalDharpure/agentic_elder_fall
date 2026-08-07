#!/usr/bin/env bash
# Full Q1 experiment suite under configs/paper_protocol.yaml
# Usage:
#   FOLDS=0 DEVICE=cuda nohup bash scripts/run_full_experiments.sh &
#   FOLDS=0,1,2,3,4 bash scripts/run_full_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
DEVICE="${DEVICE:-cuda}"
FOLDS="${FOLDS:-0}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
PRIMARY="${PRIMARY_BACKBONE:-cnn_lstm_attn}"
export PYTHONUNBUFFERED=1

echo "=== FULL EXPERIMENTS START $(date) ===" | tee results/FULL_RUN.log
echo "FOLDS=$FOLDS DEVICE=$DEVICE PAPER=$PAPER PRIMARY=$PRIMARY" | tee -a results/FULL_RUN.log

IFS=',' read -ra FOLD_ARR <<< "$FOLDS"
for FOLD in "${FOLD_ARR[@]}"; do
  FOLD="$(echo "$FOLD" | tr -d '[:space:]')"
  OUT="results/fold${FOLD}"
  mkdir -p "$OUT"
  echo "===== FOLD ${FOLD} =====" | tee -a results/FULL_RUN.log

  echo "[1/5] Backbone zoo + top-N agentic (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_backbone_benchmark.py \
    --paper-config "$PAPER" --fold "$FOLD" --device "$DEVICE" --out-dir "$OUT" \
    2>&1 | tee "$OUT/full_backbone_benchmark.log"
  echo "[1/5] DONE $(date)" | tee -a results/FULL_RUN.log

  CKPT="checkpoints/backbones/${PRIMARY}_fold${FOLD}.pt"
  MEM="data/memory/sisfall_fold${FOLD}.json"
  echo "[2/5] Build memory (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/build_memory.py \
    --checkpoint "$CKPT" --fold "$FOLD" --model "$PRIMARY" --device "$DEVICE" \
    2>&1 | tee "$OUT/build_memory.log"
  echo "[2/5] DONE $(date)" | tee -a results/FULL_RUN.log

  echo "[3/5] Ablations (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_ablations.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --device "$DEVICE" --skip-baselines --out-dir "$OUT" \
    2>&1 | tee "$OUT/full_ablation.log"
  echo "[3/5] DONE $(date)" | tee -a results/FULL_RUN.log

  echo "[4/5] Agentic eval + LLM compare (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_agentic_eval.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --backend ollama --device "$DEVICE" \
    2>&1 | tee "$OUT/full_agentic_eval.log"
  python -u scripts/run_llm_compare.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --device "$DEVICE" --out-dir "$OUT" \
    2>&1 | tee "$OUT/full_llm_compare.log"
  echo "[4/5] DONE $(date)" | tee -a results/FULL_RUN.log
done

echo "[5/5] Aggregate + export" | tee -a results/FULL_RUN.log
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee results/full_aggregate.log
python -u scripts/export_tables.py 2>&1 | tee results/full_export.log
echo "[5/5] DONE $(date)" | tee -a results/FULL_RUN.log

echo "=== ALL EXPERIMENTS COMPLETE $(date) ===" | tee -a results/FULL_RUN.log
