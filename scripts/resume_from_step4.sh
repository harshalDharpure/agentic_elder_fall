#!/usr/bin/env bash
# Resume Q1 suite after fold-0 steps 1–3 completed.
# Fold 0: only step 4 (agentic eval + LLM compare)
# Folds 1–4: full pipeline
# Then aggregate + export
#
# Usage:
#   nohup env DEVICE=cuda bash scripts/resume_from_step4.sh >> results/nohup_full_experiments.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
DEVICE="${DEVICE:-cuda}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
PRIMARY="${PRIMARY_BACKBONE:-cnn_lstm_attn}"
export PYTHONUNBUFFERED=1

echo "=== RESUME FROM STEP 4 START $(date) ===" | tee -a results/FULL_RUN.log
echo "DEVICE=$DEVICE PAPER=$PAPER PRIMARY=$PRIMARY" | tee -a results/FULL_RUN.log

# ----- Fold 0: step 4 only -----
FOLD=0
OUT="results/fold${FOLD}"
mkdir -p "$OUT"
CKPT="checkpoints/backbones/${PRIMARY}_fold${FOLD}.pt"
MEM="data/memory/sisfall_fold${FOLD}.json"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint $CKPT" | tee -a results/FULL_RUN.log
  exit 1
fi
if [[ ! -f "$MEM" ]]; then
  echo "ERROR: missing memory $MEM" | tee -a results/FULL_RUN.log
  exit 1
fi

echo "===== FOLD ${FOLD} (resume step 4 only) =====" | tee -a results/FULL_RUN.log
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

# ----- Folds 1–4: full pipeline -----
for FOLD in 1 2 3 4; do
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
