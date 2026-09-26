#!/usr/bin/env bash
# Hybrid Q1: folds 2–4 under the fast protocol.
# - Primary backbone only (cnn_lstm_attn)
# - Ablations: max_test=1000, skip gate_llm
# - Agentic eval uses Ollama (matches gate_knn_llm)
# - Then aggregate + export
#
# Usage:
#   nohup env DEVICE=cuda bash scripts/resume_folds_2_4_fast.sh >> results/nohup_full_experiments.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
DEVICE="${DEVICE:-cuda}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
PRIMARY="${PRIMARY_BACKBONE:-cnn_lstm_attn}"
ABLATION_MAX_TEST="${ABLATION_MAX_TEST:-1000}"
ABLATION_MODES="${ABLATION_MODES:-tier1_only,gate_only,gate_knn,gate_knn_llm}"
PROTOCOL_TAG="${PROTOCOL_TAG:-fast_1000_no_gate_llm}"
export PYTHONUNBUFFERED=1

echo "=== RESUME FOLDS 2–4 FAST START $(date) ===" | tee -a results/FULL_RUN.log
echo "DEVICE=$DEVICE PAPER=$PAPER PRIMARY=$PRIMARY MAX_TEST=$ABLATION_MAX_TEST MODES=$ABLATION_MODES TAG=$PROTOCOL_TAG" \
  | tee -a results/FULL_RUN.log

for FOLD in 2 3 4; do
  OUT="results/fold${FOLD}"
  mkdir -p "$OUT"
  echo "===== FOLD ${FOLD} (fast protocol) =====" | tee -a results/FULL_RUN.log

  echo "[1/5] Primary backbone only (${PRIMARY}, fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_backbone_benchmark.py \
    --paper-config "$PAPER" --fold "$FOLD" --device "$DEVICE" --out-dir "$OUT" \
    --models "$PRIMARY" \
    2>&1 | tee "$OUT/full_backbone_benchmark.log"
  echo "[1/5] DONE $(date)" | tee -a results/FULL_RUN.log

  CKPT="checkpoints/backbones/${PRIMARY}_fold${FOLD}.pt"
  MEM="data/memory/sisfall_fold${FOLD}.json"
  echo "[2/5] Build memory (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/build_memory.py \
    --checkpoint "$CKPT" --fold "$FOLD" --model "$PRIMARY" --device "$DEVICE" \
    2>&1 | tee "$OUT/build_memory.log"
  echo "[2/5] DONE $(date)" | tee -a results/FULL_RUN.log

  echo "[3/5] Ablations fast (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_ablations.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --device "$DEVICE" --skip-baselines --out-dir "$OUT" \
    --max-test "$ABLATION_MAX_TEST" \
    --modes "$ABLATION_MODES" \
    --protocol-tag "$PROTOCOL_TAG" \
    2>&1 | tee "$OUT/full_ablation.log"
  echo "[3/5] DONE $(date)" | tee -a results/FULL_RUN.log

  echo "[4/5] Agentic eval (ollama) + LLM compare (fold ${FOLD})" | tee -a results/FULL_RUN.log
  python -u scripts/run_agentic_eval.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --backend ollama --device "$DEVICE" \
    2>&1 | tee "$OUT/full_agentic_eval.log"
  python -u scripts/run_llm_compare.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --device "$DEVICE" --out-dir "$OUT" \
    2>&1 | tee "$OUT/full_llm_compare.log"
  echo "[4/5] DONE $(date)" | tee -a results/FULL_RUN.log

  cat > "$OUT/protocol_fold${FOLD}.json" <<EOF
{"fold": ${FOLD}, "protocol_tag": "${PROTOCOL_TAG}", "max_test": ${ABLATION_MAX_TEST}, "modes": "${ABLATION_MODES}", "backbone": "primary_only"}
EOF
done

echo "[5/5] Aggregate + export" | tee -a results/FULL_RUN.log
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee results/full_aggregate.log
python -u scripts/export_tables.py 2>&1 | tee results/full_export.log
echo "[5/5] DONE $(date)" | tee -a results/FULL_RUN.log

echo "=== FOLDS 2–4 FAST COMPLETE $(date) ===" | tee -a results/FULL_RUN.log
