#!/usr/bin/env bash
# Overnight strengthen for guide meeting:
# 1) Archive fold0/1 full4000 ablations
# 2) Re-run agentic eval with ollama (align with gate_knn_llm)
# 3) Re-run fold0/1 ablations at max_test=1000 for homogeneous paper tables
# 4) Re-aggregate + rebuild GUIDE_BRIEF
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
DEVICE="${DEVICE:-cuda}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
PRIMARY="${PRIMARY_BACKBONE:-cnn_lstm_attn}"
export PYTHONUNBUFFERED=1
LOG=results/FULL_RUN.log

echo "=== OVERNIGHT GUIDE STRENGTHEN START $(date) ===" | tee -a "$LOG"

archive_and_homogenize() {
  local FOLD="$1"
  local OUT="results/fold${FOLD}"
  local CKPT="checkpoints/backbones/${PRIMARY}_fold${FOLD}.pt"
  local MEM="data/memory/sisfall_fold${FOLD}.json"
  mkdir -p "$OUT"

  if [[ -f "$OUT/ablation_fold${FOLD}.json" && ! -f "$OUT/ablation_fold${FOLD}_full4000.json" ]]; then
    cp -a "$OUT/ablation_fold${FOLD}.json" "$OUT/ablation_fold${FOLD}_full4000.json"
    cp -a "$OUT/ablation_fold${FOLD}.csv" "$OUT/ablation_fold${FOLD}_full4000.csv" 2>/dev/null || true
    echo "Archived fold${FOLD} full4000 ablation" | tee -a "$LOG"
  fi

  echo "[fold ${FOLD}] agentic eval ollama $(date)" | tee -a "$LOG"
  python -u scripts/run_agentic_eval.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --backend ollama --device "$DEVICE" \
    2>&1 | tee "$OUT/full_agentic_eval_ollama.log"

  echo "[fold ${FOLD}] ablations max_test=1000 $(date)" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config "$PAPER" --fold "$FOLD" --checkpoint "$CKPT" --memory "$MEM" \
    --device "$DEVICE" --skip-baselines --out-dir "$OUT" \
    --max-test 1000 \
    --modes tier1_only,gate_only,gate_knn,gate_knn_llm \
    --protocol-tag fast_1000_no_gate_llm \
    2>&1 | tee "$OUT/full_ablation_homogenize1000.log"

  cat > "$OUT/protocol_fold${FOLD}.json" <<EOF
{"fold": ${FOLD}, "protocol_tag": "fast_1000_no_gate_llm", "max_test": 1000, "modes": "tier1_only,gate_only,gate_knn,gate_knn_llm", "backbone": "zoo", "note": "homogenized for Q1 main table; full4000 archived"}
EOF
}

for FOLD in 0 1; do
  archive_and_homogenize "$FOLD"
done

echo "[aggregate] $(date)" | tee -a "$LOG"
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee results/full_aggregate.log
python -u scripts/export_tables.py 2>&1 | tee results/full_export.log
python -u scripts/build_guide_brief.py 2>&1 | tee -a results/full_export.log

echo "=== OVERNIGHT GUIDE STRENGTHEN DONE $(date) ===" | tee -a "$LOG"
