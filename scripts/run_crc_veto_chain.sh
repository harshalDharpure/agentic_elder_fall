#!/usr/bin/env bash
# Detached CRC veto chain: val calibration → test eval → α-sweep (folds 0–4).
# Usage: bash scripts/run_crc_veto_chain.sh [WAIT_PID]
set -euo pipefail
ROOT="/DATA/tauseef_2121cs04/harshal/agentic_fall"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/results/fold0"
CHAIN_LOG="$ROOT/logs/crc_veto_chain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date -Is)] CRC veto chain starting"
echo "log=$CHAIN_LOG"
echo "cuda=$(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)')"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] Waiting for PID $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date -Is)] PID $WAIT_PID finished"
fi

COMMON=(--max-test 1000 --device cuda --skip-baselines)

run_fold() {
  local fold="$1"
  local ckpt="checkpoints/backbones/cnn_lstm_attn_fold${fold}.pt"
  local mem="data/memory/sisfall_fold${fold}.json"
  local cal="results/fold${fold}/veto_calibration_fold${fold}.json"
  local out="results/fold${fold}/ablation_fold${fold}_crc_veto.json"
  if [[ ! -f "$ckpt" || ! -f "$mem" ]]; then
    echo "[$(date -Is)] skip fold $fold (missing ckpt or memory)"
    return 0
  fi
  echo "[$(date -Is)] CALIBRATE fold=$fold"
  python -u scripts/calibrate_crc_veto.py \
    --fold "$fold" \
    --checkpoint "$ckpt" \
    --memory "$mem" \
    --device cuda \
    --max-cal 1000
  echo "[$(date -Is)] EVAL fold=$fold"
  python -u scripts/run_ablations.py \
    --fold "$fold" \
    --checkpoint "$ckpt" \
    --memory "$mem" \
    --modes "gate_knn_llm,gate_knn_llm_crc_veto" \
    --out-suffix "_crc_veto" \
    --protocol-tag "fold${fold}_crc_veto_alpha05" \
    --crc-cal "$cal" \
    --crc-alpha 0.05 \
    "${COMMON[@]}"
  echo "[$(date -Is)] SWEEP fold=$fold"
  python -u scripts/sweep_crc_alpha.py --fold "$fold" || true
  echo "[$(date -Is)] DONE fold=$fold out=$out"
}

for f in 0 1 2 3 4; do
  run_fold "$f"
done

echo "[$(date -Is)] All CRC veto jobs finished"
echo "Artifacts: results/fold*/veto_calibration_fold*.json results/fold*/ablation_fold*_crc_veto.json"
