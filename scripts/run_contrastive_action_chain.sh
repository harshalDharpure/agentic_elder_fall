#!/usr/bin/env bash
# After CRC chain finishes: run contrastive action-critique (visible Critic impact).
# Usage: bash scripts/run_contrastive_action_chain.sh [WAIT_PID]
set -euo pipefail
ROOT="/DATA/tauseef_2121cs04/harshal/agentic_fall"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/results"
CHAIN_LOG="$ROOT/logs/contrastive_action_chain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date -Is)] contrastive action-critique chain starting"
echo "log=$CHAIN_LOG"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] Waiting for PID $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date -Is)] PID $WAIT_PID finished"
fi

# Also wait if CRC calibrate/eval still running without the wait pid.
while pgrep -f 'scripts/calibrate_crc_veto.py|scripts/run_ablations.py.*crc_veto|scripts/run_crc_veto_chain.sh' >/dev/null 2>&1; do
  echo "[$(date -Is)] CRC still running — sleep 60s"
  sleep 60
done

COMMON=(--skip-baselines --max-test 1000 --device cuda)

run_fold() {
  local fold="$1"
  local out="results/fold${fold}/ablation_fold${fold}_contrastive_action.json"
  local ckpt="checkpoints/backbones/cnn_lstm_attn_fold${fold}.pt"
  local mem="data/memory/sisfall_fold${fold}.json"
  if [[ ! -f "$ckpt" || ! -f "$mem" ]]; then
    echo "[$(date -Is)] skip fold $fold (missing ckpt or memory)"
    return 0
  fi
  if [[ -f "$out" ]]; then
    echo "[$(date -Is)] skip fold $fold (already have $out)"
    return 0
  fi
  echo "[$(date -Is)] START fold=$fold"
  python -u scripts/run_ablations.py \
    --fold "$fold" \
    --checkpoint "$ckpt" \
    --memory "$mem" \
    --modes "gate_knn_llm,gate_knn_llm_contrastive_action_critique" \
    --out-suffix "_contrastive_action" \
    --protocol-tag "fold${fold}_contrastive_action_critique" \
    "${COMMON[@]}"
  echo "[$(date -Is)] DONE fold=$fold"
}

for f in 0 1 2 3 4; do
  run_fold "$f"
done

python -u scripts/summarize_critic_impact.py || true
echo "[$(date -Is)] All contrastive action-critique jobs finished"
echo "Artifacts: results/fold*/ablation_fold*_contrastive_action.json"
