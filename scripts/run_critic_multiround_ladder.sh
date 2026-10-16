#!/usr/bin/env bash
# SisFall multi-round + KFall critic ladder (paper protocol n=1000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TMPDIR="${TMPDIR:-/DATA/tauseef_2121cs04/tmp}"
mkdir -p "$TMPDIR" logs results/critic_multiround
DEVICE="${DEVICE:-cuda}"
LOG="logs/critic_multiround_ladder_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date) log=$LOG"

echo "===== SisFall multi-round folds 0-4 ====="
for F in 0 1 2 3 4; do
  echo "[SisFall fold=$F] $(date)"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$F" \
    --checkpoint "checkpoints/backbones/cnn_lstm_attn_fold${F}.pt" \
    --memory "data/memory/sisfall_fold${F}.json" \
    --model cnn_lstm_attn \
    --device "$DEVICE" \
    --skip-baselines \
    --modes gate_knn_llm_multi_round_critique \
    --protocol-tag multi_round_critique_1000 \
    --out-dir "results/fold${F}" \
    --out-suffix "_multi_round_critique"
  echo "[SisFall fold=$F] DONE $(date)"
done

echo "===== KFall critic ladder folds 0-4 ====="
KFALL_MODES="gate_knn_llm,gate_knn_llm_action_critique,gate_knn_llm_contrastive_action_critique,gate_knn_llm_multi_round_critique"
for F in 0 1 2 3 4; do
  echo "[KFall fold=$F] $(date)"
  python -u scripts/run_kfall_external.py \
    --kfall-config configs/tier1_kfall_binary.yaml \
    --agentic-config configs/agentic.yaml \
    --fold "$F" \
    --device "$DEVICE" \
    --max-test 1000 \
    --model cnn_lstm_attn \
    --skip-train \
    --out-subdir cnn_lstm_attn_critic_ladder \
    --modes "$KFALL_MODES"
  echo "[KFall fold=$F] DONE $(date)"
done

echo "===== Aggregate comparison ====="
python -u scripts/aggregate_critic_multiround.py
echo "ALL DONE $(date)"
