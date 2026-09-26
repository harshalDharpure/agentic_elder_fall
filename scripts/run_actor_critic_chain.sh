#!/usr/bin/env bash
# Background Actor–Critic experiment chain.
# Waits for an optional PID, then runs remaining Fold-0 (+ optional multi-fold) modes.
set -euo pipefail
ROOT="/DATA/tauseef_2121cs04/harshal/agentic_fall"
cd "$ROOT"
LOGDIR="$ROOT/results/fold0"
mkdir -p "$LOGDIR" "$ROOT/logs"
CHAIN_LOG="$ROOT/logs/actor_critic_chain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date -Is)] Actor–Critic background chain starting"
echo "log=$CHAIN_LOG"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] Waiting for PID $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date -Is)] PID $WAIT_PID finished"
fi

CKPT0="checkpoints/backbones/cnn_lstm_attn_fold0.pt"
MEM0="data/memory/sisfall_fold0.json"
COMMON=(--skip-baselines --max-test 1000 --device cuda)

run_modes() {
  local fold="$1"
  local ckpt="$2"
  local mem="$3"
  local modes="$4"
  local suffix="$5"
  local tag="$6"
  echo "[$(date -Is)] START fold=$fold modes=$modes suffix=$suffix"
  python -u scripts/run_ablations.py \
    --fold "$fold" \
    --checkpoint "$ckpt" \
    --memory "$mem" \
    --modes "$modes" \
    --out-suffix "$suffix" \
    --protocol-tag "$tag" \
    "${COMMON[@]}"
  echo "[$(date -Is)] DONE fold=$fold modes=$modes"
}

# 1) Ensure primary Fold-0 hybrid compare exists / refresh if missing
if [[ ! -f "$LOGDIR/ablation_fold0_actor_critic_llm.json" ]]; then
  echo "[$(date -Is)] Primary llm artifact missing — running gate_knn_llm + actor_critic"
  run_modes 0 "$CKPT0" "$MEM0" \
    "gate_knn_llm,gate_knn_llm_actor_critic" \
    "_actor_critic_llm" \
    "fold0_actor_critic_llm_hybrid_1000"
else
  echo "[$(date -Is)] Found $LOGDIR/ablation_fold0_actor_critic_llm.json"
fi

# 2) Extra Fold-0 LLM ablations (actor_only + roleswap)
run_modes 0 "$CKPT0" "$MEM0" \
  "gate_knn_llm_actor_only,gate_knn_llm_actor_critic_roleswap" \
  "_actor_critic_llm_extra" \
  "fold0_actor_critic_llm_extra_1000"

# 3) Multi-fold hybrid (folds with available checkpoints)
for f in 1 2 3 4; do
  ckpt="checkpoints/backbones/cnn_lstm_attn_fold${f}.pt"
  mem="data/memory/sisfall_fold${f}.json"
  if [[ -f "$ckpt" && -f "$mem" ]]; then
    run_modes "$f" "$ckpt" "$mem" \
      "gate_knn_llm,gate_knn_llm_actor_critic" \
      "_actor_critic_llm" \
      "fold${f}_actor_critic_llm_hybrid_1000"
  else
    echo "[$(date -Is)] skip fold $f (missing ckpt or memory)"
  fi
done

echo "[$(date -Is)] All Actor–Critic background jobs finished"
echo "Artifacts under results/fold*/ablation_fold*_actor_critic_*.json"
