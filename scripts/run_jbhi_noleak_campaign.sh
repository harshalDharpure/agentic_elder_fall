#!/usr/bin/env bash
# JBHI acceptance campaign (honest no-leak):
#   A) SisFall best ladder (fuse + action_critique) → results/noleak_lock/fold*
#   B) KFall binary 5-fold under same locks → results/noleak_lock/kfall/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/noleak_lock/kfall
DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
LOG="$ROOT/logs/jbhi_noleak_campaign_$(date +%Y%m%d_%H%M%S).log"
echo "=== JBHI NOLEAK CAMPAIGN START $(date) ===" | tee -a "$LOG"

python -u scripts/audit_prompt_leakage.py --self-test 2>&1 | tee -a "$LOG"
python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG" || {
  echo "[scrub] memory dirty; scrubbing…" | tee -a "$LOG"
  python -u scripts/scrub_memory_activity.py --memory-dir data/memory 2>&1 | tee -a "$LOG"
  python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"
}

# ---- A) SisFall best modes (fold0 already has baseline; merge fuse+critique, then folds 1-4) ----
echo "=== A SisFall best ladder ===" | tee -a "$LOG"
# Fold 0 extras first
python -u scripts/run_ablations.py \
  --paper-config configs/paper_protocol.yaml \
  --fold 0 \
  --checkpoint checkpoints/backbones/cnn_lstm_attn_fold0.pt \
  --memory data/memory/sisfall_fold0.json \
  --device "$DEVICE" \
  --skip-baselines \
  --out-dir results/noleak_lock/fold0 \
  --max-test "$MAX_TEST" \
  --modes gate_knn_llm_fuse,gate_knn_llm_action_critique \
  --protocol-tag "best_noleak_${MAX_TEST}" \
  --merge-existing \
  2>&1 | tee -a "$LOG"

for FOLD in 1 2 3 4; do
  echo "[SisFall fold=$FOLD] $(date)" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt" \
    --memory "data/memory/sisfall_fold${FOLD}.json" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "results/noleak_lock/fold${FOLD}" \
    --max-test "$MAX_TEST" \
    --modes tier1_only,gate_knn_llm,gate_knn_llm_fuse,gate_knn_llm_action_critique \
    --protocol-tag "best_noleak_${MAX_TEST}" \
    2>&1 | tee -a "$LOG"
done

# Leak audit SisFall windows
shopt -s nullglob
for RJ in results/noleak_lock/fold*/windows_*.json; do
  python -u scripts/audit_prompt_leakage.py --results "$RJ" 2>&1 | tee -a "$LOG"
done

# ---- B) KFall binary no-leak (cnn_lstm_attn, skip train, rebuild memory if needed) ----
echo "=== B KFall binary no-leak ===" | tee -a "$LOG"
# Force scrubbed / rebuilt KFall memories for paper lock
for FOLD in 0 1 2 3 4; do
  MEM="data/memory/kfall_cnn_lstm_attn_fold${FOLD}.json"
  if [[ -f "$MEM" ]]; then
    # scrub in place (case ids / evidence)
    python -u scripts/scrub_memory_activity.py --memory-dir data/memory --pattern "kfall_cnn_lstm_attn_fold${FOLD}.json" 2>&1 | tee -a "$LOG" || true
  fi
  echo "[KFall fold=$FOLD] $(date)" | tee -a "$LOG"
  python -u scripts/run_kfall_external.py \
    --fold "$FOLD" \
    --model cnn_lstm_attn \
    --skip-train \
    --max-test "$MAX_TEST" \
    --modes tier1_only,gate_knn_llm \
    --device "$DEVICE" \
    --out-subdir noleak_lock \
    2>&1 | tee -a "$LOG"
  # also copy into results/noleak_lock/kfall/
  src="results/kfall/noleak_lock/fold${FOLD}/kfall_external_fold${FOLD}.json"
  if [[ -f "$src" ]]; then
    cp -f "$src" "results/noleak_lock/kfall/kfall_external_fold${FOLD}.json"
  fi
done

python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"

echo "=== JBHI NOLEAK CAMPAIGN DONE $(date) ===" | tee -a "$LOG"
touch results/noleak_lock/JBHI_CAMPAIGN_DONE.flag
