#!/usr/bin/env bash
# Best honest Q1 ladder: no activity leakage + strongest System-2 modes.
# Writes to results/noleak_lock/ (does not touch leaky-era results/fold*).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/noleak_lock
DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# Primary headline candidates under no-leak protocol.
MODES="${MODES:-tier1_only,gate_knn_llm,gate_knn_llm_fuse,gate_knn_llm_action_critique}"
SKIP_SCRUB="${SKIP_SCRUB:-1}"
LOG="$ROOT/logs/best_noleak_ladder_$(date +%Y%m%d_%H%M%S).log"
echo "=== BEST NOLEAK LADDER START $(date) ===" | tee -a "$LOG"
echo "modes=$MODES folds=$FOLDS" | tee -a "$LOG"

python -u scripts/audit_prompt_leakage.py --self-test 2>&1 | tee -a "$LOG"
if [[ "$SKIP_SCRUB" != "1" ]]; then
  python -u scripts/scrub_memory_activity.py --memory-dir data/memory 2>&1 | tee -a "$LOG"
fi
python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"

for FOLD in $FOLDS; do
  CKPT="$ROOT/checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="$ROOT/data/memory/sisfall_fold${FOLD}.json"
  OUT="$ROOT/results/noleak_lock/fold${FOLD}"
  mkdir -p "$OUT"
  [[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
  echo "[fold=$FOLD] $(date)" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "$CKPT" \
    --memory "$MEM" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "$OUT" \
    --max-test "$MAX_TEST" \
    --modes "$MODES" \
    --protocol-tag "best_noleak_${MAX_TEST}" \
    --merge-existing \
    2>&1 | tee -a "$LOG"
  shopt -s nullglob
  for RJ in "$OUT"/windows_*.json; do
    python -u scripts/audit_prompt_leakage.py --results "$RJ" 2>&1 | tee -a "$LOG"
  done
done

echo "=== BEST NOLEAK LADDER DONE $(date) ===" | tee -a "$LOG"
touch "$ROOT/results/noleak_lock/BEST_NOLEAK_DONE.flag"
