#!/usr/bin/env bash
# Re-run System-2 paths after stripping activity-code leakage from evidence/memory.
# Writes to results/noleak_lock/ so original paper JSONs are preserved.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/noleak_lock results/prompt_audit
DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
FOLDS="${FOLDS:-0 1 2 3 4}"
LOG="$ROOT/logs/noleak_system2_ladder_$(date +%Y%m%d_%H%M%S).log"
echo "=== NOLEAK System-2 START $(date) ===" | tee -a "$LOG"

python -u scripts/audit_prompt_leakage.py --self-test 2>&1 | tee -a "$LOG"

# Scrub memories unless already clean (set SKIP_SCRUB=1 to force-skip).
if [[ "${SKIP_SCRUB:-0}" != "1" ]]; then
  if ! python -u scripts/audit_prompt_leakage.py --memory-dir data/memory >/dev/null 2>&1; then
    echo "[scrub] memory audit failed; scrubbing..." | tee -a "$LOG"
    python -u scripts/scrub_memory_activity.py --memory-dir data/memory 2>&1 | tee -a "$LOG"
  else
    echo "[scrub] memory already clean; skipping scrub" | tee -a "$LOG"
  fi
else
  echo "[scrub] SKIP_SCRUB=1" | tee -a "$LOG"
fi
python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"

for FOLD in $FOLDS; do
  CKPT="$ROOT/checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="$ROOT/data/memory/sisfall_fold${FOLD}.json"
  OUT="$ROOT/results/noleak_lock/fold${FOLD}"
  mkdir -p "$OUT"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: missing checkpoint $CKPT" | tee -a "$LOG"
    exit 1
  fi
  echo "[SisFall fold=$FOLD] $(date)" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "$CKPT" \
    --memory "$MEM" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "$OUT" \
    --max-test "$MAX_TEST" \
    --modes tier1_only,gate_only,gate_knn,gate_llm,gate_knn_llm \
    --protocol-tag "noleak_lock_${MAX_TEST}" \
    2>&1 | tee -a "$LOG"

  # Audit System-2 surfaces written for this fold (ablation JSON if present).
  shopt -s nullglob
  for RJ in "$OUT"/*.json; do
    python -u scripts/audit_prompt_leakage.py --results "$RJ" 2>&1 | tee -a "$LOG" || {
      echo "LEAK AUDIT FAILED for $RJ" | tee -a "$LOG"
      exit 1
    }
  done
done

echo "=== NOLEAK System-2 ALL DONE $(date) ===" | tee -a "$LOG"
touch "$ROOT/results/noleak_lock/NOLEAK_SYSTEM2_DONE.flag"
