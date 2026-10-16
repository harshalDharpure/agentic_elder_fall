#!/usr/bin/env bash
# Re-run System-2 paths after stripping activity-code leakage from evidence/memory.
# Writes to results/noleak/ so original paper JSONs are preserved until numbers are validated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/noleak results/prompt_audit
DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
LOG="$ROOT/logs/noleak_system2_ladder_$(date +%Y%m%d_%H%M%S).log"
echo "=== NOLEAK System-2 START $(date) ===" | tee -a "$LOG"

# Audit primary memories used by paper System-2 runs
python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"

# Fix KFall stress default backbone to match paper (CNN–LSTM–Attention)
KFALL_MODEL="${KFALL_MODEL:-cnn_lstm_attn}"

for FOLD in 0 1 2 3 4; do
  CKPT="$ROOT/checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="$ROOT/data/memory/sisfall_fold${FOLD}.json"
  OUT="$ROOT/results/noleak/fold${FOLD}"
  mkdir -p "$OUT"
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
    --protocol-tag "noleak_${MAX_TEST}" \
    2>&1 | tee -a "$LOG"
done

# SisFall ambiguous stress bench (if present)
if [[ -d data/kfold_ambiguous ]]; then
  for FOLD in 0 1 2 3 4; do
    echo "[SisFall stress fold=$FOLD] $(date)" | tee -a "$LOG"
    python -u scripts/eval_kfold_ambiguous_bench.py \
      --fold "$FOLD" \
      --checkpoint "checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt" \
      --memory "data/memory/sisfall_fold${FOLD}.json" \
      --device "$DEVICE" \
      --out-dir "results/noleak/kfold_ambiguous" \
      2>&1 | tee -a "$LOG" || echo "WARN: SisFall stress fold $FOLD failed" | tee -a "$LOG"
  done
fi

# KFall stress / external modes used in paper
if [[ -f scripts/eval_kfall_ambiguous_bench.py ]]; then
  for FOLD in 0 1 2 3 4; do
    echo "[KFall stress fold=$FOLD] $(date)" | tee -a "$LOG"
    python -u scripts/eval_kfall_ambiguous_bench.py \
      --fold "$FOLD" \
      --model "$KFALL_MODEL" \
      --device "$DEVICE" \
      --out-dir "results/noleak/kfold_ambiguous_kfall" \
      2>&1 | tee -a "$LOG" || echo "WARN: KFall stress fold $FOLD failed" | tee -a "$LOG"
  done
fi

# Prompt audit on logged System-2 prompts
python -u scripts/audit_prompt_leakage.py --prompts results/prompt_audit 2>&1 | tee -a "$LOG" || true

echo "=== NOLEAK System-2 ALL DONE $(date) ===" | tee -a "$LOG"
touch "$ROOT/results/noleak/NOLEAK_SYSTEM2_DONE.flag"
