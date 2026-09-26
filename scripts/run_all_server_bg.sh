#!/usr/bin/env bash
# Master background runner: aggregate K-fold ambiguous results + optional ablations.
# Usage:
#   nohup bash scripts/run_all_server_bg.sh > results/logs/all_server_bg.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs results/kfold_ambiguous results/kfold_ambiguous_heuristic
DEVICE="${DEVICE:-cuda}"
export PYTHONUNBUFFERED=1

MASTER_LOG="results/logs/all_server_bg_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== ALL SERVER EXPERIMENTS START $(date -Is) ==="
echo "DEVICE=$DEVICE log=$MASTER_LOG"

aggregate_kfold() {
  local OUT="$1"
  local BACKEND="$2"
  python -u scripts/aggregate_kfold_ambiguous.py --out-dir "$OUT" --backend "$BACKEND"
}

run_kfold_backend() {
  local BACKEND="$1"
  local OUT="$2"
  mkdir -p "$OUT/logs"
  echo ""
  echo "===== K-fold ambiguous bench backend=$BACKEND out=$OUT $(date -Is) ====="
  for fold in 0 1 2 3 4; do
    if [[ -f "$OUT/fold${fold}_eval.json" ]]; then
      echo "fold $fold already done — skip"
      continue
    fi
    echo "----- fold $fold $(date -Is) -----"
    python -u scripts/eval_kfold_ambiguous_bench.py \
      --fold "$fold" \
      --backend "$BACKEND" \
      --force-ambiguous \
      --out-dir "$OUT"
  done
  aggregate_kfold "$OUT" "$BACKEND"
  echo "===== DONE backend=$BACKEND $(date -Is) ====="
}

# ---------- 1) Heuristic all folds (skip eval if done, always aggregate) ----------
run_kfold_backend heuristic results/kfold_ambiguous_heuristic

# ---------- 2) Ollama all folds ----------
run_kfold_backend ollama results/kfold_ambiguous

# ---------- 3) Primary ablation gate_knn_llm (skip if already in ablation JSON) ----------
echo ""
echo "===== Primary ablations gate_knn_llm folds 0-4 $(date -Is) ====="
for FOLD in 0 1 2 3 4; do
  CKPT="checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="data/memory/sisfall_fold${FOLD}.json"
  OUT="results/fold${FOLD}"
  ABL="$OUT/ablation_fold${FOLD}.json"
  mkdir -p "$OUT"
  if [[ ! -f "$CKPT" ]]; then
    echo "Missing $CKPT — skip fold $FOLD"
    continue
  fi
  if [[ -f "$ABL" ]] && python -u - <<PY
import json
from pathlib import Path
d = json.loads(Path("$ABL").read_text())
rows = d.get("rows", d)
names = {r.get("name") for r in rows if isinstance(r, dict)}
raise SystemExit(0 if "gate_knn_llm" in names else 1)
PY
  then
    echo "fold $FOLD: gate_knn_llm already in $ABL — skip"
    continue
  fi
  echo "----- ablation fold $FOLD $(date -Is) -----"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "$CKPT" \
    --memory "$MEM" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "$OUT" \
    --modes gate_knn_llm \
    --protocol-tag kfold_ambiguous_followup \
    --merge-existing \
    2>&1 | tee "$OUT/ablation_gate_knn_llm_bg.log"
done

echo ""
echo "===== Aggregate folds $(date -Is) ====="
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee results/logs/aggregate_bg.log || true

echo "=== ALL SERVER EXPERIMENTS COMPLETE $(date -Is) ==="
echo "Reports:"
echo "  results/kfold_ambiguous_heuristic/ALL_FOLDS_REPORT.txt"
echo "  results/kfold_ambiguous/ALL_FOLDS_REPORT.txt"
echo "  $MASTER_LOG"
