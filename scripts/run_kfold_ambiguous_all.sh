#!/usr/bin/env bash
# Run complete architecture (gate_knn_llm + Ollama) on all K-fold ambiguous eval sets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/kfold_ambiguous"
mkdir -p "$OUT" "$OUT/logs"
LOG="$OUT/logs/all_folds_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date -Is) backend=ollama mode=gate_knn_llm ==="
echo "log=$LOG"

for fold in 0 1 2 3 4; do
  echo ""
  echo "===== FOLD $fold $(date -Is) ====="
  python -u scripts/eval_kfold_ambiguous_bench.py \
    --fold "$fold" \
    --backend ollama \
    --force-ambiguous \
    --out-dir results/kfold_ambiguous
done

python -u scripts/aggregate_kfold_ambiguous.py --out-dir results/kfold_ambiguous --backend ollama

echo "=== DONE $(date -Is) ==="
