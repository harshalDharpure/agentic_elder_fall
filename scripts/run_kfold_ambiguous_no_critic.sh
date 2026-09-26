#!/usr/bin/env bash
# Run K-fold ambiguous bench: tier1_only + gate_knn_llm (NO CRITIC) on all folds.
# Usage: nohup bash scripts/run_kfold_ambiguous_no_critic.sh > results/kfold_ambiguous/logs/nohup_no_critic.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/kfold_ambiguous/logs results/kfold_ambiguous_tier1/logs
export PYTHONUNBUFFERED=1
BACKEND="${BACKEND:-ollama}"
LOG="results/kfold_ambiguous/logs/no_critic_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== K-fold ambiguous bench NO CRITIC backend=$BACKEND $(date -Is) ==="

# 1) tier1_only baseline (fast, no LLM)
echo ""
echo "===== tier1_only (Bhatti-style, threshold 0.5) ====="
for fold in 0 1 2 3 4; do
  echo "--- fold $fold tier1_only ---"
  python -u scripts/eval_kfold_ambiguous_bench.py \
    --fold "$fold" \
    --mode tier1_only \
    --backend heuristic \
    --no-force-ambiguous \
    --out-dir results/kfold_ambiguous_tier1
done

# 2) gate_knn_llm full stack (no Critic)
echo ""
echo "===== gate_knn_llm (Tier-1 → gate → kNN → LLM Actor) NO CRITIC ====="
for fold in 0 1 2 3 4; do
  if [[ -f "results/kfold_ambiguous/fold${fold}_eval.json" ]] && \
     python3 -c "import json; d=json.load(open('results/kfold_ambiguous/fold${fold}_eval.json')); exit(0 if d.get('mode')=='gate_knn_llm' and d.get('backend')=='$BACKEND' else 1)" 2>/dev/null; then
    echo "fold $fold gate_knn_llm already done — skip"
    continue
  fi
  echo "--- fold $fold gate_knn_llm backend=$BACKEND ---"
  python -u scripts/eval_kfold_ambiguous_bench.py \
    --fold "$fold" \
    --mode gate_knn_llm \
    --backend "$BACKEND" \
    --force-ambiguous \
    --out-dir results/kfold_ambiguous
done

python -u scripts/aggregate_kfold_ambiguous.py --out-dir results/kfold_ambiguous --backend "$BACKEND"
python -u scripts/aggregate_kfold_ambiguous.py --out-dir results/kfold_ambiguous_tier1 --backend tier1_only
python -u scripts/compare_kfold_ambiguous_no_critic.py

echo "=== DONE $(date -Is) ==="
echo "Reports:"
echo "  results/kfold_ambiguous/ALL_FOLDS_REPORT.txt"
echo "  results/kfold_ambiguous/COMPARISON_NO_CRITIC.txt"
