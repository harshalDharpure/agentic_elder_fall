#!/usr/bin/env bash
# KFall ambiguous bench: tier1_only + gate_knn_llm (NO CRITIC), all folds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
BACKEND="${BACKEND:-ollama}"
mkdir -p results/kfold_ambiguous_kfall/logs results/kfold_ambiguous_kfall_tier1/logs
LOG="results/kfold_ambiguous_kfall/logs/run_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== BUILD KFall ambiguous dataset $(date -Is) ==="
python -u scripts/build_kfall_ambiguous_bench.py --n-ambiguous 100 --n-fall 100 --n-adl 100

echo ""
echo "=== tier1_only baseline (fast) ==="
for fold in 0 1 2 3 4; do
  python -u scripts/eval_kfall_ambiguous_bench.py \
    --fold "$fold" --mode tier1_only --backend heuristic --no-force-ambiguous
done

echo ""
echo "=== gate_knn_llm NO CRITIC backend=$BACKEND ==="
for fold in 0 1 2 3 4; do
  echo "--- fold $fold ---"
  python -u scripts/eval_kfall_ambiguous_bench.py \
    --fold "$fold" --mode gate_knn_llm --backend "$BACKEND" --force-ambiguous
done

python -u scripts/aggregate_kfold_ambiguous.py --out-dir results/kfold_ambiguous_kfall --backend "$BACKEND"
python -u scripts/aggregate_kfold_ambiguous.py --out-dir results/kfold_ambiguous_kfall_tier1 --backend tier1_only

python -u - <<'PY'
import json
from pathlib import Path
import numpy as np

def load_rows(d):
    rows = []
    for f in range(5):
        m = json.loads(Path(d / f"fold{f}_eval.json").read_text())["metrics"]
        amb = (m.get("per_bucket") or {}).get("ambiguous", {})
        rows.append(dict(
            fold=f, f1=m["f1"], rec=m["recall"], fn=m["fn"], fp=m["fp"],
            cost=m["expected_response_cost"], amb=amb.get("accuracy"),
        ))
    return rows

root = Path("results/kfold_ambiguous_kfall")
tier1 = load_rows(Path("results/kfold_ambiguous_kfall_tier1"))
stack = load_rows(root)
lines = [
    "KFall ambiguous bench — gate_knn_llm vs tier1_only (NO CRITIC)",
    "Dataset: data/kfold_ambiguous_kfall (100 D18/D19 + 100 fall + 100 ADL per fold)",
    "",
    f"{'fold':>4} {'mode':<14} {'F1':>6} {'Rec':>6} {'FN':>4} {'FP':>4} {'Cost':>6} {'AmbAcc':>7}",
    "-" * 60,
]
for f in range(5):
    t, s = tier1[f], stack[f]
    lines.append(f"{f:>4} {'tier1_only':<14} {t['f1']:6.3f} {t['rec']:6.3f} {t['fn']:4.0f} {t['fp']:4.0f} {t['cost']:6.0f} {t['amb']:7.3f}")
    lines.append(f"{f:>4} {'gate_knn_llm':<14} {s['f1']:6.3f} {s['rec']:6.3f} {s['fn']:4.0f} {s['fp']:4.0f} {s['cost']:6.0f} {s['amb']:7.3f}")
    lines.append(f"     {'Δ':<14} {s['f1']-t['f1']:+6.3f} {'':>6} {s['fn']-t['fn']:+4.0f} {s['fp']-t['fp']:+4.0f} {s['cost']-t['cost']:+6.0f}")
    lines.append("")
lines.append(f"mean tier1 Cost={np.mean([r['cost'] for r in tier1]):.0f}  stack Cost={np.mean([r['cost'] for r in stack]):.0f}")
(root / "COMPARISON_NO_CRITIC.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "=== DONE $(date -Is) ==="
