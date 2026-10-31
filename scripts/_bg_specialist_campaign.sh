#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/specialist_v2_campaign_nohup.log"
exec >>"$LOG" 2>&1
echo "==== START $(date -Is) pid=$$ ===="

# Wait for any in-flight specialist ladder on GPU (avoid double-run)
while pgrep -f "scripts/run_specialist_judge_ladder.py" >/dev/null 2>&1; do
  echo "[$(date -Is)] waiting for existing specialist job..."
  sleep 30
done

for f in 0 1 2 3 4; do
  out="$ROOT/results/noleak_lock/specialist/fold${f}/specialist_fold${f}.json"
  # Re-run if missing or not v2 protocol
  need=1
  if [[ -f "$out" ]]; then
    if python -c "import json; d=json.load(open('$out')); raise SystemExit(0 if d.get('protocol')=='noleak_specialist_judge_v2' else 1)"; then
      echo "[fold $f] already v2 — skip"
      need=0
    fi
  fi
  if [[ "$need" -eq 1 ]]; then
    echo "[fold $f] running..."
    python -u scripts/run_specialist_judge_ladder.py --fold "$f" --device cuda --max-test 1000 --max-train-spec 12000
    echo "[fold $f] exit=$?"
  fi
done

python - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path("results/noleak_lock/specialist")
modes = [
  "tier1_only","specialist_tools_full","specialist_tier1_blend",
  "specialist_val_fusion","specialist_val_fusion_safe",
  "gate_specialist_judge_f1","gate_specialist_judge_cost",
  "gate_specialist_prec_override","oracle_tier1_or_specialist",
]
by = {m: [] for m in modes}
for f in range(5):
    p = root / f"fold{f}" / f"specialist_fold{f}.json"
    if not p.exists():
        print("MISSING", p); continue
    d = json.loads(p.read_text())
    row = {r["name"]: r for r in d["rows"]}
    for m in modes:
        if m in row: by[m].append(row[m])
summary = {"protocol": "noleak_specialist_judge_v2", "n_folds": 5, "modes": {}}
print("\n=== MEAN ± STD ===")
for m in modes:
    if len(by[m]) < 5:
        print(f"{m}: incomplete n={len(by[m])}"); continue
    f1 = np.array([r["f1"] for r in by[m]])
    rec = np.array([r["recall"] for r in by[m]])
    cost = np.array([r["expected_response_cost"] for r in by[m]])
    summary["modes"][m] = {
        "f1_mean": float(f1.mean()), "f1_std": float(f1.std(ddof=1)),
        "recall_mean": float(rec.mean()), "recall_std": float(rec.std(ddof=1)),
        "cost_mean": float(cost.mean()), "cost_std": float(cost.std(ddof=1)),
        "per_fold_f1": f1.tolist(),
    }
    print(f"{m:32s} F1={f1.mean():.4f}±{f1.std(ddof=1):.4f}  R={rec.mean():.4f}  cost={cost.mean():.1f}")
t1 = np.array(summary["modes"]["tier1_only"]["per_fold_f1"])
cands = [m for m in modes if m not in ("tier1_only","oracle_tier1_or_specialist") and m in summary["modes"]]
best = max(cands, key=lambda m: summary["modes"][m]["f1_mean"])
bf = np.array(summary["modes"][best]["per_fold_f1"])
summary["best_honest"] = best
summary["delta_vs_tier1"] = float(bf.mean() - t1.mean())
summary["folds_beat_tier1"] = int((bf > t1).sum())
print(f"\nBEST={best} meanF1={bf.mean():.4f} vs tier1={t1.mean():.4f} Δ={bf.mean()-t1.mean():+.4f} beats={ (bf>t1).sum()}/5")
out = root / "summary_v2.json"
out.write_text(json.dumps(summary, indent=2))
print("Wrote", out)
PY

echo "==== DONE $(date -Is) ===="
