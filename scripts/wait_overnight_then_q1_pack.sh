#!/usr/bin/env bash
# Wait for overnight homogenize, then finish Q1 pack (aggregate + GUIDE_BRIEF + Q1_SUBMISSION_PACK).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/FULL_RUN.log
FLAG=results/Q1_READY.flag
STATUS=results/Q1_WAITER_STATUS.txt
OVERNIGHT_PID_FILE=results/OVERNIGHT.pid

echo "=== Q1 WAITER START $(date) ===" | tee "$STATUS" | tee -a "$LOG"

wait_overnight() {
  echo "Waiting for overnight homogenize..." | tee -a "$STATUS"
  while true; do
    if grep -q 'OVERNIGHT GUIDE STRENGTHEN DONE' "$LOG" 2>/dev/null; then
      echo "Overnight DONE marker found $(date)" | tee -a "$STATUS"
      return 0
    fi
    opid="$(cat "$OVERNIGHT_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${opid:-}" ]] && ! ps -p "$opid" >/dev/null 2>&1; then
      # process died — check success or fail
      if grep -q 'OVERNIGHT GUIDE STRENGTHEN DONE' "$LOG" 2>/dev/null; then
        return 0
      fi
      echo "ERROR: overnight PID $opid exited without DONE marker $(date)" | tee -a "$STATUS" | tee -a "$LOG"
      exit 1
    fi
    sleep 120
  done
}

verify_folds() {
  python - <<'PY'
import json, sys
from pathlib import Path
ok = True
for f in (0, 1):
    ag = json.loads(Path(f"results/fold{f}/agentic_fold{f}.json").read_text())["metrics"]
    if ag.get("backend") != "ollama":
        print(f"FAIL fold{f} agentic backend={ag.get('backend')}")
        ok = False
    else:
        print(f"OK fold{f} agentic ollama f1={ag.get('f1'):.3f}")
    d = json.loads(Path(f"results/fold{f}/ablation_fold{f}.json").read_text())
    tag = d.get("protocol_tag") if isinstance(d, dict) else None
    mt = d.get("max_test") if isinstance(d, dict) else None
    rows = d.get("rows", d) if isinstance(d, dict) else d
    g = next(r for r in rows if r.get("name") == "gate_knn_llm")
    row_mt = g.get("max_test")
    if tag != "fast_1000_no_gate_llm" and mt != 1000 and row_mt != 1000:
        # accept if max_test field on payload is 1000
        print(f"WARN fold{f} protocol_tag={tag} max_test={mt} row_mt={row_mt}")
    if (mt or row_mt) == 1000 or tag == "fast_1000_no_gate_llm":
        print(f"OK fold{f} ablation homogenized f1={g['f1']:.3f}")
    else:
        print(f"FAIL fold{f} not homogenized tag={tag} mt={mt}")
        ok = False
sys.exit(0 if ok else 1)
PY
}

write_submission_pack() {
  python - <<'PY'
import json
from pathlib import Path
import numpy as np

ROOT = Path('.')

def load_ablation(fold):
    d = json.loads((ROOT / f'results/fold{fold}/ablation_fold{fold}.json').read_text())
    rows = d.get('rows', d) if isinstance(d, dict) else d
    return {r['name']: r for r in rows if isinstance(r, dict) and 'name' in r}, d if isinstance(d, dict) else {}

def n_of(row, fold, meta):
    for k in ('n', 'max_test'):
        if isinstance(row.get(k), (int, float)) and int(row[k]) > 0:
            return int(row[k])
    if isinstance(meta.get('max_test'), (int, float)):
        return int(meta['max_test'])
    parts = [row.get(x) for x in ('tp', 'tn', 'fp', 'fn')]
    if all(isinstance(x, (int, float)) for x in parts):
        return int(sum(parts))
    return 1000

def cpk(row, fold, meta):
    n = n_of(row, fold, meta)
    return float(row['expected_response_cost']) / n * 1000.0

lines = []
lines.append('# Q1 Submission Pack — Agentic Fall Detection')
lines.append('')
lines.append(f'Generated after overnight homogenize. See also `GUIDE_BRIEF.md`.')
lines.append('')
lines.append('## Positioning vs Bhatti et al. (IEEE Access 2025)')
lines.append('')
lines.append('| | Bhatti (base) | This work |')
lines.append('|---|---|---|')
lines.append('| Role | Tier-1 CNN–LSTM–Attention detector | Same detector + agentic safety layer |')
lines.append('| Fair baseline in our tables | `tier1_only` / `cnn_lstm_attn` (same ckpt, same folds) | `gate_knn_llm` |')
lines.append('| Claim | Detection accuracy | Recall↑ + normalized response cost↓ under escalation |')
lines.append('')
lines.append('**Do not** claim beating Bhatti’s published table under their exact protocol; claim improvement vs **our** locked Tier-1 under subject-independent 5-fold SisFall.')
lines.append('')
lines.append('## Table — Per-fold Tier-1 vs gate_knn_llm (homogeneous main protocol)')
lines.append('')
lines.append('| Fold | n | Tier1 F1 | Tier1 Rec | Tier1 cost/1k | Stack F1 | Stack Rec | Stack cost/1k | ΔF1 |')
lines.append('|------|---|----------|-----------|---------------|----------|-----------|---------------|-----|')
f1t,f1s,rt,rs,ct,cs = [],[],[],[],[],[]
for fold in range(5):
    m, meta = load_ablation(fold)
    t, s = m['tier1_only'], m['gate_knn_llm']
    n = n_of(s, fold, meta)
    a, b = cpk(t, fold, meta), cpk(s, fold, meta)
    f1t.append(t['f1']); f1s.append(s['f1']); rt.append(t['recall']); rs.append(s['recall']); ct.append(a); cs.append(b)
    lines.append(f"| {fold} | {n} | {t['f1']:.3f} | {t['recall']:.3f} | {a:.0f} | {s['f1']:.3f} | {s['recall']:.3f} | {b:.0f} | {s['f1']-t['f1']:+.3f} |")
lines.append('')
lines.append('## 5-fold mean ± std')
lines.append('')
lines.append(f"| tier1_only | F1 {np.mean(f1t):.3f}±{np.std(f1t,ddof=1):.3f} | Rec {np.mean(rt):.3f}±{np.std(rt,ddof=1):.3f} | cost/1k {np.mean(ct):.0f}±{np.std(ct,ddof=1):.0f} |")
lines.append(f"| **gate_knn_llm** | **F1 {np.mean(f1s):.3f}±{np.std(f1s,ddof=1):.3f}** | **Rec {np.mean(rs):.3f}±{np.std(rs,ddof=1):.3f}** | **cost/1k {np.mean(cs):.0f}±{np.std(cs,ddof=1):.0f}** |")
lines.append('')
tests = json.loads((ROOT/'results/aggregate_ablation_tests.json').read_text()) if (ROOT/'results/aggregate_ablation_tests.json').exists() else []
lines.append('## Paired tests (gate_knn_llm − tier1_only)')
lines.append('')
for t in tests:
    if t.get('metric') in ('f1','recall','cost_per_1000'):
        lines.append(f"- **{t['metric']}**: mean Δ={t['mean_delta']:.4f}, bootstrap 95% CI={t.get('bootstrap_ci95')}, Wilcoxon p={t.get('wilcoxon_pvalue')}")
lines.append('')
lines.append('## LLM compare (ambiguous cases)')
lines.append('')
hf, mf, qf = [], [], []
for fold in range(5):
    rows = json.loads((ROOT/f'results/fold{fold}/llm_compare_fold{fold}.json').read_text())
    if isinstance(rows, dict):
        rows = rows.get('rows', rows)
    for r in rows:
        if r.get('name')=='heuristic': hf.append(r['f1'])
        if 'mistral' in str(r.get('name','')): mf.append(r['f1'])
        if 'qwen' in str(r.get('name','')): qf.append(r['f1'])
lines.append(f"- Heuristic mean F1={np.mean(hf):.3f} · **Mistral={np.mean(mf):.3f}** · Qwen={np.mean(qf):.3f}")
lines.append('')
lines.append('## Q1 draft checklist')
lines.append('')
lines.append('- [x] Subject-independent 5-fold SisFall')
lines.append('- [x] Same Tier-1 ckpt for fair baseline')
lines.append('- [x] Ablation ladder + LLM backend compare')
lines.append('- [x] Normalized cost_per_1000')
lines.append('- [x] Fold0/1 homogenized to max_test=1000 (full4000 archived)')
lines.append('- [x] End-to-end agentic uses ollama for full stack')
lines.append('- [ ] Paper writing (Methods/Results from this pack)')
lines.append('- [ ] KFall (optional, when raw data available)')
lines.append('')
lines.append('## Reproducibility')
lines.append('')
lines.append('```bash')
lines.append('python scripts/aggregate_folds.py --results-dir results')
lines.append('python scripts/export_tables.py')
lines.append('python scripts/build_guide_brief.py')
lines.append('```')
lines.append('')
(ROOT/'results/Q1_SUBMISSION_PACK.md').write_text('\n'.join(lines)+'\n')
print('Wrote results/Q1_SUBMISSION_PACK.md')
PY
}

wait_overnight
echo "Verifying folds..." | tee -a "$STATUS"
verify_folds | tee -a "$STATUS"

echo "Regenerating aggregates..." | tee -a "$STATUS"
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee -a results/full_aggregate.log | tee -a "$STATUS"
python -u scripts/export_tables.py 2>&1 | tee -a results/full_export.log | tee -a "$STATUS"
python -u scripts/build_guide_brief.py 2>&1 | tee -a "$STATUS"
write_submission_pack | tee -a "$STATUS"

{
  echo "Q1 evidence pack READY $(date)"
  echo "Open: results/GUIDE_BRIEF.md"
  echo "Open: results/Q1_SUBMISSION_PACK.md"
} | tee "$FLAG" | tee -a "$STATUS" | tee -a "$LOG"

echo "=== Q1 WAITER DONE $(date) ===" | tee -a "$STATUS" | tee -a "$LOG"
