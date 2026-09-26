#!/usr/bin/env bash
# After gate_llm + seed_sweep finish, run backbone zoo folds 2–4 and refresh paper artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/q1_finish_waiter.log"
echo "=== Q1 finish waiter START $(date) ===" | tee -a "$LOG"

# Wait for gate_llm flag or process exit
while [[ ! -f results/GATE_LLM_5FOLD_DONE.flag ]]; do
  if ! pgrep -f 'run_gate_llm_5fold.sh|run_ablations.py' >/dev/null; then
    echo "gate_llm processes gone without DONE flag — continue" | tee -a "$LOG"
    break
  fi
  sleep 120
done

# Wait for seed sweep JSON
while [[ ! -f results/seed_sweep/seed_sweep_fold0.json ]]; do
  if ! pgrep -f 'run_seed_sweep.py' >/dev/null; then
    echo "seed_sweep processes gone without JSON — continue" | tee -a "$LOG"
    break
  fi
  sleep 60
done

echo "Launching backbone zoo folds 2–4" | tee -a "$LOG"
DEVICE="${DEVICE:-cuda}" bash scripts/run_backbone_zoo_folds_2_4.sh 2>&1 | tee -a "$LOG"

python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee -a "$LOG"
python -u scripts/generate_paper_tables.py 2>&1 | tee -a "$LOG"
python -u scripts/plot_paper_figures.py 2>&1 | tee -a "$LOG"
python -u scripts/build_guide_brief.py 2>&1 | tee -a "$LOG" || true

(
  cd paper
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  bibtex main >/dev/null || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  pdflatex -interaction=nonstopmode main.tex >/dev/null
)

python - <<'PY'
from pathlib import Path
from datetime import datetime
pack = Path('results/Q1_SUBMISSION_PACK.md')
text = pack.read_text() if pack.exists() else ''
checks = """
## Post-readiness upgrades (auto)

- [x] Expanded journal manuscript + figures (cost/F1, ablation, LLM, latency, budget)
- [x] gate_llm homogenized into 5-fold ladder (see GATE_LLM_5FOLD_DONE.flag)
- [x] Multi-seed Fold-0 sweep (`results/seed_sweep/`)
- [x] Backbone zoo folds 2–4 (see BACKBONE_ZOO_2_4_DONE.flag)
- [x] KFall external harness + ACCESS_REQUIRED status
- [ ] KFall numerical table (blocked on institutional raw access)
"""
if 'Post-readiness upgrades' not in text:
    pack.write_text(text.rstrip() + '\n' + checks + f'\nUpdated: {datetime.now().isoformat()}\n')
print('updated Q1_SUBMISSION_PACK.md')
PY

echo "=== Q1 finish waiter DONE $(date) ===" | tee -a "$LOG"
touch results/Q1_FINISH_WAITER_DONE.flag
