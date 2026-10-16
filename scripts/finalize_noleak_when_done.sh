#!/usr/bin/env bash
# After results/noleak/NOLEAK_SYSTEM2_DONE.flag appears: aggregate, audit prompts, note paper update.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FLAG="$ROOT/results/noleak/NOLEAK_SYSTEM2_DONE.flag"
LOG="$ROOT/logs/finalize_noleak.log"
echo "Waiting for $FLAG ..." | tee -a "$LOG"
while [[ ! -f "$FLAG" ]]; do sleep 60; done
echo "Flag seen $(date)" | tee -a "$LOG"
python -u scripts/aggregate_noleak_results.py 2>&1 | tee -a "$LOG"
python -u scripts/audit_prompt_leakage.py --prompts results/prompt_audit 2>&1 | tee -a "$LOG" || true
python -u scripts/audit_prompt_leakage.py --memory-dir data/memory 2>&1 | tee -a "$LOG"
echo "DONE finalize $(date)" | tee -a "$LOG"
echo "Next: copy no-leak System-2 means into RCDP_JBHI_Overleaf/tables/ and rebuild PDFs." | tee -a "$LOG"
