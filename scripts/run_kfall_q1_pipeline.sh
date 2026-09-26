#!/usr/bin/env bash
# Single-instance KFall Q1 binary agentic pipeline (flock-guarded).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOCK="$ROOT/results/kfall/.pipeline.lock"
LOG="$ROOT/results/kfall/kfall_q1_pipeline.log"
mkdir -p "$ROOT/results/kfall"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another KFall pipeline holds $LOCK — exit"
  exit 0
fi

DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
echo "=== KFall Q1 pipeline START $(date) ===" | tee "$LOG"

# Fold 0: full ablation ladder (needs Ollama for gate_llm / gate_knn_llm)
F=0
CKPT="checkpoints/kfall/cnn_lstm_attn_binary_fold${F}.pt"
SKIP=()
if [[ -f "$CKPT" ]]; then SKIP=(--skip-train); fi
OUT="results/kfall/fold${F}/kfall_external_fold${F}.json"
if [[ -f "$OUT" ]] && python - <<PY
import json
from pathlib import Path
d=json.loads(Path("$OUT").read_text())
names={r.get("name") for r in d.get("rows",[])}
need={"tier1_only","gate_only","gate_knn","gate_llm","gate_knn_llm"}
raise SystemExit(0 if need.issubset(names) else 1)
PY
then
  echo "Fold 0 complete — skip" | tee -a "$LOG"
else
  echo "=== Fold 0 full ladder $(date) ===" | tee -a "$LOG"
  python -u scripts/run_kfall_external.py \
    --fold 0 --device "$DEVICE" --max-test "$MAX_TEST" \
    --modes tier1_only,gate_only,gate_knn,gate_llm,gate_knn_llm \
    "${SKIP[@]}" 2>&1 | tee -a "$LOG"
fi

# Folds 1–4: primary comparison only
for F in 1 2 3 4; do
  OUT="results/kfall/fold${F}/kfall_external_fold${F}.json"
  if [[ -f "$OUT" ]] && python - <<PY
import json
from pathlib import Path
d=json.loads(Path("$OUT").read_text())
names={r.get("name") for r in d.get("rows",[])}
raise SystemExit(0 if {"tier1_only","gate_knn_llm"}.issubset(names) else 1)
PY
  then
    echo "Fold $F complete — skip" | tee -a "$LOG"
    continue
  fi
  CKPT="checkpoints/kfall/cnn_lstm_attn_binary_fold${F}.pt"
  SKIP=()
  if [[ -f "$CKPT" ]]; then SKIP=(--skip-train); fi
  echo "=== Fold $F primary $(date) ===" | tee -a "$LOG"
  python -u scripts/run_kfall_external.py \
    --fold "$F" --device "$DEVICE" --max-test "$MAX_TEST" \
    --modes tier1_only,gate_knn_llm \
    "${SKIP[@]}" 2>&1 | tee -a "$LOG"
done

python -u scripts/aggregate_kfall.py 2>&1 | tee -a "$LOG"
echo "=== KFall Q1 pipeline DONE $(date) ===" | tee -a "$LOG"
touch results/kfall/KFALL_EXTERNAL_DONE.flag
