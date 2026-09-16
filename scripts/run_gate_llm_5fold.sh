#!/usr/bin/env bash
# Homogenize gate_llm across folds 0–4 at n=1000 and merge into ablation JSONs.
# Also archive full_4000 ladders for folds 2–4 (appendix).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda}"
MAX_TEST="${MAX_TEST:-1000}"
LOG="$ROOT/results/gate_llm_5fold.log"
mkdir -p "$ROOT/results"

echo "=== gate_llm 5-fold START $(date) ===" | tee -a "$LOG"

for FOLD in 0 1 2 3 4; do
  CKPT="$ROOT/checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="$ROOT/data/memory/sisfall_fold${FOLD}.json"
  if [[ ! -f "$CKPT" ]]; then
    echo "Missing checkpoint $CKPT" | tee -a "$LOG"
    exit 1
  fi
  # Skip if gate_llm already present in homogeneous ablation
  if python - <<PY
import json
from pathlib import Path
p = Path("results/fold${FOLD}/ablation_fold${FOLD}.json")
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
rows = d.get("rows", d)
names = {r.get("name") for r in rows if isinstance(r, dict)}
raise SystemExit(0 if "gate_llm" in names else 1)
PY
  then
    echo "Fold ${FOLD}: gate_llm already present — skip" | tee -a "$LOG"
    continue
  fi

  echo "Fold ${FOLD}: running gate_llm @ max_test=${MAX_TEST}" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "$CKPT" \
    --memory "$MEM" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "results/fold${FOLD}" \
    --max-test "$MAX_TEST" \
    --modes gate_llm \
    --protocol-tag "fast_${MAX_TEST}_with_gate_llm" \
    --merge-existing \
    2>&1 | tee -a "$LOG"
done

# Appendix: optional full_4000 ladders for folds 2–4 (very expensive with LLM).
# Default SKIP — Fold0/1 archives already cover the n=4000 ladder story.
if [[ "${SKIP_FULL4000:-1}" == "1" ]]; then
  echo "Skipping full_4000 appendix (SKIP_FULL4000=1). Fold0/1 archives remain canonical." | tee -a "$LOG"
else
for FOLD in 2 3 4; do
  ARCH="results/fold${FOLD}/ablation_fold${FOLD}_full4000.json"
  if [[ -f "$ARCH" ]]; then
    echo "Fold ${FOLD}: full4000 archive exists — skip" | tee -a "$LOG"
    continue
  fi
  CKPT="$ROOT/checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
  MEM="$ROOT/data/memory/sisfall_fold${FOLD}.json"
  echo "Fold ${FOLD}: full_4000 ladder (appendix)" | tee -a "$LOG"
  python -u scripts/run_ablations.py \
    --paper-config configs/paper_protocol.yaml \
    --fold "$FOLD" \
    --checkpoint "$CKPT" \
    --memory "$MEM" \
    --device "$DEVICE" \
    --skip-baselines \
    --out-dir "results/fold${FOLD}" \
    --max-test 4000 \
    --modes tier1_only,gate_only,gate_knn,gate_llm,gate_knn_llm \
    --protocol-tag full_4000 \
    --out-suffix _full4000 \
    2>&1 | tee -a "$LOG"
done
fi
python -u scripts/aggregate_folds.py --results-dir results 2>&1 | tee -a "$LOG"
python -u scripts/generate_paper_tables.py 2>&1 | tee -a "$LOG"
echo "=== gate_llm 5-fold DONE $(date) ===" | tee -a "$LOG"
touch "$ROOT/results/GATE_LLM_5FOLD_DONE.flag"
