#!/usr/bin/env bash
# G1 fix pipeline: full-data natural-prevalence retrain → eval → optional zoo.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p /DATA/tauseef_2121cs04/tmp results/standard_g1_v2 checkpoints/standard_g1 results/standard_protocol_g1_v2
export TMPDIR=/DATA/tauseef_2121cs04/tmp TMP=/DATA/tauseef_2121cs04/tmp TEMP=/DATA/tauseef_2121cs04/tmp
DEVICE="${DEVICE:-cuda}"
LOG=results/g1_fix_pipeline.log
say(){ echo "[$(date -Is)] $*"; }

say "=== G1 FIX START ==="
say "Bug: prior retrain capped 120k with 50/50 stratification vs test prev~0.34"
say "Fix: full train, natural prevalence, no mixup"

# Phase 1: primary backbone all folds
for FOLD in 0 1 2 3 4; do
  say "train cnn_lstm_attn fold=$FOLD"
  python3 scripts/run_backbone_benchmark.py \
    --paper-config configs/standard_g1_v2.yaml \
    --fold "$FOLD" --skip-agentic --models cnn_lstm_attn --device "$DEVICE" \
    --out-dir "results/standard_g1_v2/fold${FOLD}" \
    2>&1 | tee -a "results/standard_g1_v2/train_primary_fold${FOLD}.log" || true
done

say "Eval primary G1"
python3 scripts/eval_standard_protocol.py \
  --config configs/standard_g1_v2_eval.yaml \
  --folds 0,1,2,3,4 --models cnn_lstm_attn --device "$DEVICE" \
  2>&1 | tee results/standard_protocol_g1_v2/e1_primary.log || true

# Gate check
python3 - <<'PY' || true
import json,sys
from pathlib import Path
p=Path("results/standard_protocol_g1_v2/standard_protocol_summary.json")
if not p.exists():
  print("No summary yet"); sys.exit(0)
d=json.load(open(p))
prim=next(s for s in d["summary"] if s["name"]=="cnn_lstm_attn")
print(f"PRIMARY F1={prim['f1_mean']:.4f} g1_pass={prim['g1_pass']}")
Path("results/G1_STATUS.md").write_text(
  f"# G1 Status (v2 full-data fix)\n\n"
  f"**Verdict: {'PASS' if prim['g1_pass'] else 'FAIL'}**\n\n"
  f"Primary cnn_lstm_attn Youden F1 = {prim['f1_mean']:.4f} ± {prim['f1_std']:.4f}\n"
)
if not prim["g1_pass"]:
  sys.exit(2)
PY
G1_OK=$?

if [[ "$G1_OK" -eq 0 ]]; then
  say "G1 PASS — train remaining zoo models"
  for FOLD in 0 1 2 3 4; do
    say "zoo fold=$FOLD"
    python3 scripts/run_backbone_benchmark.py \
      --paper-config configs/standard_g1_v2.yaml \
      --fold "$FOLD" --skip-agentic \
      --models cnn1d tcn modern_tcn transformer \
      --device "$DEVICE" \
      --out-dir "results/standard_g1_v2/fold${FOLD}" \
      2>&1 | tee -a "results/standard_g1_v2/train_zoo_fold${FOLD}.log" || true
  done
  python3 scripts/eval_standard_protocol.py \
    --config configs/standard_g1_v2_eval.yaml \
    --folds 0,1,2,3,4 --device "$DEVICE" \
    2>&1 | tee results/standard_protocol_g1_v2/e1_zoo.log || true
  python3 scripts/build_r1_collapse.py || true
else
  say "G1 still FAIL after full-data fix — stop before zoo/AEGIS"
fi

say "=== G1 FIX PIPELINE END ==="
