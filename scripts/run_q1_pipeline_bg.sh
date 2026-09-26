#!/usr/bin/env bash
# Master Q1 experiment pipeline (background).
# Chains: wait E1 → document G1 → retrain standard → re-eval G1 → R1 table → G2 adjudicator
# Usage: nohup bash scripts/run_q1_pipeline_bg.sh >> results/q1_pipeline.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/standard_protocol results/standard_retrain results/adjudicator_compare results/r1_collapse logs
LOG="${LOG:-results/q1_pipeline.log}"
DEVICE="${DEVICE:-cuda}"
E1_PID="${E1_PID:-}"

ts() { date -Is; }
say() { echo "[$(ts)] $*"; }

say "=== Q1 pipeline start ==="
say "ROOT=$ROOT DEVICE=$DEVICE"

# --- 0) Wait for in-flight E1 if provided or auto-detect ---
if [[ -z "$E1_PID" ]]; then
  E1_PID=$(pgrep -f 'eval_standard_protocol.py --folds' | head -1 || true)
fi
if [[ -n "${E1_PID}" ]]; then
  say "Waiting for E1 PID $E1_PID ..."
  while kill -0 "$E1_PID" 2>/dev/null; do sleep 30; done
  say "E1 finished (PID $E1_PID)"
else
  say "No E1 PID found — running E1 now"
  python3 scripts/eval_standard_protocol.py --folds 0,1,2,3,4 --device "$DEVICE" \
    2>&1 | tee -a results/standard_protocol/e1_all_folds.log || true
fi

# --- 1) Summarize G1 from existing E1 ---
say "Building R1 collapse stub from current E1 + existing tables"
python3 scripts/build_r1_collapse.py 2>&1 | tee -a "$LOG" || true

# --- 2) G1 recovery retrain (primary first, then zoo) ---
say "Retrain standard protocol: cnn_lstm_attn all folds"
for FOLD in 0 1 2 3 4; do
  say "retrain cnn_lstm_attn fold=$FOLD"
  python3 scripts/run_backbone_benchmark.py \
    --paper-config configs/standard_retrain.yaml \
    --fold "$FOLD" --skip-agentic --models cnn_lstm_attn --device "$DEVICE" \
    --out-dir "results/standard_retrain/fold${FOLD}" \
    2>&1 | tee -a "results/standard_retrain/train_fold${FOLD}.log" || true
done

say "Retrain remaining zoo models fold0..4 (skip-agentic)"
for FOLD in 0 1 2 3 4; do
  say "retrain zoo fold=$FOLD"
  python3 scripts/run_backbone_benchmark.py \
    --paper-config configs/standard_retrain.yaml \
    --fold "$FOLD" --skip-agentic \
    --models cnn1d lstm cnn_lstm tcn modern_tcn tsmixer patchtst transformer \
    --device "$DEVICE" \
    --out-dir "results/standard_retrain/fold${FOLD}" \
    2>&1 | tee -a "results/standard_retrain/train_zoo_fold${FOLD}.log" || true
done

# Point standard eval at new checkpoints (pre-written config)
say "Re-eval G1 on checkpoints/standard"
mkdir -p results/standard_protocol_retrained
python3 scripts/eval_standard_protocol.py \
  --config configs/standard_sisfall_retrained.yaml \
  --folds 0,1,2,3,4 --device "$DEVICE" \
  2>&1 | tee results/standard_protocol_retrained/e1_retrain.log || true

python3 scripts/build_r1_collapse.py 2>&1 | tee -a "$LOG" || true

# --- 3) G2 adjudicator (features + GBM; LLM if ollama up) ---
say "G2 adjudicator compare fold0"
LLM_ARGS=()
if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | grep -qi mistral; then
  LLM_ARGS=(--llm-backends mistral:latest --llm-max 60)
  say "Ollama Mistral detected — including LLM arm"
else
  say "No Ollama Mistral — G2 runs sklearn + heuristic only"
fi
python3 scripts/run_adjudicator_compare.py --fold 0 --device "$DEVICE" --max-cases 400 \
  "${LLM_ARGS[@]}" \
  2>&1 | tee results/adjudicator_compare/fold0.log || true

# optional folds 1-4 without LLM for speed
for FOLD in 1 2 3 4; do
  say "G2 adjudicator fold=$FOLD (no LLM)"
  python3 scripts/run_adjudicator_compare.py --fold "$FOLD" --device "$DEVICE" --max-cases 300 \
    2>&1 | tee "results/adjudicator_compare/fold${FOLD}.log" || true
done

# --- 4) Ambiguous-bench detector-only sweep for R1 near-fall column (all backbones) ---
say "Near-fall detector-only eval (ambiguous slice) for missing R1 cells — if script exists"
if [[ -f scripts/eval_kfold_ambiguous_bench.py ]]; then
  for FOLD in 0 1 2 3 4; do
    CKPT="checkpoints/backbones/cnn_lstm_attn_fold${FOLD}.pt"
    [[ -f "$CKPT" ]] || continue
    python3 scripts/eval_kfold_ambiguous_bench.py --fold "$FOLD" --checkpoint "$CKPT" --device "$DEVICE" \
      2>&1 | tee -a results/kfold_ambiguous/r1_refresh.log || true
  done
fi

say "Manuscript check"
python3 scripts/check_manuscript.py 2>&1 | tee -a "$LOG" || true

say "=== Q1 pipeline complete ==="
say "Artifacts: results/standard_protocol* results/standard_retrain results/adjudicator_compare results/r1_collapse"
