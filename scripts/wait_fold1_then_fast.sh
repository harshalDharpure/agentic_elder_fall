#!/usr/bin/env bash
# Wait for fold-1 full protocol to finish, stop the old orchestrator before fold 2,
# re-run fold0/1 agentic eval with Ollama (align with gate_knn_llm), then launch fast folds 2–4.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export PYTHONUNBUFFERED=1
DEVICE="${DEVICE:-cuda}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
PRIMARY="${PRIMARY_BACKBONE:-cnn_lstm_attn}"
OLD_PID_FILE="results/FULL_RUN.pid"
LOG="results/FULL_RUN.log"

echo "=== WAIT FOLD1 THEN FAST $(date) ===" | tee -a "$LOG"

wait_for_fold1() {
  echo "Waiting for fold1 artifacts: ablation_fold1.json + llm_compare_fold1.json ..." | tee -a "$LOG"
  while true; do
    if [[ -f results/fold1/ablation_fold1.json && -f results/fold1/llm_compare_fold1.json ]]; then
      echo "Fold1 artifacts present $(date)" | tee -a "$LOG"
      return 0
    fi
    # If orchestrator died without finishing, surface that
    if [[ -f "$OLD_PID_FILE" ]]; then
      opid="$(cat "$OLD_PID_FILE" || true)"
      if [[ -n "${opid:-}" ]] && ! ps -p "$opid" >/dev/null 2>&1; then
        if [[ ! -f results/fold1/ablation_fold1.json ]]; then
          echo "ERROR: orchestrator $opid exited before fold1 ablations finished" | tee -a "$LOG"
          exit 1
        fi
        # ablations done but llm compare maybe still needed — keep waiting a bit if children live
        if pgrep -af 'run_llm_compare.py.*--fold 1' | grep -v pgrep >/dev/null 2>&1; then
          sleep 60
          continue
        fi
        if [[ -f results/fold1/llm_compare_fold1.json ]]; then
          return 0
        fi
        if [[ -f results/fold1/ablation_fold1.json ]] && ! pgrep -af 'run_ablations.py.*--fold 1|run_agentic_eval.py.*--fold 1|run_llm_compare.py.*--fold 1' | grep -v pgrep >/dev/null 2>&1; then
          echo "WARNING: fold1 ablations done but llm_compare missing and no fold1 jobs; will try to finish step4" | tee -a "$LOG"
          return 0
        fi
      fi
    fi
    # If fold 2 already started under the slow script, break once fold1 llm exists OR fold2 appeared after fold1 ablation
    if [[ -f results/fold1/ablation_fold1.json ]] && grep -q '===== FOLD 2 =====' "$LOG" 2>/dev/null; then
      if [[ -f results/fold1/llm_compare_fold1.json ]] || grep -q '\[4/5\] DONE' "$LOG"; then
        echo "Fold2 started in log; stopping to switch to fast protocol $(date)" | tee -a "$LOG"
        return 0
      fi
    fi
    sleep 120
  done
}

stop_old_orchestrator() {
  if [[ -f "$OLD_PID_FILE" ]]; then
    opid="$(cat "$OLD_PID_FILE" || true)"
    if [[ -n "${opid:-}" ]] && ps -p "$opid" >/dev/null 2>&1; then
      echo "Stopping old orchestrator PID $opid (and children)" | tee -a "$LOG"
      # Kill process group / children running fold2+ slow path
      pkill -P "$opid" 2>/dev/null || true
      kill "$opid" 2>/dev/null || true
      sleep 5
      # Ensure fold2+ slow jobs are not left running
      pgrep -af 'run_backbone_benchmark.py --paper-config.*--fold [234]|run_ablations.py.*--fold [234]|resume_from_step4' \
        | grep -v pgrep | awk '{print $1}' | while read -r p; do
          kill "$p" 2>/dev/null || true
        done
    fi
  fi
}

finish_fold1_step4_if_needed() {
  CKPT="checkpoints/backbones/${PRIMARY}_fold1.pt"
  MEM="data/memory/sisfall_fold1.json"
  OUT="results/fold1"
  if [[ ! -f "$OUT/llm_compare_fold1.json" ]]; then
    echo "Finishing fold1 agentic+LLM compare $(date)" | tee -a "$LOG"
    if [[ ! -f "$OUT/agentic_fold1.json" ]] || ! grep -q '"backend": "ollama"' "$OUT/agentic_fold1.json" 2>/dev/null; then
      python -u scripts/run_agentic_eval.py \
        --paper-config "$PAPER" --fold 1 --checkpoint "$CKPT" --memory "$MEM" \
        --model "$PRIMARY" --backend ollama --device "$DEVICE" \
        2>&1 | tee -a "$OUT/full_agentic_eval.log"
    fi
    python -u scripts/run_llm_compare.py \
      --paper-config "$PAPER" --fold 1 --checkpoint "$CKPT" --memory "$MEM" \
      --model "$PRIMARY" --device "$DEVICE" --out-dir "$OUT" \
      2>&1 | tee -a "$OUT/full_llm_compare.log"
  fi
  # Tag fold1 as full protocol
  cat > "$OUT/protocol_fold1.json" <<EOF
{"fold": 1, "protocol_tag": "full_4000", "max_test": 4000, "modes": "all", "backbone": "zoo"}
EOF
}

rerun_agentic_ollama_fold0() {
  CKPT="checkpoints/backbones/${PRIMARY}_fold0.pt"
  MEM="data/memory/sisfall_fold0.json"
  OUT="results/fold0"
  echo "Re-running fold0 agentic_eval with ollama (align with gate_knn_llm) $(date)" | tee -a "$LOG"
  python -u scripts/run_agentic_eval.py \
    --paper-config "$PAPER" --fold 0 --checkpoint "$CKPT" --memory "$MEM" \
    --model "$PRIMARY" --backend ollama --device "$DEVICE" \
    2>&1 | tee "$OUT/full_agentic_eval_ollama.log"
  cat > "$OUT/protocol_fold0.json" <<EOF
{"fold": 0, "protocol_tag": "full_4000", "max_test": 4000, "modes": "all", "backbone": "zoo"}
EOF
}

wait_for_fold1
stop_old_orchestrator
finish_fold1_step4_if_needed
rerun_agentic_ollama_fold0

echo "Launching resume_folds_2_4_fast.sh $(date)" | tee -a "$LOG"
nohup env DEVICE="$DEVICE" PRIMARY_BACKBONE="$PRIMARY" PAPER_CONFIG="$PAPER" \
  bash scripts/resume_folds_2_4_fast.sh >> results/nohup_full_experiments.out 2>&1 &
echo $! | tee results/FULL_RUN.pid
echo "Fast orchestrator PID $(cat results/FULL_RUN.pid)" | tee -a "$LOG"
echo "=== WAIT/HANDOFF COMPLETE $(date) ===" | tee -a "$LOG"
