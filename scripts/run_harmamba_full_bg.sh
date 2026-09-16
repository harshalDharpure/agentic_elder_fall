#!/usr/bin/env bash
# Faster HARMamba full suite: n=1000 agentic protocol, skip redundant gate_llm-only.
# nohup bash scripts/run_harmamba_full_bg.sh > results/harmamba/FULL_BG.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/harmamba results/kfall/harmamba data/memory checkpoints/backbones checkpoints/kfall

DEVICE="${DEVICE:-cuda}"
FOLDS="${FOLDS:-0,1,2,3,4}"
PAPER="${PAPER_CONFIG:-configs/paper_protocol.yaml}"
MODEL=harmamba
MAX_TEST="${MAX_TEST:-1000}"          # paper agentic_max_samples style
ABLATION_MODES="${ABLATION_MODES:-tier1_only,gate_only,gate_knn,gate_knn_llm}"
export PYTHONUNBUFFERED=1

LOG="results/harmamba/FULL_BG.log"
done_flag() { touch "results/harmamba/$1.done"; }
is_done() { [[ -f "results/harmamba/$1.done" ]]; }

echo "=== HARMamba FULL BG START $(date) ===" | tee -a "$LOG"
echo "FOLDS=$FOLDS DEVICE=$DEVICE MODEL=$MODEL MAX_TEST=$MAX_TEST MODES=$ABLATION_MODES" | tee -a "$LOG"

if ! curl -s --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[ollama] starting serve…" | tee -a "$LOG"
  nohup ollama serve >> results/harmamba/ollama_serve.log 2>&1 &
  sleep 4
fi

IFS=',' read -ra FOLD_ARR <<< "$FOLDS"
for FOLD in "${FOLD_ARR[@]}"; do
  FOLD="$(echo "$FOLD" | tr -d '[:space:]')"
  mkdir -p "results/harmamba/fold${FOLD}"
  CKPT="checkpoints/backbones/${MODEL}_fold${FOLD}.pt"
  MEM="data/memory/sisfall_${MODEL}_fold${FOLD}.json"
  MEM_ALIAS="data/memory/sisfall_fold${FOLD}.json"

  echo "===== SISFALL FOLD ${FOLD} $(date) =====" | tee -a "$LOG"

  if [[ -f "$CKPT" ]] && is_done "train_fold${FOLD}"; then
    echo "[train] skip fold ${FOLD}" | tee -a "$LOG"
  else
    echo "[train] fold ${FOLD}" | tee -a "$LOG"
    python -u scripts/train_tier1.py \
      --config configs/tier1_sisfall.yaml --paper-config "$PAPER" \
      --model "$MODEL" --fold "$FOLD" --device "$DEVICE" \
      2>&1 | tee "results/harmamba/fold${FOLD}/train.log"
    done_flag "train_fold${FOLD}"
  fi

  if [[ -f "$MEM" ]] && is_done "memory_fold${FOLD}"; then
    echo "[memory] skip fold ${FOLD}" | tee -a "$LOG"
  else
    echo "[memory] fold ${FOLD}" | tee -a "$LOG"
    python -u scripts/build_memory.py \
      --checkpoint "$CKPT" --fold "$FOLD" --model "$MODEL" --device "$DEVICE" \
      2>&1 | tee "results/harmamba/fold${FOLD}/memory.log"
    [[ -f "$MEM_ALIAS" ]] && cp -f "$MEM_ALIAS" "$MEM"
    done_flag "memory_fold${FOLD}"
  fi
  [[ -f "$MEM" ]] && cp -f "$MEM" "$MEM_ALIAS"

  if is_done "ablation_fold${FOLD}"; then
    echo "[ablation] skip fold ${FOLD}" | tee -a "$LOG"
  else
    echo "[ablation] fold ${FOLD} max_test=$MAX_TEST" | tee -a "$LOG"
    python -u scripts/run_ablations.py \
      --paper-config "$PAPER" --fold "$FOLD" \
      --checkpoint "$CKPT" --memory "$MEM_ALIAS" \
      --device "$DEVICE" --skip-baselines \
      --max-test "$MAX_TEST" \
      --modes "$ABLATION_MODES" \
      --out-dir "results/harmamba/fold${FOLD}" \
      2>&1 | tee "results/harmamba/fold${FOLD}/ablation.log"
    done_flag "ablation_fold${FOLD}"
  fi

  if is_done "agentic_fold${FOLD}"; then
    echo "[agentic] skip fold ${FOLD}" | tee -a "$LOG"
  else
    echo "[agentic] fold ${FOLD}" | tee -a "$LOG"
    python -u scripts/run_agentic_eval.py \
      --paper-config "$PAPER" --fold "$FOLD" \
      --checkpoint "$CKPT" --memory "$MEM_ALIAS" \
      --model "$MODEL" --backend ollama --device "$DEVICE" \
      --max-samples "$MAX_TEST" \
      --out "results/harmamba/fold${FOLD}/agentic_fold${FOLD}.json" \
      2>&1 | tee "results/harmamba/fold${FOLD}/agentic.log"
    done_flag "agentic_fold${FOLD}"
  fi

  if is_done "llm_compare_fold${FOLD}"; then
    echo "[llm_compare] skip fold ${FOLD}" | tee -a "$LOG"
  else
    echo "[llm_compare] fold ${FOLD}" | tee -a "$LOG"
    python -u scripts/run_llm_compare.py \
      --paper-config "$PAPER" --fold "$FOLD" \
      --checkpoint "$CKPT" --memory "$MEM_ALIAS" \
      --model "$MODEL" --device "$DEVICE" \
      --max-cases 150 \
      --out-dir "results/harmamba/fold${FOLD}" \
      2>&1 | tee "results/harmamba/fold${FOLD}/llm_compare.log" || true
    done_flag "llm_compare_fold${FOLD}"
  fi
done

echo "[aggregate] SisFall" | tee -a "$LOG"
python -u - <<'PY' 2>&1 | tee results/harmamba/aggregate_summary.log
import json, statistics as st
from pathlib import Path
root = Path("results/harmamba")
rows_by_mode = {}
for fold in range(5):
    p = root / f"fold{fold}" / f"ablation_fold{fold}.json"
    # ablations may write ablation_foldN.json or variant names
    cands = list((root / f"fold{fold}").glob("ablation*.json")) if (root / f"fold{fold}").exists() else []
    data = None
    for c in cands:
        try:
            d = json.loads(c.read_text())
            r = d.get("rows", d if isinstance(d, list) else [])
            if r:
                data = r
                break
        except Exception:
            pass
    ag = root / f"fold{fold}" / f"agentic_fold{fold}.json"
    if ag.exists():
        m = json.loads(ag.read_text())
        mm = m.get("metrics", m)
        rows_by_mode.setdefault("agentic_gate_knn_llm", []).append(mm)
    if not data:
        continue
    for r in data:
        name = r.get("name")
        if name:
            rows_by_mode.setdefault(name, []).append(r)

def mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.pstdev(vals)

summary = {}
for mode, rows in rows_by_mode.items():
    f1s = [float(r["f1"]) for r in rows if r.get("f1") is not None]
    recs = [float(r["recall"]) for r in rows if r.get("recall") is not None]
    costs = [float(r["expected_response_cost"]) for r in rows if r.get("expected_response_cost") is not None]
    mf, sf = mean_std(f1s)
    mr, sr = mean_std(recs)
    mc, sc = mean_std(costs)
    summary[mode] = {
        "n_folds": len(rows),
        "f1_mean": mf, "f1_std": sf,
        "recall_mean": mr, "recall_std": sr,
        "cost_mean": mc, "cost_std": sc,
    }
    print(mode, summary[mode])
Path("results/harmamba/aggregate_mean_std.json").write_text(json.dumps(summary, indent=2))
print("Wrote results/harmamba/aggregate_mean_std.json")
PY

echo "===== KFALL $(date) =====" | tee -a "$LOG"
if [[ ! -d data/raw/kfall ]]; then
  echo "[kfall] SKIP — data/raw/kfall missing" | tee -a "$LOG"
else
  for FOLD in "${FOLD_ARR[@]}"; do
    FOLD="$(echo "$FOLD" | tr -d '[:space:]')"
    if is_done "kfall_fold${FOLD}"; then
      echo "[kfall] skip fold ${FOLD}" | tee -a "$LOG"
      continue
    fi
    MODES="tier1_only,gate_knn_llm"
    [[ "$FOLD" == "0" ]] && MODES="tier1_only,gate_only,gate_knn,gate_knn_llm"
    echo "[kfall] fold ${FOLD}" | tee -a "$LOG"
    python -u scripts/run_kfall_external.py \
      --fold "$FOLD" --device "$DEVICE" --model "$MODEL" \
      --max-test "$MAX_TEST" --modes "$MODES" \
      2>&1 | tee "results/harmamba/kfall_fold${FOLD}.log"
    done_flag "kfall_fold${FOLD}"
  done
fi

touch results/harmamba/ALL_EXPERIMENTS_DONE.flag
echo "=== HARMamba FULL BG DONE $(date) ===" | tee -a "$LOG"
