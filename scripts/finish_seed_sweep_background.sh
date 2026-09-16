#!/usr/bin/env bash
# Finish Ollama seed sweep (42,43,44) in background, then refresh tables + PDF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/results/seed_sweep"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/finish_background.log"

exec >>"$MAIN_LOG" 2>&1
echo "=== $(date -Is) finish_seed_sweep_background.sh started ==="

# Stop any prior sweep (keep Ollama server)
pkill -f "python3 scripts/run_seed_sweep.py" 2>/dev/null || true
sleep 2

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Seeds 42,43: skip-train if ckpt exists; 44 trains then Ollama eval
python3 scripts/run_seed_sweep.py \
  --fold 0 \
  --with-ollama \
  --seeds 42,43,44 \
  --skip-train \
  --device cuda \
  --epochs 20 \
  2>&1 | tee "$LOG_DIR/ollama_sweep_all.log"

echo "=== $(date -Is) sweep done; merging ==="
python3 scripts/merge_seed_sweep.py

echo "=== $(date -Is) regenerating tables ==="
python3 scripts/generate_paper_tables.py

echo "=== $(date -Is) compiling PDF ==="
(
  cd paper
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  bibtex main >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null
  pdflatex -interaction=nonstopmode main.tex >/dev/null
)

echo "=== $(date -Is) ALL DONE ==="
echo "  JSON:  results/seed_sweep/seed_sweep_fold0.json"
echo "  PDF:   paper/main.pdf"
echo "  Log:   $MAIN_LOG"
