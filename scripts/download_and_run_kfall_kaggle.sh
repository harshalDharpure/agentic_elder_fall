#!/usr/bin/env bash
# Download KFall from Kaggle and run binary external validation.
# Requires: ~/.kaggle/kaggle.json (Kaggle → Settings → Create New Token)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/kfall/kfall_kaggle_run.log"
mkdir -p "$ROOT/results/kfall" "$ROOT/data/raw/kfall_download" "$ROOT/data/raw/kfall"

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing $HOME/.kaggle/kaggle.json"
  echo "Create token at https://www.kaggle.com/settings and place kaggle.json there (chmod 600)."
  exit 2
fi
chmod 600 "$HOME/.kaggle/kaggle.json" || true

echo "=== KFall Kaggle download START $(date) ===" | tee "$LOG"
kaggle datasets download -d usmanabbasi2002/kfall-dataset \
  -p data/raw/kfall_download --unzip 2>&1 | tee -a "$LOG"

# Normalize layout into data/raw/kfall/ (expect SA**/CSVs)
python - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
import shutil
root = Path('data/raw/kfall_download')
dst = Path('data/raw/kfall')
dst.mkdir(parents=True, exist_ok=True)
# find subject folders or csv trees
csv = list(root.rglob('*.csv'))
print(f'found {len(csv)} csv under download')
# if already has SA* folders, symlink/copy tree
sa = [p for p in root.rglob('SA*') if p.is_dir()]
if sa:
    for p in sa:
        target = dst / p.name
        if target.exists():
            continue
        if p.parent.resolve() == dst.resolve():
            continue
        shutil.copytree(p, target, dirs_exist_ok=True)
        print('copied', p, '->', target)
else:
    # flat / nested: group by parent folder name
    for c in csv:
        subj = c.parent.name
        out = dst / subj
        out.mkdir(parents=True, exist_ok=True)
        t = out / c.name
        if not t.exists():
            shutil.copy2(c, t)
# labels xlsx
for x in root.rglob('*.xlsx'):
    lab = dst / 'label'
    lab.mkdir(exist_ok=True)
    shutil.copy2(x, lab / x.name)
    print('label', x.name)
print('raw kfall subjects', len([p for p in dst.iterdir() if p.is_dir() and p.name.startswith('SA')]))
print('sample', list(dst.iterdir())[:10])
PY

echo "=== prepare ===" | tee -a "$LOG"
python -u scripts/prepare_kfall.py --config configs/tier1_kfall_binary.yaml --phase transition 2>&1 | tee -a "$LOG"

echo "=== external eval fold 0 ===" | tee -a "$LOG"
python -u scripts/run_kfall_external.py --fold 0 --device cuda --modes tier1_only,gate_knn_llm --max-test 1000 2>&1 | tee -a "$LOG"

# optional folds 1-4 if fold0 succeeds
for F in 1 2 3 4; do
  echo "=== external eval fold $F ===" | tee -a "$LOG"
  python -u scripts/run_kfall_external.py --fold "$F" --device cuda --modes tier1_only,gate_knn_llm --max-test 1000 2>&1 | tee -a "$LOG" || true
done

echo "=== KFall Kaggle run DONE $(date) ===" | tee -a "$LOG"
touch results/kfall/KFALL_KAGGLE_DONE.flag
