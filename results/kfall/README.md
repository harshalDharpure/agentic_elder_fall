# KFall External Validation — Access Status

**Status:** `ACCESS_REQUIRED` (raw data not present under `data/raw/kfall/`)

KFall is institution-gated: https://sites.google.com/view/kfalldataset

## What is ready in this repo

| Artifact | Path |
|----------|------|
| Binary config | `configs/tier1_kfall_binary.yaml` |
| Multi-class config | `configs/tier1_kfall.yaml` |
| Prepare script | `scripts/prepare_kfall.py` |
| External agentic eval | `scripts/run_kfall_external.py` |
| Access probe result | `results/kfall/ACCESS_REQUIRED.json` |

## After access is granted

```bash
# extract under data/raw/kfall/SAXX/*.csv
python scripts/prepare_kfall.py --config configs/tier1_kfall_binary.yaml --phase transition
python scripts/run_kfall_external.py --fold 0 --device cuda --modes tier1_only,gate_knn_llm
# optional: folds 1–4
for F in 1 2 3 4; do
  python scripts/run_kfall_external.py --fold $F --device cuda --modes tier1_only,gate_knn_llm
done
python scripts/generate_paper_tables.py
```

Paper positioning remains: SisFall is the primary numerical claim; KFall is external validation once data arrive.
