# RCDP — Confidence-Gated Multi-Agent Fall Detection (LaTeX package)

Self-contained IEEEtran manuscript for **IEEE Journal of Biomedical and Health
Informatics (JBHI)**. Backup venue: IEEE TNSRE.

## Current status

- Numerics reconciled (primary F1 0.759→0.887; cost 1151→200; LLM vs matched always-fall).
- Methods/configs/prompts packaged under `configs/` and `prompts/`.
- Author names/funding/conflicts are **placeholders** — fill
  [AUTHOR_DECLARATIONS.md](AUTHOR_DECLARATIONS.md) before submission.
- No local LaTeX compilation assumed; review the PDF in Overleaf.

Checklist: [RESULTS_RECONCILIATION.md](RESULTS_RECONCILIATION.md) ·
artifact map: [CONFIGS_MANIFEST.md](CONFIGS_MANIFEST.md)

## Folder map

| Path | Contents |
|------|----------|
| `main.tex` | Full manuscript (IEEEtran journal) |
| `refs.bib` | Bibliography |
| `tables/` | One `.tex` per table |
| `figures/` | TikZ sources + rendered plots |
| `algorithms/` | Algorithms 1–2 |
| `results_text/` | Counts CSV + DATA_LOCK |
| `configs/` | Locked YAML protocol snapshots |
| `prompts/` | LLM / actor / critic prompts + evidence template |
| `AUTHOR_DECLARATIONS.md` | Names, CRediT, funding, ethics (fill by hand) |
| `check_*.py` | Pre-flight arithmetic / structure checks |

## Pre-flight

```bash
python check_manuscript.py
python check_numbers.py
python check_reported_results.py
```

All must print `OK`.

## Compile (Overleaf)

Upload this folder, set main document to `main.tex`, compiler **pdfLaTeX**,
recompile twice after BibTeX. Check JBHI page limits and figure placement.
