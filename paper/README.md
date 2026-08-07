# IEEE Transactions paper draft

Complete LaTeX manuscript for the agentic fall-detection work.

## Files

| Path | Description |
|------|-------------|
| `main.tex` | Full IEEEtran journal paper |
| `refs.bib` | Bibliography |
| `table/*.tex` | Generated result tables (one file per table) |
| `fig/architecture.tex` | TikZ pipeline figure |

## Regenerate tables from results

```bash
# from repo root
python scripts/generate_paper_tables.py
```

## Compile

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a TeX distribution with `IEEEtran`, `booktabs`, `tikz`, and `biblatex`/`bibtex` (standard TeX Live / MiKTeX).

## Author / venue placeholders

Edit `\author{...}` in `main.tex` before submission. The draft uses honest claims vs Bhatti *et al.* (fair Tier-1 baseline under our SisFall protocol).
