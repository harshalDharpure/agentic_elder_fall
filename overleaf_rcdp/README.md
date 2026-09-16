# RCDP Overleaf Package (IEEE JBHI journal)

Self-contained LaTeX project for Overleaf upload.

## Folder map

| Path | Contents |
|------|----------|
| `main.tex` | Full manuscript (IEEEtran journal) |
| `refs.bib` | Bibliography |
| `tables/` | All result tables (`tab_*.tex`) |
| `figures/` | Architecture TikZ + PDF figures |
| `algorithms/` | Algorithm 1–2 |
| `results_text/` | Plain-text mirrors of every table + `DATA_LOCK.txt` |
| `IEEEtran.cls` / `IEEEtran.bst` | Local copies (Overleaf also provides these) |

## Compile (Overleaf)

1. New Project → Upload Project → select `overleaf_rcdp.zip`
2. Set main document to `main.tex`
3. Compiler: **pdfLaTeX** (BibTeX)
4. Recompile twice after first BibTeX pass if refs show `?`

## Compile (local)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Ground-truth numbers

See `results_text/DATA_LOCK.txt`. Do not invent metrics.

## Authors

Currently anonymous for double-blind review. Edit the `\author{...}` block in `main.tex` for camera-ready.
