# Overleaf upload — RCDP IEEE JBHI pack

**Zip:** `overleaf_rcdp.zip` (repo root)  
**Source folder:** `overleaf_rcdp/`

## Steps

1. Go to [overleaf.com](https://www.overleaf.com) → New Project → Upload Project.
2. Upload `overleaf_rcdp.zip`.
3. Confirm main file is `main.tex`.
4. Menu → Compiler → **pdfLaTeX**.
5. Click Recompile (run twice if bibliography needs a second pass).

## What you get

- Full rewritten manuscript with equations + algorithms
- All locked result tables under `tables/`
- Figures under `figures/`
- Plain-text cross-check dumps under `results_text/` (including `DATA_LOCK.txt`)

## Before submission

- Replace anonymous author block if the venue is not double-blind (or keep for JBHI blind).
- Skim `results_text/DATA_LOCK.txt` against abstract/results one last time.
- Do **not** claim: sub-5% SisFall duty cycle, KFall F1 breakthrough, energy profiling, conformal guarantees on the cost gate alone.

## Locked lead numbers

| Claim | Value |
|-------|-------|
| Ambiguous F1 | 0.713 → 0.896 (Δ+0.183) |
| Main F1 | 0.759±0.014 → 0.887±0.029 |
| Main cost/1000 | 1151 → 200 |
| KFall F1 | 0.987 → 0.990 (stability) |
| Seeds stack F1 | 0.905±0.005 (Ollama) |
