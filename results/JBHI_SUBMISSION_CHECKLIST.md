# IEEE JBHI Submission Checklist — Track A Complete

**Paper:** Risk-Calibrated Dual-Process (RCDP) Agentic Fall Detection  
**Target:** IEEE Journal of Biomedical and Health Informatics (JBHI)  
**Manuscript:** `paper/main.tex` → compile to `paper/main.pdf`

## Ground Truth Lock (verify before submit)

- [ ] Ambiguous bench lead: F1 **0.713 → 0.896** (Δ **+0.183**)
- [ ] Main SisFall: F1 **0.759±0.014 → 0.887±0.029**
- [ ] KFall external: F1 **0.987 → 0.990** (Δ **+0.003**) — stability framing only
- [ ] Escalation: main **~47%**, ambiguous **~59%**, KFall **~1.1%**
- [ ] G_faith: **0.436 → 0.843** (contrastive critique)
- [ ] No claims: KFall 0.62→0.81, <5% duty cycle, conformal guarantees

## Manuscript sections (Track A)

| Week | Item | Status |
|------|------|--------|
| 1 | Ambiguous stress table (`tab_ambiguous_sisfall.tex`) | Done |
| 1 | KFall stale text fixed | Done |
| 1 | RCDP framing in abstract/intro | Done |
| 2 | `tab_stats.tex` BCa CIs + 5/5 sign consistency | Done |
| 2 | Ollama seed sweep (42,43,44) | Running (~1hr+); refresh `tab_seeds.tex` when done |
| 3 | Deployment figure (`deployment_architecture.pdf`) | Done |
| 3 | β-cost table, G_faith table, escalation table | Done |
| 4 | Negative actor-critic ablation (`tab_negative_ablation.tex`) | Done |
| 4 | KFall 36-class context (`tab_kfall_multiclass.tex`) | Done (macro-F1 0.541) |
| 4 | Limitations expanded | Done |

## Pre-submission commands

```bash
cd /DATA/tauseef_2121cs04/harshal/agentic_fall
python scripts/aggregate_folds.py --results-dir results
python scripts/generate_paper_tables.py
python scripts/generate_kfall_paper_tables.py
python scripts/plot_paper_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Files to submit

- `paper/main.pdf` (anonymized if double-blind)
- Supplementary: `results/GUIDE_BRIEF.md`, reproducibility scripts
- Optional: link to code repository

## Suggested cover letter points

1. **Clinical contribution:** asymmetric cost + graded actions under FN-heavy eldercare risk
2. **Lead empirical result:** ambiguous near-fall stress benchmark (D18/D19) where Tier-1 fails
3. **Honest reporting:** escalation duty cycles, negative actor-critic ablation, modest KFall F1 delta
4. **Grounding:** quantitative G_faith metric ties LLM rationales to IMU telemetry

## Post-submission (Track B → IMWUT)

- Split conformal gate implementation + α audit
- Sensor perturbation robustness (±15°)
- Partition fallback evaluation

Updated: Track A Week 4 (2026-09-01)
