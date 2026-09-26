# EXECUTION_PLAN — AEGIS → IEEE JBHI (Q1 only)

**Target:** IEEE JBHI (Q1). A* / IMWUT out of scope.  
**System:** AEGIS — Adaptive Evidence-Grounded Individualised Selection.  
**Detector:** frozen (enables 9-backbone claim). Novelty: **N3** near-fall benchmark + **N1** budgeted Mondrian FN-rate risk control.

See also: [ARCHITECTURE_AEGIS.md](ARCHITECTURE_AEGIS.md).

## Status

| Item | Status |
|------|--------|
| D19 + elderly disclosure | Done |
| G1 (old + first retrain) | FAIL @ F1≈0.74 |
| **G1 fix v2** | **RUNNING** — full train (~1.07M), natural prev, no 50/50 cap, no mixup |
| G2 (GBM-only) | Done (folds 0–4) |
| AEGIS T1+ | Blocked until G1 PASS |

**G1 bug found:** `max_train=120000` + default `stratified=True` forced **50/50** train labels while test prevalence is **~0.34**.

```bash
tail -f results/g1_fix_pipeline.log
```

## Target abstract (fill brackets after experiments)

> Wearable inertial fall detectors report F1 above 0.95 on standard benchmarks. We show this conceals a field-wide failure: nine architectures spanning 2015–2024 all collapse to **[0.66–0.74]** F1 on near-fall activities. We introduce **AEGIS**, a post-hoc decision layer wrapping any frozen detector without retraining. Under a **[10]%** escalation budget on leave-one-subject-out evaluation over 38 subjects, AEGIS lifts all nine backbones from **[0.71]** to **[0.87+]**, cuts FN-weighted cost by **[X]%**, and holds edge false-negative rate at or below nominal on **[N/38]** subjects where marginal conformal fails on **[M]**.

**Q1 accept bar:** collapse ≤0.75 on all 9 · lift ≥0.87 on all 9 at B=10% · duty cycle ≤10% · elderly subgroup + D05/D11 proxy · simulated Action Agent in Limitations.

## Week schedule

| Week | Work | Gate |
|------|------|------|
| 0 | D19 + elderly MS fixes; E1 standard protocol (9 backbones) | **G1:** ≥0.95 F1 |
| 1 | R1 collapse: 9 × {standard, stratified, near-fall} | — |
| 2 | G2 adjudicator suite (trivial, heuristic, kNN, GBM, LLMs) | **G2:** LLM vs GBM |
| 3–4 | AEGIS T1: Mondrian CRC + FN-rate + B≤10%; risk–coverage | — |
| 5 | R2 lift: all 9 + AEGIS at B=10% | — |
| 6 | LOSO + elderly + D05/D11 (no third dataset required) | **G4** |
| 7–8 | Rename AEGIS, cut tables, Overleaf, submit JBHI | — |

## Commands

```bash
# Week 0 — G1 (eval existing checkpoints, natural prevalence)
python scripts/eval_standard_protocol.py --folds 0,1,2,3,4 --device cuda

# Manuscript guards (after TeX edits)
python scripts/check_manuscript.py
```

## Rules

- Do **not** start Mondrian CRC / LLM sweeps until G1 passes.
- Do **not** freeze abstract wording until G2 returns.
- If G2 fails (GBM ≈ Mistral): reframe T3 as GBM + LLM explanation; architecture unchanged.
- Out of scope for Q1: clinician study, N2 bottleneck, third dataset, A*.
