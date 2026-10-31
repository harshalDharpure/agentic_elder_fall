# Strong results strategy (JBHI package)

## Positioning
- **Bhatti** = frozen CNN–LSTM–Attention detector only (`tier1_only`).
- **Ours** = first **agentic** fall-response stack: confidence gate → biomechanics tools → case memory → local LLM reasoner → graded action (+ optional critique).

## Locked numbers (auto-updated by overnight runner)
| Track | SisFall F1 | Notes |
|-------|------------|-------|
| Bhatti-matched Tier-1 | 0.767783174348482 | dense windows n=1000 |
| Selected agentic | **0.8865787459634431** (Q1_historical) | activity ON; pre soft-guard |
| Literature trial-level Tier-1 | 0.9150190774047271 | impact/trial protocol |

Beats Bhatti on F1: **True**
Hit ≥0.89: **False** | Near (≥0.88): **True**

## Integrity
- Activity-ON results are trial-code-conditioned System-2 — disclose in paper.
- Noleak: `results/noleak_lock/`
- Package: `JBHI_ACCEPTANCE_PACKAGE.md` + `results/jbhi_package/DECISION_MEMO.json`

## Absolute guarantee
**None.** JBHI peer review decides. This package maximizes readiness.
