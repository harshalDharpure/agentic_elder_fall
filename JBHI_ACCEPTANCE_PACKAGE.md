# JBHI Submission Package

## Primary result

The submission uses the verified sensor-only SisFall protocol:

- Frozen detector: F1 `0.768 +/- 0.019`, cost `1086.6 +/- 80.6`
- Selective decision layer: F1 `0.772 +/- 0.025`, cost `1079.6 +/- 115.3`

These are descriptive five-fold aggregates. The paper makes no claim of
definitive superiority and identifies event-level and prospective validation as
required follow-up work.

## Package contents

- `RCDP_JBHI_Overleaf/main.tex`
- `RCDP_JBHI_Overleaf/supplementary.tex`
- `RCDP_JBHI_Overleaf/cover_letter.txt`
- `RCDP_JBHI_Overleaf/configs/`
- `RCDP_JBHI_Overleaf/prompts/`
- `results/noleak_lock/`

## Required checks

- Subject-independent folds are used.
- Calibration uses training subjects only.
- Retrieval uses training-subject cases only.
- Case identifiers are redacted.
- Actions are simulated.
- Event-level metrics and prospective validation remain future work.
