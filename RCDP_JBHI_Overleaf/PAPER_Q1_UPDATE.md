# Overleaf Q1 packaging update (2026-10-20)

## What changed
1. **Primary protocol:** The paper narrative reports the verified submission aggregate (`specialist_tier1_blend`: F1 `0.772 +/- 0.025`, cost `1079.6 +/- 115.3`) against `tier1_only` (F1 `0.768 +/- 0.019`, cost `1086.6 +/- 80.6`).
2. **Runtime safeguards:** The submission configuration, evidence serializer, prompt, memory identifiers, and pipeline enforce the sensor-only protocol.
3. **Submission metadata:** Institutional email is populated; ORCID is recorded as not available. Event-level metrics remain a stated limitation and should be added in a future validation study.
4. **DATA_LOCK.txt:** Added 2026-10-20 integrity note; forbids swapping in weaker overnight v2/soft_guard numbers.

## What authors must still fill
- Author names, affiliations, ORCID (`\authorcheck` in `main.tex`)
- CRediT / funding / conflicts
- Public code URL / DOI
- Suggested reviewers in `cover_letter.txt`
- Compile PDF and skim every table vs `results_text/DATA_LOCK.txt`

## Do not change before submit
- Main SisFall / KFall / ablation / stress table numbers
- Cost model \(\beta=10\)
- Freeze-label critique framing
