# Submission preparation and evidence reconciliation

Updated after packaging configs/prompts into the Overleaf tree and expanding
methods in `main.tex`. Numerics were already reconciled; author identities are
still placeholders.

## Resolved (numerics — unchanged)

| Issue | Resolution |
|---|---|
| Primary ablations | Tier-1 F1 0.759 → stack 0.887; cost 1151 → 200 per 1000 |
| Precision/specificity | Classification vs timing runs separated |
| Matched LLM always-fall | Mistral cost 37.3 vs always-fall 346.7 per 1000 escalated cases |
| GBM comparison | Different cases — no superiority claim |

## Completed in this package (methods / artifacts)

| Item | Where |
|---|---|
| Journal selected | **IEEE JBHI** (`main.tex` header, README, AUTHOR_DECLARATIONS.md) |
| Training / gate / budgets | `main.tex` § Training and Agentic Protocol + Appendix |
| Prompts + evidence template | `prompts/` |
| Locked YAML configs | `configs/` + CONFIGS_MANIFEST.md |
| Result export map | Parent `RESULT_EXPORTS_INDEX.md` |
| Ethics wording | Honest secondary-data statement; no invented IRB |
| Author declaration scaffold | AUTHOR_DECLARATIONS.md + blank CRediT/funding block in `main.tex` |

## Still author-only (cannot invent)

1. Real author names, affiliations, ORCID, corresponding email
2. Funding IDs, conflicts, CRediT ticks
3. Institutional secondary-data determination
4. Immutable Ollama weight digests + host inventory at run time
5. Overleaf PDF review (figures, layout, JBHI page limit)

## Stronger claims that still need new experiments

- Matched cost-tuned GBM vs Mistral on identical cases/evidence
- Full cross-backbone wrapper evaluation
- Prospective older-adult / free-living validation

**Readiness:** numerically reconciled + methods/configs packaged; **not**
camera-ready until AUTHOR_DECLARATIONS.md is filled and the Overleaf PDF is
reviewed.
