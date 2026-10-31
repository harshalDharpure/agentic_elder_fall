# JBHI Submission Checklist

**Guarantee:** none (peer review decides). Use this to avoid desk-reject / integrity reject.

## Before upload
- [ ] `python scripts/audit_prompt_leakage.py` passes on all submission artifacts
- [ ] Primary results come from `results/noleak_lock/` only
- [ ] Selected agentic mean F1 and Tier-1 mean F1 match Overleaf tables (±0.001)
- [ ] KFall external protocol is documented without unsupported superiority claims
- [ ] Cost model \(10\mathrm{FN}+\mathrm{FP}\) defined once; used consistently
- [ ] Subject-independent 5-fold stated; seeds locked in YAML
- [ ] Prompts in Overleaf `prompts/` match runtime
- [ ] No claim of “100% clinical readiness” or free-living elderly validation
- [ ] Abstract ≤250 words; manuscript ≤8 pages IEEE double-column (or justify overlength)
- [ ] Cover letter: novelty = agentic dual-process wearable fall response; cite JBHI LLM/agent precedent (e.g. RDguru)

## Framing that maximizes accept odds
1. Contribution = **agentic System-2 decision layer**, not “we beat every fall detector.”
2. Primary comparison = frozen Bhatti-style Tier-1 vs full stack.
3. Lead metrics = F1 **and** cost/recall/FAR on near-falls + graded action.
4. Limitations = simulated falls, young adults (SisFall), and missing event-level validation.

## Red flags (do not submit until fixed)
- Mixing results from different protocols
- Paper numbers ≠ JSON on disk
- Missing KFall or missing 5-fold

## Overnight status
- Log: `logs/jbhi_acceptance_overnight.log`
- Done flag: `results/jbhi_package/OVERNIGHT_DONE.flag`
