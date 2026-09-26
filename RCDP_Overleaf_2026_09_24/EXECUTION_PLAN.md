# AEGIS — 8-week execution plan to a strong JBHI submission

Decisions already made, so you can start immediately. Rationale for each is in
`Q1_PLAN.md` and `ARCHITECTURE_AEGIS.md`; this file is the build order.

| Decision | Choice |
|---|---|
| Target venue | **IEEE JBHI** (backup: Information Fusion, then EAAI) |
| Core novelty | **N3** backbone-agnostic near-fall benchmark + **N1** budgeted individualised conformal risk control |
| Stretch novelty | **N2** evidence bottleneck — only if Weeks 1–5 land early |
| Detector | Stays frozen. Non-negotiable — it is what enables the 9-backbone claim |
| Name | AEGIS |

---

## 0. The target abstract — write this first

Fill the blanks as results arrive. If a number cannot be filled, that
experiment is not done. This is your planning instrument, not prose.

> Wearable inertial fall detectors report F1 above 0.95 on standard benchmarks.
> We show that this number conceals a field-wide failure: **nine architectures
> spanning 2015–2024** — CNN, LSTM, CNN-LSTM, TCN, ModernTCN, TSMixer,
> PatchTST, Transformer and CNN-LSTM-Attention — all collapse to **[0.66–0.74]**
> F1 on near-fall activities (stumbles, jumps, chair collapses) that lie between
> falls and daily living, and that this failure is invisible to the evaluation
> protocols the field uses. We introduce **AEGIS**, a post-hoc decision layer
> that wraps any frozen detector without retraining it. AEGIS pairs
> **individualised conformal risk control** — subject-stratified calibration
> that targets the false-negative rate directly under an explicit escalation
> budget — with retrieval-grounded adjudication of the deferred band and a
> cost-sensitive response policy. Under a **[10]%** escalation budget on
> leave-one-subject-out evaluation over 38 subjects, AEGIS lifts **all nine**
> backbones from **[0.71]** to **[0.87+]** near-fall F1 and cuts FN-weighted
> clinical cost by **[X]%**, while holding edge false-negative rate at or below
> the nominal level on **[38/38]** subjects, where marginal conformal
> prediction fails on **[N]**. Gains hold on elderly subjects **[Y]**, across
> **[three]** datasets, and under sensor rotation and resampling. We release
> the near-fall benchmark definition. A negative study shows that permitting an
> LLM critic to overturn positive detections raises false negatives ninefold,
> and we report escalation duty cycle and latency in full.

Everything below exists to fill those brackets.

---

## 1. Schedule

### Week 0 — fixes and Gate G1 *(2 days)*

| Task | Output |
|---|---|
| Correct D19 label ("gently jump without falling", not "sudden sit") everywhere | manuscript |
| Disclose that elderly never performed D18/D19 | Setup + Limitations |
| Make the public repo private (double-blind) | — |
| **E1**: all 9 backbones on a *standard* SisFall binary protocol — natural prevalence, no enrichment, subject-independent | `results/standard/` |

**Gate G1: do the backbones reach ≥0.95 F1 on the standard protocol?**
If yes → your detectors are competitive and the collapse story is real.
If no → stop. Fix training first; everything downstream is invalid.

---

### Week 1 — R1, the collapse *(the paper's new spine)*

Run all 9 backbones × 3 protocols: standard / stratified (0.433) / near-fall.
One script; reuses existing checkpoints.

**Deliverables**
- Table R1: 9 rows × 3 protocols
- **Figure 1 (the money shot)**: 9 backbones, paired bars, standard vs
  near-fall. Every bar drops. One glance tells the whole story.

This week alone probably doubles acceptance probability, with no new
architecture at all.

---

### Week 2 — Gate G2, the adjudicator question

Fixed escalated set, swap the adjudicator:

| Arm | Why |
|---|---|
| majority class / always-fall / random | trivial floors you currently lack |
| **fix the broken heuristic** (F1 = 0.034 is below random — almost certainly inverted logic) | credibility |
| k-NN vote alone | is retrieval sufficient? |
| **GBM + logistic regression on the 5 evidence features** | **the question every reviewer asks** |
| Qwen2.5-1.5B / 3B | small-model floor |
| Mistral-7B | current system |
| cascade (small → large on disagreement) | latency answer |

**Gate G2: does Mistral beat GBM-on-features by a clear margin?**
- Yes → the LLM is the contribution. Headline it.
- No → **reframe**: contribution is the gate + evidence representation; LLM
  becomes the explanation generator. Paper survives, framing changes. Better to
  learn this in Week 2 than from Reviewer 2.

---

### Weeks 3–4 — AEGIS T1: budgeted individualised conformal risk control

The methodological novelty. Three components:

1. **Mondrian conformal** — calibrate within subject (fallback: age-group)
   strata instead of marginally.
2. **Risk control, not coverage** — target FN *rate* ≤ α directly.
3. **Budget constraint** — minimise `R_β` subject to escalation ≤ B.

Baselines at matched coverage: softmax-response, predictive entropy,
temperature scaling, MC-dropout, 5-seed deep ensembles, marginal split
conformal, Chow's rule.

**Deliverables**
- Table R3 + **risk–coverage curves** (deferral work is expected to be
  evaluated this way; their absence is conspicuous)
- Operating points at B = 5%, 10%, 20% ← **this kills the 47% duty-cycle
  objection permanently**
- Per-subject edge-FN violation count: yours vs marginal conformal

---

### Week 5 — R2, the lift *(headline)*

All 9 backbones × AEGIS × near-fall protocol, at B = 10%.

**Deliverable**: Table R2 — every backbone recovers. Report per-backbone Δ and
mean Δ. Pair with Figure 1 so the reader sees collapse-then-recovery in two
adjacent figures.

---

### Week 6 — R4, generalisation and edge cases

| Item | Note |
|---|---|
| **LOSO, 38 subjects** | Ends the n=5 statistics problem permanently. Run for Tier-1 + gate; keep 5-fold for the full stack if LLM cost bites, and report both |
| **Elderly subgroup** | Mandatory given the D18/D19 exclusion |
| **Elderly-feasible near-fall proxy** | D05/D11 (collapse into chair) — activities elderly *did* perform. **Gate G4**: does the effect hold? |
| **Third dataset** | UP-Fall or FallAllD — pick one with *different sensor placement* |
| Rotation / resampling / β sweep | already have these; fold into the matrix |

**Deliverable**: Table R4 — 8 edge cases × {best baseline, AEGIS}.

---

### Week 7 — clinical grounding and qualitative evidence

| Task | Fallback if blocked |
|---|---|
| 3–5 clinicians rate ~50 escalated cases (rationale soundness, action appropriateness); report Fleiss' κ | If no clinician access: 2 biomechanics-literate raters + published fall-severity criteria. Weaker but not nothing |
| Failure analysis: 6–8 cases with actual rationale text, where AEGIS helped and where it hurt | — |
| Latency Pareto across model sizes | — |
| Ground β=10 in alarm-fatigue / missed-fall cost literature | — |

The rating study is, in my judgement, the difference between JBHI accept and
major revision. It converts your weakest pillar from "simulated" to
"expert-validated."

---

### Week 8 — write and submit

---

## 2. Final paper skeleton

| Section | Content | Tables / figures |
|---|---|---|
| I Introduction | Lead with **the collapse**, not the recovery | Fig. 1 (collapse) |
| II Related work | Five families; the retraining-free differentiator | `tab_related` |
| III Problem formulation | Selective risk, Prop. 1–2, **+ new Prop. 3 on budgeted risk control** | — |
| IV Near-fall benchmark | Protocol definition; released | protocol table |
| V AEGIS | T0–T4; Alg. 1–2 | Fig. 2 (arch), Fig. 3 (gate band) |
| VI Experiments | Setup, LOSO, metrics | — |
| VII-A | **The collapse** | R1 |
| VII-B | **The lift** | R2 |
| VII-C | Deferral-mechanism comparison | R3 + risk–coverage |
| VII-D | Adjudicator comparison | R5 |
| VII-E | Edge-case robustness | R4 |
| VII-F | Positioning vs published literature | R6 (protocol column) |
| VII-G | Negative study (actor–critic) | `tab_negative` |
| VII-H | Clinical rating + failure analysis | κ table + qualitative |
| VIII Discussion | Duty cycle, limitations, deployment | Fig. 4 (latency) |

Target **8–10 main tables**. Everything else goes to supplementary — you
currently have 28, and a reviewer reading 28 tables gets annoyed before they
get convinced.

---

## 3. What "strong results" means, concretely

Aim for these. If you hit them, this is a solid JBHI paper.

| Claim | Target |
|---|---|
| Standard-protocol backbones | ≥ 0.95 F1 (G1) |
| Near-fall collapse | ≤ 0.75 F1 on all 9 |
| AEGIS lift | ≥ 0.87 on all 9, **at B = 10%** |
| Duty cycle | ≤ 10% — not 47% |
| Conformal validity | edge FN ≤ α on ≥ 36/38 subjects; marginal conformal fails on more |
| Statistics | LOSO n=38, CIs excluding zero |
| Elderly subgroup | effect holds (G4) |
| Datasets | 3 |
| Clinical | κ ≥ 0.6 agreement |

The duty-cycle row matters more than it looks: moving 47% → 10% converts your
most-attacked number into a selling point.

---

## 4. Compression options

If 8 weeks is too long:

- **4-week version**: Week 0 + Week 1 + Week 2 + Week 5, keep the existing
  5-fold cost gate instead of AEGIS T1. You lose the methodological novelty but
  keep the field-wide-collapse finding, which is the strongest single result.
  Submittable to JBHI or EAAI.
- **Do not compress by dropping Week 0 or Week 2.** Week 0 contains factual
  errors; Week 2 can change what the paper claims.

---

## 5. Standing risks

1. The collapse may not reproduce on all nine backbones. Still publishable —
   framing shifts to "which inductive biases survive near-falls." Check in
   Week 1, it is cheap.
2. Per-subject conformal strata may be too small for finite-sample guarantees
   with 38 subjects. Fallback: age-group strata.
3. Simulated falls remain simulated. Name FARSEEING-style validation as future
   work; do not overclaim deployment readiness.
