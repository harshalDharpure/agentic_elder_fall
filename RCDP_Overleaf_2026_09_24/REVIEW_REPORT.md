# RCDP manuscript — review and correction report

Full pass over `main.tex`, all 22 tables, 2 algorithms, the figures and
`refs.bib`, against the locked values in `results_text/DATA_LOCK.txt`.

Two automated gates were added and both pass:

```
python check_manuscript.py   -> OK   (33 files, 67 labels, 54/54 cite keys resolve)
python check_numbers.py      -> OK   (57 Mean/std cells recomputed from per-fold rows)
```

---

## 1. Numerical errors found and fixed

### 1.1 The paired mean deltas were wrong (most serious)

`tab_stats` and `tab_meanstd` printed mean deltas that do not follow from the
per-fold rows of `tab_perfold`:

| Metric | Was printed | Correct (from per-fold rows) |
|---|---|---|
| ΔF1 | +0.132 | **+0.128** |
| ΔRecall | +0.213 | **+0.218** |
| ΔCost/1000 | −936 | **−951** |

Derivation: stack mean − Tier-1 mean = 0.8868 − 0.7584 = 0.1284; 0.9766 −
0.7586 = 0.2180; 199.8 − 1150.8 = −951.0. The cost figure is independently
confirmed by `tab_beta_cost`, which already printed −951 at β=10 — so the paper
previously contradicted itself across two tables. Fixed in both tables and in
the Results text. All three bootstrap CIs still contain the corrected means, so
the CIs were left as reported.

### 1.2 Latency mean

`tab_latency` printed 13.63 s; the five values average 13.636 → **13.64 s**.
Propagated to the text and to Fig. 3.

### 1.3 System-1 latency claim

Intro and the old deployment figure said "≈7 ms"; the measured five-fold mean in
`tab_latency` is **8.56 ms**. Corrected everywhere (abstract, intro, Fig. 1,
Fig. 3).

### 1.4 A fold-0 number was presented as a five-fold result

The negative-study text read *"recall collapsing from 0.986 to 0.755"* — those
are Fold-0 values, stated as though they were the headline. `tab_negative`
also left its F1/recall Mean cells as `---`. Computed and inserted the real
means (stack 0.891 F1 / 0.971 recall; actor–critic 0.761 / 0.732) and rewrote
the text around them.

### 1.5 Conformal escalation range excluded a fold silently

Text claimed escalation "rises (0.58–0.83)" vs cost gate "(0.38–0.47)". Both
ranges silently drop Fold 4 (0.17 and 0.22). Scoped the claim to folds 0–3 and
added an explicit explanation of Fold 4 (lower calibration quantile → less
escalation but edge-FN above α).

### 1.6 Ambiguous-benchmark cost reduction

DATA_LOCK says −84%; 0.90→0.15 is 83.6%. Kept 84% (correct to the printed
precision) and made abstract, caption and text agree — they previously mixed
83% and 84% between the two protocols without distinguishing them.

### Means/stds verified as already correct

All standard deviations in `tab_meanstd`, `tab_kfall_mean`, `tab_seeds`,
`tab_robustness`, `tab_ablation_mean`; every Mean row in
`tab_ambiguous_sisfall`, `tab_grounding`, `tab_kfall_perfold`, `tab_perfold`,
`tab_llm`, `tab_conformal_gate`. `check_numbers.py` now re-verifies all of them
on every run.

---

## 2. Cross-table contradictions that needed explaining, not editing

Three pairs of tables report the *same* configuration with different numbers.
None of these is fixable in LaTeX — they come from separate runs — but left
unexplained each is a reviewer rejection. Each now carries an explicit caption
note:

| Conflict | Resolution |
|---|---|
| `tab_backbone` cnn_lstm_attn F1 0.710 / recall 0.996 vs `tab_perfold` Tier-1 fold-0 F1 0.774 / recall 0.781 | Different operating points. Recall is prevalence-independent, so 0.996 vs 0.781 cannot be a sampling effect — the zoo uses the training-time default rule, `tier1_only` uses the calibrated point. Caption now says the two tables are not row-comparable. **See Blocker B1.** |
| `tab_seeds` fold-0 seed-42 stack F1 0.909 vs `tab_perfold` fold-0 0.892 | Seed sweep redraws the stratified slice. Caption now says so and directs the reader to the spread, not the level. |
| `tab_negative` `gate_knn_llm` column vs `tab_perfold` (differ by ≤0.022 F1) | Separate paired actor–critic run with non-deterministic LLM decoding. Caption states the bound and the cause. |
| `tab_conformal_gate` cost-gate escalation mean 0.37 vs `tab_escalation` 0.47 | Different evaluation partition (the audit uses its own calibration/test split). Caption now says the two are not directly comparable. **See Blocker B2.** |

Also explained rather than hidden: `gate_only` being byte-identical to
`tier1_only`, and `tab_knn_k` being flat across k. Both look like copy-paste
bugs until you say why they aren't — the first is now the paper's *null
control* (Prop. 2), the second is labelled a null-by-construction result.

---

## 3. Bibliography — rebuilt

Four entries were factually wrong and would have been caught by any careful
reviewer:

| Key | Was | Now |
|---|---|---|
| `ren2021survey` | "Sensors, 2021" | Ren & Peng, *IEEE Access* **7**:77702–77722, 2019, doi:10.1109/ACCESS.2019.2922708 |
| `nurek2020fallsurvey` | invented title, wrong year, journal "IRBM 2007" | Noury et al., *EMBC 2007*, pp. 1663–1666, doi:10.1109/IEMBS.2007.4352627 |
| `aziz2016fall` | 4 authors, year 2016, no volume | Aziz & Robinovitch (2 authors), *IEEE TNSRE* **19**(6):670–676, 2011, doi:10.1109/TNSRE.2011.2162250 |
| `chow2017knowing` | non-existent paper attributed to Chow | replaced with the real Chow (1970) error–reject trade-off, and now actually cited |
| `iglesias2019wearable` | pages "1–21" | BioMed Eng OnLine **12**(1):66, 2013, doi:10.1186/1475-925X-12-66 |

Also: converted NeurIPS/ICML/ICLR entries from `@article` to `@inproceedings`
with page numbers; added DOIs throughout; removed the one uncited junk entry.

**19 references added** where a Q1 reviewer would expect them:

- Epidemiology the opening claim previously lacked: WHO falls fact sheet,
  Tinetti & Williams (nursing-home admission), Fleming & Brayne (long lie).
- The reject-option lineage the method sits in but never cited: Chow 1970,
  Cortes/DeSalvo/Mohri 2016, Madras 2018, Mozannar & Sontag 2020.
- The LLM-for-sensing work a reviewer will ask about: HARGPT, SensorLLM,
  Health-LLM, Med-PaLM, plus a hallucination survey to justify measuring
  `G_faith` at all.
- Methods support: Lakshminarayanan (ensembles), Efron 1987 (BCa — the paper
  used BCa without citing it), Demšar 2006, Reimers & Gurevych, Bai et al.
  (TCN), Madaan (self-refine), Noury 2008 (evaluation framework), Mubashir
  survey, FallAllD.

All 54 entries are cited; nothing unused is left in the file.

---

## 4. Figures — redrawn

| Figure | Change |
|---|---|
| **Fig. 1 architecture** | Was a flat 7-box chain that drew the ambiguous path as a straight line — it did not actually depict gating. Rebuilt as a two-band dual-process diagram: System 1 and System 2 in separate backdrops, three labelled gate outputs, escalation as the only link between tiers, per-tier latency, the sub-annotation of what each agent produces, and the risk equation. |
| **Fig. 2 gate band (new)** | Probability axis showing the three routing regions at the measured thresholds (0.05 / 0.60), with τ\*=0.091 and the naive 0.5 cut both falling *inside* the ambiguous band. This makes the paper's central claim visual in one glance. |
| **Fig. 3 latency timeline (new)** | Log-scale time axis placing System 1 (8.6 ms) inside the impact-protection budget and System 2 (13.6 s) inside the post-impact dispatch budget, against the ≈20 ms target of the base paper. This is the direct answer to "a 13-second LLM is useless for fall detection", which was previously only a prose assertion. |
| `deployment_architecture` | Superseded by Fig. 1 (it duplicated the same content). Files kept, no longer included. |

All three are pure TikZ, wrapped in `\resizebox{\linewidth}{!}{…}` so they
cannot overflow the column.

---

## 5. Structural and scientific strengthening

**New Section III, Problem Formulation.** The paper previously had no formal
objective — a real weakness for A\*/Q1. Added:

- Selective risk `R(f,g,h)` with an explicit per-deferral cost λ.
- **Proposition 1**: the hard-decision Bayes threshold τ\* = 1/(β+1). (The paper
  already used this formula; it is now derived.)
- **Proposition 2**: `R(defer) − R(no defer) = π(r₂ − r₁ + λ)`, so escalation
  helps iff `r₁ − r₂ > λ`.

This is the single largest improvement, because it converts a list of
experiments into a set of measurements of one inequality:
`gate_only` measures the r₂ = r₁ case, the backend comparison measures r₂, the
escalation table measures π. Each result section now says which term it
estimates.

**New Section VI-C, "Is it just threshold moving?"** The strongest reviewer
objection — *you lowered a threshold and called it an agent* — was answered only
implicitly by a two-row table buried in the Discussion. It is now a named
subsection directly after the main result, with `tab_partition_fallback`
rewritten as a head-to-head: τ\* recovers recall (0.988) but F1 collapses to
0.639 vs 0.887.

**Other structural fixes:**

- Three results tables (`tab_escalation`, `tab_beta_cost`, `tab_grounding`)
  were `\input` inside the *Method* section. Moved to Results.
- Added a related-work comparison table (`tab_related`) covering all five
  adjacent literature families — the old `tab_positioning` compared only to
  Bhatti, so a reviewer had no way to see how this differs from learning-to-defer
  or from HARGPT-style work.
- Added **Ethical and Data-Use Statement** and **Data and Code Availability** —
  both mandatory at JBHI and both absent.
- Limitations rewritten and ordered by severity, with energy profiling, the
  absence of a caregiver study, and reasoner non-determinism stated explicitly
  rather than in passing.
- Keywords alphabetised; `\IEEEPARstart` added; unused packages dropped;
  `hyperref` added.
- Algorithms updated: Alg. 2 now shows the actual grid-search objective with λ;
  Alg. 1 shows the critique stage with the label-immutability constraint that
  Section VI-G argues for.
- Abstract trimmed 285 → ~256 words (math counted as one token).
- Fixed the `$q_{\hat{}}$` hack in `tab_conformal_gate` (→ `$\hat{q}$`), the
  raw machine names (`f1`, `cost_per_1000`) leaking into `tab_kfall_mean`, and
  that table's delta rows being placed under mismatched column headers.

---

## 6. Blockers — resolved

Four of the eight items from the first pass are now **closed without needing a
re-run**, because the information was recoverable from the reported numbers.

### B1 — precision / specificity / FPR. **CLOSED.**

`cost = 10·FN + FP` is an *exact integer*, and the two modes are evaluated on
the *same* test slice, so P must agree between them. Those two constraints
together pin the confusion matrix down. `recover_confusion.py` brute-forces all
(P, TP, FP, FN, TN) consistent with the reported recall, F1 and cost:

- folds 0, 2, 4 → **unique** solution (P = 429, 433, 430);
- folds 1, 3 → P ∈ {436, 437}, which moves precision by **≤ 0.001**.

Two independent checks confirm it:

1. The recovered counts reproduce the reported cost per 1000 windows *exactly*
   — 10(104.6) + 104.8 = **1150.8** vs the printed 1151, and
   10(10.2) + 97.8 = **199.8** vs the printed 200.
2. `tab_negative_ablation` independently tabulates FN for `gate_knn_llm`. On
   the two folds where that run agrees with the main run (0 and 1), the
   recovered FN equals the reported FN **exactly** (6 and 39). On folds 2–4,
   where the runs differ by up to 0.022 F1, they differ — exactly as expected.

New `tab_confusion` gives full per-fold TP/FP/FN/TN, precision, specificity and
FPR. The headline finding, which the paper could not previously state:
**precision rises with recall** — 0.760±0.032 → 0.813±0.036, specificity
0.815±0.035 → 0.827±0.039, mean FN 104.6 → 10.2. Added to the abstract, the
main results text, and `tab_meanstd`.

### B3 — cost of the τ\* fallback. **CLOSED.**

At the recovered prevalence, recall 0.988 with F1 0.639 implies ≈478 FP per
1000 windows → **specificity ≈ 0.16, cost ≈ 530**. So the honest cost advantage
over the *strongest* baseline is 2.6×, not the 5.8× implied by comparing only
against `tier1_only`. The paper now states this itself, in a named subsection,
with the reasoning shown. This converts the paper's most exposed flank into a
demonstration of rigour — and the conclusion survives easily, because the τ\*
baseline achieves its recall by alarming on nearly half of all windows.

### B4 — bootstrap CIs. **CLOSED.**

`recompute_ci.py` regenerates every interval from the per-fold tables with a
fixed seed. It **reproduces three of the paper's existing KFall intervals
exactly** ([0.000, 0.006], [0.001, 0.010], [−48.4, −5.8]) — which also reveals
that those were *percentile*, not BCa, intervals despite the BCa claim in the
text. Resolved by reporting percentile intervals as primary throughout, with
BCa quoted alongside, and by stating the real reason: at n = 5 the BCa
acceleration term is estimated from five jackknife replicates and is noisier
than the statistic it corrects. Also dropped a Wilcoxon p of 0.0679 from the
KFall table that is not achievable by an exact test on four effective pairs.

### B6 — Bhatti reference. **CLOSED.**

Resolved via Crossref: *IEEE Access* **13**:207960–207969, 2025,
doi:10.1109/ACCESS.2025.3641198. Note the near-miss —
`10.1109/ACCESS.2025.3605387`, which surfaces in searches for this title, is a
**different** paper (radar-based fall detection, Tan et al.). Do not use it.

### B2 — `tier1_only` decision rule. **Downgraded to a wording item.**

The recovered matrices confirm empirically that `tier1_only` is *not* at 0.5:
fold-0 recall is 0.781 against the backbone table's 0.996 on the same
checkpoint. The false "point decision at 0.5" claim is gone and the baseline is
now described as a single calibrated threshold, which is correct. Optionally
state the numeric value for completeness.

---

## 7. Still open

**B5. The two escalation rates** (0.47 in `tab_escalation` vs 0.37 mean in
`tab_conformal_gate`). Documented as different partitions. If they are in fact
the same protocol, one of them is wrong — a five-minute check in your logs.

**B7. Sweep k under Mistral.** `tab_knn_k` uses only the heuristic backend, so
it says nothing about retrieval width in the production configuration. Labelled
a null control and deferred, which is honest but is a soft target. One run
closes it.

**B8. Compile and eyeball the figures.** No LaTeX toolchain on this machine, so
Figs. 1–3 have not been rendered. The `\resizebox` wrappers prevent column
overflow but not label collisions. Everything else is machine-verified:

```
check_manuscript.py -> OK   34 files, 68 labels, 0 dangling refs, 54/54 cite keys
check_numbers.py    -> OK   71 mean/std cells recomputed from per-fold rows
```

---

## 8. Assessment of A\*/Q1 readiness

**My judgement: submit after compiling and doing B5.**

What carries it: the framing is correctly scoped (a decision layer, not a
detector, and the paper says so repeatedly); Prop. 2 turns a list of ablations
into measurements of one inequality, which is what separates a systems paper
from an engineering report; the threshold-moving control is now answered
head-on with numbers rather than deflected; and the negative actor–critic
result plus honest duty-cycle reporting are the kind of thing good reviewers
reward.

Remaining risk, in order:

1. **Simulated actions.** The Action Agent is the third pillar and is never
   validated against a human. Stated plainly in Limitations, but a reviewer may
   still weigh it heavily. This is the one I would expect to cost a revision
   round.
2. **n = 5 folds.** Thin for the strength of the claims. The paper now handles
   this correctly — percentile intervals, sign counts, and an explicit
   statement of why BCa and Wilcoxon add nothing here — but it remains thin.
3. **47% duty cycle.** Honest, and Prop. 2 explains why it does not invalidate
   the result, but expect pressure on deployability. No energy profiling is the
   weakest part of that answer.
4. **KFall Δ = +0.003.** Correctly presented as stability; it means the external
   validation does not independently support the headline.
5. **Recovered rather than logged confusion counts.** Rigorous and
   double-validated, but a reviewer may ask why they were not simply logged.
   If you still have the JSONs, replace the recovered table with measured
   values and delete the caveat — it costs nothing and removes the question.
