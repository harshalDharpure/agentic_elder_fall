# AEGIS — proposed architecture and paper strategy

**AEGIS**: **A**daptive **E**vidence-**G**rounded **I**ndividualised **S**election
for wearable fall detection.

The name is a shield, which is exactly what the system is: it does not replace
the detector, it protects the decision the detector makes. The backronym maps
one-to-one onto the four contributions, so it is not decoration.

---

## 1. Strategic reframe: how to actually "beat everyone"

You cannot win on SisFall binary accuracy. The literature is at 0.97–0.99 and
the remaining headroom is noise. Any paper claiming 0.99 → 0.995 reads as
incremental and is rejected on novelty.

You win by **changing what is being measured**, and then beating everyone on
the new axis using *their own architectures*:

> We evaluate nine architectures spanning 2015–2024 under one protocol. All
> exceed 0.95 F1 on the standard SisFall benchmark. All collapse to 0.66–0.74
> on near-fall activities. AEGIS lifts every one of them above 0.87 **without
> retraining any of them**, at a bounded escalation budget.

That claim is:

- **honest** — you are not claiming a better detector,
- **general** — it holds across nine backbones, not one,
- **unfalsifiable-by-a-better-backbone** — a reviewer cannot say "try ModernTCN
  instead", because you already did,
- **and you can mostly run it today** with the zoo in `tab_backbone`.

This is the single highest-value structural change to the paper. It converts
"we improved our own model" into "we fixed a field-wide failure mode."

---

## 2. What is actually novel (three claims, not six)

Reviewers reject kitchen-sink architectures. Rest the novelty on exactly three
things and demote everything else to ablated engineering.

### N1. Subject-conditional conformal risk control with a duty-cycle budget

This is the methodological core, and it comes directly out of *your own
observed failure*. Your fold 4 violated its nominal level (edge FN 0.087 vs
α = 0.05) because split conformal guarantees **marginal** coverage under
exchangeability, and subject-independent splits break exchangeability.

Fix it properly:

1. **Mondrian (subject-stratified) conformal** — calibrate within strata
   (subject, or age group) rather than marginally.
2. **Risk control rather than coverage** — target the false-negative *rate*
   directly, which is the clinically meaningful quantity, instead of set
   coverage. (Conformal Risk Control, Angelopoulos et al.)
3. **Budgeted selection** — add an explicit constraint that escalation
   ≤ B (e.g. 10%), and solve for the threshold pair that minimises
   `R_β` subject to that. This kills the 47%-duty-cycle objection dead.

Nobody has done conformal risk control under a duty-cycle budget in wearable
fall detection. That is a real, citable, theoretically-grounded contribution,
and Prop. 2 in your current draft already gives you the risk decomposition to
state it cleanly.

### N2. Evidence bottleneck — faithfulness by construction

Today you *measure* rationale grounding post-hoc (`G_faith` = 0.436 → 0.843).
That is a metric, not a guarantee, and a reviewer will note that 0.843 still
means one rationale in six is ungrounded.

Instead, make the biomechanical evidence an explicit **concept bottleneck**:
the model predicts K interpretable concepts (free-fall duration, impact peak,
post-impact stillness, tilt change, peak angular rate), each with its own
confidence, and **every downstream decision depends only on those concepts**.
The reasoner physically cannot cite evidence that is not there.

This converts explainability from a measured property into a structural one,
and it connects you to the concept-bottleneck literature, which reviewers
respect. It also makes the system auditable in a way a clinician can actually
check.

### N3. Backbone-agnostic validation on a near-fall benchmark

Section 1. Nine architectures, one protocol, near-fall stress, cost-weighted.
Release the benchmark definition (activity codes, splits, cost model, budget)
so others can run it.

Defining the benchmark is itself a contribution and is how you get cited.

---

## 3. The architecture

```
                 ┌──────────────── AEGIS ────────────────┐
  IMU window ──► │                                        │
                 │  T0  Frozen detector f_θ  (ANY backbone)│
                 │            │ p, embedding               │
                 │            ▼                            │
                 │  T1  Individualised Risk Gate           │
                 │      · Mondrian conformal, per-subject  │
                 │      · targets FN rate, not coverage    │
                 │      · online recalibration from the    │
                 │        wearer's own unlabelled stream   │
                 │      · escalation budget B              │
                 │       ├── confident ──────────────────► action
                 │       └── ambiguous                     │
                 │            ▼                            │
                 │  T2  Evidence Bottleneck                │
                 │      · K interpretable concepts + conf. │
                 │      · downstream sees ONLY concepts    │
                 │            ▼                            │
                 │  T3  Case-Grounded Reasoner             │
                 │      · adaptive memory (write-back)     │
                 │      · small model → large only if unsure│
                 │      · MAY ABSTAIN → human              │
                 │            ▼                            │
                 │  T4  Risk-Calibrated Action Policy      │
                 │      · graded action under R_β          │
                 │      · temporal accumulation over the   │
                 │        post-impact window               │
                 │      · labels frozen; critique may only │
                 │        downgrade actions                │
                 └────────────────────────────────────────┘
```

### Why each layer earns its place

| Layer | Novel? | Justified by |
|---|---|---|
| T0 frozen detector | No — deliberately | The whole premise; also what makes the 9-backbone claim possible |
| T1 individualised gate | **Yes (N1)** | Fixes your fold-4 conformal violation; fixes young-adult→elderly threshold bias; bounds duty cycle |
| T2 evidence bottleneck | **Yes (N2)** | Turns `G_faith` from a metric into a guarantee |
| T3 reasoner | Partly | Write-back memory + abstention + cascade are refinements, ablated not headlined |
| T4 action policy | Partly | Your negative actor–critic result justifies the frozen-label constraint |

### Three refinements that are cheap and will be credited

- **Write-back memory.** Confirmed escalated cases enter the retrieval index.
  Turns static retrieval into real case-based reasoning and gives you a
  continual-learning result. It is the missing arrow in your current Fig. 1.
- **Cascaded reasoner.** 1.5B first, 7B only on disagreement. Cuts mean latency
  several-fold and directly answers the 13.6 s objection with a number.
- **Reasoner abstention.** "Insufficient evidence → human" is clinically
  honest and matches the graded action output.

### Design property worth stating explicitly

AEGIS is defined so that **the answer to G2 changes which module fills the T3
slot, not the architecture**. If a gradient-boosted tree on the bottleneck
concepts matches the LLM, T3 becomes a GBM and the LLM becomes the explanation
generator — the paper survives intact. Build it this way deliberately, and say
so; reviewers read slot-based designs as mature rather than lucky.

---

## 4. Prior-work limitations and how AEGIS answers each

| Prior work | Its stated / demonstrable limitation | AEGIS response |
|---|---|---|
| Bhatti et al. 2025 | Softmax force-labels every window; explanation limited to attention weights; no response policy | T1 deferral; T2 concept bottleneck; T4 graded action |
| Sucerquia et al. (SisFall authors) | Their own analysis: D04, D18, F11 dominate threshold crossings; called for approaches focused on these | The near-fall benchmark targets exactly these codes, using the dataset authors' own diagnosis |
| Musci et al. (LSTM), Waheed et al. (BiLSTM) | Random/k-fold splits; natural prevalence; no near-fall evaluation; BiLSTM specificity only 0.93 | LOSO; enriched near-fall protocol; specificity reported per-fold |
| Yu et al. 2021 (KFall) | Pre-impact focus; threshold/SVM/DL benchmarks with no deferral or cost model | Selective deferral under explicit `R_β` |
| Bourke et al. (threshold rules) | Fixed population thresholds, no adaptation | Per-subject online recalibration from unlabelled data |
| Field-wide | Models trained on young adults, deployed on elderly; elderly amplitudes are smaller so thresholds sit too high | T1 individualisation; elderly subgroup reported separately |
| Field-wide | Simulated falls, not real falls | Acknowledged; FARSEEING-style validation named as future work |
| HARGPT, SensorLLM | LLM over raw sensor streams; no cost model, no deferral, cloud-hosted | Local inference; concept-grounded not raw; invoked only on the ambiguous band |
| Split conformal (generic) | Marginal coverage fails under subject shift — **demonstrated by your own fold 4** | Mondrian/subject-conditional conformal risk control |
| SelectiveNet, Deep Gamblers | Require retraining the detector with a reject head | Fully post-hoc; detector frozen; works on all nine backbones |

That last row is a genuinely strong differentiator and is under-exploited in
your current draft: every learned-rejection method needs to retrain the model.
Yours does not, which is why it can be applied to nine architectures at once.

---

## 5. The results tables that make the case

**R1 — The collapse (headline).** Nine backbones × {standard protocol,
near-fall protocol}. Show all of them fall from >0.95 to 0.66–0.74.

**R2 — The lift (headline).** Same nine backbones, each wrapped in AEGIS, on
the near-fall protocol. Show every one recovers. Report mean Δ and per-backbone
Δ. *This is the table that gets the paper accepted.*

**R3 — Deferral-mechanism comparison.** Risk–coverage curves: softmax response,
entropy, temperature scaling, MC-dropout, deep ensembles, marginal conformal,
Mondrian conformal risk control (yours), at matched coverage.

**R4 — Edge-case robustness matrix.** Eight edge cases × {best baseline,
AEGIS}: near-fall activities; elderly subgroup; unseen subject (LOSO); sensor
rotation ±10°/±15°; resampling 50/100 Hz; cross-dataset transfer; low budget
(B = 5%); cost ratio β ∈ {5,10,20}. You already have four of these eight.

**R5 — Adjudicator comparison at matched escalation** (this is G2): trivial
baselines, k-NN vote, GBM on concepts, small LLM, 7B LLM, cascade.

**R6 — Published-literature positioning** with an explicit protocol column, as
in `Q1_PLAN.md` §3.

R1 and R2 are the paper. R3 and R5 are the defence. R4 is what makes a reviewer
write "thorough."

---

## 6. Build order

Do not build T1–T4 before resolving **G1** (is the detector actually
competitive on a standard protocol?) and **G2** (does the LLM beat a GBM on the
same features?). Both are in `Q1_PLAN.md` §7. Building a five-layer
architecture on an unresolved G2 is how papers get restructured three weeks
before a deadline.

| Phase | Work | Gate |
|---|---|---|
| **0** | Fix D19 label; disclose elderly exclusion; run E1 (standard protocol) | G1 |
| **1** | R1 + R2 across the nine-backbone zoo — *already mostly runnable* | — |
| **2** | E2 adjudicator comparison → R5 | G2 decides T3 |
| **3** | T1: Mondrian conformal risk control + budget → R3 | — |
| **4** | T2: evidence bottleneck; re-measure `G_faith` | — |
| **5** | LOSO; elderly subgroup; third dataset → R4 | G4 |
| **6** | Refinements: write-back, cascade, abstention; clinical rating study | — |

Phase 1 alone — with no new architecture at all — probably doubles the paper's
acceptance probability, because it converts the weak-baseline objection into
the central finding.

---

## 7. Honest risks

1. **The collapse may not reproduce across all nine backbones.** If some
   architecture is robust to near-falls, that is still a publishable finding —
   but the framing becomes "which inductive biases survive near-falls", not
   "the field has a blind spot." Check this early; it is cheap.
2. **Mondrian conformal needs calibration data per subject.** With 38 subjects
   and limited windows per subject, strata may be too small for finite-sample
   guarantees. Have a fallback: age-group strata rather than per-subject.
3. **The concept bottleneck will probably cost some accuracy.** Bottlenecks
   usually do. Budget for that and frame it as the interpretability/accuracy
   trade-off, quantified — which is itself a contribution.
4. **Scope.** This is a 2–4 month programme, not a polish pass. If you need to
   submit sooner, Phase 0 + Phase 1 + the existing stack is a legitimate
   submission on its own, and AEGIS becomes the journal extension.
