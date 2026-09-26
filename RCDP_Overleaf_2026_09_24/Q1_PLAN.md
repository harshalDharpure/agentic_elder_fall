# Getting RCDP into a Q1 journal — diagnosis, experiment plan, comparison strategy

Written as a reviewer would read it, then as a co-author would plan it.

---

## 0. Three things I found that must be fixed before anything else

### 0.1 D19 is mislabelled — factual error about the dataset

The manuscript says **"D18 (stumble) and D19 (sudden sit)"**. In the SisFall
release:

| Code | Actual activity |
|------|-----------------|
| D18 | Stumble while walking |
| D19 | **Gently jump without falling (trying to reach a high object)** |

D19 is a *jump*, not a sudden sit. Any reviewer who knows SisFall catches this
in the abstract, and it puts every other dataset claim in doubt. "Sudden sit"
is roughly D05/D11/D13 (collapse into a chair).

Fix: correct the label everywhere, and decide whether you want D19 (jump) or
the collapse-into-chair codes in the benchmark. Note the original SisFall paper
found **D04 (jogging), D18 (jump/stumble) and F11** were the activities that
most often crossed detection thresholds — you can cite that to justify the
choice of D18 directly from the dataset authors.

### 0.2 Elderly subjects never performed D18 or D19

From the SisFall paper: elderly participants did not perform falls, nor D06,
D13, D18, D19, on the sports physician's recommendation. The one exception is
SE06, a 60-year-old judo expert.

**Your lead result — the ambiguous stress benchmark — is therefore computed
entirely on young adults.** For a paper motivated by eldercare, an undisclosed
young-adult-only headline is a serious problem, and it compounds a known bias:
elderly activity amplitudes are smaller, so models trained on young adults set
thresholds too high, which is exactly the wrong direction for near-falls.

Fix (do all three):
1. State it explicitly in Setup and Limitations.
2. Report Tier-1 and stack performance on elderly subjects separately on the
   main protocol.
3. Build a second near-fall proxy from codes elderly *did* perform (D05, D11 —
   trying to get up and collapsing into a chair) and show the effect survives.
   If it does, this becomes a strength rather than a hole.

### 0.3 Your Tier-1 looks broken against the literature

Published SisFall binary detectors report:

| Work | Reported | Protocol caveat |
|---|---|---|
| Musci et al. (LSTM) | 97.16% fall acc | — |
| Waheed et al. (BiLSTM) | 97.21% acc, 96.97% sens, 93.18% spec | — |
| Sampling-rate study (CNN-LSTM @20 Hz) | 98.9% acc, 96.7% sens, 99.6% spec | multi-stage validation |
| Multi-modal CNN-LSTM + attention | 98.7% F1 | subject-wise split, extra modalities |

**Your Tier-1 reports F1 0.759.** A reviewer sees 0.76 against a literature of
0.97–0.99 and concludes your baseline is undertrained — at which point the
entire +0.128 improvement reads as "fixing your own weak model."

The likely truth is that your protocol is *deliberately much harder* (43%
prevalence, forced D18/D19 enrichment) and the same pipeline hits 0.987 on
KFall, which is near ceiling. But you never show this, so the reviewer cannot
know it.

**This is the single most important experiment in the plan — see E1.**

### 0.4 Also check: double-blind may already be broken

Public search surfaces a GitHub repo (`agentic_elder_fall`) and indexed text
quoting "SisFall ambiguous bench (D18/D19) F1 0.713 → 0.896" — your exact
headline numbers. If that is yours and public, a double-blind submission is
compromised. Either make the repo private until acceptance, or target a
single-blind venue, or disclose per the journal's preprint policy.

---

## 1. Journal targeting

| Venue | IF (approx) | Fit | What it demands that you lack |
|---|---|---|---|
| **IEEE JBHI** *(recommended)* | ~7 | Clinical decision support, cost-sensitive triage — best match for the Action Agent framing | Clinical grounding of β; some human validation (E10) |
| Information Fusion | ~14 | Multi-agent + multi-source fusion is a genuine fit; highest reward | Very competitive; needs E2 + E4 + a third dataset |
| IEEE IoT Journal | ~8 | Edge/cloud dual-process split | Energy + embedded profiling (E8) is effectively mandatory |
| Eng. Applications of AI / ESWA | ~7.5 | Applied agentic systems; faster turnaround | Less clinical weight, but E2/E4 still required |
| IEEE Sensors Journal | ~4 | Safe fallback | Lowest prestige of the set |

**Recommendation:** target **JBHI**. The paper's identity is a *clinical
decision layer*, not a sensor or a network contribution, and JBHI reviewers
reward exactly the cost-asymmetry and honest-negative-result framing you
already have.

If E2 comes back strongly in the LLM's favour, submit to **Information Fusion**
first — rejection there is fast and costs you little.

---

## 2. Experiment plan, ranked by effect on the accept/reject decision

### Tier A — the paper is not competitive without these

**E1. Standard-protocol Tier-1 reference run.** *(~half a day, no LLM)*
Run Tier-1 on a conventional SisFall binary protocol: natural prevalence, no
near-fall enrichment, subject-independent. You should land near 0.95–0.98 F1.
Then build one table:

| Protocol | Prevalence | Tier-1 F1 |
|---|---|---|
| Standard SisFall binary | natural | *expect ~0.97* |
| Homogeneous stratified (ours) | 0.433 | 0.759 |
| Near-fall stress (D18/D19) | 0.33 near-fall | 0.713 |

This single table does more for the paper than any other experiment. It turns
"your baseline is weak" into **"state-of-the-art detectors are fine on average
and collapse on near-falls"** — which is your thesis, demonstrated instead of
asserted. Without it you will be rejected on the baseline alone.

**E2. A strong non-LLM adjudicator.** *(~1 day)* ← **the experiment that decides the paper**
Your evidence serializer emits ~5 scalar biomechanical features. Train
gradient-boosted trees (and logistic regression) on those same features from
the training folds, and use them as the System-2 adjudicator on the identical
escalated set. Every reviewer will ask: *is the LLM doing anything a 20-line
classifier cannot?*

- If GBM ≈ Mistral → reframe. The contribution becomes the **gate plus the
  evidence representation**, with the LLM as the explainability layer. Still
  publishable, but the current framing would be wrong and a reviewer will find
  it.
- If Mistral ≫ GBM → this is your strongest result and belongs in the abstract.

Do not submit without this. If you run one experiment from this document, run
this one.

**E3. Diagnose the heuristic backend.** *(~2 hours)*
F1 = 0.034 on the ambiguous set is *below trivial*: always-predict-fall on a
roughly balanced set scores ≈0.67, random scores ≈0.4. A score of 0.034 is
anti-correlated, which almost always means inverted logic or a parsing bug.
Right now `tab_llm` reads as "our method beats a broken baseline." Either fix
it, or replace it with honest trivial baselines (majority class, always-fall,
random, k-NN majority vote) and report those.

**E4. Selective-prediction baseline suite.** *(~2–3 days)*
Same frozen Tier-1, same escalation budget, different deferral rules:
softmax-response, predictive entropy, temperature scaling + threshold,
MC-dropout variance, 5-seed deep ensemble, split conformal (you have it),
Chow's rule, and optionally SelectiveNet or Deep Gamblers (these need
retraining).

Report **risk–coverage curves** — deferral work is expected to be evaluated
this way, and their absence is conspicuous. Your claim should be that the
*cost-calibrated* gate dominates the others at equal coverage under β=10.

### Tier B — these move it from "revise" to "accept"

**E5. Leave-one-subject-out.** *(compute-heavy, no new code)*
SisFall has 38 subjects → 38 folds instead of 5. This ends the statistics
problem permanently: n=38 paired observations, real power, no Wilcoxon floor at
0.0625, no apologising for BCa. If LLM cost is prohibitive, run LOSO for
Tier-1 + gate and keep 5-fold for the full stack, reporting both.

**E6. A third dataset.** *(~2 days)*
UP-Fall, FallAllD or UniMiB SHAR — you already cite all three. Pick one with a
**different sensor placement** so it also tests the robustness claim. Two
datasets reads as minimum effort; three reads as generalisation.

**E9. Elderly subgroup + elderly-feasible near-fall proxy.** *(~1 day)*
See §0.2. Mandatory given the D18/D19 exclusion.

**E10. Small clinical validation of rationales and actions.** *(~1 week incl. recruitment)*
Have 3–5 clinicians rate ~50 escalated cases: is the rationale clinically
sound, and is the graded action appropriate? Report Cohen's/Fleiss' κ and
agreement with the Action Agent. This converts your weakest pillar from
"simulated" to "expert-validated" and is, in my judgement, the difference
between JBHI accept and JBHI major-revision.

### Tier C — strengthens, not required

**E7. Reasoner size/latency Pareto.** *(~1 day)* Qwen2.5-1.5B/3B/7B,
Llama-3.2-3B, Phi-3.5. Plot ambiguous-case F1 against latency. Attacks the
13.6 s objection with data rather than argument, and grounds the "distilled
on-device reasoner" future-work claim.

**E8. Energy and embedded profiling.** *(~2 days, needs hardware)* mJ/window for
System 1, mJ/escalation for System 2, projected battery life at 47% duty cycle
on a Jetson Orin Nano or RPi 5. **Mandatory if you target IoT-J.**

**E11. Evidence-serializer ablation.** LLM on raw windowed signal text vs on
Σ(x). Tests whether the biomechanical serialisation — which you claim as a
contribution — actually matters.

**E12. Qualitative failure analysis.** 6–8 escalated cases: where the LLM
rescued Tier-1, where it made things worse, with the actual rationale text.
Q1 reviewers consistently reward this and it costs nothing but writing.

**E13. Retrieval width under Mistral.** Closes the open `tab_knn_k` null.

---

## 3. Comparison strategy — exactly which papers, and how

### Tier 1: numeric comparison on shared data

| Compare against | Where | How to make it fair |
|---|---|---|
| Bhatti et al. 2025 | KFall | You already do this correctly — same backbone, binary protocol, explicit disclaimer about their multi-phase table |
| Yu et al. 2021 KFall benchmarks | KFall | The dataset paper ships benchmark algorithms (threshold, SVM, DL). Cite their numbers and note protocol deltas |
| Musci et al. (LSTM), Waheed et al. (BiLSTM), CNN-LSTM sampling-rate study | SisFall | **Report their numbers in a table with an explicit protocol column** |

On that last row, be aggressive about the protocol column — most published
SisFall numbers use random or k-fold splits, not subject-independent ones. If
you run E5 (LOSO), you can state that your protocol is *stricter* than most of
the literature. That reframes a 0.97 vs 0.99 gap from a weakness into a
methodological advantage. Build the table like this:

| Method | Split | Prevalence | F1 | Near-fall reported? |
|---|---|---|---|---|
| Musci et al. | k-fold | natural | 0.97 | no |
| Waheed et al. | k-fold | natural | 0.97 | no |
| CNN-LSTM @20 Hz | multi-stage | natural | ~0.98 | no |
| **Tier-1 (ours)** | **LOSO** | natural | *E1* | — |
| **Tier-1 (ours)** | **LOSO** | near-fall enriched | 0.713 | **yes** |
| **RCDP (ours)** | **LOSO** | near-fall enriched | 0.896 | **yes** |

The "near-fall reported?" column is the one that sells the paper: nobody else
reports it, and you can show why that matters.

### Tier 2: mechanism comparison (you reimplement, on your data)

This is the E4 suite. Chow 1970, softmax-response, entropy, temperature
scaling, MC-dropout, deep ensembles, split conformal, SelectiveNet, Deep
Gamblers, learning-to-defer (Mozannar & Sontag). All already in `refs.bib`
except SelectiveNet and Deep Gamblers — add those.

### Tier 3: position only, no numeric comparison

HARGPT, SensorLLM, Health-LLM, Med-PaLM. Different tasks; comparing numbers
would be dishonest. Your `tab_related` already handles this correctly. Add one
sentence each on *why* a numeric comparison would be meaningless.

---

## 4. Architecture improvements worth making

Ordered by return on effort.

1. **Make the memory adaptive.** Right now retrieval is static over training
   folds. Writing confirmed escalated cases back into the memory turns it into
   an actual case-based-reasoning system and gives you a continual-learning
   result. Currently the architecture diagram has no write-back because the
   system has none — it is the most obvious missing arrow.
2. **Confidence-aware escalation budget.** Instead of a fixed threshold pair,
   escalate the top-*k* most uncertain windows subject to a duty-cycle budget
   (e.g. ≤10%). This directly answers the 47% objection and gives you a
   deployable knob. Report accuracy at 5%, 10%, 20% budget.
3. **Cascade the reasoner.** Small model first; escalate to 7B only when the
   small model is itself unsure. Cuts mean latency substantially and pairs with
   E7.
4. **Let the LLM abstain.** Currently it must label. Allowing "insufficient
   evidence → escalate to human" is more honest clinically and gives a
   three-way policy that matches the Action Agent's graded output.
5. **Calibrate the reasoner's own confidence** and fold it into the Action
   Agent rather than using the label alone.

Items 1 and 2 are the two that reviewers will actually credit as architectural
novelty; 3–5 are refinements.

---

## 5. Framing changes

1. **Lead with the collapse, not the recovery.** The strongest version of this
   paper opens: *"Detectors reporting 0.97+ F1 on SisFall drop to 0.71 on
   near-fall activities — we show this and fix it."* That requires E1 and it
   makes the contribution unmissable.
2. **Retire "A\*/Q1" language from the paper itself** — the `tab_bhatti_gaps`
   caption currently says "complementary Q1 claim." Reviewers dislike it.
   (Already removed in the current draft; keep it out.)
3. **Move the KFall multi-class context table to supplementary.** It invites an
   unflattering comparison for no gain.
4. **Reduce to ~8 main tables.** You have 28. Move the rest to supplementary;
   a Q1 reviewer reading 28 tables gets annoyed before they get convinced.
5. **Name the system consistently.** RCDP appears alongside "the stack", "the
   full stack", "gate_knn_llm". Pick one.

---

## 6. Two packages

**Minimum viable Q1 submission** — E1, E2, E3, E9, plus §0.1/§0.2 fixes.
Roughly 1 week of work. Gets you past desk rejection and the "weak baseline"
and "is the LLM necessary" objections, which are the two that kill it.

**Strong submission** — add E4, E5, E6, E10, E12 and architecture items 1–2.
Roughly 4–6 weeks. This is a genuine JBHI accept in my estimation, and a
credible Information Fusion attempt.

---

## 7. Go / no-go gates

Run these in order and stop to re-plan if a gate fails.

| Gate | Test | If it fails |
|---|---|---|
| **G1** | E1: does standard-protocol Tier-1 reach ≥0.95 F1? | If no, the detector really is undertrained. Fix it before anything else — everything downstream is invalid. |
| **G2** | E2: does Mistral beat GBM-on-evidence-features by a clear margin? | If no, reframe around gate + evidence representation, with the LLM as an explainability contribution. Do not claim the LLM drives accuracy. |
| **G3** | E3: is the heuristic baseline genuinely that bad, or buggy? | If buggy, fix and re-run `tab_llm` — the current numbers would not survive scrutiny. |
| **G4** | E9: does the near-fall effect hold on elderly-feasible activities? | If no, the eldercare framing must be softened substantially. |

G1 and G2 are the real ones. Everything else is polish by comparison.
