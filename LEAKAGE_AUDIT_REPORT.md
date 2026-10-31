# RCDP/JBHI Paper Leakage Audit and Submission Status

**Audit date:** 2026-10-21  
**Repository:** `harshalDharpure/agentic_elder_fall`  
**Status:** **Do not submit the current manuscript**

## Executive summary

The current paper is not ready for submission. Its headline results are based on
an activity code supplied to System 2 through `include_activity: true`. In the
SisFall and KFall data used here, the first character of that code identifies
the class (`F` = fall and `D` = daily activity). Therefore, the deliberative
classifier receives information derived from the ground-truth label.

The current manuscript discloses this fact, but disclosure does not make the
experiment a valid code-free monitoring experiment. The reported F1 of `0.887`,
recall of `0.977`, cost of `200`, the claimed `83%` cost reduction, the `2.6x`
threshold comparison, and related stress/backbone gains are all results from
the activity-conditioned protocol. They must not be presented as evidence that
the proposed system improves fall detection when only sensor signals are
available.

No `REPORT.md` was found in the repository. The available no-leak evidence is
stored under [`results/noleak_lock/`](results/noleak_lock/), but the manuscript
and package still select and display the leaked activity-conditioned numbers.

## 1. Fatal methodological issue: label information reaches System 2

The manuscript states that System 2 receives the trial activity code:

- [`RCDP_JBHI_Overleaf/main.tex`](RCDP_JBHI_Overleaf/main.tex) uses
  `include_activity: true` in the described protocol.
- The paper explains at
  [`RCDP_JBHI_Overleaf/main.tex:219`](RCDP_JBHI_Overleaf/main.tex#L219)
  that binary labels are inherited from the trial activity code.
- The current abstract and results describe the activity code as unavailable to
  the frozen detector but available to the deliberative path.

Because `F` and `D` identify the target class, this is not merely contextual
metadata. It is label-derived information. A real monitoring service cannot
know whether a recording is a fall trial before classifying it.

Renaming the method “trial-code-conditioned triage” or adding a disclosure does
not remove the leakage. The experiment can be reported as a leakage audit or
as a tightly scoped scripted-trial conditional analysis, but it cannot support
the current clean-method headline claims.

## 2. Results currently used as headlines are leak-dependent

The following numbers are present in the current manuscript and result tables:

| Claim | Current value | Problem |
|---|---:|---|
| Tier-1 to full-stack F1 | `0.759 -> 0.887` | Activity-conditioned System 2 |
| Recall | `0.759 -> 0.977` | Activity-conditioned System 2 |
| Expected cost | `1151 -> 200` | Comparison is not code-free |
| Cost reduction | `83%` | Derived from the leaked result |
| Threshold comparison | `2.6x cheaper` | Threshold sees signal only; stack sees label-derived code |
| Stress-set gain | `0.713 -> 0.896` | Same protocol concern |
| Escalated-window Mistral F1 | `0.972` | Escalated reasoning is label-conditioned |

Verified locations include:

- [`RCDP_JBHI_Overleaf/main.tex:55`](RCDP_JBHI_Overleaf/main.tex#L55)
- [`RCDP_JBHI_Overleaf/main.tex:238`](RCDP_JBHI_Overleaf/main.tex#L238)
- [`RCDP_JBHI_Overleaf/tables/tab_meanstd.tex`](RCDP_JBHI_Overleaf/tables/tab_meanstd.tex)
- [`RCDP_JBHI_Overleaf/tables/tab_perfold.tex`](RCDP_JBHI_Overleaf/tables/tab_perfold.tex)
- [`RCDP_JBHI_Overleaf/tables/tab_ablation_mean.tex`](RCDP_JBHI_Overleaf/tables/tab_ablation_mean.tex)
- [`RCDP_JBHI_Overleaf/tables/tab_generality.tex`](RCDP_JBHI_Overleaf/tables/tab_generality.tex)
- [`RCDP_JBHI_Overleaf/tables/tab_partition_fallback.tex`](RCDP_JBHI_Overleaf/tables/tab_partition_fallback.tex)
- [`JBHI_RESULTS_TABLES.md`](JBHI_RESULTS_TABLES.md)

The package itself explicitly locks the activity-on protocol as the submitted
primary track:

- [`JBHI_ACCEPTANCE_PACKAGE.md`](JBHI_ACCEPTANCE_PACKAGE.md) calls
  `include_activity: true` the “Main lock”.
- [`results/jbhi_package/DECISION_MEMO.json`](results/jbhi_package/DECISION_MEMO.json)
  selects `Q1_historical` with F1 `0.8865787459634431`.
- [`results/jbhi_package/CLAIM_STRATEGY.md`](results/jbhi_package/CLAIM_STRATEGY.md)
  still recommends the activity-on result as the selected primary result.

## 3. Internal protocol contradiction

The manuscript says that retrieval keys redact activity codes, but the
described System-2 evidence includes the code. This creates an unresolved
method-description contradiction:

1. If the retrieval representation contains the activity code, MiniLM
   retrieval can return labelled neighbours.
2. If the code is absent from the retrieval representation, the retrieval
   memory cannot be described as code-conditioned.
3. If the code is present in the full evidence serializer, then the
   experiment is not code-free even if the retrieval key is redacted.
4. The reported ablation behaviour is consistent with the gain coming mainly
   from labelled retrieval/evidence rather than from general-purpose
   reasoning.

The statement that the reasoner relies on “retrieved, labelled neighbours”
should therefore be treated as evidence of the leakage mechanism, not as
evidence of a valid deployment configuration.

## 4. Verified no-leak artifacts

The repository contains a no-leak result directory:

- [`results/noleak_lock/`](results/noleak_lock/)
- [`results/noleak_lock/specialist/summary_v2.json`](results/noleak_lock/specialist/summary_v2.json)
- Fold-level result directories for folds `0` through `4`
- [`results/noleak_lock/kfall/`](results/noleak_lock/kfall/)
- Completion flags `NOLEAK_SYSTEM2_DONE.flag` and `JBHI_CAMPAIGN_DONE.flag`

The aggregate summary currently reports the following no-leak SisFall results:

| Mode | Mean F1 | F1 std. dev. | Mean recall | Mean cost | Interpretation |
|---|---:|---:|---:|---:|---|
| `tier1_only` | `0.767783` | `0.018944` | `0.773344` | `1086.6` | Frozen detector baseline |
| `specialist_tools_full` | `0.769379` | `0.021385` | `0.766465` | `1110.0` | Full specialist path |
| `specialist_tier1_blend` | `0.772472` | `0.025168` | `0.773842` | `1079.6` | Best honest aggregate |
| `specialist_val_fusion` | `0.757735` | `0.021786` | `0.828251` | `899.0` | Higher recall, lower F1 |
| `gate_specialist_judge_f1` | `0.735872` | `0.016380` | `0.838929` | `889.0` | Cost/recall trade-off |
| `gate_specialist_judge_cost` | `0.728691` | `0.015575` | `0.852348` | `850.6` | Lower cost, lower F1 |

The aggregate identifies `specialist_tier1_blend` as `best_honest`, with a
delta of only `+0.004689` F1 over `tier1_only` and three of five folds beating
the baseline. This does not support the current claim of an F1 increase from
`0.759` to `0.887`.

### Important discrepancy about the reported `0.687 / 758` values

The repository snapshot inspected for this audit does **not** contain a
no-leak aggregate with exactly F1 `0.687` and cost `758`. Those values may
refer to another run or an external Cursor report, but no `REPORT.md` or
matching aggregate file was found here.

The values that are directly verifiable in the checked-in no-leak summary are
the values in the table above, including:

- Best honest F1: `0.772472`
- Best honest cost: `850.6` for `gate_specialist_judge_cost`
- Baseline cost: `1086.6`
- Baseline F1: `0.767783`

Before using any alternative values in a paper, the corresponding report,
configuration, per-fold predictions, and aggregation script must be placed in
the repository and independently checked.

## 5. Package and submission red flags

The following files show that the activity-on numbers were intentionally locked
as the primary submission results:

- [`JBHI_ACCEPTANCE_PACKAGE.md`](JBHI_ACCEPTANCE_PACKAGE.md)
  - identifies the activity-on track as the submitted main lock;
  - says the primary tables use F1 `0.887`;
  - says the overnight plan aims to re-hit `0.887 -> 0.89`.
- [`RCDP_JBHI_Overleaf/PAPER_Q1_UPDATE.md`](RCDP_JBHI_Overleaf/PAPER_Q1_UPDATE.md)
  - states that the results tables are numerically unchanged;
  - states that the locked values remain `0.759 -> 0.887`;
  - records `include_activity: true` in the protocol YAML.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md)
  - acknowledges that `include_activity: true` must not be called no-leak;
  - nevertheless leaves the package structured around the activity-on result.

If these files are distributed with a code release, they should be revised or
removed from the submission package. They create an avoidable appearance that
the experiment was optimized to preserve a desired result rather than to test a
pre-specified leakage-free protocol.

## 6. Submission completeness problems

The current manuscript also has unresolved submission metadata and reporting
issues:

### Author and declarations

- [`RCDP_JBHI_Overleaf/main.tex`](RCDP_JBHI_Overleaf/main.tex) defines
  `\authorcheck` as an empty macro.
- As a result, author names, affiliations, corresponding-author information,
  ORCID details, CRediT contributions, funding, conflicts, and acknowledgments
  are not populated.
- The supplementary manuscript also suppresses its author declaration.

### Repository and reproducibility

- The manuscript contains a placeholder for the public repository URL or DOI.
- The submission package does not provide a completed public code/data
  reference in the manuscript.
- The exact code-free protocol, split manifests, prompt version, and
  per-window predictions must be tied to the reported tables.

### Event-level evaluation

- The manuscript contains a marker requesting event-level recall and false
  alarms per hour, but no completed event-level analysis is supplied.
- Window-level trial-membership labels are not equivalent to impact-event
  detection.
- The paper itself states that most windows in a fall trial do not overlap the
  impact phase, making event-level validation necessary.

### Cover letter

[`RCDP_JBHI_Overleaf/cover_letter.txt`](RCDP_JBHI_Overleaf/cover_letter.txt)
still contains unresolved placeholders:

- suggested reviewer names, affiliations, and institutional emails;
- corresponding author name, affiliation, email, and ORCID;
- preprint/conference disclosure;
- author and declaration details.

The cover letter also presents the leaked activity-conditioned result as the
main result. It should not be submitted in its current form.

## 7. Recommended publication options

### Option A: leakage-audit paper — currently supportable

Reframe the contribution as:

> Metadata leakage through retrieval memory can inflate an LLM-based fall
> detection stack from approximately the code-free baseline to the reported
> activity-conditioned result.

This paper should include:

1. Side-by-side code-free and activity-conditioned protocols.
2. Exact data-flow diagrams showing where the activity code enters.
3. Retrieval and evidence-serializer ablations.
4. Controls proving that the code-derived information drives the gain.
5. The no-leak aggregate and all per-fold results.
6. A reproducible leakage-audit checklist.
7. A clear statement that the activity-on result is not a deployable
   code-free monitoring result.

This is the strongest honest option available from the checked-in evidence.

### Option B: clean method paper — not yet supported

A clean method paper requires:

1. `include_activity: false` throughout the full pipeline.
2. No label-derived information in retrieval keys, serialized evidence,
   prompts, memory, filenames, or metadata.
3. Subject-independent test evaluation with calibration isolated from test
   subjects.
4. Event-level metrics and false alarms per hour.
5. A complete, reproducible `REPORT.md` or equivalent report with configuration
   hashes, per-fold outputs, and aggregation.
6. A defensible improvement over cheap controls, not merely a higher recall
   obtained by producing many false alarms.
7. Replacement of all leaked values in the abstract, main text, tables,
   supplement, package notes, and cover letter.

The current checked-in no-leak summary shows only a modest F1 improvement for
the best honest blend and therefore does not justify the current headline.

## 8. Minimum actions before any submission

- [ ] Stop using `0.887`, `0.977`, cost `200`, and the `83%` reduction as clean
  method claims.
- [ ] Decide explicitly between the leakage-audit paper and a new clean-method
  experiment.
- [ ] Add the authoritative no-leak report to the repository; no such
  `REPORT.md` currently exists.
- [ ] Recompute every table from the selected protocol.
- [ ] Remove or relabel all activity-conditioned numbers.
- [ ] Remove the empty `\authorcheck` suppression and complete all author and
  declaration fields.
- [ ] Complete the repository/DOI and cover-letter placeholders.
- [ ] Add event-level metrics and false alarms per hour.
- [ ] Run manuscript consistency checks after the rewrite.
- [ ] Compile the PDF and inspect the author block, tables, references, and
  supplementary material.

## Final determination

The current folder contains useful evidence for a leakage audit, but it does
not contain a clean no-leak `REPORT.md` proving that the proposed method passes
the stated clean-method criteria. The manuscript should not be submitted with
the activity-conditioned numbers as its primary contribution.

