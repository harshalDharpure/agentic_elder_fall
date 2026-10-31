# SisFall and KFall: root-cause audit

Audit date: 30 October 2026. This report examines the current code and stored experiment artifacts, and adds targeted, read-only data checks. It does not claim to be an exhaustive security audit or an independent reproduction of all historical runs. No checkpoints, datasets, or existing results were changed. The workspace contained substantial pre-existing changes. No GPU was available: `nvidia-smi` could not communicate with a driver.

## Executive assessment

The current tasks are materially different: SisFall assigns a fall-trial label to every short window across a long recording; KFall constructs positive windows from an annotated or estimated fall transition while taking negatives from entire ADL recordings. The KFall problem is therefore conditioned on class-specific preprocessing and strongly enriched for impact. This is a well-supported explanation for much of the observed gap, but not a quantified causal decomposition: a matched-protocol experiment is required.

Additional issues make the SisFall decision layer perform poorly: the LLM overrides many ambiguous detector outputs with false alarms, calibration does not account for final LLM errors, and there are concrete retrieval-memory and reporting problems. The architecture itself may be improved, but the data and evaluation mismatch must be addressed first. The high KFall score is not evidence that the same system performs equally well on continuous unsegmented recordings.

## Current comparable-within-protocol results

Five folds; five independent canonical JSON files per method; 1,000 sampled windows per fold. Cost = `10 * FN + FP` and is per 1,000 windows in these slices. The standard deviations below are sample standard deviations across folds, not confidence intervals.

| Dataset and method | F1 mean ± SD | Recall | Precision | Cost |
|---|---:|---:|---:|---:|
| SisFall detector | 0.7678 ± 0.0189 | 0.7733 | 0.7624 | 1086.6 |
| SisFall gate + kNN + LLM | 0.6654 ± 0.0145 | 0.9343 | 0.5182 | 664.2 |
| SisFall fixed tabular specialist/detector blend | 0.7725 ± 0.0252 | 0.7738 | 0.7712 | 1079.6 |
| KFall detector | 0.9874 ± 0.0047 | 0.9913 | 0.9835 | 50.0 |
| KFall gate + kNN + LLM | 0.9790 ± 0.0092 | 0.9950 | 0.9637 | 42.2 |

The SisFall blend is a separately trained gradient-boosted tabular model, not the LLM system. Its aggregate is selected from several explored approaches and should be treated as exploratory. The 10:1 cost encodes a choice, not a measured clinical cost.

## Finding 1 — SisFall target is trial membership, not an impact label

**Confirmed in code; impact scarcity quantified in a sample.** `src/agentic_fall/data/sisfall.py` derives the label from the recording filename and assigns it to every 90-sample window. At 200 Hz each window is 0.45 s, with a 0.05 s hop. Fall recordings include activity before and after impact; a window may look like a normal stationary or moving interval while still inheriting label 1.

Read-only diagnostic: seed 42, uniformly sampled 200 of the available raw fall recordings. Across 58,389 windows, median recording length was 15.0 s. Only **3.08%** of windows contained the sample with maximal acceleration magnitude in their recording; **6.55%** had a sample at or above 2 g and **4.09%** at or above 3 g. The maximum-acceleration sample is an impact proxy, not a verified impact annotation; the g thresholds are descriptive, not ground truth. Sampling is by recording, not by subject, and these values are not confidence intervals.

Implication: the model is asked to classify many local windows whose target is determined by the surrounding trial rather than visibly by the local signal. Higher F1 on impact-selected windows would measure a different target, so it must be accompanied by event-level evaluation of full recordings.

The metadata contains 24 fall-capable subjects and 14 ADL-only subjects. The latter contribute no positive examples, which can cause differences in subject and activity composition across folds; it is not itself proof of confounding or leakage. Report errors by subject and by activity.

## Finding 2 — KFall positives are created using the label and temporal phase

**Confirmed in code and supported by data.** In `src/agentic_fall/data/kfall.py`, the preprocessing branch checks whether the trial is a fall, uses workbook onset/impact when available (otherwise estimates timing), and crops the fall from onset to stability for the transition protocol. ADL windows are sampled from entire recordings. This is class-dependent offline dataset construction: it is legitimate as a defined phase-conditioned benchmark but cannot be performed on an unknown live trial using its true class and impact annotation.

A seed-42 random sample of 20,000 windows per class from the existing KFall archive shows peak acceleration ≥2 g in **79.20%** of fall windows versus **17.95%** of ADL windows; ≥3 g in **73.47%** versus **7.24%**. These figures describe different populations from the SisFall diagnostic and do not isolate model causality. They illustrate the dramatic difference in what the positive label represents.

KFall metadata records 23,760 positive and 203,915 negative windows, but the reported test sampler first targets approximately balanced classes and inserts D18/D19 quotas. SisFall similarly targets a balanced test slice with 80 windows per D18 and D19 per fold. These are engineered benchmark distributions, not free-living prevalence. F1, precision and the meaning of the fixed 10:1 cost change with prevalence.

**Padding check:** in the existing KFall transition archive, none of the 23,760 positive or 203,915 negative windows ended with a fully zero-valued frame. Thus an all-channel trailing-zero-padding shortcut is not present in that archive. This check does not rule out other shortcuts or shortcuts in other data versions.

**KFall units/rate:** the raw CSV example uses plausible g and degrees/second values but the loader does not independently validate or convert units. The binary configuration specifies 100 Hz while `configs/agentic.yaml` specifies evidence extraction at 200 Hz; `scripts/run_kfall_external.py` passes the latter into both retrieval construction and inference. Therefore current code calculates KFall durations with the wrong rate unless overridden by a different run configuration; acceleration peaks are not changed by this rate alone. Historical results lack complete configuration hashes, so exact run-specific impact is unproven. Verify units against the dataset specification and correct/rebuild data-dependent features and memory before rerunning.

## Finding 3 — LLM changes substantially increase SisFall false alarms

**Confirmed from 5,000 saved window records.** Comparing each `windows_gate_knn_llm_fold*.json` final prediction to its own detector probability thresholded at 0.5:

| Change induced relative to detector | Count |
|---|---:|
| New false positives | 1,398 |
| Corrected false positives | 20 |
| New false negatives | 9 |
| Corrected false negatives | 358 |

Of the 800 D18/D19 quota windows, the detector makes 150 false alarms and the stack makes 579. These are stress-set window counts, not alarms per event or hour. Some recorded predictions may also be affected by routing and fusion policy; the files establish the outcome of the complete variant, not which prompt phrase individually caused each error.

The confidence gate in `agents/confidence_gate.py` chooses bounds by minimizing confident-decision error plus a deferral penalty. Its calibration objective does **not** evaluate the downstream reasoner's actual mistakes. The prompt in `agents/llm_reasoner.py` emphasizes avoiding missed falls; this is a plausible contributor to the observed positive bias, but the audit did not establish causation via a controlled prompt experiment. Approximately 51.1% of SisFall windows are routed to the slow path, versus 1.46% in KFall. Logged SisFall escalated processing is roughly 2.4–2.5 seconds per window in these runs; actual hardware and service configuration affect latency.

**Improve:** first compare the detector alone at validation-selected thresholds against the LLM at matched recall or matched false-alarm rate. Tune a calibrated, bounded correction or abstention policy using a subject-disjoint validation set, then lock it before testing. Evaluate whether LLM output adds information beyond the probability, numerical features and kNN; retain it for response/rationale only if that benefit is demonstrated.

## Finding 4 — retrieval memory includes validation subjects

**Confirmed in code and an existing artifact.** `scripts/build_memory.py` selects from the full outer-training fold, whereas `make_fold_split` subsequently holds out inner validation subjects. Matching redacted SHA-256 case IDs back to the source processed dataset for existing SisFall fold 0 found **4,404 inner-training cases, 596 validation cases and 0 test cases**, out of 5,000 memory cases. This confirms validation-memory contamination for this artifact; it does not prove all folds or every historical run have the same counts. KNN retrieval of validation cases can make calibration optimistic, especially with overlapping windows from the same recording. The specialist also featurizes training windows against training memory without an explicit self/same-recording exclusion, risking self-neighbor shortcuts.

For the current five KFall CNN–LSTM–Attention memory files, tracing the stored window indices found **2,000 inner-training cases and zero validation/test cases in each fold** under the current split code. This checks subject membership of current files only; it does not prove historical files were identical or establish other provenance properties.

**Improve:** build each memory only from the final inner-training subject split; save split, dataset, checkpoint and embedder hashes; reject mismatches. For training features, use out-of-fold retrieval or exclude the same subject/recording. Use separate calibration data or nested cross-validation for threshold selection.

## Finding 5 — short windows cannot support long-term response logic

**Confirmed by configured durations and feature code.** Biomechanics is calculated independently per 90-sample window. The window length is 0.45 s on SisFall, so its observed post-impact stillness cannot reach the configured 3 s moderate or 10 s severe stillness thresholds. At the current shared 200 Hz setting KFall's computed duration is also 0.45 s, although 90 samples represent 0.9 s at 100 Hz. The system has no multi-window state in `AgenticPipeline.run` that would establish sustained immobility.

`body_tilt_deg` measures angular change between the first and last mean acceleration vectors of a single window. It is not an absolute body posture estimate and can be affected by dynamic acceleration. A rationale claiming the patient is lying down, incapacitated, or recovered would need additional evidence. Actions are simulated; no emergency-contact or clinical outcome validation is present.

**Improve:** implement a causal rolling state machine, explicitly distinguish window-local features from long-term observations, accumulate movement over seconds, and verify events/actions against independently specified annotations. Measure latency end to end, not just model inference.

## Finding 6 — operating points and training selection are inconsistent

The frozen SisFall ablation and specialist baselines use threshold 0.5, while `scripts/run_backbone_benchmark.py` can select a cost-sensitive threshold on validation. `fit_model` selects checkpoints on validation F1 at the default threshold; then a separate threshold may optimize cost. These are not necessarily wrong but do not directly optimize the same objective. Balanced training and validation subsampling can also distort calibration for natural prevalence.

The same benchmark script ranks backbones on **test-set F1** before choosing which get agentic evaluation. Results for those chosen backbones are exploratory, not a clean preselected comparison. In `run_specialist_judge_ladder.py`, the escalation-specific model can be fitted on validation examples and its confidence threshold tuned on those same examples, leading to optimistic calibration for those judge variants. The fixed blend uses the separate train-fitted specialist; it remains subject to the memory issue and post-hoc selection among variants.

The 20,000-window natural-prevalence retrained SisFall benchmark reports mean F1 0.7595 for ModernTCN and 0.7389 for CNN–LSTM–Attention using validation-selected Youden thresholds. This is evidence that trying ModernTCN is reasonable, not proof that architecture alone explains the gap: checkpoints, thresholds and protocol differ from the primary 1,000-window results. Compare candidates on identical untouched folds with a predeclared selection rule.

## Finding 7 — reporting and provenance need correction

- `RCDP_JBHI_Overleaf/main.tex` presents a gated retrieval/LLM deliberative path next to the **0.772** selected blend. That blend is the tabular specialist and runs over all test windows in the specialist script. Distinguish it explicitly from the LLM system before submission.
- Historical activity-conditioned runs can reveal fall/ADL from F/D codes. They cannot justify a sensor-only headline. Current configuration disables activity in evidence, but historical JSONs may not preserve enough configuration and memory provenance to independently certify every run.
- `scripts/eval_literature_protocol.py` tunes thresholds and impact-window width using test data; 0.915 trial F1 is exploratory and must not be used as a held-out result. Trial evaluation also depends on whether all windows or a subsample were selected.
- `scripts/run_ablations.py` computes `mean_llm_calls_escalated` using adjudication count as denominator rather than the number escalated. The field can be wrong. `eval/train_loop.py` assigns multiclass macro F1 to the fields named precision and recall; binary results in this report do not use that code path.
- `scripts/audit_prompt_leakage.py` invoked without inputs scans nothing. Result JSONs and memory should be checked explicitly; a redacted text scan alone does not prove split independence or embedding provenance.
- KFall archives do not preserve trial IDs and timestamps. They cannot support faithful event reconstruction or false alarms per hour without re-preparing from raw data.
- Several outputs are copies or merged rows. Aggregate one identified file per fold and keep protocol, checkpoint and configuration manifests.

## Recommended experimental sequence

1. **Lock a target.** Choose post-impact detection/response as the principal task, or run a separately specified pre-impact experiment. Define temporal labels from annotations where possible; distinguish uncertain labels from negatives. Keep complete trial timelines.
2. **Repair reproducibility and data flow.** Preserve trial IDs and window start times; build training-only fold memory, prevent self-neighbor retrieval, correct KFall rate and verify physical units; save explicit subject and file manifests, seeds and config/checkpoint hashes. Do not overwrite prior artifacts.
3. **Establish a continuous detector baseline.** Evaluate full held-out recordings with validation-selected threshold and predeclared alarm merging/refractory period. Report event recall, false alarms/hour, latency, per-subject and per-activity results. Use all fall/ADL periods available, with explicit caveats that trials are simulated rather than free-living.
4. **Run a controlled protocol-matching experiment.** On both datasets, compare full-trial windowing, event-centered windows and the existing class-conditioned KFall preprocessing. Use the same model family, physical window duration, subject separation and reporting unit. This is the experiment that can measure how much protocol explains the apparent gap.
5. **Test detector changes, one at a time.** Compare CNN–LSTM–Attention, smaller ModernTCN and a sensor-only peak threshold; train-only channel normalization; 0.45/0.9/2 s windows; event-focused sampling/temporal targets; and class weighting/mixup settings. Keep training and validation subjects fixed and choose hyperparameters only in inner validation. Longer context may improve event detection but must be evaluated for latency.
6. **Test the incremental decision layer.** Compare thresholded detector, calibrated score, tabular specialist, kNN and LLM at matched recall and matched false-alarm/hour. Report paired subject-clustered uncertainty, precision-recall curves, escalation rate, latency, and sensitivity to the cost ratio. Do not assume an LLM improves classification.
7. **Validate response.** Implement temporal observation and clinician/expert-defined action references before claiming response quality. Assess false emergency escalations and misses independently of rationales. Any pre-impact protection path must meet its real end-to-end deadline.
8. **Test generalization and robustness.** Use subject-disjoint seeds, sensor orientation perturbations, missing/noisy channels, and true cross-dataset transfer with justified sample-rate/unit harmonization. Training separate models within each dataset is not cross-dataset transfer.

The sequence prioritizes determining what causes the observed difference. No proposed model change has yet been demonstrated to improve untouched test performance.
