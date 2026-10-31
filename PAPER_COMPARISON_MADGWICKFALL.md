# MadgwickFall-Net comparison and publication assessment

Reviewed 30 October 2026 against `bioengineering-13-00568-v2.pdf` (28 pages), the current source code, and the canonical fold results documented in `WORKFLOW_RESULTS_REVIEW.md`. This is a methodological review, not an independent reproduction of the paper.

## Paper identity and journal status

Qijun Zhong, Jing Wang and Guiling Sun, “MadgwickFall-Net: A Lightweight Dual-Frame Feature Fusion Network for Pre-Impact Fall Detection Using Wearable IMUs,” Bioengineering 2026, 13, 568; published 16 May 2026. DOI: https://doi.org/10.3390/bioengineering13050568.

The publisher's current journal page lists Bioengineering as JCR Q2 (Engineering, Biomedical) and CiteScore Q2 (Bioengineering): https://www.mdpi.com/journal/bioengineering. Thus this paper is not evidence of a Q1 publication under those classifications. Quartiles must always specify database, category and year; a separate historical/SJR classification has not been established here.

## What the paper contributes

The method transforms acceleration and angular velocity into a gravity-aligned coordinate frame using a six-axis Madgwick orientation estimator. Four network branches process body/global acceleration and body/global angular velocity. The network uses lightweight convolution/temporal modules and feature fusion. It aims to detect falls early enough to trigger protective equipment, rather than to deliberate about post-fall responses.

Preprocessing uses an 8 Hz fourth-order Butterworth filter. Fall signals end 250 ms before annotated impact. Offline sampling anchors fall windows backward from that cutoff, while ADL windows advance forward. Inputs contain 87 samples at 100 Hz (nominally 875 ms; 870 ms at this integer sample count), with hop 34 samples. Windows retain chronological order. The text says fall-window labels depend on overlap with annotated fall phases; the exact overlap threshold is not specified in the passages describing sampling.

The benchmark uses 32 KFall subjects, 2,729 ADL recordings and 2,342 usable fall recordings after excluding four short trials. Five subject-grouped folds are used. Reported training includes seed 42, 105 epochs, AdamW, class-weighted smoothed cross-entropy, sevenfold training fall augmentation, and a fixed decision threshold of 0.5. The stated test isolation is a strength, but the paper does not clearly describe a separate inner validation protocol for model/window/filter-parameter selection.

## Reported evidence

| Evaluation | F1 | Precision | Recall | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|
| Primary trial-wise max, Table 1 | 0.9824 | 0.9773 | 0.9876 | 0.9801 | 0.9836 |
| Auxiliary window-level, Table 2 | 0.8911 | 0.9042 | 0.8789 | 0.9958 | 0.9908 |

Trial scores take the maximum window probability and threshold it at 0.5. This is an alarm-anywhere-in-trial rule, not majority voting. It combines multiple predictions, and ADL false alarms accumulate with exposure duration.

- Rolling inference: 2,302 of 2,342 falls detected before impact (98.3%); median lead time 390 ms and mean 410 ms among detected falls.
- Of detected falls, 78.3% have at least 250 ms lead time. Across all falls this is 1,803/2,342 = approximately 77.0%; missed trials must remain in the denominator when reporting timely protection coverage.
- ADL false alarms: 55/2,729 trials, or 2.02%; stumble T10 has 8.18% false positives and chair-collapse T15 has 14.65%. These are trial rates, not false alarms/hour.
- Dual-frame ablation: full F1 0.9824 versus global-only 0.9723 and body-only 0.9660. The full model also has more branches; a capacity-matched four-branch control would strengthen the causal attribution to coordinate fusion.
- Sampling ablation: reverse-fall/forward-ADL F1 0.9824 versus all-forward 0.8754. Rolling detection changes much less (98.3% versus 97.4%), showing that evaluation construction and continuous operation can produce materially different conclusions.
- FP16 weights occupy 59.7 KB; this is storage, not 59,700 parameters. Deployment simulations use FP32: 166.1 KB total flash, 38.2 KB RAM and 18.70–25.75 ms inference on three STM32 targets. These are simulated benchmarks, not a demonstrated elderly field deployment.

## Comparison with our evidence

| Dimension | MadgwickFall-Net | Our current repository |
|---|---|---|
| Main task | Pre-impact fall warning | Window detection and simulated graded response |
| Dataset evidence | KFall only | SisFall and KFall, each trained/evaluated within dataset |
| Timing | Inputs exclude last 250 ms before impact in main benchmark | SisFall trial-inherited labels; KFall onset-to-stability transition crops can include impact/post-impact |
| Primary evaluation | Full trial-wise maximum | Capped 1,000-window slices per fold |
| Sampling | 87 samples, 100 Hz, hop 34 | SisFall 90/200 Hz, hop 10; KFall 90/100 Hz, hop 10 |
| Input | Acceleration/gyro in two frames | Detector signals plus biomechanics and optional training-case retrieval |
| Decision layer | Small neural detector | Confidence gate, optional local LLM, or separate tabular specialist |
| Operational evidence | Lead times and simulated microcontroller deployment | Simulated actions; no validated continuous event benchmark |
| Generalization | Subject-held-out KFall | Subject-held-out within each dataset; no demonstrated cross-dataset transfer |

Our sensor-only KFall detector F1 is 0.987373 and LLM stack F1 is 0.978999. Neither may be ranked against their 0.9824 trial F1 or 0.8911 window F1: timing, labels, sample population, aggregation and splits differ. Merely selecting their window table does not make the comparison fair.

Our sensor-only SisFall baseline/blend/LLM F1 values are 0.767783/0.772472/0.665351. The 0.772472 result is a fixed blend with a gradient-boosted tabular specialist, not evidence of LLM benefit. The LLM improves recall at a substantial precision cost. KFall LLM aggregate FP/FN counts change from 40/21 to 91/12; cost falls only because the assumed false-negative penalty is ten times the false-positive penalty.

Historical activity-conditioned 0.887-style SisFall results are unsuitable for clean comparisons. Our exploratory 0.915 trial result also selects thresholds using test labels and cannot be compared as an untouched test estimate.

## Critical reading of the paper

Its strongest elements are a focused physical hypothesis, direct baseline retraining under a common benchmark, mechanism ablations, activity-specific false alarms, sequential timing evaluation, and explicit memory/runtime reporting. These are reporting and validation practices worth adopting.

Limitations and unresolved questions remain:

1. All experiments use simulated falls by healthy young adults from one dataset. The authors acknowledge missing older-adult and cross-dataset validation.
2. Offline phase labels and impact-anchored sampling are appropriate for supervised benchmark construction, but do not themselves establish continuous monitoring performance. This is different from exposing a target-revealing activity code to the classifier.
3. A Butterworth filter is specified, but causal implementation, initialization and group delay are not sufficiently established by the manuscript alone. Do not assume future leakage; verify whether code uses a causal filter or a noncausal forward/backward implementation before trusting real-time lead-time claims.
4. Model/window/Madgwick-beta ablations report the same five-fold benchmark. The separation of exploratory choices from final evaluation is unclear. This is a reproducibility question, not proof that the authors tuned on test labels.
5. Their statistical evidence uses only five fold pairs, with overlapping training sets and multiple baseline comparisons. A Wilcoxon comparison against lwRPPC-TCN is not significant at 0.05 (p=0.063), although a paired t-test is (p=0.014). These do not eliminate uncertainty or establish prospective clinical benefit.
6. The data statement links KFall; a public implementation/weight repository was not identified in the PDF. Exact split lists, label-overlap rule, filter behavior and deployment reproduction would improve auditability.
7. Earliest-alarm lead time must be interpreted with false alarms and the labelled fall-onset boundary. The text does not clearly establish that alarms before fall onset are excluded. Reproduction should report these separately.

## Q1 assessment and concrete route

Current assessment: our existing results do not yet support a strong Q1 claim that LLM reasoning improves fall detection. They can support an honest study of recall/false-alarm trade-offs, modest specialist fusion, and leakage risks. Publication decisions cannot be predicted from an F1 threshold.

The strongest coherent direction is selective, sensor-only fall response under explicit latency and error costs. Keep immediate protection on a fast detector; evaluate deliberation as a later response function. Seconds of LLM inference cannot be assumed compatible with a 200–400 ms pre-impact protective deadline.

Priority experiments:

1. Fix inner-training/validation-memory separation, KFall evidence sampling rate and sensor units. Store code/configuration hashes, per-subject split lists, trial IDs, window timestamps and immutable predictions. Remove activity information from inputs and audit actual prompts/memory.
2. Reproduce a common KFall benchmark with identical subject folds, phase labels, filtering, window timing, full held-out recordings and fixed or validation-selected thresholds. Include MadgwickFall-Net if an implementation is available or implement it transparently. Label a reproduction as such; do not silently assume exact equivalence.
3. Evaluate continuous recordings causally: event recall, false alarms/hour, time to detection, recall by a fixed pre-impact deadline, duplicate-alarm suppression and end-to-end latency. Restore recording/trial identifiers to KFall processed artifacts, which currently do not preserve file IDs.
4. Compare the detector, threshold tuning, calibrated probability model, tabular specialist, kNN and LLM at matched recall or matched false alarms/hour. If inexpensive controls attain the same operating point, the LLM needs another demonstrated benefit.
5. For graded response, define reference actions from a documented expert rubric or independent annotation. Measure action error, severe-action misses and unnecessary emergency escalation; evaluate factual rationale grounding separately from action correctness. Simulated actions alone do not validate response quality.
6. Lock choices before final test evaluation. Report multiple training seeds and uncertainty clustered by subject, paired differences and all tested primary comparators. Do not treat overlapping windows as independent observations.
7. Add a real generalization test: train on one dataset and test on the other with justified sensor harmonization, or use a genuinely separate prospective/external cohort. Training separate models on two datasets is useful but does not demonstrate transfer.

A focused paper with these experiments is a more defensible Q1 candidate than a broad agentic-superiority claim based on existing scores. No new training, publication action or source-code modification was performed for this review.
