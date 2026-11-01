# Exploratory research results tables

**Status:** Working tables; not a claim of submission readiness. Both cohorts were previously examined during development. Subject-cross-fitting limits one source of in-sample bias but does not establish independent validation. Percentages are calculated from pooled predictions unless explicitly labeled by fold.

## Table 1. SisFall subject-cross-fitted nine-second trial-prefix results

*Gated retrieval; each fold contains different held-out subjects. All 4,500 retained recordings are evaluated once across the five folds.*

| Outer fold | Recordings, n | Accuracy (%) | F1 (%) | Recall (%) | Specificity (%) |
|---|---:|---:|---:|---:|---:|
| 0 | 946 | 97.46 | 96.83 | 97.87 | 97.20 |
| 1 | 942 | 95.33 | 93.77 | 88.27 | 100.00 |
| 2 | 946 | 96.30 | 95.21 | 92.80 | 98.60 |
| 3 | 932 | 96.14 | 94.93 | 90.35 | 100.00 |
| 4 | 734 | 95.10 | 94.30 | 99.33 | 92.17 |
| **Pooled** | **4,500** | **96.11** | **95.05** | **93.49** | **97.85** |

Trial labels describe whole recordings, not independently timed falls. Inner and outer detector training and normalization differ.

## Table 2. KFall subject-cross-fitted impact-window results

*Gated retrieval; fall windows were selected using impact annotations and ADL control windows were selected separately. All 5,075 recordings appear once across five subject-disjoint test folds.*

| Outer fold | Recordings, n | Accuracy (%) | F1 (%) | Recall (%) | Specificity (%) |
|---|---:|---:|---:|---:|---:|
| 0 | 959 | 100.00 | 100.00 | 100.00 | 100.00 |
| 1 | 959 | 99.69 | 99.66 | 99.32 | 100.00 |
| 2 | 957 | 99.27 | 99.21 | 98.43 | 100.00 |
| 3 | 1,102 | 99.36 | 99.31 | 99.80 | 99.00 |
| 4 | 1,098 | 99.82 | 99.81 | 100.00 | 99.66 |
| **Pooled** | **5,075** | **99.63** | **99.59** | **99.53** | **99.71** |

These are impact-selected window classifications, not online detection accuracy. Results in Tables 1 and 2 use different tasks and are not directly comparable.

## Table 3. Matched classification baselines and retrieval ablations

*Pooled results. SisFall rows share the nine-second prefix task. KFall cross-fitted rows share the impact-window task.*

| Dataset and task | Method | Accuracy (%) | F1 (%) | Recall (%) | Specificity (%) |
|---|---|---:|---:|---:|---:|
| SisFall prefix | Primary detector | 95.38 | 94.18 | 93.55 | 96.60 |
| SisFall prefix | Gated retrieval | 96.11 | 95.05 | 93.49 | 97.85 |
| SisFall prefix | Always retrieve | 96.02 | 94.87 | 92.05 | 98.67 |
| KFall impact window | Primary detector | 99.49 | 99.45 | 99.45 | 99.52 |
| KFall impact window | Gated retrieval | 99.63 | 99.59 | 99.53 | 99.71 |
| KFall impact window | Always retrieve† | 99.92 | 99.91 | 99.96 | 99.89 |
| KFall impact window | Sensor features only† | 99.63 | 99.60 | 99.70 | 99.56 |

† Post hoc baselines from the earlier impact-window experiment on the same outer folds and records. They were not rerun as part of the cross-fitted head study; always-retrieve uses training-subject sensor features, while the sensor-only classifier has its own calibration threshold. No compute-budget matching was performed.

## Table 4. KFall offline streaming replay: new post-impact alarm

*A detection requires a new positive rising edge from annotated impact through two seconds after impact. Earlier alarms do not count in this strict post-impact metric.*

| Method | Eligible falls, n | New alarm matches, n | Event recall (%) | Median delay among matches (s) | ADL alarm edges, n | Monitored ADL hours | Alarm edges / monitored hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary detector | 2,341 | 1,566 | 66.89 | 0.22 | 52 | 4.94 | 10.53 |
| Gated retrieval | 2,341 | 1,814 | 77.49 | 0.22 | 43 | 4.94 | 8.70 |

Five fall recordings lack a full two seconds of follow-up. The matching rule excludes alarms that start before impact, including potentially useful alarms after fall onset. The rate excludes 1.8 seconds of warmup per ADL recording, has no refractory rule, and is measured on short scripted recordings rather than free-living monitoring. Delay excludes processing and communication time.

## Table 5. Common 1.8-second prefix and source-only transfer

*The same observation budget is used for all rows, with whole-trial labels. The within-dataset rows use separate trained models; the transfer row uses SisFall-trained models without KFall calibration.*

| Training → evaluation | Method and operating point | Recordings, n | Accuracy (%) | F1 (%) | Recall (%) | Specificity (%) |
|---|---|---:|---:|---:|---:|---:|
| SisFall → SisFall | Gated, 5:1 cost threshold | 4,500 | 60.67 | 65.16 | 92.05 | 39.79 |
| SisFall → SisFall | Gated, calibration F1† | 4,500 | 63.71 | 66.11 | 88.60 | 47.15 |
| KFall → KFall | Gated, 5:1 cost threshold | 5,075 | 78.34 | 80.48 | 96.55 | 62.70 |
| KFall → KFall | Gated, calibration F1† | 5,075 | 82.66 | 83.01 | 91.65 | 74.94 |
| SisFall → KFall | Gated, source-only ensemble | 5,075 | 53.02 | 66.29 | 99.91 | 12.72 |

† Threshold maximizes calibration F1 subject to at least 90% calibration recall; held-out recall is not guaranteed. The within-dataset head uses in-sample neural training scores; both cohorts were historically exposed. The two detector training protocols differ. Prefix classification is not event detection; most KFall falls occur after this budget.

## Sources and interpretation

- [SisFall cross-fitted results](../results/sisfall_mamba/subject_crossfit_20261101/summary.json)
- [KFall cross-fitted results](../results/kfall/impact_crossfit_20261101/summary.json)
- [KFall new-alarm replay](../results/kfall/crossfit_alarm_edge_audit_20261101/summary.json)
- [KFall matched controls](../results/research_submission_audit_20261101/kfall_matched_baselines.json)
- [Common-prefix operating points](../results/research_v2/common_prefix_operating_points_20261101/summary.json)
- [Source-only transfer](../results/research_v2/sisfall_to_kfall_prefix_transfer_20261101/summary.json)

These tables do not demonstrate novelty, clinical performance, independent external validation, or journal acceptance. The successfully reported path uses retrieval voting, not an LLM. KFall reuse authorization remains to be verified.
