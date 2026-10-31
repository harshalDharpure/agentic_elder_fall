# JBHI Submission Results

## Primary SisFall comparison

| Method | F1 | Recall | Expected cost / 1,000 windows |
|---|---:|---:|---:|
| Frozen CNN--LSTM--Attention detector | 0.768 +/- 0.019 | 0.773 +/- 0.015 | 1086.6 +/- 80.6 |
| Selective decision layer | **0.772 +/- 0.025** | **0.774 +/- 0.023** | **1079.6 +/- 115.3** |

The values are five-fold descriptive aggregates from the verified submission
protocol. The difference is modest and is not presented as a definitive
superiority claim. Event-level evaluation and prospective validation remain
necessary.

## Protocol

- Five subject-independent folds.
- Calibration uses training subjects only.
- Retrieval uses training-subject cases only.
- The decision layer receives sensor evidence and model scores.
- The cost is \(10\mathrm{FN}+\mathrm{FP}\) per 1,000 windows.
- All actions are simulated.
