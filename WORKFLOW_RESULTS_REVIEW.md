# Workflow and stored-results review

Reviewed 30 October 2026. This review traces the data, model, decision, and evaluation paths and recomputes descriptive aggregates from existing fold JSON files. It does not retrain models or independently reproduce the historical experiments. Existing workspace changes were preserved.

## Main finding

The best stored sensor-only SisFall F1 is **0.772472**, from a fixed blend of a tabular specialist and the CNN–LSTM–Attention detector. The specialist is not an LLM. The sensor-only LLM stack has substantially lower F1 (**0.665351**) despite higher recall. On KFall, the sensor-only detector has F1 **0.987373**, while the LLM stack has **0.978999**. Neither dataset currently demonstrates a large sensor-only LLM improvement in F1.

## 1. Data preparation

### SisFall

- Entry points: `scripts/prepare_sisfall.py`, `src/agentic_fall/data/sisfall.py`.
- Reads the first six raw channels: acceleration XYZ and gyroscope XYZ. Scales acceleration to g and gyroscope to degrees/second.
- Generates 90-sample windows with hop 10 at 200 Hz: 0.45-second windows, 0.05-second hops.
- Every window inherits its recording's fall/ADL label from the filename. Consequently, a positive window need not contain an actual impact.
- Stored processed metadata: 1,549,503 windows, 524,881 positive, 1,024,622 negative, 38 subjects, 4,500 files.
- D18/D19 presence is checked, and evaluation sampling enforces activity quotas.

### KFall

- Entry points: `scripts/prepare_kfall.py`, `src/agentic_fall/data/kfall.py`.
- Reads CSV accelerometer and optional gyroscope channels. Tasks 20–34 are mapped to falls; other supported tasks to ADLs.
- Uses workbook onset/impact labels where available; otherwise estimates impact from acceleration changes. Fall recordings are cropped to the requested phase; ADL recordings retain their full signal.
- Main binary configuration uses transition segments, 100 Hz, window 90, hop 10: 0.9-second windows, 0.1-second hops. Short segments are zero-padded by shared windowing code.
- Stored metadata: 227,675 windows, 23,760 positive, 203,915 negative, 36 activity classes; workbook labels were available.
- These are segmented, offline benchmark results. Ground-truth-assisted cropping limits conclusions about continuous monitoring.

## 2. Splitting and training

- `data/splits.py` constructs five subject-independent folds, stratifying subjects by whether they have fall recordings.
- `eval/protocol.py` holds out approximately 15% of outer-training subjects for validation and checks disjoint train/validation/test subjects.
- `configs/paper_protocol.yaml` sets seed 42, training cap 40,000 windows, validation cap 5,000, and test cap 1,000 per fold.
- Sampling initially targets balanced positive/negative slices, then enforces D18/D19 quotas. The final class balance can change. Results do not represent natural activity prevalence.
- The paper training configuration uses 40 epochs, Adam-style training through the shared fit routine, learning rate 0.001, weight decay 0.0001, label smoothing, mixup, mixed precision, and early stopping. Standalone YAML defaults differ; the invoked runner and overrides determine the actual protocol.
- `models/cnn_lstm_attn.py`: Conv1D stem → two dilated convolution branches → stacked LSTMs (256, 128) → four-head self-attention → temporal mean pooling → two-class classifier. Embedding size is 128.
- The model factory also supports CNN1D, LSTM, CNN–LSTM, TCN, ModernTCN, TSMixer, PatchTST, Transformer, and HAR-Mamba comparisons.

## 3. Decision workflow

```text
Raw recordings → physical sensor channels → overlapping windows
  → subject split → train detector / validate / freeze checkpoint
  → p(fall) → calibrated confidence gate
      ├─ confident ADL/fall → direct detector decision
      └─ ambiguous → biomechanical evidence
          → MiniLM text embedding → kNN labelled training cases
          → local Ollama reasoner → optional adjudication/fusion
  → graded simulated response → window metrics → fold aggregation
```

- `agents/confidence_gate.py` routes low probabilities to ADL, high probabilities to fall, and the middle band to deliberation. Validation calibration balances confident-decision errors and a deferral penalty; it does not directly optimize observed end-to-end LLM performance.
- `features/biomechanics.py` computes freefall duration, peak impact, post-impact stillness, orientation change, duration, mean acceleration, and peak gyroscope magnitude.
- `agents/evidence.py` serializes these measurements and the detector probability. Current primary configuration disables activity codes.
- `agents/knn_memory.py` retrieves cosine-nearest cases, usually k=5, with MiniLM embeddings. Neighbour labels are legitimate only when memory partitioning is correct.
- `agents/llm_reasoner.py` supports Ollama and heuristic reasoning. Current locked configuration requests Mistral and disables silent heuristic fallback.
- `agents/pipeline.py` connects routing, retrieval, reasoning, optional fusion and adjudication. Some critique modes also screen confident fall predictions.
- `agents/action_agent.py` maps decisions and severity evidence to log/monitor/caregiver/emergency responses. Responses are simulated.
- `scripts/run_ablations.py` compares routing/retrieval/LLM/fusion/critique variants; `scripts/run_kfall_external.py` trains or loads KFall-specific checkpoints and evaluates the same broad stack. This KFall runner is within-dataset validation, not direct SisFall-to-KFall zero-shot transfer.

## 4. Separate specialist workflow

`scripts/run_specialist_judge_ladder.py` collects detector probability, kNN vote, seven biomechanical measurements, and interaction terms. `agents/specialist_judge.py` trains a HistGradientBoostingClassifier. It evaluates specialist-only, fixed blend, validation-tuned fusion, confidence-gated overrides, and an oracle diagnostic.

The best stored blend uses `0.5 * p_specialist + 0.5 * p_detector >= 0.5` across test windows. It does not call the LLM and is not restricted to ambiguous windows. Its recorded escalation flag should not be interpreted as proof that specialist computation was selective.

## 5. Sensor-only SisFall results

Recomputed from exactly one canonical file for each fold 0–4 under `results/noleak_lock/fold*/ablation_fold*.json` and `results/noleak_lock/specialist/fold*/specialist_fold*.json`. Values are fold means; ± denotes sample standard deviation, not confidence intervals. Each fold evaluates 1,000 windows. Cost is `10*FN + FP`, therefore also cost per 1,000 evaluated windows here.

| Method | F1 mean ± SD | Recall | Precision | Cost mean ± SD |
|---|---:|---:|---:|---:|
| Frozen CNN–LSTM–Attention | 0.767783 ± 0.018944 | 0.773344 | 0.762409 | 1086.6 ± 80.6 |
| Gate + kNN + LLM | 0.665351 ± 0.014509 | 0.934293 | 0.518175 | 664.2 ± 121.9 |
| Gate + kNN + LLM + action critique | 0.659790 ± 0.012015 | 0.945024 | 0.508490 | 637.6 ± 144.7 |
| Gate + kNN + LLM + fusion | 0.657960 ± 0.015642 | 0.953816 | 0.503414 | 611.0 ± 101.0 |
| Tabular specialist alone | 0.769379 ± 0.021385 | 0.766465 | 0.772477 | 1110.0 ± 98.1 |
| Fixed specialist/detector blend | **0.772472 ± 0.025168** | 0.773842 | 0.771152 | **1079.6 ± 115.3** |
| Validation-tuned specialist fusion | 0.757735 ± 0.021785 | 0.828251 | 0.698859 | 899.0 ± 153.7 |
| Cost-oriented specialist judge | 0.728691 ± 0.015576 | 0.852348 | 0.637094 | 850.6 ± 175.4 |

The fixed blend improves F1 by 0.004689 (0.469 percentage points), reduces mean cost by 0.64%, and beats the detector in three of five folds. This is modest descriptive evidence. Selection among many tested variants also limits confirmatory claims. Gate-only, kNN-only and LLM-only entries in this directory exist for fold 0 only; they are not complete five-fold ablations.

## 6. Sensor-only KFall results

Recomputed from `results/kfall/noleak_lock/kfall_external_fold0.json` through fold 4. Equivalent copies exist under `results/noleak_lock/kfall/`; these are not additional independent runs.

| Method | F1 mean ± SD | Recall | Precision | Accuracy | Cost mean ± SD |
|---|---:|---:|---:|---:|---:|
| Frozen CNN–LSTM–Attention | **0.987373 ± 0.004725** | 0.991279 | 0.983539 | 0.987800 | 50.0 ± 35.5 |
| Gate + kNN + LLM | 0.978999 ± 0.009187 | **0.995028** | 0.963736 | 0.979400 | **42.2 ± 39.0** |

Across all five 1,000-window slices, detector FP/FN totals are 40/21; stack totals are 91/12. The stack prevents nine misses while adding 51 false alarms. Under the chosen 10:1 error cost this lowers mean cost by 15.6%, but reduces F1 by 0.837 percentage points. Average gate escalation is 1.46%, versus 51.1% in the SisFall LLM runs.

## 7. Newer activity-conditioned artifacts

The newest substantive result JSON files found are from 20 October; 26 October files under `_test_kfold_agg` are test fixtures. The sensor-only fold files above are from 18 October. File modification time is evidence of artifact recency, not a guaranteed experiment execution timestamp.

The later high-F1 campaign explicitly uses `configs/agentic_high_f1.yaml` and activity information. These results must be kept separate from sensor-only experiments:

| Track | Detector F1 | Decision-layer F1 | Interpretation |
|---|---:|---:|---|
| SisFall activity-on guarded | 0.767783 | 0.820135 | Activity-conditioned critique variant |
| SisFall activity-on soft guard | 0.767783 | 0.778092 | Activity-conditioned kNN/LLM variant |
| KFall CNN–LSTM–Attention high-F1 | 0.987373 | 0.988220 | Activity-conditioned campaign |
| KFall HAR-Mamba high-F1 | 0.990512 | 0.988518 | Activity-conditioned campaign |

Historical package notes select SisFall F1 0.886579. Because F/D activity prefixes reveal the class, that value cannot establish sensor-only fall detection performance. Current top-level submission tables use 0.772/0.768, but older claim-strategy and decision-memo files retain conflicting historical claims.

The latest literature-protocol summary reports trial F1 0.915019 after searching thresholds on test labels. At the fixed 0.5 threshold, trial F1 is 0.582102; dense-window F1 is 0.716735. The 0.915 value is exploratory test-optimized performance, not a held-out estimate and not comparable to the capped dense-window table. Impact-label width and threshold sweeps also use test labels.

## 8. Code and evidence limitations

1. **Validation-memory overlap:** `scripts/build_memory.py` uses the complete outer-training fold, including subjects subsequently reserved for validation by `make_fold_split`. Test subjects are excluded, but retrieval memory is not isolated from validation. Rebuild memory from the final inner-training indices before claiming clean validation calibration. Specialist training features can also retrieve training cases without explicit self-exclusion.
2. **Validation reuse in judge variants:** the specialist ladder can train `clf_esc` on escalated validation examples and tune its confidence threshold on that same validation set. This affects judge variants; the fixed blend uses the separately trained specialist.
3. **KFall physical units and rate:** the CSV loader does not convert gyroscope units, although biomechanics assumes degrees/second. The external runner passes the agentic evidence rate (200 Hz in the shared configuration), rather than overriding it with KFall's 100 Hz. Historical run-specific configuration must be checked before attributing duration-sensitive effects.
4. **Unavailable temporal context:** a 0.45-second SisFall window cannot establish the configured 3- or 10-second stillness criteria. Meaningful long-duration action logic requires state across windows.
5. **Artifact provenance:** KFall output JSON omits an activity flag and configuration hash. Campaign scripts identify the intended protocol, but JSON alone cannot independently prove it. Existing memories are reused if present. Saved configuration, split/memory manifests, code revision and prediction IDs are needed for stronger reproducibility.
6. **Duplicate and mixed outputs:** several runners write both nested fold outputs and top-level copies. Recursive aggregation can double-count these. Some ablation payloads combine rows carrying different protocol tags. This review uses five exact canonical paths per track.
7. **Metric implementation:** `expected_response_cost` is an unnormalized count-based cost; it is per 1,000 only for 1,000-window slices. `mean_llm_calls_escalated` divides by adjudication count rather than escalation count in `run_ablations.py`, so that field can be misleading. Multiclass evaluation currently stores macro F1 under precision and recall as well; binary tables here are unaffected.
8. **Leakage audit scope:** calling `audit_prompt_leakage.py` without explicit inputs scans no artifacts. Its result scanner accesses `.get` before checking list-shaped JSON, despite advertising support for memory-style lists. A passing default invocation is not certification of the complete experiment.
9. **Claims:** oracle specialist F1 0.796972 uses test truth to choose predictions and is an upper-bound diagnostic only. Neither window-level accuracy nor simulated actions establish event-level sensitivity, false alarms/hour, or clinical readiness.

## 9. Recommended next experiment

Freeze a sensor-only protocol, rebuild memory using final inner-training subjects only, verify KFall units/rate, and store configuration hashes and split manifests. Compare the detector, validation-tuned detector threshold, tabular specialist blend, and LLM stack on identical untouched test windows. Select the operating point before test evaluation. Add temporal aggregation and event-level recall/false alarms per hour before making monitoring claims.
