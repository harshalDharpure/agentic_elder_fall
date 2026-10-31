# SisFall Mamba backbone-swap campaign

## Purpose and boundaries

Replace the primary detector with the existing HARMamba-style selective state-space model in an isolated experiment, and retrain CNN–LSTM–Attention as a matched control. Keep the existing agentic architecture. This is a controlled backbone comparison, not the full sequential evidence-acquisition proposal and not an implementation of official `mamba-ssm` or Mamba-2.

No KFall runner is invoked. Existing dataset preparation, checkpoint files, shared memory aliases, configuration files, and historical results are not overwritten. Every campaign requires a new, empty output directory.

## Frozen experimental choices

- SisFall six channels; existing 90-sample / 10-sample-hop windows at 200 Hz, with existing recording-derived labels. This deliberately retains the historical task and its limitations while changing the backbone.
- Five outer subject folds, seed 42, inner subject-held-out validation fraction 0.15. Identical subject splits, sampled training/validation/test examples and optimizer budgets for both detectors.
- Existing protocol: up to 40 epochs, early stopping patience 10, 40,000 training windows, 5,000 validation windows, and the existing 1,000-window activity-enriched test slices. These distributions are benchmark slices, not free-living prevalence.
- Default HARMamba settings: width 64, three bidirectional selective-SSM blocks, state width 16, expansion 2, patch length 10. Compare at identical completed windows; do not claim per-sample causal state streaming for the bidirectional implementation.
- Evaluate detector-only threshold 0.5 through the ablation runner, and separately report validation-selected cost-sensitive threshold from the training runner. Never treat these as identical operating points. The cost is the assumed 10 FN + FP, not measured clinical utility.
- Evaluate `tier1_only`, `gate_only`, `gate_knn`, and `gate_knn_llm`. Use the real local LLM and text embedder; fail rather than substitute heuristic or random embeddings.
- Build fresh retrieval memory from final inner-training subjects only, disjoint from validation/test. Each memory has subject/index provenance. The current pipeline uses text evidence retrieval, so the trained detector is validated for loading but memory is not claimed to be learned sensor-embedding retrieval.
- Recalibrate the confidence gate for each detector. Preserve the existing gate objective for this controlled comparison; this is empirical calibration and does not provide a clinical risk certificate.
- Report five-fold mean/sample SD and paired F1 change in percentage points. Do not choose a winning model or tune hyperparameters using the new test results. Existing public benchmarks were previously explored, so results remain a controlled re-evaluation rather than untouched independent confirmation.

## Execution

Run from the repository root:

```bash
# Report missing data, GPU and local LLM prerequisites without training.
python scripts/run_sisfall_mamba.py --stage preflight --device cuda

# Actual tiny CPU training/checkpoint-reload/evaluation of both backbones.
# One fold, one epoch, 256 train / 128 validation / 128 test windows.
# Checks mechanics only; these scores are not research results.
python scripts/run_sisfall_mamba.py --stage smoke --device cpu

# Full isolated five-fold detector + agentic campaign when prerequisites pass.
python scripts/run_sisfall_mamba.py --stage full --device cuda
```

Optional `--out-dir` specifies a new directory. `--folds 0` runs a one-fold pilot, and `--models harmamba` restricts the campaign to Mamba (without a fresh matched CNN control). Explicit `--device cpu` permits CPU execution, but a full run can be very slow. Full runs do not automatically download models or start background services.

Outputs are under `results/sisfall_mamba/<stage>_<UTC timestamp>/`:

- `manifest.json`: stage/status, environment, source/config/data hashes, folds and failure details.
- `<model>/{tier1,protocol,agentic}.yaml`: effective isolated configuration snapshots.
- `<model>/checkpoints/`: checkpoints and validation-threshold detector metrics.
- `<model>/memory/`: independent fit-only memory and split manifest.
- `<model>/foldN/`: training, memory and evaluation logs plus predictions/ablation artifacts.
- `summary.json`: aggregates from exact requested fold files; smoke output explicitly marked.

If a step fails, its log and manifest remain available. No automatic resume is attempted, avoiding accidental reuse of mismatched checkpoints or memory; start a fresh directory after resolving the cause.

## Verification and next decisions

Run `python -m pytest -q tests/test_sisfall_mamba_campaign.py` and the existing core tests. Check memory split separation, checkpoint loading, output isolation, and matched-fold aggregation. Run the tiny real-data smoke before the full campaign.

Inspect all four modes. A stronger detector can still yield a weaker final system due to altered calibration and LLM routing. Report both outcomes. Longer contexts, corrected SisFall labels, normalization changes, a causal CNN–Mamba model, and a learned sequential agent require separate follow-up experiments to isolate their effects.

## Current environment limitation

Initial sandbox checks on 31 October 2026 found PyTorch 2.3.1+cu121 installed but CUDA inaccessible, and local Ollama sockets were blocked. The authorized host preflight subsequently passed: both CUDA and the required Ollama model are available outside the sandbox. Therefore the full campaign must execute with host access. A completed CPU smoke is not a completed five-fold scientific run. The host preflight manifest records actual availability.
