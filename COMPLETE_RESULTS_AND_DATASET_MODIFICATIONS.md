# Complete experimental results and dataset modifications

Self-contained index of **all protocols, dataset manipulations, and numeric results** for the RCDP / agentic fall-detection project.

- Project root: `/DATA/tauseef_2121cs04/harshal/agentic_fall/`
- Generated for GitHub upload / paper comparison
- Paper primary claims use the **enriched SisFall n=1000** protocol (§3) and the **ambiguous stress n=300** bench (§4)

---

## 0. Shared corpus baseline (before protocol caps)

| Item | Setting |
|------|---------|
| SisFall processed | `data/processed/sisfall/windows_w90_h10.npz` |
| Subjects | All 38 SisFall subjects (SA + SE); **no code-level elderly exclusion** |
| Channels / rate | 6 (ADXL345 + ITG3200), 200 Hz |
| Window / hop | **90 / 10** (~0.45 s windows, 50 ms stride) |
| Approx. windows | ~1.55×10⁶ |
| Labels | Binary Fall (F*) vs ADL (D*); D18/D19 remain **ADL (y=0)** |
| D18 | Stumble while walking (~27.8k windows) |
| D19 | Gently jump / reach high object (~27.8k windows); **operational stress ADL**, not clinical near-fall |
| Folds | 5 subject-independent; stratified by fall-capable vs ADL-only subjects |
| Cost model | `R = 10·FN + FP` (unless noted) |
| Seed (primary) | 42 |

### Dataset constraints (from SisFall authors / paper disclosure — not pipeline filters)

- Elderly subjects generally did **not** perform falls or D06/D13/D18/D19 (medical advice).
- SE06 (60y judo) is an exception.
- **No age-stratified metrics** were produced.
- **D05/D11 elderly-feasible proxy** was planned, **not run**.
- Enrichment is done by **eval/train subsampling quotas**, not by rewriting the NPZ.

### KFall shared corpus

| Item | Setting |
|------|---------|
| Processed | `data/processed/kfall/windows_transition_w90_h10.npz` |
| Phase used | **transition only** (pre/post supported in prepare, not on disk for main runs) |
| Windows | ~227.7k; falls ~23.8k |
| Rate | 100 Hz source resampled into same windowing |

---

## 1. Protocol matrix (what we changed)

| # | Protocol name | Prevalence / sampling | D18/D19 treatment | max_train | max_test | Config | Results dir |
|---|---------------|----------------------|-------------------|-----------|----------|--------|-------------|
| P1 | Paper main (enriched) | Stratified ~50/50 in caps | Force ≥80/code in test | 40000 | **1000** | `configs/paper_protocol.yaml` | `results/fold{0..4}/` |
| P2 | Paper archive | Same enrichment | Same | 40000 | **4000** | paper protocol | `results/fold*/ablation_*full4000*` |
| P3 | Ambiguous stress SisFall | Forced 100 fall / 100 D18D19 / 100 other ADL | **100** D18/D19 | — | **300** fixed | `scripts/build_kfold_ambiguous_bench.py` | `results/kfold_ambiguous*` |
| P4 | Standard / E1 natural | **Natural** (~0.33 fall) | **None** (no force-in) | eval-only | 20000 | `standard_sisfall.yaml` | `results/standard_protocol/` |
| P5 | Retrain v1 | Natural eval; train had stratified bug (50/50) | None | 120000 | 20000 | `standard_retrain.yaml` | `results/standard_retrain/` + `standard_protocol_retrained/` |
| P6 | G1 v2 full-data | Natural train+eval | None | **null (full)** | 30000 | `standard_g1_v2*.yaml` | `results/standard_protocol_g1_v2/` |
| P7 | R1 collapse table | Compare P4 vs P1 detector F1 | Near-fall column incomplete | — | — | `scripts/build_r1_collapse.py` | `results/r1_collapse/` |
| P8 | Seed sweep Fold0 | Same as P1; redraw stratified slice | Same as P1 | — | 1000 | seeds 42/43/44 | `results/seed_sweep/` |
| P9 | Sensor robustness | P1 + ±15° IMU rotation at eval | — | — | — | `run_astar_experiments.py` | `results/astar/` |
| P10 | G2 feature adjudicators | Escalated feature cases only | — | — | 160/fold | `run_adjudicator_compare.py` | `results/adjudicator_compare/` |
| P11 | LLM shared-ID compare | Shared escalated case IDs | — | — | 150/fold | `run_llm_compare.py` | `results/fold*/llm_compare*` |
| P12 | KFall binary external | Paper-like caps on KFall transition | remapped D18/D19 quota in slice | 40000 | 1000 | `tier1_kfall_binary.yaml` | `results/kfall/` |
| P13 | KFall multiclass | 36-class transition | — | — | 4000 (fold0) | `tier1_kfall.yaml` | `results/kfall/kfall_multiclass_fold0.json` |
| P14 | KFall ambiguous stress | Same 100/100/100 on KFall | KFall D18/D19 | — | 300 | `build_kfall_ambiguous_bench.py` | `results/kfold_ambiguous_kfall*` |

### What we did **not** remove / filter in code

- Did **not** drop elderly (SE*) subjects from the NPZ.
- Did **not** remove D18/D19 from training (still ADL).
- Did **not** re-window the corpus between protocols (always w90/h10).
- “Removed” windows only means: **not selected** into a capped eval/train slice.

---

## 2. Paper main protocol (P1) — enriched SisFall n=1000

**Modification:** stratified subsample + D18/D19 quota ≥80/code → fall prevalence ≈ **0.433** (not natural ~0.33).

### 2.1 Main ablation means (from supplied confusion counts / disk)

| Mode | F1 | Recall | Precision | cost/1000 | esc |
|------|-----|--------|-----------|-----------|-----|
| tier1_only | 0.759 ± 0.014 | 0.759 | 0.760 | 1150.8 | 0.000 |
| gate_only | 0.759 ± 0.014 | 0.759 | 0.760 | 1150.8 | 0.468 |
| gate_knn | 0.753 ± 0.012 | 0.723 | 0.787 | 1286.0 | ~0.46 |
| gate_llm | 0.702 ± 0.021 | 0.895 | 0.578 | 740.4 | ~0.47 |
| **gate_knn_llm (stack)** | **0.887 ± 0.029** | **0.977** | **0.813** | **199.8** | ~0.47 |

### 2.2 Per-fold main ablation (disk `results/fold*/ablation_fold*.json`)

| Fold | Mode | F1 | Rec | Prec | Acc | TP | FN | FP | TN |
|------|------|-----|-----|------|-----|----|----|----|----|
| 0 | cnn_lstm_attn | 0.774 | 0.781 | 0.767 | 0.804 | 335 | 94 | 102 | 469 |
| 0 | tier1_only | 0.774 | 0.781 | 0.767 | 0.804 | 335 | 94 | 102 | 469 |
| 0 | gate_only | 0.774 | 0.781 | 0.767 | 0.804 | 335 | 94 | 102 | 469 |
| 0 | gate_knn | 0.771 | 0.755 | 0.786 | 0.807 | 324 | 105 | 88 | 483 |
| 0 | gate_llm | 0.717 | 0.911 | 0.591 | 0.691 | 391 | 38 | 271 | 300 |
| 0 | gate_knn_llm | 0.892 | 0.986 | 0.815 | 0.898 | 423 | 6 | 96 | 475 |
| 1 | cnn_lstm_attn | 0.737 | 0.748 | 0.727 | 0.767 | 327 | 110 | 123 | 440 |
| 1 | tier1_only | 0.737 | 0.748 | 0.727 | 0.767 | 327 | 110 | 123 | 440 |
| 1 | gate_only | 0.737 | 0.748 | 0.727 | 0.767 | 327 | 110 | 123 | 440 |
| 1 | gate_knn | 0.744 | 0.714 | 0.776 | 0.785 | 312 | 125 | 90 | 473 |
| 1 | gate_llm | 0.699 | 0.849 | 0.594 | 0.680 | 371 | 66 | 254 | 309 |
| 1 | gate_knn_llm | 0.851 | 0.911 | 0.799 | 0.861 | 398 | 39 | 100 | 463 |
| 2 | cnn_lstm_attn | 0.755 | 0.783 | 0.729 | 0.780 | 339 | 94 | 126 | 441 |
| 2 | tier1_only | 0.755 | 0.783 | 0.729 | 0.780 | 339 | 94 | 126 | 441 |
| 2 | gate_only | 0.755 | 0.783 | 0.729 | 0.780 | 339 | 94 | 126 | 441 |
| 2 | gate_knn | 0.743 | 0.746 | 0.739 | 0.776 | 323 | 110 | 114 | 453 |
| 2 | gate_llm | 0.690 | 0.896 | 0.561 | 0.651 | 388 | 45 | 304 | 263 |
| 2 | gate_knn_llm | 0.874 | 0.998 | 0.777 | 0.875 | 432 | 1 | 124 | 443 |
| 3 | cnn_lstm_attn | 0.766 | 0.732 | 0.804 | 0.805 | 320 | 117 | 78 | 485 |
| 3 | tier1_only | 0.766 | 0.732 | 0.804 | 0.805 | 320 | 117 | 78 | 485 |
| 3 | gate_only | 0.766 | 0.732 | 0.804 | 0.805 | 320 | 117 | 78 | 485 |
| 3 | gate_knn | 0.748 | 0.673 | 0.842 | 0.802 | 294 | 143 | 55 | 508 |
| 3 | gate_llm | 0.729 | 0.904 | 0.611 | 0.706 | 395 | 42 | 252 | 311 |
| 3 | gate_knn_llm | 0.930 | 0.995 | 0.873 | 0.935 | 435 | 2 | 63 | 500 |
| 4 | cnn_lstm_attn | 0.760 | 0.749 | 0.772 | 0.797 | 322 | 108 | 95 | 475 |
| 4 | tier1_only | 0.760 | 0.749 | 0.772 | 0.797 | 322 | 108 | 95 | 475 |
| 4 | gate_only | 0.760 | 0.749 | 0.772 | 0.797 | 322 | 108 | 95 | 475 |
| 4 | gate_knn | 0.758 | 0.728 | 0.790 | 0.800 | 313 | 117 | 83 | 487 |
| 4 | gate_llm | 0.675 | 0.914 | 0.535 | 0.622 | 393 | 37 | 341 | 229 |
| 4 | gate_knn_llm | 0.887 | 0.993 | 0.801 | 0.891 | 427 | 3 | 106 | 464 |

### 2.3 Extended ablation aggregates (`aggregate_ablation_mean_std.csv`)

| System | F1 | Rec | Acc | cost/1k | esc |
|--------|-----|-----|-----|---------|-----|
| gate_knn_llm_action_critic | 0.892±0.000 | 0.986±0.000 | 0.898 | 156.0 | 0.443 |
| gate_knn_llm_action_critique | 0.891±0.022 | 0.971±0.032 | 0.897 | 214.6 | 0.448 |
| gate_knn_llm_crc_veto | 0.891±0.022 | 0.971±0.032 | 0.897 | 214.8 | 0.448 |
| gate_knn_llm_contrastive_action_critique | 0.878±0.020 | 0.971±0.032 | 0.883 | 228.4 | 0.448 |
| gate_knn_llm | 0.877±0.071 | 0.974±0.031 | 0.895 | 208.0 | 0.457 |
| gate_knn_llm_actor_critic_roleswap | 0.856±0.000 | 0.914±0.000 | 0.868 | 465.0 | 0.443 |
| gate_knn_actor_critic | 0.771±0.000 | 0.755±0.000 | 0.807 | 1138.0 | 0.443 |
| gate_knn_actor_critic_roleswap | 0.771±0.000 | 0.755±0.000 | 0.807 | 1138.0 | 0.443 |
| gate_knn_actor_only | 0.771±0.000 | 0.755±0.000 | 0.807 | 1138.0 | 0.443 |
| gate_knn | 0.759±0.013 | 0.735±0.030 | 0.799 | 1230.5 | 0.459 |
| gate_only | 0.759±0.014 | 0.759±0.022 | 0.791 | 1150.8 | 0.468 |
| tier1_only | 0.759±0.014 | 0.759±0.022 | 0.791 | 1150.8 | 0.000 |
| cnn_lstm_attn | 0.754±0.073 | 0.770±0.018 | 0.797 | 0.0 | 0.000 |
| gate_knn_llm_actor_critic | 0.704±0.133 | 0.730±0.015 | 0.804 | 1105.2 | 0.476 |
| gate_llm | 0.702±0.020 | 0.895±0.025 | 0.670 | 740.4 | 0.468 |
| gate_knn_llm_actor_only | 0.661±0.230 | 0.959±0.047 | 0.831 | 326.5 | 0.531 |

> Note: some extended rows are single-fold / reruns. Paper primary uses the five-fold main ladder above, not the noisier 0.877±0.071 extended `gate_knn_llm` aggregate.

### 2.4 Agentic timing run (same protocol; latency/duty-cycle)

| Fold | F1 | Rec | Prec | Acc | Spec | esc | cost | FAR_NF | lat_mean_ms |
|------|-----|-----|------|-----|------|-----|------|--------|-------------|
| 0 | 0.892 | 0.986 | 0.815 | 0.898 | 0.832 | 0.443 | 156.0 | 0.175 | 5678.8 |
| 1 | 0.850 | 0.911 | 0.798 | 0.860 | 0.821 | 0.349 | 491.0 | 0.156 | 3726.6 |
| 2 | 0.874 | 0.998 | 0.777 | 0.875 | 0.781 | 0.460 | 134.0 | 0.200 | 10515.9 |
| 3 | 0.931 | 0.995 | 0.875 | 0.936 | 0.890 | 0.544 | 82.0 | 0.081 | 5985.8 |
| 4 | 0.885 | 0.993 | 0.798 | 0.889 | 0.811 | 0.546 | 138.0 | 0.175 | 5931.7 |

**Mean:** F1=0.887±0.030, Rec=0.977, esc=0.468, cost/1k=200.2, lat_tier1=8.56 ms, lat_esc=13635 ms

Classification vs timing: keep F1/Rec from main ablation counts; use this run for latency. FP/TN differ slightly in folds 1/3/4.

---

## 3. Ambiguous / near-fall stress bench SisFall (P3) — n=300

**Modification:** per fold force **100 D18/D19 + 100 falls + 100 other ADLs** from test subjects only. Near-fall ADLs ≈ 33% (heavily oversampled vs free-living).

| Mode | F1 | Rec | Acc | esc | cost/win |
|------|-----|-----|-----|-----|----------|
| tier1_only | 0.713 ± 0.024 | 0.768 ± 0.020 | 0.793 ± 0.024 | 0.000 ± 0.000 | 0.903 ± 0.065 |
| gate_knn_llm | 0.896 ± 0.019 | 0.976 ± 0.028 | 0.924 ± 0.016 | 0.594 ± 0.042 | 0.148 ± 0.085 |

| Fold | Mode | F1 | Rec | Acc | esc | cost/win |
|------|------|-----|-----|-----|-----|----------|
| 0 | tier1_only | 0.696 | 0.790 | 0.770 | 0.000 | 0.860 |
| 1 | tier1_only | 0.738 | 0.790 | 0.813 | 0.000 | 0.817 |
| 2 | tier1_only | 0.682 | 0.750 | 0.767 | 0.000 | 0.983 |
| 3 | tier1_only | 0.714 | 0.760 | 0.797 | 0.000 | 0.923 |
| 4 | tier1_only | 0.735 | 0.750 | 0.820 | 0.000 | 0.930 |
| 0 | gate_knn_llm | 0.868 | 0.990 | 0.900 | 0.587 | 0.130 |
| 1 | gate_knn_llm | 0.894 | 0.930 | 0.927 | 0.523 | 0.283 |
| 2 | gate_knn_llm | 0.892 | 0.990 | 0.920 | 0.627 | 0.110 |
| 3 | gate_knn_llm | 0.902 | 0.970 | 0.930 | 0.613 | 0.160 |
| 4 | gate_knn_llm | 0.922 | 1.000 | 0.943 | 0.620 | 0.057 |

**Paper lead:** 0.713 → 0.896 F1; cost/win 0.903 → 0.148 (~84%).

---

## 4. Standard / E1 natural-prevalence zoo (P4)

**Modification vs paper:** `stratified=false`, **no** D18/D19 force-in, `max_test=20000`, Youden threshold. Prevalence ≈ 0.33–0.35.

| Backbone | F1 | Rec | Spec | Acc | per-fold F1 |
|----------|-----|-----|------|-----|-------------|
| modern_tcn | 0.742 ± 0.015 | 0.745 | 0.867 | 0.826 | 0.740 0.719 0.746 0.759 0.749 |
| cnn_lstm_attn | 0.718 ± 0.022 | 0.756 | 0.820 | 0.799 | 0.728 0.679 0.716 0.733 0.731 |
| tcn | 0.702 ± 0.029 | 0.733 | 0.820 | 0.791 | 0.705 0.687 0.663 0.738 0.717 |
| cnn1d | 0.701 ± 0.031 | 0.751 | 0.799 | 0.783 | 0.724 0.663 0.686 0.740 0.690 |
| cnn_lstm | 0.696 ± 0.010 | 0.741 | 0.803 | 0.782 | 0.701 0.699 0.679 0.698 0.705 |
| lstm | 0.693 ± 0.016 | 0.720 | 0.818 | 0.785 | 0.678 0.680 0.688 0.717 0.701 |
| transformer | 0.690 ± 0.012 | 0.742 | 0.792 | 0.775 | 0.689 0.681 0.676 0.695 0.708 |
| tsmixer | 0.671 ± 0.022 | 0.671 | 0.834 | 0.779 | 0.675 0.651 0.649 0.679 0.701 |
| patchtst | 0.541 ± 0.009 | 0.627 | 0.647 | 0.641 | 0.538 0.532 0.534 0.547 0.552 |

**G1 gate (F1≥0.95): all FAIL.** Best modern_tcn 0.742.

---

## 5. Retrained zoo under natural prevalence (P5)

**Modification:** retrain detectors for natural-prevalence eval. Early retrain used stratified train subsample (≈50/50) vs natural test — prevalence mismatch. Eval results below.

| Backbone | F1 | Rec | Spec | per-fold F1 |
|----------|-----|-----|------|-------------|
| modern_tcn | 0.760 ± 0.030 | 0.794 | 0.848 | 0.779 0.718 0.737 0.786 0.778 |
| tcn | 0.747 ± 0.034 | 0.798 | 0.827 | 0.758 0.693 0.735 0.779 0.768 |
| cnn_lstm_attn | 0.739 ± 0.031 | 0.769 | 0.840 | 0.751 0.687 0.743 0.766 0.748 |
| cnn1d | 0.734 ± 0.026 | 0.772 | 0.831 | 0.731 0.691 0.756 0.749 0.743 |
| cnn_lstm | 0.724 ± 0.036 | 0.784 | 0.804 | 0.761 0.693 0.742 0.744 0.679 |
| lstm | 0.711 ± 0.026 | 0.771 | 0.797 | 0.725 0.668 0.724 0.733 0.707 |
| transformer | 0.703 ± 0.017 | 0.751 | 0.804 | 0.689 0.683 0.702 0.719 0.720 |
| tsmixer | 0.686 ± 0.018 | 0.703 | 0.825 | 0.680 0.667 0.692 0.678 0.715 |
| patchtst | 0.566 ± 0.015 | 0.679 | 0.632 | 0.571 0.543 0.582 0.566 0.567 |

Still ≪ 0.95.

---

## 6. G1 v2 full-data natural prevalence (P6) — cnn_lstm_attn

**Modification:** `max_train=null` (full data), `subsample_stratified=false`, `max_test=30000`, Youden. Fixes train/test prevalence bug.

| Fold | Youden F1 | Rec | Spec | Prec | Acc | thr | Cost F1 | Cost Rec |
|------|-----------|-----|------|------|-----|-----|---------|----------|
| 0 | 0.760 | 0.853 | 0.809 | 0.685 | 0.823 | 0.347 | 0.675 | 0.959 |
| 1 | 0.683 | 0.763 | 0.756 | 0.617 | 0.758 | 0.401 | 0.615 | 0.913 |
| 2 | 0.768 | 0.775 | 0.879 | 0.761 | 0.845 | 0.563 | 0.686 | 0.916 |
| 3 | 0.783 | 0.850 | 0.838 | 0.726 | 0.842 | 0.374 | 0.699 | 0.940 |
| 4 | 0.761 | 0.797 | 0.842 | 0.728 | 0.826 | 0.500 | 0.644 | 0.950 |
| **Mean±std** | **0.751 ± 0.039** | **0.808 ± 0.042** | 0.825 ± 0.046 | 0.703 ± 0.055 | 0.819 ± 0.035 | — | 0.664 ± 0.034 | 0.936 ± 0.020 |

**G1 FAIL** (need ≥0.95). Dense SI windows are harder than many published SisFall setups.

---

## 7. R1 collapse: standard vs enriched detector F1 (P7)

**Modification compared:** same backbones under natural (P4) vs enriched main-slice context (detector-only summaries). Near-fall-specific F1 column was **not completed** (null).

| Backbone | Standard F1 | Enriched F1 | Near-fall F1 |
|----------|-------------|-------------|--------------|
| cnn1d | 0.701±0.031 | 0.705±0.015 | null |
| lstm | 0.693±0.016 | 0.704±0.008 | null |
| cnn_lstm | 0.696±0.010 | 0.700±0.006 | null |
| cnn_lstm_attn | 0.718±0.022 | 0.715±0.005 | null |
| tcn | 0.702±0.029 | 0.705±0.021 | null |
| modern_tcn | 0.742±0.015 | 0.734±0.012 | null |
| tsmixer | 0.671±0.022 | 0.688±0.007 | null |
| patchtst | 0.541±0.009 | 0.672±0.003 | null |
| transformer | 0.690±0.012 | 0.683±0.010 | null |

Notable: PatchTST jumps 0.541 → 0.672 under enriched context; most others change little.

---

## 8. G2 feature adjudicators on escalated windows (P10)

**Modification:** evaluate classical feature models only on escalated cases (n=160/fold), **not** the LLM case set.

| Method | F1 | Rec | Acc | Prec |
|--------|-----|-----|-----|------|
| gbm | 0.780 ± 0.046 | 0.831 ± 0.040 | 0.680 ± 0.058 | 0.737 ± 0.060 |
| logreg | 0.664 ± 0.102 | 0.599 ± 0.129 | 0.596 ± 0.080 | 0.752 ± 0.074 |
| heuristic | 0.039 ± 0.011 | 0.020 ± 0.006 | 0.320 ± 0.069 | 0.767 ± 0.224 |
| always_fall | 0.814 ± 0.044 | 1.000 ± 0.000 | 0.689 ± 0.063 | 0.689 ± 0.063 |
| majority_train | 0.814 ± 0.044 | 1.000 ± 0.000 | 0.689 ± 0.063 | 0.689 ± 0.063 |
| random | 0.605 ± 0.057 | 0.536 ± 0.041 | 0.520 ± 0.056 | 0.696 ± 0.093 |
| always_adl | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.311 ± 0.063 | 0.000 ± 0.000 |

Paper costs (per 1000 escalated): GBM ≈1366; always-fall ≈311. **Different cases than LLM compare — no matched superiority claim.**

---

## 9. LLM shared-ID adjudicators (P11) — n=150/fold

**Modification:** same escalated case IDs across backends.

| Fold | Model | F1 | Rec | Prec | Acc | Spec | cost |
|------|-------|-----|-----|------|-----|------|------|
| 0 | heuristic | 0.042 | 0.021 | 1.000 | 0.387 | 1.000 | 920.0 |
| 0 | mistral:latest | 0.959 | 1.000 | 0.922 | 0.947 | 0.857 | 8.0 |
| 0 | qwen2.5-coder:7b-instruct | 0.807 | 1.000 | 0.676 | 0.700 | 0.196 | 45.0 |
| 1 | heuristic | 0.105 | 0.056 | 1.000 | 0.320 | 1.000 | 1020.0 |
| 1 | mistral:latest | 0.977 | 1.000 | 0.956 | 0.967 | 0.881 | 5.0 |
| 1 | qwen2.5-coder:7b-instruct | 0.874 | 1.000 | 0.777 | 0.793 | 0.262 | 31.0 |
| 2 | heuristic | 0.000 | 0.000 | 0.000 | 0.347 | 0.981 | 971.0 |
| 2 | mistral:latest | 0.975 | 1.000 | 0.951 | 0.967 | 0.906 | 5.0 |
| 2 | qwen2.5-coder:7b-instruct | 0.819 | 1.000 | 0.693 | 0.713 | 0.189 | 43.0 |
| 3 | heuristic | 0.021 | 0.011 | 1.000 | 0.387 | 1.000 | 920.0 |
| 3 | mistral:latest | 0.984 | 1.000 | 0.969 | 0.980 | 0.947 | 3.0 |
| 3 | qwen2.5-coder:7b-instruct | 0.802 | 1.000 | 0.669 | 0.693 | 0.193 | 46.0 |
| 4 | heuristic | 0.000 | 0.000 | 0.000 | 0.340 | 0.981 | 981.0 |
| 4 | mistral:latest | 0.966 | 1.000 | 0.933 | 0.953 | 0.865 | 7.0 |
| 4 | qwen2.5-coder:7b-instruct | 0.813 | 1.000 | 0.685 | 0.700 | 0.135 | 45.0 |

| Model | F1 mean | Rec | Acc | cost/1k |
|-------|---------|-----|-----|---------|
| mistral:latest | 0.972±0.010 | 1.000 | 0.963 | 37.3 |
| qwen2.5-coder:7b-instruct | 0.823±0.029 | 1.000 | 0.720 | 280.0 |
| heuristic | 0.034±0.044 | 0.018 | 0.356 | 6416.0 |

| Matched constant control (derived from LLM set class totals) | F1 | cost/1k |
|-------------------------------------------------------------|-----|---------|
| always-fall | 0.790 ± 0.028 | 346.7 |

Mistral 0.972 / 37.3 vs always-fall 0.790 / 346.7.

---

## 10. Seed sweep Fold 0 (P8) — stratified slice redraw

**Modification:** seeds {42,43,44} redraw the n=1000 stratified slice + train randomness (paper protocol).

| Seed | Tier-1 F1 | Tier-1 Rec | Stack F1 | Stack Rec | Stack cost/1k | τ_low | τ_high |
|------|-----------|------------|----------|-----------|---------------|-------|--------|
| 42 | 0.747 | 0.716 | 0.909 | 0.998 | 95.0 | 0.071 | 0.550 |
| 43 | 0.740 | 0.710 | 0.907 | 0.995 | 105.0 | 0.113 | 0.550 |
| 44 | 0.746 | 0.725 | 0.899 | 1.000 | 96.0 | 0.071 | 0.550 |
| **Mean** | 0.745±0.004 | — | 0.905±0.005 | — | — | — | — |

---

## 11. KFall binary external validation (P12)

**Modification:** switch corpus to KFall transition binary; paper-like caps; remapped D18/D19 appear in slices (n_d18=n_d19=40/fold in summaries).

| Mode | F1 mean±std | Rec mean | cost/1000 mean | esc mean |
|------|-------------|----------|----------------|----------|
| gate_knn | 0.991 | 0.994 | 36.0 | 0.025 |
| gate_knn_llm | 0.990 ± 0.004 | 0.996 | 25.8 | 0.011 |
| gate_llm | 0.976 | 0.998 | 33.0 | 0.025 |
| gate_only | 0.992 | 0.998 | 17.0 | 0.025 |
| tier1_only | 0.987 ± 0.005 | 0.991 | 50.0 | 0.000 |

Much easier than SisFall; escalation ~0.01 for stack. Small ΔF1 (~0.987→0.990).

---

## 12. KFall multiclass Fold 0 (P13) — context only

**Modification:** 36-class transition classification (not binary). **Not** a claim vs Bhatti published multi-phase tables.

| Metric | Value |
|--------|-------|
| n | 4000 |
| n_classes | 36 |
| accuracy | 0.585 |
| macro_f1 | 0.541 |
| weighted_f1 | 0.583 |

---

## 13. KFall ambiguous stress (P14) — n=300

**Modification:** same 100/100/100 composition on KFall D18/D19 codes.

| Mode | F1 | Rec | Acc | esc |
|------|-----|-----|-----|-----|
| tier1 (heuristic backend file) | 0.968 ± 0.017 | 0.994 ± 0.005 | 0.978 ± 0.012 | 0.000 ± 0.000 |
| stack (ollama) | 0.983 ± 0.013 | 0.998 ± 0.004 | 0.989 ± 0.009 | 0.337 ± 0.005 |

| Fold | Tier1 F1 | Stack F1 | Stack esc |
|------|----------|----------|-----------|
| 0 | 0.966 | 0.990 | 0.347 |
| 1 | 0.985 | 1.000 | 0.333 |
| 2 | 0.947 | 0.976 | 0.337 |
| 3 | 0.985 | 0.966 | 0.337 |
| 4 | 0.957 | 0.985 | 0.333 |

Much easier than SisFall stress; not the paper lead result.

---

## 14. Other protocol tweaks (P9 and related)

### Sensor robustness (±15° rotation)

Eval-time IMU rotation; results in `results/astar/robustness_fold*.json` and `ASTAR_EXPERIMENT_SUMMARY.md`.

Summary keys: experiments

```json
{
  "experiments": [
    {
      "file": "conformal_gate_audit.json",
      "data": {
        "rows": [
          {
            "fold": 0,
            "alpha": 0.05,
            "q_hat": 0.830179750919342,
            "conformal_f1": 0.7736720554272517,
            "conformal_recall": 0.7808857808857809,
            "conformal_cost": 1042.0,
            "conformal_escalation_rate": 0.579,
            "edge_fn_rate": 0.0166270783847981,
            "edge_n": 421,
            "set_size_hist": {
              "0": 0,
              "1": 421,
              "2": 579
            },
            "cost_gate_f1": 0.7736720554272517,
            "cost_gate_escalation_rate": 0.383
          },
          {
            "fold": 0,
            "alpha": 0.01,
            "q_hat": 0.9014959931373596,
            "conformal_f1": 0.7736720554272517,
            "conformal_recall": 0.7808857808857809,
            "conformal_cost": 1042.0,
            "conformal_escalation_rate": 0.767,
            "edge_fn_rate": 0.008583690987124463,
            "edge_n": 233,
            "set_size_hist": {
              "0": 0,
              "1": 233,
              "2": 767
            },
            "cost_gate_f1": 0.7736720554272517,
            "cost_gate_escalation_rate": 0.383
          },
          {
            "fold": 0,
            "alpha": 0.005,
            "q_hat": 0.9105221033096313,
            "conformal_f1": 0.7736720554272517,
            "conformal_recall": 0.7808857808857809,
            "conformal_cost": 1042.0,
            "conformal_escalation_rate": 0.798,
            "edge_fn_rate": 0.0049504950495049506,
            "edge_n": 202,
            "set_size_hist": {
              "0": 0,
              "1": 202,
              "2": 798
            },
            "cost_gate_f1": 0.7736720554272517,
            "cost_gate_escalation_rate": 0.383
          },
          {
            "fold": 1,
            "alpha": 0.05,
            "q_hat": 0.926355242729187,
            "conformal_f1": 0.7373167981961668,
            "conformal_recall": 0.7482837528604119,
            "conformal_cost": 1223.0,
            "conformal_escalation_rate": 0.834,
            "edge_fn_rate": 0.030120481927710843,
            "edge_n": 166,
            "set_size_hist": {
              "0": 0,
              "1": 166,
              "2": 834
            },
            "cost_gate_f1": 0.7373167981961668,
            "cost_gate_escalation_rate": 0.466
          },
          {
            "fold": 1,
       
```

---

## 15. Paper ↔ disk checklist

| Paper claim | Protocol | Disk | Match |
|-------------|----------|------|-------|
| Main F1 0.759 → 0.887 | P1 enriched n=1000 | 0.759 → 0.887 | Yes |
| Cost 1151 → 200 | P1 | 1150.8 → 199.8 | Yes |
| Stress 0.713 → 0.896 | P3 n=300 | same | Yes |
| Mistral 0.972 / cost 37.3 | P11 | same | Yes |
| Always-fall LLM control 0.790 / 346.7 | P11 derived | same | Yes |
| GBM 0.780 (unmatched) | P10 | 0.780 | Yes |
| Zoo best ~0.76 | P5/P6 | modern_tcn 0.760; G1v2 0.751 | Yes |
| KFall ~0.987 → 0.990 | P12 | yes | Yes |
| G1 ≥ 0.95 | P4–P6 | FAIL | Yes (scoped) |

---

## 16. Result folder map

```
results/fold{0..4}/              # P1 main ablations, agentic, LLM compare
results/kfold_ambiguous*/        # P3 SisFall stress
results/standard_protocol/       # P4 E1 natural zoo
results/standard_retrain/        # P5 train logs
results/standard_protocol_retrained/
results/standard_protocol_g1_v2/ # P6 G1 v2
results/r1_collapse/             # P7
results/seed_sweep/              # P8
results/astar/                   # P9 robustness / pareto / conformal audits
results/adjudicator_compare/     # P10 G2
results/kfall/                   # P12–P13
results/kfold_ambiguous_kfall*/  # P14
configs/*.yaml                   # locked protocol YAMLs
RCDP_Overleaf_2026_09_24/        # manuscript package
```

---

## 17. Planned but **not** run

| Experiment | Status |
|------------|--------|
| Elderly subject exclusion filter | Not coded |
| Age-stratified stress metrics | Absent |
| D05/D11 elderly-feasible near-fall proxy | Planned only |
| Matched GBM vs Mistral on identical cases+evidence | Absent |
| Full cross-backbone RCDP wrap all folds | Partial / incomplete |
| Mondrian FN-rate CRC with escalation budget B≤10% (AEGIS T1) | Blocked / not run |
| Prospective free-living / clinical validation | Not done |
