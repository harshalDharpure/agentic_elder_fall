# Agentic Fall Detection

**Confidence-gated multi-agent fall detection** for wearable IMU sensors.

This repository extends Bhatti et al. (IEEE Access 2025) *Beyond Falls* CNN–LSTM–Attention detector with an **agentic wrapper**: uncertainty routing, retrieval-augmented Chain-of-Thought reasoning, and cost-sensitive response actions. It also benchmarks classical and modern time-series backbones under the same protocol for Q1-ready comparison tables.

---

## Table of contents

1. [Research goal](#1-research-goal)
2. [Architecture](#2-architecture)
3. [Code map](#3-code-map)
4. [Data](#4-data)
5. [Models (Tier-1 zoo)](#5-models-tier-1-zoo)
6. [Agentic modules](#6-agentic-modules)
7. [Setup](#7-setup)
8. [How to run](#8-how-to-run)
9. [Experiments & paper tables](#9-experiments--paper-tables)
10. [Metrics](#10-metrics)
11. [Reproducibility](#11-reproducibility)
12. [Citation](#12-citation)

---

## 1. Research goal

### Problem with edge ML fall detectors

| Edge ML limitation | Our solution |
|---|---|
| Black-box decisions | CoT Reasoning Agent (natural-language rationale) |
| Fixed threshold, no uncertainty | Confidence Gate with \(\tau_{low}\) / \(\tau_{high}\) |
| Same alert for every fall | Action Agent (emergency / notify / monitor / log) |
| Force-classify near-falls | LLM Reasoning Agent on biomechanical evidence |
| Static after training | Growing k-NN case memory |

### Positioning vs base paper

| | Bhatti et al. 2025 | This work |
|---|---|---|
| Task | Multi-phase fall / ADL classification | Detection **+** uncertainty **+** explanation **+** response |
| Backbone | CNN–LSTM–Attention | Same + modern zoo (TCN, ModernTCN, TSMixer, PatchTST, Transformer) |
| Uncertainty | Softmax only | Dual-threshold confidence gate |
| Explainability | Attention weights | Structured evidence + CoT JSON rationale |
| Memory | None | Retrieval-augmented k-NN |
| Response | None | Cost-sensitive Action Agent |

**Corrections vs early design notes:** the base paper evaluates **KFall only** (not SisFall fusion), uses **~90-frame windows**, and is **36-class**. We derive \(p(\text{fall})\) for gating. SisFall is the primary runnable dataset here (binary Fall/ADL + near-fall ADLs D18/D19).

---

## 2. Architecture

### End-to-end pipeline

```
Raw IMU window (C × T), e.g. 6 × 90
        │
        ▼
┌───────────────────────────────┐
│  Tier-1 Detector (backbone) │  CNN-LSTM-Attn / ModernTCN / …
│  → logits, embedding, p_fall│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Confidence Gate              │
│  p ≤ τ_low  → ADL (direct)    │
│  p ≥ τ_high → FALL (direct)   │
│  else       → AMBIGUOUS       │
└───────────────┬───────────────┘
                │ ambiguous only
                ▼
┌───────────────────────────────┐
│  Evidence Serializer          │  free-fall, impact, stillness, tilt
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌─────────────────┐
│ k-NN Memory  │  │ LLM CoT Agent   │
│ top-k cases  │─▶│ Ollama / heur.  │
└──────────────┘  └────────┬────────┘
                           ▼
                ┌───────────────────────────────┐
                │  Action Agent                 │
                │  emergency | notify | monitor │
                │  | log  (simulated)           │
                └───────────────────────────────┘
```

### Design principles

- **Fast path**: clear ADL / clear fall never call the LLM (~most windows).
- **Slow path**: only ambiguous windows escalate (costly but thorough).
- **Reproducible**: Action Agent is a deterministic policy over structured fields; LLM output is validated JSON with heuristic fallback.
- **Fair backbone comparison**: swap Tier-1 only; keep gate / k-NN / CoT / action fixed.

---

## 3. Code map

```
agentic_fall/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── tier1_sisfall.yaml      # SisFall training
│   ├── tier1_kfall.yaml        # KFall (drop-in when licensed)
│   ├── agentic.yaml            # gate, LLM, retrieval, action
│   ├── backbones.yaml          # zoo sweep hyperparams
│   └── experiments.yaml        # experiment registry
├── src/agentic_fall/
│   ├── data/
│   │   ├── sisfall.py          # loader, prepare, subject folds
│   │   ├── kfall.py            # Algorithm-1 post-fall + prepare
│   │   ├── windowing.py        # sliding windows / resample
│   │   └── splits.py           # subject-independent stratified folds
│   ├── features/
│   │   └── biomechanics.py     # free-fall, impact, stillness, tilt
│   ├── models/
│   │   ├── cnn_lstm_attn.py    # Bhatti backbone + classical nets
│   │   ├── backbones.py        # TCN, ModernTCN, TSMixer, PatchTST, Transformer
│   │   └── factory.py          # build_model() registry
│   ├── agents/
│   │   ├── confidence_gate.py  # τ_low / τ_high + calibration
│   │   ├── evidence.py         # feature → text
│   │   ├── knn_memory.py       # growing case store
│   │   ├── llm_reasoner.py     # Ollama CoT + heuristic fallback
│   │   ├── action_agent.py     # graded response policy
│   │   └── pipeline.py         # end-to-end AgenticPipeline
│   ├── alignment/
│   │   └── sensor_language.py  # Stage-1 InfoNCE (optional)
│   ├── eval/
│   │   ├── metrics.py          # standard + agentic + LaTeX
│   │   └── train_loop.py       # train / eval / mixup / threshold
│   └── utils/                  # config, seed, io
├── scripts/
│   ├── download_sisfall.py
│   ├── prepare_sisfall.py
│   ├── prepare_kfall.py
│   ├── train_tier1.py
│   ├── train_alignment.py
│   ├── build_memory.py
│   ├── run_agentic_eval.py
│   ├── run_ablations.py
│   ├── run_backbone_benchmark.py
│   ├── run_llm_compare.py
│   └── export_tables.py
├── tests/test_core.py
├── data/{raw,processed,memory}/
├── checkpoints/
└── results/
```

### Module responsibilities (quick)

| Path | Role |
|---|---|
| `data/sisfall.py` | Parse SisFall txt → windows NPZ; stratified subject folds |
| `data/kfall.py` | Bhatti Algorithm 1 post-fall segmentation; 36-class prepare |
| `features/biomechanics.py` | Deterministic physics-ish features from a window |
| `models/factory.py` | One entrypoint: `build_model(name, C, num_classes, **kw)` |
| `agents/pipeline.py` | Orchestrates Tier-1 → gate → RAG → LLM → action |
| `eval/metrics.py` | F1/Spec + FAR/cost/escalation/ECE + LaTeX export |

---

## 4. Data

### SisFall (ready on this server)

- Source: waist-worn IMU, young (SA*) + elderly (SE*) subjects.
- We use ADXL345 accel + ITG3200 gyro → **6 channels**.
- Window **90**, hop **10** (aligned with base-paper windowing style).
- Labels: binary Fall vs ADL; activity codes retained (D18/D19 = near-fall ADLs).
- Splits: **subject-independent 5-fold**, stratified so fall-capable and ADL-only subjects appear in every fold.

```bash
python scripts/download_sisfall.py
python scripts/prepare_sisfall.py                    # full
python scripts/prepare_sisfall.py --max-files 500    # balanced subset (faster)
```

### KFall (drop-in)

1. Request access: https://sites.google.com/view/kfalldataset  
2. Extract to `data/raw/kfall/`  
3. Run:

```bash
python scripts/prepare_kfall.py --phase transition
python scripts/train_tier1.py --config configs/tier1_kfall.yaml
```

Post-fall labels follow Bhatti et al. Algorithm 1 (`segment_postfall` in `data/kfall.py`).

---

## 5. Models (Tier-1 zoo)

All models share: **input `(B, C, T)` → `(logits, embedding)`**.

| ID | Family | Notes |
|---|---|---|
| `threshold` | Classical | Peak accel magnitude rule |
| `cnn1d` | Classical | 1D CNN |
| `lstm` | Classical | 2-layer LSTM |
| `cnn_lstm` | Classical | CNN + LSTM (no attention) |
| `cnn_lstm_attn` | Base paper | Dual dilated CNN + LSTM + multi-head attention |
| `tcn` | Modern | Dilated residual Temporal ConvNet |
| `modern_tcn` | Modern | Depthwise-separable dilated TCN + SE |
| `tsmixer` | Modern | Time/feature MLP mixer |
| `patchtst` | Modern | Patch time-series Transformer |
| `transformer` | Modern | Vanilla Transformer encoder |

```python
from agentic_fall.models import build_model, ALL_BACKBONES
model = build_model("modern_tcn", in_channels=6, num_classes=2)
logits, emb = model(x)  # x: (B, 6, 90)
```

---

## 6. Agentic modules

### Confidence Gate (`agents/confidence_gate.py`)

- Defaults: \(\tau_{low}=0.30\), \(\tau_{high}=0.85\) (overridable; **calibrated** on val by expected cost).
- Routes: `adl` | `fall` | `ambiguous`.

### Evidence (`features/biomechanics.py` + `agents/evidence.py`)

Extracts and serializes:

- Free-fall duration (s)
- Impact magnitude (g)
- Post-impact stillness (s)
- Body tilt change (deg)
- Mean accel / peak gyro

### k-NN Memory (`agents/knn_memory.py`)

- Stores embeddings (MiniLM over evidence text) + labels + features.
- Retrieves top-\(k\) similar cases into the CoT prompt.
- `grow()` supports continual evidence accumulation.

### LLM Reasoning Agent (`agents/llm_reasoner.py`)

- Backend: Ollama (`mistral:latest`, `qwen2.5-coder:7b-instruct`) or `heuristic`.
- Output JSON: `{prediction, severity, confidence, rationale, action}`.
- Heuristic fallback always available for offline / busy GPU.

### Action Agent (`agents/action_agent.py`)

Deterministic policy (simulated side-effects only):

| Prediction | Severity cues | Action |
|---|---|---|
| FALL | impact ≥ 5g or stillness ≥ 10s | `emergency` |
| FALL | impact ≥ 3g or stillness ≥ 3s | `notify_caregiver` |
| FALL | milder | `monitor` |
| ADL / near-fall | elevated impact or short free-fall | `monitor` |
| ADL | otherwise | `log` |

### Pipeline (`agents/pipeline.py`)

`AgenticPipeline.run(x, window_tc, activity=...)` returns a `PipelineResult` with prediction, rationale, action, latency, escalation flag, and retrieved cases.

---

## 7. Setup

```bash
cd /DATA/tauseef_2121cs04/harshal/agentic_fall

pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# LLM (optional for CoT; heuristic works offline)
ollama serve          # if not already running
ollama pull mistral:latest
ollama pull qwen2.5-coder:7b-instruct
```

Hardware used in development: **NVIDIA A100 80GB**, CUDA 12.x, PyTorch 2.x.

---

## 8. How to run

### A. Prepare data

```bash
python scripts/download_sisfall.py
python scripts/prepare_sisfall.py
```

### B. Train a single Tier-1 model

```bash
python scripts/train_tier1.py \
  --config configs/tier1_sisfall.yaml \
  --model modern_tcn \
  --fold 0

# Other models:
# --model cnn1d|lstm|cnn_lstm|cnn_lstm_attn|tcn|tsmixer|patchtst|transformer
```

### C. Build retrieval memory

```bash
python scripts/build_memory.py \
  --checkpoint checkpoints/sisfall/cnn_lstm_attn_fold0.pt \
  --fold 0
```

### D. Agentic evaluation

```bash
python scripts/run_agentic_eval.py \
  --checkpoint checkpoints/sisfall/cnn_lstm_attn_fold0.pt \
  --memory data/memory/sisfall_fold0.json \
  --backend heuristic \
  --calibrate --fold 0

# With Ollama:
python scripts/run_agentic_eval.py \
  --checkpoint checkpoints/sisfall/cnn_lstm_attn_fold0.pt \
  --memory data/memory/sisfall_fold0.json \
  --backend ollama --calibrate --fold 0
```

### E. Full backbone zoo (Table A + B)

```bash
python scripts/run_backbone_benchmark.py --device cuda
# logs: results/full_backbone_benchmark.log
# out:  results/backbone_compare_fold0.{csv,json,tex}
#       results/backbone_agentic_fold0.tex
```

### F. Ablations (gate / k-NN / LLM)

```bash
python scripts/run_ablations.py \
  --fold 0 --epochs 20 --max-test 2000 \
  --checkpoint checkpoints/sisfall/cnn_lstm_attn_fold0.pt \
  --memory data/memory/sisfall_fold0.json
```

### G. LLM backend comparison (Table C)

```bash
python scripts/run_llm_compare.py \
  --checkpoint checkpoints/sisfall/cnn_lstm_attn_fold0.pt \
  --memory data/memory/sisfall_fold0.json \
  --backends heuristic mistral:latest qwen2.5-coder:7b-instruct \
  --max-cases 200
```

### H. Optional sensor–language alignment

```bash
python scripts/train_alignment.py --fold 0 --epochs 10
```

### I. Tests

```bash
pytest -q
```

### J. Export combined tables

```bash
python scripts/export_tables.py
```

---

## 9. Experiments & paper tables

All paper runs go through the locked protocol in [`configs/paper_protocol.yaml`](configs/paper_protocol.yaml):

```bash
# Prepare full SisFall (must include D18/D19)
python scripts/prepare_sisfall.py --required-activities D18 D19

# Fold 0 smoke, then all folds
FOLDS=0 DEVICE=cuda bash scripts/run_full_experiments.sh
FOLDS=0,1,2,3,4 DEVICE=cuda bash scripts/run_full_experiments.sh

# Aggregate mean±std + paired tests
python scripts/aggregate_folds.py --results-dir results
```

| Table | Script | Artifact |
|---|---|---|
| **A** Backbone zoo (F1, Spec, Recall, Params, Latency) | `run_backbone_benchmark.py` | `results/fold{k}/backbone_compare_fold{k}.*` |
| **B** Top-N + agentic gain (ΔF1, cost, escalation, near-fall FAR) | same script (agentic stage) | `results/fold{k}/backbone_agentic_fold{k}.tex` |
| **C** LLM backends on ambiguous cases (paired IDs) | `run_llm_compare.py` | `results/fold{k}/llm_compare_fold{k}.*` |
| Ablations | `run_ablations.py` | `results/fold{k}/ablation_fold{k}.*` |
| End-to-end agentic | `run_agentic_eval.py` | `results/fold{k}/agentic_fold{k}.json` |
| Aggregate | `aggregate_folds.py` | `results/aggregate_*_mean_std.*`, `aggregate_*_tests.json` |

### Evidence contract (publish only if met)

- Subject-independent **5-fold** mean±std
- Near-fall FAR on **D18/D19** (`false_alarm_rate_near_fall`) and escalated-subset F1 (`escalated_f1`)
- Agentic vs same Tier-1 checkpoint: lower **normalized** cost (`cost_per_1000`) and/or better F1/recall
- LLM compare on **shared** ambiguous case IDs

### Guide meeting pack

Open **[`results/GUIDE_BRIEF.md`](results/GUIDE_BRIEF.md)** for the 1–2 page summary (per-fold wins, mean±std, LLM table, talking points).

### Hybrid eval budget (methods footnote)

| Folds | Ablation `max_test` | LLM ablation modes | Backbone |
|---|---|---|---|
| 0–1 (original / archived) | 4000 | full ladder incl. `gate_llm` | zoo |
| 2–4 | 1000 | skip `gate_llm` | primary `cnn_lstm_attn` only |
| Overnight homogenize | fold0/1 → 1000 | same as 2–4 | `*_full4000.json` archived |

Primary claim: **`gate_knn_llm` vs `tier1_only`** using **cost per 1000 windows**. Use `--backend ollama` for end-to-end agentic. See `results/aggregate_paper_footnotes.json`.

### Suggested paper claim

> On SisFall subject-independent 5-fold CV, confidence-gated retrieval-augmented CoT (`gate_knn_llm`) improves F1/recall and reduces normalized response cost versus the same Tier-1 `cnn_lstm_attn` detector. Paired local LLM backends (Mistral/Qwen) recover ambiguous cases that a heuristic reasoner misses. Near-fall FAR is reported but is not the primary differentiator.

KFall is supported in code (`data/kfall.py`, binary `p(fall)` path) but experiments are gated on placing raw data under `data/raw/kfall/`.

---

## 10. Metrics

**Detection:** accuracy, precision, recall/sensitivity, specificity, F1, params, latency (ms), val-tuned decision threshold.

**Agentic / safety (Q1 differentiators):**

- `false_alarm_rate_near_fall` — FAR on D18/D19 near-fall ADLs
- `escalated_f1` — F1 on gate-escalated windows
- Missed fall rate
- Expected response cost (\(w_{FN} \gg w_{FP}\))
- Escalation rate (% to LLM)
- ECE (calibration)
- Tier-1 vs escalated latency
- Counts: `n_d18`, `n_d19`, `n_near_fall`, `n_escalated`

---

## 11. Reproducibility

- Seed: `42` (`configs/paper_protocol.yaml`)
- Subject-independent 5-fold CV with **subject-held-out validation**
- Shared harness: `src/agentic_fall/eval/protocol.py`
- Class imbalance: `pos_weight` in CE; cost-sensitive decision threshold on val
- Same window/hop for all backbones
- Adam + cosine LR, label smoothing, Mixup
- Action side-effects are **simulated** (no real emergency calls)
- Prepare fails loudly if D18/D19 are missing from the processed archive

---

## 12. Citation

```bibtex
@article{bhatti2025beyond,
  title={Beyond Falls: A Hybrid CNN--LSTM--Attention Framework for Pre-, Transition-, and Post-Fall Detection With Wearable Inertial Sensors},
  author={Bhatti, Faraz Ahmad and Riaz, Qaiser and Kr{\"u}ger, Bj{\"o}rn},
  journal={IEEE Access},
  volume={13},
  pages={207960--207969},
  year={2025}
}
```

Datasets: **SisFall** (Sucerquia et al., Sensors 2017); **KFall** (Yu et al., Front. Aging Neurosci. 2021).
