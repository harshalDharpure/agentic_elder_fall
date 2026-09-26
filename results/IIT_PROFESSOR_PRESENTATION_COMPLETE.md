# Agentic Fall Detection — Complete Analysis Pack for IIT Professor Presentation

**Title:** Confidence-Gated Multi-Agent Fall Detection: Retrieval-Augmented Reasoning and Cost-Sensitive Response on Wearable IMU Data  

**Use this file tomorrow.** Start with **SLIDE GOLD — Final Comparison Pack** (all headline tables). Then architecture, full SisFall/KFall analysis, ablations, LLM, latency, talking points, and honest Q1 claims.

**Paper PDF:** `paper/main.pdf`  
**Architecture figure:** use your corrected diagram / `paper/fig/architecture.tex`  
**Generated:** 2026-08-09

---

## 0. Opening pitch (45 seconds)

> Sir, Bhatti et al. (IEEE Access 2025) built a strong **CNN–LSTM–Attention fall detector**. We keep that detector fixed as **Tier-1** and add an **agentic safety layer**: Confidence Gate → biomechanical Evidence → k-NN memory → local LLM chain-of-thought (Mistral via Ollama) → cost-sensitive Action Agent.  
> Under subject-independent **5-fold SisFall**, our full stack (`gate_knn_llm`) improves mean F1 **0.759 → 0.887**, recall **0.759 → 0.977**, and cuts normalized response cost by ~**83%** vs the **same** Tier-1 checkpoint.  
> On **KFall** (Bhatti’s dataset world, binary Fall/ADL), Tier-1 is already near ceiling (~0.99 F1); our stack still improves mean F1 **0.987 → 0.990**, recall **0.991 → 0.996**, and cost **50 → 26**.

**One-line difference:** Bhatti = *detection*; we = *what to do when the detector is uncertain*, under FN-heavy clinical cost.

---

## 1. What NOT to claim (builds trust)

| Avoid saying | Why |
|--------------|-----|
| “We beat Bhatti’s published ~98% multi-class KFall table” | Different task/protocol; we do **not** claim that |
| “Our detector alone is F1 ≥ 0.90 on SisFall” | SisFall Tier-1 under our cost OP is ~0.76 |
| “We solved near-fall FAR completely” | Lead with **recall + cost**; FAR gains are modest |
| “Heuristic is the main result” | Main stack needs **Ollama/Mistral** |

**Honest Q1 claim:** Same Bhatti-style backbone + agents improve **recall / cost under uncertainty** on SisFall (primary) and KFall binary (external).

---

# SLIDE GOLD — Final Comparison Pack (SisFall + KFall)

**Protocol (both datasets):** same architecture · subject-independent **5-fold** · \(n{=}1000\) · cost \(=10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\).

| Label | Definition |
|-------|------------|
| **Tier-1** | Bhatti-style CNN–LSTM–Attention alone (`tier1_only`) |
| **Ours** | Full stack `gate_knn_llm` = Gate + Evidence + k-NN + Mistral CoT + Action Agent |

---

## Pack Table A — How the architecture works (both datasets)

| Step | What happens |
|------|----------------|
| 1 | IMU window **(6 × 90)** |
| 2 | Tier-1 CNN–LSTM–Attention → \(p_{\mathrm{fall}}\) |
| 3a | Confidence Gate: **sure** ADL/Fall → Direct decision → Action Agent |
| 3b | Confidence Gate: **unsure** → Evidence → k-NN (MiniLM) → LLM CoT (Mistral) → Action Agent |
| 4 | Score: \(\mathrm{cost}=10\times\mathrm{FN}+1\times\mathrm{FP}\) |

```
IMU window (6×90)
  → Tier-1 CNN–LSTM–Attention → p_fall
  → Confidence Gate
       ├─ sure ADL/Fall → Direct decision → Action Agent
       └─ unsure → Evidence → k-NN (MiniLM) → LLM CoT (Mistral) → Action Agent
  → Score: cost = 10×FN + 1×FP
```

Same pipeline on SisFall and KFall; only the **data source / sample rate** differs.

---

## Pack Table B — Headline: Tier-1 vs Ours (5-fold mean)

| Dataset | Method | F1 | Recall | Cost / 1000 |
|---------|--------|-----|--------|-------------|
| SisFall (primary) | Tier-1 | 0.759 ± 0.014 | 0.759 ± 0.022 | 1151 ± 93 |
| SisFall (primary) | **Ours** | **0.887 ± 0.029** | **0.977 ± 0.037** | **200 ± 164** |
| KFall (external) | Tier-1 | 0.987 ± 0.005 | 0.991 ± 0.007 | 50 ± 35 |
| KFall (external) | **Ours** | **0.990 ± 0.004** | **0.996 ± 0.002** | **26 ± 12** |

### Pack Table B2 — Deltas (Ours − Tier-1)

| Dataset | Δ F1 | Δ Recall | Δ Cost |
|---------|------|----------|--------|
| SisFall | **+0.128** | **+0.218** | **−951 (~83%↓)** |
| KFall | **+0.003** | **+0.005** | **−24 (~48%↓)** |

**Read:** SisFall = big accuracy + cost gains. KFall = Tier-1 already near ceiling; ours still improves recall/cost a bit.

---

## Pack Table C — Per-fold SisFall

| Fold | Tier-1 F1 | Ours F1 | Tier-1 Rec | Ours Rec | Tier-1 cost | Ours cost | ΔF1 |
|------|-----------|---------|------------|----------|-------------|-----------|-----|
| 0 | 0.774 | **0.892** | 0.781 | **0.986** | 1042 | **156** | +0.119 |
| 1 | 0.737 | **0.851** | 0.748 | **0.911** | 1223 | **490** | +0.114 |
| 2 | 0.755 | **0.874** | 0.783 | **0.998** | 1066 | **134** | +0.119 |
| 3 | 0.766 | **0.930** | 0.732 | **0.995** | 1248 | **83** | +0.164 |
| 4 | 0.760 | **0.887** | 0.749 | **0.993** | 1175 | **136** | +0.126 |

**Ours wins every fold.**

---

## Pack Table D — Per-fold KFall (binary)

| Fold | Tier-1 F1 | Ours F1 | Tier-1 Rec | Ours Rec | Tier-1 cost | Ours cost | ΔF1 |
|------|-----------|---------|------------|----------|-------------|-----------|-----|
| 0 | 0.992 | **0.993** | 0.998 | **1.000** | 17 | **7** | +0.001 |
| 1 | 0.987 | 0.986 | 0.996 | 0.996 | 31 | 32 | −0.001 |
| 2 | 0.987 | **0.989** | 0.994 | **0.996** | 40 | **29** | +0.002 |
| 3 | 0.992 | **0.995** | 0.990 | **0.996** | 53 | **23** | +0.003 |
| 4 | 0.980 | **0.989** | 0.979 | **0.994** | 109 | **38** | +0.008 |

Most folds: slight F1↑ and/or cost↓. Fold 1 ≈ flat.

---

## Pack Table E — Ablation ladder (SisFall, 5-fold mean, n=1000)

| Mode | F1 | Recall | Cost/1000 | Meaning |
|------|-----|--------|-----------|---------|
| `tier1_only` | 0.759 | 0.759 | 1151 | Detector only |
| `gate_only` | 0.759 | 0.759 | 1151 | Gate alone |
| `gate_knn` | 0.753 | 0.723 | 1286 | Gate + memory |
| `gate_llm` | 0.702 | 0.895 | 740 | Gate + LLM |
| **`gate_knn_llm` (Ours)** | **0.887** | **0.977** | **200** | Full stack |

### Pack Table E2 — Ablation ladder (KFall Fold 0)

| Mode | F1 | Recall | Cost | Esc. F1 |
|------|-----|--------|------|---------|
| `tier1_only` | 0.992 | 0.998 | 17 | — |
| `gate_only` | 0.992 | 0.998 | 17 | 0.750 |
| `gate_knn` | 0.991 | 0.994 | 36 | 0.400 |
| `gate_llm` | 0.976 | 0.998 | 33 | 0.250 |
| **`gate_knn_llm`** | **0.993** | **1.000** | **7** | **0.889** |

**Full stack needs k-NN + LLM together.**

---

## Pack Table F — LLM backends (ambiguous cases only)

| Dataset | Heuristic F1 | **Mistral F1** | Qwen F1 |
|---------|--------------|----------------|---------|
| SisFall | 0.034 | **0.972** | 0.823 |
| KFall | 0.088 | **0.885** | 0.504 |

Main stack uses **Ollama + Mistral**.

---

## Pack Table G — Method positioning (paper claim)

| Aspect | Bhatti / Tier-1 | **Our proposed method** |
|--------|-----------------|-------------------------|
| Backbone | CNN–LSTM–Attention | Same + agents |
| Uncertainty | Softmax only | Confidence Gate |
| Memory | None | k-NN (MiniLM) |
| Reasoning | None | LLM CoT (Mistral) |
| Response | None | Action Agent |
| Fair numeric compare | `tier1_only` | `gate_knn_llm` |
| SisFall story | Weaker under cost protocol | Large F1 / recall / cost gains |
| KFall story | Already ~0.99 binary F1 | Safety layer: cost↓, small F1↑ |
| Do **not** claim | — | Beating Bhatti’s published 36-class ~98% table |

---

## Pack Table H — One-line takeaway

| Dataset | Takeaway |
|---------|----------|
| **Both** | Same architecture on SisFall and KFall |
| **SisFall** | Agents **clearly beat** Tier-1 (F1, recall, cost) |
| **KFall** | Tier-1 already excellent; agents still **lower FN-heavy cost** and slightly improve mean F1/recall |

> **One-line:** Same architecture on both datasets: on SisFall the agents clearly beat Tier-1; on KFall Tier-1 is already excellent, and the agents still lower expected FN-heavy cost while slightly improving mean F1/recall.

---

# PART A — Architectures

---

## Table 1 — Positioning: Bhatti vs Our Proposed Method

| Aspect | Bhatti et al. 2025 (base) | **Our proposed method** |
|--------|--------------------------|-------------------------|
| Role | Multi-phase fall/ADL **detector** | Detector + **agentic safety layer** |
| Backbone | CNN–LSTM–Attention | **Same** primary backbone (`cnn_lstm_attn`) |
| Uncertainty | Softmax only | Dual-threshold **Confidence Gate** (\(\tau_{low},\tau_{high}\)) |
| Explainability | Attention weights | Biomechanical **evidence** + LLM **CoT** JSON |
| Case memory | None | MiniLM **k-NN** retrieval (RAG) |
| Response policy | None | Cost-sensitive **Action Agent** |
| Cost model | Not primary claim | \(10\cdot\mathrm{FN} + 1\cdot\mathrm{FP}\) |
| Fair baseline in *our* tables | — | `tier1_only` = same ckpt, same folds |
| Original paper focus | KFall multi-class (~98% F1) | SisFall binary + D18/D19; KFall external binary |
| Main mode name | — | **`gate_knn_llm`** |

---

## Table 2 — Tier-1 Architecture (Bhatti-style detector)

**Code:** `src/agentic_fall/models/cnn_lstm_attn.py` · **Name:** `cnn_lstm_attn`  
**Config:** `configs/tier1_sisfall.yaml` / `configs/tier1_kfall_binary.yaml`

| Stage | Details |
|-------|---------|
| Input | IMU window \((C{=}6, T{=}90)\): 3-axis accel + 3-axis gyro |
| CNN stem | Conv1d → BatchNorm → ReLU |
| Parallel dilated CNN branches | dilation 1 & 2 → concatenate |
| LSTM stack | Hidden **256 → 128** (sequential) |
| Attention | Multi-head self-attention (**4 heads**) |
| Pooling | Mean over time → embedding \(\mathbf{e}\) |
| Head | Dropout → Linear → Softmax |
| Outputs | \(p_{\mathrm{fall}}\), embedding \(\mathbf{e}\) |
| Task (our tables) | Binary **Fall vs ADL** |
| At agentic inference | **Weights frozen** |

```
IMU (B,6,90)
  → CNN stem → dilated branches → fuse
  → LSTM(256) → LSTM(128)
  → Multi-Head Attention
  → mean pool → emb e
  → FC → Softmax → p_fall
```

**What to say:**  
> Tier-1 is our faithful Bhatti-style hybrid. All agentic gains are measured against **this same checkpoint**, not against a weaker baseline.

---

## Table 3 — Our Agentic / Gate Architecture (proposed stack)

**Code:** `src/agentic_fall/agents/` · **Config:** `configs/agentic.yaml`

| Module | What it does | When it runs |
|--------|--------------|--------------|
| **1. Confidence Gate** | Dual thresholds: \(p\le\tau_{low}\)→ADL; \(p\ge\tau_{high}\)→Fall; else **ambiguous** | Always |
| **2. Evidence Serializer** | Free-fall, impact, stillness, tilt → natural-language evidence | Escalated (+ feats used by Action always) |
| **3. k-NN Memory (RAG)** | MiniLM embed of evidence; retrieve \(k{=}5\) cases; fallback = Tier-1 \(e\) | Escalated (modes with knn) |
| **4. LLM Reasoner (CoT)** | Ollama **`mistral:latest`**; JSON with step-by-step **rationale**; Qwen for compare only; heuristic fallback | Escalated (modes with llm) |
| **5. Action Agent** | Map to `log → monitor → notify_caregiver → emergency`; prefer more conservative LLM suggestion | Always |

### End-to-end flow (both SisFall & KFall)

```
Wearable IMU window (6 × 90)
   → Tier-1 CNN–LSTM–Attention → p_fall (+ e)
   → Confidence Gate (τ_low / τ_high)
        ├─ High-confidence ADL / Fall
        │     → Direct decision (source=tier1)
        │     → Action Agent → Output
        └─ Ambiguous / force-escalate (e.g. D18/D19 eval)
              → Evidence text (biomechanics)
              → k-NN retrieve similar cases (MiniLM; fallback e)
              → Local LLM CoT (Mistral) → prediction, severity, rationale, action
              → Action Agent (severity cues + conservative merge)
              → Pipeline Output
```

### Cost (evaluation objective — not computed inside Action Agent online)

\[
\text{expected\_response\_cost} = 10 \times \mathrm{FN} + 1 \times \mathrm{FP}
\]

With \(n{=}1000\) windows/fold, this equals **Cost / 1000** in main tables.  
**Missed falls are 10× more expensive than false alarms.**

### Ablation mode names (Q1 ladder)

| Mode | Gate | k-NN | LLM | Meaning |
|------|------|------|-----|---------|
| `tier1_only` | collapsed | ✗ | ✗ | Bhatti-style detector alone |
| `gate_only` | ✓ | ✗ | ✗ | Selective routing only |
| `gate_knn` | ✓ | ✓ | heuristic | Retrieval without neural LLM |
| `gate_llm` | ✓ | ✗ | ✓ | LLM without memory |
| **`gate_knn_llm`** | ✓ | ✓ | ✓ | **Full proposed method** |

---

## Table 4 — Datasets & Inputs (SisFall vs KFall)

| Item | **SisFall** (primary) | **KFall** (external) |
|------|----------------------|----------------------|
| Role in paper | Main numerical claim | External validation vs Bhatti’s world |
| Subjects (approx.) | 38 | ~32 |
| Processed windows | ~1.55M | ~228k (transition phase) |
| Channels | 6 (ADXL345 + ITG3200) | 6 (accel + gyro) |
| Window / hop | 90 / 10 | 90 / 10 |
| Sample rate | **200 Hz** (~0.45 s clip) | **100 Hz** (~0.90 s clip) |
| Label (our claim) | Binary Fall/ADL | Binary Fall/ADL |
| Near-fall focus | Forced **D18/D19** in test | Hard ADLs; similar protocol |
| Architecture | **Identical stack** | **Identical stack** |
| Config | `tier1_sisfall.yaml` | `tier1_kfall_binary.yaml` |

**What to say:**  
> Same robot team, two movement libraries. SisFall is the main exam; KFall is the extra exam on Bhatti’s dataset.

### What is a fold?
Subject-independent **5-fold**: people in the test fold were **never** in training for that fold. Mean ± std = average ± spread over folds 0–4.

---

# PART B — SisFall Results (Primary Q1 Evidence)

**Protocol:** subject-independent 5-fold, homogeneous **n = 1000**/fold, forced D18/D19 (≥80 each), seed 42, cost \(10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\), full stack backend = **Ollama/Mistral**.

---

## Table 5 — SisFall Per-fold: Tier-1 vs Proposed (`gate_knn_llm`)

| Fold | n | Tier-1 F1 | Tier-1 Rec | Tier-1 Cost/1k | **Ours F1** | **Ours Rec** | **Ours Cost/1k** | ΔF1 | Cost ↓ |
|------|---|-----------|------------|----------------|-------------|--------------|------------------|-----|--------|
| 0 | 1000 | 0.774 | 0.781 | 1042 | **0.892** | **0.986** | **156** | +0.119 | ~85% |
| 1 | 1000 | 0.737 | 0.748 | 1223 | **0.851** | **0.911** | **490** | +0.114 | ~60% |
| 2 | 1000 | 0.755 | 0.783 | 1066 | **0.874** | **0.998** | **134** | +0.119 | ~87% |
| 3 | 1000 | 0.766 | 0.732 | 1248 | **0.930** | **0.995** | **83** | +0.164 | ~93% |
| 4 | 1000 | 0.760 | 0.749 | 1175 | **0.887** | **0.993** | **136** | +0.126 | ~88% |

**Talking point:** Stack wins on **every fold**.

### Why cost drops (example Fold 0)

| Method | FN | FP | Cost = 10·FN + FP |
|--------|----|----|-------------------|
| Tier-1 | 94 | 102 | **1042** |
| Ours | 6 | 96 | **156** |

FP almost flat; **FN collapses** → cost collapses (FN weight = 10).

---

## Table 6 — SisFall 5-fold Mean ± Std (Main Claim)

| Mode | F1 | Recall | Cost / 1000 |
|------|-----|--------|-------------|
| Tier-1 (`tier1_only`) | 0.759 ± 0.014 | 0.759 ± 0.022 | 1151 ± 93 |
| **Ours (`gate_knn_llm`)** | **0.887 ± 0.029** | **0.977 ± 0.037** | **200 ± 164** |

| Metric | Mean Δ (Ours − Tier-1) | Bootstrap 95% CI | Wilcoxon p |
|--------|------------------------|------------------|------------|
| F1 | **+0.128** | [0.117, 0.147] | 0.0625 |
| Recall | **+0.218** | [0.187, 0.248] | 0.0625 |
| Cost/1000 | **−951** (~83%↓) | [−1084, −825] | 0.0625 |

**About ±0.014:** mean ± **standard deviation across 5 folds** (spread), not the cost formula.  
**About p=0.0625:** with 5 folds and complete sign agreement, Wilcoxon bottoms at 0.0625; emphasize **bootstrap CIs exclude zero**.

---

## Table 7 — SisFall Ablation Ladder (5-fold mean, n=1000)

| Mode | F1 | Recall | Cost/1000 | Interpretation |
|------|-----|--------|-----------|----------------|
| `tier1_only` | 0.759 ± 0.014 | 0.759 ± 0.022 | 1151 ± 93 | Detector alone |
| `gate_only` | 0.759 ± 0.014 | 0.759 ± 0.022 | 1151 ± 93 | Gate alone ≈ no gain |
| `gate_knn` | 0.753 ± 0.012 | 0.723 ± 0.032 | 1286 ± 133 | Memory alone weak / can hurt |
| `gate_llm` | 0.702 ± 0.021 | 0.895 ± 0.027 | 740 ± 105 | LLM helps recall; F1 can drop |
| **`gate_knn_llm`** | **0.887 ± 0.029** | **0.977 ± 0.037** | **200 ± 164** | **k-NN + LLM needed together** |

**Talking point:** Modular story — retrieval and LLM are **complementary**.

### Archive detail (Fold 0, n=4000) — richer escalated metrics

| Mode | F1 | Recall | Cost | Near-fall FAR | Escalated F1 |
|------|-----|--------|------|---------------|--------------|
| tier1_only | 0.788 | 0.787 | 4596 | 0.219 | 0.000 |
| gate_only | 0.788 | 0.787 | 4596 | 0.219 | 0.175 |
| gate_knn | 0.789 | 0.764 | 4972 | 0.188 | 0.026 |
| gate_llm | 0.760 | 0.924 | 2491 | 0.625 | 0.445 |
| **gate_knn_llm** | **0.907** | **0.993** | **516** | 0.212 | **0.952** |

---

## Table 8 — SisFall LLM Backends (paired ambiguous IDs)

Same hard windows; only the reasoner changes. Main stack uses **Mistral**.

| Fold | Heuristic F1 | **Mistral F1** | Qwen2.5 F1 |
|------|--------------|----------------|------------|
| 0 | 0.042 | **0.959** | 0.807 |
| 1 | 0.105 | **0.977** | 0.874 |
| 2 | 0.000 | **0.975** | 0.819 |
| 3 | 0.021 | **0.984** | 0.802 |
| 4 | 0.000 | **0.966** | 0.813 |
| **Mean** | 0.034 | **0.972** | 0.823 |

**Talking point:** Escalation alone is not enough — **LLM CoT** recovers ambiguous cases (heuristic ~0.03 → Mistral ~0.97).

---

## Table 9 — SisFall Latency (deployment honesty)

| Fold | Tier-1 (ms) | Escalated path (s) | Mean stack (s) |
|------|-------------|--------------------|----------------|
| 0 | 8.96 | 12.81 | 5.68 |
| 1 | 8.09 | 10.66 | 3.73 |
| 2 | 8.52 | 22.85 | 10.52 |
| 3 | 8.36 | 11.00 | 5.99 |
| 4 | 8.89 | 10.86 | 5.93 |
| **Mean** | **8.56 ms** | **13.63 s** | **6.37 s** |

**Talking point:** Confident path is milliseconds; escalated path is seconds (local LLM). We report this; we do **not** claim free-living real-time on every window.

---

## Table 10 — Backbone Zoo (context; Fold 0 detector-only)

Primary agentic backbone fixed as **`cnn_lstm_attn`** for fair Bhatti-style comparison.

| Model | F1 | Recall | Spec. | Params |
|-------|-----|--------|-------|--------|
| cnn1d | 0.721 | 0.985 | 0.277 | 43,330 |
| lstm | 0.700 | 0.986 | 0.197 | 201,986 |
| cnn_lstm | 0.694 | 0.995 | 0.155 | 86,914 |
| **cnn_lstm_attn (Tier-1)** | 0.710 | 0.996 | 0.216 | 564,930 |
| tcn | 0.727 | 0.988 | 0.295 | 89,282 |
| modern_tcn | 0.737 | 0.990 | 0.326 | 30,314 |
| tsmixer | 0.695 | 0.984 | 0.181 | 224,562 |
| patchtst | 0.669 | 0.996 | 0.051 | 150,786 |
| transformer | 0.678 | 0.994 | 0.092 | 158,722 |

**Talking point:** Q1 claim is **not** “new SOTA detector”; it is **agentic wrapper on a strong fixed detector**.

---

# PART C — KFall Results (External Validation)

**Protocol:** mirrors SisFall — binary Fall/ADL, 5-fold subject-independent, n=1000, cost \(10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\), same `gate_knn_llm` stack.  
**Not** a claim against Bhatti’s published 36-class ~98% table.

---

## Table 11 — KFall Per-fold: Tier-1 vs Proposed

| Fold | Tier-1 F1 | Ours F1 | Tier-1 Rec | Ours Rec | Tier-1 Cost/1k | Ours Cost/1k | ΔF1 |
|------|-----------|---------|------------|----------|----------------|--------------|-----|
| 0 | 0.992 | **0.993** | 0.998 | **1.000** | 17 | **7** | +0.001 |
| 1 | 0.987 | 0.986 | 0.996 | 0.996 | 31 | 32 | −0.001 |
| 2 | 0.987 | **0.989** | 0.994 | **0.996** | 40 | **29** | +0.002 |
| 3 | 0.992 | **0.995** | 0.990 | **0.996** | 53 | **23** | +0.003 |
| 4 | 0.980 | **0.989** | 0.979 | **0.994** | 109 | **38** | +0.008 |

**Talking point:** On KFall, Tier-1 is already excellent; agents still help cost on most folds (Fold 1 ≈ flat).

---

## Table 12 — KFall 5-fold Mean ± Std

| Mode | F1 | Recall | Cost / 1000 |
|------|-----|--------|-------------|
| Tier-1 | 0.987 ± 0.005 | 0.991 ± 0.007 | 50 ± 35 |
| **Ours (`gate_knn_llm`)** | **0.990 ± 0.004** | **0.996 ± 0.002** | **26 ± 12** |

| Metric | Mean Δ | Bootstrap 95% CI | Note |
|--------|--------|------------------|------|
| F1 | +0.003 | [0.000, 0.006] | Small (near ceiling) |
| Recall | +0.005 | [0.001, 0.010] | Consistent direction |
| Cost/1000 | **−24.2** (~48%↓) | [−48.4, −5.8] | Main KFall safety gain |

---

## Table 13 — KFall Fold-0 Ablation Ladder

| Mode | F1 | Recall | Cost | Escalated F1 |
|------|-----|--------|------|--------------|
| tier1_only | 0.992 | 0.998 | 17 | — |
| gate_only | 0.992 | 0.998 | 17 | 0.750 |
| gate_knn | 0.991 | 0.994 | 36 | 0.400 |
| gate_llm | 0.976 | 0.998 | 33 | 0.250 |
| **gate_knn_llm** | **0.993** | **1.000** | **7** | **0.889** |

Again: **full stack needs k-NN + LLM**.

---

## Table 14 — KFall LLM Backends (paired ambiguous IDs)

| Fold | Heuristic F1 | **Mistral F1** | Qwen F1 |
|------|--------------|----------------|---------|
| 0 | 0.167 | **0.880** | 0.500 |
| 1 | 0.000 | **1.000** | 0.364 |
| 2 | 0.000 | **1.000** | 0.600 |
| 3 | 0.000 | **0.571** | 0.250 |
| 4 | 0.273 | **0.974** | 0.809 |
| **Mean** | 0.088 | **0.885** | 0.504 |

---

# PART D — Combined Cross-Dataset View (Slide Gold)

---

## Table 15 — Combined Means: SisFall + KFall + Tier-1 vs Ours

| Dataset | Role | Tier-1 F1 | **Ours F1** | Tier-1 Rec | **Ours Rec** | Tier-1 Cost/1k | **Ours Cost/1k** | Cost change |
|---------|------|-----------|-------------|------------|--------------|----------------|------------------|-------------|
| **SisFall** | Primary | 0.759 | **0.887** | 0.759 | **0.977** | 1151 | **200** | **~83% ↓** |
| **KFall** | External | 0.987 | **0.990** | 0.991 | **0.996** | 50 | **26** | **~48% ↓** |

### How to interpret for professor

| Dataset | Story |
|---------|--------|
| SisFall | Harder under our locked protocol → **large** agentic gains in F1, recall, cost |
| KFall | Binary Tier-1 already ~0.99 → agents are a **safety layer** (cost↓, small F1/recall↑) |
| Both | **Same architecture**; different data difficulty / ceiling |

---

## Table 16 — Bhatti Gaps → Our Multi-Agent Answers

| Bhatti et al. limitation | Our response |
|--------------------------|--------------|
| Softmax force-labels every window; sit/stand (D18/D19) confusion | Dual-threshold **Confidence Gate** escalates ambiguous band |
| Explainability ≈ attention only | Biomechanical **evidence** + LLM **CoT** JSON |
| No retrieval / case memory | MiniLM **k-NN Memory** |
| Detection only; no clinical response policy | Cost-sensitive **Action Agent** (\(10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\)) |
| Published focus: 36-class ~98% F1, ~20 ms | Our fair claim: same Tier-1 + agents improve recall/cost under uncertainty |

---

## Table 17 — Where CoT / Which LLM

| Setting | Model / backend | Where used |
|---------|-----------------|------------|
| **Main full stack** | Ollama + **`mistral:latest`** | SisFall + KFall `gate_knn_llm` / `gate_llm` |
| Compare only | `qwen2.5-coder:7b-instruct` | Paired LLM tables |
| Fallback / ablation | **Heuristic** step rules | `gate_knn`; Ollama failure |
| Embedder (not LLM) | **MiniLM** (`all-MiniLM-L6-v2`) | k-NN query vectors |

**CoT location:** `LLMReasoner` (`src/agentic_fall/agents/llm_reasoner.py`) on **escalated** windows only; JSON field `"rationale": "step-by-step reasoning"`.

---

# PART E — Presentation Script (5–8 minutes)

1. **Problem (30s):** Wearable fall detection fails on ambiguous near-falls; missed falls are clinically costly; binary output ≠ response policy.  
2. **Base paper (30s):** Bhatti = strong CNN–LSTM–Attention detector (esp. KFall multi-class). Gaps: no gate, no memory, no LLM explanation, no action policy.  
3. **Our method (60s):** Show architecture diagram — Tier-1 frozen → Gate → Evidence → k-NN → Mistral CoT → Action; cost = 10·FN+1·FP.  
4. **SisFall (90s):** Tables 5–6 every-fold win; 0.76→0.89 F1; 0.76→0.98 recall; cost ~83%↓. Table 7 ablation: need knn+llm. Table 8: Mistral ≫ heuristic.  
5. **KFall (45s):** Same stack; Tier-1 already strong; still cost 50→26; honest: not beating published 36-class 98%.  
6. **Combined (20s):** Table 15 — dual-dataset evidence.  
7. **Limitations (30s):** LLM latency seconds; Action simulated; 5-fold Wilcoxon coarse (use bootstrap CIs); multiclass context script optional leftover.  
8. **Ask (15s):** Approve Q1 story = *agentic safety layer around fixed Tier-1*, SisFall-primary + KFall-external.

---

## Closing line

> Sir, this is Q1-oriented evidence for a **confidence-gated multi-agent wrapper** around a Bhatti-style detector: locked 5-fold SisFall gains, complementary KFall binary validation, ablations, paired LLM compares, and honest limits — same Tier-1 checkpoint, fair comparison, clinical cost \(10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\).

---

# PART F — Files to Open in the Meeting

1. **This file** — `results/IIT_PROFESSOR_PRESENTATION_COMPLETE.md`  
2. Short brief — `results/PROFESSOR_MEETING_BRIEF.md`  
3. Pack — `results/Q1_SUBMISSION_PACK.md`  
4. Paper — `paper/main.pdf`  
5. Architecture diagram (your corrected PNG)  
6. KFall flags — `results/kfall/KFALL_EXTERNAL_DONE.flag`, `KFALL_Q1_COMPLETE.flag`

---

# Quick Glossary

| Term | Meaning |
|------|---------|
| **Tier-1** | Bhatti-style CNN–LSTM–Attention detector alone |
| **Ours / stack** | `gate_knn_llm` full agentic pipeline |
| **F1** | Balance of precision & recall for Fall vs ADL |
| **Recall** | Fraction of real falls caught (↑ = fewer missed falls) |
| **Cost/1000** | \(10\cdot\mathrm{FN}+1\cdot\mathrm{FP}\) on 1000 windows |
| **±0.014** | Std-dev across 5 folds |
| **Fold** | One subject-held-out train/test split |
| **CoT** | Step-by-step rationale from LLM / heuristic |
| **Escalation** | Uncertain window sent to evidence+kNN+LLM |

---

*End of presentation pack. Good luck tomorrow.*
