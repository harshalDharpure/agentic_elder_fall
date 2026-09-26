# AEGIS — Adaptive Evidence-Grounded Individualised Selection

Post-hoc decision layer for wearable fall detection. AEGIS does **not** replace the detector; it protects the decision the detector makes. Detector stays frozen — that is what enables the nine-backbone claim.

## Strategic claim (Q1 / JBHI)

> Nine architectures spanning 2015–2024, one protocol. All exceed 0.95 F1 on standard SisFall. All collapse to 0.66–0.74 on near-falls. AEGIS lifts every one above 0.87 **without retraining**, at a bounded escalation budget (B ≤ 10%).

Honest (not a better detector), general (nine backbones), unfalsifiable by “try ModernTCN” (already done).

## Three novelty claims

| ID | Claim |
|----|--------|
| **N1** | Subject-conditional (Mondrian) conformal **risk control** targeting FN rate, under an explicit duty-cycle budget B |
| **N2** | Evidence bottleneck (stretch; only if Weeks 1–5 early) — faithfulness by construction |
| **N3** | Backbone-agnostic near-fall benchmark, released as a protocol others can run |

## Architecture

```
IMU window → T0 frozen detector f_θ (ANY backbone)
           → T1 Individualised Risk Gate (Mondrian CRC, FN rate, budget B)
                ├── confident → action
                └── ambiguous → T2 evidence concepts
                              → T3 case-grounded reasoner (slot: LLM or GBM)
                              → T4 risk-calibrated action policy
```

**Design property:** G2 changes which module fills T3, not the architecture. If GBM on biomechanical features matches Mistral, T3 becomes GBM and the LLM becomes the explanation generator.

## Layer justification

| Layer | Novel? | Why it exists |
|-------|--------|---------------|
| T0 frozen detector | No (deliberate) | Enables 9-backbone wrap |
| T1 individualised gate | **Yes (N1)** | Fixes fold-4 marginal conformal violation; bounds duty cycle; subject shift |
| T2 evidence | Partial | Biomechanical features already in `features/biomechanics.py` |
| T3 reasoner | Slot | Write-back / cascade / abstention = refinements |
| T4 action policy | Partial | Frozen labels; critique may only downgrade |

## Prior-work answers (one-liners)

- Softmax force-labels every window → T1 deferral  
- SisFall authors: D04/D18/F11 dominate thresholds → our near-fall benchmark  
- Literature 0.97 on natural prevalence, no near-fall → R1 collapse table  
- SelectiveNet / Deep Gamblers need reject-head retrain → AEGIS is post-hoc on frozen zoo  
- Young-adult train → elderly deploy bias → T1 individualisation + elderly subgroup  

## Build order (gates first)

0. Fix D19 / elderly disclosure; run E1 standard protocol → **G1**  
1. R1 + R2 across zoo (mostly existing checkpoints)  
2. E2 adjudicator compare → **G2 decides T3**  
3. T1 Mondrian CRC + budget → risk–coverage  
4. LOSO, elderly, D05/D11  
5. Write / submit JBHI  

Do not build T1–T4 before G1 and G2.
