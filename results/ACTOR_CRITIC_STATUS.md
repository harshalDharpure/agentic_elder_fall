# Actor–Critic (Q1-safe action critique)

Updated: 2026-08-15

## Decision

Keep **`gate_knn_llm`** as the paper detection table. Critique is **not** a second fall/ADL detector.

The failed 5-fold run (`gate_knn_llm_actor_critic`) flipped fall→ADL (Fold-0 FN 6→105). That path stays as a negative ablation only.

## Q1-safe path (`gate_knn_llm_action_critique`)

On **ambiguous windows only**:

1. **Actor** = same LLM as `gate_knn_llm` (fall/ADL frozen).
2. **Critic** (heuristic, Σ(w) on 90-frame bands) may:
   - ground the rationale in free-fall / impact / stillness / tilt / kNN / p(fall)
   - downgrade action `emergency → notify_caregiver → monitor`
3. Critic **cannot** change the fall/ADL label.

Default in `configs/agentic.yaml`: `adjudication.enabled: false`.

## 5-fold result (`max_test=1000`) — **PASS** (all jobs finished)

Recall and FN match `gate_knn_llm` on every fold (drop = 0). Tiny F1/FP wiggle is a second LLM Actor draw, not a label flip.

| Fold | `gate_knn_llm` F1 / Rec / FN | `action_critique` F1 / Rec / FN | grounding |
|---|---|---|---|
| 0 | 0.892 / 0.986 / 6 | 0.892 / 0.986 / 6 | 0.429 → 0.443 |
| 1 | 0.851 / 0.911 / 39 | 0.852 / 0.911 / 39 | 0.337 → 0.349 |
| 2 | 0.896 / 0.986 / 6 | 0.896 / 0.986 / 6 | 0.470 → 0.482 |
| 3 | 0.918 / 0.993 / 3 | 0.917 / 0.993 / 3 | 0.481 → 0.495 |
| 4 | 0.895 / 0.981 / 8 | 0.897 / 0.981 / 8 | 0.461 → 0.470 |

Main paper table stays `gate_knn_llm`. Critique is the response/explanation layer.

Outputs: `results/fold{N}/ablation_fold{N}_action_critique.json`

## Old (harmful) label-flip Critic — do not use in main table

| Fold | `gate_knn_llm` F1 / Rec | `actor_critic` F1 / Rec |
|---|---|---|
| 0 | 0.893 / 0.986 | 0.768 / 0.755 |
| 1–4 | similar drop | FN explosion |
