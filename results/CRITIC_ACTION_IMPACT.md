# Critic impact (Option A): action_critique vs gate_knn_llm

Labels are frozen. Impact should appear in graded action cost and rationale grounding.

| Fold | Mode | F1 | Recall | Cost | GradedAction | EmADL | Ground |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | gate_knn_llm | 0.892 | 0.986 | 156 | 544 | 0.000 | 0.429 |
| 0 | gate_knn_llm_action_critique | 0.892 | 0.986 | 156 | 544 | 0.000 | 0.443 |
| 1 | gate_knn_llm | 0.851 | 0.911 | 490 | 895 | 0.000 | 0.337 |
| 1 | gate_knn_llm_action_critique | 0.852 | 0.911 | 489 | 891 | 0.000 | 0.349 |
| 2 | gate_knn_llm | 0.896 | 0.986 | 153 | 535 | 0.000 | 0.470 |
| 2 | gate_knn_llm_action_critique | 0.896 | 0.986 | 153 | 535 | 0.000 | 0.482 |
| 3 | gate_knn_llm | 0.918 | 0.993 | 105 | 428 | 0.000 | 0.481 |
| 3 | gate_knn_llm_action_critique | 0.917 | 0.993 | 106 | 432 | 0.000 | 0.495 |
| 4 | gate_knn_llm | 0.895 | 0.981 | 171 | 540 | 0.004 | 0.461 |
| 4 | gate_knn_llm_action_critique | 0.897 | 0.981 | 169 | 528 | 0.002 | 0.470 |

## Means

- **gate_knn_llm**: F1=0.890, graded_action=588.4, grounding=0.436
- **action_critique**: F1=0.891, graded_action=586.0, grounding=0.448

## Readout

Old action_critique barely changes detection (by design). Grounding rises a little; graded cost barely moves because Critic did not screen confident-path falls. The new `gate_knn_llm_contrastive_action_critique` mode screens those windows for action downgrades and feeds hard-negative ADLs to the Critic.
