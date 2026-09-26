# Configuration and artifact manifest

Shipped with the Overleaf package so methods are inspectable without inventing
hyperparameters.

## Configs in this package (`configs/`)

| File | Role |
|------|------|
| `paper_protocol.yaml` | Locked Q1 evaluation protocol (seeds, budgets, costs, backends) |
| `tier1_sisfall.yaml` | SisFall windowing, channels, Tier-1 architecture defaults |
| `agentic.yaml` | Gate, evidence, retrieval, LLM, action, adjudication |
| `backbones.yaml` | Zoo architecture kwargs + shared train settings |
| `standard_sisfall.yaml` | Follow-up E1 standard-protocol eval |
| `standard_retrain.yaml` | Follow-up natural-prevalence retrain |
| `standard_g1_v2.yaml` / `_eval.yaml` | G1 v2 detector train/eval |

## Prompts (`prompts/`)

| File | Role |
|------|------|
| `llm_system_prompt.txt` | Primary reasoner system prompt |
| `actor_system.txt` / `critic_system.txt` | Actor–critic variants |
| `evidence_serializer.txt` | Evidence string field template |

## Companion experiment exports (parent repo)

Not duplicated inside the zip (large). Absolute root:
`/DATA/tauseef_2121cs04/harshal/agentic_fall/`

See `../RESULT_EXPORTS_INDEX.md` for full numeric tables and paths:

| Need | Path |
|------|------|
| Per-fold metrics + samples + case_ids | `results/fold{0..4}/` |
| E1 / retrain / G1 metrics | `results/standard_protocol*` |
| G2 adjudicator | `results/adjudicator_compare/` |
| Ambiguous stress | `results/kfold_ambiguous/` |
| Checkpoints | `checkpoints/` |

## Still author-captured at release time

- Ollama version string and immutable model digests for `mistral:latest` / Qwen tag
- GPU/CPU/RAM of the timing host
- Per-fold subject-ID manifests if not already exported beside checkpoints
