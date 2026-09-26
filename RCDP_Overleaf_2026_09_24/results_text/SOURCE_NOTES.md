# Author-supplied numeric records

Source: the author's “All results (numbers)” answer on 2026-09-24. Values are transcribed from that message, not fetched from the remote Linux experiment folders. No predictions were fabricated and no experiments rerun.

## Count records

[reported_counts.csv](reported_counts.csv) preserves:

- **Main ablation**, section 7: five distinct modes × five folds. The redundant `cnn_lstm_attn` rows equal `tier1_only` and are not duplicated in the CSV. All rows sum to 1000, including rows where the pasted `n` field was blank.
- **Agentic system**, section 9: five timing-run stack count rows, stored separately.
- **G2 adjudicator**, section 5: seven methods × five folds, 160 cases/fold.
- **LLM adjudicator**, section 8: three models × five folds, 150 cases/fold. CSV names `mistral` and `qwen_coder` map to the supplied `mistral:latest` and `qwen2.5-coder:7b-instruct` tags.

The LLM always-fall row is a deterministic mathematical control derived from per-fold label totals, not a new experimental run. Equal class totals are necessary but do not prove that methods used identical case IDs; the shared-ID claim is author-reported.

## Detector records

[backbone_f1.csv](backbone_f1.csv) preserves the E1 and retrained per-fold F1 columns from sections 1 and 2. [g1_v2.csv](g1_v2.csv) preserves selected columns from the separate v2 run, section 3. All are rounded supplied metrics; aggregates are recomputed from those values. Small last-digit differences from supplied aggregate summaries are retained transparently rather than forcing inconsistent arithmetic.

In particular, rounded E1 CNN--LSTM--Attention folds average 0.7174 (printed 0.717), while the supplied aggregate says 0.718. ModernTCN averages 0.7426 (printed 0.743), versus supplied 0.742. The v2 max-F1-labelled setting averages 0.7434 (printed 0.743), versus supplied 0.744. These differences are compatible with rounding of unavailable full-precision metrics; do not infer a new experiment or change fold rows to match the supplied aggregate.

Original single-fold zoo text remains archived in [tab_backbone.txt](tab_backbone.txt), not a mirror of the current five-fold table. The primary tables and CSV records take precedence over historical text mirrors/planning documents. Missing configurations, manifests, full-precision outputs and model digests remain author-dependent.