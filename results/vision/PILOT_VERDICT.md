# Vision VLM Pilot Verdict

**Decision: DISCARD**

| Model | F1 | Recall | Spec | Cost/1k |
|---|---:|---:|---:|---:|
| CNN ResNet18 | 1.000 | 1.000 | 1.000 | 0.0 |
| Zero-shot VLM | 0.600 | 1.000 | 0.000 | 571.4 |
| LoRA VLM | 0.675 | 0.963 | 0.333 | 539.7 |

- beat_cnn: False
- beat_zeroshot: True
- criteria: `{"f1_ok": false, "recall_ok": false, "lora_minus_cnn_f1": -0.3246753246753247, "lora_minus_zs_f1": 0.07532467532467535, "lora_minus_cnn_recall": -0.03703703703703709, "lora_minus_cnn_spec": -0.6666666666666481}`
