from __future__ import annotations

from typing import Iterable

import numpy as np

from agentic_fall.eval.metrics import binary_metrics


def cost_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    m = binary_metrics(y_true, y_pred)
    n = max(int(m["tp"] + m["tn"] + m["fp"] + m["fn"]), 1)
    total = cost_fn * m["fn"] + cost_fp * m["fp"]
    m["expected_cost"] = float(total)
    m["cost_per_1000"] = float(1000.0 * total / n)
    return m


def summarize_go_nogo(
    cnn: dict[str, float],
    zeroshot: dict[str, float],
    lora: dict[str, float],
    f1_margin: float = 0.02,
    recall_margin: float = 0.05,
    spec_slack: float = 0.05,
    zs_f1_gain: float = 0.05,
) -> dict:
    """Apply locked KEEP/DISCARD criteria from the pilot plan."""
    f1_ok = lora["f1"] >= cnn["f1"] + f1_margin
    recall_ok = (
        lora["recall"] >= cnn["recall"] + recall_margin
        and lora["specificity"] >= cnn["specificity"] - spec_slack
    )
    beat_cnn = bool(f1_ok or recall_ok)
    beat_zs = bool(lora["f1"] >= zeroshot["f1"] + zs_f1_gain)
    keep = beat_cnn and beat_zs
    return {
        "decision": "KEEP" if keep else "DISCARD",
        "beat_cnn": beat_cnn,
        "beat_zeroshot": beat_zs,
        "criteria": {
            "f1_ok": f1_ok,
            "recall_ok": recall_ok,
            "lora_minus_cnn_f1": float(lora["f1"] - cnn["f1"]),
            "lora_minus_zs_f1": float(lora["f1"] - zeroshot["f1"]),
            "lora_minus_cnn_recall": float(lora["recall"] - cnn["recall"]),
            "lora_minus_cnn_spec": float(lora["specificity"] - cnn["specificity"]),
        },
        "cnn": cnn,
        "zeroshot": zeroshot,
        "lora": lora,
    }
