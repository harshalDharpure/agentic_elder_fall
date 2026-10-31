"""Regression checks for split hygiene and paired campaign reporting."""
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_memory_excludes_inner_validation_and_outer_test():
    build = runpy.run_path(str(ROOT / "scripts/build_memory.py"))["memory_training_indices"]

    class Dataset:
        subjects = [f"subject_{index // 3}" for index in range(30)]

        def fold_indices(self, n_folds, seed):
            return [(list(range(24)), list(range(24, 30)))]

    dataset = Dataset()
    config = {"train": {"seed": 42}, "splits": {"n_folds": 5}}
    indices, split = build(dataset, config, {"splits": {"val_fraction": 0.25}}, 0, 100)
    subjects = {dataset.subjects[index] for index in indices}
    assert indices
    assert set(indices) == set(split.train_idx)
    assert not subjects & {dataset.subjects[index] for index in split.val_idx + split.test_idx}
    sampled, _ = build(dataset, config, {"splits": {"val_fraction": 0.25}}, 0, 4)
    assert len(sampled) == 4 and set(sampled) <= set(indices)


def test_paired_summary_uses_requested_folds_only_and_percentage_points(tmp_path):
    aggregate = runpy.run_path(str(ROOT / "scripts/run_sisfall_mamba.py"))["aggregate"]
    for model, values in (("cnn_lstm_attn", [0.7, 0.8, 0.99]), ("harmamba", [0.8, 0.9, 0.1])):
        checkpoints = tmp_path / model / "checkpoints"
        checkpoints.mkdir(parents=True)
        for fold, value in enumerate(values):
            (checkpoints / f"{model}_fold{fold}_metrics.json").write_text(json.dumps({"test": {"f1": value}}))
    summary = aggregate(tmp_path, ["cnn_lstm_attn", "harmamba"], [0, 1], True)
    assert summary["smoke_only"] is True
    assert summary["models"]["harmamba"]["f1_mean"] == pytest.approx(0.85)
    assert summary["mamba_minus_cnn_f1_percentage_points"] == pytest.approx(10.0)
