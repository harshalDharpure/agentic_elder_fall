import numpy as np
import pytest
import torch

from agentic_fall.agents.confidence_gate import ConfidenceGate
from agentic_fall.agents.action_agent import ActionAgent
from agentic_fall.agents.evidence import serialize_evidence
from agentic_fall.agents.knn_memory import KNNMemory, MemoryCase
from agentic_fall.agents.llm_reasoner import LLMReasoner
from agentic_fall.features.biomechanics import extract_biomechanics
from agentic_fall.data.windowing import sliding_windows
from agentic_fall.data.kfall import segment_postfall, multiclass_logits_to_p_fall, fall_class_mask
from agentic_fall.data.splits import subject_folds
from agentic_fall.data.sisfall import (
    assert_required_activities,
    inventory_activity_counts,
    _force_include_activities,
    _parse_filename,
)
from agentic_fall.eval.metrics import agentic_metrics, binary_metrics
from agentic_fall.eval.protocol import (
    make_fold_split,
    subject_held_out_val,
    subsample_indices,
    tune_decision_threshold,
)
from agentic_fall.models import ALL_BACKBONES, CNNLSTMAttention, build_model, count_parameters
from pathlib import Path


def test_confidence_gate_routes():
    g = ConfidenceGate(0.3, 0.85)
    assert g.decide(0.1).route == "adl"
    assert g.decide(0.9).route == "fall"
    assert g.decide(0.5).route == "ambiguous"


def test_biomechanics_shapes():
    rng = np.random.default_rng(0)
    win = rng.normal(size=(90, 6)).astype(np.float32)
    win[20:30, :3] *= 0.1
    win[35, :3] = 4.0
    feats = extract_biomechanics(win, sample_rate_hz=200.0)
    assert feats.impact_magnitude_g > 1.0
    text = serialize_evidence(feats, activity="F01", p_fall=0.62)
    assert "Free-fall" in text
    assert "Impact" in text


def test_sliding_windows():
    x = np.zeros((200, 3), dtype=np.float32)
    w = sliding_windows(x, window=90, hop=10)
    assert w.shape[1:] == (90, 3)
    assert w.shape[0] == 12


def test_segment_postfall():
    t = 300
    acc = np.zeros((t, 3), dtype=np.float32)
    acc[:, 2] = 1.0
    acc[100] = [0, 0, 20]
    peak, stable = segment_postfall(acc, sample_rate_hz=100.0, rate_threshold=1.0)
    assert peak >= 99
    assert stable >= peak


def test_knn_memory():
    mem = KNNMemory(k=2)
    for i in range(5):
        emb = np.zeros(8, dtype=np.float32)
        emb[i % 8] = 1.0
        mem.add(
            MemoryCase(
                case_id=str(i),
                embedding=emb.tolist(),
                label=i % 2,
                evidence_text=f"case {i}",
                features={},
            )
        )
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    hits = mem.retrieve(q, k=2)
    assert len(hits) == 2


def test_heuristic_reasoner_and_action():
    win = np.zeros((90, 6), dtype=np.float32)
    win[:, 2] = 1.0
    win[40, :3] = 6.0
    win[50:80, :3] = 0.05
    feats = extract_biomechanics(win, sample_rate_hz=200.0)
    reasoner = LLMReasoner(backend="heuristic")
    res = reasoner.reason(serialize_evidence(feats), feats, p_fall=0.55, cases=[])
    assert res.prediction in ("fall", "adl")
    assert res.source == "heuristic"
    agent = ActionAgent()
    act = agent.decide(res.prediction, res.severity, feats, res.action)
    assert act.action in ("emergency", "notify_caregiver", "monitor", "log")


def test_model_forward():
    m = CNNLSTMAttention(in_channels=6, num_classes=2)
    x = torch.randn(4, 6, 90)
    logits, emb = m(x)
    assert logits.shape == (4, 2)
    assert emb.shape == (4, 128)


def test_modern_backbones_forward():
    x = torch.randn(2, 6, 90)
    for name in ALL_BACKBONES:
        m = build_model(name, in_channels=6, num_classes=2, seq_len=90)
        logits, emb = m(x)
        assert logits.shape == (2, 2), name
        assert emb.ndim == 2 and emb.size(0) == 2, name
        assert count_parameters(m) > 0


def test_assert_required_activities():
    assert_required_activities({"D18": 10, "D19": 5, "F01": 1})
    with pytest.raises(RuntimeError):
        assert_required_activities({"D01": 10, "F01": 1})


def test_force_include_activities(tmp_path: Path):
    # synthetic filenames only — Path objects need not exist for name parsing
    all_files = [
        Path("SA01/D01_SA01_R01.txt"),
        Path("SA01/D18_SA01_R01.txt"),
        Path("SA01/D19_SA01_R01.txt"),
        Path("SA01/F01_SA01_R01.txt"),
    ]
    selected = [all_files[0], all_files[3]]
    out = _force_include_activities(all_files, selected, ["D18", "D19"])
    acts = {_parse_filename(p.name)["activity"] for p in out}
    assert "D18" in acts and "D19" in acts


def test_agentic_metrics_near_fall_and_escalated():
    y_true = [0, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    esc = [True, False, True, False, False]
    acts = ["D18", "D01", "F01", "F02", "D19"]
    m = agentic_metrics(y_true, y_pred, esc, activities=acts, ambiguous_codes=["D18", "D19"])
    assert m["n_d18"] == 1
    assert m["n_d19"] == 1
    assert m["n_near_fall"] == 2
    assert m["n_escalated"] == 2
    assert m["false_alarm_rate_near_fall"] == 0.5  # one of two near-fall ADLs FP
    assert m["escalated_f1"] == m["ambiguous_f1"]
    assert m["false_alarm_rate_ambiguous"] == m["false_alarm_rate_near_fall"]


def test_agentic_metrics_no_near_fall_no_nan():
    m = agentic_metrics([0, 1], [0, 1], [False, False], activities=["D01", "F01"], ambiguous_codes=["D18", "D19"])
    assert m["n_near_fall"] == 0
    assert m["false_alarm_rate_near_fall"] == 0.0
    assert m["escalated_f1"] == 0.0


def test_subject_folds_disjoint():
    subjects = [f"S{i:02d}" for i in range(20) for _ in range(5)]
    strata = ["fall" if i < 10 else "adl" for i in range(20) for _ in range(5)]
    folds = subject_folds(subjects, n_folds=5, seed=42, strata=strata)
    for tr, te in folds:
        assert not (set(tr) & set(te))


def test_subject_held_out_val_no_leak():
    subjects = np.array([f"S{i%10}" for i in range(100)])
    idx = list(range(100))
    tr, va = subject_held_out_val(idx, subjects, val_fraction=0.2, seed=0)
    assert not ({subjects[i] for i in tr} & {subjects[i] for i in va})


def test_subsample_forces_near_falls():
    y = np.array([0] * 200 + [1] * 200)
    acts = np.array(["D01"] * 180 + ["D18"] * 10 + ["D19"] * 10 + ["F01"] * 200)
    idx = list(range(400))
    out = subsample_indices(
        idx, y, acts, n=50, seed=0, min_per_activity={"D18": 5, "D19": 5}
    )
    counts = inventory_activity_counts(acts[out])
    assert counts.get("D18", 0) >= 5
    assert counts.get("D19", 0) >= 5


def test_tune_decision_threshold():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    thr, m = tune_decision_threshold(y, p, mode="f1")
    assert 0.05 <= thr <= 0.95
    assert "f1" in m


def test_kfall_p_fall_helper():
    logits = torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    mask = np.array([False, True, False])
    p = multiclass_logits_to_p_fall(logits, mask)
    assert p.shape == (2,)
    assert float(p[0]) > float(p[1])


def test_make_fold_split_toy():
    class Toy:
        def __init__(self):
            self.subjects = np.array([f"S{i}" for i in range(10) for _ in range(20)])
            self.y = np.array([i % 2 for i in range(10) for _ in range(20)])
            self.activities = np.array(
                (["D18"] * 5 + ["D19"] * 5 + ["F01"] * 10) * 10
            )

        def fold_indices(self, n_folds=5, seed=42):
            # reuse real subject_folds
            from agentic_fall.data.splits import subject_folds

            subj = [str(s) for s in self.subjects]
            folds = subject_folds(subj, n_folds=n_folds, seed=seed)
            out = []
            for tr_s, te_s in folds:
                tr, te = set(tr_s), set(te_s)
                train_idx = [i for i, s in enumerate(subj) if s in tr]
                test_idx = [i for i, s in enumerate(subj) if s in te]
                out.append((train_idx, test_idx))
            return out

    split = make_fold_split(Toy(), fold=0, n_folds=5, seed=42, val_fraction=0.2)
    assert split.train_idx and split.val_idx and split.test_idx
