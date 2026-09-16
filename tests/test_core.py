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
                activity="D18" if i % 2 == 0 else "F01",
            )
        )
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    hits = mem.retrieve(q, k=2)
    assert len(hits) == 2
    pos, neg = mem.contrastive_retrieve(q, k_pos=2, k_neg=2)
    assert all(c.label == 1 for c in pos)
    assert all(c.label == 0 for c in neg)
    assert len(pos) <= 2 and len(neg) <= 2


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


def test_constraint_pack_and_actor_critic():
    from agentic_fall.agents.constraints import ConstraintPack, build_constraint_pack, rationale_cites_sigma
    from agentic_fall.agents.adjudicator import Adjudicator
    from agentic_fall.agents.critic_agent import ActorHypothesis, CriticAgent, _heuristic_critic

    assert rationale_cites_sigma("Free-fall 0.2s and impact 4g support a fall")
    assert not rationale_cites_sigma("looks bad")

    win = np.zeros((90, 6), dtype=np.float32)
    win[:, 2] = 1.0
    win[10:14, :3] = 0.2
    win[20, :3] = 2.5
    feats = extract_biomechanics(win, sample_rate_hz=200.0)
    pack = build_constraint_pack(feats, p_fall=0.45, cases=[])
    assert pack.freefall_band in ("short", "moderate", "long")
    assert "Free-fall" in pack.checklist_text()

    reasoner = LLMReasoner(backend="heuristic")
    critic = CriticAgent(backend="heuristic")
    adj = Adjudicator(
        reasoner=reasoner, critic=critic, enabled=True, mode="action_critique", freeze_label=True
    )
    res, trace = adj.reason(serialize_evidence(feats, p_fall=0.45), feats, 0.45, [])
    assert res.prediction in ("fall", "adl")
    assert trace.actor is not None
    assert trace.critic is not None
    assert trace.actor.prediction == res.prediction

    actor_fall = ActorHypothesis(
        hypothesis="fall",
        prediction="fall",
        severity="severe",
        confidence=0.9,
        rationale="I just know",
        suggested_action="emergency",
        source="heuristic",
    )
    v = _heuristic_critic(actor_fall, pack)
    assert v.revised_prediction == "fall"
    assert v.verdict == "revise"
    assert any("rationale" in x.lower() for x in v.violations)

    weak_pack = ConstraintPack(
        freefall_duration_s=0.04,
        impact_magnitude_g=2.5,
        post_impact_stillness_s=0.05,
        body_tilt_deg=10.0,
        p_fall=0.45,
        knn_fall_votes=0,
        knn_adl_votes=0,
        knn_fall_ratio=0.5,
        freefall_band="short",
        impact_band="high",
        stillness_band="brief",
        tilt_band="upright",
        fall_support_score=0.05,
        near_fall_likely=True,
    )
    v_down = _heuristic_critic(actor_fall, weak_pack)
    assert v_down.revised_prediction == "fall"
    assert v_down.revised_action != "emergency"
    # Stronger Critic: weak Σ + notify should become monitor
    actor_notify = ActorHypothesis(
        hypothesis="fall",
        prediction="fall",
        severity="moderate",
        confidence=0.9,
        rationale="Free-fall short; impact moderate; tilt upright; kNN mostly ADL",
        suggested_action="notify_caregiver",
        source="heuristic",
    )
    weak_pack_hard = ConstraintPack(
        freefall_duration_s=0.04,
        impact_magnitude_g=2.5,
        post_impact_stillness_s=0.05,
        body_tilt_deg=10.0,
        p_fall=0.45,
        knn_fall_votes=1,
        knn_adl_votes=4,
        knn_fall_ratio=0.2,
        freefall_band="short",
        impact_band="moderate",
        stillness_band="brief",
        tilt_band="upright",
        fall_support_score=0.05,
        near_fall_likely=True,
    )
    v_mon = _heuristic_critic(actor_notify, weak_pack_hard)
    assert v_mon.revised_prediction == "fall"
    assert v_mon.revised_action == "monitor"
    assert v_mon.verdict == "revise"

    actor_adl = ActorHypothesis(
        hypothesis="adl",
        prediction="adl",
        severity="severe",
        confidence=0.8,
        rationale="I just know",
        suggested_action="emergency",
        source="heuristic",
    )
    v_adl = _heuristic_critic(actor_adl, weak_pack)
    assert v_adl.revised_prediction == "adl"
    assert v_adl.revised_action in ("monitor", "log")

    win2 = np.zeros((90, 6), dtype=np.float32)
    win2[:, 2] = 1.0
    win2[10:40, :3] = 0.1
    win2[45, :3] = 6.0
    win2[50:85, :3] = 0.02
    win2[-10:, 0] = 1.0
    feats2 = extract_biomechanics(win2, sample_rate_hz=200.0)
    res2, trace2 = adj.reason(serialize_evidence(feats2, p_fall=0.8), feats2, 0.8, [])
    assert res2.prediction == trace2.actor.prediction
    assert trace2.mode == "action_critique"

    from agentic_fall.agents.constraints import veto_score
    from agentic_fall.eval.conformal import (
        alpha_break_even,
        apply_veto,
        empirical_bernstein_upper,
        hoeffding_upper,
        select_lambda_star,
    )

    assert 0.0 <= veto_score(weak_pack) <= 1.0
    strong = ConstraintPack(
        freefall_duration_s=0.20,
        impact_magnitude_g=6.0,
        post_impact_stillness_s=0.30,
        body_tilt_deg=70.0,
        p_fall=0.9,
        knn_fall_votes=5,
        knn_adl_votes=0,
        knn_fall_ratio=1.0,
        freefall_band="long",
        impact_band="severe",
        stillness_band="prolonged",
        tilt_band="horizontal",
        fall_support_score=0.9,
        near_fall_likely=False,
    )
    assert veto_score(weak_pack) > veto_score(strong)

    rng = np.random.default_rng(0)
    # High scores on ADL (y=0), low scores on falls (y=1) → vetoing high scores is safe.
    scores = list(rng.uniform(0.7, 1.0, 200)) + list(rng.uniform(0.0, 0.3, 50))
    labels = [0] * 200 + [1] * 50
    sel = select_lambda_star(scores, labels, alpha=0.05, delta=0.1, min_n=20)
    assert sel["feasible"] is True
    assert sel["lambda_star"] is not None
    assert hoeffding_upper(0.01, 800, 0.1) < 0.05
    assert empirical_bernstein_upper([0.0] * 200, 0.1) < 0.1
    assert abs(alpha_break_even(10.0, 1.0) - 1.0 / 11.0) < 1e-9
    assert apply_veto(["fall", "fall", "adl"], [0.9, 0.1, 0.9], 0.5) == ["adl", "fall", "adl"]

    adj_crc = Adjudicator(
        reasoner=reasoner,
        critic=critic,
        enabled=True,
        mode="crc_veto",
        freeze_label=False,
        veto_threshold=0.0,
        crc_feasible=True,
        crc_alpha=0.05,
        actor_backend="heuristic",
        critic_backend="heuristic",
    )
    res_v, tr_v = adj_crc.reason(serialize_evidence(feats, p_fall=0.45), feats, 0.45, [])
    if tr_v.actor and tr_v.actor.prediction == "fall":
        assert res_v.prediction == "adl"
        assert tr_v.vetoed is True
    adj_infeas = Adjudicator(
        reasoner=reasoner,
        critic=critic,
        enabled=True,
        mode="crc_veto",
        freeze_label=False,
        veto_threshold=0.0,
        crc_feasible=False,
        actor_backend="heuristic",
        critic_backend="heuristic",
    )
    res_i, tr_i = adj_infeas.reason(serialize_evidence(feats, p_fall=0.45), feats, 0.45, [])
    assert tr_i.vetoed is False
    assert res_i.prediction == tr_i.actor.prediction


def test_action_downgrade():
    feats = extract_biomechanics(np.zeros((90, 6), dtype=np.float32), sample_rate_hz=200.0)
    agent = ActionAgent()
    up = agent.decide("fall", "mild", feats, suggested_action="emergency", allow_downgrade=False)
    assert up.action == "emergency"
    down = agent.decide("fall", "severe", feats, suggested_action="monitor", allow_downgrade=True)
    assert down.action == "monitor"


def test_kfold_ambiguous_aggregate():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    npz = root / "data/processed/sisfall/windows_w90_h10.npz"
    if not npz.exists():
        import pytest

        pytest.skip("SisFall NPZ not available")

    spec = importlib.util.spec_from_file_location(
        "build_kfold_ambiguous_bench",
        root / "scripts" / "build_kfold_ambiguous_bench.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    from agentic_fall.data.sisfall import SisFallDataset

    ds = SisFallDataset(npz, channels=6)
    payload, meta = mod.build_fold_subset(
        ds, fold=0, n_ambiguous=50, n_fall=25, n_adl=25, seed=0
    )
    assert payload["X"].ndim == 3 and payload["X"].shape[1:] == (90, 6)
    assert meta["n_ambiguous"] == 50
    assert meta["n_d18"] + meta["n_d19"] == 50
    assert meta["n_strict_fall"] == 25
    assert meta["n_clear_adl"] == 25
    assert "source_indices" in payload
    assert all(m["source"] == "sisfall_kfold" for m in meta["cases"])

    agg_spec = importlib.util.spec_from_file_location(
        "aggregate_kfold_ambiguous",
        root / "scripts" / "aggregate_kfold_ambiguous.py",
    )
    agg = importlib.util.module_from_spec(agg_spec)
    assert agg_spec.loader is not None
    agg_spec.loader.exec_module(agg)
    tmp = root / "results" / "_test_kfold_agg"
    tmp.mkdir(parents=True, exist_ok=True)
    import json

    fake = {
        "metrics": {
            "accuracy": 0.9,
            "f1": 0.85,
            "precision": 0.8,
            "recall": 0.9,
            "fp": 10,
            "tn": 90,
            "escalate_rate": 0.5,
            "expected_response_cost": 100.0,
            "per_bucket": {"ambiguous": {"accuracy": 0.95}},
        },
        "backend": "test",
    }
    for f in range(5):
        (tmp / f"fold{f}_eval.json").write_text(json.dumps(fake))
    summary = agg.aggregate(tmp, backend="test")
    assert len(summary["folds"]) == 5
    assert (tmp / "ALL_FOLDS_REPORT.txt").exists()

