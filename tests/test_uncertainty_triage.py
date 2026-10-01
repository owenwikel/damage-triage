"""Calibration, uncertainty and triage maths on small made-up predictions."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.calibrate import calibration_report, fit_temperature, nll  # noqa: E402
from src.metrics import softmax  # noqa: E402
from src.model import build_model  # noqa: E402
from src.triage import (at_coverage, aurc, inspection_recall, oracle_recall,  # noqa: E402
                        risk_coverage_curve)
from src.uncertainty import (enable_mc_dropout, ensemble_probs, mc_dropout_probs,  # noqa: E402
                             mc_dropout_summary, method_scores, predictive_entropy)


def fake_logits(n=4000, scale=1.0, seed=0):
    """Labels drawn from softmax(z); returning z * scale makes the model over-confident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4)) * 2
    p = softmax(z)
    labels = np.array([rng.choice(4, p=row) for row in p])
    return z * scale, labels


# ---- calibration ---------------------------------------------------------------------

def test_temperature_recovers_a_known_overconfidence():
    logits, labels = fake_logits(scale=3.0)       # logits 3x too large -> T should be ~3
    t = fit_temperature(logits, labels)
    assert t == pytest.approx(3.0, rel=0.1)
    assert nll(logits, labels, t) < nll(logits, labels, 1.0)


def test_temperature_scaling_lowers_ece_and_keeps_accuracy():
    logits, labels = fake_logits(scale=3.0)
    rep = calibration_report(logits, labels, fit_temperature(logits, labels))
    assert rep["after"]["ece"] < rep["before"]["ece"] / 3
    assert rep["after"]["accuracy"] == rep["before"]["accuracy"]
    assert rep["after"]["mean_confidence"] < rep["before"]["mean_confidence"]


def test_well_calibrated_logits_get_temperature_near_one():
    logits, labels = fake_logits(scale=1.0)
    assert fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.1)


# ---- uncertainty ---------------------------------------------------------------------

def test_entropy_limits():
    assert predictive_entropy(np.array([[1.0, 0, 0, 0]]))[0] == pytest.approx(0.0, abs=1e-9)
    assert predictive_entropy(np.full((1, 4), 0.25))[0] == pytest.approx(np.log(4))


def test_ensemble_averages_probabilities():
    a = np.array([[10.0, 0, 0, 0]])
    b = np.array([[0.0, 10, 0, 0]])
    p = ensemble_probs([a, b])
    assert p.sum() == pytest.approx(1.0)
    assert p[0, 0] == pytest.approx(p[0, 1]) and p[0, 0] < 0.51     # two confident models disagree


def test_mc_dropout_keeps_batchnorm_frozen_and_dropout_on():
    model = enable_mc_dropout(build_model("siamese", pretrained=False))
    assert not model.backbone.bn1.training
    assert all(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))


def test_mc_dropout_passes_differ_but_plain_eval_does_not():
    torch.manual_seed(0)
    model = build_model("stack", pretrained=False)
    batch = [(torch.randn(6, 3, 64, 64), torch.randn(6, 3, 64, 64), torch.zeros(6))]
    probs = mc_dropout_probs(model, batch, torch.device("cpu"), passes=5)
    assert probs.shape == (5, 6, 4)
    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-5)
    assert not np.allclose(probs[0], probs[1])              # dropout is on
    assert not any(m.training for m in model.modules())     # and switched off again afterwards
    s = mc_dropout_summary(probs)
    assert s["probs"].shape == (6, 4) and s["entropy"].shape == (6,)
    assert (s["mean_max_prob"] >= s["probs"].max(axis=1) - 1e-6).all()   # mean of max >= max of mean


def test_method_scores_lists_what_is_available():
    logits, labels = fake_logits(200)
    only = method_scores(logits, labels)
    assert list(only) == ["max prob (raw)"]
    mc = mc_dropout_summary(np.stack([softmax(logits)] * 3))
    full = method_scores(logits, labels, temperature=2.0, mc=mc, ensemble_logits=[logits, logits * 0.5])
    assert list(full) == ["max prob (raw)", "max prob (temp. scaled)", "MC dropout (mean max prob)",
                          "MC dropout (entropy)", "ensemble of 2"]
    # temperature never changes the predicted class
    assert (full["max prob (raw)"]["pred"] == full["max prob (temp. scaled)"]["pred"]).all()
    assert (full["MC dropout (entropy)"]["confidence"] <= 0).all()


# ---- triage --------------------------------------------------------------------------

def test_risk_coverage_by_hand():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    cov, risk = risk_coverage_curve(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    assert aurc(conf, correct) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4]))


def test_good_ranking_beats_random_and_bad_ranking():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.8
    good = correct + rng.normal(scale=0.3, size=2000)       # confidence tracks correctness
    assert aurc(good, correct) < aurc(rng.random(2000), correct) < aurc(-good, correct)


def test_at_coverage():
    conf = np.linspace(1, 0, 10)                            # most confident first
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    rows = at_coverage(conf, y_true, y_pred, coverages=(1.0, 0.7, 0.5))
    assert [r["n"] for r in rows] == [10, 7, 5]
    assert [r["accuracy"] for r in rows] == pytest.approx([0.7, 1.0, 1.0])
    assert rows[1]["threshold"] == pytest.approx(conf[6])


def test_inspection_recall_and_oracle():
    damaged = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    priority = np.array([0.9, 0.8, 0.1, 0.7, 0, 0, 0, 0, 0, 0])
    assert inspection_recall(priority, damaged, budgets=(0.1, 0.2, 0.3)) == [0.5, 0.5, 1.0]
    assert oracle_recall(damaged, budgets=(0.1, 0.2, 0.5)) == [0.5, 1.0, 1.0]
