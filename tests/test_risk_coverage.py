"""scripts/risk_coverage.py: the fast curves against a slow, obvious calculation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")        # calibrate_ensemble (imported by the script) needs it

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import risk_coverage as rc  # noqa: E402
from src.metrics import confusion_matrix, macro_f1  # noqa: E402
from test_calibrate_ensemble import RUNS, write_predictions  # noqa: E402


def fake(n=400, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(4) * 0.7, size=n)
    y_true = np.array([rng.choice(4, p=p) for p in probs])
    return probs, y_true, probs.argmax(axis=1)


def slow(priority, y_true, y_pred, rate):
    """The definition, spelled out: review the top ``rate`` % by priority, score the rest."""
    order = np.argsort(-priority, kind="stable")
    accepted = order[int(np.floor(rate / 100 * len(order) + 1e-9)):]
    t, p = y_true[accepted], y_pred[accepted]
    return {"accuracy": (t == p).mean(), "macro_f1": macro_f1(confusion_matrix(t, p)),
            "miss_rate": ((t > 0) & (p == 0)).sum() / (y_true > 0).sum()}


@pytest.mark.parametrize("score", ["confidence", "p_damaged", "no_damage_queue"])
def test_fast_curves_match_the_definition(score):
    probs, y_true, y_pred = fake()
    priority = rc.review_priority(probs)[score]
    fast = rc.Curves(priority, y_true, y_pred)()
    for rate in (0, 1, 10, 25, 50):
        expected = slow(priority, y_true, y_pred, rate)
        for metric in rc.METRICS:
            assert fast[metric][rate] == pytest.approx(expected[metric]), (metric, rate)


def test_least_confident_and_most_likely_damaged_are_reviewed_first():
    probs = np.array([[0.97, 0.01, 0.01, 0.01],     # sure, undamaged
                      [0.40, 0.30, 0.20, 0.10],     # unsure
                      [0.02, 0.02, 0.02, 0.94]])    # sure, destroyed
    priority = rc.review_priority(probs)
    assert priority["confidence"].argmax() == 1
    assert priority["p_damaged"].argmax() == 2


def test_no_damage_queue_reviews_only_predicted_no_damage_most_doubtful_first():
    probs = np.array([[0.97, 0.01, 0.01, 0.01],     # predicted no-damage, sure
                      [0.40, 0.30, 0.20, 0.10],     # predicted no-damage, doubtful
                      [0.02, 0.02, 0.02, 0.94],     # predicted destroyed: not in the queue
                      [0.60, 0.20, 0.10, 0.10],     # predicted no-damage, in between
                      [0.30, 0.45, 0.15, 0.10]])    # predicted minor: not in the queue
    priority = rc.review_priority(probs)["no_damage_queue"]
    assert np.argsort(-priority, kind="stable").tolist() == [1, 3, 0, 2, 4]
    # Buildings 1 and 3 are damaged and called no-damage; 2 and 4 are on the map.
    y_true, y_pred = np.array([0, 2, 3, 1, 0]), probs.argmax(axis=1)
    out = rc.Curves(priority, y_true, y_pred)(rates=[0, 20, 40])
    assert out["miss_rate"] == pytest.approx([2 / 3, 1 / 3, 0.0])     # 3 damaged in all
    assert out["queue_share"] == pytest.approx(3 / 5)
    # A whole queue reviewed leaves no misses, and reviewing the map changes nothing more.
    assert rc.Curves(priority, y_true, y_pred)(rates=[60, 80])["miss_rate"] == pytest.approx([0, 0])


def test_miss_rate_counts_only_accepted_damaged_buildings_called_no_damage():
    # 4 buildings; review 50% = the 2 with the highest priority (the first two).
    priority = np.array([4.0, 3.0, 2.0, 1.0])
    y_true = np.array([1, 0, 2, 3])
    y_pred = np.array([0, 0, 0, 3])       # misses: building 0 (reviewed) and building 2
    out = rc.Curves(priority, y_true, y_pred)(rates=[0, 50])
    assert out["miss_rate"] == pytest.approx([2 / 3, 1 / 3])     # 3 damaged in total
    assert out["accuracy"] == pytest.approx([0.5, 0.5])
    # the miss rate can never rise as more is reviewed
    probs, y_true, y_pred = fake()
    curve = rc.Curves(rc.review_priority(probs)["confidence"], y_true, y_pred)()["miss_rate"]
    assert (np.diff(curve) <= 1e-12).all()


def test_weights_act_like_repeated_buildings():
    probs, y_true, y_pred = fake(n=3000)
    priority = rc.review_priority(probs)["confidence"]
    weights = np.random.default_rng(1).integers(0, 3, size=3000)
    weighted = rc.Curves(priority, y_true, y_pred)(weights, rates=[0, 20, 40])
    rep = np.repeat(np.arange(3000), weights)     # physically copy the rows instead
    copied = rc.Curves(priority[rep], y_true[rep], y_pred[rep])(rates=[0, 20, 40])
    for metric in rc.METRICS:
        # Not exactly equal: a building with weight 2 is reviewed or accepted as a whole,
        # while its two copies can land on either side of the cut. One building at most.
        assert weighted[metric] == pytest.approx(copied[metric], abs=2e-3), metric
    assert weighted["accuracy"][0] == pytest.approx(copied["accuracy"][0])    # no cut at 0%


def test_bootstrap_draws_whole_tiles_and_is_reproducible():
    tiles = np.array(["a", "a", "b", "b", "b", "c"])
    draws = list(rc.tile_bootstrap_weights(tiles, n_boot=50, seed=3))
    for w in draws:
        assert w[0] == w[1] and w[2] == w[3] == w[4]        # a tile is in or out as a whole
        assert w.sum() in {2 * a + 3 * b + c for a in range(4) for b in range(4)
                           for c in range(4) if a + b + c == 3}        # 3 tiles drawn
    again = list(rc.tile_bootstrap_weights(tiles, n_boot=50, seed=3))
    assert all((a == b).all() for a, b in zip(draws, again))
    assert len({tuple(w) for w in draws}) > 1


def test_end_to_end_on_synthetic_predictions(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=300)
    doc, figs = tmp_path / "docs" / "rc.md", tmp_path / "docs" / "figures"
    rc.main(["--pred-dir", str(pred), "--runs", *RUNS, "--n-boot", "40",
             "--out", str(tmp_path / "out"), "--fig-dir", str(figs), "--doc", str(doc)])
    curves = pd.read_csv(tmp_path / "out" / "curves.csv")
    assert set(curves["part"]) == {"val-tiles", "val"}             # never test
    main_rows = curves[curves["score"].isin(rc.SCORES)]
    assert len(main_rows) == 2 * 3 * 3 * 51                 # parts x scores x metrics x rates
    assert (main_rows["lo"] <= main_rows["hi"]).all()
    # with nobody reviewed, all ranking scores describe the same set of buildings
    zero = main_rows[main_rows["review_rate"] == 0]
    assert zero.groupby(["part", "metric"])["value"].nunique().eq(1).all()
    # random order within the no-damage queue: starts at the same miss rate, never rises
    queue_random = curves[curves["score"] == "queue_random"].sort_values(["part", "review_rate"])
    assert set(queue_random["metric"]) == {"miss_rate"} and len(queue_random) == 2 * 51
    for part, d in queue_random.groupby("part"):
        start = zero[(zero["part"] == part) & (zero["metric"] == "miss_rate")]["value"].iloc[0]
        assert d["value"].iloc[0] == pytest.approx(start)
        assert (np.diff(d["value"]) <= 1e-12).all()
    assert (figs / "risk_coverage_accuracy.png").exists()
    assert (figs / "risk_coverage_miss_rate.png").exists()
    text = doc.read_text(encoding="utf-8")
    assert "### val-tiles" in text and "### val" in text and "| 30% |" in text
