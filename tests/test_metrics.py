import numpy as np
import pytest

from src.metrics import (confusion_matrix, expected_calibration_error, macro_f1,
                         per_class_f1, per_class_recall, reliability_bins, softmax, summarize)


def test_confusion_matrix_rows_are_truth():
    cm = confusion_matrix([0, 0, 1, 3], [0, 1, 1, 0])
    assert cm.shape == (4, 4)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[3, 0] == 1
    assert cm.sum() == 4


def test_recall_and_f1_by_hand():
    # class 0: 3 true, 2 found; one class-1 building wrongly called 0
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    recall = per_class_recall(cm)
    assert recall[0] == pytest.approx(2 / 3) and recall[1] == pytest.approx(1 / 2)
    assert np.isnan(recall[2])                      # no class-2 buildings at all
    f1 = per_class_f1(cm)
    assert f1[0] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert f1[1] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    assert macro_f1(cm) == pytest.approx((f1[0] + f1[1]) / 2)   # absent classes are skipped


def test_majority_guess_has_high_accuracy_but_low_macro_f1():
    y_true = np.array([0] * 90 + [1] * 4 + [2] * 3 + [3] * 3)
    s = summarize(y_true, np.zeros(100, dtype=int))
    assert s["accuracy"] == pytest.approx(0.9)
    assert s["macro_f1"] < 0.25
    assert s["recall"] == [1.0, 0.0, 0.0, 0.0]


def test_softmax_rows_sum_to_one_and_survive_large_logits():
    p = softmax(np.array([[1000.0, 0.0, -1000.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] == pytest.approx(1.0) and np.allclose(p[1], 0.25)


def test_ece_is_zero_when_confidence_equals_accuracy():
    conf = np.full(1000, 0.8)
    correct = np.array([1] * 800 + [0] * 200)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-12)
    # always 100% sure but right only 60% of the time: off by 0.4
    assert expected_calibration_error(np.ones(10), [1] * 6 + [0] * 4) == pytest.approx(0.4)


def test_reliability_bins_cover_every_prediction():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.25, 1.0, 500)
    bins = reliability_bins(conf, rng.random(500) < conf, n_bins=10)
    assert sum(n for _, _, n in bins) == 500
