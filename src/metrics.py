"""Classification metrics, written with numpy so they can be tested without a GPU.

Why not just accuracy? About three quarters of the buildings are undamaged. A model that
always answers "no-damage" is ~75% accurate and useless. So alongside accuracy we report:

  recall of class c   of the buildings that truly are c, what share did we call c?
  precision of c      of the buildings we called c, what share truly are c?
  F1 of c             the harmonic mean of the two: high only if both are high
  macro-F1            the plain average of the four per-class F1 scores. Every class
                      counts the same, however rare, so ignoring a rare class costs a
                      full quarter of the score. This is the number used for early
                      stopping and for comparing models.
"""

import numpy as np

NUM_CLASSES = 4


def confusion_matrix(y_true, y_pred, n_classes=NUM_CLASSES):
    """cm[i, j] = number of buildings whose true class is i and predicted class is j."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    # Encode each (true, predicted) pair as one number, count them, and reshape.
    return np.bincount(y_true * n_classes + y_pred,
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_recall(cm):
    """Diagonal / row sums. NaN for a class with no true examples (nothing to recall)."""
    support = cm.sum(axis=1)
    return np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)


def per_class_precision(cm):
    predicted = cm.sum(axis=0)
    return np.where(predicted > 0, np.diag(cm) / np.maximum(predicted, 1), np.nan)


def per_class_f1(cm):
    """F1 = 2 TP / (2 TP + FP + FN). A class never predicted and never present scores NaN;
    a class that is present but never predicted (or predicted but never present) scores 0.
    """
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)       # = 2 TP + FP + FN
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def macro_f1(cm):
    """Mean F1 over the classes that occur (as truth or prediction)."""
    f1 = per_class_f1(cm)
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


def accuracy(cm):
    return float(np.diag(cm).sum() / max(cm.sum(), 1))


def summarize(y_true, y_pred, n_classes=NUM_CLASSES):
    """All the numbers we log for one set of predictions, as plain Python types."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return {
        "n": int(cm.sum()),
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "recall": [None if np.isnan(r) else float(r) for r in per_class_recall(cm)],
        "f1": [None if np.isnan(f) else float(f) for f in per_class_f1(cm)],
        "confusion": cm.tolist(),
    }


def softmax(logits):
    """Turn scores into probabilities that sum to 1 (numerically stable version)."""
    z = logits - logits.max(axis=1, keepdims=True)   # subtracting the max avoids overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(confidence, correct, n_bins=15):
    """ECE: how far stated confidence is from actual accuracy, on average.

    Sort predictions into ``n_bins`` equal-width confidence bins (0-0.067, ...). In a
    perfectly calibrated model, the predictions in the "0.8-0.87" bin are right 80-87% of
    the time. ECE is the gap |accuracy - mean confidence| per bin, averaged with each bin
    weighted by how many predictions fall in it. 0 is perfect; 0.10 means confidence is
    off by ten percentage points on average.
    """
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_bins(confidence, correct, n_bins=15):
    """Per-bin (mean confidence, accuracy, count) for a reliability diagram."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            rows.append((float(confidence[in_bin].mean()), float(correct[in_bin].mean()),
                         int(in_bin.sum())))
    return rows
