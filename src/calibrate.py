"""Calibration: make the model's confidence mean what it says.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.calibrate --run runs\\stack_aug_s0

The problem
-----------
The model outputs four scores ("logits") per building; softmax turns them into
probabilities. We would like "0.9" to mean "right 9 times out of 10", because the triage
idea depends on it: auto-accept confident predictions, send unsure ones to a person. But
networks trained with cross-entropy are usually OVER-confident: the loss keeps rewarding
pushing probabilities towards 1.0 on the training set long after accuracy has stopped
improving.

Temperature scaling
-------------------
The simplest fix that works: divide all logits by one number T before the softmax.

    probabilities = softmax(logits / T)

  T > 1  flattens the probabilities (less confident)
  T < 1  sharpens them (more confident)
  T = 1  changes nothing

Dividing every logit by the same positive number never changes which one is largest, so
the predicted class, and therefore accuracy, F1 and the confusion matrix, stay exactly
the same. Only the confidence changes. T is a single number, fitted on held-out data
(val-tiles) by minimising the negative log-likelihood (the ordinary, unweighted
cross-entropy). With one parameter there is essentially nothing to overfit.

What it cannot do: one global T cannot fix a model that is over-confident on one kind of
building and under-confident on another. And a T fitted on val-tiles (same disasters as
training) may not be right for a new disaster. That is why ECE is reported on val too.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.metrics import expected_calibration_error, reliability_bins, softmax


def nll(logits, labels, temperature=1.0):
    """Mean negative log-likelihood: -log(probability given to the true class)."""
    p = softmax(np.asarray(logits, dtype=np.float64) / temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits, labels):
    """Find the T that minimises the NLL of softmax(logits / T). Returns a float.

    We optimise log(T) rather than T so that T stays positive whatever the optimiser
    does. L-BFGS is a classic optimiser for small smooth problems like this one: no
    learning rate to tune, converges in a handful of steps.
    """
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)     # T starts at 1
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        # L-BFGS evaluates the loss several times per step, so it wants a function.
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def calibrated_probs(logits, temperature):
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def calibration_report(logits, labels, temperature):
    """ECE and NLL before and after scaling, for one set of predictions."""
    out = {}
    for name, t in (("before", 1.0), ("after", temperature)):
        p = calibrated_probs(logits, t)
        conf, pred = p.max(axis=1), p.argmax(axis=1)
        out[name] = {"ece": expected_calibration_error(conf, pred == labels),
                     "nll": nll(logits, labels, t), "mean_confidence": float(conf.mean()),
                     "accuracy": float((pred == labels).mean())}
    return out


def plot_reliability(ax, confidence, correct, title, color="#2a78d6", n_bins=15):
    """Reliability diagram: accuracy against confidence, bin by bin.

    A perfectly calibrated model sits on the diagonal. Points BELOW the diagonal are
    over-confident (said 0.9, was right 0.7). The grey bars show how many predictions
    fall in each bin, because a wild point backed by 12 buildings matters much less than
    a small gap backed by 12,000.
    """
    rows = reliability_bins(confidence, correct, n_bins)
    conf, acc, n = (np.array(c) for c in zip(*rows))
    share = n / n.sum()
    ax.bar(conf, share, width=1 / n_bins * 0.9, color="#e1e0d9", zorder=1,
           label="share of buildings")
    ax.plot([0, 1], [0, 1], color="#898781", linestyle="--", linewidth=1, zorder=2,
            label="perfect calibration")
    ax.plot(conf, acc, marker="o", markersize=4, color=color, linewidth=1.8, zorder=3,
            label="model")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ece = expected_calibration_error(confidence, correct, n_bins)
    ax.set_title(f"{title}\nECE {ece:.3f}", fontsize=10)
    ax.grid(color="#e1e0d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="a folder under runs/")
    args = ap.parse_args()
    fit = np.load(args.run / "preds_val-tiles.npz")
    t = fit_temperature(fit["logits"], fit["labels"])
    report = {"temperature": t, "fitted_on": "val-tiles"}
    for part in ("val-tiles", "val"):
        d = np.load(args.run / f"preds_{part}.npz")
        report[part] = calibration_report(d["logits"], d["labels"], t)
        b, a = report[part]["before"], report[part]["after"]
        print(f"{part:9s} ECE {b['ece']:.4f} -> {a['ece']:.4f}   NLL {b['nll']:.4f} -> {a['nll']:.4f}")
    (args.run / "temperature.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"temperature {t:.3f} (fitted on val-tiles) -> {args.run / 'temperature.json'}")


if __name__ == "__main__":
    main()
