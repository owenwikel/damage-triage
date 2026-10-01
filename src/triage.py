"""Triage evaluation: what does a confidence score buy us?

The use case: after a disaster there are far more buildings than inspectors. If the model
can say which of its own predictions to trust, the trusted ones can be accepted
automatically and the doubtful ones sent to a person. This file measures how well a
confidence score supports that, in two views.

1. Risk-coverage (selective prediction)
   Sort the buildings from most to least confident and accept the top X%.
       coverage = share of buildings auto-accepted
       risk     = error rate among the accepted ones
   A useful confidence score makes risk fall as coverage shrinks: the 50% the model is
   surest about should be nearly all correct. A useless score gives a flat line at the
   overall error rate. AURC (area under the risk-coverage curve) sums this up in one
   number; lower is better. It mixes two things, how accurate the model is and how well
   the score ranks its mistakes, so compare AURC only between methods on the same data.

2. Inspection budget
   Inspectors can visit only K% of the buildings. Which K%? Two policies:
       least confident first     visit where the model is unsure
       most likely damaged first visit where P(damaged) = 1 - P(no-damage) is highest
   Measured by: what share of the truly damaged buildings (minor, major or destroyed)
   is among those visited? Random visits reach K% of them; a perfect ranking reaches
   all of them as soon as K% covers the damaged share.
   These answer different questions. "Least confident" finds the model's likely
   mistakes; "most likely damaged" finds damage. They need not agree.
"""

import numpy as np

from src.metrics import confusion_matrix, macro_f1

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
BUDGETS = tuple(np.arange(5, 51, 5) / 100)      # 5%, 10%, ..., 50%


def risk_coverage_curve(confidence, correct):
    """Return (coverage, risk) arrays of length N.

    Entry k describes "accept the k+1 most confident predictions": coverage = (k+1)/N and
    risk = share of those that are wrong. Ties in confidence are broken by original
    order (stable sort), so the result is deterministic.
    """
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="stable")
    wrong = 1.0 - np.asarray(correct, dtype=np.float64)[order]
    n = np.arange(1, len(wrong) + 1)
    return n / len(wrong), np.cumsum(wrong) / n


def aurc(confidence, correct):
    """Area under the risk-coverage curve = mean risk over all coverage levels."""
    _, risk = risk_coverage_curve(confidence, correct)
    return float(risk.mean())


def at_coverage(confidence, y_true, y_pred, coverages=COVERAGES):
    """Accuracy and macro-F1 on the auto-accepted buildings at each coverage level.

    Returns a list of dicts with: coverage, n accepted, the confidence threshold that
    gives this coverage, accuracy and macro-F1 among the accepted.

    Note on macro-F1 here: the accepted set is a biased sample (mostly easy, mostly
    no-damage buildings), so rare classes can become very rare in it and their F1 noisy.
    Accuracy says "how often is an auto-accepted answer right"; macro-F1 says "is it
    still right across all four classes, or only on the easy one".
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        keep = order[:k]
        cm = confusion_matrix(y_true[keep], y_pred[keep])
        rows.append({"coverage": c, "n": k, "threshold": float(confidence[keep[-1]]),
                     "accuracy": float((y_true[keep] == y_pred[keep]).mean()),
                     "macro_f1": macro_f1(cm)})
    return rows


def inspection_recall(priority, is_damaged, budgets=BUDGETS):
    """Share of truly damaged buildings reached when visiting the top K% by ``priority``.

    ``priority``: higher = visit sooner. Returns one value per budget.
    """
    priority = np.asarray(priority, dtype=np.float64)
    is_damaged = np.asarray(is_damaged, dtype=bool)
    order = np.argsort(-priority, kind="stable")
    found = np.cumsum(is_damaged[order])
    total = max(int(is_damaged.sum()), 1)
    return [float(found[max(1, int(round(b * len(order)))) - 1] / total) for b in budgets]


def oracle_recall(is_damaged, budgets=BUDGETS):
    """Best possible: visit damaged buildings first."""
    share = float(np.mean(is_damaged))
    return [min(1.0, b / share) if share > 0 else 0.0 for b in budgets]


# ---- plots -------------------------------------------------------------------------

# One colour per method, fixed across every chart so a method is recognisable anywhere.
METHOD_COLORS = ["#898781", "#2a78d6", "#eda100", "#e34948", "#4a3aa7", "#1f9d6b"]


def plot_risk_coverage(ax, methods, y_true, title):
    """All methods' risk-coverage curves on one axis. ``methods`` = src.uncertainty output."""
    for (name, m), color in zip(methods.items(), METHOD_COLORS):
        cov, risk = risk_coverage_curve(m["confidence"], m["pred"] == y_true)
        # the first few points average over a handful of buildings and jump around
        skip = max(1, len(cov) // 200)
        ax.plot(cov[skip:], risk[skip:], label=f"{name} (AURC {risk.mean():.3f})",
                color=color, linewidth=1.8)
    ax.set_xlabel("coverage (share of buildings auto-accepted)")
    ax.set_ylabel("risk (error rate among accepted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=10)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")


def plot_inspection(ax, curves, is_damaged, title, budgets=BUDGETS):
    """Damaged buildings reached vs inspection budget.

    ``curves`` = {label: priority} or {label: (priority, colour, line style)}; the second
    form lets the caller use colour for the method and line style for the policy.
    """
    pct = [100 * b for b in budgets]
    ax.plot(pct, [100 * r for r in oracle_recall(is_damaged, budgets)], color="#0b0b0b",
            linestyle=":", linewidth=1.2, label="perfect ranking")
    for (name, value), default in zip(curves.items(), METHOD_COLORS[1:] * 3):
        priority, color, style = value if isinstance(value, tuple) else (value, default, "-")
        ax.plot(pct, [100 * r for r in inspection_recall(priority, is_damaged, budgets)],
                marker="o", markersize=3, linewidth=1.6, color=color, linestyle=style, label=name)
    ax.plot(pct, pct, color="#898781", linestyle="--", linewidth=1.2, label="random visits")
    ax.set_xlabel("inspection budget K (% of buildings visited)")
    ax.set_ylabel("% of truly damaged buildings reached")
    ax.set_ylim(0, 102)
    ax.set_title(title, fontsize=10)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
