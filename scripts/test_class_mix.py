"""EXPLORATORY, after the final evaluation: how much of the test/val difference in the
missed-damage rate is the class mix? -> section "exploratory" of docs/test_results.md

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\test_class_mix.py --final

This is NOT part of the protocol's fixed metrics (docs/test_protocol.md, section 3). It
was written after the test results were seen, to explain them, and is labelled as
exploratory in the document, as rule 5 of the protocol requires. It changes no model,
no queue and no headline number; it only cuts the same predictions another way.

The idea
--------
The missed-damage rate averages over all damaged buildings. Damaged buildings come in
three classes, and the model misses them at very different rates (it calls a
minor-damage building "no-damage" far more often than a destroyed one). So a part with
many minor-damage buildings has a high miss rate even if the model is exactly as good,
class by class, as elsewhere.

To separate the two, the miss rate is split by true class, and then recombined with
ANOTHER part's class mix ("standardised"): "what would test's miss rate be if its
damaged buildings were minor / major / destroyed in val's proportions?" If that number
is close to val's, the difference was the mix; if it stays far away, the model really
does worse on test's buildings of the same class.

Reads the test predictions behind the same --final and lock check as
scripts/test_results.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import risk_coverage as rc  # noqa: E402
import test_results as tr  # noqa: E402
from calibrate_ensemble import DEFAULT_RUNS, load_ensemble, logits_of  # noqa: E402
from src.dataset import CLASS_NAMES  # noqa: E402
from src.metrics import softmax  # noqa: E402
from src.reporting import df_to_markdown, write_section  # noqa: E402
from src.split import TEST_PART, require_final  # noqa: E402

DAMAGED = CLASS_NAMES[1:]           # minor-damage, major-damage, destroyed
RATE = tr.HEADLINE_RATE             # 20% review, as in the headline


def per_class_misses(table, rate=RATE):
    """For each damaged class: number of buildings, share called no-damage, and share
    still missed after the ``rate``% "doubtful no-damage first" review."""
    probs = softmax(logits_of(table))
    y, pred = table["label"].to_numpy(), probs.argmax(axis=1)
    # The same review order as the headline: risk_coverage's priority, highest first.
    order = np.argsort(-rc.review_priority(probs)[tr.HEADLINE], kind="stable")
    reviewed = np.zeros(len(y), dtype=bool)
    reviewed[order[:int(np.floor(rate / 100 * len(y) + 1e-9))]] = True
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        if c == 0:
            continue
        of_class = y == c
        n = int(of_class.sum())
        out[name] = {"n": n,
                     "missed, nobody reviewed": float((pred[of_class] == 0).mean()) if n else np.nan,
                     f"missed after {rate}% review":
                         float(((pred == 0) & ~reviewed)[of_class].mean()) if n else np.nan}
    return out


def standardised(rates, mix):
    """Overall miss rate from per-class miss rates, weighted by a class mix (counts)."""
    total = sum(mix.values())
    return sum(rates[c] * mix[c] / total for c in DAMAGED)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--pred-dir", type=Path, default=tr.REPO / "outputs" / "predictions_test")
    ap.add_argument("--dev-pred-dir", type=Path, default=tr.REPO / "outputs" / "predictions")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--lock", type=Path, default=tr.REPO / "TEST_EVALUATION.lock")
    ap.add_argument("--doc", type=Path, default=tr.REPO / "docs" / "test_results.md")
    args = ap.parse_args(argv)
    require_final(TEST_PART, args.final)
    tr.check_lock(args.lock, args.pred_dir, args.runs)

    groups = {f"{p} (all)": load_ensemble(args.dev_pred_dir, args.runs, p) for p in ("val-tiles", "val")}
    groups.update(tr.split_groups(tr.load_tables(args.pred_dir, args.runs, TEST_PART, final=True), TEST_PART))
    stats = {name: per_class_misses(t) for name, t in groups.items()}

    rows = []
    for name, s in stats.items():
        damaged = sum(s[c]["n"] for c in DAMAGED)
        for c in DAMAGED:
            rows.append({"buildings": name, "true class": c, "n": s[c]["n"],
                         "share of the damaged buildings": s[c]["n"] / damaged,
                         "missed, nobody reviewed": s[c]["missed, nobody reviewed"],
                         f"missed after {RATE}% review": s[c][f"missed after {RATE}% review"]})
    per_class = pd.DataFrame(rows)

    # Standardise: each group's per-class miss rates with each reference group's mix.
    references = ["val-tiles (all)", "val (all)", f"{TEST_PART}: both disasters"]
    std_rows = []
    for key in ("missed, nobody reviewed", f"missed after {RATE}% review"):
        for name, s in stats.items():
            rates = {c: s[c][key] for c in DAMAGED}
            if any(np.isnan(v) for v in rates.values()):
                continue
            row = {"miss rate": key, "per-class miss rates of": name,
                   "with its own class mix": standardised(rates, {c: s[c]["n"] for c in DAMAGED})}
            for ref in references:
                row[f"with the mix of {ref}"] = standardised(rates, {c: stats[ref][c]["n"] for c in DAMAGED})
            std_rows.append(row)
    std = pd.DataFrame(std_rows)

    md = "\n\n".join([
        "## Exploratory: the class mix (not part of the protocol's metrics)",
        "Generated by `scripts/test_class_mix.py`, **written after the test results were "
        "seen**, to explain them (protocol rule 5). No headline number depends on it.",
        "Miss rate by true class. 'Missed' = called no-damage; after review = called "
        f"no-damage and not among the {RATE}% reviewed under 'doubtful no-damage first'.",
        df_to_markdown(per_class, "{:.3f}", index=False),
        "The overall miss rate recombined from the per-class rates of one group and the "
        "class mix of another. 'With its own class mix' reproduces the group's real miss "
        "rate. Reading across a row shows what the mix alone does.",
        df_to_markdown(std, "{:.3f}", index=False)])
    write_section(args.doc, "exploratory", md)
    print(per_class.round(3).to_string(index=False))
    print(std.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
