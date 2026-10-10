"""Apply the KEEP / DROP rule of docs/pre_image_plan.md to a candidate setting.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\stop_rule.py --candidate stack_aug_samepair

This file was written and committed BEFORE any candidate was trained, so that the rule
cannot drift after the results are known. It reads only runs/<setting>_s<seed>/
metrics.json (val-tiles and val numbers written by src/train.py). Nothing here can read
the test part.

The rule (baseline = stack_aug, candidate = the new setting; both with seeds 0, 1, 2,
each run kept under the 3-epoch selection rule of src/train.py):

    KEEP if ALL three hold, otherwise DROP:
      1. mean val macro-F1 (candidate)       >= mean val macro-F1 (baseline) + 0.03
      2. mean val accuracy (candidate)       >= mean val accuracy (baseline)
      3. mean val-tiles macro-F1 (candidate) >= mean val-tiles macro-F1 (baseline) - 0.02

"mean" is the plain average over the three seeds of each run's metrics at its kept
epoch. The comparisons use the unrounded numbers. A candidate with fewer than three
finished seeds is DROPPED: an unfinished setting has not met the rule.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
MIN_VAL_F1_GAIN = 0.03          # condition 1
MAX_VAL_TILES_F1_LOSS = 0.02    # condition 3


def setting_means(runs_dir, setting, seeds=SEEDS):
    """Mean over seeds of (val macro-F1, val accuracy, val-tiles macro-F1), or None if a
    seed has not finished. Also checks that every run used the same selection rule."""
    rows = []
    for seed in seeds:
        path = Path(runs_dir) / f"{setting}_s{seed}" / "metrics.json"
        if not path.exists():
            return None
        m = json.loads(path.read_text(encoding="utf-8"))
        # The rule compares like with like only if both were selected the same way.
        assert m["selected_on"] == "mean" and m["select_window"] == 3, f"{path}: other selection rule"
        best = m["best"]
        rows.append((best["val"]["macro_f1"], best["val"]["accuracy"], best["val-tiles"]["macro_f1"]))
    val_f1, val_acc, tiles_f1 = np.mean(rows, axis=0)
    return {"val_macro_f1": float(val_f1), "val_accuracy": float(val_acc),
            "val_tiles_macro_f1": float(tiles_f1)}


def decide(baseline, candidate):
    """Returns ("KEEP" or "DROP", list of (condition text, passed))."""
    if candidate is None:
        return "DROP", [("three finished seeds", False)]
    checks = [
        (f"val macro-F1 {candidate['val_macro_f1']:.4f} >= "
         f"{baseline['val_macro_f1']:.4f} + {MIN_VAL_F1_GAIN} = "
         f"{baseline['val_macro_f1'] + MIN_VAL_F1_GAIN:.4f}",
         candidate["val_macro_f1"] >= baseline["val_macro_f1"] + MIN_VAL_F1_GAIN),
        (f"val accuracy {candidate['val_accuracy']:.4f} >= {baseline['val_accuracy']:.4f}",
         candidate["val_accuracy"] >= baseline["val_accuracy"]),
        (f"val-tiles macro-F1 {candidate['val_tiles_macro_f1']:.4f} >= "
         f"{baseline['val_tiles_macro_f1']:.4f} - {MAX_VAL_TILES_F1_LOSS} = "
         f"{baseline['val_tiles_macro_f1'] - MAX_VAL_TILES_F1_LOSS:.4f}",
         candidate["val_tiles_macro_f1"] >= baseline["val_tiles_macro_f1"] - MAX_VAL_TILES_F1_LOSS),
    ]
    return ("KEEP" if all(ok for _, ok in checks) else "DROP"), checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", help="setting name, e.g. stack_aug_samepair; omit to "
                                        "print the baseline only")
    ap.add_argument("--baseline", default="stack_aug")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args(argv)
    baseline = setting_means(args.runs_dir, args.baseline)
    assert baseline is not None, "the baseline needs three finished seeds"
    print(f"baseline {args.baseline}: " + ", ".join(f"{k} {v:.6f}" for k, v in baseline.items()))
    if not args.candidate:
        return None
    candidate = setting_means(args.runs_dir, args.candidate)
    if candidate:
        print(f"candidate {args.candidate}: " + ", ".join(f"{k} {v:.6f}" for k, v in candidate.items()))
    decision, checks = decide(baseline, candidate)
    for text, ok in checks:
        print(f"  [{'pass' if ok else 'FAIL'}] {text}")
    print(f"decision: {decision}")
    return decision


if __name__ == "__main__":
    main()
