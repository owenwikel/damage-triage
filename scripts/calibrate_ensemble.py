"""Calibration of the 3-seed ensemble -> docs/calibration_results.md, outputs/calib/

Usage (from the repo root; needs scripts/export_predictions.py to have been run):
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_ensemble.py

What it answers
---------------
The ensemble says "I am 80% sure". Is it right 80% of the time? On the disasters it
trained on (val-tiles)? On disasters it never saw (val)? And if not, does temperature
scaling (src/calibrate.py) repair it?

    1. main result    fit ONE temperature on val-tiles, apply it to val-tiles and val
    2. alternative    fit the temperature on val instead. This is NOT something we could
                      do for a real new disaster (it needs that disaster's true labels);
                      it shows the best any single temperature could do on val.
    3. breakdowns     ECE per val disaster, and accuracy per true class

What goes in
------------
outputs/predictions/<run>/<part>.csv for parts val-tiles and val only. This script cannot
touch the test part: "test" is not an allowed value for --parts, no test predictions
were ever exported, and no model or image is loaded here at all. It runs on the CPU in a
few seconds.

How the ensemble is formed
--------------------------
For each building, each seed's four logits are turned into probabilities (softmax), and
the three probability vectors are averaged ("mean softmax"). Temperature scaling needs
logits, and an average of probabilities has none of its own. But softmax(log(p)) = p for
any probabilities p, so log(p) can serve as the ensemble's logits: with T = 1 nothing
changes, and dividing by T flattens or sharpens the averaged probabilities exactly as
it would for a single model. src/uncertainty.py does the same.

What is written
---------------
    outputs/calib/reliability_fit_on_val-tiles.png      main result
    outputs/calib/reliability_fit_on_val.png            the alternative
    outputs/calib/reliability_val_per_disaster.png
    outputs/calib/calibration.json                      every number, unrounded
    docs/figures/calibration_*.png                      the same three figures, for the doc
    docs/calibration_results.md                         the "tables" section only; the
                                                        text around it is written by hand
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # draw to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import (calibrated_probs, calibration_report, fit_temperature,  # noqa: E402
                           plot_reliability)
from src.colors import INK, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES  # noqa: E402
from src.metrics import expected_calibration_error  # noqa: E402
from src.reporting import df_to_markdown, write_section  # noqa: E402
from src.split import require_final  # noqa: E402
from src.uncertainty import ensemble_probs  # noqa: E402

PARTS = ("val-tiles", "val")        # the only parts this script will read
DEFAULT_RUNS = ["stack_aug_s0", "stack_aug_s1", "stack_aug_s2"]
LOGIT_COLUMNS = [f"logit_{i}" for i in range(len(CLASS_NAMES))]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_ensemble(pred_dir, runs, part):
    """One table per building for ``part``, with the ensemble's "logits".

    Returns a DataFrame with uid, disaster, tile_id, label and ens_0..ens_3, where
    ens_k = log(mean over seeds of softmax(seed's logits))[k].
    """
    require_final(part)                 # raises for "test"; final=True is never passed
    tables = [pd.read_csv(Path(pred_dir) / run / f"{part}.csv", dtype={"tile_id": str})
              for run in runs]
    first = tables[0]
    for run, t in zip(runs, tables):
        # Averaging row by row is only right if row i is the same building in every
        # file. The export writes them in the same order; check instead of trusting.
        assert (t["uid"] == first["uid"]).all(), f"{run}/{part}: buildings out of step"
        assert (t["label"] == first["label"]).all(), f"{run}/{part}: labels differ"
    probs = ensemble_probs([t[LOGIT_COLUMNS].to_numpy() for t in tables])     # [N, 4]
    out = first[["uid", "disaster", "tile_id", "label"]].copy()
    # clip: log(0) is minus infinity. A probability that small never occurs here, but
    # one such value would poison every average.
    out[[f"ens_{i}" for i in range(len(CLASS_NAMES))]] = np.log(np.clip(probs, 1e-12, None))
    return out


def logits_of(table):
    return table[[f"ens_{i}" for i in range(len(CLASS_NAMES))]].to_numpy()


def confidence_and_correct(table, temperature):
    """Top probability and "was the prediction right" for every building, at one T."""
    p = calibrated_probs(logits_of(table), temperature)
    return p.max(axis=1), p.argmax(axis=1) == table["label"].to_numpy()


def before_after_table(data, temperature):
    """ECE / NLL / mean confidence before and after scaling with ``temperature``, per part."""
    rows = []
    for part, table in data.items():
        report = calibration_report(logits_of(table), table["label"].to_numpy(), temperature)
        for when in ("before", "after"):
            r = report[when]
            rows.append({"part": part, "temperature scaling": when, "ECE": r["ece"],
                         "NLL": r["nll"], "mean confidence": r["mean_confidence"],
                         "accuracy": r["accuracy"]})
    return pd.DataFrame(rows)


def per_disaster_table(table, temperatures):
    """One row per disaster (plus "all"): ECE at each temperature in ``temperatures``.

    ``temperatures`` is {column label: T}. ECE is recomputed inside each disaster: the
    bins are filled with that disaster's buildings only.
    """
    groups = [("all", np.ones(len(table), dtype=bool))]
    groups += [(d, (table["disaster"] == d).to_numpy()) for d in sorted(table["disaster"].unique())]
    rows = []
    for name, mask in groups:
        sub = table[mask]
        row = {"disaster": name, "buildings": int(mask.sum())}
        for label, t in temperatures.items():
            conf, correct = confidence_and_correct(sub, t)
            row["accuracy"] = float(correct.mean())       # the same at every temperature
            row[f"mean confidence, {label}"] = float(conf.mean())
            row[f"ECE, {label}"] = expected_calibration_error(conf, correct)
        rows.append(row)
    cols = (["disaster", "buildings", "accuracy"] + [f"mean confidence, {l}" for l in temperatures]
            + [f"ECE, {l}" for l in temperatures])
    return pd.DataFrame(rows)[cols]


def per_class_table(groups):
    """Accuracy per TRUE class: of the buildings that really are class c, the share the
    ensemble called c (also known as the recall of class c).

    ``groups`` is {name: table}. Temperature plays no part here: it never changes which
    class is predicted. Also given: how many buildings of the class there are (a share
    out of 187 buildings is much shakier than one out of 16,000) and what the ensemble
    most often says instead.
    """
    rows = []
    for name, table in groups.items():
        y = table["label"].to_numpy()
        pred = logits_of(table).argmax(axis=1)
        for c, class_name in enumerate(CLASS_NAMES):
            is_c = y == c
            wrong = pred[is_c & (pred != c)]
            row = {"buildings": name, "true class": class_name, "n": int(is_c.sum()),
                   "accuracy": float((pred[is_c] == c).mean()) if is_c.any() else float("nan"),
                   "most common wrong answer": ""}
            if len(wrong):
                top = np.bincount(wrong, minlength=len(CLASS_NAMES)).argmax()
                row["most common wrong answer"] = (
                    f"{CLASS_NAMES[top]} ({(wrong == top).sum() / is_c.sum():.0%} of the class)")
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_figure(panels, path, suptitle):
    """A grid of reliability diagrams. ``panels`` is a list of rows, each a list of
    (title, confidence, correct). plot_reliability (src/calibrate.py) draws one panel."""
    n_rows, n_cols = len(panels), len(panels[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.9 * n_rows), squeeze=False)
    for r, row in enumerate(panels):
        for c, (title, conf, correct) in enumerate(row):
            plot_reliability(axes[r][c], conf, correct, title)
    axes[0][0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    for p in path if isinstance(path, (list, tuple)) else [path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="folders under --pred-dir; their mean softmax is the ensemble")
    ap.add_argument("--out", type=Path, default=Path("outputs/calib"))
    ap.add_argument("--doc", type=Path, default=Path("docs/calibration_results.md"))
    ap.add_argument("--doc-fig-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    data = {part: load_ensemble(args.pred_dir, args.runs, part) for part in PARTS}
    # .copy(): pandas hands out a read-only view, and PyTorch (inside fit_temperature)
    # warns about arrays it is not allowed to write to.
    labels = {part: t["label"].to_numpy().copy() for part, t in data.items()}

    # ---- the two temperatures --------------------------------------------------------
    # fit_temperature finds the T that makes the true classes most probable on average
    # (lowest NLL) on the data it is given. The only difference between the main result
    # and the alternative is WHICH data that is.
    t_main = fit_temperature(logits_of(data["val-tiles"]), labels["val-tiles"])
    t_alt = fit_temperature(logits_of(data["val"]), labels["val"])
    print(f"temperature fitted on val-tiles {t_main:.3f}, fitted on val {t_alt:.3f}")

    main_table = before_after_table(data, t_main)
    alt_table = before_after_table(data, t_alt)
    temperatures = {"before": 1.0, "T from val-tiles": t_main, "T from val (alternative)": t_alt}
    disaster_table = per_disaster_table(data["val"], temperatures)
    groups = {"val-tiles (all)": data["val-tiles"], "val (all)": data["val"]}
    for d in sorted(data["val"]["disaster"].unique()):
        groups[f"val: {d}"] = data["val"][data["val"]["disaster"] == d]
    class_table = per_class_table(groups)

    # ---- figures ---------------------------------------------------------------------
    def both(name):                      # every figure is saved twice, see the docstring
        return [args.out / f"reliability_{name}.png",
                args.doc_fig_dir / f"calibration_{name}.png"]

    for name, t, what in (("fit_on_val-tiles", t_main, "temperature fitted on val-tiles"),
                          ("fit_on_val", t_alt, "ALTERNATIVE: temperature fitted on val")):
        panels = [[(f"{part}, before", *confidence_and_correct(data[part], 1.0)),
                   (f"{part}, after (T = {t:.2f})", *confidence_and_correct(data[part], t))]
                  for part in PARTS]
        reliability_figure(panels, both(name), f"Ensemble of {len(args.runs)}: {what}")
    disasters = sorted(data["val"]["disaster"].unique())
    panels = [[(f"{d}, {label}", *confidence_and_correct(data["val"][data["val"]["disaster"] == d], t))
               for label, t in temperatures.items()] for d in disasters]
    reliability_figure(panels, both("val_per_disaster"), "Ensemble on val, per disaster")

    # ---- numbers and document ----------------------------------------------------------
    (args.out / "calibration.json").write_text(json.dumps({
        "runs": args.runs, "n_bins": 15,
        "temperature_fitted_on_val-tiles": t_main, "temperature_fitted_on_val": t_alt,
        "fit_on_val-tiles": main_table.to_dict("records"),
        "fit_on_val_alternative": alt_table.to_dict("records"),
        "val_per_disaster": disaster_table.to_dict("records"),
        "per_class": class_table.to_dict("records")}, indent=2), encoding="utf-8")

    fmt = "{:.4f}"
    md = "\n\n".join([
        "## Tables",
        f"Generated by `scripts/calibrate_ensemble.py` from `outputs/predictions/` "
        f"(runs: {', '.join(f'`{r}`' for r in args.runs)}; mean softmax over the seeds). "
        f"val-tiles: {len(data['val-tiles']):,} buildings; val: {len(data['val']):,}. "
        "ECE uses 15 equal-width confidence bins. NLL = mean of -log(probability given to "
        "the true class). Temperature scaling never changes the predicted class, so "
        "accuracy is the same before and after.",
        f"### 1. Temperature fitted on val-tiles (T = {t_main:.3f}), applied to both parts",
        df_to_markdown(main_table, fmt, index=False),
        "![reliability, temperature fitted on val-tiles](figures/calibration_fit_on_val-tiles.png)",
        f"### 2. Alternative: temperature fitted on val (T = {t_alt:.3f})",
        "**Not the main result.** Here the temperature is fitted on the same val buildings "
        "it is then scored on, which needs val's true labels. The val rows are therefore "
        "the best any single temperature can do on val, not what would happen on a new "
        "disaster.",
        df_to_markdown(alt_table, fmt, index=False),
        "![reliability, temperature fitted on val](figures/calibration_fit_on_val.png)",
        "### 3a. ECE per disaster on val",
        df_to_markdown(disaster_table, fmt, index=False),
        "![reliability per val disaster](figures/calibration_val_per_disaster.png)",
        "### 3b. Accuracy per true class",
        "Of the buildings that truly belong to a class, the share the ensemble assigned "
        "to it (the class's recall). The same at every temperature.",
        df_to_markdown(class_table, "{:.3f}", index=False),
    ])
    write_section(args.doc, "tables", md)
    print(main_table.round(4).to_string(index=False))
    print(alt_table.round(4).to_string(index=False))
    print(disaster_table.round(4).to_string(index=False))
    print(class_table.round(3).to_string(index=False))
    print(f"Wrote {args.doc}, figures and calibration.json in {args.out}")


if __name__ == "__main__":
    main()
