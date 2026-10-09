"""Risk-coverage analysis of the 3-seed ensemble -> docs/risk_coverage.md, docs/figures/

Usage (from the repo root; needs scripts/export_predictions.py to have been run):
    .\\.venv\\Scripts\\python.exe scripts\\risk_coverage.py

The question
------------
After a disaster there are more buildings than people to look at them. Suppose a person
can review R% of the buildings and the model's answer is accepted for the rest. Which
R% should go to the person, and how good are the answers that are accepted?

    review rate   share of buildings sent to a person (0%, 1%, ..., 50%)
    coverage      the rest: share of buildings whose model answer is auto-accepted
                  (coverage = 1 - review rate; "risk-coverage" is the textbook name)

Two ways to choose who gets reviewed ("ranking scores"):

    least confident first       review the buildings with the LOWEST top probability:
                                the ones where the model is least sure of its answer,
                                whatever that answer is.
    most likely damaged first   review the buildings with the HIGHEST 1 - P(no-damage):
                                the ones the model most suspects of being damaged.

Three numbers on the auto-accepted rest, for every review rate:

    accuracy    share of auto-accepted buildings whose predicted class is right
    macro-F1    plain average of the four classes' F1 on the auto-accepted buildings
    miss rate   of ALL truly damaged buildings in the part (label 1, 2 or 3), the share
                that is auto-accepted AND predicted no-damage. These are the buildings
                nobody will look at and the model called fine: the costly mistake.
                The denominator does not shrink with the review rate, so the miss rate
                can only go down as more is reviewed.

The model
---------
Ensemble = for each building, the mean of the three seeds' softmax probabilities.
Predicted class = the largest of the four mean probabilities; confidence = that largest
value. No temperature scaling: dividing by a temperature changes how large the
probabilities are, but this analysis only uses their ORDER.

Uncertainty of the numbers: bootstrap by tile
---------------------------------------------
Every number here comes from one finite set of buildings; another set of tiles from the
same disasters would give slightly different numbers. The bootstrap estimates by how
much: draw a new dataset of the same size from the one we have, WITH replacement (some
items twice, some not at all), recompute everything, repeat 1000 times, and report the
range that holds the middle 95% of the results.

What is drawn is whole TILES, not single buildings. Buildings on one tile share the
image, the neighbourhood and the damage, so they are not independent pieces of evidence:
a tile of 200 flooded houses is closer to one observation than to 200. Resampling single
buildings would pretend otherwise and give intervals that are too narrow.

What the intervals do NOT cover: a different disaster. val has two disasters; the
interval says how much the val numbers would move with other tiles of those two, not
what to expect on a third.

Inputs and outputs
------------------
Reads outputs/predictions/<run>/{val-tiles,val}.csv. The test part cannot be read here:
"test" is refused by load_ensemble (src/split.py), and no test predictions exist.

    docs/figures/risk_coverage_accuracy.png     accuracy vs review rate
    docs/figures/risk_coverage_miss_rate.png    miss rate vs review rate
    outputs/risk_coverage/curves.csv            every review rate, with intervals
    docs/risk_coverage.md                       the "tables" section; the reading around
                                                it is written by hand
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # draw to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# load_ensemble reads the three seeds' tables, checks that they list the same buildings
# in the same order, and returns log(mean softmax). Importing it also sets the charts'
# colours and fonts to the project's style.
from calibrate_ensemble import DEFAULT_RUNS, PARTS, load_ensemble, logits_of  # noqa: E402
from src.metrics import NUM_CLASSES, softmax  # noqa: E402
from src.reporting import df_to_markdown, write_section  # noqa: E402

REVIEW_RATES = np.arange(0, 51)             # per cent: 0, 1, ..., 50
TABLE_RATES = (0, 10, 20, 30)               # the rows shown in the document
METRICS = {"accuracy": "accuracy", "macro_f1": "macro-F1", "miss_rate": "miss rate"}
# name -> (label, colour). Colours are the project's (src/triage.py, METHOD_COLORS).
SCORES = {"confidence": ("least confident first", "#2a78d6"),
          "p_damaged": ("most likely damaged first", "#e34948")}


def review_priority(probs):
    """For each ranking score, a number per building: HIGHER = reviewed SOONER.

    probs   [N, 4] ensemble probabilities (columns: no-damage, minor, major, destroyed)
    """
    return {
        # Low confidence should be reviewed first, so flip the sign.
        "confidence": -probs.max(axis=1),
        # 1 - P(no-damage) = the probability the model gives to "some damage".
        "p_damaged": 1.0 - probs[:, 0],
    }


class Curves:
    """Computes the three metrics at every review rate, fast enough to repeat 1000 times.

    The slow way would be: for each review rate, pick the accepted buildings, build
    their confusion matrix, compute the metrics. With 51 rates, 2 scores and 1000
    bootstrap rounds that is 100,000 confusion matrices per part.

    The fast way rests on one observation: at every review rate, the reviewed buildings
    are a PREFIX of the same sorted list (the first k buildings by priority). So we
    sort once, keep a running total of the confusion matrix along the list, and read
    off the total at each cut. "Accepted" is then "everything" minus "reviewed".

    The bootstrap fits in by giving each building a WEIGHT: the number of times its
    tile was drawn (0, 1, 2, ...). A building with weight 2 counts as two buildings.
    That is the same as physically copying rows, without the copying (up to one
    building at the cut: a building of weight 2 is reviewed or accepted as a whole).
    """

    def __init__(self, priority, y_true, y_pred):
        # kind="stable": buildings with equal priority keep their original order, so
        # the result is the same on every run.
        self.order = np.argsort(-np.asarray(priority, dtype=np.float64), kind="stable")
        y_true, y_pred = np.asarray(y_true)[self.order], np.asarray(y_pred)[self.order]
        # One-hot of the confusion-matrix cell each building falls in. A 4x4 matrix has
        # 16 cells; (true class t, predicted class p) is cell number t * 4 + p.
        # self.cell[i, c] is 1 if building i (in review order) sits in cell c, else 0.
        self.cell = np.zeros((len(y_true), NUM_CLASSES ** 2))
        self.cell[np.arange(len(y_true)), y_true * NUM_CLASSES + y_pred] = 1.0

    def __call__(self, weights=None, rates=REVIEW_RATES):
        """Metrics at each review rate (per cent). ``weights``: one number per building
        in the ORIGINAL order; None = every building counts once.

        Returns {"accuracy": [len(rates)], "macro_f1": [...], "miss_rate": [...]}.
        """
        w = np.ones(len(self.order)) if weights is None else np.asarray(weights, float)[self.order]
        total_weight = w.sum()
        # running[k] = confusion matrix (flattened to 16 numbers) of the first k
        # buildings in review order. A row of zeros in front stands for "none reviewed".
        running = np.vstack([np.zeros(NUM_CLASSES ** 2), np.cumsum(self.cell * w[:, None], axis=0)])
        # How many buildings from the front of the list make up r% of the total weight?
        # cumsum(w) is the weight of the first 1, 2, 3, ... buildings; searchsorted finds
        # how many of them fit within the budget.
        k = np.searchsorted(np.cumsum(w), np.asarray(rates) / 100 * total_weight, side="right")
        everything = running[-1]
        accepted = (everything - running[k]).reshape(len(rates), NUM_CLASSES, NUM_CLASSES)

        diagonal = np.einsum("rii->ri", accepted)            # correct ones, per class
        n_accepted = accepted.sum(axis=(1, 2))
        accuracy = diagonal.sum(axis=1) / np.maximum(n_accepted, 1e-12)

        # F1 of class c = 2 TP / (2 TP + FP + FN) = 2 * diagonal / (row sum + column sum).
        # A class that neither occurs nor is predicted among the accepted has no F1 and
        # is left out of the average, as in src/metrics.py (macro_f1).
        denom = accepted.sum(axis=2) + accepted.sum(axis=1)
        f1 = np.where(denom > 0, 2 * diagonal / np.maximum(denom, 1e-12), np.nan)
        macro_f1 = np.nanmean(f1, axis=1)

        # Misses: true class 1, 2 or 3 (rows 1..3) predicted no-damage (column 0), among
        # the accepted. Divided by ALL damaged buildings, reviewed or not.
        all_damaged = everything.reshape(NUM_CLASSES, NUM_CLASSES)[1:].sum()
        miss_rate = accepted[:, 1:, 0].sum(axis=1) / max(all_damaged, 1e-12)
        return {"accuracy": accuracy, "macro_f1": macro_f1, "miss_rate": miss_rate}


def tile_bootstrap_weights(tile_ids, n_boot, seed):
    """Yield ``n_boot`` weight vectors, one weight per building.

    Each round draws as many tiles as there are, with replacement; a building's weight
    is how often its tile was drawn. All buildings of a tile therefore get the same
    weight: a tile is in or out (or in twice) as a whole.
    """
    # np.unique with return_inverse numbers the tiles 0..T-1 and tells us, for every
    # building, the number of its tile.
    tiles, tile_of_building = np.unique(np.asarray(tile_ids), return_inverse=True)
    rng = np.random.default_rng(seed)         # seeded: the intervals are reproducible
    for _ in range(n_boot):
        drawn = rng.integers(0, len(tiles), size=len(tiles))
        times_drawn = np.bincount(drawn, minlength=len(tiles))
        yield times_drawn[tile_of_building]


def analyse_part(table, n_boot=1000, seed=0, rates=REVIEW_RATES):
    """All curves, with intervals, for one part. Returns a long DataFrame:

        score, metric, review_rate, value, lo, hi

    plus rows with score "difference": most-likely-damaged minus least-confident, with
    the interval of that difference. The difference is computed inside each bootstrap
    round (the same tiles for both scores), which is what makes it a fair comparison:
    whatever a particular draw of tiles does to one score, it does to the other too.
    """
    probs = softmax(logits_of(table))             # back from log(mean softmax) to probabilities
    y_true = table["label"].to_numpy()
    y_pred = probs.argmax(axis=1)
    # A tile id alone is not unique: every disaster numbers its tiles from 0.
    tile_ids = (table["disaster"] + "_" + table["tile_id"]).to_numpy()
    curves = {name: Curves(p, y_true, y_pred) for name, p in review_priority(probs).items()}

    point = {name: c() for name, c in curves.items()}
    boot = {name: {m: [] for m in METRICS} for name in curves}
    for weights in tile_bootstrap_weights(tile_ids, n_boot, seed):
        for name, c in curves.items():
            for metric, values in c(weights).items():
                boot[name][metric].append(values)

    rows = []

    def add(score, metric, value, samples):
        # The middle 95% of the bootstrap results: cut off 2.5% at each end.
        lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
        rows.extend({"score": score, "metric": metric, "review_rate": int(r), "value": v,
                     "lo": a, "hi": b} for r, v, a, b in zip(rates, value, lo, hi))

    for metric in METRICS:
        for name in curves:
            add(name, metric, point[name][metric], np.array(boot[name][metric]))
        add("difference", metric, point["p_damaged"][metric] - point["confidence"][metric],
            np.array(boot["p_damaged"][metric]) - np.array(boot["confidence"][metric]))
    # Each score against reviewing at random, again inside each bootstrap round. Random
    # review leaves accuracy at its 0% value and scales the misses by the share accepted
    # (see plot_metric), so no simulation is needed: both follow from column 0.
    accepted_share = 1 - np.asarray(rates) / 100
    for name in curves:
        acc, miss = np.array(boot[name]["accuracy"]), np.array(boot[name]["miss_rate"])
        add(f"{name}_minus_random", "accuracy",
            point[name]["accuracy"] - point[name]["accuracy"][0], acc - acc[:, :1])
        add(f"{name}_minus_random", "miss_rate",
            point[name]["miss_rate"] - point[name]["miss_rate"][0] * accepted_share,
            miss - miss[:, :1] * accepted_share)
    info = {"buildings": len(table), "tiles": len(np.unique(tile_ids)),
            "damaged": int((y_true > 0).sum()), "predicted_damaged": int((y_pred > 0).sum())}
    return pd.DataFrame(rows), info


def plot_metric(results, metric, path, title, ylabel):
    """One panel per part; one line per ranking score with its 95% band."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 4.3), squeeze=False)
    for ax, (part, df) in zip(axes[0], results.items()):
        for name, (label, color) in SCORES.items():
            d = df[(df["score"] == name) & (df["metric"] == metric)].sort_values("review_rate")
            ax.fill_between(d["review_rate"], d["lo"], d["hi"], color=color, alpha=0.18, linewidth=0)
            ax.plot(d["review_rate"], d["value"], color=color, linewidth=1.8, label=label)
        # What reviewing buildings at RANDOM would give, as a yardstick. Random review
        # leaves the accepted buildings a fair sample of all of them: accuracy stays
        # where it was, and the misses shrink in step with the share accepted.
        start = df[(df["metric"] == metric) & (df["review_rate"] == 0)]["value"].iloc[0]
        r = np.asarray(REVIEW_RATES)
        random = start * (1 - r / 100) if metric == "miss_rate" else np.full(len(r), start)
        ax.plot(r, random, color="#898781", linestyle="--", linewidth=1.2, label="random review")
        ax.set_title(part, fontsize=10)
        ax.set_xlabel("review rate (% of buildings sent to a person)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 50)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    if metric == "miss_rate":
        for ax in axes[0]:
            ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle(title + "  (bands: 95% bootstrap intervals, resampled by tile)", fontsize=10.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def table_at(df, rates=TABLE_RATES):
    """The document's table for one part: rows = review rate x ranking score."""
    rows = []
    for r in rates:
        for name, (label, _) in SCORES.items():
            row = {"review rate": f"{r}%", "who is reviewed": label if r else "nobody"}
            for metric, title in METRICS.items():
                d = df[(df["score"] == name) & (df["metric"] == metric) & (df["review_rate"] == r)].iloc[0]
                row[title] = f"{d['value']:.3f} ({d['lo']:.3f}-{d['hi']:.3f})"
            rows.append(row)
            if r == 0:                    # with nobody reviewed the two scores coincide
                break
    return pd.DataFrame(rows)


def difference_table(df, score="difference", metrics=tuple(METRICS), rates=TABLE_RATES[1:]):
    """A difference with its interval, per review rate. ``score`` picks which one:
    "difference" (most-likely-damaged minus least-confident) or "<score>_minus_random"."""
    rows = []
    for r in rates:
        row = {"review rate": f"{r}%"}
        for metric in metrics:
            title = METRICS[metric]
            d = df[(df["score"] == score) & (df["metric"] == metric) & (df["review_rate"] == r)].iloc[0]
            row[title] = f"{d['value']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/risk_coverage"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--doc", type=Path, default=Path("docs/risk_coverage.md"))
    args = ap.parse_args(argv)

    results, infos = {}, {}
    for part in PARTS:                    # ("val-tiles", "val"): never "test"
        table = load_ensemble(args.pred_dir, args.runs, part)
        results[part], infos[part] = analyse_part(table, args.n_boot, args.seed)
        print(f"{part}: {infos[part]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat([df.assign(part=part) for part, df in results.items()]).to_csv(
        args.out / "curves.csv", index=False, float_format="%.6f", lineterminator="\n")
    plot_metric(results, "accuracy", args.fig_dir / "risk_coverage_accuracy.png",
                "Accuracy on the auto-accepted buildings", "accuracy of auto-accepted answers")
    plot_metric(results, "miss_rate", args.fig_dir / "risk_coverage_miss_rate.png",
                "Damaged buildings missed", "miss rate (share of all damaged buildings)")

    md = ["## Tables",
          f"Generated by `scripts/risk_coverage.py` from `outputs/predictions/` (runs: "
          f"{', '.join(f'`{r}`' for r in args.runs)}; mean softmax over the seeds, no "
          f"temperature). Each cell: value (95% interval). Intervals: {args.n_boot} "
          "bootstrap resamples of whole tiles, percentile method. All three numbers are "
          "computed on the auto-accepted buildings; the miss rate is their undetected "
          "damage as a share of all truly damaged buildings in the part."]
    for part in PARTS:
        i = infos[part]
        md += [f"### {part}",
               f"{i['buildings']:,} buildings on {i['tiles']:,} tiles; {i['damaged']:,} truly "
               f"damaged ({i['damaged'] / i['buildings']:.1%}); {i['predicted_damaged']:,} "
               f"predicted damaged ({i['predicted_damaged'] / i['buildings']:.1%}).",
               df_to_markdown(table_at(results[part]), index=False),
               "Most likely damaged first minus least confident first (same tiles in every "
               "resample; an interval that excludes 0 means the difference is not just "
               "tile-to-tile noise):",
               df_to_markdown(difference_table(results[part]), index=False)]
        print(part)
        print(table_at(results[part]).to_string(index=False))
        print(difference_table(results[part]).to_string(index=False))
        for name, (label, _) in SCORES.items():
            vs_random = difference_table(results[part], f"{name}_minus_random",
                                         ("accuracy", "miss_rate"))
            md += [f"{label[0].upper() + label[1:]} minus random review (negative miss rate "
                   "= fewer misses than random):",
                   df_to_markdown(vs_random, index=False)]
            print(vs_random.to_string(index=False))
    md += ["### Plots",
           "![accuracy vs review rate](figures/risk_coverage_accuracy.png)",
           "![miss rate vs review rate](figures/risk_coverage_miss_rate.png)",
           "The dashed line is what reviewing buildings at random would give. Every review "
           "rate from 0 to 50% is in `outputs/risk_coverage/curves.csv` (not in git)."]
    write_section(args.doc, "tables", "\n\n".join(md))
    print(f"Wrote {args.doc}, {args.out / 'curves.csv'} and two figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
