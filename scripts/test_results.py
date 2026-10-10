"""THE FINAL EVALUATION, STEP 2: risk-coverage on the test predictions -> docs/test_results.md

Usage (after scripts/export_test_predictions.py has run):
    .\\.venv\\Scripts\\python.exe scripts\\test_results.py --final

A rehearsal on the val predictions, where the answers are already known (writes to a
scratch folder, touches nothing of the test set and needs no --final):
    .\\.venv\\Scripts\\python.exe scripts\\test_results.py --part val --pred-dir outputs\\predictions --doc outputs\\test_rehearsal\\rehearsal.md --fig-dir outputs\\test_rehearsal --out outputs\\test_rehearsal

What it does
------------
docs/test_protocol.md fixes what is measured. This script only carries it out:

  * the ensemble is the mean of the three seeds' softmax probabilities, no temperature;
  * every risk-coverage number comes from the UNCHANGED functions of
    scripts/risk_coverage.py (``analyse_part`` and what it calls): the three rankings,
    the review rates 0-50%, the miss rate, accuracy and macro-F1 on the accepted
    buildings, both random baselines, and the 1000-resample tile bootstrap;
  * two things the protocol asks for that risk_coverage.py does not compute are added
    here with the same bootstrap resamples: the share of predicted damage that is real,
    and per-class recall;
  * each disaster of the part is scored separately, then both together;
  * the same computation is repeated on val-tiles and val for the comparison table.

scripts/risk_coverage.py itself refuses the test part (its loader calls
``require_final`` without ``final``). So the tables are loaded here, behind this
script's own ``--final`` and behind the lock file written by the export: for the test
part the lock must say "completed" and the files must still have the checksums recorded
in it.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import risk_coverage as rc  # noqa: E402  (unchanged; see the docstring)
from calibrate_ensemble import DEFAULT_RUNS, LOGIT_COLUMNS, load_ensemble, logits_of  # noqa: E402
from src.dataset import CLASS_NAMES  # noqa: E402
from src.metrics import NUM_CLASSES, softmax  # noqa: E402
from src.reporting import df_to_markdown, write_section  # noqa: E402
from src.split import TEST_PART, require_final  # noqa: E402
from src.uncertainty import ensemble_probs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HEADLINE_RATE = 20                       # per cent; fixed by the protocol
HEADLINE = "no_damage_queue"             # "doubtful no-damage first"
TABLE_RATES = (0, 10, 20, 30)


def check_lock(lock_path, pred_dir, runs):
    """The test tables may be read only if the export finished and nothing was altered."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise SystemExit(f"{lock_path} not found: run scripts/export_test_predictions.py first.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "completed":
        raise SystemExit(f"{lock_path} says '{lock.get('status')}': the export did not finish. "
                         "See docs/test_protocol.md, section 7.")
    for run in runs:
        path = Path(pred_dir) / run / f"{TEST_PART}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != lock["files"][run]["sha256"]:
            raise SystemExit(f"{path} differs from the file recorded in the lock.")
    return lock


def load_tables(pred_dir, runs, part, final=False):
    """The ensemble table of one part: uid, disaster, tile_id, label, ens_0..ens_3.

    The same steps as calibrate_ensemble.load_ensemble (which the development scripts
    use and which cannot read the test part), with ``final`` passed on to the guard.
    """
    require_final(part, final)             # raises for "test" unless final=True
    tables = [pd.read_csv(Path(pred_dir) / run / f"{part}.csv", dtype={"tile_id": str})
              for run in runs]
    first = tables[0]
    for run, t in zip(runs, tables):
        assert (t["uid"] == first["uid"]).all(), f"{run}/{part}: buildings out of step"
        assert (t["label"] == first["label"]).all(), f"{run}/{part}: labels differ"
    probs = ensemble_probs([t[LOGIT_COLUMNS].to_numpy() for t in tables])
    out = first[["uid", "disaster", "tile_id", "label"]].copy()
    out[[f"ens_{i}" for i in range(NUM_CLASSES)]] = np.log(np.clip(probs, 1e-12, None))
    return out


def split_groups(table, part):
    """{name: table}: each disaster of the part, then all of them together."""
    groups = {f"{part}: {d}": g.reset_index(drop=True) for d, g in table.groupby("disaster")}
    if len(groups) > 1:
        groups[f"{part}: both disasters"] = table.reset_index(drop=True)
    return groups


def extra_metrics(table, n_boot=1000, seed=0):
    """Share of predicted damage that is real, per-class recall and the class mix, each
    with a 95% interval from the same tile resamples as risk_coverage.analyse_part.

    Returns {name: (value, lo, hi)}.
    """
    y = table["label"].to_numpy()
    pred = softmax(logits_of(table)).argmax(axis=1)
    tile_ids = (table["disaster"] + "_" + table["tile_id"]).to_numpy()
    # One-hot of each building's confusion-matrix cell, as in risk_coverage.Curves: a
    # weighted sum of these rows IS the confusion matrix of a bootstrap resample.
    cell = np.zeros((len(y), NUM_CLASSES ** 2))
    cell[np.arange(len(y)), y * NUM_CLASSES + pred] = 1.0

    def measure(weights):
        cm = (cell * weights[:, None]).sum(axis=0).reshape(NUM_CLASSES, NUM_CLASSES)
        true_n, total = cm.sum(axis=1), cm.sum()
        predicted_damaged = cm[:, 1:].sum()
        out = {
            # rows 1..3 = truly damaged, columns 1..3 = predicted damaged
            "share of predicted damage that is real":
                cm[1:, 1:].sum() / predicted_damaged if predicted_damaged else np.nan,
            "share of buildings predicted damaged": predicted_damaged / total,
        }
        for c, name in enumerate(CLASS_NAMES):
            out[f"recall {name}"] = cm[c, c] / true_n[c] if true_n[c] else np.nan
            out[f"share of buildings truly {name}"] = true_n[c] / total
        return out

    point = measure(np.ones(len(y)))
    boot = [measure(w.astype(float)) for w in rc.tile_bootstrap_weights(tile_ids, n_boot, seed)]
    result = {}
    for name, value in point.items():
        lo, hi = np.nanpercentile([b[name] for b in boot], [2.5, 97.5])
        result[name] = (value, lo, hi)
    result["_counts"] = {name: int((y == c).sum()) for c, name in enumerate(CLASS_NAMES)}
    return result


def analyse(groups, n_boot, seed):
    """{group: (curves DataFrame, info dict, extra metrics)} using the unchanged code."""
    out = {}
    for name, table in groups.items():
        curves, info = rc.analyse_part(table, n_boot, seed)
        out[name] = (curves, info, extra_metrics(table, n_boot, seed))
        print(f"analysed {name}: {info}", flush=True)
    return out


def cell(curves, score, metric, rate, signed=False):
    """'value (lo-hi)' for one row of an analyse_part table."""
    d = curves[(curves["score"] == score) & (curves["metric"] == metric)
               & (curves["review_rate"] == rate)].iloc[0]
    if signed:
        return f"{d['value']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})"
    return f"{d['value']:.3f} ({d['lo']:.3f}-{d['hi']:.3f})"


def value(curves, score, metric, rate):
    return float(curves[(curves["score"] == score) & (curves["metric"] == metric)
                        & (curves["review_rate"] == rate)]["value"].iloc[0])


def headline_table(results, rate=HEADLINE_RATE):
    """Missed-damage rate at the headline review rate, with both baselines."""
    rows = []
    for name, (curves, info, _) in results.items():
        nobody = value(curves, HEADLINE, "miss_rate", 0)
        rows.append({
            "buildings": name,
            "nobody reviewed": cell(curves, HEADLINE, "miss_rate", 0),
            f"{rate}% random review": f"{nobody * (1 - rate / 100):.3f}",
            f"{rate}%: the queue in random order": cell(curves, "queue_random", "miss_rate", rate),
            f"{rate}%: doubtful no-damage first": cell(curves, HEADLINE, "miss_rate", rate),
            "queue minus random review": cell(curves, f"{HEADLINE}_minus_random", "miss_rate", rate, True),
            "queue minus the queue in random order":
                cell(curves, f"{HEADLINE}_minus_queue_random", "miss_rate", rate, True),
        })
    return pd.DataFrame(rows)


def accepted_table(results):
    """Miss rate, accuracy and macro-F1 on the accepted buildings at 0/10/20/30% review."""
    rows = []
    for name, (curves, _, _) in results.items():
        for rate in TABLE_RATES:
            rows.append({"buildings": name, "review rate": f"{rate}%",
                         "missed-damage rate": cell(curves, HEADLINE, "miss_rate", rate),
                         "accuracy on accepted": cell(curves, HEADLINE, "accuracy", rate),
                         "macro-F1 on accepted": cell(curves, HEADLINE, "macro_f1", rate)})
    return pd.DataFrame(rows)


def data_table(results):
    rows = []
    for name, (_, info, extra) in results.items():
        row = {"buildings": name, "n": info["buildings"], "tiles": info["tiles"]}
        for c in CLASS_NAMES:
            row[c] = f"{extra['_counts'][c]:,} ({extra[f'share of buildings truly {c}'][0]:.1%})"
        row["predicted damaged"] = f"{extra['share of buildings predicted damaged'][0]:.1%}"
        rows.append(row)
    return pd.DataFrame(rows)


def recall_table(results):
    rows = []
    for name, (_, _, extra) in results.items():
        row = {"buildings": name}
        for key in ["share of predicted damage that is real"] + [f"recall {c}" for c in CLASS_NAMES]:
            v, lo, hi = extra[key]
            row[key] = "" if np.isnan(v) else f"{v:.3f} ({lo:.3f}-{hi:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(all_results, rate=HEADLINE_RATE):
    """One column per group (validation parts first), one row per number: the plain
    values, without intervals, to see everything side by side."""
    rows = {}

    def put(label, name, text):
        rows.setdefault(label, {})[name] = text

    for name, (curves, info, extra) in all_results.items():
        nobody = value(curves, HEADLINE, "miss_rate", 0)
        put("buildings", name, f"{info['buildings']:,}")
        put("tiles", name, f"{info['tiles']:,}")
        for c in CLASS_NAMES:
            put(f"share truly {c}", name, f"{extra[f'share of buildings truly {c}'][0]:.1%}")
        put("share predicted damaged", name, f"{extra['share of buildings predicted damaged'][0]:.1%}")
        put("accuracy, nobody reviewed", name, f"{value(curves, HEADLINE, 'accuracy', 0):.3f}")
        put("macro-F1, nobody reviewed", name, f"{value(curves, HEADLINE, 'macro_f1', 0):.3f}")
        for c in CLASS_NAMES:
            v = extra[f"recall {c}"][0]
            put(f"recall {c}", name, "" if np.isnan(v) else f"{v:.3f}")
        put("share of predicted damage that is real", name,
            f"{extra['share of predicted damage that is real'][0]:.3f}")
        put("missed-damage rate, nobody reviewed", name, f"{nobody:.3f}")
        put(f"missed-damage rate, {rate}% random review", name, f"{nobody * (1 - rate / 100):.3f}")
        put(f"missed-damage rate, {rate}%, the queue in random order", name,
            f"{value(curves, 'queue_random', 'miss_rate', rate):.3f}")
        put(f"missed-damage rate, {rate}%, least confident first", name,
            f"{value(curves, 'confidence', 'miss_rate', rate):.3f}")
        put(f"**missed-damage rate, {rate}%, doubtful no-damage first**", name,
            f"**{value(curves, HEADLINE, 'miss_rate', rate):.3f}**")
        put(f"share of the misses caught by the {rate}% queue", name,
            f"{1 - value(curves, HEADLINE, 'miss_rate', rate) / nobody:.0%}" if nobody else "")
        put(f"accuracy on accepted, {rate}%", name, f"{value(curves, HEADLINE, 'accuracy', rate):.3f}")
        put(f"macro-F1 on accepted, {rate}%", name, f"{value(curves, HEADLINE, 'macro_f1', rate):.3f}")
    table = pd.DataFrame(rows).T
    table.insert(0, "", table.index)
    return table.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default=TEST_PART, choices=[TEST_PART, "val", "val-tiles"],
                    help="'test' for the real thing (needs --final); 'val' for a rehearsal")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--pred-dir", type=Path, default=None,
                    help="default: outputs/predictions_test for test, outputs/predictions otherwise")
    ap.add_argument("--dev-pred-dir", type=Path, default=REPO / "outputs" / "predictions",
                    help="val-tiles and val predictions, for the comparison")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--lock", type=Path, default=REPO / "TEST_EVALUATION.lock")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doc", type=Path, default=REPO / "docs" / "test_results.md")
    ap.add_argument("--fig-dir", type=Path, default=REPO / "docs" / "figures")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "test_results")
    args = ap.parse_args(argv)
    is_test = args.part == TEST_PART
    pred_dir = args.pred_dir or REPO / "outputs" / ("predictions_test" if is_test else "predictions")
    if is_test:
        require_final(args.part, args.final)          # raises without --final
        check_lock(args.lock, pred_dir, args.runs)

    table = load_tables(pred_dir, args.runs, args.part, final=args.final and is_test)
    results = analyse(split_groups(table, args.part), args.n_boot, args.seed)
    # The validation parts through exactly the same functions, for the comparison.
    dev = {f"{p} (all)": load_ensemble(args.dev_pred_dir, args.runs, p) for p in ("val-tiles", "val")}
    all_results = {**analyse(dev, args.n_boot, args.seed), **results}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat([c.assign(group=name) for name, (c, _, _) in results.items()]).to_csv(
        args.out / f"curves_{args.part}.csv", index=False, float_format="%.6f", lineterminator="\n")
    curves_only = {name: c for name, (c, _, _) in results.items()}
    stem = f"{args.part}_risk_coverage"
    rc.plot_metric(curves_only, "miss_rate", args.fig_dir / f"{stem}_miss_rate.png",
                   f"{args.part.upper()}: damaged buildings missed", "miss rate (share of all damaged buildings)")
    rc.plot_metric(curves_only, "accuracy", args.fig_dir / f"{stem}_accuracy.png",
                   f"{args.part.upper()}: accuracy on the auto-accepted buildings",
                   "accuracy of auto-accepted answers")

    # where the figures are, as seen from the document (docs/test_results.md -> figures/)
    fig_rel = Path(os.path.relpath(args.fig_dir, args.doc.parent))
    md = ["## Tables",
          f"Generated by `scripts/test_results.py` from `{pred_dir.name}/` (runs: "
          f"{', '.join(f'`{r}`' for r in args.runs)}; mean softmax over the seeds, no "
          f"temperature). Part: **{args.part}**. Brackets: 95% intervals from {args.n_boot} "
          "bootstrap resamples of whole tiles, within the group of the row. The risk-coverage "
          "numbers are computed by the unchanged functions of `scripts/risk_coverage.py`.",
          "### 1. The data",
          "Number of buildings and tiles, the class mix (count and share), and the share of "
          "buildings the ensemble predicts as damaged.",
          df_to_markdown(data_table(results), index=False),
          f"### 2. Headline: missed-damage rate at a {HEADLINE_RATE}% review rate",
          "Of all truly damaged buildings, the share that nobody reviews and the ensemble "
          "calls no-damage. The two right-hand columns are differences computed inside "
          "each resample; negative = fewer misses than the baseline.",
          df_to_markdown(headline_table(results), index=False),
          f"![{args.part}: miss rate against review rate]({fig_rel.as_posix()}/{stem}_miss_rate.png)",
          "### 3. Doubtful no-damage first: the accepted buildings at each review rate",
          df_to_markdown(accepted_table(results), index=False),
          f"![{args.part}: accuracy against review rate]({fig_rel.as_posix()}/{stem}_accuracy.png)",
          "### 4. Share of predicted damage that is real, and per-class recall",
          "Before any review. Recall of a class = of the buildings truly in it, the share "
          "the ensemble assigns to it.",
          df_to_markdown(recall_table(results), index=False),
          "### 5. The other two rankings, for context",
          "All three rankings at 0/10/20/30% review (accuracy and macro-F1 on the accepted "
          "buildings, miss rate), as in `docs/risk_coverage.md`."]
    for name, (curves, _, _) in results.items():
        md += [f"**{name}**", df_to_markdown(rc.table_at(curves), index=False)]
    md += ["### 6. Side by side with the validation sets",
           "The same computation on val-tiles (held-out tiles of the training disasters) and "
           "val (two unseen flood disasters). Plain values; the intervals are above and in "
           "`docs/risk_coverage.md`.",
           df_to_markdown(comparison_table(all_results), index=False)]
    write_section(args.doc, "tables", "\n\n".join(md))
    print(f"Wrote {args.doc}, figures in {args.fig_dir}, curves in {args.out}")


if __name__ == "__main__":
    main()
