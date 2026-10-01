"""Collect the training runs into docs/results.md and docs/figures/.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\make_results.py

Reads runs/<run>/metrics.json and the saved predictions; trains nothing and needs no GPU,
so it can be re-run at any time. Sections whose runs are missing are skipped.
Only val-tiles and val are ever read: the training runs never save test predictions, and
CropDataset refuses to load the test part (src/split.py).

Sections: baselines, sanity check, architecture comparison (with confusion matrices),
calibration, uncertainty methods (ECE, risk-coverage), triage (coverage table,
inspection budget).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import calibration_report, fit_temperature, plot_reliability  # noqa: E402
from src.colors import DAMAGE_COLORS, INK, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES, read_index  # noqa: E402
from src.metrics import expected_calibration_error, summarize  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402
from src.triage import (BUDGETS, COVERAGES, at_coverage, aurc, inspection_recall,  # noqa: E402
                        plot_inspection, plot_risk_coverage)
from src.uncertainty import method_scores  # noqa: E402

PARTS = ["val-tiles", "val"]
SHORT = ["no-dmg", "minor", "major", "destr."]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_run(run_dir):
    """metrics.json plus predictions of one run, or None if it has not finished."""
    run_dir = Path(run_dir)
    if not (run_dir / "metrics.json").exists() or not (run_dir / "preds_val.npz").exists():
        return None
    run = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run["dir"] = run_dir
    run["preds"] = {p: dict(np.load(run_dir / f"preds_{p}.npz")) for p in PARTS}
    return run


def metric_row(m):
    """accuracy, macro-F1 and the four recalls of a metrics dict, as table cells."""
    row = {"accuracy": m["accuracy"], "macro-F1": m["macro_f1"]}
    for name, r in zip(SHORT, m["recall"]):
        row[f"recall {name}"] = np.nan if r is None else r
    return row


# ---- 1. baselines --------------------------------------------------------------------

def baselines(labels_by_part, train_labels, draws=200, seed=0):
    """Majority class and frequency-weighted random guessing.

    Both use only the class frequencies of the TRAINING set, as a real baseline would.
    Random guessing is averaged over ``draws`` repetitions.
    """
    freq = np.bincount(train_labels, minlength=4) / len(train_labels)
    majority = int(freq.argmax())
    rng = np.random.default_rng(seed)
    rows = []
    for part, y in labels_by_part.items():
        rows.append({"split": part, "baseline": f"majority class ({CLASS_NAMES[majority]})",
                     **metric_row(summarize(y, np.full_like(y, majority)))})
        sims = [metric_row(summarize(y, rng.choice(4, size=len(y), p=freq))) for _ in range(draws)]
        rows.append({"split": part, "baseline": "random, weighted by train class frequency",
                     **pd.DataFrame(sims).mean().to_dict()})
    return pd.DataFrame(rows), freq


# ---- figures -------------------------------------------------------------------------

def plot_confusions(runs, path):
    """Row-normalised confusion matrices: one row of panels per run, one column per split."""
    fig, axes = plt.subplots(len(runs), len(PARTS), figsize=(4.3 * len(PARTS), 3.9 * len(runs)),
                             squeeze=False)
    for i, run in enumerate(runs):
        for j, part in enumerate(PARTS):
            cm = np.array(run["best"][part]["confusion"], dtype=float)
            share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax = axes[i, j]
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for r in range(4):
                for c in range(4):
                    ax.text(c, r, f"{share[r, c]:.0%}\n{int(cm[r, c]):,}", ha="center",
                            va="center", fontsize=7.5,
                            color="white" if share[r, c] > 0.55 else INK)
            ax.set_xticks(range(4), SHORT)
            ax.set_yticks(range(4), SHORT)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{run['run_name']} on {part}\n(each row sums to 100%)", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability_grid(run, temperature, path):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    for j, part in enumerate(PARTS):
        logits, labels = run["preds"][part]["logits"], run["preds"][part]["labels"]
        for i, (name, t) in enumerate((("before", 1.0), (f"after (T = {temperature:.2f})", temperature))):
            m = method_scores(logits, labels, temperature=t)["max prob (temp. scaled)"]
            plot_reliability(axes[i, j], m["confidence"], m["pred"] == labels, f"{part}, {name}")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- the document ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--cache-dir", type=Path, default=Path(r"C:\Users\wikel\dev\xbd-data\crops"))
    ap.add_argument("--archs", nargs="+", default=["siamese", "stack"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("docs/results.md"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = args.fig_dir.relative_to(args.out.parent).as_posix()

    index = read_index(args.cache_dir)
    labels = {p: index[index["part"] == p].sort_values("row")["label"].to_numpy() for p in PARTS}
    train_labels = index[index["part"] == "train"]["label"].to_numpy()

    L = ["# Results (validation only)\n",
         "Generated by `scripts/make_results.py` from `runs/`. **Every number here is on "
         "val-tiles or val. The test set (hurricane-michael, santa-rosa-wildfire) has not "
         "been loaded, evaluated or looked at**; `src/split.py` blocks it unless `--final` "
         "is passed.\n",
         "- **val-tiles**: held-out spatial blocks of the six *training* disasters "
         f"({len(labels['val-tiles']):,} buildings). Used for early stopping and for "
         "fitting the temperature, so scores on it are optimistic.",
         f"- **val**: hurricane-florence and midwest-flooding, never trained on "
         f"({len(labels['val']):,} buildings). The honest estimate for a new disaster, but "
         "floods only, with few minor-damage and destroyed buildings "
         f"({int((labels['val'] == 1).sum())} and {int((labels['val'] == 3).sum())}), so its "
         "per-class numbers for those two classes are noisy.\n",
         "Un-classified buildings are dropped (`docs/data_decisions.md`). Recall of a class "
         "= share of its buildings that were predicted as that class. Macro-F1 = plain "
         "average of the four per-class F1 scores.\n"]

    # 1. baselines
    base, freq = baselines(labels, train_labels)
    L += ["## 1. Baselines\n",
          "What you get without looking at any image. Class shares in train: "
          + ", ".join(f"{n} {f:.1%}" for n, f in zip(CLASS_NAMES, freq)) + ".\n",
          df_to_markdown(base, float_fmt="{:.3f}", index=False) + "\n",
          "Majority-class accuracy is high because most buildings are undamaged; its "
          "macro-F1 shows it is useless. A model has to beat **both** numbers.\n"]

    # 2. sanity check
    sanity = args.runs_dir / "sanity_overfit" / "metrics.json"
    if sanity.exists():
        s = json.loads(sanity.read_text(encoding="utf-8"))
        m = s["best"]["overfit-subset"]
        L += ["## 2. Sanity check: overfit 256 crops\n",
              f"Before any real training, the siamese model was trained on {m['n']} training "
              "crops only (no augmentation) to check that the pipeline can learn at all: "
              f"**{m['accuracy']:.1%} accuracy on those crops after {s['epochs_run']} epochs** "
              f"(training loss {s['history'][0]['train_loss']:.2f} -> "
              f"{s['history'][-1]['train_loss']:.3f}). So labels, gradients and learning "
              "rate are wired up correctly; it says nothing yet about generalisation.\n"]

    # 3. architectures
    runs = {a: load_run(args.runs_dir / f"{a}_s0") for a in args.archs}
    runs = {a: r for a, r in runs.items() if r}
    better = None
    if runs:
        rows, cost = [], []
        for arch, r in runs.items():
            for part in PARTS:
                rows.append({"model": arch, "split": part, **metric_row(r["best"][part])})
            cost.append({"model": arch, "best epoch": r["best_epoch"], "epochs run": r["epochs_run"],
                         "batch size": r["batch_size"],
                         "train s/epoch": r["train_seconds_per_epoch"],
                         "s/epoch incl. eval": r["seconds_per_epoch"],
                         "GPU peak GB (allocated)": r["gpu_peak_allocated_gb"],
                         "GPU peak GB (reserved)": r["gpu_peak_reserved_gb"]})
        better = max(runs, key=lambda a: runs[a]["best"]["val-tiles"]["macro_f1"])
        plot_confusions(list(runs.values()), args.fig_dir / "confusion_matrices.png")
        L += ["## 3. Siamese vs stack\n",
              "Same data, loss, optimiser, schedule and seed (0); only the architecture "
              "differs (`src/model.py`). Weights are those of the best epoch by val-tiles "
              "macro-F1.\n",
              df_to_markdown(pd.DataFrame(rows), float_fmt="{:.3f}", index=False) + "\n",
              df_to_markdown(pd.DataFrame(cost), float_fmt="{:.1f}", index=False) + "\n"]
        if len(runs) > 1:
            a, b = (runs[x]["best"] for x in runs)
            names = list(runs)
            L.append(f"**Better on val-tiles macro-F1 (the selection metric): `{better}`** "
                     f"({runs[better]['best']['val-tiles']['macro_f1']:.3f} vs "
                     f"{min(a['val-tiles']['macro_f1'], b['val-tiles']['macro_f1']):.3f}). "
                     f"On val: {names[0]} {a['val']['macro_f1']:.3f}, {names[1]} "
                     f"{b['val']['macro_f1']:.3f}. This is one seed per architecture, so a "
                     "difference of a point or two is within what a different seed could "
                     "produce (compare the three seeds in section 5). The later sections "
                     f"use `{better}`.\n")
        # Say plainly how the model compares with the no-image baselines on each split.
        for part in PARTS:
            maj = base[(base["split"] == part) & base["baseline"].str.startswith("majority")].iloc[0]
            m = runs[better]["best"][part]
            verdict = ("above" if m["accuracy"] > maj["accuracy"] else "**below**")
            L.append(f"- On **{part}**, `{better}` has accuracy {m['accuracy']:.3f}, {verdict} "
                     f"the majority-class baseline ({maj['accuracy']:.3f}), and macro-F1 "
                     f"{m['macro_f1']:.3f} against the baselines' "
                     f"{base[base['split'] == part]['macro-F1'].max():.3f}.")
        gap = runs[better]["best"]["val-tiles"]["macro_f1"] - runs[better]["best"]["val"]["macro_f1"]
        L.append(f"\nThe gap between val-tiles and val (macro-F1 {gap:+.3f}) is the main "
                 "result so far: the model has learned the training disasters, not damage in "
                 "general. `docs/error_analysis.md` breaks this down per disaster.\n"
                 if gap > 0.15 else "")
        missing = [a for a in args.archs if a not in runs]
        if missing:
            L.append(f"**Not trained yet: {', '.join(f'`{a}`' for a in missing)}** (see "
                     "`BLOCKERS.md`), so there is no architecture comparison in this "
                     f"version of the document. The later sections use `{better}`.\n")
        L += [f"![confusion matrices]({fig_rel}/confusion_matrices.png)\n"]
        for arch, r in runs.items():
            epochs = pd.DataFrame([{
                "epoch": h["epoch"], "train loss": h["train_loss"], "train acc": h["train_acc"],
                **{f"{p} {k}": h[p][v] for p in PARTS
                   for k, v in (("loss", "loss"), ("acc", "accuracy"), ("macro-F1", "macro_f1"))},
                "seconds": int(round(h["epoch_seconds"]))} for h in r["history"]])
            L += [f"Per epoch, `{r['run_name']}` (best epoch by val-tiles macro-F1: "
                  f"**{r['best_epoch']}**; training accuracy is measured on augmented "
                  "batches while the weights are still changing):\n",
                  df_to_markdown(epochs, float_fmt="{:.3f}", index=False) + "\n"]
        L.append("Per-class recall for every epoch is in `runs/<run>/log.jsonl` (not in git).\n")

    # 4-6 need the better architecture's seed-0 run
    if better:
        run = runs[better]
        temperature = fit_temperature(run["preds"]["val-tiles"]["logits"],
                                      run["preds"]["val-tiles"]["labels"])
        cal = []
        for part in PARTS:
            rep = calibration_report(run["preds"][part]["logits"], run["preds"][part]["labels"],
                                     temperature)
            for when in ("before", "after"):
                cal.append({"split": part, "temperature scaling": when,
                            "ECE": rep[when]["ece"], "NLL": rep[when]["nll"],
                            "mean confidence": rep[when]["mean_confidence"],
                            "accuracy": rep[when]["accuracy"]})
        plot_reliability_grid(run, temperature, args.fig_dir / "reliability.png")
        L += [f"## 4. Calibration (`{run['run_name']}`)\n",
              f"Temperature scaling (`src/calibrate.py`): one number T divides the logits. "
              f"Fitted on val-tiles: **T = {temperature:.3f}** "
              + ("(above 1: the model was over-confident and is made less sure)."
                 if temperature > 1 else "(below 1: the model was under-confident).")
              + " ECE = average gap between confidence and accuracy (15 bins); NLL = negative "
              "log-likelihood. Accuracy cannot change, because dividing by T never changes "
              "the top class.\n",
              df_to_markdown(pd.DataFrame(cal), float_fmt="{:.4f}", index=False) + "\n",
              f"![reliability diagrams]({fig_rel}/reliability.png)\n",
              "T was fitted on val-tiles, so the val-tiles 'after' number is the best case. "
              "The val rows show how well that one T carries over to unseen disasters.\n"]

        # 5. uncertainty methods
        seeds = [load_run(args.runs_dir / f"{better}_s{s}") for s in args.seeds]
        seeds = [r for r in seeds if r]
        methods = {}
        for part in PARTS:
            mc_path = run["dir"] / f"mc_dropout_{part}.npz"
            methods[part] = method_scores(
                run["preds"][part]["logits"], run["preds"][part]["labels"], temperature,
                mc=dict(np.load(mc_path)) if mc_path.exists() else None,
                ensemble_logits=[r["preds"][part]["logits"] for r in seeds])
        rows = []
        for part in PARTS:
            y = labels[part]
            for name, m in methods[part].items():
                s = summarize(y, m["pred"])
                top = m["probs"].max(axis=1)     # ECE needs a probability, not e.g. -entropy
                rows.append({"split": part, "method": name, "accuracy": s["accuracy"],
                             "macro-F1": s["macro_f1"],
                             "ECE": expected_calibration_error(
                                 m["confidence"] if m["confidence"].min() >= 0 else top,
                                 m["pred"] == y),
                             "AURC": aurc(m["confidence"], m["pred"] == y)})
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for ax, part in zip(axes, PARTS):
            plot_risk_coverage(ax, methods[part], labels[part], f"Risk-coverage on {part}")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "risk_coverage.png", dpi=130)
        plt.close(fig)
        have = list(methods["val"])
        L += [f"## 5. Uncertainty methods (`{better}`)\n",
              "Each method gives every prediction a confidence (`src/uncertainty.py`):\n",
              "- **max prob**: the top softmax probability of the seed-0 model, raw and after "
              "temperature scaling;",
              "- **MC dropout**: 20 passes with dropout left on; confidence = mean of the "
              "passes' top probabilities, or minus the entropy of the averaged probabilities"
              + ("" if any("MC" in h for h in have) else " *(not run yet)*") + ";",
              ("- **ensemble**: probabilities averaged over models trained with different "
               "seeds " + (f"({', '.join(r['run_name'] for r in seeds)})" if len(seeds) > 1 else
                           f"*(not available: only {seeds[0]['run_name']} has been trained)*")
               + ".\n"),
              ("" if len(have) > 2 else
               "**Only the max-probability method is available in this version; MC dropout "
               "and the ensemble are waiting on the runs listed in `BLOCKERS.md`.**\n"),
              "ECE: lower = confidence closer to accuracy. AURC (area under the "
              "risk-coverage curve): lower = mistakes are ranked lower, and/or fewer "
              "mistakes. For the entropy row, ECE is computed on the top averaged "
              "probability, because minus-entropy is not a probability.\n",
              df_to_markdown(pd.DataFrame(rows), float_fmt="{:.4f}", index=False) + "\n",
              f"![risk-coverage curves]({fig_rel}/risk_coverage.png)\n"]
        if len(seeds) > 1:
            per_seed = [{"run": r["run_name"], "best epoch": r["best_epoch"],
                         **{f"{p} {k}": r["best"][p][v] for p in PARTS
                            for k, v in (("accuracy", "accuracy"), ("macro-F1", "macro_f1"))}}
                        for r in seeds]
            L += ["The individual ensemble members, to show how much a seed alone changes:\n",
                  df_to_markdown(pd.DataFrame(per_seed), float_fmt="{:.3f}", index=False) + "\n"]

        # 6. triage
        rows = []
        for part in PARTS:
            for name, m in methods[part].items():
                for r in at_coverage(m["confidence"], labels[part], m["pred"]):
                    rows.append({"split": part, "method": name,
                                 "coverage": f"{r['coverage']:.0%}", "accuracy": r["accuracy"],
                                 "macro-F1": r["macro_f1"]})
        t = pd.DataFrame(rows)
        wide = t.pivot(index=["split", "method"], columns="coverage", values=["accuracy", "macro-F1"])
        order = [f"{c:.0%}" for c in COVERAGES]
        acc = wide["accuracy"][order].reindex([(p, n) for p in PARTS for n in methods[p]])
        f1 = wide["macro-F1"][order].reindex(acc.index)
        acc.index = [f"{p} / {n}" for p, n in acc.index]
        f1.index = acc.index
        acc.index.name = f1.index.name = "split / method"

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        for ax, part in zip(axes, PARTS):
            m = methods[part]
            cal_m = m["max prob (temp. scaled)"]
            best_name = [n for n in m if n.startswith("ensemble")] or ["max prob (temp. scaled)"]
            curves = {"least confident first (max prob, temp. scaled)": -cal_m["confidence"],
                      "most likely damaged first (1 - P(no-damage))": 1 - cal_m["probs"][:, 0]}
            if best_name[0].startswith("ensemble"):
                curves[f"most likely damaged first ({best_name[0]})"] = 1 - m[best_name[0]]["probs"][:, 0]
            damaged = labels[part] > 0
            plot_inspection(ax, curves, damaged,
                            f"Inspection budget on {part} ({damaged.mean():.0%} of buildings damaged)")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "inspection_budget.png", dpi=130)
        plt.close(fig)
        budget_rows = []
        for part in PARTS:
            cal_m = methods[part]["max prob (temp. scaled)"]
            damaged = labels[part] > 0
            for name, pr in (("least confident first", -cal_m["confidence"]),
                             ("most likely damaged first", 1 - cal_m["probs"][:, 0])):
                rec = inspection_recall(pr, damaged)
                budget_rows.append({"split": part, "policy": name,
                                    **{f"K={b:.0%}": r for b, r in zip(BUDGETS, rec) if round(b * 100) % 10 == 0}})
        L += ["## 6. Triage\n",
              "**Auto-accept view.** Keep the most confident X% of predictions "
              "(`src/triage.py`), send the rest to a person. Accuracy on the accepted "
              "buildings:\n",
              df_to_markdown(acc, float_fmt="{:.3f}") + "\n",
              "Macro-F1 on the accepted buildings (the accepted set is mostly easy, "
              "no-damage buildings; rare classes get rarer in it, so this number is "
              "noisier and can fall while accuracy rises):\n",
              df_to_markdown(f1, float_fmt="{:.3f}") + "\n",
              "**Inspection-budget view.** Inspectors can visit K% of the buildings. Share "
              "of the truly damaged buildings (minor, major or destroyed) among those "
              "visited, for the temperature-scaled seed-0 model:\n",
              df_to_markdown(pd.DataFrame(budget_rows), float_fmt="{:.1%}", index=False) + "\n",
              f"![inspection budget]({fig_rel}/inspection_budget.png)\n",
              "The two policies answer different questions: 'least confident first' looks "
              "for the model's likely mistakes, 'most likely damaged first' looks for "
              "damage. Random visits reach K% of the damaged buildings.\n"]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}" + (f" (better architecture: {better})" if better else ""))


if __name__ == "__main__":
    main()
