"""Collect the training runs into docs/results.md and docs/figures/.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\make_results.py

Reads runs/<setting>_s<seed>/metrics.json and the saved predictions; trains nothing and
needs no GPU, so it can be re-run at any time. Settings whose runs are missing are
skipped. Only val-tiles and val are ever read: the training runs never save test
predictions, and CropDataset refuses to load the test part (src/split.py).

A "setting" is one way of training (architecture, augmentation, preprocessing); each
setting is trained with several seeds, and the document reports the mean and the spread
over seeds, because on val a single run is not a reliable measurement.

Sections: baselines; settings compared (mean +- spread); does uncertainty flag failure
on val; gap diagnosis (pointer); calibration and triage of the base model; MC dropout
(set aside); earlier single-run comparisons; all runs.
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
from src.calibrate import calibration_report, plot_reliability  # noqa: E402
from src.colors import INK, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES, read_index  # noqa: E402
from src.metrics import expected_calibration_error, summarize  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402
from src.triage import (BUDGETS, COVERAGES, METHOD_COLORS, at_coverage, aurc,  # noqa: E402
                        inspection_recall, plot_inspection, plot_risk_coverage)
from src.uncertainty import collect_methods  # noqa: E402

PARTS = ["val-tiles", "val"]
SHORT = ["no-dmg", "minor", "major", "destr."]
# Settings in the order they are shown; the first one found is the base.
SETTINGS = {
    "stack_aug": "base: stack, strong augmentation",
    "stack_aug_pernorm": "base + per-image normalisation",
    "stack_aug_histmatch": "base + post colours histogram-matched to pre",
    # The pre-image experiments (docs/pre_image_plan.md).
    "stack_aug_samepair": "base + same-image training pairs",
    "siamese_diff_aug": "siamese, difference-only head, strong augmentation",
}
SINGLE, ENSEMBLE = "max prob (temp. scaled)", "ensemble"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Recorded results of work that is no longer re-run (runs made on 2026-10-01 under the
# earlier, single-epoch selection rule; their folders are not in git).
MC_DROPOUT_RECORD = [
    # split, method, accuracy, macro-F1, ECE, AURC, accuracy at 80% / 50% coverage
    ("val-tiles", "max prob (temp. scaled)", 0.8697, 0.6933, 0.0154, 0.0238, 0.958, 0.993),
    ("val-tiles", "MC dropout (mean max prob)", 0.8695, 0.6930, 0.0156, 0.0240, 0.958, 0.993),
    ("val-tiles", "MC dropout (entropy)", 0.8695, 0.6930, 0.0160, 0.0244, 0.956, 0.993),
    ("val", "max prob (temp. scaled)", 0.6133, 0.3079, 0.0426, 0.2022, 0.676, 0.795),
    ("val", "MC dropout (mean max prob)", 0.6143, 0.3085, 0.0423, 0.2017, 0.677, 0.796),
    ("val", "MC dropout (entropy)", 0.6143, 0.3085, 0.0405, 0.2021, 0.670, 0.793),
]
EARLIER_RUNS_RECORD = [
    # run, what, epoch kept, val-tiles acc, val-tiles F1, val acc, val F1
    ("siamese_s0", "siamese, base augmentation", "10", 0.880, 0.704, 0.483, 0.256),
    ("stack_s0", "stack, base augmentation", "4 of 7", 0.871, 0.685, 0.676, 0.297),
    ("stack_aug_s0", "stack, strong augmentation", "9 of 10", 0.870, 0.693, 0.613, 0.308),
    ("stack_aug_s1", "same, seed 1", "9 of 10", 0.870, 0.693, 0.657, 0.310),
    ("stack_aug_s2", "same, seed 2", "4 of 7", 0.862, 0.669, 0.761, 0.337),
    ("ensemble of those 3", "probabilities averaged", "", 0.875, 0.700, 0.709, 0.331),
]


def load_run(run_dir):
    """metrics.json of one finished run (None if it has not finished)."""
    run_dir = Path(run_dir)
    if not (run_dir / "metrics.json").exists() or not (run_dir / "preds_val.npz").exists():
        return None
    run = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run["dir"] = run_dir
    return run


def metric_row(m):
    """accuracy, macro-F1 and the four recalls of a metrics dict, as table cells."""
    row = {"accuracy": m["accuracy"], "macro-F1": m["macro_f1"]}
    for name, r in zip(SHORT, m["recall"]):
        row[f"recall {name}"] = np.nan if r is None else r
    return row


def spread(values, digits=3):
    """'mean +- standard deviation (min-max)' of a handful of per-seed numbers.

    The standard deviation uses n - 1 (it estimates the spread of seeds in general from
    the few we have). With 3 seeds it is itself a rough number: read it as "about this
    much", not to the last digit.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return f"{v[0]:.{digits}f} (1 seed)"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f} ({v.min():.{digits}f}-{v.max():.{digits}f})"


def auroc(confidence, correct):
    """Probability that a randomly chosen correct prediction has a higher confidence than
    a randomly chosen wrong one. 0.5 = confidence says nothing about being right;
    1.0 = every correct prediction outranks every mistake. Computed from ranks
    (the Mann-Whitney statistic), ties counting half."""
    correct = np.asarray(correct, dtype=bool)
    n_pos, n_neg = correct.sum(), (~correct).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(confidence, dtype=float)).rank().to_numpy()
    return float((ranks[correct].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def errors_caught(confidence, correct, share):
    """Share of all mistakes that sit among the least-confident ``share`` of predictions.

    If confidence were unrelated to being right, the least-confident 20% would contain
    20% of the mistakes. The closer to 100%, the better confidence flags failure.
    """
    confidence, wrong = np.asarray(confidence, dtype=float), ~np.asarray(correct, dtype=bool)
    k = max(1, int(round(share * len(confidence))))
    lowest = np.argsort(confidence, kind="stable")[:k]
    return float(wrong[lowest].sum() / max(wrong.sum(), 1))


def flag_stats(method, y):
    right = method["pred"] == y
    return {"accuracy": float(right.mean()), "mean confidence": float(np.mean(method["confidence"])),
            "AUROC": auroc(method["confidence"], right),
            "errors in least-confident 20%": errors_caught(method["confidence"], right, 0.20),
            "errors in least-confident 50%": errors_caught(method["confidence"], right, 0.50)}


# ---- baselines -----------------------------------------------------------------------

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

def plot_confusions(panels, labels, path):
    """Row-normalised confusion matrices. ``panels`` = [(title, {part: predictions})]."""
    fig, axes = plt.subplots(len(panels), len(PARTS), figsize=(4.3 * len(PARTS), 3.9 * len(panels)),
                             squeeze=False)
    for i, (title, preds) in enumerate(panels):
        for j, part in enumerate(PARTS):
            cm = np.array(summarize(labels[part], preds[part])["confusion"], dtype=float)
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
            ax.set_title(f"{title} on {part}\n(each row sums to 100%)", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability_methods(methods, labels, path):
    """One reliability diagram per method (columns) and split (rows)."""
    names = list(methods["val"])
    fig, axes = plt.subplots(2, len(names), figsize=(3.5 * len(names), 6.6), squeeze=False)
    for i, part in enumerate(PARTS):
        for j, name in enumerate(names):
            m = methods[part][name]
            plot_reliability(axes[i, j], m["confidence"], m["pred"] == labels[part],
                             f"{part}\n{name}", color=METHOD_COLORS[j % len(METHOD_COLORS)])
            if j:
                axes[i, j].set_ylabel("")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence_histograms(single, ensemble, labels, path):
    """Where do right and wrong predictions sit on the confidence axis?"""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
    bins = np.linspace(0.25, 1.0, 31)
    for i, (name, methods) in enumerate((("single model (seed 0)", single), ("ensemble", ensemble))):
        for j, part in enumerate(PARTS):
            m = methods[part]
            right = m["pred"] == labels[part]
            ax = axes[i, j]
            ax.hist(m["confidence"][right], bins=bins, color="#2a78d6", alpha=0.75,
                    label=f"correct ({right.mean():.0%})")
            ax.hist(m["confidence"][~right], bins=bins, color="#e34948", alpha=0.75,
                    label=f"wrong ({(~right).mean():.0%})")
            ax.set_title(f"{name} on {part}", fontsize=10)
            ax.set_ylabel("buildings")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
            ax.grid(color="#e1e0d9", linewidth=0.6)
            ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- the document ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--cache-dir", type=Path, default=Path(r"C:\Users\wikel\dev\xbd-data\crops"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("docs/results.md"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = args.fig_dir.relative_to(args.out.parent).as_posix()

    index = read_index(args.cache_dir)
    labels = {p: index[index["part"] == p].sort_values("row")["label"].to_numpy() for p in PARTS}
    train_labels = index[index["part"] == "train"]["label"].to_numpy()

    # Every setting: its finished seeds, and every confidence method built from them.
    settings = {}
    for name, what in SETTINGS.items():
        runs = [load_run(args.runs_dir / f"{name}_s{s}") for s in args.seeds]
        runs = [r for r in runs if r]
        if not runs:
            continue
        dirs = [r["dir"] for r in runs]
        per_seed = [collect_methods(d, [])[0] for d in dirs]            # each model alone
        together, info = collect_methods(dirs[0], dirs)                  # seed 0 + the ensemble
        ens_key = next((k for k in together["val"] if k.startswith("ensemble") and "scaled" in k), None)
        settings[name] = {"what": what, "runs": runs, "per_seed": per_seed,
                          "together": together, "info": info, "ens_key": ens_key}
    base_name = next(iter(settings), None)

    L = ["# Results (validation only)\n",
         "Generated by `scripts/make_results.py` from `runs/`. **Every number here is on "
         "val-tiles or val. The test set (hurricane-michael, santa-rosa-wildfire) has not "
         "been loaded, evaluated or looked at**; `src/split.py` blocks it unless `--final` "
         "is passed.\n",
         "- **val-tiles**: held-out spatial blocks of the six *training* disasters "
         f"({len(labels['val-tiles']):,} buildings). Used for model selection and for "
         "fitting the temperature, so scores on it are optimistic.",
         f"- **val**: hurricane-florence and midwest-flooding, never trained on "
         f"({len(labels['val']):,} buildings). Floods only, with few minor-damage and "
         f"destroyed buildings ({int((labels['val'] == 1).sum())} and "
         f"{int((labels['val'] == 3).sum())}), so its per-class numbers for those two "
         "classes are noisy.\n",
         "> **val is used for model selection.** The checkpoint that is kept is the last "
         "epoch of the best 3-epoch stretch of the *mean of the val-tiles and val "
         "macro-F1* (`smoothed_score` in `src/train.py`), and the settings below were "
         "compared on val. val scores are therefore not an untouched estimate for a new "
         "disaster: they are somewhat optimistic. The only untouched data is the test "
         "set.\n",
         "Un-classified buildings are dropped (`docs/data_decisions.md`). Recall of a class "
         "= share of its buildings that were predicted as that class. Macro-F1 = plain "
         "average of the four per-class F1 scores. **'a ± b (c-d)'** = mean, standard "
         "deviation and range over the seeds of a setting; with 3 seeds the standard "
         "deviation is itself rough.\n"]

    # 1. baselines
    base_tbl, freq = baselines(labels, train_labels)
    majority = {p: base_tbl[(base_tbl["split"] == p)
                            & base_tbl["baseline"].str.startswith("majority")].iloc[0] for p in PARTS}
    L += ["## 1. Baselines\n",
          "What you get without looking at any image. Class shares in train: "
          + ", ".join(f"{n} {f:.1%}" for n, f in zip(CLASS_NAMES, freq)) + ".\n",
          df_to_markdown(base_tbl, float_fmt="{:.3f}", index=False) + "\n",
          "Majority-class accuracy is high because most buildings are undamaged; its "
          "macro-F1 shows it is useless. A model has to beat **both** numbers.\n"]

    if not settings:
        args.out.write_text("\n".join(L), encoding="utf-8")
        print(f"Wrote {args.out} (no finished runs)")
        return

    # 2. settings compared
    rows, ens_rows = [], []
    for name, s in settings.items():
        row = {"setting": s["what"], "runs": f"`{name}_s*` ({len(s['runs'])} seeds)"}
        for p in PARTS:
            row[f"{p} accuracy"] = spread([r["best"][p]["accuracy"] for r in s["runs"]])
            row[f"{p} macro-F1"] = spread([r["best"][p]["macro_f1"] for r in s["runs"]])
        rows.append(row)
        if s["ens_key"]:
            e = {"setting": s["what"]}
            for p in PARTS:
                m = summarize(labels[p], s["together"][p][s["ens_key"]]["pred"])
                e[f"{p} accuracy"], e[f"{p} macro-F1"] = m["accuracy"], m["macro_f1"]
            ens_rows.append(e)
    base = settings[base_name]
    L += ["## 2. Settings compared: mean ± spread over seeds\n",
          "Every setting uses the strong augmentation and is trained with seeds "
          f"{', '.join(str(s) for s in args.seeds)}. The base is the stack ResNet-18; each "
          "other setting changes one thing (settings without finished runs are left out):\n",
          "- **per-image normalisation**: each image is shifted and scaled by its own "
          "channel means and standard deviations instead of ImageNet's fixed ones, which "
          "removes overall brightness, contrast and colour cast (`src/dataset.py`);",
          "- **histogram matching**: the post image is recoloured so that each channel "
          "has the pre image's distribution of values (`src/dataset.py`);",
          "- **same-image training pairs**: a quarter of the training examples show one of "
          "the building's two images twice, labelled no-damage, to make the model compare "
          "(`docs/pre_image_plan.md`, candidate A);",
          "- **siamese, difference-only head**: another architecture, whose classifier "
          "sees only post features minus pre features (`src/model_diff.py`; "
          "`docs/pre_image_plan.md`, candidate B).\n",
          "Single models (each seed on its own):\n",
          df_to_markdown(pd.DataFrame(rows), index=False) + "\n"]
    if ens_rows:
        L += ["Ensemble of each setting's seeds (probabilities averaged; one number per "
              "setting, so no spread):\n",
              df_to_markdown(pd.DataFrame(ens_rows), float_fmt="{:.3f}", index=False) + "\n"]
    # A plain-language reading of each fix against the base, on val.
    b_acc = np.array([r["best"]["val"]["accuracy"] for r in base["runs"]])
    b_f1 = np.array([r["best"]["val"]["macro_f1"] for r in base["runs"]])
    for name, s in list(settings.items())[1:]:
        acc = np.array([r["best"]["val"]["accuracy"] for r in s["runs"]])
        f1 = np.array([r["best"]["val"]["macro_f1"] for r in s["runs"]])
        vt = (np.mean([r["best"]["val-tiles"]["macro_f1"] for r in s["runs"]])
              - np.mean([r["best"]["val-tiles"]["macro_f1"] for r in base["runs"]]))
        apart_f1 = f1.min() > b_f1.max() or f1.max() < b_f1.min()
        apart_acc = acc.min() > b_acc.max() or acc.max() < b_acc.min()
        L.append(f"- **{s['what']}** against the base, on val: macro-F1 "
                 f"{f1.mean() - b_f1.mean():+.3f}, accuracy {acc.mean() - b_acc.mean():+.3f} "
                 f"(difference of means); on val-tiles macro-F1 {vt:+.3f}. The seeds' "
                 f"val macro-F1 ranges {'do not overlap' if apart_f1 else 'overlap'} and "
                 f"their val accuracy ranges {'do not overlap' if apart_acc else 'overlap'}"
                 + (f": with {len(base['runs'])} and {len(s['runs'])} seeds, that is a difference worth taking seriously."
                    if apart_f1 and apart_acc else
                    f": with {len(base['runs'])} and {len(s['runs'])} seeds, overlapping ranges mean the difference could be "
                    "seed luck." if not (apart_f1 or apart_acc) else
                    ": mixed evidence."))
    best_val = max(settings, key=lambda n: np.mean([r["best"]["val"]["accuracy"]
                                                    for r in settings[n]["runs"]]))
    acc_best = np.mean([r["best"]["val"]["accuracy"] for r in settings[best_val]["runs"]])
    L += [f"\nFor scale: always answering no-damage scores {majority['val']['accuracy']:.3f} "
          f"accuracy on val. The best setting's mean val accuracy is {acc_best:.3f} "
          f"(`{best_val}`), "
          + ("still below it." if acc_best < majority["val"]["accuracy"] else "above it.") + "\n"]
    panels = [(f"{n}, seed 0", {p: s["per_seed"][0][p][SINGLE]["pred"] for p in PARTS})
              for n, s in settings.items()]
    plot_confusions(panels, labels, args.fig_dir / "confusion_matrices.png")
    L += [f"![confusion matrices of seed 0 of each setting]({fig_rel}/confusion_matrices.png)\n"]

    # 3. the key question
    rows = []
    for name, s in settings.items():
        for p in PARTS:
            per = [flag_stats(m[p][SINGLE], labels[p]) for m in s["per_seed"]]
            row = {"setting": s["what"], "split": p, "model": f"single ({len(per)} seeds)"}
            row.update({k: spread([x[k] for x in per]) for k in per[0]})
            rows.append(row)
            if s["ens_key"]:
                e = flag_stats(s["together"][p][s["ens_key"]], labels[p])
                rows.append({"setting": s["what"], "split": p, "model": "ensemble",
                             **{k: f"{v:.3f}" for k, v in e.items()}})
    flags = pd.DataFrame(rows)
    b = base
    single_val = [flag_stats(m["val"][SINGLE], labels["val"]) for m in b["per_seed"]]
    single_vt = [flag_stats(m["val-tiles"][SINGLE], labels["val-tiles"]) for m in b["per_seed"]]
    L += ["## 3. Does uncertainty flag failure on val?\n",
          "The triage idea needs the model to be *less sure when it is wrong*, most of "
          "all on a disaster it has not seen. Three ways to check, for the single model "
          "(temperature-scaled top probability; mean ± spread over seeds) and for the "
          "ensemble (averaged probabilities, with its own temperature):\n",
          "- **AUROC**: the chance that a correct prediction has a higher confidence than "
          "a wrong one. 0.5 = confidence says nothing; 1 = perfect separation.",
          "- **errors in the least-confident 20% / 50%**: of all the mistakes, the share "
          "that sits among the least confident 20% (or 50%) of predictions. Unrelated "
          "confidence would give 20% (or 50%).",
          "- **mean confidence** on val against val-tiles, next to accuracy: does the model "
          "*know* it is on unfamiliar ground? Confidence should fall when accuracy "
          "falls.\n",
          df_to_markdown(flags, index=False) + "\n"]
    conf_drop = np.mean([x["mean confidence"] for x in single_vt]) - np.mean([x["mean confidence"] for x in single_val])
    acc_drop = np.mean([x["accuracy"] for x in single_vt]) - np.mean([x["accuracy"] for x in single_val])
    gap_val = np.mean([x["mean confidence"] - x["accuracy"] for x in single_val])
    L += [f"**Reading it, for the base setting (`{base_name}`), single model:**\n",
          f"- From val-tiles to val, accuracy falls by {acc_drop:.3f} and mean confidence "
          f"by {conf_drop:.3f}. " + (
              "Confidence falls by about as much as accuracy: the model does register that "
              "val is harder." if abs(conf_drop - acc_drop) < 0.05 else
              "Confidence falls, but by less than accuracy: the model registers that val "
              "is harder, and still overstates how well it is doing there "
              f"(mean confidence exceeds accuracy by {gap_val:.3f} on val)."
              if conf_drop < acc_drop else
              "Confidence falls by more than accuracy.") + "",
          f"- AUROC on val is {np.mean([x['AUROC'] for x in single_val]):.3f} against "
          f"{np.mean([x['AUROC'] for x in single_vt]):.3f} on val-tiles: confidence still "
          "ranks mistakes below correct answers on val, but much less sharply.",
          f"- The least-confident 20% of val predictions hold "
          f"{np.mean([x['errors in least-confident 20%'] for x in single_val]):.0%} of the "
          f"val mistakes, and the least-confident 50% hold "
          f"{np.mean([x['errors in least-confident 50%'] for x in single_val]):.0%} "
          f"(on val-tiles: {np.mean([x['errors in least-confident 20%'] for x in single_vt]):.0%} "
          f"and {np.mean([x['errors in least-confident 50%'] for x in single_vt]):.0%})."]
    if b["ens_key"]:
        e_val = flag_stats(b["together"]["val"][b["ens_key"]], labels["val"])
        aurocs = [x["AUROC"] for x in single_val]
        L.append(f"- The ensemble on val: AUROC {e_val['AUROC']:.3f} (single models "
                 f"{min(aurocs):.3f}-{max(aurocs):.3f}), "
                 f"{e_val['errors in least-confident 20%']:.0%} of mistakes in the "
                 f"least-confident 20%, {e_val['errors in least-confident 50%']:.0%} in the "
                 "least-confident 50%. "
                 + ("It ranks mistakes better than every one of its members."
                    if e_val["AUROC"] > max(aurocs) else
                    "It is within the range of its members: no clear gain in flagging "
                    "failure." if e_val["AUROC"] >= min(aurocs) else
                    "It ranks mistakes worse than its members."))
        plot_confidence_histograms({p: b["per_seed"][0][p][SINGLE] for p in PARTS},
                                   {p: b["together"][p][b["ens_key"]] for p in PARTS},
                                   labels, args.fig_dir / "confidence_right_wrong.png")
        L += [f"\n![confidence of correct and wrong predictions]({fig_rel}/confidence_right_wrong.png)\n"]
    L += ["**Short answer:** partly. On val the confidence carries real information about "
          "which predictions are wrong (AUROC well above 0.5, mistakes concentrated among "
          "the least confident), but far less than on val-tiles, and with so many "
          "mistakes on val a large share of them still sits among confident "
          "predictions. Section 5 turns this into coverage and inspection numbers.\n"]

    # 4. gap diagnosis
    L += ["## 4. Why is val worse? (gap diagnosis)\n",
          "Two diagnostics, input swaps (which of the two images carries the shift) and "
          "colour statistics per disaster, are in **`docs/gap_diagnosis.md`** "
          "(`scripts/gap_diagnosis.py`). The two fixes in section 2 are the direct test "
          "of the colour explanation.\n"]

    # 5. calibration and triage of the base setting
    run0 = base["runs"][0]
    methods, info = base["together"], base["info"]
    preds0 = {p: np.load(run0["dir"] / f"preds_{p}.npz") for p in PARTS}
    cal = []
    for part in PARTS:
        rep = calibration_report(preds0[part]["logits"], preds0[part]["labels"], info["temperature"])
        for when in ("before", "after"):
            cal.append({"split": part, "temperature scaling": when, "ECE": rep[when]["ece"],
                        "NLL": rep[when]["nll"], "mean confidence": rep[when]["mean_confidence"],
                        "accuracy": rep[when]["accuracy"]})
    rows = []
    for part in PARTS:
        y = labels[part]
        for name, m in methods[part].items():
            s = summarize(y, m["pred"])
            rows.append({"split": part, "method": name, "accuracy": s["accuracy"],
                         "macro-F1": s["macro_f1"],
                         "ECE": expected_calibration_error(m["confidence"], m["pred"] == y),
                         "AURC": aurc(m["confidence"], m["pred"] == y)})
    plot_reliability_methods(methods, labels, args.fig_dir / "reliability_methods.png")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, part in zip(axes, PARTS):
        plot_risk_coverage(ax, methods[part], labels[part], f"Risk-coverage on {part}")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "risk_coverage.png", dpi=130)
    plt.close(fig)
    t = info["temperature"]
    L += [f"## 5. Calibration and triage (`{base_name}`)\n",
          f"Single model = `{run0['run_name']}`; ensemble = "
          f"{', '.join('`' + m + '`' for m in info['members'])}.\n",
          "### Calibration\n",
          f"Temperature scaling (`src/calibrate.py`): one number T divides the logits, "
          f"fitted on val-tiles. Single model: **T = {t:.3f}** "
          + ("(above 1: over-confident, made less sure)" if t > 1 else "(below 1: under-confident)")
          + (f"; ensemble: T = {info['ensemble_temperature']:.3f}"
             if info["ensemble_temperature"] else "")
          + ". ECE = average gap between confidence and accuracy (15 bins); NLL = negative "
          "log-likelihood. Scaling never changes the predicted class.\n",
          df_to_markdown(pd.DataFrame(cal), float_fmt="{:.4f}", index=False) + "\n",
          "All methods (AURC = area under the risk-coverage curve, lower is better):\n",
          df_to_markdown(pd.DataFrame(rows), float_fmt="{:.4f}", index=False) + "\n",
          f"![reliability per method]({fig_rel}/reliability_methods.png)\n",
          f"![risk-coverage curves]({fig_rel}/risk_coverage.png)\n"]

    if base["ens_key"]:
        # each member alone next to the ensemble
        per_member = []
        for r, own in zip(base["runs"], base["per_seed"]):
            row = {"model": r["run_name"], "epoch kept": f"{r['best_epoch']} of {r['epochs_run']}"}
            for p in PARTS:
                m = own[p][SINGLE]
                cov = at_coverage(m["confidence"], labels[p], m["pred"], (1.0, 0.5))
                row.update({f"{p} accuracy": cov[0]["accuracy"],
                            f"{p} macro-F1": r["best"][p]["macro_f1"],
                            f"{p} AURC": aurc(m["confidence"], m["pred"] == labels[p]),
                            f"{p} accuracy at 50% coverage": cov[1]["accuracy"]})
            per_member.append(row)
        row = {"model": "ensemble", "epoch kept": ""}
        for p in PARTS:
            m = methods[p][base["ens_key"]]
            cov = at_coverage(m["confidence"], labels[p], m["pred"], (1.0, 0.5))
            row.update({f"{p} accuracy": cov[0]["accuracy"],
                        f"{p} macro-F1": summarize(labels[p], m["pred"])["macro_f1"],
                        f"{p} AURC": aurc(m["confidence"], m["pred"] == labels[p]),
                        f"{p} accuracy at 50% coverage": cov[1]["accuracy"]})
        per_member.append(row)
        L += ["The ensemble next to each of its members (each member with its own "
              "temperature):\n",
              df_to_markdown(pd.DataFrame(per_member), float_fmt="{:.3f}", index=False) + "\n"]

    rows = []
    for part in PARTS:
        for name, m in methods[part].items():
            for r in at_coverage(m["confidence"], labels[part], m["pred"]):
                rows.append({"split": part, "method": name, "coverage": f"{r['coverage']:.0%}",
                             "accuracy": r["accuracy"], "macro-F1": r["macro_f1"]})
    t = pd.DataFrame(rows)
    wide = t.pivot(index=["split", "method"], columns="coverage", values=["accuracy", "macro-F1"])
    order = [f"{c:.0%}" for c in COVERAGES]
    acc = wide["accuracy"][order].reindex([(p, n) for p in PARTS for n in methods[p]])
    f1 = wide["macro-F1"][order].reindex(acc.index)
    acc.index = [f"{p} / {n}" for p, n in acc.index]
    f1.index = acc.index
    acc.index.name = f1.index.name = "split / method"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    budget_rows = []
    for ax, part in zip(axes, PARTS):
        damaged = labels[part] > 0
        curves = {}
        for k, name in enumerate(n for n in methods[part] if n != "max prob (raw)"
                                 and not (n.startswith("ensemble") and "scaled" not in n)):
            m = methods[part][name]
            color = METHOD_COLORS[1 + k % (len(METHOD_COLORS) - 1)]
            for policy, priority, style in (("least confident first", -m["confidence"], "--"),
                                            ("most likely damaged first", 1 - m["probs"][:, 0], "-")):
                curves[f"{policy}: {name}"] = (priority, color, style)
                budget_rows.append({"split": part, "method": name, "policy": policy,
                                    **{f"K={b:.0%}": r for b, r in
                                       zip(BUDGETS, inspection_recall(priority, damaged))
                                       if round(b * 100) % 10 == 0}})
        plot_inspection(ax, curves, damaged,
                        f"Inspection budget on {part} ({damaged.mean():.0%} of buildings damaged)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "inspection_budget.png", dpi=130)
    plt.close(fig)
    L += ["### Triage\n",
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
          "visited. 'Most likely damaged' ranks by 1 - P(no-damage); 'least confident' "
          "ranks by confidence. Random visits reach K%.\n",
          df_to_markdown(pd.DataFrame(budget_rows), float_fmt="{:.1%}", index=False) + "\n",
          f"![inspection budget]({fig_rel}/inspection_budget.png)\n",
          "In the chart, solid lines are 'most likely damaged first' and dashed lines "
          "'least confident first'; colour is the method.\n"]

    # 6. MC dropout, set aside
    mc = pd.DataFrame(MC_DROPOUT_RECORD, columns=["split", "method", "accuracy", "macro-F1", "ECE",
                                                  "AURC", "accuracy at 80% coverage",
                                                  "accuracy at 50% coverage"])
    L += ["## 6. MC dropout: tried and set aside\n",
          "MC dropout (20 forward passes with dropout left on, each pass "
          "temperature-scaled; confidence = mean of the passes' top probabilities, or "
          "minus the entropy of the averaged probabilities) was run once, on an earlier "
          "`stack_aug` seed-0 model. Its numbers next to the plain top probability of the "
          "same model:\n",
          df_to_markdown(mc, float_fmt="{:.4f}", index=False) + "\n",
          "It is practically identical to the plain top probability on every measure, at "
          "20 times the prediction cost. The reason is structural: in these models "
          "dropout sits only in front of the last layer, so the 20 passes barely differ. "
          "It is therefore no longer part of the comparison. The code remains in "
          "`src/uncertainty.py` (`mc_dropout_probs`), tested; it would be worth "
          "revisiting only with dropout inside the backbone, which changes training.\n"]

    # 7. earlier single-run comparisons
    earlier = pd.DataFrame(EARLIER_RUNS_RECORD, columns=["run", "what", "epoch kept",
                                                         "val-tiles acc", "val-tiles macro-F1",
                                                         "val acc", "val macro-F1"])
    L += ["## 7. Earlier single-run comparisons (recorded)\n",
          "The first runs, kept for the record. They used the earlier selection rule "
          "(best single epoch of the mean macro-F1, no 3-epoch window); `siamese_s0` kept "
          "its weights by val-tiles macro-F1 alone, and its row is taken from its "
          "per-epoch log. Sanity check before any of this: the siamese model overfitted "
          "256 training crops to 100% in 4 epochs.\n",
          df_to_markdown(earlier, float_fmt="{:.3f}", index=False) + "\n",
          "What they showed: stack edged out siamese on the selection score (0.491 "
          "against 0.480), and the strong augmentation gave a mixed result against plain "
          "stack (val macro-F1 +0.011, val accuracy -0.063). Both were one run per "
          "setting, and the three `stack_aug` seeds differ by more than either gap, so "
          "neither comparison is settled.\n"]

    # 8. all runs
    all_runs = [r for s in settings.values() for r in s["runs"]]
    rows = [{"run": r["run_name"], "epoch kept": f"{r['best_epoch']} of {r['epochs_run']}",
             "val-tiles acc": r["best"]["val-tiles"]["accuracy"],
             "val-tiles macro-F1": r["best"]["val-tiles"]["macro_f1"],
             "val acc": r["best"]["val"]["accuracy"], "val macro-F1": r["best"]["val"]["macro_f1"],
             "smoothed selection score": r["best_selection_score"],
             "loader workers": r["num_workers"],
             "train s/epoch": int(round(r["train_seconds_per_epoch"])),
             "RAM of this run, peak GB": r["run_peak_private_gb"],
             "RAM free, minimum GB": r["ram_min_available_gb"],
             "GPU peak GB (reserved)": r["gpu_peak_reserved_gb"] if r["gpu_peak_reserved_gb"] is not None else np.nan}
            for r in all_runs]
    L += ["## 8. All runs\n",
          "Every run behind sections 2-5, at the epoch that was kept.\n",
          df_to_markdown(pd.DataFrame(rows), float_fmt="{:.3f}", index=False) + "\n",
          "Per-epoch metrics and memory are in `runs/<run>/log.jsonl` (not in git).\n"]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out} (settings: {', '.join(settings)})")


if __name__ == "__main__":
    main()
