"""Collect the training runs into docs/results.md and docs/figures/.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\make_results.py --base siamese_aug

Reads runs/<run>/metrics.json and the saved predictions; trains nothing and needs no GPU,
so it can be re-run at any time. Sections whose runs are missing are skipped.
Only val-tiles and val are ever read: the training runs never save test predictions, and
CropDataset refuses to load the test part (src/split.py).

Run names: <arch>_s<seed> for the base augmentation, <arch>_aug_s<seed> for the strong
augmentation. ``--base`` names the model the uncertainty sections are built on (without
the seed suffix); by default it is chosen from the runs, see ``choose_base``.

Sections: baselines, sanity check, siamese vs stack, augmentation experiment,
calibration, uncertainty methods, triage, table of all runs.
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
from src.colors import INK, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES, read_index  # noqa: E402
from src.metrics import expected_calibration_error, summarize  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402
from src.triage import (BUDGETS, COVERAGES, METHOD_COLORS, at_coverage, aurc,  # noqa: E402
                        inspection_recall, plot_inspection, plot_risk_coverage)
from src.uncertainty import collect_methods, method_scores  # noqa: E402

PARTS = ["val-tiles", "val"]
SHORT = ["no-dmg", "minor", "major", "destr."]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def mean_score(h):
    """The selection score: mean of the val-tiles and val macro-F1 of one epoch."""
    return (h["val-tiles"]["macro_f1"] + h["val"]["macro_f1"]) / 2


def load_run(run_dir):
    """metrics.json plus predictions of one finished run, or None.

    Adds ``rule_epoch`` / ``rule`` / ``rule_score``: the epoch the current selection rule
    (mean of val-tiles and val macro-F1) picks, and that epoch's metrics from the
    per-epoch log. For runs trained under this rule that is simply the saved best epoch.
    ``saved_matches_rule`` is False for the first run, which kept its weights by
    val-tiles macro-F1 alone.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "metrics.json").exists() or not (run_dir / "preds_val.npz").exists():
        return None
    run = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run["dir"] = run_dir
    run["preds"] = {p: dict(np.load(run_dir / f"preds_{p}.npz")) for p in PARTS}
    scores = [mean_score(h) for h in run["history"]]
    run["rule_epoch"] = int(np.argmax(scores)) + 1
    run["rule"] = run["history"][run["rule_epoch"] - 1]
    run["rule_score"] = float(max(scores))
    run["saved_matches_rule"] = run["rule_epoch"] == run["best_epoch"]
    return run


def metric_row(m):
    """accuracy, macro-F1 and the four recalls of a metrics dict, as table cells."""
    row = {"accuracy": m["accuracy"], "macro-F1": m["macro_f1"]}
    for name, r in zip(SHORT, m["recall"]):
        row[f"recall {name}"] = np.nan if r is None else r
    return row


def comparison_rows(runs):
    return pd.DataFrame([{"model": r["run_name"], "split": part, **metric_row(r["rule"][part])}
                         for r in runs for part in PARTS])


def cost_rows(runs):
    rows = []
    for r in runs:
        rows.append({
            "run": r["run_name"], "epoch kept": r["rule_epoch"], "epochs run": r["epochs_run"],
            "loader workers": r.get("num_workers", 8),
            "train s/epoch": int(round(r["train_seconds_per_epoch"])),
            "s/epoch incl. eval": int(round(r["seconds_per_epoch"])),
            "RAM used, peak GB (whole machine)": r.get("ram_peak_used_gb", np.nan),
            "RAM of this run, peak GB": r.get("run_peak_private_gb", np.nan),
            "GPU peak GB (allocated / reserved)":
                (f"{r['gpu_peak_allocated_gb']:.1f} / {r['gpu_peak_reserved_gb']:.1f}"
                 if r['gpu_peak_allocated_gb'] is not None else "no GPU"),
        })
    return pd.DataFrame(rows)


def choose_base(plain, aug):
    """The model the rest of the document is built on.

    The augmentation experiment is judged on val macro-F1 (the unseen disasters), which
    is the gap it was meant to close.
    """
    if aug is None:
        return plain
    return aug if aug["rule"]["val"]["macro_f1"] > plain["rule"]["val"]["macro_f1"] else plain


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
            cm = np.array(run["rule"][part]["confusion"], dtype=float)
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


def plot_reliability_methods(methods, labels, path):
    """One reliability diagram per method (columns) and split (rows)."""
    names = [n for n in methods["val"] if n != "MC dropout (entropy)"]   # not a probability
    fig, axes = plt.subplots(2, len(names), figsize=(3.5 * len(names), 6.6), squeeze=False)
    for i, part in enumerate(PARTS):
        for j, name in enumerate(names):
            m = methods[part][name]
            plot_reliability(axes[i, j], m["confidence"], m["pred"] == labels[part],
                             f"{part}\n{name}", color=METHOD_COLORS[j % len(METHOD_COLORS)])
            if j:
                axes[i, j].set_ylabel("")
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
    ap.add_argument("--base", default=None,
                    help="run name without the seed suffix, e.g. siamese_aug (default: chosen)")
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
         f"({len(labels['val-tiles']):,} buildings). Used for model selection and for "
         "fitting the temperature, so scores on it are optimistic.",
         f"- **val**: hurricane-florence and midwest-flooding, never trained on "
         f"({len(labels['val']):,} buildings). Floods only, with few minor-damage and "
         f"destroyed buildings ({int((labels['val'] == 1).sum())} and "
         f"{int((labels['val'] == 3).sum())}), so its per-class numbers for those two "
         "classes are noisy.\n",
         "> **val is now used for model selection.** The epoch whose weights are kept is "
         "the one with the highest *mean of the val-tiles and val macro-F1*, and the "
         "architecture and augmentation choices below were judged on val. val scores are "
         "therefore no longer an untouched estimate for a new disaster: they are somewhat "
         "optimistic. The only untouched data is the test set.\n",
         "Un-classified buildings are dropped (`docs/data_decisions.md`). Recall of a class "
         "= share of its buildings that were predicted as that class. Macro-F1 = plain "
         "average of the four per-class F1 scores.\n"]

    # 1. baselines
    base_tbl, freq = baselines(labels, train_labels)
    L += ["## 1. Baselines\n",
          "What you get without looking at any image. Class shares in train: "
          + ", ".join(f"{n} {f:.1%}" for n, f in zip(CLASS_NAMES, freq)) + ".\n",
          df_to_markdown(base_tbl, float_fmt="{:.3f}", index=False) + "\n",
          "Majority-class accuracy is high because most buildings are undamaged; its "
          "macro-F1 shows it is useless. A model has to beat **both** numbers.\n"]

    def versus_baseline(run):
        out = []
        for part in PARTS:
            maj = base_tbl[(base_tbl["split"] == part)
                           & base_tbl["baseline"].str.startswith("majority")].iloc[0]
            m = run["rule"][part]
            verdict = "above" if m["accuracy"] > maj["accuracy"] else "**below**"
            out.append(f"- On **{part}**, `{run['run_name']}` has accuracy {m['accuracy']:.3f}, "
                       f"{verdict} the majority-class baseline ({maj['accuracy']:.3f}), and "
                       f"macro-F1 {m['macro_f1']:.3f} against the baselines' "
                       f"{base_tbl[base_tbl['split'] == part]['macro-F1'].max():.3f}.")
        return out

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
    archs = {a: load_run(args.runs_dir / f"{a}_s0") for a in args.archs}
    archs = {a: r for a, r in archs.items() if r}
    better = base = None
    if archs:
        better = max(archs, key=lambda a: archs[a]["rule_score"])
        plot_confusions(list(archs.values()), args.fig_dir / "confusion_matrices.png")
        L += ["## 3. Siamese vs stack\n",
              "Same data, loss, optimiser, schedule, augmentation and seed (0); only the "
              "architecture differs (`src/model.py`). Each row is the epoch with the highest "
              "mean of val-tiles and val macro-F1.\n",
              df_to_markdown(comparison_rows(archs.values()), float_fmt="{:.3f}", index=False) + "\n"]
        for r in archs.values():
            if not r["saved_matches_rule"]:
                L.append(f"*`{r['run_name']}` was trained before the selection rule changed: its "
                         f"saved weights are from epoch {r['best_epoch']} (best val-tiles "
                         f"macro-F1), while the row above is epoch {r['rule_epoch']}, taken "
                         "from its per-epoch log. The comparison is therefore like for like, "
                         "but those epoch-{0} weights no longer exist.*\n".format(r["rule_epoch"]))
        L += [df_to_markdown(cost_rows(archs.values()), float_fmt="{:.1f}", index=False) + "\n"]
        if len(archs) > 1:
            other = [a for a in archs if a != better][0]
            b, o = archs[better], archs[other]
            L.append(f"**Better by the selection score: `{better}`** (mean macro-F1 "
                     f"{b['rule_score']:.3f} against {o['rule_score']:.3f}; val-tiles "
                     f"{b['rule']['val-tiles']['macro_f1']:.3f} vs "
                     f"{o['rule']['val-tiles']['macro_f1']:.3f}, val "
                     f"{b['rule']['val']['macro_f1']:.3f} vs {o['rule']['val']['macro_f1']:.3f}). "
                     "One seed per architecture: a difference of a point or two is within "
                     "what another seed could produce (see the ensemble members in section "
                     "6).\n")
        else:
            missing = [a for a in args.archs if a not in archs]
            L.append(f"**Not trained: {', '.join(f'`{a}`' for a in missing)}** (see "
                     "`BLOCKERS.md`).\n")
        L += versus_baseline(archs[better])
        L += [f"\n![confusion matrices]({fig_rel}/confusion_matrices.png)\n"]

        # 4. augmentation experiment
        plain = archs[better]
        aug = load_run(args.runs_dir / f"{better}_aug_s0")
        base = load_run(args.runs_dir / f"{args.base}_s0") if args.base else choose_base(plain, aug)
        if aug:
            d_val = aug["rule"]["val"]["macro_f1"] - plain["rule"]["val"]["macro_f1"]
            d_vt = aug["rule"]["val-tiles"]["macro_f1"] - plain["rule"]["val-tiles"]["macro_f1"]
            d_acc = aug["rule"]["val"]["accuracy"] - plain["rule"]["val"]["accuracy"]
            plot_confusions([plain, aug], args.fig_dir / "confusion_augmentation.png")
            L += ["## 4. Augmentation experiment for the val gap\n",
                  f"One extra run of `{better}` with the \"strong\" augmentation "
                  "(`augment_pair` in `src/dataset.py`): brightness, contrast and saturation "
                  "within +-30% and a +-10% colour cast, drawn independently for pre and "
                  "post, plus a random shift of the post image against the pre image by up "
                  "to 4 px each way (in the 128 px crop; about 2 px in the original tile for "
                  "a typical building). Everything else is unchanged. The original uses "
                  "+-10% brightness and contrast and no shift.\n",
                  df_to_markdown(comparison_rows([plain, aug]), float_fmt="{:.3f}", index=False) + "\n",
                  f"**Judged on val (the unseen disasters): macro-F1 {d_val:+.3f}, accuracy "
                  f"{d_acc:+.3f}; on val-tiles macro-F1 {d_vt:+.3f}.** "
                  + ("The stronger augmentation helps on val. "
                     if d_val > 0.01 else
                     "The stronger augmentation does not help on val. "
                     if d_val < -0.01 else
                     "On val macro-F1 the two are within a point of each other. ")
                  + "This is a single run per setting, so small differences are not "
                  "evidence either way.\n"]
            L += versus_baseline(aug)
            L += [f"\n![confusion matrices, original vs strong augmentation]({fig_rel}/confusion_augmentation.png)\n",
                  df_to_markdown(cost_rows([aug]), float_fmt="{:.1f}", index=False) + "\n"]
        L += [f"**Base model for the rest of this document: `{base['run_name'][:-3]}`**"
              + (" (chosen by hand with `--base`)." if args.base else
                 " (the one with the higher val macro-F1 of the two above)." if aug else ".")
              + "\n"]
        for r in dict.fromkeys([plain["run_name"], *( [aug["run_name"]] if aug else [])]):
            run_ = plain if r == plain["run_name"] else aug
            epochs = pd.DataFrame([{
                "epoch": h["epoch"], "train loss": h["train_loss"], "train acc": h["train_acc"],
                **{f"{p} {k}": h[p][v] for p in PARTS
                   for k, v in (("acc", "accuracy"), ("macro-F1", "macro_f1"))},
                "selection score": mean_score(h),
                "RAM free GB": h.get("memory", {}).get("ram_available_gb", np.nan),
            } for h in run_["history"]])
            L += [f"Per epoch, `{r}` (kept: epoch **{run_['rule_epoch']}**):\n",
                  df_to_markdown(epochs, float_fmt="{:.3f}", index=False) + "\n"]
        L.append("Per-class recall and memory for every epoch are in `runs/<run>/log.jsonl` "
                 "(not in git).\n")

    # 5-7 need the base model's seed-0 run
    if base:
        base_name = base["run_name"][:-3]
        if not base["saved_matches_rule"]:
            L.append(f"*Note: the saved weights of `{base['run_name']}` are from epoch "
                     f"{base['best_epoch']}, not the rule's epoch {base['rule_epoch']}; the "
                     "sections below use the saved weights.*\n")
        run = base
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
        L += [f"## 5. Calibration (`{run['run_name']}`)\n",
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

        # 6. uncertainty methods
        seeds = [load_run(args.runs_dir / f"{base_name}_s{s}") for s in args.seeds]
        seeds = [r for r in seeds if r]
        # All methods, built from the saved predictions. The ensemble gets its own
        # temperature: log(mean probability) is treated as its logits and one T is
        # fitted on val-tiles, exactly as for a single model.
        methods, info = collect_methods(run["dir"], [r["dir"] for r in seeds])
        ens_t = info["ensemble_temperature"]
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
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        for ax, part in zip(axes, PARTS):
            plot_risk_coverage(ax, methods[part], labels[part], f"Risk-coverage on {part}")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "risk_coverage.png", dpi=130)
        plt.close(fig)
        plot_reliability_methods(methods, labels, args.fig_dir / "reliability_methods.png")
        have = list(methods["val"])
        has_mc, has_ens = any("MC" in h for h in have), len(seeds) > 1
        mc_t = None
        mc_meta = run["dir"] / "mc_dropout.json"
        if mc_meta.exists():
            mc_t = json.loads(mc_meta.read_text(encoding="utf-8")).get("temperature")
        L += [f"## 6. Uncertainty methods (`{base_name}`)\n",
              "Each method gives every prediction a confidence (`src/uncertainty.py`):\n",
              "- **max prob**: the top softmax probability of the seed-0 model, raw and after "
              "temperature scaling;",
              "- **MC dropout**: 20 passes of the seed-0 model with dropout left on"
              + (f", each pass temperature-scaled with the same T ({mc_t:.3f})" if mc_t else "")
              + "; confidence = mean of the passes' top probabilities, or minus the entropy "
              "of the averaged probabilities" + ("" if has_mc else " *(not run)*") + ";",
              "- **ensemble**: probabilities averaged over models trained with different "
              "seeds " + (f"({', '.join(r['run_name'] for r in seeds)}), raw and with its own "
                          f"temperature (T = {ens_t:.3f}, fitted on val-tiles)" if has_ens else
                          f"*(not available: only {seeds[0]['run_name']} has been trained)*")
              + ".\n",
              ("" if has_mc and has_ens else
               "**Some methods are missing in this version; see `BLOCKERS.md`.**\n"),
              "ECE: lower = confidence closer to accuracy. AURC (area under the "
              "risk-coverage curve): lower = mistakes are ranked lower, and/or fewer "
              "mistakes. For the entropy row, ECE is computed on the top averaged "
              "probability, because minus-entropy is not a probability. MC dropout and the "
              "ensemble also change the predictions, so their accuracy can differ from the "
              "single model's.\n",
              df_to_markdown(pd.DataFrame(rows), float_fmt="{:.4f}", index=False) + "\n",
              f"![risk-coverage curves]({fig_rel}/risk_coverage.png)\n",
              f"![reliability per method]({fig_rel}/reliability_methods.png)\n"]
        if has_ens:
            per_seed = [{"run": r["run_name"], "epoch kept": r["best_epoch"],
                         **{f"{p} {k}": r["best"][p][v] for p in PARTS
                            for k, v in (("accuracy", "accuracy"), ("macro-F1", "macro_f1"))}}
                        for r in seeds]
            L += ["The individual ensemble members, to show how much a seed alone changes:\n",
                  df_to_markdown(pd.DataFrame(per_seed), float_fmt="{:.3f}", index=False) + "\n"]

        # 7. triage
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

        # Inspection budget, for every method that yields probabilities. The entropy
        # variant shares MC dropout's probabilities, so it only differs in "least
        # confident first".
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        budget_rows = []
        for ax, part in zip(axes, PARTS):
            damaged = labels[part] > 0
            curves = {}
            shown = [n for n in methods[part] if n != "max prob (raw)"]
            for k, name in enumerate(shown):
                m = methods[part][name]
                color = METHOD_COLORS[1 + k % (len(METHOD_COLORS) - 1)]
                curves[f"least confident first: {name}"] = (-m["confidence"], color, "--")
                budget_rows.append({"split": part, "method": name, "policy": "least confident first",
                                    **{f"K={b:.0%}": r for b, r in
                                       zip(BUDGETS, inspection_recall(-m["confidence"], damaged))
                                       if round(b * 100) % 10 == 0}})
                if name != "MC dropout (entropy)":
                    p_damaged = 1 - m["probs"][:, 0]
                    curves[f"most likely damaged first: {name}"] = (p_damaged, color, "-")
                    budget_rows.append({"split": part, "method": name,
                                        "policy": "most likely damaged first",
                                        **{f"K={b:.0%}": r for b, r in
                                           zip(BUDGETS, inspection_recall(p_damaged, damaged))
                                           if round(b * 100) % 10 == 0}})
            plot_inspection(ax, curves, damaged,
                            f"Inspection budget on {part} ({damaged.mean():.0%} of buildings damaged)")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "inspection_budget.png", dpi=130)
        plt.close(fig)
        budget = pd.DataFrame(budget_rows).sort_values(
            ["split", "policy"], key=lambda c: c.map({"val-tiles": 0, "val": 1}).fillna(0)
            if c.name == "split" else c, kind="stable")
        L += ["## 7. Triage\n",
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
              "visited. 'Most likely damaged' ranks by 1 - P(no-damage) under each "
              "method's probabilities; 'least confident' ranks by each method's "
              "confidence. Random visits reach K%.\n",
              df_to_markdown(budget, float_fmt="{:.1%}", index=False) + "\n",
              f"![inspection budget]({fig_rel}/inspection_budget.png)\n",
              "In the chart, solid lines are 'most likely damaged first' and dashed lines "
              "'least confident first'; colour is the method. The two policies answer "
              "different questions: one looks for damage, the other for the model's "
              "likely mistakes.\n"]

    # 8. all runs
    all_runs = [load_run(d) for d in sorted(args.runs_dir.iterdir()) if d.is_dir()]
    all_runs = [r for r in all_runs if r]
    if all_runs:
        rows = [{"run": r["run_name"], "augmentation": r.get("augment", "base"), "seed": r["seed"],
                 "epoch kept": r["rule_epoch"] if r["saved_matches_rule"]
                 else f"{r['rule_epoch']} (weights saved: {r['best_epoch']})",
                 "val-tiles acc": r["rule"]["val-tiles"]["accuracy"],
                 "val-tiles macro-F1": r["rule"]["val-tiles"]["macro_f1"],
                 "val acc": r["rule"]["val"]["accuracy"],
                 "val macro-F1": r["rule"]["val"]["macro_f1"],
                 "selection score": r["rule_score"]} for r in all_runs]
        L += ["## 8. All runs\n",
              "Every finished run, at the epoch the selection rule picks. Selection score = "
              "mean of the val-tiles and val macro-F1.\n",
              df_to_markdown(pd.DataFrame(rows), float_fmt="{:.3f}", index=False) + "\n",
              "Time and memory per run:\n",
              df_to_markdown(cost_rows(all_runs), float_fmt="{:.1f}", index=False) + "\n",
              "'RAM of this run' is the memory owned by the training process and its "
              "loader workers; the first run (8 + 8 + 8 workers) did not log it and used "
              "about 59 GB of the machine's 62 (`BLOCKERS.md`).\n"]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}" + (f" (base model: {base['run_name'][:-3]})" if base else ""))


if __name__ == "__main__":
    main()
