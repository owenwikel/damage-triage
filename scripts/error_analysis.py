"""Where is the model wrong, and does its confidence know? -> docs/error_analysis.md

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\error_analysis.py --run runs\\siamese_s0

Uses the saved predictions of one run on val-tiles and val (never test), with confidence
= top probability after temperature scaling (T fitted on val-tiles, src/calibrate.py).

Sections
    1. metrics per disaster
    2. metrics per building-size bucket (longest bounding-box side)
    3. buildings cut by the tile edge vs the rest
    4. label-noise candidates (method of docs/label_noise.md): are they lower-confidence?
    5. confidence vs distance to the nearest building with a different label
       (hypothesis 2 in docs/observations.md: uncertainty is highest at damage boundaries)
    6. gallery: most confident mistakes, least confident correct predictions
    7. sections 4 and 5 again for every uncertainty method (MC dropout, ensemble), when
       the base run has them
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import calibrated_probs, fit_temperature  # noqa: E402
from src.colors import DAMAGE_COLORS, INK, INK_SECONDARY, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES, read_index  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402
from src.uncertainty import collect_methods  # noqa: E402

PARTS = ["val-tiles", "val"]
SHORT = ["no-dmg", "minor", "major", "destr."]
SIZE_BINS = [0, 20, 40, 80, np.inf]
SIZE_LABELS = ["< 20 px", "20-40 px", "40-80 px", "> 80 px"]
DIST_BINS = [0, 25, 50, 100, 200, np.inf]
DIST_LABELS = ["< 25 px", "25-50 px", "50-100 px", "100-200 px", "> 200 px"]
NO_OTHER = "no other label in tile"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_predictions(run_dir, cache_dir):
    """One row per val-tiles/val building: index columns + pred, confidence, correct."""
    index = read_index(cache_dir)
    fit = np.load(run_dir / "preds_val-tiles.npz")
    temperature = fit_temperature(fit["logits"], fit["labels"])
    frames = []
    for part in PARTS:
        d = np.load(run_dir / f"preds_{part}.npz")
        sub = index[index["part"] == part].sort_values("row").reset_index(drop=True)
        assert (sub["label"].to_numpy() == d["labels"]).all(), "predictions out of step with index"
        p = calibrated_probs(d["logits"], temperature)
        sub["pred"] = p.argmax(axis=1)
        sub["confidence"] = p.max(axis=1)
        sub["correct"] = sub["pred"] == sub["label"]
        frames.append(sub)
    return pd.concat(frames, ignore_index=True), temperature


def group_table(df, by, order=None):
    """n, accuracy, macro-F1, mean confidence and per-class recall for each group."""
    rows = []
    for (part, key), g in df.groupby(["part", by], observed=True, sort=False):
        s = summarize(g["label"].to_numpy(), g["pred"].to_numpy())
        rows.append({"split": part, by: key, "buildings": len(g), "accuracy": s["accuracy"],
                     "macro-F1": s["macro_f1"], "mean confidence": g["confidence"].mean(),
                     **{f"recall {n}": (np.nan if r is None else r)
                        for n, r in zip(SHORT, s["recall"])}})
    t = pd.DataFrame(rows)
    t["split"] = pd.Categorical(t["split"], PARTS)
    if order is not None:
        t[by] = pd.Categorical(t[by], order)
    return t.sort_values(["split", by]).reset_index(drop=True)


def noise_candidates(df, k=5):
    """docs/label_noise.md's rule: a building whose label differs from the label of every
    one of its k nearest neighbours (pixel centroid, same tile). Tiles with fewer than
    k+1 buildings are not eligible. Returns (is_candidate, is_eligible, neighbour majority).
    """
    cand = np.zeros(len(df), dtype=bool)
    eligible = np.zeros(len(df), dtype=bool)
    majority = np.full(len(df), -1)
    for _, t in df.groupby(["disaster", "tile_id"], sort=False):
        if len(t) < k + 1:
            continue
        xy = t[["centroid_x", "centroid_y"]].to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :k]
        lab = t["label"].to_numpy()
        pos = t.index.to_numpy()
        eligible[pos] = True
        cand[pos] = (lab[nn] != lab[:, None]).all(axis=1)
        majority[pos] = [np.bincount(row, minlength=4).argmax() for row in lab[nn]]
    return cand, eligible, majority


def distance_to_other_label(df):
    """Pixel distance from each building to the nearest building in the same tile whose
    label is different. inf if every building in the tile has the same label."""
    out = np.full(len(df), np.inf)
    for _, t in df.groupby(["disaster", "tile_id"], sort=False):
        lab = t["label"].to_numpy()
        if len(np.unique(lab)) < 2:
            continue
        xy = t[["centroid_x", "centroid_y"]].to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        d[lab[:, None] == lab[None, :]] = np.inf       # ignore same-label buildings
        out[t.index.to_numpy()] = d.min(axis=1)
    return out


def plot_boundary(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, part in zip(axes, PARTS):
        t = table[table["split"] == part]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["mean confidence"], width=0.4, color="#2a78d6", label="mean confidence")
        ax.bar(x + 0.2, t["accuracy"], width=0.4, color="#eda100", label="accuracy")
        for xi, n in zip(x, t["buildings"]):
            ax.text(xi, 1.01, f"n={n:,}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        ax.set_xticks(x, t["distance"], rotation=20, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{part}: by distance to the nearest building with a different label", fontsize=10)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_gallery(rows, cache_dir, title, path):
    """Contact sheet: for each building, pre crop | post crop, with true/predicted labels."""
    arrays = {p: (np.load(Path(cache_dir) / f"{p}_pre.npy", mmap_mode="r"),
                  np.load(Path(cache_dir) / f"{p}_post.npy", mmap_mode="r")) for p in PARTS}
    cols = 5
    n_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.3 * cols, 2.25 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows.itertuples()):
        pre, post = arrays[r.part]
        ax.imshow(np.concatenate([pre[r.row], np.full((128, 4, 3), 255, np.uint8), post[r.row]], axis=1))
        ax.set_title(f"true: {CLASS_NAMES[r.label]}", fontsize=8, loc="left", pad=14,
                     color=DAMAGE_COLORS[CLASS_NAMES[r.label]], fontweight="bold")
        ax.text(0, 1.02, f"predicted: {CLASS_NAMES[r.pred]} ({r.confidence:.0%})   "
                f"{r.disaster.split('-')[-1]} {r.tile_id[-4:]}, {r.part}",
                transform=ax.transAxes, fontsize=6.8, color=INK, va="bottom")
    fig.suptitle(title + "   (left: before, right: after)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=Path(r"C:\Users\wikel\dev\xbd-data\crops"))
    ap.add_argument("--out", type=Path, default=Path("docs/error_analysis.md"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--gallery", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="ensemble members: sibling runs <name>_s<seed> of --run")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = args.fig_dir.relative_to(args.out.parent).as_posix()

    df, temperature = load_predictions(args.run, args.cache_dir)
    L = ["# Error analysis (validation only)\n",
         f"Generated by `scripts/error_analysis.py` from `{args.run.as_posix()}` "
         f"({len(df):,} buildings on val-tiles and val; the test set is not touched). "
         f"Confidence = top class probability after temperature scaling (T = {temperature:.3f}). "
         "Overall numbers are in `docs/results.md`.\n",
         "Reading the tables: a group with few buildings of some class has a noisy recall "
         "for it, and macro-F1 averages only over classes present in the group, so compare "
         "groups with care. Blank recall = no buildings of that class in the group.\n"]

    # 1. per disaster
    L += ["## 1. Per disaster\n",
          df_to_markdown(group_table(df, "disaster"), float_fmt="{:.3f}", index=False) + "\n"]

    # 2. size buckets
    df["size"] = pd.cut(df["box_side"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    L += ["## 2. By building size\n",
          "Longest side of the bounding box, in pixels of the original tile (every crop is "
          "resized to 128 px, so a small building is magnified and blurry).\n",
          df_to_markdown(group_table(df, "size", SIZE_LABELS), float_fmt="{:.3f}", index=False) + "\n"]

    # 3. edge
    df["position"] = np.where(df["at_edge"], "cut by tile edge", "inside tile")
    L += ["## 3. Cut by the tile edge vs not\n",
          "A building counts as cut if its bounding box is within 1 px of the tile border; "
          "part of its crop is then black padding.\n",
          df_to_markdown(group_table(df, "position"), float_fmt="{:.3f}", index=False) + "\n"]

    # 4. label-noise candidates
    cand, eligible, majority = noise_candidates(df)
    df["noise"] = np.where(cand, "candidate", "other eligible")
    e = df[eligible].copy()
    e["agrees_with_neighbours"] = e["pred"].to_numpy() == majority[eligible]
    rows = []
    for (part, kind), g in e.groupby(["part", "noise"], sort=False):
        rows.append({"split": part, "group": kind, "buildings": len(g),
                     "mean confidence": g["confidence"].mean(),
                     "median confidence": g["confidence"].median(),
                     "model agrees with the label": g["correct"].mean(),
                     "model agrees with neighbours' majority": g["agrees_with_neighbours"].mean()})
    noise = pd.DataFrame(rows)
    noise["split"] = pd.Categorical(noise["split"], PARTS)
    noise = noise.sort_values(["split", "group"]).reset_index(drop=True)
    L += ["## 4. Label-noise candidates\n",
          "Same rule as `docs/label_noise.md`, applied here to val-tiles and val only: a "
          "building is a *candidate* if its label differs from the label of all 5 of its "
          "nearest neighbours in the tile. 'Other eligible' = the remaining buildings in "
          "tiles with at least 6 classified buildings. A candidate is not necessarily "
          "mislabelled: one burnt house on an intact street is real.\n",
          df_to_markdown(noise, float_fmt="{:.3f}", index=False) + "\n",
          "How to read it: if candidates were mostly real damage, the model should agree "
          "with their labels about as often as elsewhere. If they were mostly label "
          "mistakes, the model should side with the neighbours. Neither can be told apart "
          "from 'the model just follows the neighbourhood', because the crop includes "
          "context around the building.\n"]

    # 5. distance to a different label
    dist = distance_to_other_label(df)
    df["distance"] = pd.cut(dist, DIST_BINS, labels=DIST_LABELS, right=False).astype(object)
    df.loc[np.isinf(dist), "distance"] = NO_OTHER
    boundary = group_table(df, "distance", DIST_LABELS + [NO_OTHER])
    plot_boundary(boundary, args.fig_dir / "confidence_vs_boundary.png")
    corr = {}
    for p in PARTS:
        keep = ((df["part"] == p) & np.isfinite(dist)).to_numpy()
        # Spearman = ordinary (Pearson) correlation of the ranks; done by hand to avoid
        # needing scipy.
        a = pd.Series(dist[keep]).rank()
        b = pd.Series(df["confidence"].to_numpy()[keep]).rank()
        corr[p] = float(a.corr(b)) if keep.sum() > 2 else float("nan")
    L += ["## 5. Is uncertainty higher near damage boundaries?\n",
          "Hypothesis 2 in `docs/observations.md`. For each building: distance (pixel "
          "centroids) to the nearest building *in the same tile* with a different label. "
          "Small distance = on a damage boundary.\n",
          df_to_markdown(boundary[["split", "distance", "buildings", "accuracy",
                                   "mean confidence", "macro-F1"]],
                         float_fmt="{:.3f}", index=False) + "\n",
          f"![confidence vs distance to a different label]({fig_rel}/confidence_vs_boundary.png)\n",
          "Spearman rank correlation between distance and confidence (buildings with a "
          "different label somewhere in their tile): "
          + ", ".join(f"{p} {c:+.3f}" for p, c in corr.items())
          + ". Positive = confidence rises with distance from the boundary.\n",
          "Caveats: the distance only sees neighbours inside the same tile, so a boundary "
          "just across the tile border is missed; and 'near a different label' is also "
          "where a wrong label is most likely, so lower accuracy there mixes model errors "
          "with label noise.\n"]

    # 6. gallery
    gallery_dir = args.run / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    wrong = df[~df["correct"]].sort_values("confidence", ascending=False).head(args.gallery)
    unsure = df[df["correct"]].sort_values("confidence").head(args.gallery)
    draw_gallery(wrong, args.cache_dir, f"The {len(wrong)} most confident mistakes",
                 gallery_dir / "confident_mistakes.png")
    draw_gallery(unsure, args.cache_dir, f"The {len(unsure)} least confident correct predictions",
                 gallery_dir / "unsure_correct.png")
    cols = ["part", "disaster", "tile_id", "uid", "damage", "pred", "confidence", "box_side", "at_edge"]
    wrong[cols].to_csv(gallery_dir / "confident_mistakes.csv", index=False)
    unsure[cols].to_csv(gallery_dir / "unsure_correct.csv", index=False)

    def tally(g):
        pairs = (g["label"].map(lambda i: SHORT[i]) + " -> " + g["pred"].map(lambda i: SHORT[i]))
        return ", ".join(f"{k} ({v})" for k, v in pairs.value_counts().items())

    L += ["## 6. Gallery\n",
          f"Written to `{gallery_dir.as_posix()}/` (not in git): `confident_mistakes.png` "
          f"and `unsure_correct.png`, each with pre and post crops, plus a CSV of the same "
          "buildings.\n",
          f"- The {len(wrong)} most confident mistakes (confidence "
          f"{wrong['confidence'].min():.3f}-{wrong['confidence'].max():.3f}), as true -> "
          f"predicted: {tally(wrong)}. By disaster: "
          + ", ".join(f"{k} ({v})" for k, v in wrong["disaster"].value_counts().items()) + ".",
          f"- The {len(unsure)} least confident correct predictions (confidence "
          f"{unsure['confidence'].min():.3f}-{unsure['confidence'].max():.3f}): classes "
          + ", ".join(f"{SHORT[k]} ({v})" for k, v in unsure["label"].value_counts().items()) + ".\n",
          "Confident mistakes are the ones to look at by eye: each is either a model "
          "failure worth understanding or a label worth questioning.\n"]

    # 7. the same questions for every uncertainty method
    stem = args.run.name.rsplit("_s", 1)[0]
    methods, info = collect_methods(args.run, [args.run.parent / f"{stem}_s{s}" for s in args.seeds])
    rows = []
    for part in PARTS:
        sel = (df["part"] == part).to_numpy()
        sub = df[sel]
        d, c, e = dist[sel], cand[sel], eligible[sel]
        for name, m in methods[part].items():
            conf = np.asarray(m["confidence"], dtype=float)
            right = m["pred"] == sub["label"].to_numpy()
            fin = np.isfinite(d)
            rho = float(pd.Series(d[fin]).rank().corr(pd.Series(conf[fin]).rank())) if fin.sum() > 2 else np.nan
            # AUROC of "is this prediction right?": the chance that a randomly chosen
            # correct prediction gets a higher confidence than a randomly chosen wrong
            # one. 0.5 = the confidence says nothing; 1.0 = it separates them perfectly.
            ranks = pd.Series(conf).rank().to_numpy()
            n_pos, n_neg = right.sum(), (~right).sum()
            auroc = ((ranks[right].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                     if n_pos and n_neg else np.nan)
            # share of candidates among the least-confident 10% vs their share overall
            low = conf <= np.quantile(conf[e], 0.10) if e.any() else np.zeros_like(c)
            rows.append({"split": part, "method": name, "accuracy": right.mean(),
                         "AUROC: confidence separates right from wrong": auroc,
                         "rank corr.: distance to boundary vs confidence": rho,
                         "noise candidates: share in least-confident 10%":
                             (c & low & e).sum() / max(c.sum(), 1)})
    L += ["## 7. The same questions for every uncertainty method\n",
          "Sections 4 and 5 used the temperature-scaled top probability of the seed-0 "
          "model. Here every available method is asked the same things"
          + (f" (ensemble members: {', '.join(info['members'])})" if len(info["members"]) > 1 else "")
          + ":\n",
          "- **AUROC**: probability that a correct prediction gets a higher confidence than "
          "a wrong one (0.5 = useless, 1 = perfect);",
          "- **rank correlation** between distance to the nearest differently-labelled "
          "building and confidence (section 5; positive = less sure near boundaries);",
          "- **noise candidates in the least-confident 10%**: if confidence were unrelated "
          "to the label-noise candidates of section 4, this would be 10%.\n",
          df_to_markdown(pd.DataFrame(rows), float_fmt="{:.3f}", index=False) + "\n",
          "For entropy the 'confidence' is minus the predictive entropy, so only its "
          "ranking is meaningful.\n"]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out} and {gallery_dir}")


if __name__ == "__main__":
    main()
