"""Where is the model wrong, and does its confidence know? -> docs/error_analysis.md

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\error_analysis.py --run runs\\stack_aug_s0

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
    8. confusion: for every part and disaster, where the buildings of each true class go
    9. missed damage (truly damaged, predicted no-damage): by disaster, class and size
   10. example val tiles with the most missed damage -> docs/figures/ (needs --data-dir)
   11. are the most confident "errors" on val label noise? Uses the candidates of
       outputs/label_noise/candidates.csv. No label is changed anywhere.

Full run, with the tile figures:
    .\\.venv\\Scripts\\python.exe scripts\\error_analysis.py --run runs\\stack_aug_s0 --data-dir C:\\Users\\wikel\\dev\\xbd-data
"""

import argparse
import re
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


# ---- sections 8-11: confusion, missed damage, example tiles, label noise -------------------

def confusion_table(df):
    """Long-format confusion matrices: one row per (group of buildings, true class).

    Groups: each part as a whole, then every disaster. The four "-> class" columns are
    COUNTS of buildings of that true class given each predicted class; "found" is the
    share on the diagonal (the class's recall).
    """
    groups = [(f"{part} (all)", df[df["part"] == part]) for part in PARTS]
    for part in PARTS:
        sub = df[df["part"] == part]
        groups += [(f"{part}: {d}", g) for d, g in sub.groupby("disaster")]
    rows = []
    for name, g in groups:
        cm = summarize(g["label"].to_numpy(), g["pred"].to_numpy())["confusion"]
        for c, class_name in enumerate(CLASS_NAMES):
            n = sum(cm[c])
            if n == 0:                     # this class does not occur in the group
                continue
            rows.append({"buildings": name, "true class": class_name, "n": n,
                         **{f"-> {SHORT[j]}": cm[c][j] for j in range(len(SHORT))},
                         "found": cm[c][c] / n})
    return pd.DataFrame(rows)


def missed_table(df, by, order=None):
    """Missed damage per group: of the truly damaged buildings (label 1-3) in the group,
    how many were predicted no-damage. One table per call, both parts in it."""
    damaged = df[df["label"] > 0]
    rows = []
    for (part, key), g in damaged.groupby(["part", by], observed=True, sort=False):
        missed = int((g["pred"] == 0).sum())
        all_missed = int((damaged.loc[damaged["part"] == part, "pred"] == 0).sum())
        rows.append({"split": part, by: key, "damaged buildings": len(g), "missed": missed,
                     "miss rate": missed / len(g),
                     "share of the split's misses": missed / max(all_missed, 1),
                     "mean confidence of the misses":
                         g.loc[g["pred"] == 0, "confidence"].mean() if missed else np.nan})
    t = pd.DataFrame(rows)
    t["split"] = pd.Categorical(t["split"], PARTS)
    if order is not None:
        t[by] = pd.Categorical(t[by], order)
    return t.sort_values(["split", by]).reset_index(drop=True)


def draw_missed_tile(tile, data_dir, path):
    """One val tile: before | after, with the missed buildings picked out.

    Three kinds of outline on the "after" panel:
      thick, in the colour of the TRUE class   damaged, but the model said no-damage
      thin black                               damaged, and the model did report damage
      thin white                               truly undamaged
    The images and outlines come from the tile's own files; only val and val-tiles tiles
    can be looked up (load_split refuses the test part).
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image

    from src.labels import parse_label_file
    from src.split import find_folder, load_split

    first = tile.iloc[0]
    disaster, tile_id = first["disaster"], first["tile_id"]
    allowed = load_split()                       # development parts only
    assert ((allowed["disaster"] == disaster) & (allowed["tile_id"] == tile_id)).any()
    folder = find_folder(data_dir, disaster, tile_id)
    stem = f"{disaster}_{tile_id}"
    polys = {r["uid"]: r["polygon"] for r in parse_label_file(
        folder / "labels" / f"{stem}_post_disaster.json", folder / "images")}
    pre = np.asarray(Image.open(folder / "images" / f"{stem}_pre_disaster.png").convert("RGB"))
    post = np.asarray(Image.open(folder / "images" / f"{stem}_post_disaster.png").convert("RGB"))

    def patches(rows):
        return [MplPolygon(np.asarray(polys[u].exterior.coords), closed=True) for u in rows["uid"]]

    damaged = tile["label"] > 0
    missed = tile[damaged & (tile["pred"] == 0)]
    caught = tile[damaged & (tile["pred"] > 0)]
    fine = tile[~damaged]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), dpi=100)
    axes[0].imshow(pre)
    axes[0].set_title("Before", loc="left", fontsize=12, color=INK)
    ax = axes[1]
    ax.imshow(post)
    ax.set_title("After: damaged buildings the model called no-damage", loc="left",
                 fontsize=12, color=INK)
    if len(fine):
        ax.add_collection(PatchCollection(patches(fine), facecolor="none", edgecolor="white",
                                          linewidth=0.9, alpha=0.9))
    if len(caught):
        ax.add_collection(PatchCollection(patches(caught), facecolor="none", edgecolor="white",
                                          linewidth=2.4))
        ax.add_collection(PatchCollection(patches(caught), facecolor="none", edgecolor="#0b0b0b",
                                          linewidth=1.1))
    if len(missed):
        colors = [DAMAGE_COLORS[CLASS_NAMES[i]] for i in missed["label"]]
        ax.add_collection(PatchCollection(patches(missed), facecolor="none", edgecolor="white",
                                          linewidth=4.2))
        ax.add_collection(PatchCollection(patches(missed), facecolor=colors, edgecolor=colors,
                                          linewidth=2.4, alpha=0.55))
        ax.add_collection(PatchCollection(patches(missed), facecolor="none", edgecolor=colors,
                                          linewidth=2.4))
    for a in axes:
        a.set_xlim(0, post.shape[1])
        a.set_ylim(post.shape[0], 0)
        a.axis("off")
    handles = [Line2D([0], [0], color=DAMAGE_COLORS[c], linewidth=4, label=f"missed, truly {c}")
               for c in CLASS_NAMES[1:]]
    handles += [Line2D([0], [0], color="#0b0b0b", linewidth=1.4, label="damaged, reported as damaged"),
                Line2D([0], [0], color="#898781", linewidth=1.0, label="truly undamaged (white outline)")]
    legend = fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False, fontsize=9,
                        labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.005))
    legend.set_in_layout(False)
    fig.suptitle(f"{disaster}, tile {tile_id}   |   unseen disaster (val)   |   model: "
                 f"predictions of the base model\n{int(damaged.sum())} damaged buildings of "
                 f"{len(tile)}: {len(missed)} missed (called no-damage), {len(caught)} reported "
                 "as damaged", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 88})
    plt.close(fig)
    return {"disaster": disaster, "tile": tile_id, "buildings": len(tile),
            "damaged": int(damaged.sum()), "missed": len(missed), "file": Path(path).name}


def load_val_noise_candidates(csv_path, val):
    """Label-noise candidates (docs/label_noise.md) that are val buildings.

    Returns a DataFrame indexed by uid with the column ``neighbour_majority`` (the most
    common class among the building's 5 nearest neighbours), or None without the file.

    CAREFUL: the candidates file was written during the first data exploration, before
    the split existed, and it also lists buildings of the two TEST disasters. Those rows
    are thrown away while the file is being read: only rows of the val disasters are
    kept, and of those only buildings that are in val. Nothing about a test building is
    used, shown or stored.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    val_disasters = set(val["disaster"])
    kept = []
    # Read in pieces and filter each piece at once, so that the unwanted rows never sit
    # in a table that later code could reach.
    for chunk in pd.read_csv(csv_path, usecols=["disaster", "uid", "neighbour_majority"], chunksize=500):
        kept.append(chunk[chunk["disaster"].isin(val_disasters)])
    cand = pd.concat(kept, ignore_index=True)
    cand = cand[cand["uid"].isin(set(val["uid"]))]
    return cand.set_index("uid")[["neighbour_majority"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="xBD folder; needed only for the example tiles of section 10")
    ap.add_argument("--noise-csv", type=Path, default=Path("outputs/label_noise/candidates.csv"),
                    help="label-noise candidates for section 11 (skipped if the file is missing)")
    ap.add_argument("--tiles", type=int, default=6, help="example tiles in section 10")
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

    # 8. confusion per part and per disaster
    L += ["## 8. Confusion per part, per disaster and per class\n",
          "One row per true class: how many of its buildings the model put in each class "
          "(counts), and the share it got right ('found'). Classes that do not occur in a "
          "group are left out.\n",
          df_to_markdown(confusion_table(df), float_fmt="{:.3f}", index=False) + "\n"]

    # 9. missed damage
    df["true class"] = df["label"].map(lambda i: CLASS_NAMES[i])
    L += ["## 9. Missed damage: which buildings?\n",
          "A *miss* is a truly damaged building (minor, major or destroyed) that the model "
          "calls no-damage. miss rate = misses / damaged buildings in the group. 'Share of "
          "the split's misses' says where the bulk of the misses is; a group can have a high "
          "miss rate and still hold few of them.\n",
          "By disaster:\n",
          df_to_markdown(missed_table(df, "disaster"), float_fmt="{:.3f}", index=False) + "\n",
          "By true class:\n",
          df_to_markdown(missed_table(df, "true class", CLASS_NAMES[1:]), float_fmt="{:.3f}",
                         index=False) + "\n",
          "By building size (longest side of the bounding box, tile pixels):\n",
          df_to_markdown(missed_table(df, "size", SIZE_LABELS), float_fmt="{:.3f}", index=False) + "\n",
          "Cut by the tile edge or not:\n",
          df_to_markdown(missed_table(df, "position"), float_fmt="{:.3f}", index=False) + "\n"]
    val = df[df["part"] == "val"]
    val_damaged = val[val["label"] > 0].copy()
    val_damaged["group"] = val_damaged["disaster"] + ", " + val_damaged["true class"]
    by_both = (val_damaged.assign(missed=val_damaged["pred"] == 0)
               .groupby("group")["missed"].agg(["size", "sum", "mean"]).reset_index())
    by_both.columns = ["val: disaster, true class", "damaged buildings", "missed", "miss rate"]
    L += ["val, disaster and class together:\n",
          df_to_markdown(by_both, float_fmt="{:.3f}", index=False) + "\n"]

    # 10. example tiles
    if args.data_dir is not None:
        per_tile = (val.assign(missed=(val["label"] > 0) & (val["pred"] == 0))
                    .groupby(["disaster", "tile_id"])
                    .agg(missed=("missed", "sum"), n=("label", "size")).reset_index())
        # Half of the tiles from each val disaster, most misses first, so that one
        # disaster does not fill the page. Tiles with a readable number of buildings.
        per_tile = per_tile[per_tile["n"] <= 200].sort_values(
            ["missed", "n"], ascending=[False, True], kind="stable")
        share = max(1, args.tiles // max(per_tile["disaster"].nunique(), 1))
        picks = per_tile.groupby("disaster", sort=True).head(share).head(args.tiles)
        infos = []
        for r in picks.itertuples():
            tile = val[(val["disaster"] == r.disaster) & (val["tile_id"] == r.tile_id)]
            path = args.fig_dir / f"missed_damage_val_{r.disaster}_{r.tile_id}.jpg"
            infos.append(draw_missed_tile(tile, args.data_dir, path))
            print(f"wrote {path}")
        L += ["## 10. Example tiles of missed damage (val)\n",
              f"The {len(infos)} val tiles with the most missed buildings, {share} per "
              "disaster (tiles with at most 200 buildings). **They are the worst cases, picked "
              "because the model failed on them, not typical tiles.** All are from val, the "
              "unseen disasters; every figure says so in its title.\n"]
        for i in infos:
            L += [f"**{i['disaster']}, tile {i['tile']} (val)**: {i['damaged']} damaged buildings "
                  f"of {i['buildings']}, {i['missed']} missed.\n",
                  f"![missed damage, {i['disaster']} {i['tile']}, val]({fig_rel}/{i['file']})\n"]

    # 11. are the most confident "errors" on val label noise?
    noise_val = load_val_noise_candidates(args.noise_csv, val)
    if noise_val is not None:
        errors = val[~val["correct"]].sort_values("confidence", ascending=False).copy()
        errors["candidate"] = errors["uid"].isin(noise_val.index)
        errors["neighbours"] = errors["uid"].map(noise_val["neighbour_majority"])
        # "Looks like label noise" here means two independent hints agree: the label
        # disagrees with all 5 neighbours, AND the model predicts what the neighbours are.
        errors["sides_with_neighbours"] = errors["candidate"] & (
            errors["pred"].map(lambda i: CLASS_NAMES[i]) == errors["neighbours"])
        n_val, n_cand = len(val), int(val["uid"].isin(noise_val.index).sum())
        rows = []
        for top in (50, 100, 200, 500, 1000, len(errors)):
            head = errors.head(top)
            rows.append({"errors looked at": ("all " if top == len(errors) else "top ") + f"{len(head):,}",
                         "lowest confidence among them": head["confidence"].min(),
                         "label-noise candidates": int(head["candidate"].sum()),
                         "share": head["candidate"].mean(),
                         "of those, model predicts the neighbours' class":
                             int(head["sides_with_neighbours"].sum())})
        L += ["## 11. Are the most confident 'errors' on val label noise?\n",
              f"An 'error' is a val building whose prediction differs from its label; "
              f"{len(errors):,} of {n_val:,}. They are sorted by the model's confidence, highest "
              "first. A *label-noise candidate* (`docs/label_noise.md`, "
              f"`{args.noise_csv.as_posix()}`) is a building whose label differs from the label "
              f"of all 5 nearest neighbours in its tile; {n_cand:,} val buildings are candidates "
              f"({n_cand / n_val:.1%} of val), so that is the share to expect if confident "
              "errors had nothing to do with label noise. **No label was changed.** The "
              "candidates file also contains test-disaster rows from the first data "
              "exploration; they are discarded on reading and play no part here.\n",
              df_to_markdown(pd.DataFrame(rows), float_fmt="{:.3f}", index=False) + "\n"]
        examples = errors[errors["candidate"]].head(6)
        if len(examples):
            path = args.fig_dir / "label_noise_val_examples.png"
            draw_gallery(examples, args.cache_dir,
                         f"val: the {len(examples)} most confident 'errors' that are label-noise "
                         "candidates", path)
            ex = examples[["disaster", "tile_id", "uid"]].copy()
            ex["label"] = examples["label"].map(lambda i: CLASS_NAMES[i])
            ex["predicted"] = examples["pred"].map(lambda i: CLASS_NAMES[i])
            ex["confidence"] = examples["confidence"]
            ex["neighbours' most common class"] = examples["neighbours"]
            L += [f"The {len(examples)} most confident errors that are candidates (all val):\n",
                  df_to_markdown(ex, float_fmt="{:.3f}", index=False) + "\n",
                  f"![label-noise examples on val]({fig_rel}/{path.name})\n"]

    # A hand-written reading may sit in the document between two marker comments. It is
    # carried over to the new version, right after the introduction, so that running
    # this script again does not wipe it. (Whoever regenerates the tables must still
    # check that the reading matches them.)
    if args.out.exists():
        kept = re.search(r"<!-- reading:start -->.*?<!-- reading:end -->",
                         args.out.read_text(encoding="utf-8"), re.DOTALL)
        if kept:
            L.insert(3, kept.group(0) + "\n")
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out} and {gallery_dir}")


if __name__ == "__main__":
    main()
