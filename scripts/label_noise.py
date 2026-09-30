"""Find buildings whose damage class disagrees with all of their nearest neighbours.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\label_noise.py --data-dir C:\\Users\\wikel\\dev\\xbd-data

A building is a *candidate* if its damage class differs from the class of every one of its
k (default 5) nearest neighbours in the same tile, by pixel centroid. Damage is spatially
clustered (docs/observations.md), so an isolated disagreement is worth a look, but it can
be perfectly real (one burnt house on an intact street). This script only lists and shows
candidates; it does not relabel or remove anything.

Only the four real classes take part: un-classified buildings are neither candidates nor
neighbours (they would always count as "different"). Tiles with fewer than k+1 classified
buildings are skipped.

Writes docs/label_noise.md, outputs/label_noise/candidates.csv and gallery PNGs.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.colors import DAMAGE_COLORS, INK, INK_SECONDARY, SURFACE  # noqa: E402
from src.labels import DAMAGE_CLASSES, load_labels  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402

REAL = DAMAGE_CLASSES[:4]
SHORT = {"no-damage": "no", "minor-damage": "minor", "major-damage": "major", "destroyed": "destr."}
# Fixed so an empty result still has the right columns.
CANDIDATE_COLUMNS = ["disaster", "tile_id", "uid", "damage", "neighbour_classes", "neighbour_majority",
                     "nearest_px", "fifth_px", "area_px", "tile_buildings", "row_index"]


def find_candidates(df, k=5):
    """Return (candidates DataFrame, eligible-building count per (disaster, damage))."""
    df = df[df["damage"].isin(REAL)]
    rows, eligible = [], []
    for (disaster, tile_id), t in df.groupby(["disaster", "tile_id"], observed=True):
        if len(t) < k + 1:
            continue
        xy = t[["centroid_x", "centroid_y"]].to_numpy()
        # pairwise distances; a building is not its own neighbour
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :k]
        cls = t["damage"].astype(str).to_numpy()
        eligible.append(t[["disaster", "damage"]])
        differs = (cls[nn] != cls[:, None]).all(axis=1)
        for i in np.flatnonzero(differs):
            nb = nn[i]
            rows.append({
                "disaster": disaster, "tile_id": tile_id, "uid": t["uid"].iloc[i],
                "damage": cls[i], "neighbour_classes": ", ".join(cls[nb]),
                "neighbour_majority": pd.Series(cls[nb]).mode().iloc[0],
                "nearest_px": float(d[i, nb[0]]), "fifth_px": float(d[i, nb[-1]]),
                "area_px": float(t["area_px"].iloc[i]), "tile_buildings": len(t),
                "row_index": t.index[i],
            })
    elig = pd.concat(eligible) if eligible else df.iloc[:0][["disaster", "damage"]]
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), elig


def pick_gallery(cands, n, seed):
    """Up to n candidates, spread as evenly as possible over the four classes."""
    rng = np.random.default_rng(seed)
    per_class = {c: cands[cands["damage"] == c] for c in REAL}
    quota = {c: 0 for c in REAL}
    remaining = n
    while remaining > 0 and any(quota[c] < len(per_class[c]) for c in REAL):
        for c in REAL:
            if remaining and quota[c] < len(per_class[c]):
                quota[c] += 1
                remaining -= 1
    picks = [g.iloc[np.sort(rng.choice(len(g), size=quota[c], replace=False))]
             for c, g in per_class.items() if quota[c]]
    return pd.concat(picks).reset_index(drop=True) if picks else cands.iloc[:0]


def crop(img, cx, cy, side):
    half = side // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, img.shape[1]), min(y0 + side, img.shape[0])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, x0, y0


def draw_gallery(gallery, df, out_dir, per_page=10, side=160, max_side=400):
    paths = []
    for page in range(0, len(gallery), per_page):
        chunk = gallery.iloc[page:page + per_page]
        cols = 5
        rows = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.9, rows * 2.9 + 0.6), dpi=110, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes.ravel():
            ax.axis("off")
        for j, (gid, c) in enumerate(chunk.iterrows()):
            b = df.loc[c["row_index"]]
            tile = df[(df["disaster"] == c["disaster"]) & (df["tile_id"] == c["tile_id"])]
            # big enough to show the building and its 5th-nearest neighbour, within limits
            bw = max(b["polygon"].bounds[2] - b["polygon"].bounds[0], b["polygon"].bounds[3] - b["polygon"].bounds[1])
            s = int(np.clip(max(side, bw + 60, 2 * c["fifth_px"] + 60), side, max_side))
            for k, col in enumerate(("pre_image", "post_image")):
                ax = axes[j // cols, (j % cols) * 2 + k]
                img = np.asarray(Image.open(b[col]).convert("RGB"))
                cimg, x0, y0 = crop(img, b["centroid_x"], b["centroid_y"], s)
                ax.imshow(cimg, extent=(0, s, s, 0))
                # neighbours (thin, their class colour) and the candidate (thick, white halo)
                for _, o in tile.iterrows():
                    if o.name == b.name or o["damage"] == "un-classified":
                        continue
                    xy = np.asarray(o["polygon"].exterior.coords) - (x0, y0)
                    ax.plot(xy[:, 0], xy[:, 1], color=DAMAGE_COLORS[o["damage"]], linewidth=0.8, alpha=0.9)
                xy = np.asarray(b["polygon"].exterior.coords) - (x0, y0)
                ax.plot(xy[:, 0], xy[:, 1], color="white", linewidth=3.2)
                ax.plot(xy[:, 0], xy[:, 1], color=DAMAGE_COLORS[b["damage"]], linewidth=1.8)
                ax.set_xlim(0, s)
                ax.set_ylim(s, 0)
                ax.set_title("pre" if k == 0 else "post", fontsize=7, color=INK_SECONDARY)
            nb = ", ".join(SHORT[x.strip()] for x in c["neighbour_classes"].split(","))
            axes[j // cols, (j % cols) * 2].text(
                0, -0.2, f"#{gid + 1}  {b['disaster']} {b['tile_id']}\n"
                         f"label: {b['damage']}   neighbours: {nb}",
                transform=axes[j // cols, (j % cols) * 2].transAxes, fontsize=6.5, color=INK, va="top")
        handles = [plt.Line2D([], [], color=DAMAGE_COLORS[c], linewidth=2, label=c) for c in REAL]
        fig.legend(handles=handles, loc="upper right", ncols=4, frameon=False, fontsize=8)
        fig.suptitle("Label-noise candidates: thick outline = the building, thin outlines = other "
                     "buildings (colour = labelled class)", x=0.01, ha="left", fontsize=10, color=INK)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.1, wspace=0.04, hspace=0.6)
        path = out_dir / f"gallery_{page // per_page + 1:02d}.png"
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gallery", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doc", type=Path, default=Path("docs/label_noise.md"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/label_noise"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for s in args.splits:
        f = load_labels(args.data_dir / s)
        f.insert(0, "split", s)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["disaster"] = df["disaster"].astype(str)

    cands, elig = find_candidates(df, args.k)
    cands.drop(columns=["row_index"]).to_csv(args.out_dir / "candidates.csv", index=False)

    elig_d = elig.groupby("disaster").size()
    by_dis = pd.DataFrame({"eligible buildings": elig_d,
                           "candidates": cands.groupby("disaster").size().reindex(elig_d.index, fill_value=0)})
    by_dis["share"] = by_dis["candidates"] / by_dis["eligible buildings"]
    by_dis.loc["all"] = [int(by_dis["eligible buildings"].sum()), int(by_dis["candidates"].sum()),
                         by_dis["candidates"].sum() / by_dis["eligible buildings"].sum()]
    by_dis[["eligible buildings", "candidates"]] = by_dis[["eligible buildings", "candidates"]].astype(int)

    elig_c = elig.groupby("damage", observed=False).size().reindex(REAL)
    by_cls = pd.DataFrame({"eligible buildings": elig_c,
                           "candidates": cands.groupby("damage").size().reindex(REAL, fill_value=0)})
    by_cls["share"] = by_cls["candidates"] / by_cls["eligible buildings"]
    by_cls.index.name = "labelled class"
    maj = (pd.crosstab(cands["damage"], cands["neighbour_majority"]) if len(cands)
           else pd.DataFrame()).reindex(index=REAL, columns=REAL, fill_value=0)
    maj.index.name = "labelled class \\ neighbours' most common class"

    gallery = pick_gallery(cands, args.gallery, args.seed)
    paths = draw_gallery(gallery, df, args.out_dir)
    print(f"{len(cands):,} candidates; gallery of {len(gallery)} in {len(paths)} images")

    g = gallery.assign(**{"#": np.arange(1, len(gallery) + 1)})
    g["tile"] = g["disaster"] + "_" + g["tile_id"]
    g["nearest / 5th neighbour px"] = g["nearest_px"].round(0).astype(int).astype(str) + " / " + \
        g["fifth_px"].round(0).astype(int).astype(str)
    g["looks mislabelled? (yes / no / unsure)"] = ""
    table = g[["#", "tile", "uid", "damage", "neighbour_classes", "nearest / 5th neighbour px",
               "looks mislabelled? (yes / no / unsure)"]].set_index("#")

    L = ["# Label-noise candidates\n",
         f"Generated by `scripts/label_noise.py` from `{args.data_dir}` (splits: "
         f"{', '.join(args.splits)}). **Nothing has been relabelled or removed.**\n",
         f"**Definition.** A building is a candidate if its damage class differs from the "
         f"class of *every one* of its {args.k} nearest neighbours in the same tile (by pixel "
         "centroid). Only the four real classes take part: un-classified buildings are "
         f"neither candidates nor neighbours. Tiles with fewer than {args.k + 1} classified "
         "buildings are skipped, so 'eligible buildings' below is the number checked.\n",
         "**What a candidate is, and isn't.** Damage is spatially clustered "
         "(`docs/observations.md`), so a building that disagrees with all its neighbours is "
         "unusual, and worth checking by eye. It isn't evidence of a mistake: a single "
         "building really can be the only one destroyed (or the only one spared) on a "
         "street. The rule is also blind to mistakes in *clusters*, e.g. a whole block "
         "labelled one class when it should be another.\n",
         "## Counts per disaster\n", df_to_markdown(by_dis, float_fmt="{:.1%}") + "\n",
         "## Counts per labelled class\n", df_to_markdown(by_cls, float_fmt="{:.1%}") + "\n",
         "What the neighbours are, for each labelled class (candidates only):\n",
         df_to_markdown(maj) + "\n",
         "The rarer classes are flagged far more often, partly for a mechanical reason: a "
         "rare-class building is more likely to be surrounded by the common class. So the "
         "per-class shares are not an estimate of per-class error rates.\n",
         "## Gallery for review\n",
         f"{len(gallery)} candidates, spread over the four labelled classes and otherwise "
         f"random (seed {args.seed}): " + ", ".join(f"`outputs/label_noise/{p.name}`" for p in paths)
         + " (not tracked in git; regenerate with the command in the script). Each example "
         "shows the pre and post crop around the building. The building has a thick outline "
         "in its labelled class colour; other buildings have thin outlines in theirs. The "
         "last column is for your judgement.\n",
         df_to_markdown(table) + "\n",
         "All candidates: `outputs/label_noise/candidates.csv`.\n"]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.doc}")


if __name__ == "__main__":
    main()
