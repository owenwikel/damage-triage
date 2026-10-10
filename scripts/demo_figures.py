"""Demo figures: what the model says about whole tiles -> outputs/demo/ and docs/demo.md.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\demo_figures.py --run runs\\stack_aug_s0 --data-dir C:\\Users\\wikel\\dev\\xbd-data

For six validation tiles from six different disasters, draws three panels:
    before                      the pre-disaster image
    after + model               every building coloured by the PREDICTED damage class;
                                buildings the model is not sure about are outlined REVIEW
    after + annotators' labels  the same tile coloured by the true labels, to compare

REVIEW = confidence (top probability after temperature scaling) below the threshold that
auto-accepts 80% of val-tiles buildings. One threshold for every tile, fixed on val-tiles,
as it would be in use: you cannot tune a threshold on a disaster that has just happened.

Only val and val-tiles tiles can be drawn; the test part is blocked by src/split.py.

For the README: unseen-disaster tiles only, three per disaster, into docs/figures/, with
a document of their own (docs/demo.md is left as it is):
    .\\.venv\\Scripts\\python.exe scripts\\demo_figures.py --run runs\\stack_aug_s0 --data-dir C:\\Users\\wikel\\dev\\xbd-data --parts val --per-disaster 3 --out-dir docs\\figures --prefix demo_val_ --doc docs\\demo_val.md
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import calibrated_probs, fit_temperature  # noqa: E402
from src.colors import DAMAGE_COLORS, INK, INK_SECONDARY, SURFACE  # noqa: E402
from src.dataset import CLASS_NAMES, read_index  # noqa: E402
from src.labels import parse_label_file  # noqa: E402
from src.split import DEFAULT_SPLIT, find_folder, load_split  # noqa: E402
from src.triage import at_coverage  # noqa: E402

# Two unseen disasters (val) and four training disasters' held-out tiles (val-tiles).
DISASTERS = ["hurricane-florence", "midwest-flooding", "hurricane-harvey",
             "hurricane-matthew", "palu-tsunami", "socal-fire"]
REVIEW_COVERAGE = 0.80


def predictions(run_dir, cache_dir):
    """val-tiles and val buildings with pred, confidence and the REVIEW flag."""
    index = read_index(cache_dir)
    fit = np.load(run_dir / "preds_val-tiles.npz")
    temperature = fit_temperature(fit["logits"], fit["labels"])
    frames, threshold = [], None
    for part in ("val-tiles", "val"):
        d = np.load(run_dir / f"preds_{part}.npz")
        sub = index[index["part"] == part].sort_values("row").reset_index(drop=True)
        p = calibrated_probs(d["logits"], temperature)
        sub["pred"], sub["confidence"] = p.argmax(axis=1), p.max(axis=1)
        if part == "val-tiles":
            threshold = at_coverage(sub["confidence"], sub["label"], sub["pred"],
                                    coverages=(REVIEW_COVERAGE,))[0]["threshold"]
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["review"] = df["confidence"] < threshold
    return df, threshold


def pick_tiles(df, disasters, min_buildings=25, max_buildings=160, per_disaster=1):
    """``per_disaster`` tiles per disaster: those showing the most kinds of damage, then
    the most damaged buildings. Chosen from the true labels, not from how well the model
    did."""
    picks = []
    for disaster in disasters:
        d = df[df["disaster"] == disaster]
        stats = d.groupby("tile_id").agg(n=("label", "size"), kinds=("label", "nunique"),
                                         damaged=("label", lambda s: int((s > 0).sum())))
        ok = stats[(stats["n"] >= min_buildings) & (stats["n"] <= max_buildings)]
        stats = ok if len(ok) else stats
        if stats.empty:
            continue
        ranked = stats.sort_values(["kinds", "damaged", "n"], ascending=False, kind="stable")
        picks += [(disaster, tile_id) for tile_id in ranked.index[:per_disaster]]
    return picks


def patches_for(polys):
    return [MplPolygon(np.asarray(p.exterior.coords), closed=True) for p in polys]


def draw_tile(tile, polys, pre, post, threshold):
    """tile: prediction rows for one tile; polys: {uid: shapely polygon}."""
    polys = [polys[u] for u in tile["uid"]]
    pred_colors = [DAMAGE_COLORS[CLASS_NAMES[i]] for i in tile["pred"]]
    true_colors = [DAMAGE_COLORS[CLASS_NAMES[i]] for i in tile["label"]]
    review = tile["review"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    axes[0].imshow(pre)
    axes[0].set_title("Before", loc="left", fontsize=12, color=INK)
    for ax, colors, title in ((axes[1], pred_colors, "After, coloured by the model's prediction"),
                              (axes[2], true_colors, "After, coloured by the annotators' labels")):
        ax.imshow(post)
        # white halo, then a faint fill and a coloured outline: visible on any background
        ax.add_collection(PatchCollection(patches_for(polys), facecolor="none",
                                          edgecolor="white", linewidth=2.6, alpha=0.7))
        ax.add_collection(PatchCollection(patches_for(polys), facecolor=colors,
                                          edgecolor="none", alpha=0.30))
        ax.add_collection(PatchCollection(patches_for(polys), facecolor="none",
                                          edgecolor=colors, linewidth=1.4))
        ax.set_title(title, loc="left", fontsize=12, color=INK)
    # REVIEW: a dashed black-on-white ring around the building's bounding box, so it
    # reads as a flag on top of the colour rather than as a fifth class.
    for poly, flag in zip(polys, review):
        if flag:
            x0, y0, x1, y1 = poly.bounds
            pad = 6
            for color, style, width in (("white", "-", 2.6), ("#0b0b0b", (0, (3, 2)), 1.4)):
                axes[1].add_patch(plt.Rectangle((x0 - pad, y0 - pad), x1 - x0 + 2 * pad,
                                                y1 - y0 + 2 * pad, fill=False, edgecolor=color,
                                                linestyle=style, linewidth=width))
    for ax in axes:
        ax.set_xlim(0, post.shape[1])
        ax.set_ylim(post.shape[0], 0)
        ax.axis("off")

    handles = [Patch(facecolor=DAMAGE_COLORS[c], edgecolor=DAMAGE_COLORS[c], label=c)
               for c in CLASS_NAMES]
    handles.append(Line2D([0], [0], color="#0b0b0b", linestyle=(0, (3, 2)), linewidth=1.4,
                          label=f"REVIEW: confidence below {threshold:.2f}"))
    leg = fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False, fontsize=10,
                     labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.005))
    leg.set_in_layout(False)
    first = tile.iloc[0]
    acc = (tile["pred"] == tile["label"]).mean()
    kept = tile[~tile["review"]]
    kept_acc = (kept["pred"] == kept["label"]).mean() if len(kept) else float("nan")
    where = ("unseen disaster (val)" if first["part"] == "val"
             else "held-out tile of a training disaster (val-tiles)")
    fig.suptitle(f"{first['disaster']}, tile {first['tile_id']}   |   {where}\n"
                 f"{len(tile)} buildings: {acc:.0%} predicted correctly; {int(review.sum())} "
                 f"flagged for review; {kept_acc:.0%} correct among the {len(kept)} not flagged",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    return fig, {"disaster": first["disaster"], "tile": first["tile_id"], "part": first["part"],
                 "buildings": len(tile), "accuracy": acc, "flagged": int(review.sum()),
                 "accuracy not flagged": kept_acc}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None, help="default: <data-dir>/crops")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    ap.add_argument("--doc", type=Path, default=Path("docs/demo.md"))
    # No "test" among the choices: this script has no way to draw a test tile.
    ap.add_argument("--parts", nargs="+", default=["val-tiles", "val"], choices=["val-tiles", "val"],
                    help="draw tiles of these parts only")
    ap.add_argument("--per-disaster", type=int, default=1, help="tiles per disaster")
    ap.add_argument("--prefix", default="", help="put in front of every figure's file name")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, threshold = predictions(args.run, args.cache_dir or args.data_dir / "crops")
    allowed = load_split()         # dev parts only; "test" would need final=True
    allowed = set(zip(allowed["disaster"], allowed["tile_id"]))
    rows = []
    # The threshold above is always fixed on val-tiles; --parts only limits what is drawn.
    shown = df[df["part"].isin(args.parts)]
    disasters = [d for d in DISASTERS if d in set(shown["disaster"])]
    for disaster, tile_id in pick_tiles(shown, disasters, per_disaster=args.per_disaster):
        assert (disaster, tile_id) in allowed
        folder = find_folder(args.data_dir, disaster, tile_id)
        stem = f"{disaster}_{tile_id}"
        polys = {r["uid"]: r["polygon"] for r in parse_label_file(
            folder / "labels" / f"{stem}_post_disaster.json", folder / "images")}
        pre = np.asarray(Image.open(folder / "images" / f"{stem}_pre_disaster.png").convert("RGB"))
        post = np.asarray(Image.open(folder / "images" / f"{stem}_post_disaster.png").convert("RGB"))
        tile = df[(df["disaster"] == disaster) & (df["tile_id"] == tile_id)]
        fig, info = draw_tile(tile, polys, pre, post, threshold)
        path = args.out_dir / f"{args.prefix}{stem}.jpg"
        fig.savefig(path, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 88})
        plt.close(fig)
        info["file"] = path.name
        rows.append(info)
        print(f"wrote {path}")

    # Where the figures are, seen from the document: docs/demo.md -> ../outputs/demo
    rel = Path(os.path.relpath(args.out_dir, args.doc.parent))
    share = df.groupby("part")["review"].mean()
    L = ["# Demo: predictions on whole tiles (validation only)\n",
         f"Generated by `scripts/demo_figures.py` from `{args.run.as_posix()}`. "
         + ("One tile per disaster: two disasters the model never trained on (val) and four "
            "held-out tiles of training disasters (val-tiles). "
            if set(args.parts) == {"val-tiles", "val"} and args.per_disaster == 1 else
            f"Parts shown: {', '.join(args.parts)}; {args.per_disaster} tile(s) per disaster. "
            "Every figure says in its title which part its tile is from. ")
         + "No test-set tile is shown.\n",
         "Tiles were picked from the annotators' labels (the tile with the most kinds of "
         "damage, then the most damaged buildings, among tiles with 25-160 buildings), "
         "**not** by how well the model did on them. They are examples, not a summary: "
         "the numbers that describe the model are in `docs/results.md`.\n",
         f"**REVIEW** marks buildings whose confidence is below {threshold:.3f}, the "
         f"threshold that auto-accepts {REVIEW_COVERAGE:.0%} of val-tiles buildings. With "
         f"that one threshold, {share.get('val-tiles', float('nan')):.0%} of val-tiles "
         f"buildings and {share.get('val', float('nan')):.0%} of val buildings are flagged.\n"]
    for r in rows:
        L += [f"## {r['disaster']}, tile {r['tile']} ({r['part']})\n",
              f"{r['buildings']} buildings, {r['accuracy']:.0%} predicted correctly, "
              f"{r['flagged']} flagged; {r['accuracy not flagged']:.0%} correct among those "
              "not flagged.\n",
              f"![{r['disaster']} {r['tile']}]({(rel / r['file']).as_posix()})\n"]
    args.doc.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.doc}")


if __name__ == "__main__":
    main()
