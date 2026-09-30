"""Building bounding-box sizes and how they fit fixed crop sizes.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\crop_sizes.py --data-dir C:\\Users\\wikel\\dev\\xbd-data

Writes the "crop-sizes" section of docs/data_quality.md and an example figure to
outputs/crops_example.png.

A crop setting (S, P) means: take the building's bounding box, add P pixels of context on
every side, make it square (side = longest bbox side + 2P) around the box centre, and
resize that square to S x S. The building "fits" if the square is at most S pixels, i.e.
it can be used at native resolution (or upsampled) without being downscaled.
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
from src.reporting import df_to_markdown, write_section  # noqa: E402

CROPS = (32, 64, 128)
PADS = (0, 16, 32)
TILE_SIZE = 1024


def bbox_sizes(df):
    b = np.array([p.bounds for p in df["polygon"]])
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return df.assign(bbox_w=w, bbox_h=h, bbox_side=np.maximum(w, h),
                     cx=(b[:, 0] + b[:, 2]) / 2, cy=(b[:, 1] + b[:, 3]) / 2)


def fit_table(df, groupby=None):
    """% of buildings whose padded square (side + 2P) fits in S, per (S, P)."""
    cols = {}
    for s in CROPS:
        for p in PADS:
            fits = (df["bbox_side"] + 2 * p) <= s
            cols[f"{s} px, pad {p}"] = fits.groupby(df[groupby], observed=False).mean() if groupby else [fits.mean()]
    return pd.DataFrame(cols, index=None if groupby else pd.Index(["all buildings"], name=""))


def crop_square(img, cx, cy, side, out):
    """Square of `side` px around (cx, cy), zero-padded outside the image, resized to out x out."""
    half = side / 2
    x0, y0 = int(np.floor(cx - half)), int(np.floor(cy - half))
    n = int(np.ceil(side))
    canvas = np.zeros((n, n, 3), dtype=np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + n, img.shape[1]), min(y0 + n, img.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    resized = Image.fromarray(canvas).resize((out, out), Image.BILINEAR)
    return np.asarray(resized), (x0, y0, n)


def pick_examples(df, seed, n=6):
    """One building at each of several size percentiles, from different disasters, not at an edge."""
    rng = np.random.default_rng(seed)
    inner = df[(df["cx"] > 80) & (df["cx"] < TILE_SIZE - 80) & (df["cy"] > 80) & (df["cy"] < TILE_SIZE - 80)
               & (df["damage"] != "un-classified")]
    wanted = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99][:n]
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed", "no-damage", "destroyed"]
    used, rows = set(), []
    for q, cls in zip(wanted, classes):
        target = df["bbox_side"].quantile(q)
        cand = inner[(inner["bbox_side"].between(target * 0.9, target * 1.1)) & (~inner["disaster"].isin(used))]
        pref = cand[cand["damage"] == cls]
        pool = pref if len(pref) else cand
        if len(pool):
            row = pool.iloc[rng.integers(len(pool))]
        else:  # nothing near this size (tiny datasets): take the closest-sized building
            fallback = inner if len(inner) else df
            row = fallback.iloc[int((fallback["bbox_side"] - target).abs().to_numpy().argmin())]
        used.add(row["disaster"])
        rows.append((q, row))
    return rows


def example_figure(examples, path):
    settings = [(s, p) for s in CROPS for p in PADS]
    n_rows, n_cols = len(examples), len(settings) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.95, n_rows * 1.25 + 0.9), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for r, (q, b) in enumerate(examples):
        pre = np.asarray(Image.open(b["pre_image"]).convert("RGB"))
        post = np.asarray(Image.open(b["post_image"]).convert("RGB"))
        xy = np.asarray(b["polygon"].exterior.coords)
        for c, (s, p) in enumerate(settings):
            side = b["bbox_side"] + 2 * p
            for k, img in enumerate((pre, post)):
                ax = axes[r, 2 * c + k]
                crop, (x0, y0, n) = crop_square(img, b["cx"], b["cy"], side, s)
                ax.imshow(crop, extent=(0, s, s, 0), interpolation="nearest")
                scale = s / n
                ax.plot((xy[:, 0] - x0) * scale, (xy[:, 1] - y0) * scale,
                        color="white" if k == 0 else DAMAGE_COLORS[b["damage"]], linewidth=0.7)
                ax.set_xlim(0, s)
                ax.set_ylim(s, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title("pre" if k == 0 else "post", fontsize=6.5, color=INK_SECONDARY)
                if c == 0 and k == 0:
                    ax.set_ylabel(f"p{int(q * 100)} size\n{b['bbox_side']:.0f} px\n{b['damage']}\n"
                                  f"{b['disaster']}", fontsize=6.5, color=INK_SECONDARY, rotation=0,
                                  ha="right", va="center", labelpad=4)
                if side > s and k == 1:
                    ax.text(s * 0.97, s * 0.97, "downscaled", color="white", fontsize=5.5,
                            ha="right", va="bottom",
                            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "linewidth": 0})
    fig.suptitle("The same 6 buildings at each crop setting: box + pad px of context, squared, "
                 "resized to crop px (pre left, post right; outline = building polygon)",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.84, bottom=0.01, wspace=0.05, hspace=0.08)
    # One label per setting, centred over its pre/post pair.
    for c, (s, p) in enumerate(settings):
        left, right = axes[0, 2 * c].get_position(), axes[0, 2 * c + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.045, f"crop {s} px, pad {p}",
                 ha="center", va="bottom", fontsize=7.5, color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--doc", type=Path, default=Path("docs/data_quality.md"))
    ap.add_argument("--figure", type=Path, default=Path("outputs/crops_example.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.concat([load_labels(args.data_dir / s) for s in args.splits], ignore_index=True)
    df["disaster"] = df["disaster"].astype(str)
    df = bbox_sizes(df)

    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    pct = df.groupby("disaster")["bbox_side"].quantile(qs).unstack()
    pct.columns = [f"p{int(q * 100)}" for q in qs]
    pct.loc["all"] = df["bbox_side"].quantile(qs).to_numpy()
    pct.insert(0, "buildings", df.groupby("disaster").size().reindex(pct.index).fillna(len(df)).astype(int))

    overall = fit_table(df)
    by_class = fit_table(df, "damage")
    by_class.index.name = "damage"
    by_disaster = pd.DataFrame({f"{s} px, pad 16": ((df["bbox_side"] + 32) <= s).groupby(df["disaster"]).mean()
                                for s in CROPS})

    examples = pick_examples(df, args.seed)
    example_figure(examples, args.figure)
    print(f"Wrote {args.figure}")

    L = ["## 7. Crop sizes\n",
         "Generated by `scripts/crop_sizes.py`. Building size here = the **longer side of its "
         "axis-aligned bounding box**, in pixels.\n",
         "### Bounding-box size per disaster (px)\n",
         df_to_markdown(pct, float_fmt="{:.0f}") + "\n",
         "### Share of buildings that fit each crop setting\n",
         "A setting (crop S, pad P) takes the bounding box plus P px of context on each side, "
         "squared (side = longest box side + 2P), and resizes it to S x S. A building **fits** "
         "if that square is at most S px, so it needs no downscaling (smaller squares are "
         "upsampled). Buildings that don't fit lose detail when shrunk to S.\n",
         df_to_markdown(overall, float_fmt="{:.1%}", index=False) + "\n",
         "By damage class:\n",
         df_to_markdown(by_class, float_fmt="{:.1%}") + "\n",
         "By disaster, with 16 px padding:\n",
         df_to_markdown(by_disaster, float_fmt="{:.1%}") + "\n",
         f"Example figure: `outputs/crops_example.png` (not tracked): 6 buildings at the "
         f"5th to 99th size percentiles, from different disasters, each shown at all 9 "
         f"settings, pre and post side by side. Crops marked 'downscaled' did not fit. The "
         f"32 px settings with padding fit nothing by construction: 16 px of context on each "
         f"side already fills a 32 px crop.\n",
         "Two things are visible in the example crops (judged by eye, not measured). "
         "First, the polygon lines up with the building in the pre image, but in some post "
         "images the building sits several pixels off it: about 10 px for the Palu example, "
         "a few pixels for the Michael and Florence ones. That is direct visual evidence of "
         "the pre/post misalignment that section 6 could not measure. Second, the Mexico "
         "example's post image is much blurrier than its pre image, so the two captures "
         "can differ in effective resolution.\n",
         "**For modelling:** " + (
             f"the median building is {df['bbox_side'].median():.0f} px on its longest side, so a "
             f"64 px crop with 16 px of context fits {overall.iloc[0]['64 px, pad 16']:.0%} of "
             f"buildings, and 128 px with 32 px fits {overall.iloc[0]['128 px, pad 32']:.0%}. "
             "Small crops force a choice between shrinking larger buildings (losing detail) and "
             "cutting them off. Large crops leave small buildings as a few pixels in a mostly-"
             "context window, or require upsampling. Classes differ in size (destroyed "
             "buildings are the smallest, major-damage the largest; section 3), so the crop "
             "setting changes each class's input differently. Context padding would also absorb "
             "a few pixels of pre/post misalignment (visible above) and gives the model surroundings (debris, "
             "water, burn scars), which the observations suggest carry damage information. "
             "Larger crops cost GPU memory quadratically: 128 px is 4x the pixels of 64 px."),
         ]
    write_section(args.doc, "crop-sizes", "\n".join(L))
    print(f"Wrote the 'crop-sizes' section of {args.doc}")


if __name__ == "__main__":
    main()
