"""Before/after images for random xBD tiles, with building outlines coloured by damage class.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\visualize.py --data-root C:\\Users\\wikel\\dev\\xbd-data\\test

Writes one PNG per tile to outputs/viz/ (override with --out-dir). The same polygons are
drawn on both images, coloured by the POST-disaster damage class, so you can look at a
building before and after and see what the annotators decided.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.colors import DAMAGE_COLORS, INK, INK_SECONDARY, SURFACE  # noqa: E402
from src.labels import DAMAGE_CLASSES, load_labels  # noqa: E402


def polygon_patches(tile_df):
    """Turn each shapely polygon into a matplotlib patch; return (patches, colours)."""
    patches, colors = [], []
    for poly, damage in zip(tile_df["polygon"], tile_df["damage"]):
        # exterior.coords is a sequence of (x, y) pixel coordinates; the last repeats the first
        xy = np.asarray(poly.exterior.coords)
        patches.append(MplPolygon(xy, closed=True))
        colors.append(DAMAGE_COLORS[damage])
    return patches, colors


def draw_tile(tile_df, out_path):
    first = tile_df.iloc[0]
    pre = np.asarray(Image.open(first["pre_image"]).convert("RGB"))
    post = np.asarray(Image.open(first["post_image"]).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax, img, title in ((axes[0], pre, "Before (pre-disaster)"), (axes[1], post, "After (post-disaster)")):
        ax.imshow(img)  # image row 0 is at the top, matching the label's y-down pixel coords
        patches, colors = polygon_patches(tile_df)
        # A PatchCollection draws thousands of polygons far faster than adding them one by one.
        # White halo first, coloured outline on top: keeps dark outlines visible on dark ground.
        ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor="white",
                                          linewidth=2.8, alpha=0.7))
        ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=colors, linewidth=1.3))
        if title.startswith("After"):
            # a faint fill on the post image makes small buildings easier to spot
            ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.25))
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")

    counts = tile_df["damage"].value_counts()
    legend = [Patch(facecolor=DAMAGE_COLORS[c], edgecolor=DAMAGE_COLORS[c],
                    label=f"{c} ({counts.get(c, 0)})") for c in DAMAGE_CLASSES]
    fig.legend(handles=legend, loc="lower center", ncols=5, frameon=False, fontsize=10,
               labelcolor=INK_SECONDARY)
    fig.suptitle(f"{first['disaster']} ({first['disaster_type']}), tile {first['tile_id']}: "
                 f"{len(tile_df)} buildings", x=0.02, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/viz"))
    ap.add_argument("--n", type=int, default=10, help="number of tiles to draw")
    ap.add_argument("--seed", type=int, default=0, help="change this to get different tiles")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labels(args.data_root)
    # Only tiles that contain buildings appear in df, so every chosen tile has something to draw.
    tiles = df[["disaster", "tile_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    chosen = tiles.iloc[rng.choice(len(tiles), size=min(args.n, len(tiles)), replace=False)]

    for disaster, tile_id in chosen.itertuples(index=False):
        tile_df = df[(df["disaster"] == disaster) & (df["tile_id"] == tile_id)]
        out = args.out_dir / f"{disaster}_{tile_id}.png"
        draw_tile(tile_df, out)
        print(f"  wrote {out} ({len(tile_df)} buildings)")


if __name__ == "__main__":
    main()
