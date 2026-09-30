"""Before/after images for the most-damaged xBD tiles.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\visualize.py --data-root C:\\Users\\wikel\\dev\\xbd-data\\test

Picks tiles by damage (default: 4 with the most destroyed buildings, 3 with the most
major-damage, 3 with the most minor-damage; within each group every tile comes from a
different disaster) and writes one PNG per tile to outputs/viz/ (override with --out-dir).

The before panel shows plain white building outlines, so it doesn't hint at what happened.
The after panel colours each building by its post-disaster damage class, with a legend.

If an output file is locked (e.g. open in an image viewer), the figure is written next to
it with a suffix (``_new``, ``_new2``, ...) and the script lists those files at the end.
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

PRE_OUTLINE = "white"

# Short labels for the figure title.
SHORT = {"no-damage": "no", "minor-damage": "minor", "major-damage": "major",
         "destroyed": "destroyed", "un-classified": "uncl."}


def polygon_patches(tile_df):
    """Turn each shapely polygon into a matplotlib patch; return (patches, colours)."""
    patches, colors = [], []
    for poly, damage in zip(tile_df["polygon"], tile_df["damage"]):
        # exterior.coords is a sequence of (x, y) pixel coordinates; the last repeats the first
        xy = np.asarray(poly.exterior.coords)
        patches.append(MplPolygon(xy, closed=True))
        colors.append(DAMAGE_COLORS[damage])
    return patches, colors


def select_tiles(df, quotas=(("destroyed", 4), ("major-damage", 3), ("minor-damage", 3))):
    """Pick tiles with the most buildings of each damage class.

    For each (class, n) in ``quotas``, in order, take the n tiles with the highest count of
    that class, such that within the group every tile is from a different disaster and no
    tile is picked twice overall. Tiles with zero buildings of the class are never picked,
    so a group can come back short. Returns a list of (group, disaster, tile_id, count).
    """
    counts = (df.groupby(["disaster", "tile_id", "damage"], observed=True).size()
                .unstack("damage", fill_value=0)
                .reindex(columns=DAMAGE_CLASSES, fill_value=0))
    picked, used_tiles = [], set()
    for cls, n in quotas:
        ranked = counts[cls].sort_values(ascending=False, kind="stable")
        used_disasters = set()
        for (disaster, tile_id), count in ranked.items():
            if len(used_disasters) == n or count == 0:
                break
            if disaster in used_disasters or (disaster, tile_id) in used_tiles:
                continue
            picked.append((cls, disaster, tile_id, int(count)))
            used_disasters.add(disaster)
            used_tiles.add((disaster, tile_id))
    return picked


def figure_title(tile_df, group=None):
    first = tile_df.iloc[0]
    name = Path(first["post_image"]).name.replace("_post_disaster.png", "")
    counts = tile_df["damage"].value_counts()
    per_class = ", ".join(f"{SHORT[c]} {counts.get(c, 0)}" for c in DAMAGE_CLASSES)
    why = f"  [most {group}]" if group else ""
    return (f"{name}  ({first['disaster']}, {first['disaster_type']}){why}\n"
            f"{len(tile_df)} buildings: {per_class}")


def make_figure(tile_df, group=None):
    """Build the before/after figure for one tile and return it (caller saves/closes it)."""
    first = tile_df.iloc[0]
    pre = np.asarray(Image.open(first["pre_image"]).convert("RGB"))
    post = np.asarray(Image.open(first["post_image"]).convert("RGB"))

    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(14, 7.9), dpi=110)
    fig.patch.set_facecolor(SURFACE)

    # Before: plain white outlines, no damage information.
    patches, colors = polygon_patches(tile_df)
    ax_pre.imshow(pre)  # image row 0 is at the top, matching the label's y-down pixel coords
    ax_pre.add_collection(PatchCollection(patches, facecolor="none", edgecolor=PRE_OUTLINE,
                                          linewidth=1.2))
    ax_pre.set_title("Before (pre-disaster): building outlines", color=INK, fontsize=11, loc="left")

    # After: damage colours. A PatchCollection can't be shared between axes, so rebuild it.
    patches, colors = polygon_patches(tile_df)
    ax_post.imshow(post)
    # White halo first, coloured outline on top: keeps dark outlines visible on dark ground.
    ax_post.add_collection(PatchCollection(patches, facecolor="none", edgecolor="white",
                                           linewidth=2.8, alpha=0.7))
    ax_post.add_collection(PatchCollection(patches, facecolor="none", edgecolor=colors,
                                           linewidth=1.3))
    # A faint fill makes small buildings easier to spot.
    ax_post.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.25))
    ax_post.set_title("After (post-disaster): coloured by damage class", color=INK, fontsize=11,
                      loc="left")

    for ax, img in ((ax_pre, pre), (ax_post, post)):
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")

    counts = tile_df["damage"].value_counts()
    legend = [Patch(facecolor=DAMAGE_COLORS[c], edgecolor=DAMAGE_COLORS[c],
                    label=f"{c} ({counts.get(c, 0)})") for c in DAMAGE_CLASSES]
    # Legend sits under the after panel only, since only that panel uses the colours.
    # One row of five, so no legend row can fall off the bottom of the figure.
    leg = ax_post.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=5,
                         frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, handlelength=1.2,
                         handletextpad=0.4, columnspacing=0.9)
    # Keep the legend out of tight_layout (otherwise it shrinks the images to make room);
    # the bottom 5% of the figure is reserved for it instead.
    leg.set_in_layout(False)
    fig.suptitle(figure_title(tile_df, group), x=0.02, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def save_figure(fig, path):
    """Save ``fig`` to ``path``; if that file is locked, try path_new.png, path_new2.png, ...

    Returns the path actually written.
    """
    path = Path(path)
    candidates = [path] + [path.with_name(f"{path.stem}_new{'' if i == 1 else i}{path.suffix}")
                           for i in range(1, 10)]
    for candidate in candidates:
        try:
            fig.savefig(candidate, facecolor=fig.get_facecolor())
            return candidate
        except OSError:  # Windows raises PermissionError or Errno 22 for a file open elsewhere
            continue
    raise OSError(f"Could not write {path} or any suffixed alternative")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/viz"))
    ap.add_argument("--n-destroyed", type=int, default=4)
    ap.add_argument("--n-major", type=int, default=3)
    ap.add_argument("--n-minor", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labels(args.data_root)
    chosen = select_tiles(df, (("destroyed", args.n_destroyed), ("major-damage", args.n_major),
                               ("minor-damage", args.n_minor)))

    redirected = []
    for group, disaster, tile_id, count in chosen:
        tile_df = df[(df["disaster"] == disaster) & (df["tile_id"] == tile_id)]
        wanted = args.out_dir / f"{disaster}_{tile_id}.png"
        fig = make_figure(tile_df, group)
        written = save_figure(fig, wanted)
        plt.close(fig)
        if written != wanted:
            redirected.append((wanted, written))
        print(f"  [{group}: {count}] wrote {written} ({len(tile_df)} buildings)")

    if redirected:
        print("\nThese files were locked, so the figure was written under a new name:")
        for wanted, written in redirected:
            print(f"  {wanted}  ->  {written}")


if __name__ == "__main__":
    main()
