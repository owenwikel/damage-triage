"""Write the final train / val-tiles / val / test split -> splits/split_v1.csv.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\make_split.py --data-dir C:\\Users\\wikel\\dev\\xbd-data

The split is "P1 + val-tiles" from docs/split_candidates.md: whole disasters are held out
for val and test, and a second validation set (val-tiles) takes ~10% of the tiles of each
training disaster. val-tiles is chosen by spatial block rather than by single tile, so that
neighbouring tiles (same streets, same kind of damage) mostly land on the same side.

How a block is built:
  1. Each tile's lng/lat footprint is recovered from its label file: every building is
     listed in pixel and in lng/lat coordinates, so a straight-line fit between the two
     gives the position of the tile's centre and its width and height in degrees.
  2. Tiles are binned into a fixed lng/lat grid (--block-deg, default 0.02 degrees, about
     4 tiles across). A grid cell is a block.
  3. Where two tiles' footprints overlap, their blocks are merged, so overlapping tiles
     can never be separated.
  4. Per training disaster, blocks are shuffled (--seed) and walked once; a block is held
     out whenever that brings the held-out tile count closer to the 10% target.

Tiles without buildings have no coordinates in xBD and cannot be placed in a block. They
stay in train (they produce no building crops) and are left out of the 10% target.

Outputs:
  splits/split_v1.csv          disaster, tile_id, part   (the source of truth)
  splits/split_v1_blocks.csv   disaster, tile_id, lng, lat, block   (how it was derived)
  docs/split_final.md          class counts per part, block size, remaining leak
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import load_labels  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402

TRAIN = ["guatemala-volcano", "hurricane-harvey", "hurricane-matthew", "mexico-earthquake",
         "palu-tsunami", "socal-fire"]
VAL = ["hurricane-florence", "midwest-flooding"]
TEST = ["hurricane-michael", "santa-rosa-wildfire"]
PARTS = ["train", "val-tiles", "val", "test"]

REAL = ["no-damage", "minor-damage", "major-damage", "destroyed"]
BLOCK_DEG = 0.02       # block edge in degrees; a tile is ~0.005 deg wide and ~0.0045 deg tall
VAL_TILES_FRAC = 0.10
TOUCH_TOL = 0.05       # two tiles "touch" if the gap between them is under 5% of a tile
OVERLAP_MIN = 0.01     # footprints "overlap" if they share over 1% of the smaller tile's area


def tile_footprint(label_path):
    """Return (lng, lat, width, height) of a tile in degrees, or NaNs if it has no buildings.

    Fits lng/lat = A @ (x, y) + b over every polygon vertex in the label file.
    """
    with open(label_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    xy, ll = doc["features"]["xy"], doc["features"]["lng_lat"]
    nan = (np.nan,) * 4
    if not xy or len(xy) != len(ll):
        return nan
    px = np.vstack([np.asarray(shapely_wkt.loads(f["wkt"]).exterior.coords)[:-1] for f in xy])
    geo = np.vstack([np.asarray(shapely_wkt.loads(f["wkt"]).exterior.coords)[:-1] for f in ll])
    if len(px) != len(geo):
        return nan
    # centre both sides first: longitudes like -118.9 next to pixel values up to 1024
    # would otherwise cost precision
    px0, geo0 = px.mean(axis=0), geo.mean(axis=0)
    design = np.c_[px - px0, np.ones(len(px))]
    coef, _, rank, _ = np.linalg.lstsq(design, geo - geo0, rcond=None)
    if rank < 3:
        return nan
    meta = doc.get("metadata", {})
    width, height = meta.get("width", 1024), meta.get("height", 1024)
    centre = np.array([width / 2 - px0[0], height / 2 - px0[1], 1.0]) @ coef + geo0
    return centre[0], centre[1], abs(coef[0, 0]) * width, abs(coef[1, 1]) * height


def load_tiles(data_dir, folders):
    """One row per tile: disaster, tile_id, lng, lat, w, h (NaN where the tile is empty)."""
    rows = []
    for folder in folders:
        for p in sorted((Path(data_dir) / folder / "labels").glob("*_post_disaster.json")):
            disaster, tile_id = p.name.replace("_post_disaster.json", "").rsplit("_", 1)
            rows.append((disaster, tile_id, *tile_footprint(p)))
    tiles = pd.DataFrame(rows, columns=["disaster", "tile_id", "lng", "lat", "w", "h"])
    return tiles.sort_values(["disaster", "tile_id"], ignore_index=True)


def pair_geometry(g):
    """For the tiles of one disaster: boolean matrices (touch, overlap) between tiles."""
    lng, lat, w, h = (g[c].to_numpy() for c in ("lng", "lat", "w", "h"))
    dx = np.abs(lng[:, None] - lng[None, :])
    dy = np.abs(lat[:, None] - lat[None, :])
    sw = (w[:, None] + w[None, :]) / 2   # centre distance at which two tiles just touch
    sh = (h[:, None] + h[None, :]) / 2
    touch = (dx <= sw * (1 + TOUCH_TOL)) & (dy <= sh * (1 + TOUCH_TOL))
    shared = np.clip(sw - dx, 0, None) * np.clip(sh - dy, 0, None)
    area = w * h
    overlap = shared > OVERLAP_MIN * np.minimum(area[:, None], area[None, :])
    np.fill_diagonal(touch, False)
    np.fill_diagonal(overlap, False)
    return touch, overlap


def assign_blocks(g, block_deg):
    """Block label per tile of one disaster (all tiles must have coordinates).

    Starts from grid cells, then merges cells that hold a pair of overlapping tiles.
    """
    cells = [f"{int(np.floor(x / block_deg))}_{int(np.floor(y / block_deg))}"
             for x, y in zip(g["lng"], g["lat"])]
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    _, overlap = pair_geometry(g)
    for i, j in zip(*np.nonzero(np.triu(overlap))):
        a, b = sorted((find(cells[i]), find(cells[j])))
        parent[b] = a  # the alphabetically first cell names the merged block
    return [find(c) for c in cells]


def pick_val_blocks(blocks, frac, rng):
    """Choose blocks to hold out so that about ``frac`` of the tiles are held out.

    Walks the blocks once in random order and takes a block whenever that brings the
    held-out count closer to the target. Never takes every block.
    """
    sizes = pd.Series(blocks).value_counts()
    target = frac * len(blocks)
    order = rng.permutation(sorted(sizes.index))
    held, picked = 0, []
    for b in order:
        if abs(held + sizes[b] - target) < abs(held - target):
            picked.append(b)
            held += sizes[b]
    return picked


def make_split(tiles, block_deg=BLOCK_DEG, frac=VAL_TILES_FRAC, seed=0):
    """Add ``block`` and ``part`` columns to the tile table and return it."""
    unknown = sorted(set(tiles["disaster"]) - set(TRAIN + VAL + TEST))
    if unknown:
        raise ValueError(f"Disasters not assigned to any part: {unknown}")
    tiles = tiles.copy()
    tiles["block"] = ""
    tiles["part"] = np.select([tiles["disaster"].isin(VAL), tiles["disaster"].isin(TEST)],
                              ["val", "test"], default="train")
    rng = np.random.default_rng(seed)
    for disaster in sorted(tiles["disaster"].unique()):
        idx = tiles.index[(tiles["disaster"] == disaster) & tiles["lng"].notna()]
        if len(idx) == 0:
            continue
        tiles.loc[idx, "block"] = assign_blocks(tiles.loc[idx], block_deg)
        if disaster in TRAIN:
            picked = pick_val_blocks(tiles.loc[idx, "block"].tolist(), frac, rng)
            tiles.loc[idx[tiles.loc[idx, "block"].isin(picked)], "part"] = "val-tiles"
    return tiles


def leak_table(tiles):
    """Per training disaster: how much of val-tiles still sits next to train."""
    rows = []
    for disaster in TRAIN:
        g = tiles[(tiles["disaster"] == disaster) & tiles["lng"].notna()].reset_index(drop=True)
        if g.empty:
            continue
        touch, overlap = pair_geometry(g)
        is_val = (g["part"] == "val-tiles").to_numpy()
        cross = touch & is_val[:, None] & ~is_val[None, :]   # [val-tiles tile, train tile]
        val_blocks = g.loc[is_val, "block"]
        rows.append({
            "disaster": disaster,
            "tiles with buildings": len(g),
            "blocks": g["block"].nunique(),
            "val-tiles tiles": int(is_val.sum()),
            "share": is_val.mean(),
            "val-tiles blocks": val_blocks.nunique(),
            "blocks touching a train block": g.loc[cross.any(axis=1), "block"].nunique(),
            "val-tiles tiles touching a train tile": int(cross.any(axis=1).sum()),
            "train tiles touching a val-tiles tile": int(cross.any(axis=0).sum()),
            "overlapping pairs split": int((overlap & cross).sum()),
        })
    t = pd.DataFrame(rows).set_index("disaster")
    total = t.sum()
    total["share"] = total["val-tiles tiles"] / total["tiles with buildings"]
    t.loc["all"] = total
    return t.astype({c: int for c in t.columns if c != "share"})


def class_counts(tiles, df):
    """Tiles and buildings per damage class in each part."""
    b = df.merge(tiles[["disaster", "tile_id", "part"]], on=["disaster", "tile_id"])
    counts = pd.crosstab(b["part"], b["damage"]).reindex(columns=REAL + ["un-classified"],
                                                        fill_value=0)
    counts.insert(0, "tiles", tiles["part"].value_counts())
    counts["buildings"] = counts[REAL + ["un-classified"]].sum(axis=1)
    counts = counts.reindex([p for p in PARTS if p in counts.index])
    share = counts[REAL].div(counts[REAL].sum(axis=1), axis=0)
    share.columns = [f"% {c}" for c in share.columns]
    return counts, share


def report(tiles, df, args, n_overlap):
    counts, share = class_counts(tiles, df)
    leak = leak_table(tiles)
    all_ = leak.loc["all"].drop("share").astype(int)
    n_train_dis = int(tiles["disaster"].isin(TRAIN).sum())
    n_empty_train = int((tiles["disaster"].isin(TRAIN) & tiles["lng"].isna()).sum())
    km = args.block_deg * 111
    L = ["# Final split: P1 + val-tiles by spatial block (split_v1)\n",
         "Generated by `scripts/make_split.py`; the split itself is `splits/split_v1.csv` "
         f"({len(tiles):,} tiles; columns `disaster`, `tile_id`, `part`). The block each tile "
         "belongs to is in `splits/split_v1_blocks.csv`. This replaces the tile-by-tile "
         "val-tiles of the \"P1 + val-tiles\" candidate in `docs/split_candidates.md`, so "
         "the numbers here differ from the ones there.\n",
         f"- **train**: {', '.join(TRAIN)}",
         f"- **val-tiles**: whole spatial blocks of the train disasters, about "
         f"{args.frac:.0%} of each disaster's tiles (seed {args.seed})",
         f"- **val**: {', '.join(VAL)}",
         f"- **test**: {', '.join(TEST)}\n",
         "## Class counts per part\n",
         df_to_markdown(counts) + "\n",
         "Class mix within each part (un-classified excluded):\n",
         df_to_markdown(share, float_fmt="{:.1%}") + "\n",
         "## Blocks\n",
         f"**Block size: {args.block_deg} degrees** of longitude and latitude (about "
         f"{km:.1f} km north-south; a tile is about 0.005 degrees wide and 0.0045 tall, so a "
         "block is roughly 4 tiles across). Each tile's position comes from fitting its "
         "buildings' pixel coordinates to their lng/lat coordinates; a block is one cell of "
         "a fixed lng/lat grid. Per disaster the blocks were shuffled and walked once, "
         "holding out a block whenever that brought the held-out tile count closer to "
         f"{args.frac:.0%}.\n",
         "Two things the grid alone does not handle:\n",
         f"- **Tiles without buildings have no coordinates** ({n_empty_train:,} tiles in the "
         "train disasters, most of them socal-fire). They cannot be placed in a block, so "
         f"they all stay in train, and the {args.frac:.0%} target counts only tiles with "
         f"buildings: val-tiles is {leak.loc['all', 'share']:.1%} of the tiles with "
         f"buildings and {all_['val-tiles tiles'] / n_train_dis:.1%} of all tiles in the "
         "train disasters. Empty tiles contribute no building crops to either side.",
         f"- **Some tile footprints overlap** ({n_overlap:,} pairs in the train disasters "
         "share more than 1% of a tile's area). Grid cells holding such a pair were merged "
         "into one block, so no overlapping pair is split between train and val-tiles "
         "(last column below). A one-off check of the pre-disaster images found that "
         "overlapping tiles do not show the same picture in the shared area, which agrees "
         "with check 2 in `docs/split_candidates.md` (no building appears in two tiles): "
         "they behave like neighbours, not duplicates.\n",
         "## Remaining leak\n",
         "Two tiles *touch* when the gap between their footprints is under 5% of a tile, "
         "corner-to-corner included. A val-tiles block *touches a train block* when at "
         "least one of its tiles touches a train tile.\n",
         df_to_markdown(leak, float_fmt="{:.1%}") + "\n",
         f"**{all_['blocks touching a train block']:,} of {all_['val-tiles blocks']:,} "
         "val-tiles blocks sit next to a train block** "
         f"({all_['val-tiles tiles touching a train tile']:,} of "
         f"{all_['val-tiles tiles']:,} val-tiles tiles touch a train tile). Blocking keeps "
         "a held-out tile's neighbours inside its own block where it can, but a block's "
         "outer edge still borders train, so val-tiles remains a same-disaster score and "
         "will read higher than val. Empty train tiles are not in this count because their "
         "position is unknown.\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--folders", nargs="+", default=["train", "test"],
                    help="xBD folders to pool (their own train/test naming is ignored)")
    ap.add_argument("--out", type=Path, default=Path("splits/split_v1.csv"))
    ap.add_argument("--doc", type=Path, default=Path("docs/split_final.md"))
    ap.add_argument("--block-deg", type=float, default=BLOCK_DEG)
    ap.add_argument("--frac", type=float, default=VAL_TILES_FRAC)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = make_split(load_tiles(args.data_dir, args.folders), args.block_deg, args.frac,
                       args.seed)
    if tiles.duplicated(["disaster", "tile_id"]).any():
        raise ValueError("A tile appears in more than one xBD folder")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator: same bytes on Windows and Linux
    tiles[["disaster", "tile_id", "part"]].to_csv(args.out, index=False, lineterminator="\n")
    blocks_out = args.out.with_name(args.out.stem + "_blocks.csv")
    tiles[["disaster", "tile_id", "lng", "lat", "block"]].to_csv(
        blocks_out, index=False, lineterminator="\n", float_format="%.6f")

    df = pd.concat([load_labels(args.data_dir / f) for f in args.folders], ignore_index=True)
    df["disaster"] = df["disaster"].astype(str)
    n_overlap = 0
    for disaster in TRAIN:
        g = tiles[(tiles["disaster"] == disaster) & tiles["lng"].notna()]
        if len(g):
            n_overlap += int(np.triu(pair_geometry(g)[1]).sum())
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(report(tiles, df, args, n_overlap), encoding="utf-8")

    print(f"Wrote {args.out}, {blocks_out} and {args.doc}")
    print(tiles["part"].value_counts().reindex(PARTS).to_string())


if __name__ == "__main__":
    main()
