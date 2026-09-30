"""Compare candidate disaster-held-out splits -> docs/split_candidates.md.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\split_candidates.py --data-dir C:\\Users\\wikel\\dev\\xbd-data

Pools the train and test folders (no tile appears in both; see docs/data_quality.md) and,
for each candidate split, reports building counts per damage class in each part, the share
of each class that is held out, and the disaster types in each part. Also checks how much
a tile-level split *within* a disaster could leak. This script describes the candidates;
it does not choose one.
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

REAL = ["no-damage", "minor-damage", "major-damage", "destroyed"]
TILE_SIZE = 1024

CANDIDATES = {
    "P1": {"desc": "front-runner in docs/split_proposal.md",
           "val": ["hurricane-florence", "midwest-flooding"],
           "test": ["hurricane-michael", "santa-rosa-wildfire"]},
    "P1 + val-tiles": {"desc": "P1, plus a second val set: 10% of the training disasters' "
                               "tiles, split by tile and stratified by disaster",
                       "val": ["hurricane-florence", "midwest-flooding"],
                       "test": ["hurricane-michael", "santa-rosa-wildfire"],
                       "val_tiles_frac": 0.10},
    "P2": {"desc": "P1 with SoCal (fire) in val instead of Midwest",
           "val": ["hurricane-florence", "socal-fire"],
           "test": ["hurricane-michael", "santa-rosa-wildfire"]},
    "P3": {"desc": "a minor-damage-rich hurricane in each held-out set",
           "val": ["hurricane-michael"],
           "test": ["hurricane-matthew", "santa-rosa-wildfire"]},
}


def assign(df, tiles, cand, seed):
    """Return a Series part-name per tile (index = (disaster, tile_id))."""
    part = pd.Series("train", index=pd.MultiIndex.from_frame(tiles[["disaster", "tile_id"]]))
    dis = part.index.get_level_values("disaster")
    part[dis.isin(cand["val"])] = "val"
    part[dis.isin(cand["test"])] = "test"
    frac = cand.get("val_tiles_frac")
    if frac:
        rng = np.random.default_rng(seed)
        for d in sorted(set(dis[part.to_numpy() == "train"])):
            idx = np.flatnonzero((dis == d) & (part.to_numpy() == "train"))
            pick = rng.choice(idx, size=max(1, round(frac * len(idx))), replace=False)
            part.iloc[pick] = "val-tiles"
    return part


def describe(df, tiles, part, types):
    b = df.join(part.rename("part"), on=["disaster", "tile_id"])
    counts = pd.crosstab(b["part"], b["damage"]).reindex(columns=REAL + ["un-classified"], fill_value=0)
    counts.insert(0, "tiles", part.value_counts())
    counts["buildings"] = counts[REAL + ["un-classified"]].sum(axis=1)
    order = [p for p in ("train", "val-tiles", "val", "test") if p in counts.index]
    counts = counts.loc[order]
    held = counts.loc[[p for p in order if p != "train"], REAL].div(counts[REAL].sum()).T
    held.columns = [f"% in {c}" for c in held.columns]
    held["% held out (all but train)"] = held.sum(axis=1)
    kinds = {p: sorted({types[d] for d in part.index.get_level_values("disaster")[part.to_numpy() == p]})
             for p in order}
    names = {p: sorted(set(part.index.get_level_values("disaster")[part.to_numpy() == p])) for p in order}
    return counts, held, kinds, names


def border_proximity(df, dist=50):
    """Per disaster: tiles with a building within `dist` px of the border, and such buildings."""
    b = np.array([p.bounds for p in df["polygon"]])
    near = ((b[:, 0] < dist) | (b[:, 1] < dist) | (b[:, 2] > TILE_SIZE - dist) | (b[:, 3] > TILE_SIZE - dist))
    d = df.assign(near=near)
    per_tile = d.groupby(["disaster", "tile_id"])["near"].any()
    t = pd.DataFrame({
        "tiles with buildings": per_tile.groupby("disaster").size(),
        f"tiles with a building < {dist} px from border": per_tile.groupby("disaster").sum(),
        "buildings": d.groupby("disaster").size(),
        f"buildings < {dist} px from border": d.groupby("disaster")["near"].sum(),
    })
    t["share of tiles"] = t.iloc[:, 1] / t.iloc[:, 0]
    t["share of buildings"] = t.iloc[:, 3] / t.iloc[:, 2]
    tot = t.iloc[:, :4].sum()
    t.loc["all"] = [*tot, tot.iloc[1] / tot.iloc[0], tot.iloc[3] / tot.iloc[2]]
    for c in t.columns[:4]:
        t[c] = t[c].astype(int)
    return t


def cross_tile_duplicates(data_dir, splits, grid=1e-5):
    """Buildings whose lng/lat centroid (~1 m grid) also appears in a *different* tile.

    Direct evidence of the same building being annotated in two overlapping tiles.
    """
    cells = {}
    for split in splits:
        for p in sorted((data_dir / split / "labels").glob("*_post_disaster.json")):
            stem = p.name.replace("_post_disaster.json", "")
            disaster, tile_id = stem.rsplit("_", 1)
            with open(p, encoding="utf-8") as fh:
                feats = json.load(fh)["features"]["lng_lat"]
            for f in feats:
                c = shapely_wkt.loads(f["wkt"]).centroid
                key = (disaster, round(c.x / grid), round(c.y / grid))
                cells.setdefault(key, set()).add(tile_id)
    dup = [(k[0], len(v)) for k, v in cells.items() if len(v) > 1]
    if not dup:
        return pd.DataFrame(columns=["disaster", "locations in >1 tile"]), 0
    s = pd.DataFrame(dup, columns=["disaster", "n_tiles"]).groupby("disaster").size()
    return s.rename("locations in >1 tile").to_frame(), len(dup)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out", type=Path, default=Path("docs/split_candidates.md"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = [load_labels(args.data_dir / s) for s in args.splits]
    df = pd.concat(frames, ignore_index=True)
    df["disaster"] = df["disaster"].astype(str)
    types = df.groupby("disaster")["disaster_type"].first().to_dict()
    tiles = []
    for s in args.splits:
        for p in sorted((args.data_dir / s / "labels").glob("*_post_disaster.json")):
            d, t = p.name.replace("_post_disaster.json", "").rsplit("_", 1)
            tiles.append((d, t))
    tiles = pd.DataFrame(tiles, columns=["disaster", "tile_id"])

    L = ["# Split candidates\n",
         f"Generated by `scripts/split_candidates.py` from `{args.data_dir}` (pooled: "
         f"{', '.join(args.splits)}; {len(tiles):,} tiles, {len(df):,} buildings). "
         "Background and reasoning: `docs/split_proposal.md`. **No candidate is chosen here.**\n",
         "Every candidate holds out whole disasters for val and test. Percentages are the "
         "share of *all* buildings of that class (train + test folders pooled) that land in "
         "each part. Un-classified buildings are counted but excluded from the percentages.\n"]
    for name, cand in CANDIDATES.items():
        part = assign(df, tiles, cand, args.seed)
        counts, held, kinds, names = describe(df, tiles, part, types)
        L.append(f"## {name}\n")
        L.append(f"{cand['desc']}.\n")
        for p in counts.index:
            if p == "val-tiles":
                L.append(f"- **{p}**: {cand['val_tiles_frac']:.0%} of the tiles of each train "
                         f"disaster (seed {args.seed}); types: {', '.join(kinds[p])}")
            else:
                L.append(f"- **{p}**: {', '.join(names[p])} (types: {', '.join(kinds[p])})")
        L.append("")
        L.append(df_to_markdown(counts) + "\n")
        L.append(df_to_markdown(held, float_fmt="{:.1%}") + "\n")
        missing = sorted(set(types.values()) - set(kinds["train"]))
        note = (" Harvey and Florence were hurricanes too, but xBD labels them `flooding`."
                if "wind" in missing else "")
        L.append("Disaster types missing from train: "
                 + (", ".join(missing) if missing else "none") + "." + note + "\n")

    L.append("## Would a tile-level split within a disaster leak?\n")
    L.append("The `val-tiles` idea splits training disasters by tile, so neighbouring tiles can "
             "end up in different parts. Two checks:\n")
    L.append("**1. Buildings near the tile border.** A tile whose buildings sit close to its "
             "edge probably has neighbours showing the same street (and, for buildings cut "
             "by the edge, part of the same building).\n")
    L.append(df_to_markdown(border_proximity(df), float_fmt="{:.1%}") + "\n")
    dups, n_dup = cross_tile_duplicates(args.data_dir, args.splits)
    L.append("**2. The same building in two tiles.** Each building's longitude/latitude "
             "centroid was snapped to a ~1 m grid; a grid cell holding buildings from two "
             "different tiles of the same disaster means the tiles overlap on the ground.\n")
    if n_dup:
        L.append(f"Found **{n_dup:,}** such locations:\n")
        L.append(df_to_markdown(dups) + "\n")
    else:
        L.append("None found: no building appears whole in two tiles. This catches overlapping "
                 "tiles, but not a building cut by a shared border: its two partial outlines "
                 "have different centroids.\n")
    L.append("**What this means:** almost every tile has buildings near its border, so most "
             "tiles have neighbours with shared context (same streets, same kind of damage; "
             "damage is spatially clustered, see `docs/observations.md`). " +
             ("No building appears whole in two tiles, so tiles don't seem to overlap, but "
              if not n_dup else
              "Some tiles do overlap, so the same building can appear in two parts, and ") +
             "a val-tiles score will likely read optimistic compared with a new disaster. That "
             "is the point of reporting it next to the disaster-held-out val set: the gap "
             "between the two measures how much the model relies on familiar context. "
             "Grouping neighbouring tiles together (e.g. by lng/lat blocks) before splitting "
             "would reduce the leak.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
