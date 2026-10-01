"""Cut every building out of its tile once and save the crops -> <data-dir>/crops/.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\build_crops.py --data-dir C:\\Users\\wikel\\dev\\xbd-data

Why a cache? Training reads every building many times (once per epoch). Opening two
1024x1024 PNGs, cutting a square and resizing it, 135,000 times per epoch, would make the
disk and the CPU the bottleneck while the GPU waits. So we do that work once and store the
result as plain arrays that can be read back almost instantly.

What is written (per part: train, val-tiles, val):
    <part>_pre.npy     uint8 array [N, 128, 128, 3]: crops from the pre-disaster image
    <part>_post.npy    same buildings, same windows, from the post-disaster image
    index.csv          one row per building, all parts together; ``row`` is the building's
                       position inside its part's arrays
    meta.json          crop settings and counts

uint8 (one byte per colour value, 0-255) is what the PNGs hold anyway; storing floats
would take four times the space for no extra information.

The test part is NOT built unless --final is given (see src/split.py). Un-classified
buildings are dropped and invalid polygons repaired (docs/data_decisions.md).
The output lives next to the data, outside the repo: it is large and can be regenerated.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crops import OUT, PAD, at_edge, crop_window, cut_crop  # noqa: E402
from src.labels import DAMAGE_LEVEL  # noqa: E402
from src.split import DEFAULT_SPLIT, DEV_PARTS, load_part_labels, require_final  # noqa: E402

INDEX_COLUMNS = ["row", "uid", "disaster", "part", "tile_id", "label", "damage",
                 "box_w", "box_h", "box_side", "at_edge", "centroid_x", "centroid_y",
                 "win_left", "win_top", "win_side"]


def build_index(labels, pad):
    """One row per building: where it is, how big it is, and the crop window to cut."""
    rows = []
    for r in labels.itertuples():
        bounds = r.polygon.bounds
        left, top, right, _ = crop_window(bounds, pad)
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        rows.append({
            "uid": r.uid, "disaster": r.disaster, "part": r.part, "tile_id": r.tile_id,
            "label": DAMAGE_LEVEL[r.damage], "damage": r.damage,
            "box_w": round(w, 2), "box_h": round(h, 2), "box_side": round(max(w, h), 2),
            "at_edge": at_edge(bounds),
            "centroid_x": round(r.centroid_x, 2), "centroid_y": round(r.centroid_y, 2),
            "win_left": left, "win_top": top, "win_side": right - left,
            "pre_image": r.pre_image, "post_image": r.post_image,
        })
    index = pd.DataFrame(rows)
    # Keep each tile's buildings together and each part in one block, so that ``row``
    # can simply count up within a part.
    index = index.sort_values(["part", "disaster", "tile_id"], kind="stable", ignore_index=True)
    index.insert(0, "row", index.groupby("part").cumcount())
    return index


def crop_tile(task):
    """Worker: open one tile's two images and cut all of its buildings. Runs in a pool."""
    pre_path, post_path, windows, out = task
    crops = []
    for path in (pre_path, post_path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            crops.append(np.stack([cut_crop(im, w, out) for w in windows]))
    return crops  # [pre crops, post crops], each [n, out, out, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--out", type=Path, default=None, help="default: <data-dir>/crops")
    ap.add_argument("--parts", nargs="+", default=list(DEV_PARTS))
    ap.add_argument("--final", action="store_true",
                    help="required to build crops for the test part; never during development")
    ap.add_argument("--size", type=int, default=OUT)
    ap.add_argument("--pad", type=int, default=PAD)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    parts = require_final(args.parts, args.final)   # raises if "test" is asked without --final
    out_dir = args.out or args.data_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    labels = load_part_labels(args.data_dir, parts, args.split, final=args.final,
                              drop_unclassified=True, repair=True)
    index = build_index(labels, args.pad)
    print(f"{len(index):,} buildings in parts {parts} ({time.time() - t0:.0f}s to read labels)")

    for part in parts:
        sub = index[index["part"] == part]
        if sub.empty:
            continue
        shape = (len(sub), args.size, args.size, 3)
        # open_memmap creates the .npy file on disk at its full size and lets us fill it
        # piece by piece, so the whole array never has to sit in memory.
        pre = np.lib.format.open_memmap(out_dir / f"{part}_pre.npy", mode="w+",
                                        dtype=np.uint8, shape=shape)
        post = np.lib.format.open_memmap(out_dir / f"{part}_post.npy", mode="w+",
                                         dtype=np.uint8, shape=shape)
        tasks, starts = [], []
        for _, tile in sub.groupby(["disaster", "tile_id"], sort=False):
            windows = [(r.win_left, r.win_top, r.win_left + r.win_side, r.win_top + r.win_side)
                       for r in tile.itertuples()]
            tasks.append((tile["pre_image"].iloc[0], tile["post_image"].iloc[0], windows,
                          args.size))
            starts.append(int(tile["row"].iloc[0]))
        # Decoding PNGs is the slow part and each tile is independent, so spread the tiles
        # over several processes. ``map`` returns results in task order.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for start, (pre_c, post_c) in zip(starts, pool.map(crop_tile, tasks, chunksize=4)):
                pre[start:start + len(pre_c)] = pre_c
                post[start:start + len(post_c)] = post_c
        pre.flush()
        post.flush()
        del pre, post
        print(f"  {part}: {len(sub):,} buildings from {len(tasks):,} tiles "
              f"({time.time() - t0:.0f}s elapsed)")

    # If only some parts were rebuilt, keep the other parts' rows in the index.
    index_path = out_dir / "index.csv"
    index = index[INDEX_COLUMNS]
    if index_path.exists():
        old = pd.read_csv(index_path, dtype={"tile_id": str})
        index = pd.concat([old[~old["part"].isin(parts)], index], ignore_index=True)
    index.to_csv(index_path, index=False, lineterminator="\n")
    counts = index.groupby("part")["label"].value_counts().unstack(fill_value=0)
    meta = {"size": args.size, "pad": args.pad, "split": str(args.split),
            "buildings": {p: int(n) for p, n in index["part"].value_counts().items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_gb = sum(f.stat().st_size for f in out_dir.iterdir()) / 2**30
    print(f"Wrote {out_dir}  ({size_gb:.2f} GB on disk, {time.time() - t0:.0f}s)")
    print("Buildings per part and label (0 no-damage .. 3 destroyed):")
    print(counts.to_string())


if __name__ == "__main__":
    main()
