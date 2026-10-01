"""Data-quality audit of the xBD train and test folders -> docs/data_quality.md.



Usage:

    .\\.venv\\Scripts\\python.exe scripts\\audit_data.py --data-dir C:\\Users\\wikel\\dev\\xbd-data



--data-dir is the folder holding the split folders (train/ and test/ by default; see

--splits). The script only reads the data. It writes the "audit" section of

docs/data_quality.md (other sections, e.g. the crop-size analysis, are left alone) and a

per-tile CSV of the pre/post shift estimates to outputs/audit/.

"""



import argparse

import json

import sys

from pathlib import Path



import numpy as np

import pandas as pd

from PIL import Image

from shapely import wkt as shapely_wkt

from shapely.validation import explain_validity, make_valid



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import estimate_shift  # noqa: E402

from src.labels import DAMAGE_CLASSES, load_labels  # noqa: E402

from src.reporting import df_to_markdown, write_section  # noqa: E402



TILE_SIZE = 1024       # every xBD image is 1024 x 1024 (see outputs/data_report.md)

EDGE_TOL = 1.0         # a polygon within 1 px of the border counts as touching it

MAX_SHIFT = 16         # patch shifts are searched within +-16 px





def load_all(data_dir, splits):

    frames = []

    for split in splits:

        df = load_labels(data_dir / split, repair=False)  # the audit reports the raw outlines

        df.insert(0, "split", split)

        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    df["disaster"] = df["disaster"].astype(str)

    return df





def tile_table(data_dir, splits):

    """One row per tile (including tiles with no buildings), from the post label files."""

    rows = []

    for split in splits:

        for p in sorted((data_dir / split / "labels").glob("*_post_disaster.json")):

            stem = p.name.replace("_post_disaster.json", "")

            disaster, tile_id = stem.rsplit("_", 1)

            rows.append({"split": split, "disaster": disaster, "tile_id": tile_id,

                         "pre_image": str(data_dir / split / "images" / f"{stem}_pre_disaster.png"),

                         "post_image": str(data_dir / split / "images" / f"{stem}_post_disaster.png"),

                         "pre_label": str(p).replace("_post_disaster.json", "_pre_disaster.json"),

                         "post_label": str(p)})

    return pd.DataFrame(rows)





# --- individual checks -------------------------------------------------------------------



def invalid_polygons(df):

    rows = []

    for _, r in df[~df["polygon"].map(lambda p: p.is_valid)].iterrows():

        fixed = make_valid(r["polygon"])

        buffered = r["polygon"].buffer(0)

        rows.append({

            "split": r["split"], "tile": f"{r['disaster']}_{r['tile_id']}", "uid": r["uid"],

            "damage": r["damage"], "problem": explain_validity(r["polygon"]),

            "vertices": len(r["polygon"].exterior.coords), "area_px": r["area_px"],

            "make_valid": f"{fixed.geom_type}, area {fixed.area:.1f}",

            "buffer(0)": f"{buffered.geom_type}, area {buffered.area:.1f}",

        })

    return pd.DataFrame(rows)





def out_of_bounds(df):

    b = np.array([p.bounds for p in df["polygon"]])  # minx, miny, maxx, maxy

    return {"min_x": b[:, 0].min(), "min_y": b[:, 1].min(), "max_x": b[:, 2].max(), "max_y": b[:, 3].max(),

            "n_outside": int(((b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > TILE_SIZE) | (b[:, 3] > TILE_SIZE)).sum())}





def unclassified_by_disaster(df):

    t = pd.crosstab(df["disaster"], df["split"], values=(df["damage"] == "un-classified"), aggfunc="sum")

    t = t.fillna(0).astype(int)

    t["total"] = t.sum(axis=1)

    t["share of buildings"] = t["total"] / df.groupby("disaster").size()

    t.loc["all"] = [*t[[c for c in t.columns if c != "share of buildings"]].sum(),

                    t["total"].sum() / len(df)]

    t[[c for c in t.columns if c != "share of buildings"]] = t[[c for c in t.columns if c != "share of buildings"]].astype(int)

    return t.sort_values("total", ascending=False)





def tiny_buildings(df):

    pct = df.groupby("damage", observed=False)["area_px"].quantile([0.01, 0.05, 0.25, 0.5]).unstack()

    pct.columns = [f"p{int(q * 100)}" for q in pct.columns]

    for thr in (10, 25, 50):

        pct[f"< {thr} pxÂ²"] = df.groupby("damage", observed=False)["area_px"].apply(lambda s, t=thr: int((s < t).sum()))

    overall = {f"p{int(q * 100)}": df["area_px"].quantile(q) for q in (0.01, 0.05, 0.25, 0.5)}

    overall.update({f"< {thr} pxÂ²": int((df["area_px"] < thr).sum()) for thr in (10, 25, 50)})

    pct.loc["all"] = overall

    for thr in (10, 25, 50):

        pct[f"< {thr} pxÂ²"] = pct[f"< {thr} pxÂ²"].astype(int)

    return pct





def edge_buildings(df):

    b = np.array([p.bounds for p in df["polygon"]])

    touches = ((b[:, 0] <= EDGE_TOL) | (b[:, 1] <= EDGE_TOL) |

               (b[:, 2] >= TILE_SIZE - EDGE_TOL) | (b[:, 3] >= TILE_SIZE - EDGE_TOL))

    df = df.assign(at_edge=touches)

    t = df.groupby("disaster").agg(buildings=("at_edge", "size"), at_edge=("at_edge", "sum"))

    t["share"] = t["at_edge"] / t["buildings"]

    t.loc["all"] = [len(df), int(touches.sum()), touches.mean()]

    t[["buildings", "at_edge"]] = t[["buildings", "at_edge"]].astype(int)

    by_class = df.groupby("damage", observed=False)["at_edge"].mean()

    # Edge buildings are cut off, so their visible area is smaller than the full footprint.

    med_area = df.groupby("at_edge")["area_px"].median()

    return t.sort_values("share", ascending=False), by_class, med_area





def pre_post_consistency(tiles):

    """Compare the pre and post label files of every tile, building by building (by uid).



    Polygon text can't be compared directly: the same outline may be written with different

    float rounding or start at a different vertex. So each shared building gets the

    Hausdorff distance (the largest distance from any point of one outline to the other) and

    the IoU of the two polygons, plus the centroid offset (post minus pre).

    """

    per_tile, per_building = [], []

    for t in tiles.itertuples(index=False):

        with open(t.pre_label, encoding="utf-8") as fh:

            pre = {f["properties"]["uid"]: f["wkt"] for f in json.load(fh)["features"]["xy"]}

        with open(t.post_label, encoding="utf-8") as fh:

            post = {f["properties"]["uid"]: f["wkt"] for f in json.load(fh)["features"]["xy"]}

        shared = pre.keys() & post.keys()

        for u in shared:

            a, b = shapely_wkt.loads(pre[u]), shapely_wkt.loads(post[u])

            # buffer(0) repairs invalid outlines, which would otherwise make union() fail

            a, b = (g if g.is_valid else g.buffer(0) for g in (a, b))

            union = a.union(b).area

            per_building.append({"disaster": t.disaster, "hausdorff": a.hausdorff_distance(b),

                                 "iou": a.intersection(b).area / union if union else 1.0,

                                 "cdx": b.centroid.x - a.centroid.x, "cdy": b.centroid.y - a.centroid.y})

        per_tile.append({"split": t.split, "disaster": t.disaster, "tile_id": t.tile_id,

                         "n_pre": len(pre), "n_post": len(post), "only_pre": len(pre.keys() - post.keys()),

                         "only_post": len(post.keys() - pre.keys()), "shared": len(shared)})

    return pd.DataFrame(per_tile), pd.DataFrame(per_building)





def split_overlap(tiles):

    keys = {s: set(zip(g["disaster"], g["tile_id"])) for s, g in tiles.groupby("split")}

    splits = sorted(keys)

    return {(a, b): len(keys[a] & keys[b]) for i, a in enumerate(splits) for b in splits[i + 1:]}





def misalignment(tiles, per_disaster, seed):

    rng = np.random.default_rng(seed)

    parts = []

    for disaster, g in tiles.groupby("disaster"):

        idx = rng.choice(len(g), size=min(per_disaster, len(g)), replace=False)

        parts.append(g.iloc[np.sort(idx)])

    sample = pd.concat(parts, ignore_index=True)

    rows = []

    for i, t in enumerate(sample.itertuples(index=False), start=1):

        pre = np.asarray(Image.open(t.pre_image).convert("RGB"))

        post = np.asarray(Image.open(t.post_image).convert("RGB"))

        est = estimate_shift(pre, post, max_shift=MAX_SHIFT)

        rows.append({"split": t.split, "disaster": t.disaster, "tile_id": t.tile_id,
                     "pre_image": t.pre_image, "post_image": t.post_image, **est})

        if i % 50 == 0:

            print(f"    shift estimates: {i}/{len(sample)}")

    return pd.DataFrame(rows)





def misalignment_null(sample, n, offset=7):
    """Null control: run the same estimator on UNRELATED image pairs.

    Each pre image is paired with the post image of a different sampled tile, so there is no
    true shift to find. Whatever the estimator reports here is its noise floor.
    """
    sample = sample.reset_index(drop=True)
    idx = np.linspace(0, len(sample) - 1, min(n, len(sample))).astype(int)
    rows = []
    for i in idx:
        t, u = sample.iloc[i], sample.iloc[(i + offset) % len(sample)]
        pre = np.asarray(Image.open(t["pre_image"]).convert("RGB"))
        post = np.asarray(Image.open(u["post_image"]).convert("RGB"))
        est = estimate_shift(pre, post, max_shift=MAX_SHIFT)
        rows.append({"magnitude": est["magnitude"], "patch_agreement": est["patch_agreement"]})
    return pd.DataFrame(rows)


def misalignment_summary(shifts):

    g = shifts.groupby("disaster")

    over = f"global est. > {MAX_SHIFT} px"

    t = pd.DataFrame({

        "sampled": g.size(),

        "median px": g["magnitude"].median(),

        "p90 px": g["magnitude"].quantile(0.9),

        "max px": g["magnitude"].max(),

        "share > 2 px": g["magnitude"].apply(lambda s: (s > 2).mean()),

        "share > 5 px": g["magnitude"].apply(lambda s: (s > 5).mean()),

        "median patch agreement": g["patch_agreement"].median(),

        "global est. median px": g["global_magnitude"].median(),

        over: g["global_magnitude"].apply(lambda s: int((s > MAX_SHIFT).sum())),

    })

    m, gm = shifts["magnitude"], shifts["global_magnitude"]

    t.loc["all"] = [len(shifts), m.median(), m.quantile(0.9), m.max(), (m > 2).mean(), (m > 5).mean(),

                    shifts["patch_agreement"].median(), gm.median(), int((gm > MAX_SHIFT).sum())]

    t["sampled"] = t["sampled"].astype(int)

    t[over] = t[over].astype(int)

    return t





# --- report ------------------------------------------------------------------------------



def build_report(args, df, tiles, results):

    inv, oob, uncls, tiny, (edge, edge_cls, edge_area), (pp, pb), overlap, shifts, null = results

    shift_tab = misalignment_summary(shifts)

    L = []

    L.append("# Data-quality audit\n")

    L.append(f"Generated by `scripts/audit_data.py` from `{args.data_dir}` "

             f"(splits: {', '.join(args.splits)}). Regenerate rather than editing the "

             "numbers by hand. Findings are stated with what they mean for modelling; "

             "no modelling decisions are made here.\n")

    L.append(f"Scope: **{len(tiles):,} tiles**, **{len(df):,} buildings** "

             f"({', '.join(f'{s}: {(df.split == s).sum():,}' for s in args.splits)}).\n")



    L.append("## 1. Invalid polygons\n")

    if len(inv):

        L.append(df_to_markdown(inv, float_fmt="{:,.1f}", index=False) + "\n")

        L.append("`problem` is shapely's `explain_validity`. A self-intersection means the "

                 "outline crosses itself (a 'bow-tie'), usually a digitising slip where a "

                 "vertex was placed on the wrong side of an edge. `make_valid` and `buffer(0)` "

                 "are the two standard repairs; their resulting type and area are shown.\n")

        L.append("**For modelling:** one polygon out of the whole set can't move any metric, "

                 "but it can crash code that assumes valid geometry (area, intersection, "

                 "rasterising with some libraries). Suggested handling: repair with "

                 "`buffer(0)` (keeps a single Polygon when the area is unchanged) or "

                 "`make_valid` and keep the largest polygon part, or drop that one building. "

                 "Whichever is chosen, do it in one place (the label loader) so every "

                 "downstream step sees the same geometry.\n")

    else:

        L.append("None found.\n")

    L.append(f"Coordinate range over all polygons: x {oob['min_x']:.2f} to {oob['max_x']:.2f}, "

             f"y {oob['min_y']:.2f} to {oob['max_y']:.2f}; {oob['n_outside']:,} polygons extend "

             f"slightly outside the 0-{TILE_SIZE} image. The overshoot is under half a pixel, so "

             "clipping to the image bounds changes nothing visible, but code that indexes "

             "pixels from polygon coordinates should clip first.\n")



    L.append("## 2. Un-classified buildings\n")

    L.append(df_to_markdown(uncls, float_fmt="{:.1%}") + "\n")

    L.append("In the official target masks these buildings are painted as background (0), "

             "not as a separate class (checked in `outputs/data_report.md`).\n")

    L.append("**For modelling:** they are buildings with no usable label. Options are to drop "

             "them, to ignore them in the loss (common for segmentation: an 'ignore' index), "

             "or to keep them only for building-vs-background tasks. Treating them as "

             "background, as the official masks do, teaches a segmentation model that some "

             "buildings aren't buildings. Guatemala stands out, with a far higher share than any "

             "other disaster, so any Guatemala result depends heavily on this choice.\n")



    L.append("## 3. Tiny buildings\n")

    L.append("Footprint area in pixels (pxÂ²), by damage class. `pN` = Nth percentile.\n")

    L.append(df_to_markdown(tiny, float_fmt="{:,.1f}") + "\n")

    L.append("**For modelling:** at xBD's resolution a building under ~50 pxÂ² is only a few "

             "pixels across, which leaves very little to classify, especially after "

             "downsampling. These buildings also make up a meaningful part of the smallest "

             "classes' tail. Filtering by a size threshold, upweighting, or accepting that they "

             "will be hard are all possible; the choice affects per-class recall directly.\n")



    L.append("## 4. Buildings cut off at the tile edge\n")

    L.append(f"A building counts as 'at the edge' if its bounding box is within {EDGE_TOL:g} px "

             "of the image border, i.e. the footprint is probably cut off.\n")

    L.append(df_to_markdown(edge, float_fmt="{:.1%}") + "\n")

    L.append("Share at the edge by damage class: "

             + ", ".join(f"{c} {v:.1%}" for c, v in edge_cls.items()) + ". "

             f"Median area of edge buildings {edge_area.get(True, np.nan):,.0f} pxÂ² vs "

             f"{edge_area.get(False, np.nan):,.0f} pxÂ² for the rest.\n")

    L.append("**For modelling:** a building crop centred on an edge building will be partly "

             "empty (padding), and the model sees only part of the footprint. For a "

             "per-building classifier this is a source of hard, possibly misleading examples. "

             "For tile-level segmentation it matters less. The same building may also appear in "

             "a neighbouring tile, which is relevant to tile-level splits (see "

             "`docs/split_candidates.md`).\n")



    L.append("## 5. Pre vs post label consistency\n")

    mism_count = int((pp["n_pre"] != pp["n_post"]).sum())

    h = pb["hausdorff"]

    L.append(f"- Tiles compared: {len(pp):,}. Tiles where pre and post list a different "

             f"number of buildings: **{mism_count:,}**. Buildings only in the pre file: "

             f"{pp['only_pre'].sum():,}; only in the post file: {pp['only_post'].sum():,}; "

             f"in both (same uid): {pp['shared'].sum():,}.")

    L.append("- The polygon text is never byte-identical (float rounding and starting vertex "

             "differ), so outlines are compared as shapes: the Hausdorff distance between the "

             "pre and post outline of each building.\n")

    bands = [(0, 0.01), (0.01, 0.5), (0.5, 2), (2, 5), (5, np.inf)]

    band_tab = pd.DataFrame({

        "buildings": [int(((h >= lo) & (h < hi)).sum()) for lo, hi in bands],

        "share": [((h >= lo) & (h < hi)).mean() for lo, hi in bands],

        "median IoU": [pb.loc[(h >= lo) & (h < hi), "iou"].median() for lo, hi in bands],

    }, index=pd.Index([f"{lo:g} to {hi:g} px" if np.isfinite(hi) else f">= {lo:g} px" for lo, hi in bands],

                      name="outline difference"))

    L.append(df_to_markdown(band_tab, float_fmt="{:.3f}") + "\n")

    moved = pb[h >= 0.5]

    per_dis = pb.assign(m=h >= 0.5).groupby("disaster")["m"].mean().sort_values(ascending=False)

    L.append(f"For the {len(moved):,} buildings whose outlines differ by 0.5 px or more, the "

             f"centroid offset (post minus pre) has mean ({moved['cdx'].mean():+.2f}, "

             f"{moved['cdy'].mean():+.2f}) px in (x, y), with standard deviation "

             f"({moved['cdx'].std():.2f}, {moved['cdy'].std():.2f}). There is no consistent "

             "direction, so these look like small independent edits to individual outlines, "

             "not the post polygons being moved to follow the post image. Share differing by "

             ">= 0.5 px, per disaster: " + ", ".join(f"{d} {v:.1%}" for d, v in per_dis.items()) + ".\n")

    same = (h < 0.01).mean()

    L.append(f"**For modelling:** both files list the same buildings. For {same:.0%} of them the "

             "outlines are identical, and almost all the rest differ by about a pixel. So "

             "either file's polygons can be used for both images with very little difference "

             "(the loader currently uses the post polygons). The labels do not correct the "

             "image misalignment in section 6, since the outline isn't redrawn to follow the "

             "post image. The low IoUs in the table come from small buildings, where a 1 px "

             "edit is a big share of the area.\n")

    L.append("Train/test tile overlap (same disaster and tile id in two splits): "

             + ", ".join(f"{a}/{b}: {n}" for (a, b), n in overlap.items()) + ". "

             + ("No tile appears in both, so pooling the two folders (as the split proposal "

                "does) doesn't duplicate any tile." if all(n == 0 for n in overlap.values())

                else "**Some tiles appear in both; pooling would duplicate them.**") + "\n")



    L.append("## 6. Pre/post image misalignment\n")

    L.append(f"Estimated for {len(shifts)} tiles: {args.shift_per_disaster} per disaster, sampled "

             f"at random (seed {args.seed}) from all tiles, including ones without buildings. "

             "Method (`src/alignment.py`): each tile is split into overlapping 128x128 patches; "

             "patches with no-data or no texture are skipped; each patch's shift is found with "

             f"low-pass phase correlation, searched within +-{MAX_SHIFT} px; the tile's shift is "

             "the median over its patches. On synthetic tests the method recovers known "

             "shifts exactly (`tests/test_alignment.py`).\n")

    L.append(df_to_markdown(shift_tab, float_fmt="{:.2f}") + "\n")

    L.append("`median/p90/max px` = size of the tile's (median-of-patches) shift. "

             "`median patch agreement` = share of a tile's patches within 1.5 px of the tile's "

             "shift. `global est.` = a single phase correlation over the whole tile with no "

             "search limit, as a cross-check; the last column counts tiles where it lands "

             f"beyond +-{MAX_SHIFT} px.\n")

    real_m, null_m = shifts["magnitude"], null["magnitude"]
    real_a, null_a = shifts["patch_agreement"], null["patch_agreement"]
    ctrl = pd.DataFrame({
        "median shift px": [real_m.median(), null_m.median()],
        "p75 shift px": [real_m.quantile(0.75), null_m.quantile(0.75)],
        "p90 shift px": [real_m.quantile(0.9), null_m.quantile(0.9)],
        "median patch agreement": [real_a.median(), null_a.median()],
    }, index=pd.Index([f"real pre/post pairs (n={len(shifts)})",
                       f"unrelated pairs, null control (n={len(null)})"], name="input"))
    L.append("**Null control: how much of this is noise?** The same estimator was run on "
             f"{len(null)} *unrelated* pairs (one tile's pre image with a different tile's post "
             "image), where there is no true shift to find. Whatever it reports there is the "
             "method's noise floor:\n")
    L.append(df_to_markdown(ctrl, float_fmt="{:.2f}") + "\n")
    L.append("**How far to trust this.** The shift sizes for real pairs are barely above the "
             "null control, so **this method cannot measure the misalignment**; the "
             "per-disaster 'median px' values above are mostly its noise floor, not a real "
             "offset. What the data does support:\n")
    L.append(f"- there *is* some shared structure: patch agreement is higher on real pairs "
             f"({real_a.median():.2f}) than on unrelated ones ({null_a.median():.2f}), so the "
             "images are related, but most patches still don't agree with each other;")
    L.append("- the tile-level estimates are not large: they stay within a few pixels of the "
             "noise floor, and none of the other methods tried during development (whole-tile "
             "phase correlation on intensity and gradient images, with and without low-pass "
             "filtering) gave consistent large shifts either. The unbounded whole-tile "
             f"estimates beyond +-{MAX_SHIFT} px in the table are spurious peaks, and they "
             "don't agree with the patch estimates for the same tile.\n")
    L.append("Why most patches disagree can't be settled from this: the shift could vary across "
             "a tile (parallax, since the two captures were taken from different viewing "
             "angles and taller buildings move more than the ground), or the two captures could "
             "simply differ too much (lighting, season, sensor, damage) for patch matching to "
             "work. A more direct measurement would need hand-marked matching points on a few "
             "tiles, or a feature-matching method (e.g. SIFT/ORB with outlier rejection), which "
             "needs a library not currently installed (OpenCV). Per-tile values: "
             "`outputs/audit/shift_estimates.csv` and `outputs/audit/shift_null_control.csv`.\n")
    L.append("**For modelling:** misalignment between pre and post images is unmeasured, not "
             "known to be zero. If it's a few pixels, it's small next to a typical building "
             "(median footprint ~650 px², roughly 25x25 px) but large for the tiniest ones "
             "(section 3). Inputs are affected differently:\n")
    L.append("- crops cut tightly to the polygon may clip the post-image building; context "
             "padding reduces that (see the crop-size section);")
    L.append("- a pixel-difference input (post minus pre) turns any misalignment into false "
             "'change' along edges, even for undamaged buildings;")
    L.append("- channel stacking and siamese features are more tolerant of a few pixels of "
             "shift, because convolution and pooling blur small offsets.\n")
    L.append("Visual inspection of the tile figures in `outputs/viz/` is currently the best "
             "check available: outlines drawn from the same polygon on both images can be "
             "compared by eye against the buildings.\n")
    return "\n".join(L)





def main():

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--data-dir", required=True, type=Path, help="folder holding the split folders")

    ap.add_argument("--splits", nargs="+", default=["train", "test"])

    ap.add_argument("--doc", type=Path, default=Path("docs/data_quality.md"))

    ap.add_argument("--csv-dir", type=Path, default=Path("outputs/audit"))

    ap.add_argument("--shift-per-disaster", type=int, default=20,

                    help="tiles per disaster for the misalignment estimate (10 disasters -> 200)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-pairs", type=int, default=50,
                    help="unrelated image pairs for the misalignment noise-floor control")

    args = ap.parse_args()



    print("Loading labels ...")

    df = load_all(args.data_dir, args.splits)

    tiles = tile_table(args.data_dir, args.splits)

    print(f"  {len(df):,} buildings, {len(tiles):,} tiles")



    print("Checking geometry, classes, sizes, edges ...")

    inv, oob = invalid_polygons(df), out_of_bounds(df)

    uncls, tiny, edge = unclassified_by_disaster(df), tiny_buildings(df), edge_buildings(df)

    print("Comparing pre and post label files ...")

    pp = pre_post_consistency(tiles)

    overlap = split_overlap(tiles)

    print("Estimating pre/post shifts ...")

    shifts = misalignment(tiles, args.shift_per_disaster, args.seed)
    print("Null control on unrelated pairs ...")
    null = misalignment_null(shifts, args.null_pairs)



    args.csv_dir.mkdir(parents=True, exist_ok=True)

    shifts.drop(columns=["pre_image", "post_image"]).to_csv(args.csv_dir / "shift_estimates.csv", index=False)
    null.to_csv(args.csv_dir / "shift_null_control.csv", index=False)

    pp[0].to_csv(args.csv_dir / "pre_post_consistency.csv", index=False)



    report = build_report(args, df, tiles, (inv, oob, uncls, tiny, edge, pp, overlap, shifts, null))

    write_section(args.doc, "audit", report)

    print(f"Wrote the 'audit' section of {args.doc}")





if __name__ == "__main__":

    main()

