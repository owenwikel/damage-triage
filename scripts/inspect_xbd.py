"""Walk an xBD split folder and write a plain-language report of what is in it.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\inspect_xbd.py --data-root C:\\Users\\wikel\\dev\\xbd-data\\test

The data root must contain images/, labels/ and targets/ subfolders.
The report is written to outputs/data_report.md (override with --out).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# xBD file names look like: socal-fire_00000123_post_disaster.png
NAME_RE = re.compile(r"^(?P<disaster>.+)_(?P<tile>\d+)_(?P<phase>pre|post)_disaster")


def parse_name(path):
    """Return (disaster, tile_id, phase) from an xBD file name, or None if it doesn't match."""
    m = NAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("disaster"), m.group("tile"), m.group("phase")


def count_by_disaster_phase(files):
    """Count files per (disaster, phase). Returns {disaster: Counter({'pre': n, 'post': n})}."""
    counts = defaultdict(Counter)
    for f in files:
        parsed = parse_name(f)
        if parsed:
            disaster, _, phase = parsed
            counts[disaster][phase] += 1
    return counts


def tile_keys(files):
    """Set of (disaster, tile_id, phase) keys, used to spot missing pairs."""
    return {k for k in (parse_name(f) for f in files) if k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/data_report.md"))
    ap.add_argument("--target-sample", type=int, default=50,
                    help="how many target PNGs to open when checking their pixel values")
    args = ap.parse_args()

    root = args.data_root
    dirs = {name: root / name for name in ("images", "labels", "targets")}
    for name, d in dirs.items():
        if not d.is_dir():
            raise SystemExit(f"Missing folder: {d}")

    images = sorted(dirs["images"].glob("*.png"))
    labels = sorted(dirs["labels"].glob("*.json"))
    targets = sorted(dirs["targets"].glob("*.png"))
    other = [p for d in dirs.values() for p in d.iterdir()
             if p.suffix.lower() not in (".png", ".json")]
    unparsed = [p for p in images + labels + targets if parse_name(p) is None]

    # --- Pairing checks: every image should have a label and a target, and every pre a post.
    img_k, lab_k, tgt_k = tile_keys(images), tile_keys(labels), tile_keys(targets)
    img_no_label = sorted(img_k - lab_k)
    img_no_target = sorted(img_k - tgt_k)
    tiles = {(d, t) for d, t, _ in img_k}
    pre_no_post = sorted(t for t in tiles if (t[0], t[1], "post") not in img_k)
    post_no_pre = sorted(t for t in tiles if (t[0], t[1], "pre") not in img_k)

    # --- Image sizes / modes. Image.open only reads the header, so this is fast.
    sizes, modes = Counter(), Counter()
    for p in images:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1

    # --- Label JSON contents.
    top_keys, feature_keys, prop_keys, meta_keys = Counter(), Counter(), Counter(), Counter()
    feature_types = Counter()
    subtype_by_phase = defaultdict(Counter)          # phase -> Counter of subtype strings
    subtype_by_disaster = defaultdict(Counter)       # disaster -> Counter (post only)
    disaster_types = {}
    geom_types = Counter()
    lnglat_xy_mismatch = 0
    sample_label = None

    for p in labels:
        disaster, tile, phase = parse_name(p)
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        top_keys.update(d.keys())
        feature_keys.update(d["features"].keys())
        meta_keys.update(d["metadata"].keys())
        disaster_types[d["metadata"].get("disaster", disaster)] = d["metadata"].get("disaster_type")
        xy = d["features"]["xy"]
        if len(xy) != len(d["features"].get("lng_lat", [])):
            lnglat_xy_mismatch += 1
        for feat in xy:
            props = feat["properties"]
            prop_keys.update(props.keys())
            feature_types[props.get("feature_type")] += 1
            sub = props.get("subtype", "<no subtype key>")
            subtype_by_phase[phase][sub] += 1
            if phase == "post":
                subtype_by_disaster[disaster][sub] += 1
            geom_types[feat["wkt"].split(" ", 1)[0]] += 1
        if sample_label is None and phase == "post" and xy:
            sample_label = (p.name, d["metadata"], xy[0])

    # --- Target masks: which pixel values appear? (sampled, because decoding is slower)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(targets), size=min(args.target_sample, len(targets)), replace=False)
    target_values = {"pre": Counter(), "post": Counter()}
    target_modes = Counter()
    for i in sample_idx:
        p = targets[i]
        _, _, phase = parse_name(p)
        with Image.open(p) as im:
            target_modes[im.mode] += 1
            arr = np.asarray(im)
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            target_values[phase][v] += c

    # --- Write the report.
    L = []
    L.append("# xBD data report\n")
    L.append(f"Data root: `{root}`\n")
    L.append("## File counts\n")
    L.append("| folder | files | pre | post |")
    L.append("|---|---:|---:|---:|")
    for name, files in (("images", images), ("labels", labels), ("targets", targets)):
        ph = Counter(parse_name(f)[2] for f in files if parse_name(f))
        L.append(f"| {name} | {len(files)} | {ph['pre']} | {ph['post']} |")
    L.append("")
    L.append(f"- Unique tiles (disaster + tile id): **{len(tiles)}**")
    L.append(f"- Files not matching the `<disaster>_<tile>_<pre|post>_disaster` pattern: {len(unparsed)}")
    L.append(f"- Non-PNG/JSON files in the three folders: {len(other)}")
    L.append(f"- Images without a label file: {len(img_no_label)}")
    L.append(f"- Images without a target file: {len(img_no_target)}")
    L.append(f"- Tiles with pre but no post image: {len(pre_no_post)}")
    L.append(f"- Tiles with post but no pre image: {len(post_no_pre)}")
    for title, items in (("Unmatched names", unparsed), ("Other files", other),
                         ("Missing labels", img_no_label), ("Missing targets", img_no_target),
                         ("Missing post", pre_no_post), ("Missing pre", post_no_pre)):
        if items:
            L.append(f"\n{title} (first 20):\n")
            L.extend(f"- `{x}`" for x in items[:20])
    L.append("")

    L.append("## Images per disaster\n")
    L.append("| disaster | disaster_type | pre images | post images |")
    L.append("|---|---|---:|---:|")
    img_counts = count_by_disaster_phase(images)
    for dis in sorted(img_counts):
        c = img_counts[dis]
        L.append(f"| {dis} | {disaster_types.get(dis, '?')} | {c['pre']} | {c['post']} |")
    L.append("")

    L.append("## Image format\n")
    L.append("| size (w x h) | images |")
    L.append("|---|---:|")
    for (w, h), n in sizes.most_common():
        L.append(f"| {w} x {h} | {n} |")
    L.append("")
    L.append("Colour modes: " + ", ".join(f"`{m}` ({n})" for m, n in modes.most_common()) + "\n")

    L.append("## Label file format\n")
    L.append("Each image has one JSON label file with the same stem. Structure:\n")
    L.append("```text")
    L.append(f"top-level keys:          {sorted(top_keys)}")
    L.append(f"features keys:           {sorted(feature_keys)}")
    L.append(f"feature properties keys: {sorted(prop_keys)}")
    L.append(f"metadata keys:           {sorted(meta_keys)}")
    L.append("```\n")
    L.append("- `features.xy` holds polygons in **pixel coordinates** of the 1024x1024 image; "
             "`features.lng_lat` holds the same polygons in longitude/latitude. "
             f"Files where the two lists differ in length: {lnglat_xy_mismatch}.")
    L.append("- Each polygon is a WKT string. Geometry types found: "
             + ", ".join(f"`{g}` ({n})" for g, n in geom_types.most_common()) + ".")
    L.append("- `feature_type` values: "
             + ", ".join(f"`{t}` ({n})" for t, n in feature_types.most_common()) + ".")
    L.append("- The damage class is the `subtype` property. It exists only in **post** label files; "
             "pre files list the same buildings (same `uid`) without a damage class.\n")
    if sample_label:
        name, meta, feat = sample_label
        wkt = feat["wkt"]
        L.append(f"Example feature from `{name}`:\n")
        L.append("```json")
        L.append(json.dumps({"properties": feat["properties"],
                             "wkt": wkt[:120] + ("..." if len(wkt) > 120 else "")}, indent=2))
        L.append("```\n")
        L.append("Example metadata:\n")
        L.append("```json")
        L.append(json.dumps(meta, indent=2))
        L.append("```\n")

    L.append("## Damage-label strings (exact `subtype` values)\n")
    L.append("| subtype | post-file buildings | pre-file buildings |")
    L.append("|---|---:|---:|")
    all_subs = sorted(set(subtype_by_phase["post"]) | set(subtype_by_phase["pre"]))
    for s in all_subs:
        L.append(f"| `{s}` | {subtype_by_phase['post'][s]} | {subtype_by_phase['pre'][s]} |")
    L.append("")

    L.append("### Per disaster (post files)\n")
    subs = sorted(set().union(*subtype_by_disaster.values())) if subtype_by_disaster else []
    L.append("| disaster | " + " | ".join(f"`{s}`" for s in subs) + " | total |")
    L.append("|---|" + "---:|" * (len(subs) + 1))
    for dis in sorted(subtype_by_disaster):
        c = subtype_by_disaster[dis]
        L.append(f"| {dis} | " + " | ".join(str(c[s]) for s in subs) + f" | {sum(c.values())} |")
    L.append("")

    L.append(f"## Target masks (sample of {len(sample_idx)} files)\n")
    L.append("Target PNGs are per-pixel masks rendered from the label polygons. "
             "Modes: " + ", ".join(f"`{m}` ({n})" for m, n in target_modes.most_common()) + ".\n")
    L.append("| phase | pixel value | pixels (sample) |")
    L.append("|---|---:|---:|")
    for phase in ("pre", "post"):
        for v, n in sorted(target_values[phase].items()):
            L.append(f"| {phase} | {v} | {n} |")
    L.append("")
    L.append("Standard xBD convention: pre masks are 0 = background, 1 = building; post masks are "
             "0 = background, 1 = no-damage, 2 = minor-damage, 3 = major-damage, 4 = destroyed.\n")

    # Where do un-classified buildings end up in the post mask? Look at the mask pixel under
    # a point guaranteed to be inside each polygon (shapely's representative_point).
    from shapely import wkt as shapely_wkt
    uncls_vals = Counter()
    for p in labels:
        if parse_name(p)[2] != "post" or sum(uncls_vals.values()) >= 50:
            continue
        with open(p, encoding="utf-8") as fh:
            feats = [f for f in json.load(fh)["features"]["xy"]
                     if f["properties"].get("subtype") == "un-classified"]
        if not feats:
            continue
        mask = np.asarray(Image.open(dirs["targets"] / f"{p.stem}_target.png"))
        for f in feats:
            pt = shapely_wkt.loads(f["wkt"]).representative_point()
            uncls_vals[int(mask[int(pt.y), int(pt.x)])] += 1
    # The 50 limit is checked per file, so the last file can push the total past 50.
    L.append(f"Mask value found inside `un-classified` polygons ({sum(uncls_vals.values())} "
             "checked, from the first post files containing any, stopping once 50 were reached): "
             + ", ".join(f"value {v}: {n}" for v, n in sorted(uncls_vals.items()))
             + ". A value of 0 means un-classified buildings are painted as background.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"images={len(images)} labels={len(labels)} targets={len(targets)} tiles={len(tiles)}")
    print("damage subtypes (post):", dict(subtype_by_phase["post"]))


if __name__ == "__main__":
    main()
