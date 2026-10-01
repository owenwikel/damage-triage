"""Parse xBD label JSON files into a pandas DataFrame with one row per building.

Each xBD tile has a pre-disaster and a post-disaster image, each with its own JSON label
file. Both files list the same buildings (matched by ``uid``), but only the post file
carries the damage class (the ``subtype`` property). This module reads the post files,
takes the polygon and damage class from there, and records the paths of both images.

Two data decisions from docs/data_decisions.md are applied here, so that every later step
sees the same buildings and the same geometry:
  - an invalid polygon (an outline that crosses itself) is repaired with ``buffer(0)``;
  - un-classified buildings can be dropped with ``load_labels(..., drop_unclassified=True)``.

Example:
    >>> from src.labels import load_labels
    >>> df = load_labels(r"C:\\Users\\wikel\\dev\\xbd-data\\test")
    >>> df["damage"].value_counts()
"""

import json
import re
from pathlib import Path

import pandas as pd
from shapely import wkt as shapely_wkt

# The exact subtype strings found in xBD post-disaster labels (see outputs/data_report.md).
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]

# Ordinal level for the four real damage classes. "un-classified" means the annotators
# couldn't tell, so it gets no level (pandas <NA>) rather than a made-up number.
DAMAGE_LEVEL = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}

NAME_RE = re.compile(r"^(?P<disaster>.+)_(?P<tile>\d+)_post_disaster\.json$")

COLUMNS = [
    "disaster", "disaster_type", "tile_id", "uid",
    "pre_image", "post_image", "polygon", "damage", "damage_level",
    "area_px", "centroid_x", "centroid_y",
]


def repair_polygon(poly):
    """Return ``poly`` if it is valid, otherwise a repaired copy made with ``buffer(0)``.

    ``buffer(0)`` redraws the outline without the crossing. If that splits the shape into
    several pieces, the largest one is kept, so the result is always a single Polygon.
    """
    if poly.is_valid:
        return poly
    fixed = poly.buffer(0)
    if fixed.geom_type == "MultiPolygon":
        fixed = max(fixed.geoms, key=lambda g: g.area)
    if fixed.is_empty or fixed.geom_type != "Polygon":
        raise ValueError(f"buffer(0) could not repair polygon: {poly.wkt[:80]}...")
    return fixed


def parse_label_file(post_json_path, images_dir, repair=True):
    """Return a list of row dicts, one per building, for a single post-disaster label file.

    ``images_dir`` is the folder holding the matching PNGs; paths are built, not checked.
    ``repair=False`` keeps invalid polygons exactly as annotated (used by the data audit).
    """
    post_json_path = Path(post_json_path)
    m = NAME_RE.match(post_json_path.name)
    if m is None:
        raise ValueError(f"Not an xBD post-disaster label file name: {post_json_path.name}")
    disaster, tile_id = m.group("disaster"), m.group("tile")

    with open(post_json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("metadata", {})

    images_dir = Path(images_dir)
    pre_image = images_dir / f"{disaster}_{tile_id}_pre_disaster.png"
    post_image = images_dir / f"{disaster}_{tile_id}_post_disaster.png"

    rows = []
    for feat in data["features"]["xy"]:  # "xy" = pixel coordinates, not lng/lat
        props = feat["properties"]
        if props.get("feature_type", "building") != "building":
            continue
        damage = props.get("subtype")
        if damage not in DAMAGE_CLASSES:
            raise ValueError(f"Unknown damage subtype {damage!r} in {post_json_path.name}")
        poly = shapely_wkt.loads(feat["wkt"])
        if repair:
            poly = repair_polygon(poly)
        centroid = poly.centroid
        rows.append({
            "disaster": meta.get("disaster", disaster),
            "disaster_type": meta.get("disaster_type"),
            "tile_id": tile_id,
            "uid": props.get("uid"),
            "pre_image": str(pre_image),
            "post_image": str(post_image),
            "polygon": poly,
            "damage": damage,
            "damage_level": DAMAGE_LEVEL.get(damage),
            "area_px": poly.area,
            "centroid_x": centroid.x,
            "centroid_y": centroid.y,
        })
    return rows


def load_labels(data_root, drop_unclassified=False, repair=True):
    """Load every post-disaster label under ``data_root/labels`` into one DataFrame.

    ``data_root`` is a split folder such as ``xbd-data/test`` containing ``images/`` and
    ``labels/``. Returns one row per building with the columns in ``COLUMNS``.

    ``drop_unclassified=True`` leaves out buildings whose damage is "un-classified"; use it
    for training and evaluation. The default keeps them, because the data reports count
    them. ``repair=False`` keeps invalid polygons as annotated instead of repairing them.
    """
    data_root = Path(data_root)
    labels_dir = data_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No labels folder at {labels_dir}")

    rows = []
    for path in sorted(labels_dir.glob("*_post_disaster.json")):
        rows.extend(parse_label_file(path, data_root / "images", repair=repair))

    df = pd.DataFrame(rows, columns=COLUMNS)
    # Categorical columns store each distinct string once; faster and self-documenting.
    df["damage"] = pd.Categorical(df["damage"], categories=DAMAGE_CLASSES)
    df["disaster"] = df["disaster"].astype("category")
    # Int64 (capital I) is pandas' nullable integer type, so un-classified can be <NA>.
    df["damage_level"] = df["damage_level"].astype("Int64")
    if drop_unclassified:
        # the "un-classified" category stays in the dtype, with zero rows
        df = df[df["damage"] != "un-classified"].reset_index(drop=True)
    return df
