"""Parse xBD label JSON files into a pandas DataFrame with one row per building.

Each xBD tile has a pre-disaster and a post-disaster image, each with its own JSON label
file. Both files list the same buildings (matched by ``uid``), but only the post file
carries the damage class (the ``subtype`` property). This module reads the post files,
takes the polygon and damage class from there, and records the paths of both images.

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


def parse_label_file(post_json_path, images_dir):
    """Return a list of row dicts, one per building, for a single post-disaster label file.

    ``images_dir`` is the folder holding the matching PNGs; paths are built, not checked.
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


def load_labels(data_root):
    """Load every post-disaster label under ``data_root/labels`` into one DataFrame.

    ``data_root`` is a split folder such as ``xbd-data/test`` containing ``images/`` and
    ``labels/``. Returns one row per building with the columns in ``COLUMNS``.
    """
    data_root = Path(data_root)
    labels_dir = data_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No labels folder at {labels_dir}")

    rows = []
    for path in sorted(labels_dir.glob("*_post_disaster.json")):
        rows.extend(parse_label_file(path, data_root / "images"))

    df = pd.DataFrame(rows, columns=COLUMNS)
    # Categorical columns store each distinct string once; faster and self-documenting.
    df["damage"] = pd.Categorical(df["damage"], categories=DAMAGE_CLASSES)
    df["disaster"] = df["disaster"].astype("category")
    # Int64 (capital I) is pandas' nullable integer type, so un-classified can be <NA>.
    df["damage_level"] = df["damage_level"].astype("Int64")
    return df
