import json
from pathlib import Path

import pandas as pd
import pytest
from shapely import wkt as shapely_wkt

from src.labels import DAMAGE_CLASSES, load_labels, parse_label_file, repair_polygon


def _feature(subtype, wkt, uid):
    props = {"feature_type": "building", "uid": uid}
    if subtype is not None:
        props["subtype"] = subtype
    return {"properties": props, "wkt": wkt}


SQUARE = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"          # area 100, centroid (5, 5)
RECT = "POLYGON ((20 30, 60 30, 60 40, 20 40, 20 30))"       # area 400, centroid (40, 35)


def _write_tile(root, disaster, tile, subtypes_and_wkts, disaster_type="flooding"):
    """Write a fake pre/post label pair in the xBD layout under root/labels."""
    labels = root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    meta = {"disaster": disaster, "disaster_type": disaster_type}
    post = [_feature(s, w, f"{tile}-{i}") for i, (s, w) in enumerate(subtypes_and_wkts)]
    pre = [_feature(None, w, f"{tile}-{i}") for i, (_, w) in enumerate(subtypes_and_wkts)]
    for phase, feats in (("post", post), ("pre", pre)):
        doc = {"features": {"xy": feats, "lng_lat": feats}, "metadata": meta}
        (labels / f"{disaster}_{tile}_{phase}_disaster.json").write_text(json.dumps(doc))


@pytest.fixture
def fake_root(tmp_path):
    _write_tile(tmp_path, "hurricane-x", "00000001", [("no-damage", SQUARE), ("destroyed", RECT)])
    _write_tile(tmp_path, "hurricane-x", "00000002", [("un-classified", SQUARE)])
    _write_tile(tmp_path, "fire-y", "00000007", [("minor-damage", RECT)], disaster_type="fire")
    _write_tile(tmp_path, "fire-y", "00000008", [])  # tile with no buildings
    return tmp_path


def test_one_row_per_building(fake_root):
    df = load_labels(fake_root)
    assert len(df) == 4
    assert df["uid"].is_unique


def test_pre_files_are_ignored(fake_root):
    # pre files have no subtype; if they were read we'd get 8 rows or a ValueError
    df = load_labels(fake_root)
    assert df["damage"].notna().all()


def test_columns_and_paths(fake_root):
    df = load_labels(fake_root)
    row = df[df["tile_id"] == "00000007"].iloc[0]
    assert row["disaster"] == "fire-y"
    assert row["disaster_type"] == "fire"
    assert Path(row["pre_image"]) == fake_root / "images" / "fire-y_00000007_pre_disaster.png"
    assert Path(row["post_image"]) == fake_root / "images" / "fire-y_00000007_post_disaster.png"


def test_geometry(fake_root):
    df = load_labels(fake_root)
    rect = df[df["uid"] == "00000001-1"].iloc[0]
    assert rect["polygon"].geom_type == "Polygon"
    assert rect["area_px"] == pytest.approx(400)
    assert (rect["centroid_x"], rect["centroid_y"]) == pytest.approx((40, 35))


def test_damage_dtypes(fake_root):
    df = load_labels(fake_root)
    assert isinstance(df["damage"].dtype, pd.CategoricalDtype)
    assert list(df["damage"].cat.categories) == DAMAGE_CLASSES
    assert str(df["damage_level"].dtype) == "Int64"
    levels = df.set_index("uid")["damage_level"]
    assert levels["00000001-0"] == 0
    assert levels["00000001-1"] == 3
    assert levels["00000007-0"] == 1
    assert pd.isna(levels["00000002-0"])


def test_empty_folder_gives_empty_frame(tmp_path):
    (tmp_path / "labels").mkdir()
    df = load_labels(tmp_path)
    assert len(df) == 0
    assert "damage" in df.columns


def test_missing_labels_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path)


def test_unknown_subtype_raises(tmp_path):
    _write_tile(tmp_path, "quake-z", "00000001", [("totally-flattened", SQUARE)])
    with pytest.raises(ValueError, match="totally-flattened"):
        load_labels(tmp_path)


# A square whose outline runs up a spike and back down the same line: invalid, area 100.
SPIKE = "POLYGON ((0 0, 10 0, 10 10, 5 10, 5 15, 5 10, 0 10, 0 0))"
# A bow-tie: buffer(0) splits it, and the loader keeps one piece.
BOWTIE = "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"


def test_invalid_polygon_is_repaired(tmp_path):
    _write_tile(tmp_path, "quake-z", "00000001", [("no-damage", SPIKE), ("destroyed", SQUARE)])
    df = load_labels(tmp_path)
    assert len(df) == 2  # repaired, not dropped
    spike = df.iloc[0]
    assert spike["polygon"].is_valid
    assert spike["polygon"].geom_type == "Polygon"
    assert spike["area_px"] == pytest.approx(100)
    assert (spike["centroid_x"], spike["centroid_y"]) == pytest.approx((5, 5))


def test_valid_polygon_is_left_untouched(tmp_path):
    _write_tile(tmp_path, "quake-z", "00000001", [("no-damage", RECT)])
    poly = load_labels(tmp_path).iloc[0]["polygon"]
    assert list(poly.exterior.coords) == [(20, 30), (60, 30), (60, 40), (20, 40), (20, 30)]


def test_repair_can_be_switched_off(tmp_path):
    _write_tile(tmp_path, "quake-z", "00000001", [("no-damage", SPIKE)])
    assert not load_labels(tmp_path, repair=False).iloc[0]["polygon"].is_valid


def test_repair_always_gives_one_valid_polygon():
    fixed = repair_polygon(shapely_wkt.loads(BOWTIE))
    assert fixed.is_valid and fixed.geom_type == "Polygon" and fixed.area > 0


def test_drop_unclassified(fake_root):
    kept = load_labels(fake_root, drop_unclassified=True)
    assert len(kept) == 3
    assert "un-classified" not in set(kept["damage"])
    assert kept["damage_level"].notna().all()
    assert list(kept.index) == [0, 1, 2]
    # nothing else changes: same rows as the default minus the un-classified one
    everything = load_labels(fake_root)
    assert set(everything["uid"]) - set(kept["uid"]) == {"00000002-0"}
    assert list(kept["damage"].cat.categories) == DAMAGE_CLASSES


def test_unclassified_is_kept_by_default(fake_root):
    assert (load_labels(fake_root)["damage"] == "un-classified").sum() == 1


def test_bad_file_name(tmp_path):
    bad = tmp_path / "notes.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        parse_label_file(bad, tmp_path)
