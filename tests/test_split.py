"""Checks on the committed split (splits/split_v1.csv) and on the code that makes it.

The CSV checks need no dataset, so they run in CI. Set XBD_DATA_DIR to also check that
scripts/make_split.py still reproduces the committed files from the real data.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_split  # noqa: E402

from conftest import write_fake_split  # noqa: E402

N_TILES = 3732  # train + test folders pooled, see docs/split_candidates.md


@pytest.fixture(scope="module")
def split():
    # dtype=str keeps the leading zeros of tile ids like 00000012
    return pd.read_csv(REPO / "splits" / "split_v1.csv", dtype=str)


@pytest.fixture(scope="module")
def blocks():
    return pd.read_csv(REPO / "splits" / "split_v1_blocks.csv", dtype=str, keep_default_na=False)


def test_columns_and_parts(split):
    assert list(split.columns) == ["disaster", "tile_id", "part"]
    assert set(split["part"]) == {"train", "val-tiles", "val", "test"}


def test_every_tile_appears_exactly_once(split):
    assert not split.duplicated(["disaster", "tile_id"]).any()
    assert len(split) == N_TILES


def test_no_disaster_overlap_between_train_val_test(split):
    # val-tiles is cut out of the train disasters on purpose, so it counts as train here
    group = split["part"].replace("val-tiles", "train")
    assert (split.assign(group=group).groupby("disaster")["group"].nunique() == 1).all()
    by_group = split.assign(group=group).groupby("group")["disaster"].agg(set)
    assert by_group["train"] == set(make_split.TRAIN)
    assert by_group["val"] == set(make_split.VAL)


def test_test_contains_only_michael_and_santa_rosa(split):
    assert set(split.loc[split["part"] == "test", "disaster"]) == {"hurricane-michael",
                                                                  "santa-rosa-wildfire"}
    # ...and all of them: no tile of those two disasters is anywhere else
    held = split[split["disaster"].isin(["hurricane-michael", "santa-rosa-wildfire"])]
    assert (held["part"] == "test").all()


def test_no_block_is_split_between_train_and_val_tiles(split, blocks):
    assert not blocks.duplicated(["disaster", "tile_id"]).any()
    merged = split.merge(blocks, on=["disaster", "tile_id"], how="left", validate="1:1")
    assert merged["block"].notna().all(), "a tile in the split has no row in the blocks file"
    placed = merged[merged["block"] != ""]
    assert (placed.groupby(["disaster", "block"])["part"].nunique() == 1).all()
    # tiles without coordinates cannot be in a block, so they must not be in val-tiles
    assert (merged.loc[merged["block"] == "", "part"] != "val-tiles").all()


def test_val_tiles_is_about_ten_percent_of_each_train_disaster(split, blocks):
    merged = split.merge(blocks, on=["disaster", "tile_id"])
    placed = merged[(merged["block"] != "") & merged["disaster"].isin(make_split.TRAIN)]
    share = (placed["part"] == "val-tiles").groupby(placed["disaster"]).mean()
    assert set(share.index) == set(make_split.TRAIN)
    assert share.between(0.05, 0.15).all(), share


# --- the code, on made-up tiles -------------------------------------------------------

def grid_tiles(disaster, nx, ny, w=0.005, h=0.0045, lng0=10.0, lat0=20.0):
    """nx * ny tiles laid edge to edge, like a mosaic."""
    rows = [(disaster, f"{i * ny + j:08d}", lng0 + (i + 0.5) * w, lat0 + (j + 0.5) * h, w, h)
            for i in range(nx) for j in range(ny)]
    return pd.DataFrame(rows, columns=["disaster", "tile_id", "lng", "lat", "w", "h"])


def test_touch_and_overlap():
    g = pd.DataFrame({"lng": [0.0, 0.005, 0.010, 0.0005], "lat": [0.0, 0.0045, 0.0, 0.0],
                      "w": 0.005, "h": 0.0045})
    touch, overlap = make_split.pair_geometry(g)
    assert touch[0, 1]          # corner to corner
    assert not touch[0, 2]      # one tile apart
    assert overlap[0, 3] and touch[0, 3]
    assert not overlap[0, 1]
    assert not touch.diagonal().any() and not overlap.diagonal().any()


def test_overlapping_tiles_share_a_block():
    # two tiles on either side of a grid line at lng 10.02, overlapping by 80%
    g = pd.DataFrame({"lng": [10.0195, 10.0205], "lat": [20.001, 20.001],
                      "w": 0.005, "h": 0.0045})
    blocks = make_split.assign_blocks(g, 0.02)
    assert blocks[0] == blocks[1]
    apart = g.assign(lng=[10.0150, 10.0250])  # same cells, no overlap: two blocks
    assert len(set(make_split.assign_blocks(apart, 0.02))) == 2


def test_pick_val_blocks_gets_close_to_target_and_never_takes_everything():
    rng = np.random.default_rng(0)
    blocks = [f"b{i // 4}" for i in range(200)]  # 50 blocks of 4 tiles
    picked = make_split.pick_val_blocks(blocks, 0.10, rng)
    assert sum(b in picked for b in blocks) == 20
    assert make_split.pick_val_blocks(["only"] * 7, 0.10, rng) == []


def test_make_split_on_synthetic_tiles():
    tiles = pd.concat([grid_tiles("hurricane-harvey", 20, 20),
                       grid_tiles("hurricane-florence", 3, 3),
                       grid_tiles("hurricane-michael", 3, 3)], ignore_index=True)
    # one harvey tile without buildings: no coordinates
    tiles.loc[0, ["lng", "lat", "w", "h"]] = np.nan
    out = make_split.make_split(tiles, block_deg=0.02, frac=0.10, seed=0)
    assert out.loc[0, "part"] == "train" and out.loc[0, "block"] == ""
    assert set(out.loc[out["disaster"] == "hurricane-florence", "part"]) == {"val"}
    assert set(out.loc[out["disaster"] == "hurricane-michael", "part"]) == {"test"}
    harvey = out[(out["disaster"] == "hurricane-harvey") & (out["block"] != "")]
    assert (harvey.groupby("block")["part"].nunique() == 1).all()
    assert 0.07 <= (harvey["part"] == "val-tiles").mean() <= 0.13
    # same seed, same answer
    again = make_split.make_split(tiles, block_deg=0.02, frac=0.10, seed=0)
    assert out["part"].equals(again["part"])
    leak = make_split.leak_table(out)
    assert leak.loc["all", "overlapping pairs split"] == 0
    assert leak.loc["all", "blocks touching a train block"] <= leak.loc["all", "val-tiles blocks"]


def test_unknown_disaster_is_an_error():
    with pytest.raises(ValueError, match="not assigned"):
        make_split.make_split(grid_tiles("fake-fire", 2, 2))


def test_tile_footprint_from_label_file(tmp_path):
    # conftest's fake tiles map pixels to lng/lat as (-120 + x * 1e-5, 38 - y * 1e-5)
    write_fake_split(tmp_path, "hurricane-harvey", "00000001", "flooding")
    lng, lat, w, h = make_split.tile_footprint(
        tmp_path / "labels" / "hurricane-harvey_00000001_post_disaster.json")
    assert lng == pytest.approx(-120 + 512e-5, abs=1e-9)
    assert lat == pytest.approx(38 - 512e-5, abs=1e-9)
    assert w == pytest.approx(1024e-5) and h == pytest.approx(1024e-5)


# --- the real data, when it is available ----------------------------------------------

@pytest.mark.skipif(not os.environ.get("XBD_DATA_DIR"), reason="XBD_DATA_DIR not set")
def test_committed_split_matches_the_data(split, blocks):
    tiles = make_split.make_split(make_split.load_tiles(os.environ["XBD_DATA_DIR"],
                                                        ["train", "test"]))
    assert len(tiles) == N_TILES
    fresh = tiles[["disaster", "tile_id", "part"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, split)
    assert tiles["block"].tolist() == blocks["block"].tolist()
