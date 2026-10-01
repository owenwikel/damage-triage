"""The test part must be unreachable without an explicit final=True / --final."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_fake_split
from src.split import DEV_PARTS, TestSplitError, load_part_labels, load_split, require_final

REPO = Path(__file__).resolve().parents[1]


def test_dev_parts_pass_and_test_is_refused():
    assert require_final(["train", "val-tiles", "val"]) == ["train", "val-tiles", "val"]
    assert require_final("val") == ["val"]
    with pytest.raises(TestSplitError, match="final"):
        require_final("test")
    with pytest.raises(TestSplitError):
        require_final(["train", "test"])
    assert require_final(["test"], final=True) == ["test"]


def test_unknown_part_is_an_error():
    with pytest.raises(ValueError, match="Unknown"):
        require_final("holdout")


def test_load_split_never_returns_test_tiles_by_default():
    tiles = load_split()
    assert set(tiles["part"]) == set(DEV_PARTS)
    assert not tiles["disaster"].isin(["hurricane-michael", "santa-rosa-wildfire"]).any()
    with pytest.raises(TestSplitError):
        load_split(["test"])


@pytest.fixture
def fake_data(tmp_path):
    """Three tiles on disk plus a test tile that exists ONLY in the split file."""
    data = tmp_path / "xbd"
    write_fake_split(data / "train", "hurricane-harvey", "00000001", "flooding")
    write_fake_split(data / "test", "hurricane-harvey", "00000002", "flooding")
    write_fake_split(data / "train", "hurricane-florence", "00000003", "flooding")
    split = tmp_path / "split.csv"
    pd.DataFrame([("hurricane-harvey", "00000001", "train"),
                  ("hurricane-harvey", "00000002", "val-tiles"),
                  ("hurricane-florence", "00000003", "val"),
                  ("hurricane-michael", "00000004", "test")],
                 columns=["disaster", "tile_id", "part"]).to_csv(split, index=False)
    return data, split


def test_load_part_labels_reads_only_the_requested_parts(fake_data):
    data, split = fake_data
    # The michael tile has no files. If the loader tried to open it, this would crash.
    df = load_part_labels(data, split_csv=split)
    assert set(df["part"]) == set(DEV_PARTS)
    assert len(df) == 6                       # 3 tiles x 2 classified buildings
    assert "un-classified" not in set(df["damage"])
    assert len(load_part_labels(data, ["val"], split, drop_unclassified=False)) == 3
    with pytest.raises(TestSplitError):
        load_part_labels(data, ["test"], split)


def test_build_crops(fake_data, tmp_path):
    data, split = fake_data
    out = tmp_path / "crops"
    script = [sys.executable, str(REPO / "scripts" / "build_crops.py"), "--data-dir", str(data),
              "--split", str(split), "--out", str(out), "--workers", "1"]
    refused = subprocess.run(script + ["--parts", "test"], capture_output=True, text=True)
    assert refused.returncode != 0 and "TestSplitError" in refused.stderr
    assert not (out / "test_pre.npy").exists()

    done = subprocess.run(script, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    index = pd.read_csv(out / "index.csv", dtype={"tile_id": str})
    assert sorted(index["part"].unique()) == ["train", "val", "val-tiles"]
    assert list(index[index["part"] == "train"]["label"]) == [0, 3]
    pre, post = np.load(out / "train_pre.npy"), np.load(out / "train_post.npy")
    assert pre.shape == post.shape == (2, 128, 128, 3) and pre.dtype == np.uint8
    # conftest darkens the destroyed building in the post image only
    assert post[1].mean() < 0.8 * pre[1].mean()
    # box 60 x 50 px -> window side 60 + 2 * 16
    assert index.loc[0, ["box_w", "box_h", "win_side"]].tolist() == [60.0, 50.0, 92]
