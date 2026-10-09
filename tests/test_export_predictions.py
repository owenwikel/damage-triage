"""scripts/export_predictions.py on a tiny synthetic cache and a 1-epoch model."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import export_predictions as ep  # noqa: E402
from conftest import write_fake_cache  # noqa: E402
from src.dataset import read_index  # noqa: E402
from src.split import TestSplitError  # noqa: E402
from src.train import Config, run  # noqa: E402

# Different sizes per part, so a file written for the wrong part cannot pass by accident.
SIZES = {"train": 24, "val-tiles": 12, "val": 16}


def make_cache(cache_dir):
    """A fake cache with a different number of buildings in each part."""
    frames = []
    for part, n in SIZES.items():
        write_fake_cache(cache_dir, n=n, parts=(part,))
        frames.append(pd.read_csv(cache_dir / "index.csv", dtype={"tile_id": str}))
    pd.concat(frames, ignore_index=True).to_csv(cache_dir / "index.csv", index=False)
    return cache_dir


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """(cache folder, runs folder) with two 1-epoch seeds of the stack model."""
    tmp = tmp_path_factory.mktemp("export")
    cache = make_cache(tmp / "cache")
    for seed in (0, 1):
        run(Config(arch="stack", seed=seed, cache_dir=str(cache), runs_dir=str(tmp / "runs"),
                   epochs=1, batch_size=8, eval_batch_size=8, pretrained=False, num_workers=0,
                   amp=False))
    return cache, tmp / "runs"


def test_row_counts_match_the_label_table_per_part(trained, tmp_path):
    cache, runs = trained
    out = tmp_path / "predictions"
    ep.main(["--runs", str(runs / "stack_s0"), str(runs / "stack_s1"), "--out", str(out),
             "--batch-size", "8"])
    labels = read_index(cache)                      # the label table: one row per building
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["runs"]) == {"stack_s0", "stack_s1"}
    for run_name in ("stack_s0", "stack_s1"):
        assert sorted(p.name for p in (out / run_name).iterdir()) == [
            "train.csv", "val-tiles.csv", "val.csv"]            # and no test.csv
        for part, n in SIZES.items():
            expected = labels[labels["part"] == part].sort_values("row")
            table = pd.read_csv(out / run_name / f"{part}.csv", dtype={"tile_id": str})
            assert len(table) == len(expected) == n
            assert manifest["runs"][run_name]["rows"][part] == n
            assert list(table.columns) == ep.COLUMNS
            # the same buildings, in the same order, with their own labels
            for col in ("uid", "disaster", "tile_id", "label"):
                assert table[col].tolist() == expected[col].tolist(), col
            assert table["uid"].is_unique
            assert table["label"].isin([0, 1, 2, 3]).all()
            assert np.isfinite(table[ep.LOGIT_COLUMNS].to_numpy()).all()


def test_exported_logits_are_the_ones_training_saved(trained, tmp_path):
    cache, runs = trained
    ep.export_run(runs / "stack_s0", tmp_path, parts=["val"], batch_size=5)
    table = pd.read_csv(tmp_path / "stack_s0" / "val.csv")
    saved = np.load(runs / "stack_s0" / "preds_val.npz")
    # Not bit-for-bit: another batch size (and, on a GPU, another convolution algorithm)
    # changes the last digits.
    assert table[ep.LOGIT_COLUMNS].to_numpy() == pytest.approx(saved["logits"], abs=1e-2)
    assert table["label"].tolist() == saved["labels"].tolist()


def test_the_test_part_cannot_be_exported(trained, tmp_path):
    cache, runs = trained
    with pytest.raises(SystemExit):                 # argparse: not an allowed --parts value
        ep.main(["--runs", str(runs / "stack_s0"), "--out", str(tmp_path), "--parts", "test"])
    with pytest.raises(TestSplitError):
        ep.export_run(runs / "stack_s0", tmp_path, parts=["val", "test"])
    assert not list(tmp_path.rglob("*.csv"))        # refused before anything was written


def test_unfinished_runs_are_skipped(trained, tmp_path):
    cache, runs = trained
    partial = tmp_path / "runs" / "stack_s9"
    partial.mkdir(parents=True)
    (partial / "best.pt").write_bytes((runs / "stack_s0" / "best.pt").read_bytes())
    with pytest.raises(SystemExit, match="No finished run"):
        ep.main(["--runs", str(partial), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
