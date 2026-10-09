"""scripts/calibrate_ensemble.py on small synthetic prediction tables."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_ensemble as ce  # noqa: E402
from src.metrics import softmax  # noqa: E402
from src.split import TestSplitError  # noqa: E402

RUNS = ["m_s0", "m_s1", "m_s2"]


def write_predictions(pred_dir, n=600, sharpen=3.0, seed=0):
    """Three "seeds" of over-confident logits for val-tiles and val, in the export's layout.

    The true label is drawn from softmax(z); the saved logits are z * sharpen, so the
    right temperature is about ``sharpen``.
    """
    rng = np.random.default_rng(seed)
    for part, disasters in (("val-tiles", ["hurricane-harvey"]),
                            ("val", ["hurricane-florence", "midwest-flooding"])):
        z = rng.normal(size=(n, 4)) * 1.5
        p = softmax(z)
        labels = np.array([rng.choice(4, p=row) for row in p])
        for run in RUNS:
            table = pd.DataFrame({"uid": [f"{part}-{i}" for i in range(n)],
                                  "disaster": [disasters[i % len(disasters)] for i in range(n)],
                                  "tile_id": [f"{i // 50:08d}" for i in range(n)],
                                  "label": labels})
            noisy = z * sharpen + rng.normal(scale=0.05, size=z.shape)
            table[ce.LOGIT_COLUMNS] = noisy
            (pred_dir / run).mkdir(parents=True, exist_ok=True)
            table.to_csv(pred_dir / run / f"{part}.csv", index=False)
    return pred_dir


def test_end_to_end_writes_tables_figures_and_numbers(tmp_path):
    pred = write_predictions(tmp_path / "predictions")
    out, doc, figs = tmp_path / "calib", tmp_path / "docs" / "calibration.md", tmp_path / "figs"
    ce.main(["--pred-dir", str(pred), "--runs", *RUNS, "--out", str(out), "--doc", str(doc),
             "--doc-fig-dir", str(figs)])
    numbers = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    # the logits were sharpened 3x, so both fits should find a temperature near 3
    assert numbers["temperature_fitted_on_val-tiles"] == pytest.approx(3.0, rel=0.2)
    assert numbers["temperature_fitted_on_val"] == pytest.approx(3.0, rel=0.2)
    for key in ("fit_on_val-tiles", "fit_on_val_alternative"):
        rows = {(r["part"], r["temperature scaling"]): r for r in numbers[key]}
        assert set(rows) == {(p, w) for p in ("val-tiles", "val") for w in ("before", "after")}
        for part in ("val-tiles", "val"):
            before, after = rows[part, "before"], rows[part, "after"]
            assert after["ECE"] < before["ECE"] and after["NLL"] < before["NLL"]
            assert after["accuracy"] == before["accuracy"]      # scaling keeps the class
    assert [r["disaster"] for r in numbers["val_per_disaster"]] == [
        "all", "hurricane-florence", "midwest-flooding"]
    assert sum(r["buildings"] for r in numbers["val_per_disaster"][1:]) == 600
    per_class = pd.DataFrame(numbers["per_class"])
    assert per_class.groupby("buildings")["n"].sum().to_dict() == {
        "val-tiles (all)": 600, "val (all)": 600,
        "val: hurricane-florence": 300, "val: midwest-flooding": 300}
    for name in ("fit_on_val-tiles", "fit_on_val", "val_per_disaster"):
        assert (out / f"reliability_{name}.png").exists()
        assert (figs / f"calibration_{name}.png").exists()
    text = doc.read_text(encoding="utf-8")
    assert "### 2. Alternative: temperature fitted on val" in text
    assert "**Not the main result.**" in text


def test_hand_written_text_around_the_tables_survives_a_rerun(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=200)
    doc = tmp_path / "calibration.md"
    args = ["--pred-dir", str(pred), "--runs", *RUNS, "--out", str(tmp_path / "calib"),
            "--doc", str(doc), "--doc-fig-dir", str(tmp_path / "figs")]
    ce.main(args)
    doc.write_text("# My reading\n\n" + doc.read_text(encoding="utf-8"), encoding="utf-8")
    ce.main(args)
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("# My reading") and text.count("## Tables") == 1


def test_ensemble_logits_are_the_log_of_the_mean_softmax(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=40)
    table = ce.load_ensemble(pred, RUNS, "val")
    members = [pd.read_csv(pred / r / "val.csv")[ce.LOGIT_COLUMNS].to_numpy() for r in RUNS]
    mean = np.mean([softmax(m) for m in members], axis=0)
    assert softmax(ce.logits_of(table)) == pytest.approx(mean)


def test_the_test_part_and_misaligned_seeds_are_refused(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=40)
    with pytest.raises(TestSplitError):
        ce.load_ensemble(pred, RUNS, "test")
    path = pred / "m_s1" / "val.csv"
    pd.read_csv(path, dtype={"tile_id": str}).iloc[::-1].to_csv(path, index=False)
    with pytest.raises(AssertionError, match="out of step"):
        ce.load_ensemble(pred, RUNS, "val")
