"""The final-evaluation scripts, on synthetic data only: the three locks of the test
export, and the results script. Nothing here reads the real test set."""

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

import export_test_predictions as etp  # noqa: E402
import test_results as tr  # noqa: E402
from conftest import write_fake_cache  # noqa: E402
from src.split import TestSplitError  # noqa: E402
from src.train import Config, run  # noqa: E402
from test_calibrate_ensemble import RUNS, write_predictions  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Two tiny trained runs, a development cache and a separate fake "test" cache."""
    tmp = tmp_path_factory.mktemp("final")
    dev = write_fake_cache(tmp / "cache", n=24)
    test_cache = write_fake_cache(tmp / "cache_test", n=20, parts=("test",))
    for seed in (0, 1):
        run(Config(arch="stack", seed=seed, cache_dir=str(dev), runs_dir=str(tmp / "runs"),
                   epochs=1, batch_size=8, eval_batch_size=8, pretrained=False, num_workers=0,
                   amp=False))
    return tmp, test_cache, [tmp / "runs" / "stack_s0", tmp / "runs" / "stack_s1"]


def args_for(tmp, test_cache, runs, out="out", lock="TEST.lock"):
    protocol = tmp / "protocol.md"
    protocol.write_text("protocol")
    return ["--cache-dir", str(test_cache), "--runs", *map(str, runs), "--out", str(tmp / out),
            "--lock", str(tmp / lock), "--protocol", str(protocol), "--batch-size", "8"]


def test_export_needs_final_then_a_committed_protocol(setup, monkeypatch):
    tmp, test_cache, runs = setup
    args = args_for(tmp, test_cache, runs, out="out_a", lock="a.lock")
    with pytest.raises(TestSplitError):
        etp.main(args)                                   # no --final
    monkeypatch.setattr(etp, "protocol_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="not committed"):
        etp.main(["--final", *args])
    assert not (tmp / "a.lock").exists() and not (tmp / "out_a").exists()   # nothing was loaded


def test_export_runs_once_and_the_lock_stops_a_second_run(setup, monkeypatch):
    tmp, test_cache, runs = setup
    monkeypatch.setattr(etp, "protocol_is_committed", lambda path: True)
    args = ["--final", *args_for(tmp, test_cache, runs, out="out_b", lock="b.lock")]
    etp.main(args)
    lock = json.loads((tmp / "b.lock").read_text())
    assert lock["status"] == "completed" and lock["runs"] == ["stack_s0", "stack_s1"]
    for name in ("stack_s0", "stack_s1"):
        table = pd.read_csv(tmp / "out_b" / name / "test.csv", dtype={"tile_id": str})
        assert len(table) == lock["files"][name]["rows"] == 20
        assert lock["files"][name]["sha256"] == etp.sha256(tmp / "out_b" / name / "test.csv")
        assert list(table.columns) == etp.COLUMNS
    before = (tmp / "out_b" / "stack_s0" / "test.csv").read_bytes()
    with pytest.raises(SystemExit, match="already been exported"):
        etp.main(args)
    assert (tmp / "out_b" / "stack_s0" / "test.csv").read_bytes() == before


def test_a_crashed_export_leaves_a_started_lock_that_also_blocks(setup, monkeypatch):
    tmp, test_cache, runs = setup
    monkeypatch.setattr(etp, "protocol_is_committed", lambda path: True)

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(etp, "export_run_test", boom)
    args = ["--final", *args_for(tmp, test_cache, runs, out="out_c", lock="c.lock")]
    with pytest.raises(RuntimeError):
        etp.main(args)
    assert json.loads((tmp / "c.lock").read_text())["status"] == "started"
    with pytest.raises(SystemExit):
        etp.main(args)
    with pytest.raises(SystemExit, match="did not finish"):
        tr.check_lock(tmp / "c.lock", tmp / "out_c", ["stack_s0"])


def test_protocol_check_is_false_for_a_file_outside_git(tmp_path):
    assert etp.protocol_is_committed(tmp_path / "missing.md") is False
    (tmp_path / "loose.md").write_text("x")
    assert etp.protocol_is_committed(tmp_path / "loose.md") is False


def test_the_development_export_still_refuses_the_test_part():
    import export_predictions as ep
    with pytest.raises(SystemExit):
        ep.main(["--parts", "test"])


def test_results_script_needs_final_and_an_intact_lock_for_test(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=120)
    for run_name in RUNS:      # fake "test" tables: copies of the val tables
        (pred / run_name / "test.csv").write_bytes((pred / run_name / "val.csv").read_bytes())
    common = ["--pred-dir", str(pred), "--dev-pred-dir", str(pred), "--runs", *RUNS, "--n-boot", "20",
              "--doc", str(tmp_path / "docs" / "r.md"), "--fig-dir", str(tmp_path / "docs" / "figures"),
              "--out", str(tmp_path / "out"), "--lock", str(tmp_path / "t.lock")]
    with pytest.raises(TestSplitError):
        tr.main(common)                                      # part test, no --final
    with pytest.raises(SystemExit, match="not found"):
        tr.main(["--final", *common])                        # no lock
    files = {r: {"rows": 120, "sha256": etp.sha256(pred / r / "test.csv")} for r in RUNS}
    (tmp_path / "t.lock").write_text(json.dumps({"status": "completed", "files": files}))
    tr.main(["--final", *common])
    text = (tmp_path / "docs" / "r.md").read_text(encoding="utf-8")
    assert "Part: **test**" in text and "### 6. Side by side with the validation sets" in text
    assert "(figures/test_risk_coverage_miss_rate.png)" in text
    assert (tmp_path / "docs" / "figures" / "test_risk_coverage_miss_rate.png").exists()
    # a file changed after the export is noticed
    with open(pred / RUNS[0] / "test.csv", "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="differs"):
        tr.main(["--final", *common])


def test_extra_metrics_match_a_direct_calculation(tmp_path):
    pred = write_predictions(tmp_path / "predictions", n=200)
    table = tr.load_tables(pred, RUNS, "val")
    extra = tr.extra_metrics(table, n_boot=30, seed=0)
    y = table["label"].to_numpy()
    p = tr.softmax(tr.logits_of(table)).argmax(axis=1)
    real = ((y > 0) & (p > 0)).sum() / (p > 0).sum()
    assert extra["share of predicted damage that is real"][0] == pytest.approx(real)
    for c, name in enumerate(tr.CLASS_NAMES):
        assert extra[f"recall {name}"][0] == pytest.approx((p[y == c] == c).mean())
        lo, hi = extra[f"recall {name}"][1:]
        assert lo <= hi
    assert sum(extra["_counts"].values()) == 200
    groups = tr.split_groups(table, "val")
    assert list(groups) == ["val: hurricane-florence", "val: midwest-flooding", "val: both disasters"]
    assert len(groups["val: both disasters"]) == 200


def test_exploratory_class_mix_needs_final_and_recombines_correctly():
    import test_class_mix as tcm
    with pytest.raises(TestSplitError):
        tcm.main([])                                     # no --final: nothing is read
    # per-class rates recombined with their own mix give back the overall rate
    rates = {"minor-damage": 0.6, "major-damage": 0.3, "destroyed": 0.1}
    mix = {"minor-damage": 10, "major-damage": 30, "destroyed": 60}
    assert tcm.standardised(rates, mix) == pytest.approx(0.06 + 0.09 + 0.06)
    assert tcm.standardised(rates, {k: 1 for k in mix}) == pytest.approx(1 / 3)
