"""End to end on a tiny fake cache: train a few models, then build the reports from them."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import write_fake_cache  # noqa: E402
from src.train import Config, run  # noqa: E402
from src.uncertainty import collect_methods  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def command(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=REPO,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{args[0]} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("reports")
    cache = write_fake_cache(root / "cache", n=24)
    common = dict(cache_dir=str(cache), runs_dir=str(root / "runs"), epochs=1, batch_size=8,
                  eval_batch_size=8, pretrained=False, num_workers=0, amp=False)
    run(Config(arch="siamese", seed=0, **common))
    run(Config(arch="siamese", seed=1, **common))
    run(Config(arch="stack", seed=0, **common))
    # whichever architecture scores higher gets the augmentation experiment in the report
    run(Config(arch="siamese", augment="strong", **common))
    run(Config(arch="stack", augment="strong", **common))
    # MC dropout for the base run, through the command line as it is used for real
    command("-m", "src.uncertainty", "--run", root / "runs" / "siamese_s0", "--passes", 3,
            "--batch-size", 8)
    return root


def test_mc_dropout_files(workspace):
    d = np.load(workspace / "runs" / "siamese_s0" / "mc_dropout_val.npz")
    assert d["probs"].shape == (24, 4) and d["entropy"].shape == (24,)
    assert (workspace / "runs" / "siamese_s0" / "mc_dropout.json").exists()
    assert not (workspace / "runs" / "siamese_s0" / "mc_dropout_test.npz").exists()


def test_collect_methods(workspace):
    runs = workspace / "runs"
    methods, info = collect_methods(runs / "siamese_s0", [runs / f"siamese_s{s}" for s in (0, 1, 2)])
    assert info["members"] == ["siamese_s0", "siamese_s1"]        # seed 2 was never trained
    assert list(methods["val"]) == ["max prob (raw)", "max prob (temp. scaled)",
                                    "MC dropout (mean max prob)", "MC dropout (entropy)",
                                    "ensemble of 2", "ensemble of 2 (temp. scaled)"]
    for m in methods["val-tiles"].values():
        assert len(m["pred"]) == len(m["confidence"]) == 24


def test_make_results(workspace):
    out = workspace / "docs" / "results.md"
    out.parent.mkdir(exist_ok=True)
    command(REPO / "scripts" / "make_results.py", "--runs-dir", workspace / "runs",
            "--cache-dir", workspace / "cache", "--out", out,
            "--fig-dir", workspace / "docs" / "figures", "--base", "siamese")
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Baselines", "## 3. Siamese vs stack", "## 4. Augmentation experiment",
                    "## 5. Calibration", "## 6. Uncertainty methods", "## 7. Triage",
                    "## 8. All runs"):
        assert heading in text, heading
    assert "ensemble of 2 (temp. scaled)" in text and "MC dropout (entropy)" in text
    assert "val is now used for model selection" in text
    assert "has not been loaded" in text
    for run_name in ("siamese_s0", "siamese_s1", "stack_s0", "siamese_aug_s0", "stack_aug_s0"):
        assert f"| {run_name} |" in text
    for fig in ("confusion_matrices", "confusion_augmentation", "reliability",
                "reliability_methods", "risk_coverage", "inspection_budget"):
        assert (workspace / "docs" / "figures" / f"{fig}.png").exists(), fig


def test_error_analysis(workspace):
    out = workspace / "docs" / "error_analysis.md"
    out.parent.mkdir(exist_ok=True)
    command(REPO / "scripts" / "error_analysis.py", "--run", workspace / "runs" / "siamese_s0",
            "--cache-dir", workspace / "cache", "--out", out,
            "--fig-dir", workspace / "docs" / "figures", "--gallery", 4)
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Per disaster", "## 2. By building size", "## 3. Cut by the tile edge",
                    "## 4. Label-noise candidates", "## 5. Is uncertainty higher", "## 6. Gallery",
                    "## 7. The same questions for every uncertainty method"):
        assert heading in text
    assert "ensemble of 2" in text and "MC dropout (mean max prob)" in text
    gallery = workspace / "runs" / "siamese_s0" / "gallery"
    assert (gallery / "confident_mistakes.png").exists() or (gallery / "unsure_correct.png").exists()
