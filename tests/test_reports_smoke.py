"""End to end on a tiny fake cache: train two models, then build the reports from them."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import write_fake_cache  # noqa: E402
from src.train import Config, run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def script(name, *args):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / name), *map(str, args)],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{name} failed:\n{result.stdout}\n{result.stderr}"
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
    return root


def test_make_results(workspace):
    out = workspace / "docs" / "results.md"
    out.parent.mkdir(exist_ok=True)
    script("make_results.py", "--runs-dir", workspace / "runs", "--cache-dir", workspace / "cache",
           "--out", out, "--fig-dir", workspace / "docs" / "figures", "--seeds", 0, 1)
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Baselines", "## 3. Siamese vs stack", "## 4. Calibration",
                    "## 5. Uncertainty methods", "## 6. Triage"):
        assert heading in text
    assert "ensemble of 2" in text
    assert "The test set" in text and "has not" in text
    for fig in ("confusion_matrices", "reliability", "risk_coverage", "inspection_budget"):
        assert (workspace / "docs" / "figures" / f"{fig}.png").exists()


def test_error_analysis(workspace):
    out = workspace / "docs" / "error_analysis.md"
    out.parent.mkdir(exist_ok=True)
    script("error_analysis.py", "--run", workspace / "runs" / "siamese_s0", "--cache-dir",
           workspace / "cache", "--out", out, "--fig-dir", workspace / "docs" / "figures",
           "--gallery", 4)
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Per disaster", "## 2. By building size", "## 3. Cut by the tile edge",
                    "## 4. Label-noise candidates", "## 5. Is uncertainty higher", "## 6. Gallery"):
        assert heading in text
    gallery = workspace / "runs" / "siamese_s0" / "gallery"
    assert (gallery / "confident_mistakes.png").exists() or (gallery / "unsure_correct.png").exists()
