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
sys.path.insert(0, str(REPO / "scripts"))
import make_results  # noqa: E402


def command(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=REPO,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{args[0]} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("reports")
    cache = write_fake_cache(root / "cache", n=24)
    common = dict(arch="stack", augment="strong", cache_dir=str(cache),
                  runs_dir=str(root / "runs"), epochs=1, batch_size=8, eval_batch_size=8,
                  pretrained=False, num_workers=0, amp=False)
    for seed in (0, 1):
        run(Config(seed=seed, **common))                                  # stack_aug
        run(Config(seed=seed, normalize="per_image", **common))           # stack_aug_pernorm
    run(Config(seed=0, match_hist=True, **common))                        # stack_aug_histmatch
    return root


def test_collect_methods(workspace):
    runs = workspace / "runs"
    methods, info = collect_methods(runs / "stack_aug_s0", [runs / f"stack_aug_s{s}" for s in (0, 1, 2)])
    assert info["members"] == ["stack_aug_s0", "stack_aug_s1"]        # seed 2 was never trained
    # MC dropout is set aside: it is not among the methods even if its files existed
    assert list(methods["val"]) == ["max prob (raw)", "max prob (temp. scaled)",
                                    "ensemble of 2", "ensemble of 2 (temp. scaled)"]
    for m in methods["val-tiles"].values():
        assert len(m["pred"]) == len(m["confidence"]) == 24
    alone, info = collect_methods(runs / "stack_aug_s0", [])
    assert list(alone["val"]) == ["max prob (raw)", "max prob (temp. scaled)"]


def test_make_results(workspace):
    out = workspace / "docs" / "results.md"
    out.parent.mkdir(exist_ok=True)
    command(REPO / "scripts" / "make_results.py", "--runs-dir", workspace / "runs",
            "--cache-dir", workspace / "cache", "--out", out,
            "--fig-dir", workspace / "docs" / "figures", "--seeds", 0, 1)
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Baselines", "## 2. Settings compared", "## 3. Does uncertainty flag",
                    "## 4. Why is val worse", "## 5. Calibration and triage",
                    "## 6. MC dropout: tried and set aside", "## 7. Earlier single-run",
                    "## 8. All runs"):
        assert heading in text, heading
    assert "val is used for model selection" in text and "has not been loaded" in text
    assert "base + per-image normalisation" in text and "histogram-matched" in text
    assert "(1 seed)" in text                    # the histogram-matching setting has one run
    assert " ± " in text
    for run_name in ("stack_aug_s0", "stack_aug_s1", "stack_aug_pernorm_s1", "stack_aug_histmatch_s0"):
        assert f"| {run_name} |" in text
    for fig in ("confusion_matrices", "confidence_right_wrong", "reliability_methods",
                "risk_coverage", "inspection_budget"):
        assert (workspace / "docs" / "figures" / f"{fig}.png").exists(), fig


def test_flagging_statistics():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
    right = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)   # the 3 least confident are wrong
    assert make_results.auroc(conf, right) == pytest.approx(1.0)
    assert make_results.auroc(-conf, right) == pytest.approx(0.0)
    assert make_results.auroc(np.ones(10), right) == pytest.approx(0.5)         # all tied
    assert make_results.errors_caught(conf, right, 0.20) == pytest.approx(2 / 3)
    assert make_results.errors_caught(conf, right, 0.50) == pytest.approx(1.0)
    assert make_results.spread([0.5]) == "0.500 (1 seed)"
    assert make_results.spread([0.4, 0.5, 0.6]) == "0.500 ± 0.100 (0.400-0.600)"


def test_error_analysis(workspace):
    out = workspace / "docs" / "error_analysis.md"
    out.parent.mkdir(exist_ok=True)
    command(REPO / "scripts" / "error_analysis.py", "--run", workspace / "runs" / "stack_aug_s0",
            "--cache-dir", workspace / "cache", "--out", out,
            "--fig-dir", workspace / "docs" / "figures", "--gallery", 4)
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Per disaster", "## 2. By building size", "## 3. Cut by the tile edge",
                    "## 4. Label-noise candidates", "## 5. Is uncertainty higher", "## 6. Gallery",
                    "## 7. The same questions for every uncertainty method"):
        assert heading in text
    assert "ensemble of 2" in text and "MC dropout" not in text
    gallery = workspace / "runs" / "stack_aug_s0" / "gallery"
    assert (gallery / "confident_mistakes.png").exists() or (gallery / "unsure_correct.png").exists()


def test_gap_diagnosis(workspace):
    out = workspace / "docs" / "gap_diagnosis.md"
    out.parent.mkdir(exist_ok=True)
    runs = workspace / "runs"
    command(REPO / "scripts" / "gap_diagnosis.py", "--runs", runs / "stack_aug_s0",
            runs / "stack_aug_s1", "--cache-dir", workspace / "cache", "--out", out,
            "--fig-dir", workspace / "docs" / "figures", "--per-group", 10)
    text = out.read_text(encoding="utf-8")
    for variant in ("normal", "both = pre", "both = post", "pre blank", "post blank"):
        assert f"| val (all) | {variant} |" in text
    assert "## (b) Colour statistics per disaster" in text
    assert (workspace / "docs" / "figures" / "colour_statistics.png").exists()
