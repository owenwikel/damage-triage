"""End-to-end smoke tests: run each script on the synthetic data from conftest.py."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(script, *args, cwd):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / script), *map(str, args)],
                            cwd=cwd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def test_inspect_xbd(fake_xbd, tmp_path):
    out = tmp_path / "report.md"
    run("inspect_xbd.py", "--data-root", fake_xbd / "train", "--out", out, cwd=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "| images | 2 | 1 | 1 |" in text
    assert "`destroyed` | 1 |" in text


def test_dataset_stats(fake_xbd, tmp_path):
    out = tmp_path / "stats"
    run("dataset_stats.py", "--data-root", fake_xbd / "train", "--out-dir", out, cwd=tmp_path)
    assert (out / "summary.md").exists()
    assert len(list(out.glob("*.png"))) == 5


def test_visualize(fake_xbd, tmp_path):
    out = tmp_path / "viz"
    stdout = run("visualize.py", "--data-root", fake_xbd / "train", "--out-dir", out, cwd=tmp_path)
    assert [p.name for p in out.glob("*.png")] == ["fake-fire_00000001.png"]
    assert "[destroyed: 1]" in stdout


def test_crop_sizes(fake_xbd, tmp_path):
    doc, fig = tmp_path / "dq.md", tmp_path / "crops.png"
    # only 3 buildings, so ask for as many examples as exist
    run("crop_sizes.py", "--data-dir", fake_xbd, "--doc", doc, "--figure", fig, cwd=tmp_path)
    assert "## 7. Crop sizes" in doc.read_text(encoding="utf-8")
    assert fig.exists()


def test_audit_data(fake_xbd, tmp_path):
    doc = tmp_path / "dq.md"
    run("audit_data.py", "--data-dir", fake_xbd, "--doc", doc, "--csv-dir", tmp_path / "csv",
        "--shift-per-disaster", 1, "--null-pairs", 2, cwd=tmp_path)
    text = doc.read_text(encoding="utf-8")
    assert "## 6. Pre/post image misalignment" in text
    assert "Tiles where pre and post list a different number of buildings: **0**" in text


def test_split_candidates(fake_xbd, tmp_path):
    out = tmp_path / "splits.md"
    run("split_candidates.py", "--data-dir", fake_xbd, "--out", out, cwd=tmp_path)
    assert "No candidate is chosen here" in out.read_text(encoding="utf-8")
