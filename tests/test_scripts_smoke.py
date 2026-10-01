"""End-to-end smoke tests: run each script on the synthetic data from conftest.py."""

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import write_fake_split

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


def test_make_split(tmp_path):
    # make_split.py only accepts the real disaster names, so it gets its own fake data
    data = tmp_path / "xbd"
    write_fake_split(data / "train", "hurricane-harvey", "00000001", "flooding")
    write_fake_split(data / "train", "hurricane-florence", "00000002", "flooding")
    write_fake_split(data / "test", "hurricane-michael", "00000003", "wind")
    out, doc = tmp_path / "splits" / "split.csv", tmp_path / "split.md"
    run("make_split.py", "--data-dir", data, "--out", out, "--doc", doc, cwd=tmp_path)
    assert out.read_text(encoding="utf-8").splitlines() == [
        "disaster,tile_id,part",
        "hurricane-florence,00000002,val",
        "hurricane-harvey,00000001,train",  # a disaster's only tile is never held out
        "hurricane-michael,00000003,test",
    ]
    assert (tmp_path / "splits" / "split_blocks.csv").exists()
    assert "## Final split" in doc.read_text(encoding="utf-8")


def test_disaster_table(fake_xbd, tmp_path):
    out = tmp_path / "disasters.md"
    run("disaster_table.py", "--data-dir", fake_xbd, "--out", out, cwd=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "| fake-fire | fire (2) | 1 |" in text
    assert "FAKESAT" in text


def test_label_noise_with_no_candidates(fake_xbd, tmp_path):
    # the synthetic tiles have fewer than k+1 buildings, so nothing is eligible
    doc = tmp_path / "noise.md"
    run("label_noise.py", "--data-dir", fake_xbd, "--doc", doc, "--out-dir", tmp_path / "out",
        cwd=tmp_path)
    assert "Nothing has been relabelled or removed" in doc.read_text(encoding="utf-8")


def test_label_noise_finds_isolated_building(fake_xbd, tmp_path):
    # with k=1, the destroyed building's nearest classified neighbour is no-damage
    doc = tmp_path / "noise.md"
    run("label_noise.py", "--data-dir", fake_xbd, "--doc", doc, "--out-dir", tmp_path / "out",
        "--k", 1, "--gallery", 2, cwd=tmp_path)
    csv = (tmp_path / "out" / "candidates.csv").read_text(encoding="utf-8")
    assert "uid-b" in csv and "uid-a" in csv
    assert (tmp_path / "out" / "gallery_01.png").exists()
