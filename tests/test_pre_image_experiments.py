"""Tests for the pre-image experiments: the stop rule, the pre-image-use test and the
difference-only siamese model (docs/pre_image_plan.md)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import pre_image_use as piu  # noqa: E402
import stop_rule  # noqa: E402
from conftest import write_fake_cache  # noqa: E402
from src.model_diff import SiameseDiffNet  # noqa: E402
from src.split import TestSplitError  # noqa: E402
from src.train import Config, run  # noqa: E402


# ---- stop rule ---------------------------------------------------------------------------

BASE = {"val_macro_f1": 0.30, "val_accuracy": 0.60, "val_tiles_macro_f1": 0.70}


@pytest.mark.parametrize("candidate, expected", [
    ({"val_macro_f1": 0.34, "val_accuracy": 0.60, "val_tiles_macro_f1": 0.69}, "KEEP"),
    ({"val_macro_f1": 0.3299, "val_accuracy": 0.70, "val_tiles_macro_f1": 0.75}, "DROP"),  # F1 gain too small
    ({"val_macro_f1": 0.40, "val_accuracy": 0.5999, "val_tiles_macro_f1": 0.75}, "DROP"),  # accuracy fell
    ({"val_macro_f1": 0.40, "val_accuracy": 0.70, "val_tiles_macro_f1": 0.6799}, "DROP"),  # val-tiles fell too far
    (None, "DROP"),                                                                        # unfinished
])
def test_stop_rule_needs_all_three_conditions(candidate, expected):
    assert stop_rule.decide(BASE, candidate)[0] == expected


def test_stop_rule_reads_means_over_seeds_and_drops_unfinished_settings(tmp_path):
    def write(setting, seed, val_f1, val_acc, tiles_f1):
        d = tmp_path / f"{setting}_s{seed}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({
            "selected_on": "mean", "select_window": 3,
            "best": {"val": {"macro_f1": val_f1, "accuracy": val_acc},
                     "val-tiles": {"macro_f1": tiles_f1}}}))
    for seed, f1 in enumerate((0.29, 0.30, 0.31)):
        write("base", seed, f1, 0.60, 0.70)
        write("good", seed, f1 + 0.04, 0.61, 0.69)
    write("partial", 0, 0.50, 0.90, 0.90)
    assert stop_rule.setting_means(tmp_path, "base")["val_macro_f1"] == pytest.approx(0.30)
    assert stop_rule.main(["--candidate", "good", "--baseline", "base",
                           "--runs-dir", str(tmp_path)]) == "KEEP"
    assert stop_rule.main(["--candidate", "partial", "--baseline", "base",
                           "--runs-dir", str(tmp_path)]) == "DROP"


# ---- pre-image-use test ------------------------------------------------------------------

def test_swap_sources_stay_in_the_disaster_and_never_point_at_themselves():
    disasters = np.array(["a"] * 7 + ["b"] * 5 + ["c"] * 1 + ["a"] * 3)
    source = piu.swap_sources(disasters)
    many = disasters != "c"
    assert (source[many] != np.arange(len(disasters))[many]).all()
    assert (disasters[source] == disasters).all()
    assert source[disasters == "c"].tolist() == [12]              # nobody to swap with
    # every building's pre image is handed out exactly once
    assert sorted(source.tolist()) == list(range(len(disasters)))
    assert (piu.swap_sources(disasters) == source).all()          # fixed, not random per call


def test_swapped_dataset_takes_pre_from_the_source_and_keeps_post_and_label(tmp_path):
    from src.dataset import CropDataset
    data = CropDataset(write_fake_cache(tmp_path / "cache", n=8), "val")
    source = piu.swap_sources(data.index["disaster"].to_numpy())
    swapped = piu.SwappedPre(data, source)
    assert len(swapped) == 8
    for i in range(8):
        pre, post, label = swapped[i]
        assert torch.equal(pre, data[int(source[i])][0])
        assert torch.equal(post, data[i][1]) and label == data[i][2]


def test_pre_image_use_on_a_tiny_run_and_never_on_test(tmp_path):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    run(Config(arch="stack", cache_dir=str(cache), runs_dir=str(tmp_path / "runs"), epochs=1,
               batch_size=8, eval_batch_size=8, pretrained=False, num_workers=0, amp=False))
    rows = piu.test_run(tmp_path / "runs" / "stack_s0", parts=["val"], batch_size=8)
    assert len(rows) == 1 and rows[0]["buildings"] == 24
    assert rows[0]["drop_macro_f1"] == pytest.approx(
        rows[0]["macro_f1_real"] - rows[0]["macro_f1_swapped"])
    assert 0.0 <= rows[0]["changed_share"] <= 1.0
    with pytest.raises(TestSplitError):
        piu.test_run(tmp_path / "runs" / "stack_s0", parts=["test"])


# ---- difference-only siamese model ---------------------------------------------------------

def test_siamese_diff_sees_only_the_difference():
    torch.manual_seed(0)
    model = SiameseDiffNet(pretrained=False).eval()
    a, b, c = (torch.randn(2, 3, 64, 64) for _ in range(3))
    with torch.no_grad():
        same_1, same_2 = model(a, a), model(b, b)
        real, other_pre = model(a, b), model(c, b)
    assert real.shape == (2, 4)
    # the same image twice: the difference is zero, so the answer is one constant
    assert torch.allclose(same_1, same_2, atol=1e-5)
    assert torch.allclose(same_1[0], same_1[1], atol=1e-5)
    # a different pre image changes the answer
    assert not torch.allclose(real, other_pre, atol=1e-3)
    # gradients reach the backbone through both inputs
    model.train()
    model(a, b).sum().backward()
    assert model.backbone.conv1.weight.grad.abs().sum() > 0


def test_siamese_diff_trains_end_to_end_under_its_own_run_name(tmp_path):
    from src.model import build_model
    assert isinstance(build_model("siamese_diff", pretrained=False), SiameseDiffNet)
    cache = write_fake_cache(tmp_path / "cache", n=24)
    summary = run(Config(arch="siamese_diff", augment="strong", cache_dir=str(cache),
                         runs_dir=str(tmp_path / "runs"), epochs=1, batch_size=8,
                         eval_batch_size=8, pretrained=False, num_workers=0, amp=False))
    assert summary["run_name"] == "siamese_diff_aug_s0" and summary["arch"] == "siamese_diff"
    # the saved checkpoint can be rebuilt by the scripts that load models from best.pt
    rows = piu.test_run(tmp_path / "runs" / "siamese_diff_aug_s0", parts=["val"], batch_size=8)
    assert rows[0]["buildings"] == 24
