import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import write_fake_cache  # noqa: E402
from src.train import (Config, class_weights, lr_factor, parse_config, run,  # noqa: E402
                       selection_score, smoothed_score)


def test_class_weights_are_inverse_sqrt_frequency():
    labels = np.array([0] * 64 + [1] * 16 + [2] * 16 + [3] * 4)
    w = class_weights(labels).numpy()
    assert w.mean() == pytest.approx(1.0)
    assert w[1] / w[0] == pytest.approx(2.0)      # 4x rarer -> sqrt(4) = 2x the weight
    assert w[3] / w[0] == pytest.approx(4.0)      # 16x rarer -> 4x
    assert w[1] == pytest.approx(w[2])


def test_class_weights_ignore_absent_classes():
    w = class_weights(np.array([0, 0, 0, 1])).numpy()
    assert w[2] == 0 and w[3] == 0 and w[0] > 0


def test_lr_schedule_warms_up_then_decays_to_zero():
    total, warm = 100, 10
    f = [lr_factor(s, total, warm) for s in range(total)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(f[:10], f[1:10]))          # rising
    assert all(a >= b for a, b in zip(f[9:], f[10:]))            # then falling
    assert f[-1] < 0.01
    assert lr_factor(55, total, warm) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))


def test_config_precedence_defaults_then_file_then_flags(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"arch": "stack", "epochs": 3}))
    cfg = parse_config(["--config", str(path), "--epochs", "7", "--amp", "false"])
    assert (cfg.arch, cfg.epochs, cfg.amp, cfg.lr) == ("stack", 7, False, Config().lr)


@pytest.mark.parametrize("arch", ["siamese", "stack"])
def test_training_run_end_to_end_on_a_tiny_cache(tmp_path, arch):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    cfg = Config(arch=arch, cache_dir=str(cache), runs_dir=str(tmp_path / "runs"), epochs=2,
                 batch_size=8, eval_batch_size=8, pretrained=False, num_workers=0, amp=False)
    summary = run(cfg)
    out = tmp_path / "runs" / f"{arch}_s0"
    assert summary["selected_on"] == "mean"
    # Selection: mean of val-tiles and val macro-F1, averaged over a 3-epoch window. The
    # run has only 2 epochs, so the window shrinks to 2 and only epoch 2 can be kept.
    scores = [(h["val-tiles"]["macro_f1"] + h["val"]["macro_f1"]) / 2 for h in summary["history"]]
    assert [h["selection_score"] for h in summary["history"]] == pytest.approx(scores)
    assert summary["history"][0]["smoothed_score"] is None
    assert summary["best_epoch"] == 2
    assert summary["best_selection_score"] == pytest.approx(sum(scores) / 2)
    # RAM (and, on a GPU, GPU memory) is logged every epoch
    for h in summary["history"]:
        assert h["memory"]["ram_used_gb"] > 0 and h["memory"]["run_private_gb"] > 0
    assert summary["ram_min_available_gb"] > 0
    assert set(summary["best"]) == {"val-tiles", "val"}        # never "test"
    for name in ("config.json", "log.jsonl", "best.pt", "metrics.json",
                 "preds_val-tiles.npz", "preds_val.npz"):
        assert (out / name).exists(), name
    preds = np.load(out / "preds_val.npz")
    assert preds["logits"].shape == (24, 4) and preds["labels"].shape == (24,)
    lines = (out / "log.jsonl").read_text().splitlines()
    assert len(lines) == summary["epochs_run"]
    first = json.loads(lines[0])
    assert {"train_loss", "val-tiles", "val"} <= set(first)
    assert len(first["val"]["recall"]) == 4


def test_selection_score_rules():
    row = {"val-tiles": {"macro_f1": 0.7}, "val": {"macro_f1": 0.3}}
    assert selection_score(row, "mean") == pytest.approx(0.5)
    assert selection_score(row, "val-tiles") == 0.7
    assert selection_score(row, "val") == 0.3


def test_smoothed_score_needs_a_full_window():
    scores = [0.40, 0.50, 0.30, 0.60]
    assert smoothed_score(scores[:1], 3, 10) is None
    assert smoothed_score(scores[:2], 3, 10) is None
    assert smoothed_score(scores[:3], 3, 10) == pytest.approx(0.40)
    assert smoothed_score(scores, 3, 10) == pytest.approx((0.5 + 0.3 + 0.6) / 3)
    # a run shorter than the window: the window shrinks to the run length
    assert smoothed_score([0.4], 3, 1) == pytest.approx(0.4)
    assert smoothed_score([0.4, 0.6], 3, 2) == pytest.approx(0.5)


def test_a_single_spike_does_not_win_under_the_window(tmp_path, monkeypatch):
    # Fake the per-epoch scores: epoch 2 is a spike, epochs 4-6 are steadily good.
    import src.train as train
    fake = iter([0.30, 0.90, 0.30, 0.60, 0.62, 0.61])
    monkeypatch.setattr(train, "selection_score", lambda row, select_on: next(fake))
    cache = write_fake_cache(tmp_path / "cache", n=24)
    summary = run(Config(cache_dir=str(cache), runs_dir=str(tmp_path / "runs"), epochs=6,
                         patience=6, batch_size=8, eval_batch_size=8, pretrained=False,
                         num_workers=0, amp=False))
    assert summary["best_epoch"] == 6                     # last epoch of the best stretch
    assert summary["best_selection_score"] == pytest.approx((0.60 + 0.62 + 0.61) / 3)


def test_preprocessing_options_reach_the_datasets_and_the_run_name(tmp_path):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    cfg = Config(arch="stack", augment="strong", normalize="per_image", match_hist=True,
                 cache_dir=str(cache), runs_dir=str(tmp_path / "runs"), epochs=1, batch_size=8,
                 eval_batch_size=8, pretrained=False, num_workers=0, amp=False)
    summary = run(cfg)
    assert summary["run_name"] == "stack_aug_pernorm_histmatch_s0"
    assert summary["normalize"] == "per_image" and summary["match_hist"] is True
    saved = torch.load(tmp_path / "runs" / summary["run_name"] / "best.pt", weights_only=True)
    assert saved["config"]["normalize"] == "per_image" and saved["config"]["match_hist"] is True


def test_run_name_and_augmentation_follow_the_config(tmp_path):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    cfg = Config(arch="stack", augment="strong", select_on="val-tiles", cache_dir=str(cache),
                 runs_dir=str(tmp_path / "runs"), epochs=1, batch_size=8, eval_batch_size=8,
                 pretrained=False, num_workers=0, amp=False)
    summary = run(cfg)
    assert summary["run_name"] == "stack_aug_s0"
    assert summary["augment"] == "strong" and summary["selected_on"] == "val-tiles"
    assert (tmp_path / "runs" / "stack_aug_s0" / "best.pt").exists()


def test_overfit_mode_trains_on_a_subset_and_skips_validation(tmp_path):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    cfg = Config(cache_dir=str(cache), runs_dir=str(tmp_path / "runs"), run_name="sanity",
                 epochs=1, overfit=8, batch_size=4, pretrained=False, num_workers=0, amp=False)
    summary = run(cfg)
    assert list(summary["best"]) == ["overfit-subset"]
    assert summary["best"]["overfit-subset"]["n"] == 8
    assert not (tmp_path / "runs" / "sanity" / "preds_val.npz").exists()


def test_same_pair_option_reaches_the_training_set_only(tmp_path):
    cache = write_fake_cache(tmp_path / "cache", n=24)
    cfg = Config(arch="stack", augment="strong", same_pair_prob=0.25, cache_dir=str(cache),
                 runs_dir=str(tmp_path / "runs"), epochs=1, batch_size=8, eval_batch_size=8,
                 pretrained=False, num_workers=0, amp=False)
    summary = run(cfg)
    assert summary["run_name"] == "stack_aug_samepair_s0"
    assert summary["same_pair_prob"] == 0.25
    # class weights come from the true labels (6 of each class), so they are all equal
    assert summary["class_weights"] == pytest.approx([1.0] * 4)
    # the evaluation sets are scored on their true labels: all four classes are present
    assert all(r is not None for r in summary["best"]["val"]["recall"])
