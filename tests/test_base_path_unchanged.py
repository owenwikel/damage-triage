"""The base code path must not change when experimental options are added.

tests/golden/base_path_golden.npz was recorded from commit 7b96b05, BEFORE the pre-image
experiments (docs/pre_image_plan.md) touched src/dataset.py, src/model.py or
src/train.py. The tests below recompute the same things with the current code and
compare. If one fails, an option that should be off by default has changed what the
base model sees or computes.

To look at how the file was made: ``base_path_outputs`` below is the exact recipe. Do
not regenerate the file from changed code; that would make the test agree with itself.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import write_fake_cache  # noqa: E402
from src.dataset import CropDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import Config  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "base_path_golden.npz"


def base_path_outputs(cache_dir):
    """Everything the base path produces from fixed seeds, as {name: array}."""
    cache = write_fake_cache(cache_dir, n=8, seed=0)
    out = {}
    # 1. What the dataset hands to the model, for every augmentation setting. The crops
    #    are thinned to every 8th pixel to keep the file small; the mean covers the rest.
    for augment in (False, "base", "strong"):
        dataset = CropDataset(cache, "train", augment=augment)
        torch.manual_seed(0)                     # the augmentation's random numbers
        for i in range(4):
            pre, post, label = dataset[i]
            key = f"data_{augment}_{i}"
            out[key + "_pre"] = pre[:, ::8, ::8].numpy()
            out[key + "_post"] = post[:, ::8, ::8].numpy()
            out[key + "_stats"] = np.array([pre.mean().item(), post.mean().item(), label])
        # how far the random generator has advanced: catches an extra random draw
        out[f"rng_after_{augment}"] = torch.rand(3).numpy()
    # 2. What the two existing architectures compute from fixed weights and inputs.
    for arch in ("stack", "siamese"):
        torch.manual_seed(0)                     # the initial weights
        model = build_model(arch, pretrained=False, dropout=0.3).eval()
        torch.manual_seed(1)
        pre, post = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out[f"logits_{arch}"] = model(pre, post).numpy()
        out[f"n_weights_{arch}"] = np.array(sum(p.numel() for p in model.parameters()))
    return out


def test_base_path_gives_the_recorded_outputs(tmp_path):
    golden = np.load(GOLDEN)
    current = base_path_outputs(tmp_path / "cache")
    assert set(current) == set(golden.files)
    for name in golden.files:
        # Data and random numbers are simple arithmetic and must agree closely; logits go
        # through many convolutions, whose last digits differ between machines.
        tolerance = 1e-3 if name.startswith("logits") else 1e-5
        assert current[name] == pytest.approx(golden[name], abs=tolerance), name


def test_experimental_options_are_off_by_default():
    cfg = Config()
    assert (cfg.arch, cfg.augment, cfg.normalize, cfg.match_hist) == (
        "siamese", "base", "imagenet", False)
    # Options added by later experiments: every one must default to "off".
    for name, off in (("same_pair_prob", 0.0),):
        if hasattr(cfg, name):
            assert getattr(cfg, name) == off, name
