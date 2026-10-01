import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import write_fake_cache  # noqa: E402
from src.dataset import CropDataset, augment_pair, unnormalize  # noqa: E402
from src.split import TestSplitError  # noqa: E402


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return write_fake_cache(tmp_path_factory.mktemp("cache"), n=24,
                            parts=("train", "val-tiles", "val", "test"))


def test_item_shapes_types_and_labels(cache):
    ds = CropDataset(cache, "train")
    assert len(ds) == 24
    pre, post, label = ds[3]
    assert pre.shape == post.shape == (3, 128, 128)
    assert pre.dtype == torch.float32
    assert label == 3 and isinstance(label, int)


def test_pixels_match_the_cache_after_unnormalizing(cache):
    ds = CropDataset(cache, "val")
    pre, post, _ = ds[5]
    stored = np.load(cache / "val_pre.npy")[5] / 255.0
    assert np.allclose(unnormalize(pre), stored, atol=1e-5)
    assert not np.allclose(unnormalize(post), unnormalize(ds[6][1]))


def test_test_part_needs_final(cache):
    with pytest.raises(TestSplitError):
        CropDataset(cache, "test")
    assert len(CropDataset(cache, "test", final=True)) == 24


def test_missing_part_is_a_clear_error(tmp_path):
    write_fake_cache(tmp_path, n=4, parts=("train",))
    with pytest.raises(FileNotFoundError, match="build_crops"):
        CropDataset(tmp_path, "val")


def test_no_augmentation_means_identical_reads(cache):
    ds = CropDataset(cache, "val-tiles")
    assert torch.equal(ds[0][0], ds[0][0])


def test_augmentation_moves_pre_and_post_together():
    # A single bright pixel at the same place in pre and post must stay at the same
    # place in both, whatever flip/rotation is drawn.
    torch.manual_seed(0)
    seen = set()
    for _ in range(60):
        pre = torch.full((3, 16, 16), 0.2)
        post = torch.full((3, 16, 16), 0.2)
        pre[:, 2, 5] = post[:, 2, 5] = 1.0
        a, b = augment_pair(pre, post)
        assert a.shape == b.shape == (3, 16, 16)
        assert a[0].argmax() == b[0].argmax()
        assert 0.0 <= a.min() and a.max() <= 1.0
        seen.add(int(a[0].argmax()))
    assert len(seen) == 8        # 4 rotations x mirrored or not: all 8 symmetries appear


def test_augmented_dataset_changes_the_image_but_not_the_label(cache):
    torch.manual_seed(1)
    plain, aug = CropDataset(cache, "train"), CropDataset(cache, "train", augment=True)
    assert aug[2][2] == plain[2][2]
    assert not torch.equal(aug[2][0], plain[2][0])


def test_works_with_a_dataloader(cache):
    loader = torch.utils.data.DataLoader(CropDataset(cache, "train"), batch_size=8)
    pre, post, label = next(iter(loader))
    assert pre.shape == (8, 3, 128, 128) and label.dtype == torch.int64
