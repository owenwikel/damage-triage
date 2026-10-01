"""PyTorch Dataset over the crop cache written by scripts/build_crops.py.

Background for a PyTorch newcomer
---------------------------------
PyTorch separates "what is one training example?" from "how do I feed examples to the
model?":

  * a ``Dataset`` answers the first question. It needs just two methods:
    ``__len__`` (how many examples) and ``__getitem__(i)`` (give me example number i).
  * a ``DataLoader`` (built in src/train.py) answers the second: it asks the Dataset for
    examples in a shuffled order, in several background processes, and glues them into
    batches.

One example here is a triple ``(pre, post, label)``:
    pre, post   float32 tensors of shape [3, 128, 128] (channels, height, width)
    label       an integer 0..3 (no-damage, minor-damage, major-damage, destroyed)

Note the axis order. Image files and numpy use height x width x channels ("HWC");
PyTorch's convolution layers expect channels first ("CHW"). ``__getitem__`` converts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.split import require_final

# The pretrained ResNet was trained on ImageNet photos that had been shifted and scaled
# with exactly these per-channel (red, green, blue) numbers. Its filters expect inputs in
# that range, so we must apply the same normalisation or the pretrained weights are
# working on data that "looks wrong" to them.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def read_index(cache_dir, part=None):
    """Read index.csv (one row per building); optionally keep one part, in array order."""
    index = pd.read_csv(Path(cache_dir) / "index.csv", dtype={"tile_id": str})
    if part is not None:
        index = index[index["part"] == part].sort_values("row").reset_index(drop=True)
        # ``row`` must be 0, 1, 2, ...: position i of the index is position i of the arrays.
        assert (index["row"].to_numpy() == np.arange(len(index))).all(), "index out of step"
    return index


def augment_pair(pre, post):
    """Randomly flip, rotate and slightly recolour a (pre, post) pair. Tensors in [0, 1].

    Augmentation shows the network altered copies of each example so it cannot simply
    memorise the training images; it has to learn what stays the same under the changes.

    WHICH changes are safe depends on the data:
      * Flips and 90-degree rotations. A satellite looks straight down, so there is no
        "up": a house rotated by 90 degrees is an equally plausible house. (For photos of
        people or street scenes a vertical flip would create nonsense.) These changes
        also cost nothing in quality, because they only move pixels around, unlike
        arbitrary-angle rotations, which have to blend neighbouring pixels.
      * The SAME flip/rotation must be applied to pre and post. The model compares the
        two images; if they were turned differently, the comparison would be meaningless.
      * Colour jitter is drawn SEPARATELY for pre and post, and kept light. The two
        images were taken on different days, often by different satellites, so brightness
        and contrast already differ between them for reasons unrelated to damage. Small
        independent changes teach the model not to read "post is a bit darker" as
        damage. Kept small (about +-10%) because colour can also be real signal: burnt
        ground and flood water are genuinely darker.

    Random numbers come from torch's generator. The DataLoader gives each worker process
    its own seed derived from the run's seed, so the augmentation is reproducible.
    """
    if torch.rand(1).item() < 0.5:                     # left-right flip
        pre, post = pre.flip(-1), post.flip(-1)        # -1 = last axis = width
    if torch.rand(1).item() < 0.5:                     # up-down flip
        pre, post = pre.flip(-2), post.flip(-2)        # -2 = height
    k = int(torch.randint(0, 4, (1,)).item())          # 0, 90, 180 or 270 degrees
    if k:
        pre, post = torch.rot90(pre, k, dims=(-2, -1)), torch.rot90(post, k, dims=(-2, -1))
    return _jitter(pre), _jitter(post)


def _jitter(img, strength=0.1):
    """Scale brightness and contrast by random factors in [1 - strength, 1 + strength]."""
    brightness, contrast = (1 + strength * (2 * torch.rand(2) - 1)).tolist()
    mean = img.mean()
    # contrast: stretch or squeeze values around the image's mean; brightness: scale all
    img = ((img - mean) * contrast + mean) * brightness
    return img.clamp(0.0, 1.0)


class CropDataset(Dataset):
    """Buildings of one split part, as (pre, post, label) triples.

    cache_dir   folder written by scripts/build_crops.py
    part        "train", "val-tiles" or "val" ("test" needs final=True)
    augment     True for training only. Validation must see the same images every epoch,
                otherwise the score would wobble for reasons unrelated to the model.
    final       must be True to load the test part; see src/split.py for why.
    """

    def __init__(self, cache_dir, part, augment=False, final=False):
        require_final(part, final)          # raises TestSplitError for "test" by default
        self.cache_dir = Path(cache_dir)
        self.part = part
        self.augment = augment
        self.index = read_index(cache_dir, part)
        self.labels = self.index["label"].to_numpy(dtype=np.int64)
        for name in ("pre", "post"):
            if not (self.cache_dir / f"{part}_{name}.npy").exists():
                raise FileNotFoundError(f"No crop cache for part '{part}' in {cache_dir}; "
                                        "run scripts/build_crops.py first")
        # The arrays are opened lazily (see _open). On Windows, DataLoader workers are
        # new processes that receive a pickled copy of this object; an open file handle
        # cannot be pickled, but a path can.
        self._pre = self._post = None

    def _open(self):
        # mmap_mode="r" maps the file into memory without reading it: only the crops we
        # actually index are fetched from disk, and the operating system caches them.
        self._pre = np.load(self.cache_dir / f"{self.part}_pre.npy", mmap_mode="r")
        self._post = np.load(self.cache_dir / f"{self.part}_post.npy", mmap_mode="r")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self._pre is None:
            self._open()
        # np.array(...) copies the crop out of the read-only file; then HWC uint8 0..255
        # becomes CHW float 0..1.
        pre = torch.from_numpy(np.array(self._pre[i])).permute(2, 0, 1).float() / 255.0
        post = torch.from_numpy(np.array(self._post[i])).permute(2, 0, 1).float() / 255.0
        if self.augment:
            pre, post = augment_pair(pre, post)
        pre = (pre - IMAGENET_MEAN) / IMAGENET_STD
        post = (post - IMAGENET_MEAN) / IMAGENET_STD
        return pre, post, int(self.labels[i])


def unnormalize(img):
    """Undo the ImageNet normalisation: CHW tensor -> HWC float array in [0, 1] to plot."""
    img = img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
