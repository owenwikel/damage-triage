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


def augment_pair(pre, post, strength="base"):
    """Randomly flip, rotate and recolour a (pre, post) pair. Tensors in [0, 1].

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
      * Colour jitter is drawn SEPARATELY for pre and post. The two images were taken on
        different days, often by different satellites, so brightness and contrast
        already differ between them for reasons unrelated to damage. Independent changes
        teach the model not to read "post is a bit darker" as damage.

    Two strengths:
      "base"    brightness and contrast within about +-10%. Careful, because colour can
                also be real signal: burnt ground and flood water are genuinely darker.
      "strong"  the experiment for the val gap (docs/results.md). The first model learned
                the training disasters but failed on unseen ones, whose images are
                visibly darker and greener. So: brightness, contrast and saturation
                within +-30%, a random colour cast (each of red, green, blue scaled by
                +-10%), and the post image shifted against the pre image by up to 4
                pixels in each direction, to imitate the misalignment between the two
                captures (docs/data_quality.md section 6). The bet: if the model cannot
                rely on exact colours or exact alignment, it has to rely on structure,
                which should carry over to a new disaster better.

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
    if strength == "strong":
        post = _shift(post, max_shift=4)
        return (_jitter(pre, 0.3, saturation=True, cast=0.1),
                _jitter(post, 0.3, saturation=True, cast=0.1))
    return _jitter(pre, 0.1), _jitter(post, 0.1)


def _jitter(img, strength=0.1, saturation=False, cast=0.0):
    """Random brightness and contrast (and optionally saturation and a colour cast).

    Every factor is drawn uniformly from [1 - strength, 1 + strength].
    """
    brightness, contrast, sat = (1 + strength * (2 * torch.rand(3) - 1)).tolist()
    # contrast: stretch or squeeze values around the image's mean; brightness: scale all
    mean = img.mean()
    img = ((img - mean) * contrast + mean) * brightness
    if saturation:
        # saturation: move every pixel towards (or away from) its own grey value
        grey = img.mean(dim=0, keepdim=True)
        img = (img - grey) * sat + grey
    if cast:
        # colour cast: scale red, green and blue separately, as a different sensor or a
        # different time of day would
        img = img * (1 + cast * (2 * torch.rand(3, 1, 1) - 1))
    return img.clamp(0.0, 1.0)


def _shift(img, max_shift=4):
    """Move the image by a random whole number of pixels, up to ``max_shift`` each way.

    The image is padded by repeating its border pixels and then cut back to its original
    size at an offset, so nothing wraps around and no black bars appear.

    The shift is in pixels of the 128x128 crop, not of the original tile. A crop of a
    typical building is magnified about 2x, so 4 crop pixels are about 2 tile pixels.
    """
    dx, dy = torch.randint(-max_shift, max_shift + 1, (2,)).tolist()
    if dx == 0 and dy == 0:
        return img
    h, w = img.shape[-2:]
    padded = torch.nn.functional.pad(img.unsqueeze(0), (max_shift,) * 4, mode="replicate")[0]
    top, left = max_shift + dy, max_shift + dx
    return padded[:, top:top + h, left:left + w]


def match_histogram(image, reference, bins=256):
    """Recolour ``image`` so that each colour channel has the same distribution of values
    as in ``reference``. Both are CHW float tensors in [0, 1]; returns a new tensor.

    This is classic histogram matching. Per channel:
      1. count how many pixels have each of the 256 levels, in both images;
      2. turn the counts into cumulative shares (the CDF): "what share of pixels is at or
         below this level?";
      3. send each level of ``image`` to the level of ``reference`` that has the same
         cumulative share. The darkest 10% of pixels in ``image`` get the values of the
         darkest 10% in ``reference``, and so on.

    Used for one of the two "cheap fixes" for the val gap: the post image is matched to
    the pre image, so that a difference in overall colour between the two captures
    (another sensor, another season, haze) is removed before the model sees them. What
    is left is WHERE things are bright or dark, which is the structure we want compared.
    The risk: real damage also changes colour (burn scars, flood water), and that part
    of the signal is flattened too.
    """
    channels = image.shape[0]
    # Levels 0..255 as integers, one row per channel: [channels, pixels]
    src = (image * (bins - 1)).round().long().reshape(channels, -1)
    ref = (reference * (bins - 1)).round().long().reshape(channels, -1)
    # All channels are counted in one bincount call by giving each channel its own block
    # of 256 numbers (channel 0 -> 0..255, channel 1 -> 256..511, ...). One call for
    # three channels is much faster than three calls: this function runs for every
    # training example.
    offset = torch.arange(channels).view(channels, 1) * bins
    src_cdf = torch.bincount((src + offset).flatten(), minlength=channels * bins)
    ref_cdf = torch.bincount((ref + offset).flatten(), minlength=channels * bins)
    src_cdf = src_cdf.view(channels, bins).cumsum(1).float() / src.shape[1]
    ref_cdf = ref_cdf.view(channels, bins).cumsum(1).float() / ref.shape[1]
    # Lookup table per channel: for each source level, the first reference level whose
    # cumulative share reaches the source level's cumulative share.
    lut = torch.searchsorted(ref_cdf, src_cdf).clamp_(max=bins - 1)
    matched = torch.gather(lut, 1, src).float() / (bins - 1)
    return matched.view_as(image)


def normalize_image(img, mode="imagenet"):
    """Shift and scale a CHW image in [0, 1] before it goes into the network.

    "imagenet"   subtract ImageNet's channel means, divide by its channel standard
                 deviations: the same fixed numbers for every image. An image that is
                 dark overall stays dark relative to the others.
    "per_image"  subtract THIS image's own channel means and divide by its own channel
                 standard deviations. Every image then has mean 0 and spread 1 in each
                 channel, so overall brightness, contrast and colour cast are removed:
                 a dark, green-tinted disaster looks like a bright, grey one. The other
                 "cheap fix" for the val gap. The risk is the same as for histogram
                 matching: overall darkening can be real damage.
    """
    img = img.contiguous()
    if mode == "per_image":
        flat = img.view(img.shape[0], -1)                 # [channels, pixels]
        mean = flat.mean(dim=1).view(-1, 1, 1)
        centred = img - mean
        # The standard deviation by hand (root of the mean squared distance from the
        # mean): on this machine torch.std took 1.5 ms per image, this takes 0.1 ms.
        # clamp: a blank crop has spread 0 and must not be divided by it.
        std = centred.pow(2).view(img.shape[0], -1).mean(dim=1).sqrt().clamp_min(0.02)
        return centred / std.view(-1, 1, 1)
    if mode == "imagenet":
        return (img - IMAGENET_MEAN) / IMAGENET_STD
    raise ValueError(f"Unknown normalisation {mode!r}; use 'imagenet' or 'per_image'")


class CropDataset(Dataset):
    """Buildings of one split part, as (pre, post, label) triples.

    cache_dir   folder written by scripts/build_crops.py
    part        "train", "val-tiles" or "val" ("test" needs final=True)
    augment     False, "base" (True means the same) or "strong"; see ``augment_pair``.
                For training only. Validation must see the same images every epoch,
                otherwise the score would wobble for reasons unrelated to the model.
    final       must be True to load the test part; see src/split.py for why.
    normalize   "imagenet" (default) or "per_image"; see ``normalize_image``.
    match_hist  True: recolour the post image to the pre image's colour distribution
                (``match_histogram``) before normalising.

    ``normalize`` and ``match_hist`` change what the model sees, so they must be the same
    in training and in every later evaluation of that model. ``from_config`` builds the
    dataset from a run's saved config to make that automatic.
    """

    def __init__(self, cache_dir, part, augment=False, final=False, normalize="imagenet",
                 match_hist=False):
        require_final(part, final)          # raises TestSplitError for "test" by default
        if normalize not in ("imagenet", "per_image"):
            raise ValueError(f"Unknown normalisation {normalize!r}; use 'imagenet' or 'per_image'")
        self.normalize = normalize
        self.match_hist = bool(match_hist)
        self.cache_dir = Path(cache_dir)
        self.part = part
        self.augment = "base" if augment is True else augment
        if self.augment in (None, "none"):
            self.augment = False
        if self.augment not in (False, "base", "strong"):
            raise ValueError(f"Unknown augmentation {augment!r}; use False, 'base' or 'strong'")
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
        pre, post = self._to_tensor(self._pre[i]), self._to_tensor(self._post[i])
        if self.augment:
            pre, post = augment_pair(pre, post, self.augment)
        if self.match_hist:
            # After the augmentation, so that training and evaluation agree: the model
            # always receives a post image carrying the pre image's colours.
            post = match_histogram(post, pre)
        # A flipped or rotated tensor is a "view" that reads the same memory in a
        # different order; normalize_image makes it contiguous first (see _to_tensor).
        return (normalize_image(pre, self.normalize), normalize_image(post, self.normalize),
                int(self.labels[i]))

    @classmethod
    def from_config(cls, config, part, augment=False, final=False):
        """The dataset as a finished run saw it: ``config`` is the dict saved in best.pt
        or config.json. Runs from before these options existed lack the keys and get the
        defaults they were trained with."""
        return cls(config["cache_dir"], part, augment=augment, final=final,
                   normalize=config.get("normalize", "imagenet"),
                   match_hist=config.get("match_hist", False))

    @staticmethod
    def _to_tensor(crop):
        """HWC uint8 (0..255) from the cache -> CHW float (0..1).

        np.array(...) copies the crop out of the read-only file. ``.contiguous()``
        matters for speed: ``permute`` does not move any data, it only changes the order
        in which the axes are read, and arithmetic on such a tensor is several times
        slower. Making it contiguous first cut the time per item from about 2 ms to a
        fraction of that, which is what lets two loader workers keep up with the GPU.
        """
        return torch.from_numpy(np.array(crop)).permute(2, 0, 1).contiguous().float().div_(255.0)


def unnormalize(img):
    """Undo the ImageNet normalisation: CHW tensor -> HWC float array in [0, 1] to plot."""
    img = img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
