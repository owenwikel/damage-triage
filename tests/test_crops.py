import numpy as np
from PIL import Image

from src.crops import at_edge, crop_window, cut_crop


def test_window_is_square_centred_and_padded():
    left, top, right, bottom = crop_window((100, 200, 160, 220), pad=16)
    assert right - left == bottom - top == 60 + 32      # longest side + 2 * pad
    assert (left + right) / 2 == 130 and (top + bottom) / 2 == 210


def test_crop_is_resized_and_black_outside_the_image():
    img = Image.fromarray(np.full((1024, 1024, 3), 200, dtype=np.uint8))
    crop = cut_crop(img, (-32, 100, 32, 164), out=128)
    assert crop.shape == (128, 128, 3) and crop.dtype == np.uint8
    assert (crop[:, :60] == 0).all()         # left half was outside the tile
    assert (crop[:, 68:] == 200).all()


def test_crop_keeps_content_in_place():
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    arr[500:510, 300:340] = 255                          # a bright 40 x 10 "building"
    crop = cut_crop(Image.fromarray(arr), crop_window((300, 500, 340, 510)), out=72)
    ys, xs = np.nonzero(crop[..., 0] > 128)
    # window side is 72, so no resize: the building sits centred with 16 px of padding
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (16, 55, 31, 40)


def test_at_edge():
    assert at_edge((0.5, 300, 40, 340))
    assert at_edge((990, 700, 1024, 740))
    assert not at_edge((100, 100, 160, 150))
