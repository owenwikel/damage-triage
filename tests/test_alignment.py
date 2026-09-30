import numpy as np
import pytest

from src.alignment import (estimate_shift, fill_nodata, patch_shift, phase_correlation,
                           to_gray)


def texture(size, seed=0):
    """Random but smooth grayscale texture (sum of a few blurred noise scales), 0..255."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    for scale in (4, 16, 64):
        coarse = rng.normal(size=(size // scale + 2, size // scale + 2))
        img += np.kron(coarse, np.ones((scale, scale)))[:size, :size]
    img = (img - img.min()) / (img.max() - img.min())
    return 20 + 200 * img


def shifted_pair(dy, dx, size=256, pad=40, seed=0):
    """(a, b) crops of one bigger texture, where b is a shifted by (dy, dx), no wrap-around."""
    big = texture(size + 2 * pad, seed)
    a = big[pad:pad + size, pad:pad + size]
    b = big[pad - dy:pad - dy + size, pad - dx:pad - dx + size]
    return a, b


@pytest.mark.parametrize("dy,dx", [(0, 0), (3, -5), (-7, 2), (12, 11)])
def test_phase_correlation_recovers_shift(dy, dx):
    a, b = shifted_pair(dy, dx)
    got_dy, got_dx, psr = phase_correlation(a, b)
    assert (got_dy, got_dx) == pytest.approx((dy, dx), abs=0.25)
    assert psr > 10


def test_phase_correlation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        phase_correlation(np.zeros((4, 4)), np.zeros((4, 5)))


@pytest.mark.parametrize("dy,dx", [(0, 0), (2, -3), (-6, 5)])
def test_patch_shift_recovers_shift(dy, dx):
    a, b = shifted_pair(dy, dx, size=128)
    assert patch_shift(a, b, max_shift=16) == (dy, dx)


def test_patch_shift_never_exceeds_search_window():
    a, b = shifted_pair(0, 0, size=128, seed=1)
    b = texture(128, seed=2)  # unrelated image: whatever it returns must stay in the window
    dy, dx = patch_shift(a, b, max_shift=4)
    assert abs(dy) <= 4 and abs(dx) <= 4


def test_fill_nodata_replaces_black_with_valid_mean():
    g = np.array([[0.0, 100.0], [200.0, 0.0]])
    out = fill_nodata(g)
    assert out.tolist() == [[150.0, 100.0], [200.0, 150.0]]
    assert g[0, 0] == 0.0  # input untouched


def test_estimate_shift_on_rgb_tile_with_nodata():
    a, b = shifted_pair(4, -3, size=512, seed=3)
    rgb_a = np.repeat(a[..., None], 3, axis=2).astype(np.uint8)
    rgb_b = np.repeat(b[..., None], 3, axis=2).astype(np.uint8)
    rgb_a[:, :100] = 0  # a black no-data strip on the left of both images
    rgb_b[:, :100] = 0
    est = estimate_shift(rgb_a, rgb_b)
    assert (est["dy"], est["dx"]) == (4, -3)
    assert est["patch_agreement"] > 0.9
    assert est["nodata_frac"] == pytest.approx(100 / 512)
    assert est["n_patches"] > 0


def test_to_gray_weights():
    assert to_gray(np.array([[[255, 255, 255]]], dtype=np.uint8))[0, 0] == pytest.approx(255)
