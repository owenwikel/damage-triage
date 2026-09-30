"""Estimate the translation between two images with phase correlation.

Phase correlation finds the single (dy, dx) shift that best lines up two images. It works
in the frequency domain: a shift in space is a phase change in the Fourier transform, so
normalising away the magnitudes leaves a spectrum whose inverse transform is a sharp peak
at the shift.

Limitations: it measures one global translation per image pair, so it can't see local
misalignment (e.g. tall buildings leaning differently because the two captures were taken
from different viewing angles). And when the two images differ a lot (heavy damage,
clouds, flooding, different sensors), the peak gets weak and the estimate unreliable, which
is what the peak-to-sidelobe ratio (PSR) measures.
"""

import numpy as np

# Pixels this dark in grayscale are treated as no-data (the black areas outside the swath).
NODATA_MAX = 1.0


def to_gray(rgb):
    """(H, W, 3) uint8 RGB -> (H, W) float64 luminance."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def fill_nodata(gray):
    """Replace no-data pixels with the mean of the valid ones.

    Otherwise the sharp edge of a black no-data region dominates the correlation and pulls
    the estimate towards wherever the two edges line up.
    """
    gray = np.array(gray, dtype=np.float64)
    valid = gray > NODATA_MAX
    gray[~valid] = gray[valid].mean() if valid.any() else 0.0
    return gray


def correlation_surface(a, b):
    """Normalised cross-power spectrum, transformed back to image space."""
    h, w = a.shape
    # A Hann window fades the borders to zero so the image edges don't dominate.
    window = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-12
    return np.fft.ifft2(cross).real


def peak_to_sidelobe(r, iy, ix, exclude=5):
    """(peak - mean) / std of the surface, ignoring a (2*exclude+1)^2 box around the peak.

    For pure noise the highest of ~1M values sits about 5 standard deviations up, so a PSR
    well above that (say >= 10) means a real, distinct peak.
    """
    h, w = r.shape
    mask = np.ones_like(r, dtype=bool)
    ys = np.arange(iy - exclude, iy + exclude + 1) % h
    xs = np.arange(ix - exclude, ix + exclude + 1) % w
    mask[np.ix_(ys, xs)] = False
    side = r[mask]
    return float((r[iy, ix] - side.mean()) / (side.std() + 1e-12))


def _subpixel(left, centre, right, i):
    """Parabolic peak refinement along one axis."""
    denom = left - 2 * centre + right
    offset = 0.0 if denom == 0 else 0.5 * (left - right) / denom
    return i + float(np.clip(offset, -0.5, 0.5))


def phase_correlation(a, b):
    """Return (dy, dx, psr) such that ``b`` is approximately ``a`` shifted by (dy, dx).

    ``a`` and ``b`` are 2-D grayscale arrays of the same shape. Positive dy = down, positive
    dx = right, refined to sub-pixel precision. ``psr`` is the peak-to-sidelobe ratio (see
    ``peak_to_sidelobe``): higher = more trustworthy.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"need two 2-D arrays of equal shape, got {a.shape} and {b.shape}")
    h, w = a.shape
    r = correlation_surface(a, b)
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    dy = _subpixel(r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix], iy)
    dx = _subpixel(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w], ix)
    # The FFT is periodic: a peak at index h-2 means a shift of -2.
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx, peak_to_sidelobe(r, iy, ix)


def patch_shift(a, b, max_shift=16, lowpass=0.15):
    """Integer (dy, dx) shift of ``b`` relative to ``a``, searched only within +-max_shift.

    Same phase correlation as above, with two changes that suit small patches: a Gaussian
    low-pass on the cross-power spectrum (suppresses pixel noise, which dominates small
    patches) and a bounded search window (co-registered imagery shouldn't be off by more).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    cross = np.fft.fft2((b - b.mean()) * window) * np.conj(np.fft.fft2((a - a.mean()) * window))
    cross /= np.abs(cross) + 1e-12
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    cross *= np.exp(-(fx ** 2 + fy ** 2) / (2 * lowpass ** 2))
    r = np.fft.fftshift(np.fft.ifft2(cross).real)  # zero shift now sits at the centre
    cy, cx = h // 2, w // 2
    sub = r[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return int(iy - max_shift), int(ix - max_shift)


def estimate_shift(pre_rgb, post_rgb, patch=128, stride=96, max_shift=16, agree_px=1.5):
    """Post-vs-pre misalignment for one tile, measured two ways.

    Patch estimate (the main one): split the tile into overlapping ``patch``-sized squares,
    skip ones with no-data or no texture, estimate each patch's shift within +-max_shift,
    and take the median. ``patch_agreement`` is the share of patches within ``agree_px`` of
    that median: high = one consistent shift, low = the shift varies across the tile or the
    patch estimates are noisy.

    Whole-image estimate (a cross-check): one phase correlation over the full tile with
    no-data filled, unbounded, with its PSR.
    """
    pre, post = to_gray(pre_rgb), to_gray(post_rgb)
    nodata = float((pre <= NODATA_MAX).mean())
    shifts = []
    h, w = pre.shape
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            pa, pb = pre[y:y + patch, x:x + patch], post[y:y + patch, x:x + patch]
            if (pa <= NODATA_MAX).mean() > 0.05 or (pb <= NODATA_MAX).mean() > 0.05 or pa.std() < 5:
                continue
            shifts.append(patch_shift(pa, pb, max_shift))
    out = {"n_patches": len(shifts), "nodata_frac": nodata}
    if shifts:
        s = np.array(shifts, dtype=float)
        med = np.median(s, axis=0)
        out.update(dy=float(med[0]), dx=float(med[1]), magnitude=float(np.hypot(*med)),
                   patch_agreement=float((np.hypot(*(s - med).T) <= agree_px).mean()))
    else:
        out.update(dy=np.nan, dx=np.nan, magnitude=np.nan, patch_agreement=np.nan)
    gdy, gdx, psr = phase_correlation(fill_nodata(pre), fill_nodata(post))
    out.update(global_dy=gdy, global_dx=gdx, global_magnitude=float(np.hypot(gdy, gdx)), global_psr=psr)
    return out
