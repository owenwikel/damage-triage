"""Cut a square image crop around one building.

The model does not look at whole 1024x1024 tiles. It looks at one building at a time: a
small square cut out of the pre-disaster image and the same square cut out of the
post-disaster image. This file defines that square.

    side of the square = longest side of the building's bounding box + 2 * PAD pixels

Why a square around the *bounding box* (the smallest upright rectangle holding the
outline) rather than the outline itself?
  - A network needs a fixed-size, rectangular input.
  - The padding gives context: debris around a house, water on the street, scorched ground.
    Damage is often easier to see around a building than on its roof.
  - Pre and post images are not perfectly aligned (docs/data_quality.md, section 6). With
    16 px of padding, a post-disaster building shifted by a few pixels is still inside
    the crop.

Every crop is then resized to OUT x OUT pixels, so a shed and a warehouse arrive at the
network at the same size. That throws away the building's true size, which is why the
index file keeps ``box_side``: we can still report accuracy by size later.
"""

import numpy as np
from PIL import Image

PAD = 16          # context pixels added on every side of the bounding box
OUT = 128         # crops are resized to OUT x OUT
TILE_SIZE = 1024
EDGE_TOL = 1.0    # same "touches the tile border" rule as docs/data_quality.md section 4


def crop_window(bounds, pad=PAD):
    """Square window (left, top, right, bottom) centred on a bounding box, in whole pixels.

    ``bounds`` is shapely's ``polygon.bounds``: (min x, min y, max x, max y).
    The window may stick out of the image; ``cut_crop`` fills that part with black.
    """
    x0, y0, x1, y1 = bounds
    side = max(x1 - x0, y1 - y0) + 2 * pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = max(int(round(side)), 1)
    return left, top, left + side, top + side


def cut_crop(image, window, out=OUT):
    """Cut ``window`` out of a PIL image and resize it to ``out`` x ``out``; uint8 array.

    PIL's ``crop`` pads with black where the window leaves the image, which is what we
    want for buildings at the tile border: the network sees "nothing here" rather than a
    stretched or mirrored picture. BILINEAR in PIL averages over the right number of
    source pixels when shrinking, so big buildings are not aliased.
    """
    crop = image.crop(window).resize((out, out), Image.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def at_edge(bounds, tile_size=TILE_SIZE, tol=EDGE_TOL):
    """True if the bounding box touches the tile border (the footprint is probably cut)."""
    x0, y0, x1, y1 = bounds
    return bool(x0 <= tol or y0 <= tol or x1 >= tile_size - tol or y1 >= tile_size - tol)
