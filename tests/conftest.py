"""Shared test fixtures. Nothing here reads the real dataset."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

TILE = 1024

# Three buildings: one intact, one destroyed, one un-classified touching the tile edge.
BUILDINGS = [
    ("uid-a", "no-damage", [(100, 100), (160, 100), (160, 150), (100, 150)]),
    ("uid-b", "destroyed", [(400, 380), (470, 380), (470, 460), (400, 460)]),
    ("uid-c", "un-classified", [(990, 700), (1024, 700), (1024, 740), (990, 740)]),
]
MASK_VALUE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}


def _texture(seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((TILE, TILE))
    for scale in (8, 32, 128):
        coarse = rng.normal(size=(TILE // scale + 1, TILE // scale + 1))
        img += np.kron(coarse, np.ones((scale, scale)))[:TILE, :TILE]
    img = (img - img.min()) / (img.max() - img.min())
    return np.repeat((30 + 200 * img)[..., None], 3, axis=2).astype(np.uint8)


def _wkt(pts):
    ring = pts + [pts[0]]
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"


def write_fake_split(root, disaster="fake-fire", tile_id="00000001", disaster_type="fire"):
    """Write one xBD-style tile (images, labels, targets) under ``root``; return ``root``."""
    for sub in ("images", "labels", "targets"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    base = _texture(seed=1)
    # post = pre shifted 3 px right, with the destroyed building darkened
    post = np.roll(base, 3, axis=1).copy()
    x0, y0, x1, y1 = 400, 380, 470, 460
    post[y0:y1, x0:x1] //= 3
    stem = f"{disaster}_{tile_id}"
    for phase, img in (("pre", base), ("post", post)):
        Image.fromarray(img).save(root / "images" / f"{stem}_{phase}_disaster.png")
        feats = []
        for uid, dmg, pts in BUILDINGS:
            props = {"feature_type": "building", "uid": uid}
            if phase == "post":
                props["subtype"] = dmg
            # lng_lat: a fake linear mapping so geographic code has something to read
            ll = [(-120 + x * 1e-5, 38 - y * 1e-5) for x, y in pts]
            feats.append({"properties": props, "wkt": _wkt(pts), "_ll": _wkt(ll)})
        doc = {
            "features": {
                "xy": [{"properties": f["properties"], "wkt": f["wkt"]} for f in feats],
                "lng_lat": [{"properties": f["properties"], "wkt": f["_ll"]} for f in feats],
            },
            "metadata": {"disaster": disaster, "disaster_type": disaster_type,
                         "capture_date": "2020-01-01T00:00:00.000Z", "sensor": "FAKESAT",
                         "gsd": 2.0, "off_nadir_angle": 10.0, "width": TILE, "height": TILE,
                         "img_name": f"{stem}_{phase}_disaster.png"},
        }
        (root / "labels" / f"{stem}_{phase}_disaster.json").write_text(json.dumps(doc))
        mask = Image.new("L", (TILE, TILE), 0)
        draw = ImageDraw.Draw(mask)
        for _, dmg, pts in BUILDINGS:
            value = 1 if phase == "pre" else MASK_VALUE.get(dmg, 0)
            if value:
                draw.polygon(pts, fill=value)
        mask.save(root / "targets" / f"{stem}_{phase}_disaster_target.png")
    return root


@pytest.fixture(scope="session")
def fake_xbd(tmp_path_factory):
    """A data folder with a ``train`` and a ``test`` split, one synthetic tile each."""
    data = tmp_path_factory.mktemp("xbd")
    write_fake_split(data / "train", "fake-fire", "00000001", "fire")
    write_fake_split(data / "test", "fake-flood", "00000002", "flooding")
    return data
