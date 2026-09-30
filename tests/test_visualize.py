import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import visualize  # noqa: E402
from src.colors import DAMAGE_COLORS  # noqa: E402


@pytest.fixture
def tile(tmp_path):
    """A one-building tile whose pre and post images are different solid colours."""
    pre = tmp_path / "x_00000001_pre_disaster.png"
    post = tmp_path / "x_00000001_post_disaster.png"
    Image.fromarray(np.full((32, 32, 3), 10, np.uint8)).save(pre)
    Image.fromarray(np.full((32, 32, 3), 200, np.uint8)).save(post)
    df = pd.DataFrame([{
        "disaster": "x", "disaster_type": "fire", "tile_id": "00000001", "uid": "a",
        "pre_image": str(pre), "post_image": str(post),
        "polygon": shapely_wkt.loads("POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))"),
        "damage": "destroyed",
    }])
    return df, pre, post


def test_panels_load_different_files(tile, tmp_path, monkeypatch):
    df, pre, post = tile
    opened = []
    real_open = visualize.Image.open

    def recording_open(path, *args, **kwargs):
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(visualize.Image, "open", recording_open)
    visualize.draw_tile(df, tmp_path / "out.png")

    assert opened == [pre, post], "left panel must load the pre image, right panel the post image"
    assert opened[0] != opened[1]


def test_panels_show_different_pixels(tile, tmp_path, monkeypatch):
    df, _, _ = tile
    shown = []
    real_imshow = visualize.plt.Axes.imshow

    def recording_imshow(self, img, *args, **kwargs):
        shown.append(np.asarray(img).copy())
        return real_imshow(self, img, *args, **kwargs)

    monkeypatch.setattr(visualize.plt.Axes, "imshow", recording_imshow)
    visualize.draw_tile(df, tmp_path / "out.png")

    assert len(shown) == 2
    assert np.abs(shown[0].astype(int) - shown[1].astype(int)).mean() > 0


def test_polygon_colour_follows_damage(tile):
    df, _, _ = tile
    _, colors = visualize.polygon_patches(df)
    assert colors == [DAMAGE_COLORS["destroyed"]]
