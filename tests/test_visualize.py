import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import visualize  # noqa: E402
from src.colors import DAMAGE_COLORS  # noqa: E402
from src.labels import DAMAGE_CLASSES  # noqa: E402

SQUARE = shapely_wkt.loads("POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))")
RECT = shapely_wkt.loads("POLYGON ((12 12, 20 12, 20 20, 12 20, 12 12))")


@pytest.fixture
def tile(tmp_path):
    """A two-building tile whose pre and post images are different solid colours."""
    pre = tmp_path / "x_00000001_pre_disaster.png"
    post = tmp_path / "x_00000001_post_disaster.png"
    Image.fromarray(np.full((32, 32, 3), 10, np.uint8)).save(pre)
    Image.fromarray(np.full((32, 32, 3), 200, np.uint8)).save(post)
    rows = [{"disaster": "x", "disaster_type": "fire", "tile_id": "00000001", "uid": uid,
             "pre_image": str(pre), "post_image": str(post), "polygon": poly, "damage": dmg}
            for uid, poly, dmg in (("a", SQUARE, "destroyed"), ("b", RECT, "no-damage"))]
    return pd.DataFrame(rows), pre, post


def _edge_rgbs(ax):
    """Every distinct edge colour (as an RGB tuple) used by the patch collections on ax."""
    out = set()
    for coll in ax.collections:
        for rgba in coll.get_edgecolor():
            out.add(tuple(np.round(rgba[:3], 3)))
    return out


def _rgb(color):
    return tuple(np.round(mcolors.to_rgb(color), 3))


# --- panels load the right images ------------------------------------------------------

def test_panels_load_different_files(tile, monkeypatch):
    df, pre, post = tile
    opened = []
    real_open = visualize.Image.open

    def recording_open(path, *args, **kwargs):
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(visualize.Image, "open", recording_open)
    plt.close(visualize.make_figure(df))

    assert opened == [pre, post], "left panel must load the pre image, right panel the post image"
    assert opened[0] != opened[1]


def test_panels_show_different_pixels(tile):
    df, _, _ = tile
    fig = visualize.make_figure(df)
    left, right = (np.asarray(ax.images[0].get_array()) for ax in fig.axes[:2])
    plt.close(fig)
    assert np.abs(left.astype(int) - right.astype(int)).mean() > 0


# --- outline colours and legend --------------------------------------------------------

def test_before_panel_is_plain_white(tile):
    df, _, _ = tile
    fig = visualize.make_figure(df)
    ax_pre = fig.axes[0]
    assert _edge_rgbs(ax_pre) == {_rgb("white")}
    assert ax_pre.get_legend() is None
    plt.close(fig)


def test_after_panel_uses_damage_colours_and_has_legend(tile):
    df, _, _ = tile
    fig = visualize.make_figure(df)
    ax_post = fig.axes[1]
    edges = _edge_rgbs(ax_post)
    assert _rgb(DAMAGE_COLORS["destroyed"]) in edges
    assert _rgb(DAMAGE_COLORS["no-damage"]) in edges
    legend = ax_post.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels[0] == "no-damage (1)" and "destroyed (1)" in labels
    # each legend swatch colour matches its class
    for handle, cls in zip(legend.legend_handles, DAMAGE_CLASSES):
        assert _rgb(handle.get_facecolor()) == _rgb(DAMAGE_COLORS[cls])
    plt.close(fig)


def test_legend_fits_inside_figure(tile):
    # Long labels like "minor-damage (1062)" once pushed a legend row off the bottom edge.
    df, _, _ = tile
    df = df.assign(damage=pd.Categorical(["minor-damage", "un-classified"], DAMAGE_CLASSES))
    fig = visualize.make_figure(df)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.axes[1].get_legend().get_window_extent(renderer)
    fig_box = fig.bbox
    plt.close(fig)
    assert box.x0 >= fig_box.x0 and box.y0 >= fig_box.y0
    assert box.x1 <= fig_box.x1 and box.y1 <= fig_box.y1


def test_title_has_name_disaster_and_counts(tile):
    df, _, _ = tile
    title = visualize.figure_title(df, "destroyed")
    assert "x_00000001" in title
    assert "(x, fire)" in title
    assert "2 buildings" in title and "destroyed 1" in title and "no 1" in title


def test_polygon_colour_follows_damage(tile):
    df, _, _ = tile
    _, colors = visualize.polygon_patches(df)
    assert colors == [DAMAGE_COLORS["destroyed"], DAMAGE_COLORS["no-damage"]]


# --- tile selection --------------------------------------------------------------------

def _building_rows(disaster, tile_id, **counts):
    return [{"disaster": disaster, "tile_id": tile_id, "damage": cls.replace("_", "-")}
            for cls, n in counts.items() for _ in range(n)]


@pytest.fixture
def many_tiles():
    rows = []
    rows += _building_rows("fire-a", "1", destroyed=9)
    rows += _building_rows("fire-a", "2", destroyed=8)            # same disaster as tile 1
    rows += _building_rows("storm-b", "1", destroyed=5, major_damage=7, minor_damage=2)
    rows += _building_rows("storm-c", "1", major_damage=6, minor_damage=9)
    rows += _building_rows("quake-d", "1", no_damage=50)
    df = pd.DataFrame(rows)
    df["damage"] = pd.Categorical(df["damage"], categories=DAMAGE_CLASSES)
    return df


def test_select_tiles_ranks_by_class_and_keeps_disasters_distinct(many_tiles):
    picked = visualize.select_tiles(many_tiles, (("destroyed", 2), ("major-damage", 2)))
    assert picked == [
        ("destroyed", "fire-a", "1", 9),       # fire-a/2 (8) skipped: same disaster
        ("destroyed", "storm-b", "1", 5),
        ("major-damage", "storm-c", "1", 6),   # storm-b/1 (7) skipped: already picked
    ]


def test_select_tiles_never_picks_zero_count(many_tiles):
    picked = visualize.select_tiles(many_tiles, (("minor-damage", 5),))
    assert [(d, n) for _, d, _, n in picked] == [("storm-c", 9), ("storm-b", 2)]


# --- locked output files ---------------------------------------------------------------

def test_save_figure_writes_suffixed_file_when_locked(tmp_path, monkeypatch):
    fig = plt.figure()
    wanted = tmp_path / "tile.png"
    real_savefig = fig.savefig
    locked = {wanted, tmp_path / "tile_new.png"}

    def savefig(path, *args, **kwargs):
        if Path(path) in locked:
            raise PermissionError(13, "file is open in another program")
        return real_savefig(path, *args, **kwargs)

    monkeypatch.setattr(fig, "savefig", savefig)
    written = visualize.save_figure(fig, wanted)
    plt.close(fig)
    assert written == tmp_path / "tile_new2.png"
    assert written.exists() and not wanted.exists()


def test_save_figure_uses_original_name_when_free(tmp_path):
    fig = plt.figure()
    written = visualize.save_figure(fig, tmp_path / "tile.png")
    plt.close(fig)
    assert written == tmp_path / "tile.png" and written.exists()
