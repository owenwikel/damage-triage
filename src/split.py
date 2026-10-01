"""Read the split (splits/split_v1.csv) and guard the test set.

The project has four parts (see docs/split_final.md and docs/split_decision.md):

    train       what the model learns from
    val-tiles   held-out tiles of the *training* disasters: early stopping, calibration
    val         two disasters the model never trains on: honest "new disaster" check
    test        two more unseen disasters: the final exam, looked at ONCE, at the very end

Why a guard? Every time you look at a test score and then change something (a learning
rate, an architecture, a threshold), a little information about the test set leaks into
your decisions. Do it often enough and the test score stops being an honest estimate of
how the model does on new data. So the rule is: nothing loads the test part unless the
caller says, explicitly, "this is the final evaluation" (``final=True`` in code,
``--final`` on the command line). Every loader in this project goes through
``require_final`` below, so forgetting is an error rather than a silent leak.

A naming trap: xBD ships its files in folders called ``train`` and ``test``. Those folder
names mean nothing here: our parts are defined by *disaster*, and tiles of every part are
spread over both folders. In this file "folder" always means the xBD folder and "part"
always means our split.
"""

from pathlib import Path

import pandas as pd

from src.labels import COLUMNS, DAMAGE_CLASSES, parse_label_file

DEV_PARTS = ("train", "val-tiles", "val")   # safe to use during development
TEST_PART = "test"
ALL_PARTS = DEV_PARTS + (TEST_PART,)
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "splits" / "split_v1.csv"


class TestSplitError(RuntimeError):
    """Raised when code tries to touch the test part without ``final=True``."""

    __test__ = False  # tell pytest this is not a test class, despite the name


def require_final(parts, final=False):
    """Raise unless every requested part is a development part, or ``final`` is set.

    ``parts`` may be one name or a list of names. Returns the parts as a list.
    """
    parts = [parts] if isinstance(parts, str) else list(parts)
    unknown = sorted(set(parts) - set(ALL_PARTS))
    if unknown:
        raise ValueError(f"Unknown split part(s) {unknown}; choose from {ALL_PARTS}")
    if TEST_PART in parts and not final:
        raise TestSplitError(
            "The test part (hurricane-michael, santa-rosa-wildfire) is locked. It may only "
            "be loaded for the final evaluation: pass final=True (or --final on the command "
            "line). During development use train, val-tiles and val.")
    return parts


def load_split(parts=DEV_PARTS, split_csv=DEFAULT_SPLIT, final=False):
    """Return the tiles (disaster, tile_id, part) of the requested parts."""
    parts = require_final(parts, final)
    # dtype=str keeps tile ids like "00000012" as text; as numbers they would lose the zeros
    split = pd.read_csv(split_csv, dtype=str)
    return split[split["part"].isin(parts)].reset_index(drop=True)


def find_folder(data_dir, disaster, tile_id, folders=("train", "test")):
    """Return the xBD folder (``data_dir/train`` or ``data_dir/test``) holding this tile."""
    for folder in folders:
        if (Path(data_dir) / folder / "labels" / f"{disaster}_{tile_id}_post_disaster.json").exists():
            return Path(data_dir) / folder
    raise FileNotFoundError(f"No label file for {disaster}_{tile_id} under {data_dir}")


def load_part_labels(data_dir, parts=DEV_PARTS, split_csv=DEFAULT_SPLIT, final=False,
                     drop_unclassified=True, repair=True):
    """One row per building for the requested parts, with a ``part`` column.

    Unlike ``src.labels.load_labels`` (which reads a whole xBD folder), this opens only
    the label files of tiles that belong to the requested parts. Test tiles are therefore
    never even read from disk unless ``final=True``.

    The defaults apply the decisions in docs/data_decisions.md: un-classified buildings
    are dropped and invalid polygons are repaired.
    """
    tiles = load_split(parts, split_csv, final)
    rows = []
    for t in tiles.itertuples():
        folder = find_folder(data_dir, t.disaster, t.tile_id)
        label_file = folder / "labels" / f"{t.disaster}_{t.tile_id}_post_disaster.json"
        for row in parse_label_file(label_file, folder / "images", repair=repair):
            row["part"] = t.part
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS + ["part"])
    df["damage"] = pd.Categorical(df["damage"], categories=DAMAGE_CLASSES)
    df["damage_level"] = df["damage_level"].astype("Int64")
    if drop_unclassified:
        df = df[df["damage"] != "un-classified"].reset_index(drop=True)
    return df
