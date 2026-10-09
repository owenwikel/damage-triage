"""Save every building's logits for the base model's seeds -> outputs/predictions/

Usage (from the repo root):
    .\\.venv\\Scripts\\python.exe scripts\\export_predictions.py
    .\\.venv\\Scripts\\python.exe scripts\\export_predictions.py --runs runs\\stack_aug_s0 --parts val

What this is for
----------------
Training (src/train.py) already saves logits for val-tiles and val, but only as two bare
arrays (logits, labels): to know WHICH building a row belongs to you have to line it up
with the crop cache's index.csv yourself, and nothing is saved for train. This script
writes one self-describing table per run and part, so the uncertainty work that comes
next (docs/uncertainty_plan.md) can start from files instead of from the GPU.

What is written
---------------
    outputs/predictions/<run_name>/<part>.csv      one row per building
        uid          xBD's id of the building (unique across the whole dataset)
        disaster     e.g. "hurricane-harvey"
        tile_id      the tile it sits on, e.g. "00000012". Read it back as TEXT:
                         pd.read_csv(path, dtype={"tile_id": str})
                     otherwise pandas turns it into the number 12.
        label        the true class: 0 no-damage, 1 minor-damage, 2 major-damage,
                     3 destroyed
        logit_0..3   the model's four raw scores, one per class, in that same order
    outputs/predictions/manifest.json              which runs and parts, with row counts

A "logit" is the number the network outputs for a class BEFORE the softmax turns the
four of them into probabilities. Logits are saved, not probabilities, because you can
always get probabilities from logits (softmax) but not the other way round, and
temperature scaling (src/calibrate.py) needs the logits.

Every run's file for a part has the same rows in the same order, so the seeds can be
averaged into an ensemble row by row (and ``uid`` is there to check that).

What is NOT done here
---------------------
  * The test part is never exported. There is no --final flag in this script on purpose:
    "test" is not an allowed value for --parts, and CropDataset would refuse it anyway
    (src/split.py).
  * Un-classified buildings do not appear: they were dropped when the crop cache was
    built (scripts/build_crops.py, docs/data_decisions.md). ``export_run`` checks that.
  * The train rows are predictions on data the model LEARNED FROM. They look far better
    than the model really is and must not be used to measure accuracy or to fit a
    temperature. They are exported for diagnostics (for example: which training
    buildings does the model still get wrong?).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Scripts live in scripts/, the library code in src/. Python only finds ``src`` if the
# repo root is on its search path, so we add it by hand.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CLASS_NAMES, CropDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.split import DEV_PARTS, require_final  # noqa: E402
from src.train import predict  # noqa: E402

# The base model: stack ResNet-18, strong augmentation, kept under the 3-epoch selection
# rule. Three seeds of it exist; together they are the ensemble (docs/results.md).
DEFAULT_RUNS = ["runs/stack_aug_s0", "runs/stack_aug_s1", "runs/stack_aug_s2"]
LOGIT_COLUMNS = [f"logit_{i}" for i in range(len(CLASS_NAMES))]
COLUMNS = ["uid", "disaster", "tile_id", "label"] + LOGIT_COLUMNS


def load_model(run_dir, device):
    """Rebuild a trained model from runs/<run>/best.pt. Returns (model, config dict).

    A checkpoint does not contain "the model". It contains a ``state_dict``: a
    dictionary from layer names to tensors of numbers (the weights). To use it you
    first build the same architecture again, empty, and then pour the numbers in.
    That is why the config is saved next to the weights: it says which architecture.
    """
    # map_location: the weights were saved from the GPU; this loads them onto whatever
    # device we have now (so the script also works on a machine without a GPU).
    # weights_only=True: only read tensors and plain values, never run code hidden in
    # the file. A good habit for any checkpoint.
    state = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    cfg = state["config"]
    # pretrained=False: no need to download ImageNet weights, every weight is about to
    # be overwritten by ours.
    model = build_model(cfg["arch"], pretrained=False, dropout=cfg["dropout"]).to(device)
    # load_state_dict is strict by default: a missing or unexpected layer name is an
    # error, so the wrong architecture cannot be loaded silently.
    model.load_state_dict(state["model"])
    return model, cfg


def export_run(run_dir, out_dir, parts=DEV_PARTS, batch_size=512, num_workers=0,
               cache_dir=None, device=None):
    """Write <out_dir>/<run name>/<part>.csv for one finished run.

    Returns {part: number of rows written}.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    # Raises for "test": final=True is never passed anywhere in this file.
    parts = require_final(parts)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)
    if cache_dir is not None:      # the crop cache has moved since the run was trained
        cfg = {**cfg, "cache_dir": str(cache_dir)}

    counts = {}
    for part in parts:
        # from_config prepares the images exactly as this run saw them in training
        # (normalisation, histogram matching). augment=False even for train: we want
        # the model's answer for each building as it is, not for a randomly flipped
        # and recoloured copy, and we want the same answer every time we run this.
        dataset = CropDataset.from_config(cfg, part, augment=False)
        index = dataset.index      # one row per building, in the order of the crops
        assert not (index["damage"] == "un-classified").any(), "un-classified in the cache"
        assert index["label"].between(0, len(CLASS_NAMES) - 1).all(), "label outside 0..3"

        # shuffle is off (the default), so batch after batch the loader walks through
        # buildings 0, 1, 2, ... and row i of the logits is row i of the index. That
        # alignment is the whole point of this script; it is checked below.
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        # predict() (src/train.py) puts the model in eval mode (dropout off, BatchNorm
        # on its stored statistics), runs without gradients, and returns numpy arrays:
        # logits [N, 4] and the labels the loader handed out [N].
        logits, labels = predict(model, loader, device, amp=cfg.get("amp", True))

        # Three cheap checks that the table we are about to write is the right one.
        assert len(logits) == len(index), f"{part}: {len(logits)} logits, {len(index)} buildings"
        assert (labels == index["label"].to_numpy()).all(), f"{part}: labels out of step"
        _compare_with_training(run_dir, part, logits)

        table = index[["uid", "disaster", "tile_id", "label"]].copy()
        table[LOGIT_COLUMNS] = logits
        (out_dir / run_dir.name).mkdir(parents=True, exist_ok=True)
        path = out_dir / run_dir.name / f"{part}.csv"
        # Six decimals keep everything a 32-bit logit holds that matters, at about a
        # third of the file size of full precision.
        table[COLUMNS].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        counts[part] = len(table)
        accuracy = (logits.argmax(axis=1) == labels).mean()
        print(f"  {part}: {len(table):,} buildings, accuracy {accuracy:.3f} -> {path}", flush=True)
    return counts


def _compare_with_training(run_dir, part, logits):
    """If training saved logits for this part, ours must be (nearly) the same.

    Not bit-for-bit: mixed precision and a different batch size change the last digits.
    But a real difference would mean the wrong weights or the wrong preprocessing.
    """
    saved = run_dir / f"preds_{part}.npz"
    if not saved.exists():         # train: training never saved these
        return
    old = np.load(saved)["logits"]
    assert old.shape == logits.shape, f"{part}: shape differs from {saved}"
    same_answer = (old.argmax(axis=1) == logits.argmax(axis=1)).mean()
    print(f"  {part}: largest difference to {saved.name} {np.abs(old - logits).max():.4f}, "
          f"same predicted class for {same_answer:.2%}", flush=True)
    assert same_answer > 0.995, f"{part}: predictions differ from those saved in training"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=[Path(r) for r in DEFAULT_RUNS],
                    help="run folders under runs/; default: the three seeds of the base model")
    # choices: argparse itself rejects "test" before any code runs.
    ap.add_argument("--parts", nargs="+", default=list(DEV_PARTS), choices=DEV_PARTS)
    ap.add_argument("--out", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 = load in the main process: slower, but no extra memory")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="crop cache, if it is no longer where the run's config says")
    args = ap.parse_args(argv)

    manifest = {"parts": list(args.parts), "class_names": CLASS_NAMES, "columns": COLUMNS,
                "runs": {}}
    for run_dir in args.runs:
        # metrics.json is the last file a training run writes. Without it the run was
        # interrupted: its best.pt is whatever was best so far, not a selected model.
        if not (run_dir / "metrics.json").exists():
            print(f"{run_dir}: no metrics.json (unfinished run), skipped", flush=True)
            continue
        print(f"{run_dir}:", flush=True)
        counts = export_run(run_dir, args.out, args.parts, args.batch_size, args.num_workers,
                            args.cache_dir)
        summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest["runs"][run_dir.name] = {"seed": summary["seed"], "arch": summary["arch"],
                                          "best_epoch": summary["best_epoch"], "rows": counts}
    if not manifest["runs"]:
        raise SystemExit("No finished run among --runs: nothing exported.")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":   # required on Windows if --num-workers > 0 (see src/train.py)
    main()
