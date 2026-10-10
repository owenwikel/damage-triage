"""Does a model's answer depend on the pre image? -> outputs/pre_image_use/, docs/pre_image_plan.md

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\pre_image_use.py --settings stack_aug
    .\\.venv\\Scripts\\python.exe scripts\\pre_image_use.py --settings stack_aug stack_aug_samepair

The test (defined in docs/pre_image_plan.md section 5, before any candidate was trained)
----------------------------------------------------------------------------------------
Every building keeps its post image and its label, but gets the PRE image of a
different building of the same disaster. A model that ignores the pre image answers as
before; a model that compares before with after is handed a wrong "before" and should
do worse.

    drop = macro-F1 with the real pairs  -  macro-F1 with the swapped pre images

The larger the drop, the more the model relies on the pre image. The base model
(stack_aug) is the reference: a change that is meant to make the model use the pre
image should give a larger drop than the base's.

Which "different building" is fixed once and is the same for every model: inside each
disaster the buildings are put in a random order (seed 0) and each one receives the pre
image of the next one in that order. Nobody keeps their own image, and two models are
always compared on exactly the same swapped pairs.

Parts: val (the test proper) and val-tiles (context). The test part cannot be used:
"test" is not an allowed --parts value and CropDataset refuses it (src/split.py).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_predictions import load_model  # noqa: E402  (rebuilds a model from best.pt)
from src.dataset import CropDataset  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.reporting import df_to_markdown, write_section  # noqa: E402
from src.split import require_final  # noqa: E402
from src.train import predict  # noqa: E402

PARTS = ("val", "val-tiles")
SEEDS = (0, 1, 2)


def swap_sources(disasters, seed=0):
    """For every building, the row of the building whose pre image it receives.

    disasters   one disaster name per building, in dataset order
    Returns an integer array ``source`` with source[i] != i, where building source[i]
    is from the same disaster as building i.
    """
    disasters = np.asarray(disasters)
    rng = np.random.default_rng(seed)
    source = np.arange(len(disasters))
    for d in np.unique(disasters):               # np.unique sorts: a fixed order of disasters
        rows = np.flatnonzero(disasters == d)
        if len(rows) < 2:                        # a lone building has nobody to swap with
            continue
        order = rng.permutation(rows)
        # np.roll(order, -1) is the same list moved one place: element k becomes the
        # old element k + 1, and the last one wraps round to the first. So each
        # building in ``order`` is paired with its successor.
        source[order] = np.roll(order, -1)
    return source


class SwappedPre(Dataset):
    """A CropDataset whose pre images come from other buildings.

    Item i is (pre image of building source[i], post image of building i, label of i).
    The wrapped dataset does the loading and normalising, so the images are prepared
    exactly as for an ordinary evaluation.
    """

    def __init__(self, dataset, source):
        self.dataset, self.source = dataset, source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        other_pre, _, _ = self.dataset[int(self.source[i])]
        _, post, label = self.dataset[i]
        return other_pre, post, label


def test_run(run_dir, parts=PARTS, batch_size=512, device=None, cache_dir=None):
    """Real-pair and swapped-pre metrics of one finished run. Returns a list of dicts."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)
    if cache_dir is not None:
        cfg = {**cfg, "cache_dir": str(cache_dir)}
    rows = []
    for part in require_final(parts):            # raises for "test"
        dataset = CropDataset.from_config(cfg, part, augment=False)
        source = swap_sources(dataset.index["disaster"].to_numpy())
        result = {}
        for name, data in (("real", dataset), ("swapped", SwappedPre(dataset, source))):
            loader = DataLoader(data, batch_size=batch_size, pin_memory=device.type == "cuda")
            logits, labels = predict(model, loader, device, amp=cfg.get("amp", True))
            result[name] = (logits.argmax(axis=1), labels)
        (pred, y), (pred_swapped, _) = result["real"], result["swapped"]
        real, swapped = summarize(y, pred), summarize(y, pred_swapped)
        rows.append({
            "run": Path(run_dir).name, "part": part, "buildings": len(y),
            "macro_f1_real": real["macro_f1"], "macro_f1_swapped": swapped["macro_f1"],
            "drop_macro_f1": real["macro_f1"] - swapped["macro_f1"],
            "accuracy_real": real["accuracy"], "accuracy_swapped": swapped["accuracy"],
            # how many answers change at all, right or wrong
            "changed_share": float((pred != pred_swapped).mean()),
            "predicted_no_damage_real": float((pred == 0).mean()),
            "predicted_no_damage_swapped": float((pred_swapped == 0).mean()),
        })
        r = rows[-1]
        print(f"  {r['run']} {part}: macro-F1 {r['macro_f1_real']:.3f} -> "
              f"{r['macro_f1_swapped']:.3f} (drop {r['drop_macro_f1']:+.3f}), "
              f"{r['changed_share']:.1%} of answers change", flush=True)
    return rows


def summary_table(rows):
    """One row per setting and part: mean over the seeds, and the range of the drop."""
    df = pd.DataFrame(rows)
    out = []
    for (setting, part), g in df.groupby(["setting", "part"], sort=False):
        out.append({
            "setting": setting, "part": part, "seeds": len(g),
            "macro-F1, real pairs": g["macro_f1_real"].mean(),
            "macro-F1, swapped pre": g["macro_f1_swapped"].mean(),
            "drop": g["drop_macro_f1"].mean(),
            "drop, range over seeds": f"{g['drop_macro_f1'].min():+.3f} to {g['drop_macro_f1'].max():+.3f}",
            "accuracy, real pairs": g["accuracy_real"].mean(),
            "accuracy, swapped pre": g["accuracy_swapped"].mean(),
            "answers that change": g["changed_share"].mean(),
        })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=["stack_aug"],
                    help="run-name prefixes; runs/<setting>_s<seed> for every seed")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pre_image_use"))
    ap.add_argument("--doc", type=Path, default=Path("docs/pre_image_plan.md"))
    ap.add_argument("--section", default="pre-image-use",
                    help="name of the generated section in --doc")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for setting in args.settings:
        cached = args.out / f"{setting}.json"
        if cached.exists():
            # Each setting is tested once and its numbers are kept, so that adding a
            # candidate later does not re-run (or silently change) the reference.
            print(f"{setting}: read from {cached}")
            rows += json.loads(cached.read_text(encoding="utf-8"))
            continue
        setting_rows = []
        for seed in args.seeds:
            run_dir = args.runs_dir / f"{setting}_s{seed}"
            if not (run_dir / "metrics.json").exists():
                print(f"{run_dir}: unfinished, skipped")
                continue
            setting_rows += [{**r, "setting": setting, "seed": seed}
                             for r in test_run(run_dir, args.parts)]
        if setting_rows:
            cached.write_text(json.dumps(setting_rows, indent=2), encoding="utf-8")
        rows += setting_rows

    table = summary_table(rows)
    per_seed = pd.DataFrame(rows)[["setting", "seed", "part", "macro_f1_real", "macro_f1_swapped",
                                   "drop_macro_f1", "changed_share"]]
    print(table.round(4).to_string(index=False))
    md = "\n\n".join([
        "## Pre-image-use test: results",
        "Generated by `scripts/pre_image_use.py`. Each building's pre image is replaced by "
        "the pre image of another building of the same disaster (section 5). drop = "
        "macro-F1 with the real pairs minus macro-F1 with the swapped pre images; a "
        "larger drop means the model relies more on the pre image. Means over the seeds.",
        df_to_markdown(table, "{:.3f}", index=False),
        "Per seed:",
        df_to_markdown(per_seed, "{:.3f}", index=False),
    ])
    write_section(args.doc, args.section, md)
    print(f"Wrote section '{args.section}' of {args.doc}")


if __name__ == "__main__":
    main()
