"""THE FINAL EVALUATION, STEP 1: save the three base models' logits on the TEST part.

    Run this ONCE. Read docs/test_protocol.md first.

Usage (after the test crops have been built into their own folder):
    .\\.venv\\Scripts\\python.exe scripts\\build_crops.py --data-dir C:\\Users\\wikel\\dev\\xbd-data --final --parts test --out C:\\Users\\wikel\\dev\\xbd-data\\crops_test_final
    .\\.venv\\Scripts\\python.exe scripts\\export_test_predictions.py --final --cache-dir C:\\Users\\wikel\\dev\\xbd-data\\crops_test_final

Why this is a separate script
-----------------------------
scripts/export_predictions.py cannot export the test part, on purpose: during
development nothing should be able to. This script is the one door, and it has three
locks on it:

  1. ``--final`` must be given. Without it nothing is loaded (the same rule as
     everywhere else in the project, src/split.py).
  2. docs/test_protocol.md must exist and be committed, unchanged: the rules of the
     evaluation are fixed before the data is looked at.
  3. TEST_EVALUATION.lock must not exist. The script writes that file BEFORE it loads
     anything and refuses to start while it is there. So a second run, by accident or
     out of curiosity after a change to the model, does not happen: someone would have
     to delete the lock by hand, and the protocol says when that is allowed (only after
     a crash before any result was seen).

What is written
---------------
    outputs/predictions_test/<run>/test.csv    uid, disaster, tile_id, label, logit_0..3
    TEST_EVALUATION.lock                        when, which commit, which runs, row counts
                                                and a checksum of every file written

The test predictions go to their own folder, away from outputs/predictions/, so that
the development scripts (calibration, risk-coverage on val) never see them.

The models are loaded exactly as scripts/export_predictions.py loads them, and the
images are prepared exactly as for val: same normalisation, no augmentation.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_predictions import COLUMNS, DEFAULT_RUNS, LOGIT_COLUMNS, load_model  # noqa: E402
from src.dataset import CLASS_NAMES, CropDataset  # noqa: E402
from src.split import TEST_PART, TestSplitError  # noqa: E402
from src.train import predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO / "TEST_EVALUATION.lock"
DEFAULT_PROTOCOL = REPO / "docs" / "test_protocol.md"


def protocol_is_committed(protocol):
    """True if the protocol file is tracked by git and identical to the last commit."""
    protocol = Path(protocol)
    if not protocol.exists():
        return False
    # "git status --porcelain <file>" prints nothing for a tracked, unmodified file, and
    # a line starting with "??" or " M" otherwise. "git ls-files" tells tracked from absent.
    def git(*args):
        return subprocess.run(["git", *args], cwd=protocol.parent, capture_output=True, text=True)
    tracked = git("ls-files", "--error-unmatch", protocol.name)
    status = git("status", "--porcelain", "--", protocol.name)
    return tracked.returncode == 0 and status.returncode == 0 and status.stdout.strip() == ""


def git_head():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def sha256(path):
    """Checksum of a file: a short fingerprint that changes if a single byte changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_run_test(run_dir, out_dir, cache_dir, batch_size=512, device=None):
    """Write <out_dir>/<run name>/test.csv for one run. Returns (row count, file path)."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)
    # The run's config names the DEVELOPMENT crop cache, which holds no test crops. The
    # test crops live in their own folder; everything else comes from the run's config.
    cfg = {**cfg, "cache_dir": str(cache_dir)}
    # final=True: the one place in the project where this is passed for a model.
    dataset = CropDataset.from_config(cfg, TEST_PART, augment=False, final=True)
    index = dataset.index
    assert not (index["damage"] == "un-classified").any(), "un-classified in the cache"
    assert index["label"].between(0, len(CLASS_NAMES) - 1).all(), "label outside 0..3"
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    logits, labels = predict(model, loader, device, amp=cfg.get("amp", True))
    assert len(logits) == len(index), f"{len(logits)} logits, {len(index)} buildings"
    assert (labels == index["label"].to_numpy()).all(), "labels out of step"
    table = index[["uid", "disaster", "tile_id", "label"]].copy()
    table[LOGIT_COLUMNS] = logits
    (out_dir / run_dir.name).mkdir(parents=True, exist_ok=True)
    path = out_dir / run_dir.name / f"{TEST_PART}.csv"
    table[COLUMNS].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    # Deliberately NO accuracy printed here: the numbers are read from the results
    # document, all at once, not glimpsed run by run.
    print(f"  {run_dir.name}: {len(table):,} buildings -> {path}", flush=True)
    return len(table), path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true",
                    help="required: confirms that this is the one final evaluation")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="the folder holding the TEST crops (built with build_crops.py --final)")
    ap.add_argument("--runs", nargs="+", type=Path, default=[REPO / r for r in DEFAULT_RUNS])
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "predictions_test")
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args(argv)

    # ---- the three locks, in order; nothing is loaded before all three pass ----
    if not args.final:
        raise TestSplitError("The test part is locked. This script is the final evaluation: "
                             "read docs/test_protocol.md, then pass --final.")
    if args.lock.exists():
        raise SystemExit(f"{args.lock} exists: the test predictions have already been exported "
                         "(or a run was started). The final evaluation runs once. See "
                         "docs/test_protocol.md, section 7, before doing anything else.")
    if not protocol_is_committed(args.protocol):
        raise SystemExit(f"{args.protocol} is missing or not committed unchanged. The protocol "
                         "has to be fixed in git before the test set is opened.")
    unfinished = [str(r) for r in args.runs if not (r / "metrics.json").exists()]
    if unfinished:
        raise SystemExit(f"Unfinished or missing runs: {unfinished}. Nothing was loaded.")

    # ---- write the lock FIRST: from here on the test set counts as opened ----
    lock = {"status": "started", "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_head(), "protocol": str(args.protocol.name),
            "runs": [r.name for r in args.runs], "cache_dir": str(args.cache_dir)}
    args.lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    files = {}
    for run_dir in args.runs:
        rows, path = export_run_test(run_dir, args.out, args.cache_dir, args.batch_size)
        files[run_dir.name] = {"rows": rows, "sha256": sha256(path)}
    lock.update({"status": "completed", "files": files,
                 "completed": datetime.datetime.now().isoformat(timespec="seconds")})
    args.lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Wrote {args.lock}. The test predictions are exported; this script will not run again.")


if __name__ == "__main__":
    main()
