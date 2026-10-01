"""Train a damage classifier on the crop cache.

Usage (from the repo root):
    .\\.venv\\Scripts\\python.exe -m src.train --config configs\\siamese.json
    .\\.venv\\Scripts\\python.exe -m src.train --config configs\\siamese.json --seed 1 --run-name siamese_s1
    .\\.venv\\Scripts\\python.exe -m src.train --config configs\\siamese.json --overfit 256 --run-name sanity_overfit

Everything that defines a run lives in a ``Config``: defaults below, overridden by a JSON
file (--config), overridden by command-line flags. The final config is saved with the
results, so every number in docs/results.md can be traced back to its settings.

What a training run does, in order
    1. fix the random seeds
    2. build the datasets and loaders (train with augmentation; val-tiles and val without)
    3. build the model, the loss, the optimiser and the learning-rate schedule
    4. for each epoch: one pass over the training set, then evaluate on val-tiles and val
    5. keep the weights of the best epoch (by val-tiles macro-F1); stop early if there is
       no improvement for ``patience`` epochs
    6. save predictions of the best model on val-tiles and val, for the later analysis

Outputs in runs/<run_name>/ (not in git):
    config.json     the settings
    log.jsonl       one line of metrics per epoch, written as training goes
    best.pt         weights of the best epoch
    metrics.json    summary: best epoch, metrics there, time per epoch, GPU memory
    preds_<part>.npz   logits and labels of the best model on val-tiles and val

The test part is never touched here: ``CropDataset`` refuses to load it (src/split.py).
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import CLASS_NAMES, CropDataset
from src.metrics import summarize
from src.model import build_model, count_parameters


@dataclass
class Config:
    arch: str = "siamese"            # "siamese" or "stack" (src/model.py)
    run_name: str = ""               # folder under runs/; default "<arch>_s<seed>"
    seed: int = 0
    cache_dir: str = r"C:\Users\wikel\dev\xbd-data\crops"
    runs_dir: str = "runs"
    epochs: int = 10
    patience: int = 3                # early stopping: epochs without improvement allowed
    batch_size: int = 0              # 0 = probe the GPU for the largest batch that fits
    max_batch_size: int = 512        # ...but never above this (see find_batch_size)
    eval_batch_size: int = 512       # evaluation stores no gradients, so memory is not the limit
    lr: float = 5e-4                 # peak learning rate
    weight_decay: float = 0.01
    warmup_frac: float = 0.05        # share of all steps spent ramping the lr up from 0
    dropout: float = 0.3
    pretrained: bool = True
    amp: bool = True                 # mixed precision (see train_one_epoch)
    num_workers: int = 6             # background processes that prepare training batches
    eval_num_workers: int = 0        # 0 = evaluation batches are prepared in the main process
    overfit: int = 0                 # N > 0: sanity check, train on N crops only


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

def seed_everything(seed):
    """Seed every random number generator that training touches.

    Randomness enters in several places: the initial weights of the new layers, the order
    in which examples are shuffled, the augmentation, and dropout. Fixing the seed makes
    a run repeatable enough to debug and to compare ("was that change real, or luck?").

    It is not bit-for-bit repeatable: some GPU operations are not deterministic, and
    forcing them to be costs speed. Two runs with the same seed end up close but not
    identical. That is also why the ensemble in step 8 uses *different* seeds on purpose:
    different seeds give genuinely different models.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)             # CPU generator; DataLoader workers derive from it
    torch.cuda.manual_seed_all(seed)    # GPU generator (dropout, weight init on GPU)


# --------------------------------------------------------------------------------------
# Loss: weighted cross-entropy
# --------------------------------------------------------------------------------------

def class_weights(labels, n_classes=4):
    """Per-class loss weights proportional to 1 / sqrt(class frequency), mean 1.

    Cross-entropy adds up one term per training example. With ~74% no-damage buildings,
    most of the loss, and so most of the gradient, comes from the majority class; the
    model can get a low loss while mostly ignoring the rare classes. Weighting each
    example's loss by its class makes rare-class mistakes more expensive.

    How strongly? Two obvious choices:
      * 1 / frequency          every class contributes equally in total. With a 9:1
                               imbalance a rare example counts 9x; the model then tends
                               to over-predict rare classes, and precision collapses.
      * no weights             the majority dominates.
    1 / sqrt(frequency) sits in between: with 74% vs 8% the ratio is about 3x rather
    than 9x. It is a common compromise, not a law; it is one of the first things worth
    trying to change.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    w[counts == 0] = 0.0                       # a class with no examples gets no weight
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Learning-rate schedule: linear warm-up, then cosine decay
# --------------------------------------------------------------------------------------

def lr_factor(step, total_steps, warmup_steps):
    """Multiplier (0..1) applied to the peak learning rate at a given optimiser step.

    Warm-up: the new head starts from random weights, so its first gradients are large
    and somewhat arbitrary. Big steps at that moment can wreck the pretrained backbone.
    Starting near zero and ramping up over the first few percent of training avoids that.

    Cosine decay: large steps early (move quickly), ever smaller steps later (settle
    into a minimum instead of bouncing around it). The cosine shape is just a smooth
    way from 1 down to 0; it has no deeper meaning, but works reliably and has no extra
    knobs to tune.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# --------------------------------------------------------------------------------------
# Batch size
# --------------------------------------------------------------------------------------

def find_batch_size(model, device, amp, start=32, limit=4096, budget=0.80, image=128):
    """Largest power-of-two batch whose training step stays within ``budget`` of GPU memory.

    Tries one real training step (forward + backward) on random data at 32, 64, 128, ...
    and measures the peak memory. Returns (largest batch that fits, {batch: peak GB}).

    Why measure instead of waiting for an out-of-memory error? On Windows the NVIDIA
    driver can quietly spill into ordinary system RAM when the GPU is full. Nothing
    crashes; training just becomes many times slower. So we stop at 80% of the card,
    which also leaves room for the desktop and for the evaluation batches.
    """
    total = torch.cuda.get_device_properties(device).total_memory
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, peaks, bs = start, {}, start
    model.train()
    while bs <= limit:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            pre = torch.randn(bs, 3, image, image, device=device)
            post = torch.randn(bs, 3, image, image, device=device)
            target = torch.randint(0, 4, (bs,), device=device)
            with torch.autocast("cuda", enabled=amp):
                loss = criterion(model(pre, post), target)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
        except torch.cuda.OutOfMemoryError:
            break
        peaks[bs] = round(peak / 2**30, 2)
        if peak > budget * total:
            break
        best = bs
        bs *= 2
    del pre, post, target, loss
    torch.cuda.empty_cache()
    return best, peaks


# --------------------------------------------------------------------------------------
# One epoch of training, and evaluation
# --------------------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp):
    """One pass over the training data. Returns (mean loss, accuracy) on the batches seen."""
    # model.train() switches on training behaviour: dropout drops, and BatchNorm layers
    # use the statistics of the current batch (and update their running averages).
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for pre, post, label in loader:
        # non_blocking=True lets the copy to the GPU overlap with other work; it needs
        # pin_memory=True in the DataLoader.
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # Mixed precision: inside autocast most operations run in 16-bit floats instead
        # of 32-bit. Half the memory per number and noticeably faster on modern GPUs,
        # with practically no loss in accuracy.
        with torch.autocast("cuda", enabled=amp):
            logits = model(pre, post)
            loss = criterion(logits, label)

        # The classic three lines, zero_grad / backward / step, with one extra:
        #   zero_grad  gradients accumulate by default; clear the previous batch's
        #   backward   compute d(loss)/d(weight) for every weight
        #   step       move each weight a little against its gradient
        # The GradScaler exists because of 16-bit floats: small gradients would round
        # to zero. It multiplies the loss by a large factor before backward and divides
        # the gradients again before the optimiser step.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()                 # the schedule advances per batch, not per epoch

        # .item() copies a number from the GPU to Python. detach() says "no gradients
        # needed for this bookkeeping".
        n = label.shape[0]
        total_loss += loss.detach().item() * n
        correct += (logits.detach().argmax(dim=1) == label).sum().item()
        seen += n
    return total_loss / seen, correct / seen


@torch.no_grad()   # evaluation needs no gradients: saves memory and time
def predict(model, loader, device, amp=True):
    """Run the model over a loader. Returns (logits [N, 4], labels [N]) as numpy arrays."""
    # model.eval(): dropout off, BatchNorm uses its stored running statistics. Forgetting
    # this is a classic bug: predictions would depend on what else is in the batch.
    model.eval()
    all_logits, all_labels = [], []
    for pre, post, label in loader:
        pre, post = pre.to(device, non_blocking=True), post.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(pre, post)
        all_logits.append(logits.float().cpu())     # back to 32-bit for the metrics
        all_labels.append(label)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def evaluate(model, loader, criterion, device, amp=True):
    """Metrics of the model on one loader, plus the raw logits and labels."""
    logits, labels = predict(model, loader, device, amp)
    # The same weighted loss as in training, computed once over the whole set on the CPU.
    loss = nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels),
                                       weight=criterion.weight.cpu()).item()
    metrics = summarize(labels, logits.argmax(axis=1))
    metrics["loss"] = loss
    return metrics, logits, labels


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------

def make_loader(dataset, batch_size, shuffle, cfg, seed=None):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)      # fixes the shuffling order for this run
    # Each worker is a separate Python process that loads and augments crops while the
    # GPU is busy. Workers are not free:
    #   - on Windows each takes several seconds to start (it imports PyTorch from scratch)
    #     and about 1.6 GB of memory;
    #   - each keeps 2 finished batches waiting ("prefetch"). A batch of 512 pairs of
    #     128x128 float images is 200 MB.
    # The first full run used 8 workers for training AND for each of the two evaluation
    # loaders, with evaluation batches of 1024: 24 processes and about 59 of the
    # machine's 62 GB (BLOCKERS.md). Evaluation needs no augmentation and runs once per
    # epoch, so by default it now loads in the main process (0 workers): a few seconds
    # slower per epoch, a fraction of the memory.
    workers = cfg.num_workers if shuffle else min(cfg.num_workers, cfg.eval_num_workers)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers,
        pin_memory=True,                         # faster CPU -> GPU copies
        persistent_workers=workers > 0,
        # Training drops the last, smaller batch: BatchNorm statistics from a handful of
        # examples are noisy. Evaluation must keep every example.
        drop_last=shuffle,
    )


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.run_name = cfg.run_name or f"{cfg.arch}_s{cfg.seed}"
    out_dir = Path(cfg.runs_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.amp = cfg.amp and device.type == "cuda"    # mixed precision is a GPU feature
    seed_everything(cfg.seed)

    # ---- data ----
    train_set = CropDataset(cfg.cache_dir, "train", augment=not cfg.overfit)
    train_labels = train_set.labels
    eval_sets = {}
    if cfg.overfit:
        # Sanity check: can the model memorise a handful of examples? If it cannot push
        # training accuracy to ~100% on 256 crops, something is broken (labels shuffled,
        # gradients not flowing, learning rate off by orders of magnitude), and there is
        # no point training on everything. No augmentation: we WANT memorisation here.
        pick = np.random.default_rng(cfg.seed).choice(len(train_set), cfg.overfit, replace=False)
        train_set = Subset(train_set, pick.tolist())
        train_labels = train_labels[pick]
        eval_sets = {"overfit-subset": train_set}
    else:
        eval_sets = {p: CropDataset(cfg.cache_dir, p) for p in ("val-tiles", "val")}
    select_on = "overfit-subset" if cfg.overfit else "val-tiles"

    # ---- model ----
    model = build_model(cfg.arch, cfg.pretrained, cfg.dropout).to(device)
    probe = {}
    if cfg.batch_size <= 0:
        # "Largest that fits" is an upper limit, not a goal. A bigger batch means fewer
        # weight updates per epoch (135k crops / 512 = 263 steps; at 2048 only 65), and
        # with ~10 epochs that is too few updates to fine-tune well. So the probe stops
        # at max_batch_size. (It also must not try sizes beyond the card: see the note
        # about Windows in find_batch_size. An early version probed 1024 and the run
        # crawled for minutes before training even began.)
        cfg.batch_size, probe = find_batch_size(model, device, cfg.amp, limit=cfg.max_batch_size)
        print(f"batch-size probe (peak GB per batch size): {probe} -> using {cfg.batch_size}",
              flush=True)
    if cfg.overfit:
        cfg.batch_size = min(cfg.batch_size, 64)
    # Let cuDNN time several convolution algorithms on the first batch and keep the
    # fastest. Worth it when every batch has the same shape, as here. Switched on only
    # after the probe: the trial algorithms use extra memory and would distort it.
    torch.backends.cudnn.benchmark = True

    train_loader = make_loader(train_set, cfg.batch_size, True, cfg, cfg.seed)
    eval_loaders = {p: make_loader(d, cfg.eval_batch_size, False, cfg) for p, d in eval_sets.items()}

    # ---- loss, optimiser, schedule ----
    weights = class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # AdamW = Adam with "decoupled" weight decay: every step also shrinks all weights
    # slightly towards zero, a mild pressure against overfitting. Adam adapts the step
    # size per weight, which makes fine-tuning forgiving about the exact learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"run {cfg.run_name}: {cfg.arch}, {count_parameters(model) / 1e6:.1f}M weights, "
          f"{len(train_set):,} training crops, batch {cfg.batch_size}, "
          f"{len(train_loader)} steps/epoch, class weights {weights.cpu().numpy().round(2)}",
          flush=True)

    # ---- epochs ----
    history, best, best_epoch, since_best = [], -1.0, 0, 0
    log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                scheduler, scaler, device, cfg.amp)
        train_time = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "lr": optimizer.param_groups[0]["lr"], "train_seconds": round(train_time, 1)}
        for part, loader in eval_loaders.items():
            row[part], _, _ = evaluate(model, loader, criterion, device, cfg.amp)
        row["epoch_seconds"] = round(time.time() - t0, 1)
        history.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()

        line = f"epoch {epoch:2d}  train loss {train_loss:.3f} acc {train_acc:.3f}"
        for part in eval_loaders:
            m = row[part]
            recalls = " ".join("  - " if r is None else f"{r:.2f}" for r in m["recall"])
            line += (f" | {part}: loss {m['loss']:.3f} acc {m['accuracy']:.3f} "
                     f"F1 {m['macro_f1']:.3f} recall [{recalls}]")
        print(line + f" | {row['epoch_seconds']:.0f}s", flush=True)

        # Early stopping. The model keeps improving on the training set for as long as we
        # let it; what we care about is held-out data. So after every epoch we check
        # val-tiles macro-F1, keep a copy of the weights whenever it is the best so far,
        # and stop when it has not improved for ``patience`` epochs.
        # (Why val-tiles and not val? docs/split_decision.md: val is floods only and has
        # too few minor-damage and destroyed buildings to steer by.)
        score = row[select_on]["accuracy" if cfg.overfit else "macro_f1"]
        if score > best:
            best, best_epoch, since_best = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch},
                       out_dir / "best.pt")
        else:
            since_best += 1
            if since_best >= cfg.patience and not cfg.overfit:
                print(f"no improvement on {select_on} for {cfg.patience} epochs: stopping")
                break
        if cfg.overfit and score >= 0.999:
            break
    log.close()

    # ---- wrap up: reload the best weights and save their predictions ----
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    final = {}
    for part, loader in eval_loaders.items():
        final[part], logits, labels = evaluate(model, loader, criterion, device, cfg.amp)
        if not cfg.overfit:
            np.savez_compressed(out_dir / f"preds_{part}.npz", logits=logits, labels=labels)
    summary = {
        "run_name": cfg.run_name, "arch": cfg.arch, "seed": cfg.seed,
        "best_epoch": best_epoch, "epochs_run": len(history), "selected_on": select_on,
        "batch_size": cfg.batch_size, "batch_size_probe_peak_gb": probe,
        "train_seconds_per_epoch": float(np.mean([h["train_seconds"] for h in history])),
        "seconds_per_epoch": float(np.mean([h["epoch_seconds"] for h in history])),
        "gpu_peak_allocated_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                                  if device.type == "cuda" else None),
        "gpu_peak_reserved_gb": (torch.cuda.max_memory_reserved(device) / 2**30
                                 if device.type == "cuda" else None),
        "class_weights": weights.cpu().tolist(), "class_names": CLASS_NAMES,
        "best": final, "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"best epoch {best_epoch}: " + ", ".join(
        f"{p} acc {m['accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}" for p, m in final.items()))
    return summary


def parse_config(argv=None):
    """Config = defaults, then the JSON file, then any command-line flags that were given."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="JSON file with Config fields")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or f.type == "bool":
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, type=type(f.default), default=None)
    args = ap.parse_args(argv)
    values = {}
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    values.update({f.name: getattr(args, f.name) for f in fields(Config)
                   if getattr(args, f.name) is not None})
    return Config(**values)


if __name__ == "__main__":   # required on Windows: DataLoader workers re-import this file
    run(parse_config())
