"""Ways to get a confidence score for each prediction.

In use: (a) max softmax probability and (c) the deep ensemble. Both work from the
predictions saved by src/train.py; ``collect_methods`` builds them.

Set aside: (b) MC dropout. It was tried on the base model and its confidence turned out
practically identical to (a), at 20 times the cost (docs/results.md has the numbers).
The reason is structural: dropout sits only in front of the last layer, so the 20 passes
differ very little. The code stays here, tested, because it documents what was tried and
because it becomes interesting again if dropout is ever put inside the backbone. It is
no longer part of the reports. To run it anyway:
    .\\.venv\\Scripts\\python.exe -m src.uncertainty --run runs\\stack_aug_s0 --passes 20

A confidence score is only useful for triage if it RANKS predictions well: mistakes
should get low scores, correct predictions high scores. The methods differ in where the
doubt comes from and what they cost.

a) Max softmax probability (MSP), after temperature scaling
   Confidence = the largest of the four class probabilities. Free: one forward pass, no
   change to the model. It measures how close a building is to the model's decision
   boundary, but one model can be confidently wrong about things it never saw.

b) MC dropout ("Monte Carlo dropout")
   Dropout normally runs only in training. Leave it ON at prediction time and run the
   same building through the model 20 times: each pass drops a different random 30% of
   the head's inputs, so each pass is a slightly different model. If the 20 answers
   agree, the prediction does not hinge on any particular features; if they scatter,
   it does. We average the 20 probability vectors and report two scores:
     - mean max prob: the average, over the 20 passes, of each pass's top probability
     - predictive entropy of the averaged probabilities: 0 when all mass is on one
       class, log(4) = 1.39 when it is spread evenly. Confidence = -entropy.
   Cost: 20 forward passes. Caveat for THIS model: dropout sits only in the classifier
   head, so the backbone features are identical in every pass. MC dropout here probes
   the head's uncertainty, not the whole network's.

c) Deep ensemble
   Train the same architecture several times with different random seeds (different
   initial head weights, different shuffling and augmentation) and average the
   probabilities. Each model makes its own mistakes; where they disagree the average is
   flat, so confidence is low. Usually the strongest of the three, and it tends to
   improve accuracy too, but costs N trainings and N forward passes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.calibrate import fit_temperature
from src.dataset import CropDataset
from src.metrics import softmax
from src.model import build_model


def enable_mc_dropout(model):
    """Put the model in eval mode, then switch ONLY the dropout layers back to training.

    The order matters. ``model.eval()`` also fixes BatchNorm to its stored statistics,
    which we want: BatchNorm in training mode would make a building's prediction depend
    on the other buildings in its batch. Dropout is the only layer that should stay
    random.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


@torch.no_grad()
def mc_dropout_probs(model, loader, device, passes=20, amp=True, temperature=1.0):
    """Class probabilities from ``passes`` stochastic forward passes: array [passes, N, 4].

    Each batch is loaded once and pushed through the model ``passes`` times, which is
    much cheaper than iterating over the dataset 20 times.

    ``temperature`` divides the logits of every pass before the softmax (see
    src/calibrate.py), so that MC dropout starts from the same calibrated probabilities
    as the plain "max prob" method and the two can be compared fairly.
    """
    enable_mc_dropout(model)
    out = []
    for pre, post, _ in loader:
        pre, post = pre.to(device, non_blocking=True), post.to(device, non_blocking=True)
        batch = []
        for _ in range(passes):
            with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(pre, post)
            batch.append(torch.softmax(logits.float() / temperature, dim=1).cpu())
        out.append(torch.stack(batch))            # [passes, B, 4]
    model.eval()
    return torch.cat(out, dim=1).numpy()


def predictive_entropy(probs):
    """Entropy of each row of probabilities, in nats. Higher = more uncertain."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def mc_dropout_summary(pass_probs):
    """Reduce [passes, N, 4] to what we keep: mean probs, mean max prob, entropy."""
    mean = pass_probs.mean(axis=0)
    return {"probs": mean.astype(np.float32),
            "mean_max_prob": pass_probs.max(axis=2).mean(axis=0).astype(np.float32),
            "entropy": predictive_entropy(mean).astype(np.float32)}


def ensemble_probs(list_of_logits):
    """Average the probabilities (not the logits) of several models: array [N, 4].

    Averaging probabilities is the standard choice: a model that is very sure and very
    wrong contributes at most 1.0 to a class. Averaging logits would let one model's
    extreme scores dominate the others.
    """
    return np.mean([softmax(np.asarray(l, dtype=np.float64)) for l in list_of_logits], axis=0)


def method_scores(logits, labels, temperature=None, mc=None, ensemble_logits=None,
                  ensemble_temperature=None):
    """Predictions and confidence for every method we can build from what is given.

    Returns {method name: {"pred": [N], "confidence": [N], "probs": [N, 4]}}.
    ``confidence`` is "higher = more sure" for every method, so the triage code can
    treat them all the same way.
    """
    out = {}
    p = softmax(np.asarray(logits, dtype=np.float64))
    out["max prob (raw)"] = {"probs": p, "pred": p.argmax(1), "confidence": p.max(1)}
    if temperature is not None:
        p = softmax(np.asarray(logits, dtype=np.float64) / temperature)
        out["max prob (temp. scaled)"] = {"probs": p, "pred": p.argmax(1), "confidence": p.max(1)}
    if mc is not None:
        pred = mc["probs"].argmax(1)
        out["MC dropout (mean max prob)"] = {"probs": mc["probs"], "pred": pred,
                                             "confidence": mc["mean_max_prob"]}
        # entropy is "higher = less sure", so flip the sign to get a confidence
        out["MC dropout (entropy)"] = {"probs": mc["probs"], "pred": pred,
                                       "confidence": -mc["entropy"]}
    if ensemble_logits is not None and len(ensemble_logits) > 1:
        p = ensemble_probs(ensemble_logits)
        out[f"ensemble of {len(ensemble_logits)}"] = {"probs": p, "pred": p.argmax(1),
                                                      "confidence": p.max(1)}
        if ensemble_temperature is not None:
            # An average of several models is usually LESS confident than each member
            # (they disagree a little), so the ensemble needs its own temperature. Its
            # "logits" are the log of the averaged probabilities.
            p = softmax(np.log(np.clip(p, 1e-12, None)) / ensemble_temperature)
            out[f"ensemble of {len(ensemble_logits)} (temp. scaled)"] = {
                "probs": p, "pred": p.argmax(1), "confidence": p.max(1)}
    return out


def collect_methods(run_dir, seed_dirs=(), parts=("val-tiles", "val")):
    """Build every available method for one base run from the files in runs/.

    run_dir     the single model: its logits give the max-prob methods
    seed_dirs   all ensemble members' run folders, INCLUDING run_dir; fewer than two
                finished members means no ensemble

    Returns ({part: method_scores(...)}, info) where info holds the temperatures.
    Both temperatures are fitted on val-tiles only.
    """
    run_dir = Path(run_dir)
    members = [Path(d) for d in seed_dirs if (Path(d) / "preds_val.npz").exists()]
    preds = {p: np.load(run_dir / f"preds_{p}.npz") for p in parts}
    temperature = fit_temperature(preds["val-tiles"]["logits"], preds["val-tiles"]["labels"])
    member_logits = {p: [np.load(d / f"preds_{p}.npz")["logits"] for d in members] for p in parts}
    ens_t = None
    if len(members) > 1:
        ens = ensemble_probs(member_logits["val-tiles"])
        ens_t = fit_temperature(np.log(np.clip(ens, 1e-12, None)), preds["val-tiles"]["labels"])
    methods = {}
    for p in parts:
        # MC dropout is deliberately not loaded here any more (see the module docstring).
        methods[p] = method_scores(
            preds[p]["logits"], preds[p]["labels"], temperature,
            ensemble_logits=member_logits[p], ensemble_temperature=ens_t)
    return methods, {"temperature": temperature, "ensemble_temperature": ens_t,
                     "members": [d.name for d in members]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="a folder under runs/")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--parts", nargs="+", default=["val-tiles", "val"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 = load in the main process: slower, but no extra memory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.run / "best.pt", map_location=device, weights_only=True)
    cfg = state["config"]
    model = build_model(cfg["arch"], pretrained=False, dropout=cfg["dropout"]).to(device)
    model.load_state_dict(state["model"])
    torch.manual_seed(args.seed)          # dropout masks are random: fix them
    torch.cuda.manual_seed_all(args.seed)
    # The same temperature as the plain max-prob method: fitted on this model's ordinary
    # (dropout off) val-tiles predictions.
    fit = np.load(args.run / "preds_val-tiles.npz")
    temperature = fit_temperature(fit["logits"], fit["labels"])
    (args.run / "mc_dropout.json").write_text(json.dumps(
        {"passes": args.passes, "temperature": temperature, "seed": args.seed}), encoding="utf-8")
    print(f"temperature {temperature:.3f} (from val-tiles), {args.passes} passes", flush=True)
    for part in args.parts:
        # CropDataset refuses part="test" (src/split.py); there is no --final here on purpose.
        loader = DataLoader(CropDataset.from_config(cfg, part), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
        summary = mc_dropout_summary(mc_dropout_probs(model, loader, device, args.passes,
                                                      temperature=temperature))
        np.savez_compressed(args.run / f"mc_dropout_{part}.npz", **summary)
        print(f"{part}: {len(summary['entropy']):,} buildings x {args.passes} passes -> "
              f"{args.run / f'mc_dropout_{part}.npz'}", flush=True)


if __name__ == "__main__":
    main()
