"""Why is val so much worse than val-tiles? Two diagnostics -> docs/gap_diagnosis.md

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\gap_diagnosis.py --runs runs\\stack_aug_s0 runs\\stack_aug_s1 runs\\stack_aug_s2

(a) Input swaps: which of the two images carries the problem?
    The trained models are evaluated again with one input tampered with:
        normal        (pre, post) as usual
        both = pre    the post image is replaced by the pre image: nothing changed at all
        both = post   the pre image is replaced by the post image: also "no change"
        pre blank     the pre image is replaced by a blank (mean-colour) image
        post blank    the post image is replaced by a blank image
    A model that truly compares before and after should answer "no-damage" almost always
    when both inputs are the same image. If it still reports damage on val for
    "both = post", then the post image's look alone (flood water, haze, darkness) is
    enough to trigger it; if "both = pre" triggers it, the pre image is unusual too.
    The blank variants show how much each input matters at all.

(b) Colour statistics per disaster, for pre and post crops: are the val disasters
    outside the range the model saw in training?

Validation and training crops only; the crop cache contains no test crops and
CropDataset refuses to load the test part (src/split.py).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.colors import INK, SURFACE  # noqa: E402
from src.dataset import CropDataset, read_index  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.model import build_model  # noqa: E402
from src.reporting import df_to_markdown  # noqa: E402
from src.split import require_final  # noqa: E402

PARTS = ["val-tiles", "val"]
VARIANTS = ["normal", "both = pre", "both = post", "pre blank", "post blank"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#898781", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ---- (a) input swaps -------------------------------------------------------------------

def variant_inputs(pre, post):
    """The five (pre, post) pairs fed to the model for one batch.

    "Blank" is an all-zero tensor. After normalisation, zero is the average colour, so a
    blank image is a flat mid-grey: no structure, no information.
    """
    blank = torch.zeros_like(pre)
    return {"normal": (pre, post), "both = pre": (pre, pre), "both = post": (post, post),
            "pre blank": (blank, post), "post blank": (pre, blank)}


@torch.no_grad()
def swap_predictions(run_dir, device, batch_size=512):
    """{part: {variant: probabilities [N, 4]}} for one trained run."""
    state = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    cfg = state["config"]
    model = build_model(cfg["arch"], pretrained=False, dropout=cfg["dropout"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    out = {}
    for part in require_final(PARTS):     # never "test": there is no final flag here
        # from_config: the same normalisation / histogram matching the model trained with.
        # Note for histogram-matched models: the swap happens after matching.
        loader = DataLoader(CropDataset.from_config(cfg, part), batch_size=batch_size)
        probs = {v: [] for v in VARIANTS}
        for pre, post, _ in loader:
            pre, post = pre.to(device), post.to(device)
            for name, (a, b) in variant_inputs(pre, post).items():
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(a, b)
                probs[name].append(torch.softmax(logits.float(), dim=1).cpu())
        out[part] = {v: torch.cat(p).numpy() for v, p in probs.items()}
    return out


def swap_table(per_run, index):
    """Mean over runs (and the range) of accuracy, macro-F1, share predicted no-damage
    and mean top probability, for each group of buildings and each input variant."""
    groups = [("val-tiles (all)", index["part"] == "val-tiles"), ("val (all)", index["part"] == "val")]
    for d in sorted(index.loc[index["part"] == "val", "disaster"].unique()):
        groups.append((f"val: {d}", (index["part"] == "val") & (index["disaster"] == d)))
    rows = []
    for name, mask in groups:
        part = "val" if name.startswith("val:") or name.startswith("val (") else "val-tiles"
        sub = index[index["part"] == part].sort_values("row")
        pick = mask[sub.index].to_numpy()
        y = sub["label"].to_numpy()[pick]
        for v in VARIANTS:
            stats = []
            for run in per_run:
                p = run[part][v][pick]
                pred = p.argmax(axis=1)
                s = summarize(y, pred)
                stats.append((s["accuracy"], s["macro_f1"], (pred == 0).mean(), p.max(axis=1).mean()))
            a = np.array(stats)
            rows.append({"buildings": name, "input": v, "accuracy": a[:, 0].mean(),
                         "macro-F1": a[:, 1].mean(),
                         "predicted no-damage": a[:, 2].mean(),
                         "predicted no-damage, range over seeds":
                             f"{a[:, 2].min():.2f}-{a[:, 2].max():.2f}",
                         "mean top probability": a[:, 3].mean(),
                         "truly no-damage": float((y == 0).mean())})
    return pd.DataFrame(rows)


# ---- (b) colour statistics ---------------------------------------------------------------

def colour_stats(cache_dir, index, per_group=3000, seed=0):
    """Per disaster and image (pre / post): mean colour, brightness and contrast of the
    crops, ignoring the black padding of crops that stick out of the tile."""
    rng = np.random.default_rng(seed)
    arrays = {}
    rows = []
    for (disaster, part), g in index.groupby(["disaster", "part"]):
        require_final(part)
        if part not in arrays:
            arrays[part] = {w: np.load(Path(cache_dir) / f"{part}_{w}.npy", mmap_mode="r")
                            for w in ("pre", "post")}
        take = np.sort(rng.choice(g["row"].to_numpy(), min(per_group, len(g)), replace=False))
        row = {"disaster": disaster, "part": part, "crops": len(take)}
        pre = np.asarray(arrays[part]["pre"][take], dtype=np.float32)
        post = np.asarray(arrays[part]["post"][take], dtype=np.float32)
        # padding is black in BOTH images (same crop window), real pixels almost never are
        real = (pre.sum(axis=-1) > 0) | (post.sum(axis=-1) > 0)
        for which, img in (("pre", pre), ("post", post)):
            px = img[real]                                   # [pixels, 3]
            grey = px.mean(axis=1)
            row.update({f"{which} R": px[:, 0].mean(), f"{which} G": px[:, 1].mean(),
                        f"{which} B": px[:, 2].mean(), f"{which} brightness": grey.mean(),
                        f"{which} contrast": grey.std(),
                        f"{which} green excess": (px[:, 1] - (px[:, 0] + px[:, 2]) / 2).mean()})
        row["post - pre brightness"] = row["post brightness"] - row["pre brightness"]
        rows.append(row)
    t = pd.DataFrame(rows)
    # train and val-tiles are the same disasters: pool them, weighted by crops sampled
    t["role"] = np.where(t["part"] == "val", "val (unseen)", "training disaster")
    num = [c for c in t.columns if c not in ("disaster", "part", "role", "crops")]
    pooled = (t.groupby(["role", "disaster"])
               .apply(lambda g: pd.Series({**{c: np.average(g[c], weights=g["crops"]) for c in num},
                                           "crops": g["crops"].sum()}), include_groups=False)
               .reset_index())
    return pooled.sort_values(["role", "disaster"]).reset_index(drop=True)


def plot_colours(stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (x, y, xl, yl) in zip(axes, (
            ("brightness", "green excess", "brightness (0-255)", "green excess: G - (R + B) / 2"),
            ("brightness", "contrast", "brightness (0-255)", "contrast (std of grey value)"))):
        for _, r in stats.iterrows():
            color = "#e34948" if r["role"].startswith("val") else "#2a78d6"
            pre = (r[f"pre {x}"], r[f"pre {y}"])
            post = (r[f"post {x}"], r[f"post {y}"])
            ax.annotate("", xy=post, xytext=pre,
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.3))
            ax.scatter(*pre, s=26, color=color, zorder=3)
            ax.annotate(r["disaster"].replace("hurricane-", "").replace("-", " "), post,
                        textcoords="offset points", xytext=(4, 3), fontsize=7.5, color=color)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title("Each arrow: one disaster, from its pre crops (dot) to its post crops (tip)\n"
                      "blue = training disasters, red = val (unseen)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def outside_training_range(stats, column):
    """For each val disaster: is ``column`` outside the min-max range of the training
    disasters, and by how much?"""
    train = stats[stats["role"] == "training disaster"][column]
    lo, hi = train.min(), train.max()
    out = []
    for _, r in stats[stats["role"] != "training disaster"].iterrows():
        v = r[column]
        where = "below" if v < lo else "above" if v > hi else "inside"
        out.append(f"{r['disaster']} {v:.0f} ({where})")
    return f"training disasters {lo:.0f}-{hi:.0f}; " + ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=Path(r"C:\Users\wikel\dev\xbd-data\crops"))
    ap.add_argument("--out", type=Path, default=Path("docs/gap_diagnosis.md"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--per-group", type=int, default=3000, help="crops sampled per disaster and part")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = args.fig_dir.relative_to(args.out.parent).as_posix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = read_index(args.cache_dir)
    index = index[index["part"].isin(["train", "val-tiles", "val"])].reset_index(drop=True)
    runs = [r for r in args.runs if (r / "best.pt").exists()]
    per_run = []
    for r in runs:
        per_run.append(swap_predictions(r, device))
        print(f"input swaps done for {r.name}", flush=True)
    swaps = swap_table(per_run, index)
    stats = colour_stats(args.cache_dir, index, args.per_group)
    plot_colours(stats, args.fig_dir / "colour_statistics.png")

    names = ", ".join(f"`{r.name}`" for r in runs)
    show = ["role", "disaster", "crops", "pre brightness", "post brightness",
            "post - pre brightness", "pre contrast", "post contrast", "pre green excess",
            "post green excess"]
    rgb = ["role", "disaster", "pre R", "pre G", "pre B", "post R", "post G", "post B"]
    L = ["# Gap diagnosis: why is val worse than val-tiles? (validation only)\n",
         f"Generated by `scripts/gap_diagnosis.py` from {names}. Training and validation "
         "crops only; the test set is not touched.\n",
         "## (a) Which input carries the shift?\n",
         "The trained models are run again with one input replaced. Numbers are the mean "
         f"over {len(runs)} seed(s). 'predicted no-damage' is the share of buildings the "
         "model calls undamaged; compare it with 'truly no-damage'.\n",
         "- **both = pre / both = post**: the model gets the same image twice, so nothing "
         "has changed between its two inputs. A model that compares before and after "
         "should answer no-damage for nearly everything.",
         "- **pre blank / post blank**: one input is a flat grey image. Shows how much the "
         "model depends on that input.\n",
         df_to_markdown(swaps, float_fmt="{:.3f}", index=False) + "\n",
         "How to read it:\n",
         "- If 'both = post' on val predicts far less no-damage than on val-tiles, the "
         "post images of the unseen disasters look like damage to the model by "
         "themselves, whatever the pre image shows.",
         "- If 'both = pre' on val still predicts a lot of damage, the pre images are "
         "unfamiliar too.",
         "- If 'pre blank' barely changes the predictions compared with 'normal', the "
         "model is mostly reading the post image and hardly comparing.\n",
         "Caveat: a pair of identical images is itself unlike anything in training (real "
         "pairs always differ a little), so these are probes of the model's behaviour, "
         "not measurements of accuracy.\n",
         "## (b) Colour statistics per disaster\n",
         f"Up to {args.per_group:,} crops per disaster and part, black padding excluded; "
         "training disasters pool their train and val-tiles crops. Brightness = mean of "
         "R, G, B (0-255); contrast = standard deviation of the grey value; green excess "
         "= G - (R + B) / 2.\n",
         df_to_markdown(stats[show], float_fmt="{:.1f}", index=False) + "\n",
         "Mean colour per channel:\n",
         df_to_markdown(stats[rgb], float_fmt="{:.1f}", index=False) + "\n",
         f"![colour statistics per disaster]({fig_rel}/colour_statistics.png)\n",
         "Are the val disasters outside what the model saw in training?\n",
         f"- pre brightness: {outside_training_range(stats, 'pre brightness')}",
         f"- post brightness: {outside_training_range(stats, 'post brightness')}",
         f"- pre green excess: {outside_training_range(stats, 'pre green excess')}",
         f"- post green excess: {outside_training_range(stats, 'post green excess')}",
         f"- post contrast: {outside_training_range(stats, 'post contrast')}\n",
         "These are averages over whole disasters. Being inside the range of averages "
         "does not mean the images look alike, and being outside it does not prove that "
         "colour is the cause of the gap; the fixes in `docs/results.md` test that "
         "directly.\n"]
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
