"""Class counts, building sizes and class imbalance for one xBD split.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\dataset_stats.py --data-root C:\\Users\\wikel\\dev\\xbd-data\\test

Writes PNG charts, CSV tables and summary.md to outputs/stats/ (override with --out-dir).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files only; no window pops up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import src...` works
from src.colors import DAMAGE_COLORS, GRID, INK, INK_SECONDARY, MUTED, SURFACE  # noqa: E402
from src.labels import DAMAGE_CLASSES, load_labels  # noqa: E402

REAL_CLASSES = DAMAGE_CLASSES[:4]  # everything except "un-classified"


def style_axes(ax):
    """Recessive axes: hairline grid behind the data, no top/right frame."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def new_figure(width=9, height=5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {path}")


def counts_table(df):
    """Rows = disasters, columns = damage classes, values = building counts."""
    table = pd.crosstab(df["disaster"], df["damage"])
    table = table.reindex(columns=DAMAGE_CLASSES, fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def imbalance_table(df):
    """Per-class share, imbalance ratio and inverse-frequency weight (real classes only)."""
    real = df[df["damage"].isin(REAL_CLASSES)]
    counts = real["damage"].value_counts().reindex(REAL_CLASSES)
    out = pd.DataFrame({"count": counts})
    out["share"] = out["count"] / out["count"].sum()
    # How many times more common the biggest class is than this one.
    out["majority_ratio"] = out["count"].max() / out["count"]
    # A common starting point for loss weights: total / (n_classes * count). Mean weight = 1
    # for a perfectly balanced set. Informational only; nothing is trained here.
    out["inverse_freq_weight"] = out["count"].sum() / (len(out) * out["count"])
    return out


def plot_overall(df, path):
    counts = df["damage"].value_counts().reindex(DAMAGE_CLASSES)
    fig, ax = new_figure(8, 4.5)
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=[DAMAGE_COLORS[c] for c in counts.index], width=0.6)
    total = counts.sum()
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,}\n({n / total:.1%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("buildings")
    ax.set_title(f"Buildings per damage class (n = {total:,})", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, path)


def plot_by_disaster(table, path, share=False):
    """Horizontal stacked bars, one per disaster. share=True normalises each bar to 100%."""
    data = table[DAMAGE_CLASSES]
    if share:
        data = data.div(data.sum(axis=1), axis=0)
    data = data.iloc[::-1]  # biggest disaster at the top
    fig, ax = new_figure(9, 0.45 * len(data) + 1.6)
    left = np.zeros(len(data))
    for cls in DAMAGE_CLASSES:
        vals = data[cls].to_numpy(dtype=float)
        # white edge = the thin gap between stacked segments
        ax.barh(data.index.astype(str), vals, left=left, color=DAMAGE_COLORS[cls],
                edgecolor=SURFACE, linewidth=1.5, height=0.7, label=cls)
        left += vals
    if share:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("share of buildings")
        ax.set_title("Damage-class mix per disaster", loc="left")
    else:
        for y, total in enumerate(table["total"].iloc[::-1]):
            ax.annotate(f"{total:,}", (total, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, table["total"].max() * 1.1)
        ax.set_xlabel("buildings")
        ax.set_title("Buildings per disaster, by damage class", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12 if share else -0.1),
              frameon=False, fontsize=9)
    save(fig, path)


def plot_size_hist(df, path):
    """Histogram of building area on a log x-axis (sizes span several orders of magnitude)."""
    area = df.loc[df["area_px"] > 0, "area_px"]
    bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 60)
    fig, ax = new_figure(8, 4.5)
    ax.hist(area, bins=bins, color=DAMAGE_COLORS["no-damage"], edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    for q, label in ((0.5, "median"), (0.05, "5th pct"), (0.95, "95th pct")):
        v = area.quantile(q)
        ax.axvline(v, color=INK_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate(f"{label}\n{v:,.0f} px", (v, ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("building footprint area (pixels, log scale)")
    ax.set_ylabel("buildings")
    ax.set_title("Building-size distribution (all buildings)", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    save(fig, path)


def plot_size_by_class(df, path):
    """Box plot of area per damage class: do damaged buildings tend to be bigger/smaller?"""
    groups = [df.loc[(df["damage"] == c) & (df["area_px"] > 0), "area_px"] for c in DAMAGE_CLASSES]
    fig, ax = new_figure(8, 4.5)
    bp = ax.boxplot(groups, tick_labels=DAMAGE_CLASSES, patch_artist=True, showfliers=False,
                    widths=0.5, medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, cls in zip(bp["boxes"], DAMAGE_CLASSES):
        patch.set_facecolor(DAMAGE_COLORS[cls])
        patch.set_edgecolor(DAMAGE_COLORS[cls])
        patch.set_alpha(0.85)
    for i, g in enumerate(groups, start=1):
        # label sits just right of the box (boxes are 0.5 wide) so it never overlaps a fill
        ax.annotate(f"{g.median():,.0f}", (i + 0.27, g.median()), xytext=(2, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylabel("footprint area (pixels, log scale)")
    ax.set_title("Building size by damage class (line = median; box = middle 50%; whiskers = 1.5 IQR)",
                 loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    save(fig, path)


def df_to_markdown(df, float_fmt="{:.3f}"):
    """Tiny markdown-table writer so we don't need the optional `tabulate` package."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # itertuples keeps each column's own type (iterrows would turn ints into floats).
    for idx, *row in df.itertuples():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("")
            elif isinstance(v, (float, np.floating)):
                cells.append(float_fmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stats"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {args.data_root} ...")
    df = load_labels(args.data_root)
    n_tiles = df.groupby(["disaster", "tile_id"], observed=True).ngroups
    # Tiles with zero buildings produce no rows, so count them from the label files instead.
    n_label_tiles = len(list((args.data_root / "labels").glob("*_post_disaster.json")))
    print(f"  {len(df):,} buildings on {n_tiles:,} tiles ({n_label_tiles - n_tiles:,} tiles have none)")

    table = counts_table(df)
    imb = imbalance_table(df)
    size_desc = df.groupby("damage", observed=False)["area_px"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    size_desc.loc["all"] = df["area_px"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    # Per-disaster imbalance: majority real class / minority real class (0 counts -> NaN, blank).
    real = table[REAL_CLASSES]
    per_disaster = pd.DataFrame({
        "buildings": table["total"],
        "tiles": df.groupby("disaster", observed=True)["tile_id"].nunique(),
        "damaged_share": real[REAL_CLASSES[1:]].sum(axis=1) / real.sum(axis=1),
        "unclassified_share": table["un-classified"] / table["total"],
        "rarest_class": real.idxmin(axis=1),
        "majority_to_minority": real.max(axis=1) / real.min(axis=1).replace(0, np.nan),
    }).reindex(table.index)  # same order as the counts table (largest disaster first)

    # Data-quality checks worth knowing about before modelling.
    invalid = (~df["polygon"].map(lambda p: p.is_valid)).sum()
    tiny = (df["area_px"] < 10).sum()
    buildings_per_tile = df.groupby(["disaster", "tile_id"], observed=True).size()

    table.to_csv(args.out_dir / "class_counts_by_disaster.csv")
    imb.to_csv(args.out_dir / "class_imbalance.csv")
    size_desc.to_csv(args.out_dir / "building_size_by_class.csv")
    per_disaster.to_csv(args.out_dir / "imbalance_by_disaster.csv")

    print("Drawing charts ...")
    plot_overall(df, args.out_dir / "class_counts_overall.png")
    plot_by_disaster(table, args.out_dir / "class_counts_by_disaster.png")
    plot_by_disaster(table, args.out_dir / "class_share_by_disaster.png", share=True)
    plot_size_hist(df, args.out_dir / "building_size_hist.png")
    plot_size_by_class(df, args.out_dir / "building_size_by_class.png")

    md = [
        "# Dataset statistics\n",
        f"Data root: `{args.data_root}`\n",
        f"- Buildings: **{len(df):,}** on **{n_tiles:,}** tiles across "
        f"**{df['disaster'].nunique()}** disasters",
        f"- Tiles with no buildings at all: {n_label_tiles - n_tiles:,} of {n_label_tiles:,}",
        f"- Buildings per tile: median {buildings_per_tile.median():.0f}, "
        f"max {buildings_per_tile.max():,}",
        f"- Invalid polygons (self-intersecting etc.): {invalid:,}",
        f"- Tiny polygons (< 10 px area): {tiny:,}",
        "- Areas are in image pixels. The label `gsd` field (metres per pixel) varies by "
        "sensor/tile, so pixel areas are not directly comparable across disasters.\n",
        "## Class counts per disaster\n",
        df_to_markdown(table),
        "\n![counts](class_counts_by_disaster.png)\n\n![share](class_share_by_disaster.png)\n",
        "## Class imbalance (excluding un-classified)\n",
        df_to_markdown(imb),
        "\n`majority_ratio` = how many times more common the largest class is. "
        "`inverse_freq_weight` = total / (4 x count), a typical starting loss weight.\n",
        "![overall](class_counts_overall.png)\n",
        "## Imbalance per disaster\n",
        df_to_markdown(per_disaster),
        "\n`damaged_share` = (minor + major + destroyed) / classified buildings. "
        "`majority_to_minority` is blank when a class is absent from that disaster.\n",
        "## Building size (pixels)\n",
        df_to_markdown(size_desc, float_fmt="{:,.1f}"),
        "\n![size](building_size_hist.png)\n\n![size by class](building_size_by_class.png)\n",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
