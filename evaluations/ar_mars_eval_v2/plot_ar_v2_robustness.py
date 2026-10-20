#!/usr/bin/env python3
"""
plot_ar_v2_robustness.py — is the AR skill against MARS stable, or does it rest
on the years the W-Net was trained on?

The whole-record IoU of 0.301 says nothing about stability. This figure asks
four separate stability questions of the same monthly record, all of which the
single number hides:

  (a) does skill drift across the 45-year record, and do the held-out years
      stand out from the years the model saw
  (b) is skill on the training years different from the validation and test
      years, which is the generalisation test
  (c) does skill depend on the time of year
  (d) does the model reproduce the month-to-month variability of AR area, not
      just its mean

Panels (a) to (c) use unweighted pixel counts, as accumulated in
ar_monthly_v2.csv. Panel (d) needs no weighting, since it is a correlation of
two series computed the same way.

The split years are the module constants in wnet_split_common (VAL_YEARS,
TEST_YEARS), which are fixed for every Phase-2 run, so they apply to whichever
Phase-2 checkpoint produced the v2 product.

Usage
  python plot_ar_v2_robustness.py --v2-dir . --out ar_v2_robustness
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

VAL_YEARS = [1987, 1998, 2007, 2016]
TEST_YEARS = [1983, 1994, 2003, 2012, 2021]

INK = "#000000"
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
FS_TITLE, FS_LAB, FS_TICK, FS_LEG = 13, 11.5, 10.5, 10.5
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def metrics(tp, fp, fn):
    """Same definitions as block() in phase4_ar_vs_mars_v2.py."""
    tp, fp, fn = float(tp), float(fp), float(fn)
    return dict(IoU=tp / max(tp + fp + fn, 1),
                F1=2 * tp / max(2 * tp + fp + fn, 1),
                POD=tp / max(tp + fn, 1),
                FAR=fp / max(tp + fp, 1),
                BIAS=(tp + fp) / max(tp + fn, 1))


def load_monthly(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(year=int(r["year"]), month=int(r["month"]),
                             tp=float(r["tp"]), fp=float(r["fp"]), fn=float(r["fn"]),
                             n_t=float(r["n_timesteps"])))
    return rows


def agg(rows):
    return metrics(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows),
                   sum(r["fn"] for r in rows))


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=FS_TITLE, color=INK, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LAB, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LAB, color=INK)
    ax.tick_params(labelsize=FS_TICK, colors=INK)
    for s in ax.spines.values():
        s.set_edgecolor("#8a8983")
    ax.grid(alpha=0.25)


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "ar_v2_robustness"))
    args = ap.parse_args()

    rows = load_monthly(os.path.join(args.v2_dir, "ar_monthly_v2.csv"))
    z = np.load(os.path.join(args.v2_dir, "ar_v2_merged.npz"))
    npix = float(len(z["lat"]) * len(z["lon"]))

    years = sorted({r["year"] for r in rows})
    ann = {y: agg([r for r in rows if r["year"] == y]) for y in years}
    split_of = {y: ("validation" if y in VAL_YEARS else
                    "test" if y in TEST_YEARS else "training") for y in years}
    splits = [("training", C_TRAIN), ("validation", C_VAL), ("test", C_TEST)]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.08,
                        hspace=0.33, wspace=0.20)

    # ---- (a) annual skill across the record, held-out years marked
    ax = axes[0, 0]
    for key, ls, lab in (("IoU", "-", "IoU"), ("POD", "--", "POD"),
                         ("FAR", ":", "FAR")):
        ax.plot(years, [ann[y][key] for y in years], ls, color=INK, lw=1.6,
                zorder=2, label=lab)
    for name, col in splits[1:]:
        ys = [y for y in years if split_of[y] == name]
        ax.scatter(ys, [ann[y]["IoU"] for y in ys], s=64, color=col, zorder=4,
                   edgecolor="white", linewidth=1.0, label=f"{name} year")
    sl = np.polyfit(years, [ann[y]["IoU"] for y in years], 1)[0]
    ax.text(0.015, 0.06, f"IoU trend {sl * 10:+.4f} per decade",
            transform=ax.transAxes, fontsize=FS_LEG, color=INK)
    style(ax, "(a)  Annual skill against MARS", "Year", "Score")
    ax.set_ylim(0.15, 0.68)
    ax.legend(fontsize=FS_LEG, frameon=False, ncol=5, loc="upper left",
              labelcolor=INK, columnspacing=1.1, handletextpad=0.5)

    # ---- (b) training versus held-out years
    ax = axes[0, 1]
    keys = ("IoU", "F1", "POD", "FAR")
    x = np.arange(len(keys))
    w = 0.26
    for i, (name, col) in enumerate(splits):
        m = agg([r for r in rows if split_of[r["year"]] == name])
        bars = ax.bar(x + (i - 1) * w, [m[k] for k in keys], w, color=col,
                      label=f"{name} ({sum(1 for y in years if split_of[y] == name)} yr)")
        for b, k in zip(bars, keys):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{m[k]:.3f}", ha="center", fontsize=9.5, color=INK)
    style(ax, "(b)  Skill by W-Net data split", None, "Score")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=FS_LAB, color=INK)
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=FS_LEG, frameon=False, ncol=3, loc="upper left",
              labelcolor=INK)

    # ---- (c) skill by month of year
    ax = axes[1, 0]
    for key, mark in (("IoU", "o"), ("POD", "s"), ("FAR", "^")):
        v = [agg([r for r in rows if r["month"] == m])[key] for m in range(1, 13)]
        ax.plot(range(1, 13), v, marker=mark, color=INK, lw=1.6, ms=5,
                markerfacecolor="white", label=key)
    style(ax, "(c)  Skill by month of year", "Month", "Score")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=FS_TICK, color=INK)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=FS_LEG, frameon=False, ncol=3, labelcolor=INK)

    # ---- (d) does it track month-to-month AR area variability
    ax = axes[1, 1]
    mars = np.array([(r["tp"] + r["fn"]) / (r["n_t"] * npix) for r in rows])
    wnet = np.array([(r["tp"] + r["fp"]) / (r["n_t"] * npix) for r in rows])
    mon = np.array([r["month"] for r in rows])
    cols = [dict(training=C_TRAIN, validation=C_VAL,
                 test=C_TEST)[split_of[r["year"]]] for r in rows]
    ax.scatter(mars, wnet, s=16, c=cols, alpha=0.75, linewidth=0)
    # zoomed to the data, not anchored at 0, or the cloud collapses into a corner
    lo = min(mars.min(), wnet.min()) * 0.96
    lim = [lo, max(mars.max(), wnet.max()) * 1.02]
    ax.plot(lim, lim, ls="--", color=INK, lw=1.2)
    r_raw = np.corrcoef(mars, wnet)[0, 1]
    # de-seasonalised: the seasonal cycle alone would inflate the correlation
    ma = mars - np.array([mars[mon == m].mean() for m in mon])
    wa = wnet - np.array([wnet[mon == m].mean() for m in mon])
    r_des = np.corrcoef(ma, wa)[0, 1]
    ax.text(0.03, 0.93, f"r = {r_raw:.3f}  (540 months)\n"
                        f"r = {r_des:.3f}  after removing the seasonal cycle",
            transform=ax.transAxes, fontsize=FS_LEG, color=INK, va="top")
    style(ax, "(d)  Monthly AR area, W-Net v2 against MARS",
          "MARS AR area (fraction of grid)", "W-Net v2 AR area (fraction of grid)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=n)
                       for n, c in splits] +
                      [Line2D([], [], ls="--", color=INK, label="1:1 line")],
              fontsize=FS_LEG, frameon=False, loc="lower right", labelcolor=INK)

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)

    # the numbers behind the figure, for the run log
    print(f"\n{'split':<12}{'years':>6}{'IoU':>9}{'F1':>9}{'POD':>9}{'FAR':>9}{'BIAS':>9}")
    for name, _ in splits:
        m = agg([r for r in rows if split_of[r["year"]] == name])
        n = sum(1 for y in years if split_of[y] == name)
        print(f"{name:<12}{n:>6}{m['IoU']:>9.4f}{m['F1']:>9.4f}{m['POD']:>9.4f}"
              f"{m['FAR']:>9.4f}{m['BIAS']:>9.4f}")
    iou = np.array([ann[y]["IoU"] for y in years])
    print(f"\nannual IoU: min {iou.min():.4f} ({years[int(iou.argmin())]})  "
          f"max {iou.max():.4f} ({years[int(iou.argmax())]})  "
          f"spread {np.ptp(iou):.4f}  sd {iou.std(ddof=1):.4f}")
    print(f"monthly AR-area correlation: r={r_raw:.4f}, de-seasonalised r={r_des:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
