#!/usr/bin/env python3
"""
plot_phase2_history.py — convergence figures from Phase 2 history.json.

Two modes:

  single run   Four panels: per-class validation IoU against each class's trivial
               single-threshold baseline; the same skill expressed as a ratio to
               that baseline (so classes with base rates an order of magnitude
               apart become comparable); per-class training loss; and the two
               regularisers.

  comparison   One panel per class, one line per run — for the loss-configuration
               sweep, where the whole point is that a setting good for one class
               is bad for another.

Design notes (these are deliberate, please keep them if you edit):
  * The dashed horizontal line in each IoU panel is the trivial baseline — the
    best IoU obtainable by thresholding ONE raw predictor at a single global
    value. Without it a reader cannot tell 0.23 (7x baseline, real learning) from
    0.92 (below baseline, worse than one threshold). It is the single most
    important thing on the figure.
  * No dual axes anywhere. Quantities on different scales get their own panel.
  * Every series is direct-labelled at its right end as well as legended, because
    two of the four palette slots fall below 3:1 contrast on a light surface and
    must not rely on colour alone.
  * Series colours are fixed per class and never reassigned by rank, so a class
    keeps its hue across every figure in the paper.

Usage
  python3 plot_phase2_history.py --run ARs_Hurricanes_ETCs_MCS/Phase2
  python3 plot_phase2_history.py --compare A=.../Phase2_sweep/A B=.../Phase2_sweep/B
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots 1-4 (light mode), fixed per class.
CLASS_COLOR = {
    "ar":        "#2a78d6",   # blue
    "hurricane": "#eb6834",   # orange
    "etc":       "#1baf7a",   # aqua
    "mcs":       "#eda100",   # yellow
}
CLASS_LABEL = {"ar": "AR", "hurricane": "Hurricane", "etc": "ETC", "mcs": "MCS"}
CLASSES = ["ar", "hurricane", "etc", "mcs"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"

TRIVIAL_BASELINE = {"ar": 0.9371, "hurricane": 0.0333,
                    "etc": 0.0898, "mcs": 0.2124}


def style_axes(ax):
    """Recessive grid and axes — the data should be the darkest thing present."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e5e4df", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=1.0)
    # Do NOT re-set the title here. matplotlib keeps left/center/right titles as
    # separate artists, so `set_title(get_title(), loc="left")` adds a second one
    # instead of restyling the first, and both get drawn.


def load_history(path):
    """Accept both the new {config, epochs} form and the old bare list."""
    p = path if path.endswith(".json") else os.path.join(path, "history.json")
    with open(p) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("epochs", []), raw.get("config", {})
    return raw, {}


def series(hist, key):
    ep = [r["epoch"] + 1 for r in hist if r.get(key) is not None]
    v = [r[key] for r in hist if r.get(key) is not None]
    return np.array(ep, dtype=float), np.array(v, dtype=float)


def end_label(ax, x, y, text, color):
    """Direct label at the right end of a line.

    Text uses ink, not the series colour — the coloured line running into it is
    what carries identity. Required here because aqua and yellow fall below 3:1
    against the light surface.
    """
    if len(x) == 0:
        return
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=INK, zorder=5)


def plot_single(hist, config, out, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.08,
                        hspace=0.34, wspace=0.22)

    baseline = config.get("trivial_baseline", TRIVIAL_BASELINE)

    # (a) validation IoU vs the trivial baseline ─────────────────────────────
    ax = axes[0, 0]
    for c in CLASSES:
        x, y = series(hist, f"val_{c}_iou")
        if len(x) == 0:
            continue
        ax.plot(x, y, color=CLASS_COLOR[c], linewidth=2.0,
                marker="o", markersize=4, label=CLASS_LABEL[c], zorder=3)
        b = baseline.get(c)
        if b is not None:
            ax.axhline(b, color=CLASS_COLOR[c], linewidth=1.2, linestyle=(0, (4, 3)),
                       alpha=0.55, zorder=2)
        end_label(ax, x, y, CLASS_LABEL[c], CLASS_COLOR[c])
    ax.set_title("(a)  Validation IoU  ·  dashed = single-threshold baseline", color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("IoU", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    # Not "upper left": AR and its dashed baseline both live at the top of this
    # panel, and the legend lands on them.
    leg = ax.legend(frameon=False, fontsize=8.5, loc="center left", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK)

    # (b) skill relative to baseline ─────────────────────────────────────────
    ax = axes[0, 1]
    for c in CLASSES:
        x, y = series(hist, f"val_{c}_iou")
        b = baseline.get(c)
        if len(x) == 0 or not b:
            continue
        r = y / b
        ax.plot(x, r, color=CLASS_COLOR[c], linewidth=2.0,
                marker="o", markersize=4, label=CLASS_LABEL[c], zorder=3)
        end_label(ax, x, r, CLASS_LABEL[c], CLASS_COLOR[c])
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.4, linestyle="-", zorder=2)
    ax.annotate("parity with one threshold", xy=(0.02, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_title("(b)  Skill relative to the trivial baseline", color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("IoU / baseline IoU", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    style_axes(ax)

    # (c) per-class training loss ────────────────────────────────────────────
    ax = axes[1, 0]
    for c in CLASSES:
        x, y = series(hist, f"train_{c}")
        if len(x) == 0:
            continue
        ax.plot(x, y, color=CLASS_COLOR[c], linewidth=2.0,
                marker="o", markersize=4, label=CLASS_LABEL[c], zorder=3)
        end_label(ax, x, y, CLASS_LABEL[c], CLASS_COLOR[c])
    ax.set_title("(c)  Training loss by class", color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("loss", color=INK_2, fontsize=9)
    style_axes(ax)

    # (d) regularisers ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    for key, colour, lab in (("train_rec", "#4a3aa7", "reconstruction (MSE)"),
                             ("train_ncut", "#e34948", "N-cut")):
        x, y = series(hist, key)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=colour, linewidth=2.0, marker="o", markersize=4,
                label=lab, zorder=3)
        end_label(ax, x, y, lab.split()[0], colour)
    ax.set_title("(d)  Regulariser terms", color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel("loss", color=INK_2, fontsize=9)
    style_axes(ax)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle(title, color=INK, fontsize=13, x=0.07, ha="left", y=0.97)
    sub = (f"{config.get('n_train_months','?')} training months · "
           f"{config.get('n_train_samples','?')} samples · "
           f"val years {config.get('val_years','?')} · "
           f"discretize={config.get('discretize','?')} · "
           f"ncut_w={config.get('ncut_weight','?')}")
    fig.text(0.07, 0.935, sub, color=INK_2, fontsize=8.5, ha="left")

    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def plot_compare(runs, out, title):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.16, wspace=0.26)

    # One hue per RUN here (the run is the identity being compared), fixed order.
    run_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

    for j, c in enumerate(CLASSES):
        ax = axes[j]
        for i, (name, (hist, cfg)) in enumerate(runs.items()):
            x, y = series(hist, f"val_{c}_iou")
            if len(x) == 0:
                continue
            ax.plot(x, y, color=run_colors[i % len(run_colors)], linewidth=2.0,
                    marker="o", markersize=4, label=name, zorder=3)
        b = TRIVIAL_BASELINE[c]
        ax.axhline(b, color=INK_MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"baseline {b:.3f}", xy=(0.03, b),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.set_title(CLASS_LABEL[c], color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
        if j == 0:
            ax.set_ylabel("validation IoU", color=INK_2, fontsize=9)
        ax.set_ylim(bottom=0)
        style_axes(ax)

    leg = axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle(title, color=INK, fontsize=13, x=0.05, ha="left", y=0.96)
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a run directory (or history.json path)")
    ap.add_argument("--compare", nargs="+",
                    help="NAME=path entries for a multi-run comparison")
    ap.add_argument("--out-dir", default="phase2_figures")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.run:
        hist, cfg = load_history(args.run)
        if not hist:
            print(f"no epochs in {args.run}")
            return 1
        tag = cfg.get("tag") or os.path.basename(os.path.normpath(args.run))
        plot_single(hist, cfg,
                    os.path.join(args.out_dir, f"convergence_{tag or 'run'}.png"),
                    args.title or f"Phase 2 convergence — {tag}")

    if args.compare:
        runs = {}
        for entry in args.compare:
            name, _, path = entry.partition("=")
            if not path:
                name, path = os.path.basename(os.path.normpath(entry)), entry
            h, c = load_history(path)
            if h:
                runs[name] = (h, c)
        if not runs:
            print("no runs loaded for comparison")
            return 1
        plot_compare(runs, os.path.join(args.out_dir, "sweep_comparison.png"),
                     args.title or "Phase 2 loss-configuration sweep")

    if not args.run and not args.compare:
        ap.error("give --run and/or --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
