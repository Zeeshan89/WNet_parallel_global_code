#!/usr/bin/env python3
"""
plot_paper_selected.py — the four submission figures for the Phase-2 W-Net
training/validation results, in their final paper layout.

    Figure1_iou_train_vs_val      (a) Training | (b) Validation IoU vs epoch
    Figure2_final_skill_summary   (a) Training | (b) Validation, final-epoch scores
    Figure3_optimisation          (a) Supervised loss | (b) Total training loss |
                                  (c) Reconstruction loss | (d) One-cycle LR schedule
    Figure4_areal_bias            (a) Training | (b) Validation areal bias vs epoch

Layout rules for this set (paper conventions, set by the authors):
  * no figure title or subtitle; the caption carries that information
  * one legend per figure, on top, in a single row
  * no annotation text inside the plotting area of the curve figures; values and
    the meaning of dashed reference lines go in the caption
  * Figure 2 keeps its per-marker value labels: there the numbers are the content

Data loading, colours and font setup are imported from plot_paper_figs_phase2.py
so this set cannot drift from the analysis set. Output names match the files in
phase2_figures/paper_selected_e300/, so a re-run replaces them in place.

Usage
  python plot_paper_selected.py \
      --run ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e300 \
      --out phase2_figures/paper_selected_e300
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_paper_figs_phase2 as P  # noqa: E402  (applies the shared rcParams)

INK, GRID, AXIS, SURFACE = P.INK, P.GRID, P.AXIS, P.SURFACE
CLASSES, COLOR, LABEL = P.CLASSES, P.COLOR, P.LABEL
SPLITS = [("tr_", "(a) Training"), ("", "(b) Validation")]

# The submission set runs one size up from the analysis set: these figures are
# placed at column width in the paper, where the analysis defaults read small.
# Scoped to this script; plot_paper_figs_phase2.py keeps its own sizes.
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                     "legend.fontsize": 11})
FS_TITLE, FS_LEGEND, FS_VALUE = 12.5, 11, 9.5


# ---------------------------------------------------------------- helpers
def class_handles():
    return [Line2D([], [], color=COLOR[c], lw=2.2, label=LABEL[c]) for c in CLASSES]


def top_legend(fig, handles, ncol, y=1.0, fontsize=FS_LEGEND):
    """Single-row legend centred above every panel title."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, labelcolor=INK, fontsize=fontsize, handlelength=1.9,
               columnspacing=2.2, handletextpad=0.6, borderaxespad=0.0)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=FS_TITLE, pad=7)


def need_train(d):
    if not d["has_train_metrics"]:
        raise SystemExit("this run has no train_* metrics; re-train with "
                         "main_phase2.py --train-eval-stride (default 4)")


# ---------------------------------------------------------------- Figure 1
def figure1(d, out):
    need_train(d)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.12, wspace=0.07)
    x = d["epoch"]
    for ax, (pre, name) in zip(axes, SPLITS):
        for c in CLASSES:
            ax.plot(x, d[f"{pre}{c}_iou"], color=COLOR[c], lw=2, solid_capstyle="round")
            # single-threshold floor for this class; explained in the caption
            ax.axhline(d["base"][c], color=COLOR[c], lw=0.9, ls=(0, (4, 3)), alpha=0.6)
        ax.set_xlim(0, x[-1] + 1); ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("epoch")
        title(ax, name)
    axes[0].set_ylabel("IoU")
    top_legend(fig, class_handles(), ncol=4)
    P.save(fig, out, "Figure1_iou_train_vs_val")


# ---------------------------------------------------------------- Figure 2
def figure2(d, out):
    need_train(d)
    metrics = [("iou", "IoU", "o"), ("precision", "Precision", "s"),
               ("recall", "Recall", "^"), ("dice", "F1", "D")]
    # a little taller than before so the larger per-marker values keep their spacing
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.83, bottom=0.12, wspace=0.07)
    yrow = {c: i for i, c in enumerate(CLASSES[::-1])}
    off = np.linspace(0.27, -0.27, len(metrics))
    for ax, (pre, name) in zip(axes, SPLITS):
        for c in CLASSES:
            y0 = yrow[c]
            # No last-50-epoch range bar: the widest range across both splits is
            # 0.011 IoU, narrower than a marker, so the bar hid under the dot and
            # its legend entry pointed at nothing. Stability is stated in the caption.
            b = d["base"][c]
            ax.plot([b, b], [y0 + off[0] - 0.09, y0 + off[0] + 0.09], color=INK, lw=1.4, zorder=4)
            ax.annotate("", xy=(d[f"{pre}{c}_iou"][-1], y0 + off[0]), xytext=(b, y0 + off[0]),
                        arrowprops=dict(arrowstyle="->", color=AXIS, lw=0.9, shrinkA=0, shrinkB=4),
                        zorder=1)
            for (key, _, mk), dy in zip(metrics, off):
                v = d[f"{pre}{c}_{key}"][-1]
                ax.scatter([v], [y0 + dy], s=64, marker=mk, facecolor=COLOR[c],
                           edgecolor=SURFACE, lw=1.4, zorder=5)
                ax.text(v + 0.018, y0 + dy, f"{v:.2f}", fontsize=FS_VALUE, color=INK,
                        va="center", ha="left", zorder=6)
        ax.set_xlim(0, 1.10); ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("score at the final epoch")
        ax.grid(axis="y", visible=False)
        for i in range(len(CLASSES) - 1):
            ax.axhline(i + 0.5, color=GRID, lw=0.8)
        title(ax, name)
    axes[0].set_yticks(list(yrow.values()))
    axes[0].set_yticklabels([LABEL[c] for c in yrow], color=INK)
    handles = [Line2D([], [], marker=mk, ls="", color=INK, markersize=7, label=nm)
               for _, nm, mk in metrics]
    # the floor is drawn as a vertical tick, so its legend symbol is the same tick
    handles += [Line2D([], [], color=INK, marker="|", ls="", markersize=13,
                       markeredgewidth=1.6, label="single-threshold IoU floor")]
    top_legend(fig, handles, ncol=len(handles))
    P.save(fig, out, "Figure2_final_skill_summary")


# ---------------------------------------------------------------- Figure 3
def figure3(d, out):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.15, wspace=0.36)
    x = d["epoch"]

    ax = axes[0]
    for c in CLASSES:
        ax.plot(x, d[f"loss_{c}"], color=COLOR[c], lw=2, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylabel("loss (CE + Tversky)")
    title(ax, "(a) Supervised loss")

    ax = axes[1]
    ax.plot(x, d["train_total"], color=INK, lw=2, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3]); ax.set_yticklabels(["0.5", "1", "2", "3"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("total loss")
    title(ax, "(b) Total training loss")

    ax = axes[2]
    ax.plot(x, d["train_rec"], color=INK, lw=2, solid_capstyle="round")
    ax.set_ylim(0, d["train_rec"].max() * 1.12)
    ax.set_ylabel("MSE")
    title(ax, "(c) Reconstruction loss")

    ax = axes[3]
    ax.plot(x, d["lr"] * 1e3, color=INK, lw=2, solid_capstyle="round")
    ax.set_ylim(0, d["lr"].max() * 1e3 * 1.12)
    ax.set_ylabel("learning rate  (×10⁻³)")
    title(ax, "(d) One-cycle learning rate schedule")

    for ax in axes:
        ax.set_xlim(0, x[-1] + 1)
        ax.set_xlabel("epoch")
    # only (a) carries class colour; the legend still sits on top for the set
    top_legend(fig, class_handles(), ncol=4)
    P.save(fig, out, "Figure3_optimisation")


# ---------------------------------------------------------------- Figure 4
def figure4(d, out):
    need_train(d)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.12, wspace=0.07)
    x = d["epoch"]
    for ax, (pre, name) in zip(axes, SPLITS):
        for c in CLASSES:
            ax.plot(x, d[f"{pre}{c}_bias"], color=COLOR[c], lw=2, solid_capstyle="round")
        # parity (predicted area = observed area); explained in the caption
        ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (4, 3)))
        # range chosen for the convergence; the epoch-1 spikes (up to ~46) clip at the top
        ax.set_yscale("log"); ax.set_ylim(0.6, 12)
        ticks = [0.6, 0.8, 1, 1.5, 2, 3, 5, 10]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0, x[-1] + 1)
        ax.set_xlabel("epoch")
        title(ax, name)
    axes[0].set_ylabel("areal bias  (TP + FP) / (TP + FN)")
    top_legend(fig, class_handles(), ncol=4)
    P.save(fig, out, "Figure4_areal_bias")


# ---------------------------------------------------------------- Table 1
TABLE_CAPTION = (
    "Detection scores of the W-Net at the final training epoch for atmospheric rivers "
    "(AR), tropical cyclones (TC), extratropical cyclones (ETC) and mesoscale convective "
    "systems (MCS), computed on the training years and on the held-out validation years "
    "(1987, 1998, 2007 and 2016). IoU is the intersection over union, and F1 is the "
    "harmonic mean of precision and recall. The areal bias is (TP + FP)/(TP + FN), where "
    "a value of 1 indicates that the predicted area equals the reference area. The IoU "
    "floor is the highest IoU that can be obtained by applying a single threshold to one "
    "of the six input fields, and IoU/floor is the ratio of the W-Net IoU to that floor.")
TABLE_COLS = ["IoU", "Precision", "Recall", "F1", "Areal bias", "IoU floor", "IoU/floor"]


def table_rows(d, pre):
    """Final-epoch values for one split, rounded from the raw history, not the CSV."""
    rows = []
    for c in CLASSES:
        iou, b = d[f"{pre}{c}_iou"][-1], d["base"][c]
        rows.append([P.SHORT[c], f"{iou:.3f}", f"{d[f'{pre}{c}_precision'][-1]:.3f}",
                     f"{d[f'{pre}{c}_recall'][-1]:.3f}", f"{d[f'{pre}{c}_dice'][-1]:.3f}",
                     # floor at its stored 4 decimals so IoU/floor checks by hand (TC 0.033 would not)
                     f"{d[f'{pre}{c}_bias'][-1]:.2f}", f"{b:.4f}", f"{iou / b:.2f}"])
    return rows


def table1(d, out):
    need_train(d)
    blocks = [("Training years", table_rows(d, "tr_")),
              ("Validation years", table_rows(d, ""))]
    ncol = len(TABLE_COLS) + 1

    # LaTeX (needs \usepackage{booktabs}); caption above, as tables usually take it
    tex = ["% Table 1, generated by plot_paper_selected.py; requires \\usepackage{booktabs}",
           "\\begin{table}[t]", "\\centering",
           "\\caption{" + TABLE_CAPTION.replace("%", "\\%") + "}",
           "\\label{tab:final_skill}",
           "\\begin{tabular}{l" + "c" * len(TABLE_COLS) + "}", "\\toprule",
           "Storm type & " + " & ".join(TABLE_COLS) + " \\\\", "\\midrule"]
    for i, (name, rows) in enumerate(blocks):
        if i:
            tex.append("\\midrule")
        tex.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{name}}}}} \\\\")
        tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    p = os.path.join(out, "Table1_final_skill.tex")
    open(p, "w").write("\n".join(tex) + "\n"); print(f"  wrote {p}")

    # Markdown / plain version for pasting into a word processor
    md = ["**Table 1.** " + TABLE_CAPTION, "",
          "| Storm type | " + " | ".join(TABLE_COLS) + " |",
          "|---|" + "---:|" * len(TABLE_COLS)]
    for name, rows in blocks:
        md.append(f"| *{name}* |" + " |" * len(TABLE_COLS))
        md += ["| " + " | ".join(r) + " |" for r in rows]
    p = os.path.join(out, "Table1_final_skill.md")
    open(p, "w").write("\n".join(md) + "\n"); print(f"  wrote {p}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing history.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", type=int, default=None)
    a = ap.parse_args()
    d = P.load(a.run)
    figs = {1: figure1, 2: figure2, 3: figure3, 4: figure4}
    for k in (a.only or sorted(figs)):
        figs[k](d, a.out)
    table1(d, a.out)   # cheap and deterministic, so it always tracks the figures
    return 0


if __name__ == "__main__":
    sys.exit(main())
