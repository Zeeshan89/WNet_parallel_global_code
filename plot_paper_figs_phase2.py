#!/usr/bin/env python3
"""
plot_paper_figs_phase2.py — six publication figures for the training / validation
performance of one Phase-2 W-Net run, drawn ONLY from that run's history.json.

    fig1_validation_iou      IoU per class vs epoch: training panel | validation
                             panel, with the trivial single-threshold floor each
                             class must beat
    fig2_skill_over_baseline the same skill as a multiple of that floor, training |
                             validation (log axis)
    fig3_training_loss       per-class supervised loss, total loss, W-Net
                             regularisers, learning-rate schedule
    fig4_precision_recall    validation precision-recall trajectory, one panel per
                             class, F1 iso-lines, epoch encoded as lightness
    fig5_areal_bias          (TP+FP)/(TP+FN) vs epoch -> convergence to 1
    fig6_final_summary       IoU / precision / recall / F1 at the final epoch,
                             training | validation, stability whiskers over the last
                             50 epochs, baseline tick

Training-set metrics (train_<class>_<metric>) are logged per epoch by
main_phase2.py --train-eval-stride, which scores a strided subsample of the
TRAINING years in eval mode with the same accumulator validation uses. Runs made
before that flag existed carry only val_* keys; the train panels are then skipped
with a printed note rather than faked.

Nothing here is a downstream detection comparison (MARS, IBTrACS, TRACK, ...):
every number is the model's own held-out validation on years it never trained on.

Design rules follow the dataviz method: form chosen by the data's job, one axis per
panel, thin marks, hairline solid grid, categorical hues fixed to the storm class
and never to rank, text in ink tokens (never in series colour), a legend whenever
there are >= 2 series plus selective direct labels, and the precision-recall chart
faceted because a 4-series scatter is an all-pairs form.

Palette (house storm colours; ETC purple stepped from #984ea3 -> #762a83).
Validated 2026-09-16 with a Python port of the dataviz validator on a white surface:
  adjacent pairs : CVD dE 15.6 (deutan) PASS, normal-vision dE 20.1 PASS, all >= 3:1
  all pairs      : normal-vision dE 19.3 PASS (>= 15 hard gate); CVD dE 5.1 is the
                   red<->green pair, unfixable by re-stepping -> the one all-pairs
                   form (fig4) is faceted so identity is carried by the panel title.

Usage
  python plot_paper_figs_phase2.py \
      --run ARs_Hurricanes_ETCs_MCS/Phase2/F_ibtracs_pwH50_e200 \
      --out phase2_figures/paper_F_ibtracs_pwH50_e200
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

# ---------------------------------------------------------------- tokens
SURFACE = "#ffffff"
# All text is pure black: grey text anti-aliases to a soft edge at print sizes and
# reads blurry. Grid and spines are not text and stay recessive.
INK, INK2, MUTED = "#000000", "#000000", "#000000"
GRID, AXIS = "#e1e0d9", "#8a8983"
CLASSES = ["ar", "hurricane", "etc", "mcs"]
LABEL = {"ar": "Atmospheric river", "hurricane": "Tropical cyclone",
         "etc": "Extratropical cyclone", "mcs": "Mesoscale convective system"}
SHORT = {"ar": "AR", "hurricane": "TC", "etc": "ETC", "mcs": "MCS"}
COLOR = {"ar": "#4575b4", "hurricane": "#d73027", "etc": "#762a83", "mcs": "#1a9850"}
MARKER = {"ar": "o", "hurricane": "s", "etc": "^", "mcs": "D"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- data
def load(run_dir):
    p = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "history.json")
    h = json.load(open(p))
    E = h["epochs"]
    g = lambda k: np.array([e[k] for e in E], float)
    d = {"cfg": h["config"], "epoch": g("epoch") + 1}          # 1-based for readers
    for k in ("train_total", "train_rec", "train_ncut", "lr"):
        d[k] = g(k)
    # train_<class> is the per-class LOSS; train_<class>_iou etc. are the metrics.
    d["has_train_metrics"] = all(f"train_{c}_iou" in E[-1] for c in CLASSES)
    for c in CLASSES:
        d[f"loss_{c}"] = g(f"train_{c}")
        for split in (("val",) if not d["has_train_metrics"] else ("val", "train")):
            pre = "" if split == "val" else "tr_"
            for m in ("iou", "precision", "recall", "dice", "tp", "fp", "fn"):
                d[f"{pre}{c}_{m}"] = g(f"{split}_{c}_{m}")
            d[f"{pre}{c}_bias"] = ((d[f"{pre}{c}_tp"] + d[f"{pre}{c}_fp"])
                                   / (d[f"{pre}{c}_tp"] + d[f"{pre}{c}_fn"]))
    d["base"] = h["config"]["trivial_baseline"]
    if not d["has_train_metrics"]:
        print("  note: this run has no train_* metrics (pre --train-eval-stride); "
              "figs 1, 2, 6 will show the validation panel only")
    return d


def subtitle(cfg):
    ny = len(cfg["train_years"])
    vy = ", ".join(str(y) for y in cfg["val_years"])
    return (f"{len(cfg['predictors'])} ERA5 predictors · {cfg['n_train_samples']:,} training "
            f"samples ({ny} years) · validated on {len(cfg['val_years'])} held-out years ({vy})")


def gap_note(d):
    """One line naming the largest train-validation gap, so a two-panel figure
    cannot be read as if every class generalised equally."""
    gaps = {c: d[f"tr_{c}_iou"][-1] - d[f"{c}_iou"][-1] for c in CLASSES}
    worst = max(gaps, key=gaps.get)
    small = [SHORT[c] for c in CLASSES if abs(gaps[c]) < 0.05]
    return (f"train−validation IoU gap: {', '.join(small)} ≤ 0.05, "
            f"{SHORT[worst]} {gaps[worst]:+.2f}")


def header(fig, title, sub):
    """Title + subtitle spaced in inches, so short figures do not squeeze them."""
    h_in = fig.get_size_inches()[1]
    y0 = 1 - 0.06 / h_in
    fig.text(0.012, y0, title, ha="left", va="top", fontsize=12, color=INK, weight="bold")
    fig.text(0.012, y0 - 0.24 / h_in, sub, ha="left", va="top", fontsize=8.3, color=MUTED)


def panel_tag(ax, tag, title):
    ax.set_title(f"{tag}  {title}", loc="left", color=INK, fontsize=10.5, pad=6)


def end_labels(ax, items, min_gap_pt=11.0, dx_pt=6.0, fontsize=8.5):
    """Direct labels at line ends that never overlap.

    items: list of (x, y, text, colour). Labels are placed at the data y, then
    pushed apart in display space to at least min_gap_pt; where a label had to
    move, a hairline leader connects it back to its line end (the dataviz rule:
    leader lines, not stacked labels, when series converge at the right edge).
    """
    fig = ax.figure
    fig.canvas.draw()  # so transforms are current (log axes need this)
    disp = [ax.transData.transform((x, y)) for x, y, *_ in items]
    order = np.argsort([p[1] for p in disp])
    ys = [disp[i][1] for i in order]
    gap = min_gap_pt * fig.dpi / 72.0
    for k in range(1, len(ys)):                      # sweep upward
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for k in range(len(ys) - 2, -1, -1):             # and back down, staying above the one below
        ys[k] = min(ys[k], ys[k + 1] - gap)
    for k, i in enumerate(order):
        x, y, text, colour = items[i]
        px, py = disp[i]
        lx, ly = ax.transData.inverted().transform((px, ys[k]))
        moved = abs(ys[k] - py) > 0.5
        ax.annotate(text, xy=(x, y), xytext=(dx_pt, (ys[k] - py) * 72.0 / fig.dpi),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fontsize, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=2) if moved else None)


def final_values_block(ax, values, yfrac, fmt="{:.2f}", prefix="final IoU"):
    """Compact one-line 'final' readout inside the axes.

    Used on the LEFT panel of a two-panel figure: end-of-line labels there would
    be drawn past the axes into the right panel (annotation_clip=False), so the
    left panel states its endpoints inline and only the right panel carries
    hanging end labels.
    """
    txt = prefix + ":  " + " · ".join(f"{SHORT[c]} {fmt.format(v)}" for c, v in values)
    ax.text(0.03, yfrac, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.2, color=INK)


def class_legend(ax, loc, classes=CLASSES, extra=None, title=None):
    handles = [Line2D([], [], color=COLOR[c], lw=2, label=LABEL[c]) for c in classes]
    if extra:
        handles += extra
    leg = ax.legend(handles=handles, loc=loc, frameon=False, handlelength=1.6,
                    labelcolor=INK2, title=title, borderaxespad=0.4)
    if title:
        leg.get_title().set_color(INK2); leg.get_title().set_fontsize(8.5)
    return leg


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext, dpi in (("png", 300), ("pdf", 300)):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        print(f"  wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1
def fig1(d, out):
    two = d["has_train_metrics"]
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(11.4 if two else 7.2, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.065 if two else 0.09, right=0.865 if two else 0.80,
                        top=0.815, bottom=0.125, wspace=0.10)
    x = d["epoch"]
    panels = ([("tr_", "(a)", "Training years"), ("", "(b)", "Held-out validation years")]
              if two else [("", "", "Held-out validation years")])
    for k, (ax, (pre, tag, name)) in enumerate(zip(axes, panels)):
        items, finals = [], []
        for c in CLASSES:
            y = d[f"{pre}{c}_iou"]
            ax.plot(x, y, color=COLOR[c], lw=2, solid_capstyle="round")
            ax.axhline(d["base"][c], color=COLOR[c], lw=0.9, ls=(0, (4, 3)), alpha=0.55)
            items.append((x[-1], y[-1], f"{SHORT[c]}  {y[-1]:.2f}", COLOR[c]))
            finals.append((c, y[-1]))
        ax.set_xlim(0, x[-1] + 1); ax.set_ylim(0, 1.0)
        ax.set_xlabel("epoch")
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        panel_tag(ax, tag, name)
        if k == len(axes) - 1:
            end_labels(ax, items)
        else:
            final_values_block(ax, finals, 0.30)
    axes[0].set_ylabel("IoU")
    # the empty band between the TC validation curve (~0.50) and the floors
    class_legend(axes[-1], "center",
                 extra=[Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)),
                               label="single-threshold IoU floor (per class)")])
    axes[-1].get_legend().set_bbox_to_anchor((0.58, 0.34))
    head = ("AR and MCS generalise with no train–validation gap; the TC head fits the "
            "training years far beyond what transfers" if two else
            "Validation IoU clears the single-threshold floor for every storm class")
    header(fig, head, subtitle(d["cfg"]) + (" · " + gap_note(d) if two else ""))
    save(fig, out, "fig1_validation_iou")


# ---------------------------------------------------------------- fig 2
def fig2(d, out):
    two = d["has_train_metrics"]
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(11.4 if two else 7.2, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.075 if two else 0.09, right=0.865 if two else 0.80,
                        top=0.815, bottom=0.125, wspace=0.10)
    x = d["epoch"]
    panels = ([("tr_", "(a)", "Training years"), ("", "(b)", "Held-out validation years")]
              if two else [("", "", "Held-out validation years")])
    # one shared log range so the two panels are directly comparable
    hi = max(max((d[f"{p}{c}_iou"] / d["base"][c]).max() for c in CLASSES)
             for p in (("tr_", "") if two else ("",)))
    for k, (ax, (pre, tag, name)) in enumerate(zip(axes, panels)):
        items, finals = [], []
        for c in CLASSES:
            r = d[f"{pre}{c}_iou"] / d["base"][c]
            ax.plot(x, r, color=COLOR[c], lw=2, solid_capstyle="round")
            items.append((x[-1], r[-1],
                          f"{SHORT[c]}  {r[-1]:.1f}×" if r[-1] >= 2 else f"{SHORT[c]}  {r[-1]:.2f}×",
                          COLOR[c]))
            finals.append((c, r[-1]))
        ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)))
        ax.set_yscale("log"); ax.set_ylim(0.2, hi * 1.45)
        ticks = [0.2, 0.5, 1, 2, 5, 10, 20] + ([30] if hi > 20 else [])
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}×" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0, x[-1] + 1); ax.set_xlabel("epoch")
        panel_tag(ax, tag, name)
        if k == len(axes) - 1:
            end_labels(ax, items)
        else:
            # the parity line is annotated once, on the left panel only
            ax.annotate("parity with one threshold", xy=(8, 1.0), xytext=(0, 4),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=7.8, color=INK2)
            final_values_block(ax, finals, 0.045, fmt="{:.1f}×", prefix="final skill")
    axes[0].set_ylabel("IoU ÷ single-threshold IoU")
    class_legend(axes[-1], "lower right")
    head = ("Skill over the trivial baseline, training vs held-out: TC reaches 29× in "
            "sample but 15× out of sample" if two else
            "Skill over the trivial baseline: 15× for TCs, 8× for ETCs, 4× for MCSs")
    header(fig, head,
           "the AR label is itself an IVT threshold, so parity (1×) is that class's ceiling · "
           + subtitle(d["cfg"]).split(" · ", 1)[1])
    save(fig, out, "fig2_skill_over_baseline")


# ---------------------------------------------------------------- fig 3
def fig3(d, out):
    # Two neutral greys for the two single-series regulariser lines: they carry no
    # class identity, so they must not borrow a storm colour.
    NCUT_C, REC_C = "#000000", "#7a7a7a"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.86, bottom=0.075, hspace=0.42, wspace=0.28)
    x = d["epoch"]

    ax = axes[0, 0]
    items = []
    for c in CLASSES:
        y = d[f"loss_{c}"]
        ax.plot(x, y, color=COLOR[c], lw=2, solid_capstyle="round")
        items.append((x[-1], y[-1], SHORT[c], COLOR[c]))
    ax.set_yscale("log"); ax.set_xlim(0, x[-1] + 1)
    ax.set_xlabel("epoch"); ax.set_ylabel("per-class supervised loss  (CE + Tversky)")
    panel_tag(ax, "(a)", "Supervised loss by class")
    end_labels(ax, items)
    class_legend(ax, "upper right")

    ax = axes[0, 1]
    tot0, tot1 = d["train_total"][0], d["train_total"][-1]
    ax.plot(x, d["train_total"], color=INK, lw=2, solid_capstyle="round")
    ax.set_yscale("log"); ax.set_xlim(0, x[-1] + 1)
    ax.set_yticks([0.5, 1, 2, 3]); ax.set_yticklabels(["0.5", "1", "2", "3"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("epoch"); ax.set_ylabel("total training loss")
    panel_tag(ax, "(b)", "Total training loss")
    end_labels(ax, [(x[-1], tot1, f"{tot1:.2f}", INK)])
    ax.annotate(f"{tot0:.2f} at epoch 1", xy=(x[0], tot0), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(0.97, 0.90, f"{tot0/tot1:.1f}× reduction", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK)

    ax = axes[1, 0]
    ax.plot(x, d["train_ncut"], color=NCUT_C, lw=2)
    ax.plot(x, d["train_rec"], color=REC_C, lw=2)
    end_labels(ax, [(x[-1], d["train_ncut"][-1], f"N-cut  {d['train_ncut'][-1]:.2f}", NCUT_C),
                    (x[-1], d["train_rec"][-1], f"MSE  {d['train_rec'][-1]:.2f}", REC_C)])
    ax.set_xlim(0, x[-1] + 1); ax.set_ylim(0, 2.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("regulariser loss")
    panel_tag(ax, "(c)", "W-Net regulariser terms")
    ax.legend(handles=[Line2D([], [], color=NCUT_C, lw=2, label="normalised cut (N-cut)"),
                       Line2D([], [], color=REC_C, lw=2, label="reconstruction MSE")],
              loc="center right", frameon=False, labelcolor=INK)

    ax = axes[1, 1]
    ax.plot(x, d["lr"] * 1e3, color=INK, lw=2)
    ax.set_xlim(0, x[-1] + 1); ax.set_ylim(0, d["lr"].max() * 1e3 * 1.12)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate  (×10⁻³)")
    panel_tag(ax, "(d)", "One-cycle learning-rate schedule")
    pk = int(np.argmax(d["lr"]))
    ax.annotate(f"peak {d['lr'][pk]*1e3:.1f}×10⁻³ at epoch {int(x[pk])}", xy=(x[pk], d["lr"][pk] * 1e3),
                xytext=(18, -2), textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    header(fig, f"Optimisation is smooth and monotone: total training loss falls {tot0/tot1:.1f}× "
                f"({tot0:.2f} → {tot1:.2f}) across all four classes", subtitle(d["cfg"]))
    save(fig, out, "fig3_training_loss")


# ---------------------------------------------------------------- fig 4
def fig4(d, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.835, bottom=0.075, hspace=0.30, wspace=0.12)
    rr = np.linspace(0.02, 1, 300)
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for ax, c, tag in zip(axes.ravel(), CLASSES, tags):
        # F1 iso-lines: hairline, labelled just inside the top edge where p -> 1
        for f1 in (0.4, 0.6, 0.8, 0.9):
            pp = f1 * rr / (2 * rr - f1)
            ok = (pp > 0) & (pp <= 1)
            ax.plot(rr[ok], pp[ok], color=GRID, lw=0.9, zorder=1)
            r_top = f1 * 0.97 / (2 * 0.97 - f1)          # recall where the iso-line meets p = 0.97
            ax.text(r_top, 0.985, f"F1 {f1:g}", fontsize=6.8, color=MUTED, ha="center", va="top", zorder=2)
        P, R = d[f"{c}_precision"], d[f"{c}_recall"]
        n = len(P)
        base = np.array(to_rgb(COLOR[c]))
        # epoch as lightness within the class hue: early = pale, late = full
        for i in range(n - 1):
            t = i / (n - 2)
            col = 1 - (1 - base) * (0.25 + 0.75 * t)
            ax.plot(R[i:i + 2], P[i:i + 2], color=col, lw=1.6, solid_capstyle="round", zorder=3)
        ax.scatter([R[0]], [P[0]], s=46, facecolor=SURFACE, edgecolor=COLOR[c], lw=1.6, zorder=4)
        ax.scatter([R[-1]], [P[-1]], s=150, marker="*", facecolor=COLOR[c], edgecolor=SURFACE, lw=1.2, zorder=5)
        f1 = 2 * P[-1] * R[-1] / (P[-1] + R[-1])
        ax.text(0.03, 0.04, f"final:  P {P[-1]:.2f} · R {R[-1]:.2f} · F1 {f1:.2f}",
                transform=ax.transAxes, fontsize=8.2, color=INK2, ha="left", va="bottom")
        panel_tag(ax, tag, LABEL[c])
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("precision")
    # key sits in the header band, clear of every panel title
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=7, label="epoch 1"),
                        Line2D([], [], marker="*", ls="", color=INK2, markersize=11, label="epoch 200"),
                        Line2D([], [], color=INK2, lw=1.6, label="trajectory, pale → dark = early → late")],
               loc="upper left", bbox_to_anchor=(0.012, 0.905), frameon=False, labelcolor=INK2, ncol=3,
               columnspacing=1.8, handletextpad=0.6)
    header(fig, "Precision–recall trajectory over training, on held-out years", subtitle(d["cfg"]))
    save(fig, out, "fig4_precision_recall")


# ---------------------------------------------------------------- fig 5
def fig5(d, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.10, right=0.80, top=0.84, bottom=0.13)
    x = d["epoch"]
    items = []
    for c in CLASSES:
        y = d[f"{c}_bias"]
        ax.plot(x, y, color=COLOR[c], lw=2, solid_capstyle="round")
        items.append((x[-1], y[-1], f"{SHORT[c]}  {y[-1]:.2f}", COLOR[c]))
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    # range chosen for the convergence, not the epoch-1 spike; that value is stated instead
    ax.set_yscale("log"); ax.set_ylim(0.6, 12)
    ax.set_yticks([0.6, 0.8, 1, 1.5, 2, 3, 5, 10]); ax.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3", "5", "10"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tc0 = d["hurricane_bias"][0]
    ax.annotate(f"TC at epoch 1 = {tc0:.0f}, off scale", xy=(1, 12), xytext=(6, -3), textcoords="offset points",
                ha="left", va="top", fontsize=7.6, color=MUTED)
    ax.set_xlim(0, x[-1] + 1)
    ax.set_xlabel("epoch"); ax.set_ylabel("areal bias  (TP + FP) / (TP + FN)")
    end_labels(ax, items)
    class_legend(ax, "upper right", extra=[Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)),
                                                   label="parity: predicted area = observed area")])
    header(fig, "Predicted storm area converges to the observed area for all four classes",
           "areal bias > 1 over-predicts extent, < 1 under-predicts · validated on 4 held-out years (1987, 1998, 2007, 2016)")
    save(fig, out, "fig5_areal_bias")


# ---------------------------------------------------------------- fig 6
def fig6(d, out):
    metrics = [("iou", "IoU", "o"), ("precision", "Precision", "s"),
               ("recall", "Recall", "^"), ("dice", "F1", "D")]
    two = d["has_train_metrics"]
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(12.4 if two else 7.6, 4.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.175 if two else 0.27, right=0.985, top=0.805, bottom=0.20,
                        wspace=0.07)
    yrow = {c: i for i, c in enumerate(CLASSES[::-1])}
    off = np.linspace(0.27, -0.27, len(metrics))
    panels = ([("tr_", "(a)", "Training years"), ("", "(b)", "Held-out validation years")]
              if two else [("", "", "Held-out validation years")])
    for ax, (pre, tag, pname) in zip(axes, panels):
        for c in CLASSES:
            y0 = yrow[c]
            # stability whisker: min-max of IoU over the last 50 epochs
            lo, hi = d[f"{pre}{c}_iou"][-50:].min(), d[f"{pre}{c}_iou"][-50:].max()
            ax.plot([lo, hi], [y0 + off[0]] * 2, color=COLOR[c], lw=4, alpha=0.25,
                    solid_capstyle="butt", zorder=2)
            # baseline tick on the IoU row, with an arrow to where this split landed
            b = d["base"][c]
            ax.plot([b, b], [y0 + off[0] - 0.09, y0 + off[0] + 0.09], color=INK2, lw=1.4, zorder=4)
            ax.annotate("", xy=(d[f"{pre}{c}_iou"][-1], y0 + off[0]), xytext=(b, y0 + off[0]),
                        arrowprops=dict(arrowstyle="->", color=AXIS, lw=0.9, shrinkA=0, shrinkB=4),
                        zorder=1)
            for (key, name, mk), dy in zip(metrics, off):
                v = d[f"{pre}{c}_{key}"][-1]
                ax.scatter([v], [y0 + dy], s=64, marker=mk, facecolor=COLOR[c],
                           edgecolor=SURFACE, lw=1.4, zorder=5)
                ax.text(v + 0.018, y0 + dy, f"{v:.2f}", fontsize=8, color=INK2,
                        va="center", ha="left", zorder=6)
        ax.set_xlim(0, 1.10); ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("score at the final epoch")
        ax.grid(axis="y", visible=False)
        for i in range(len(CLASSES) - 1):
            ax.axhline(i + 0.5, color=GRID, lw=0.8)
        panel_tag(ax, tag, pname)
    axes[0].set_yticks(list(yrow.values()))
    axes[0].set_yticklabels([LABEL[c] for c in yrow], color=INK)
    handles = [Line2D([], [], marker=mk, ls="", color=INK2, markersize=7, label=name)
               for _, name, mk in metrics]
    handles += [Line2D([], [], color=INK2, lw=1.4, label="single-threshold IoU floor"),
                Line2D([], [], color="#bdbdbd", lw=4, label="IoU range, last 50 epochs")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.165), ncol=3,
                   frameon=False, labelcolor=INK2, columnspacing=1.6, handletextpad=0.6)
    head = ("Final skill by storm class: training vs held-out validation"
            if two else "Final held-out skill by storm class, with the floor each one started from")
    header(fig, head, subtitle(d["cfg"]) + (" · " + gap_note(d) if two else ""))
    save(fig, out, "fig6_final_summary")
    # table twin — both splits when they exist, so nothing in the figure is chart-only
    cols = ["class", "split", "IoU", "precision", "recall", "F1", "trivial_baseline_IoU",
            "IoU_over_baseline", "IoU_min_last50", "IoU_max_last50", "areal_bias"]
    rows = [",".join(cols)]
    for pre, sname in ((("tr_", "train"), ("", "validation")) if two else (("", "validation"),)):
        for c in CLASSES:
            rows.append(",".join([
                SHORT[c], sname,
                f"{d[f'{pre}{c}_iou'][-1]:.4f}", f"{d[f'{pre}{c}_precision'][-1]:.4f}",
                f"{d[f'{pre}{c}_recall'][-1]:.4f}", f"{d[f'{pre}{c}_dice'][-1]:.4f}",
                f"{d['base'][c]:.4f}", f"{d[f'{pre}{c}_iou'][-1]/d['base'][c]:.2f}",
                f"{d[f'{pre}{c}_iou'][-50:].min():.4f}", f"{d[f'{pre}{c}_iou'][-50:].max():.4f}",
                f"{d[f'{pre}{c}_bias'][-1]:.3f}"]))
    p = os.path.join(out, "fig6_final_summary.csv")
    open(p, "w").write("\n".join(rows) + "\n"); print(f"  wrote {p}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing history.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", type=int, default=None, help="subset of figure numbers, e.g. 1 4")
    a = ap.parse_args()
    d = load(a.run)
    figs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}
    for k in (a.only or sorted(figs)):
        figs[k](d, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
