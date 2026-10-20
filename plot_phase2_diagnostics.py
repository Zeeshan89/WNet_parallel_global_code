#!/usr/bin/env python3
"""
plot_phase2_diagnostics.py — the metrics history.json carries that the convergence
figure does not use.

plot_phase2_history.py shows IoU, skill-vs-baseline and the loss terms. That leaves
the per-class confusion counts (tp/fp/fn), precision, recall, dice and the timing —
which is where the interesting behaviour is, because IoU alone cannot say WHY a class
is limited. A class stuck at IoU 0.5 because it under-detects needs a different fix
from one stuck because it over-detects.

Figure 1 — diagnostics for one run
  (a) validation precision per class
  (b) validation recall per class
  (c) precision-recall trajectory, start -> end, so the trade is visible
  (d) AREAL BIAS (tp+fp)/(tp+fn): >1 over-predicts area, <1 under-predicts.
      This is the panel that says whether a class is limited by misses or by
      false alarms, which IoU deliberately conflates.
  (e) false positives and false negatives, normalised by the positives present
  (f) generalisation: per-class training loss against validation IoU

Figure 2 — cross-run comparison
  AR / ETC / MCS share identical labels across runs, so those are directly
  comparable. TC is NOT: C_full* were trained on the CyTRACK mask and E_ibtracs* on
  IBTrACS observations, so their TC columns are measured against different targets
  and are marked accordingly rather than plotted as if comparable.

Usage
  plot_phase2_diagnostics.py --run <dir>
  plot_phase2_diagnostics.py --compare C_full80 C_full200 E_ibtracs200
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["ar", "hurricane", "etc", "mcs"]
LABELS = {"ar": "AR", "hurricane": "Hurricane", "etc": "ETC", "mcs": "MCS"}
COL = {"ar": "#1f77b4", "hurricane": "#e8613c", "etc": "#2ca02c", "mcs": "#e8b31c"}
INK = "#131a1f"
PHASE2 = "ARs_Hurricanes_ETCs_MCS/Phase2"


def load(run):
    p = run if run.endswith(".json") else os.path.join(run, "history.json")
    if not os.path.exists(p):
        p = os.path.join(PHASE2, os.path.basename(run.rstrip("/")), "history.json")
    with open(p) as f:
        return json.load(f)


def series(ep, key):
    return np.array([e.get(key, np.nan) for e in ep], dtype=float)


def diagnostics(h, tag, outdir):
    ep = h["epochs"]
    x = series(ep, "epoch")
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.2))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.885, bottom=0.07,
                        hspace=0.30, wspace=0.24)

    # (a) precision, (b) recall
    for k, (met, ttl) in enumerate([("precision", "(a)  Validation precision"),
                                    ("recall", "(b)  Validation recall")]):
        a = ax[0][k]
        for c in CLASSES:
            a.plot(x, series(ep, f"val_{c}_{met}"), lw=1.4, color=COL[c],
                   label=LABELS[c])
        a.set_xlabel("epoch"); a.set_ylabel(met); a.set_ylim(0, 1.02)
        a.grid(alpha=0.25); a.set_title(ttl, loc="left", fontsize=12)
        if k == 0:
            a.legend(fontsize=9, ncol=2, loc="lower right")

    # (c) precision-recall trajectory
    a = ax[0][2]
    for c in CLASSES:
        p = series(ep, f"val_{c}_precision"); r = series(ep, f"val_{c}_recall")
        a.plot(r, p, lw=1.0, alpha=0.55, color=COL[c])
        a.scatter(r[0], p[0], s=42, marker="o", facecolor="white",
                  edgecolor=COL[c], lw=1.6, zorder=4)
        a.scatter(r[-1], p[-1], s=64, marker="*", color=COL[c], zorder=5)
        a.annotate(LABELS[c], (r[-1], p[-1]), textcoords="offset points",
                   xytext=(6, -9), fontsize=9, color=COL[c])
    for f1 in (0.4, 0.6, 0.8):                      # iso-F1 reference curves
        rr = np.linspace(0.01, 1, 200)
        pp = f1 * rr / (2 * rr - f1)
        m = (pp > 0) & (pp <= 1)
        a.plot(rr[m], pp[m], ls=":", lw=0.8, color="0.65", zorder=1)
        if m.any():
            a.annotate(f"F1={f1}", (rr[m][-1], pp[m][-1]), fontsize=7.5,
                       color="0.55")
    a.set_xlabel("recall"); a.set_ylabel("precision")
    a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.grid(alpha=0.25)
    a.set_title("(c)  Precision-recall trajectory  (o start, * end)",
                loc="left", fontsize=12)

    # (d) areal bias
    a = ax[1][0]
    for c in CLASSES:
        tp = series(ep, f"val_{c}_tp"); fp = series(ep, f"val_{c}_fp")
        fn = series(ep, f"val_{c}_fn")
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = (tp + fp) / (tp + fn)
        a.plot(x, bias, lw=1.4, color=COL[c], label=LABELS[c])
    a.axhline(1.0, color="0.35", lw=1.0, ls="--")
    a.annotate("perfect areal extent", (x[len(x) // 2], 1.0),
               textcoords="offset points", xytext=(0, 5), fontsize=8.5,
               color="0.35")
    a.set_xlabel("epoch"); a.set_ylabel("(TP+FP) / (TP+FN)")
    a.set_yscale("log"); a.grid(alpha=0.25)
    a.set_title("(d)  Areal bias — >1 over-predicts, <1 under-predicts",
                loc="left", fontsize=12)
    a.legend(fontsize=9, ncol=2)

    # (e) FP and FN normalised by positives present
    a = ax[1][1]
    for c in CLASSES:
        tp = series(ep, f"val_{c}_tp"); fp = series(ep, f"val_{c}_fp")
        fn = series(ep, f"val_{c}_fn")
        pos = tp + fn
        a.plot(x, fp / pos, lw=1.3, color=COL[c], label=f"{LABELS[c]} FP")
        a.plot(x, fn / pos, lw=1.3, color=COL[c], ls="--", label=f"{LABELS[c]} FN")
    a.set_xlabel("epoch"); a.set_ylabel("count / positives present")
    a.set_yscale("log"); a.grid(alpha=0.25)
    a.set_title("(e)  False positives (solid) vs false negatives (dashed)",
                loc="left", fontsize=12)
    a.legend(fontsize=7.5, ncol=2)

    # (f) generalisation: training loss vs validation IoU
    a = ax[1][2]
    a2 = a.twinx()
    for c in CLASSES:
        a.plot(x, series(ep, f"train_{c}"), lw=1.2, color=COL[c], alpha=0.85)
        a2.plot(x, series(ep, f"val_{c}_iou"), lw=1.2, color=COL[c], ls=":")
    a.set_xlabel("epoch"); a.set_ylabel("training loss (solid)")
    a2.set_ylabel("validation IoU (dotted)"); a2.set_ylim(0, 1.02)
    a.set_yscale("log"); a.grid(alpha=0.25)
    a.set_title("(f)  Training loss vs validation IoU", loc="left", fontsize=12)

    cfg = h.get("config", {})
    fig.suptitle(
        f"Phase 2 diagnostics — {tag}\n"
        f"{cfg.get('n_train_months','?')} training months · "
        f"{cfg.get('n_train_samples','?')} samples · val years "
        f"{cfg.get('val_years','?')} · pos_weight {cfg.get('max_pos_weight','?')}",
        fontsize=14, color=INK, x=0.055, ha="left", y=0.975)
    out = os.path.join(outdir, f"diagnostics_{tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def compare(runs, outdir):
    hs = [(r, load(r)) for r in runs]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.10,
                        hspace=0.34, wspace=0.22)
    styles = ["-", "--", ":", "-."]

    # (a) mean IoU over AR/ETC/MCS only — the classes whose labels are identical
    a = ax[0][0]
    for i, (r, h) in enumerate(hs):
        ep = h["epochs"]; x = series(ep, "epoch")
        m = np.mean([series(ep, f"val_{c}_iou") for c in ("ar", "etc", "mcs")], axis=0)
        a.plot(x, m, lw=1.5, ls=styles[i % 4], label=os.path.basename(r))
    a.set_xlabel("epoch"); a.set_ylabel("mean IoU (AR, ETC, MCS)")
    a.grid(alpha=0.25); a.legend(fontsize=9)
    a.set_title("(a)  Shared-label classes only — directly comparable",
                loc="left", fontsize=12)

    # (b) TC, plotted but flagged as measured against different targets
    a = ax[0][1]
    for i, (r, h) in enumerate(hs):
        ep = h["epochs"]
        a.plot(series(ep, "epoch"), series(ep, "val_hurricane_iou"), lw=1.5,
               ls=styles[i % 4], color=COL["hurricane"],
               label=os.path.basename(r))
    a.set_xlabel("epoch"); a.set_ylabel("val hurricane IoU")
    a.grid(alpha=0.25); a.legend(fontsize=9)
    a.set_title("(b)  TC — NOT comparable across runs\n"
                "     C_full* score against the CyTRACK mask, "
                "E_ibtracs* against IBTrACS", loc="left", fontsize=11)

    # (c) final precision / recall per class, grouped bars
    a = ax[1][0]
    w = 0.8 / max(len(hs), 1)
    idx = np.arange(len(CLASSES))
    for i, (r, h) in enumerate(hs):
        ep = h["epochs"]
        best = max(ep, key=lambda e: np.mean([e[f"val_{c}_iou"] for c in CLASSES]))
        p = [best[f"val_{c}_precision"] for c in CLASSES]
        rc = [best[f"val_{c}_recall"] for c in CLASSES]
        a.bar(idx + i * w - 0.4 + w / 2, p, w * 0.45, label=f"{os.path.basename(r)} P")
        a.bar(idx + i * w - 0.4 + w, rc, w * 0.45, alpha=0.6,
              label=f"{os.path.basename(r)} R")
    a.set_xticks(idx); a.set_xticklabels([LABELS[c] for c in CLASSES])
    a.set_ylabel("at best-mean-IoU epoch"); a.set_ylim(0, 1.05)
    a.grid(alpha=0.25, axis="y"); a.legend(fontsize=7.5, ncol=2)
    a.set_title("(c)  Precision and recall at the best epoch", loc="left",
                fontsize=12)

    # (d) cost: epoch time and LR schedule
    a = ax[1][1]
    for i, (r, h) in enumerate(hs):
        ep = h["epochs"]
        a.plot(series(ep, "epoch"), series(ep, "epoch_seconds"), lw=1.2,
               ls=styles[i % 4], label=f"{os.path.basename(r)} s/epoch")
    a.set_xlabel("epoch"); a.set_ylabel("seconds per epoch")
    a.grid(alpha=0.25)
    a2 = a.twinx()
    for i, (r, h) in enumerate(hs):
        ep = h["epochs"]
        a2.plot(series(ep, "epoch"), series(ep, "lr"), lw=1.0, color="0.55",
                ls=styles[i % 4])
    a2.set_ylabel("learning rate (grey)")
    a.legend(fontsize=8)
    a.set_title("(d)  Epoch time and LR schedule", loc="left", fontsize=12)

    fig.suptitle("Phase 2 run comparison", fontsize=14, color=INK, x=0.07,
                 ha="left", y=0.955)
    out = os.path.join(outdir, "run_comparison_diagnostics.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run dir or history.json")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--out-dir", default="phase2_figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.run:
        h = load(args.run)
        diagnostics(h, h.get("config", {}).get("tag")
                    or os.path.basename(args.run.rstrip("/")), args.out_dir)
    if args.compare:
        compare(args.compare, args.out_dir)
    if not args.run and not args.compare:
        ap.error("give --run and/or --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
