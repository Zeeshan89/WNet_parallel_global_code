#!/usr/bin/env python3
"""
plot_ar_v2_timeseries.py — monthly skill and interannual AR-day trends.

Top     monthly IoU and F1 versus MARS, thin monthly line plus a 12-month
        centred rolling mean, with the 2006 out-of-period / primary split marked
Middle  monthly POD and FAR, same treatment
Bottom  annual global-mean AR days per year, MARS and WNet v2, each with an
        ordinary least-squares trend line and its slope in days/decade

The area-weighted global mean is used for the annual series, because an
unweighted mean over a 721x1440 regular grid is dominated by polar rows.

Usage
  python plot_ar_v2_timeseries.py --v2-dir .
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import cos_w  # noqa: E402

SPLIT = 2006.0


def roll(a, n=12):
    """Centred rolling mean, NaN-padded at the ends."""
    a = np.asarray(a, float)
    out = np.full(a.shape, np.nan)
    if len(a) >= n:
        k = np.convolve(a, np.ones(n) / n, mode="valid")
        off = (n - 1) // 2
        out[off:off + len(k)] = k
    return out


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "ar_v2_timeseries"))
    args = ap.parse_args()

    z = np.load(os.path.join(args.v2_dir, "ar_v2_merged.npz"))
    lat, lon = z["lat"], z["lon"]
    w = cos_w(lat)
    wsum = float(w.sum() * lon.size)

    mr = z["month_rows"]
    t = mr[:, 0] + (mr[:, 1] - 0.5) / 12.0
    tp, fp, fn = mr[:, 2].astype(float), mr[:, 3].astype(float), mr[:, 4].astype(float)
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    pod = tp / np.maximum(tp + fn, 1)
    far = fp / np.maximum(tp + fp, 1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.32, left=0.07, right=0.97, top=0.95, bottom=0.06)

    for ax, series, title in (
            (axes[0], [("IoU", iou, "#2a78d6"), ("F1", f1, "#1baf7a")],
             "Monthly pixel skill of WNet v2 AR versus MARS"),
            (axes[1], [("POD", pod, "#1baf7a"), ("FAR", far, "#eb6834")],
             "Monthly detection rate and false-alarm rate")):
        for name, y, col in series:
            ax.plot(t, y, color=col, lw=0.5, alpha=0.32)
            ax.plot(t, roll(y), color=col, lw=2.0, label=f"{name} (12-mo mean)")
        ax.axvline(SPLIT, color="#555555", lw=1.0, ls="--")
        ax.text(SPLIT + 0.3, ax.get_ylim()[1], " primary →", va="top", fontsize=8,
                color="#555555")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(1980, 2025)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False, ncol=2)
        ax.tick_params(labelsize=8)

    # ---- annual AR days, area-weighted global mean, with trends
    ax = axes[2]
    year0 = int(z["year0"])
    for key, name, col in (("yr_days_t", "MARS", "#2a78d6"),
                           ("yr_days", "WNet v2", "#1baf7a")):
        # Bind the array once. Indexing an NpzFile re-reads and re-decompresses
        # the whole 187 MB array on every lookup, so leaving z[key] inside the
        # comprehension below would decompress it 45 times per series.
        arr = z[key]
        years = np.arange(year0, year0 + arr.shape[0])
        ann = np.array([float((arr[i] * w).sum() / wsum) for i in range(arr.shape[0])])
        ax.plot(years, ann, color=col, lw=1.6, marker="o", ms=3, label=name)
        sl, ic = np.polyfit(years, ann, 1)
        ax.plot(years, sl * years + ic, color=col, lw=1.2, ls="--", alpha=0.8,
                label=f"{name} trend {sl * 10:+.2f} d/decade")
    ax.set_title("Global-mean AR days per year (cos-lat weighted)", fontsize=11)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("AR days / yr", fontsize=9)
    ax.set_xlim(1979, 2025)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    ax.tick_params(labelsize=8)

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
