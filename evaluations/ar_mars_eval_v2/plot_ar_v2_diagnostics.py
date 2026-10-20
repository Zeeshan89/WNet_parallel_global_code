#!/usr/bin/env python3
"""
plot_ar_v2_diagnostics.py — what the single global skill number hides.

The whole-record IoU is 0.301 for both the v1 and the v2 AR product, which reads
as "no change". Splitting the same cache by latitude shows that is two large
errors cancelling: v2 is better than v1 through the midlatitudes, where ARs are
a meaningful category, and much worse in the deep tropics, where a bare
IVT-threshold label with no geometry test flags the ITCZ almost continuously.

Panels
  a  v2 minus MARS AR frequency
  b  v1 minus MARS AR frequency
  c  zonal-mean AR frequency, MARS / v1 / v2
  d  skill by latitude band, v1 versus v2

Usage
  python plot_ar_v2_diagnostics.py --v1-dir ../ar_mars_eval_v1_recomputed --v2-dir .
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import (  # noqa: E402
    BANDS, LV_DIFF, LV_F1, cos_w, disc, disc_diff, map_metrics, to_pm180)

PC = ccrs.PlateCarree()


def diff_map(ax, field, lon, lat, title, levels=None, diverging=True):
    lonp, fld = to_pm180(field, lon)
    levels = LV_DIFF if levels is None else levels
    cmap, norm = disc_diff(levels) if diverging else disc(levels)
    ax.set_global()
    m = ax.pcolormesh(lonp, lat, fld, cmap=cmap, norm=norm, transform=PC,
                      shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#333333")
    ax.set_title(title, fontsize=13, color="#000000")
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 10, "color": "#000000"}
    plt.colorbar(m, ax=ax, extend="both", fraction=0.026, pad=0.015, ticks=levels)


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--v1-dir", default=os.path.join(HERE, "..", "ar_mars_eval_v1_recomputed"))
    ap.add_argument("--out", default=os.path.join(HERE, "ar_v2_diagnostics"))
    args = ap.parse_args()

    z2 = np.load(os.path.join(args.v2_dir, "ar_v2_merged.npz"))
    lat, lon = z2["lat"], z2["lon"]
    n2 = int(z2["n_t"])
    w = cos_w(lat)
    alat = np.abs(np.asarray(lat, float))

    fm = z2["all_targ"] / n2
    f2 = z2["all_pred"] / n2
    p1 = os.path.join(args.v1_dir, "ar_v1_merged.npz")
    z1 = np.load(p1) if os.path.exists(p1) else None
    f1 = z1["all_pred"] / int(z1["n_t"]) if z1 is not None else None

    fig = plt.figure(figsize=(19, 9.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0], projection=PC)
    diff_map(ax, f2 - fm, lon, lat, "(a)  WNet v2 minus MARS, AR frequency")
    ax = fig.add_subplot(gs[0, 1], projection=PC)
    if f1 is not None:
        diff_map(ax, f1 - fm, lon, lat, "(b)  WNet v1 minus MARS, AR frequency")
    else:
        # without a v1 cache the slot shows where v2 and MARS agree instead
        f1_map = 2 * z2["all_tp"] / (2 * z2["all_tp"] + z2["all_fp"]
                                     + z2["all_fn"] + 1e-12)
        diff_map(ax, f1_map, lon, lat, "(b)  Per-pixel F1, WNet v2 against MARS",
                 levels=LV_F1, diverging=False)

    # ---- (c) zonal-mean frequency
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(fm.mean(1), lat, color="#2a78d6", lw=2, label="MARS")
    if f1 is not None:
        axc.plot(f1.mean(1), lat, color="#eb6834", lw=2, label="WNet v1")
    axc.plot(f2.mean(1), lat, color="#1baf7a", lw=2, label="WNet v2")
    axc.axhspan(-15, 15, color="#cccccc", alpha=0.35, lw=0)
    axc.text(0.98, 0.5, "deep tropics", transform=axc.transAxes, rotation=90,
             ha="right", va="center", fontsize=8, color="#777777")
    axc.set_title("(c)  Zonal-mean AR frequency", fontsize=13)
    axc.set_xlabel("fraction of 6-hourly steps flagged as AR", fontsize=11.5)
    axc.set_ylabel("Latitude", fontsize=11.5)
    axc.set_ylim(-90, 90)
    axc.set_yticks(range(-90, 91, 30))
    axc.grid(alpha=0.3)
    axc.legend(fontsize=11, frameon=False)
    axc.tick_params(labelsize=10.5, colors="#000000")

    # ---- (d) skill by latitude band
    axd = fig.add_subplot(gs[1, 1])
    names = [b[0] for b in BANDS]
    keys = ("IoU", "POD", "FAR", "BIAS")
    x = np.arange(len(keys))
    width = 0.8 / (len(BANDS) * (2 if z1 is not None else 1))
    colors = plt.get_cmap("tab20")(np.linspace(0, 0.95, len(BANDS)))
    i = 0
    for bi, (bname, sel) in enumerate(BANDS):
        msk = sel(alat)
        sets = [("v2", z2, "")]
        if z1 is not None:
            sets = [("v1", z1, "//"), ("v2", z2, "")]
        for tag, zz, hatch in sets:
            m = map_metrics(zz["all_tp"], zz["all_fp"], zz["all_fn"], w=w, mask=msk)
            vals = [min(m[k], 2.0) for k in keys]      # BIAS is clipped for display
            axd.bar(x + i * width - 0.4 + width / 2, vals, width,
                    color=colors[bi], hatch=hatch, edgecolor="white", linewidth=0.4,
                    label=f"{bname} [{tag}]")
            i += 1
    axd.axhline(1.0, color="#888888", lw=0.8, ls="--")
    axd.set_xticks(x)
    axd.set_xticklabels(keys)
    axd.set_title("(d)  Skill by latitude band (cos-lat weighted; BIAS clipped at 2)",
                  fontsize=13)
    axd.grid(alpha=0.3, axis="y")
    axd.legend(fontsize=9.5, frameon=False, ncol=2, loc="upper left")
    axd.tick_params(labelsize=10.5, colors="#000000")

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)

    # print the table the figure draws, so it lands in the run log too
    print(f"\n{'band':<24}{'model':>7}{'IoU':>9}{'POD':>9}{'FAR':>9}{'BIAS':>9}")
    for bname, sel in BANDS:
        msk = sel(alat)
        for tag, zz in ([("v1", z1)] if z1 is not None else []) + [("v2", z2)]:
            m = map_metrics(zz["all_tp"], zz["all_fp"], zz["all_fn"], w=w, mask=msk)
            print(f"{bname:<24}{tag:>7}{m['IoU']:>9.4f}{m['POD']:>9.4f}"
                  f"{m['FAR']:>9.4f}{m['BIAS']:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
