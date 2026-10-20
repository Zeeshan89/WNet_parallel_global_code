#!/usr/bin/env python3
"""
plot_ar_v2_2col.py — the AR climatology figure reduced to two columns.

              AR frequency | mean AR days per year
  Row 1  MARS (benchmark)
  Row 2  WNet

Same fields and same colour breaks as plot_ar_v2_main.py, so the panels are
directly comparable with the three-row figure; the F1 and duration columns are
dropped. Both rows share one colour bar per column, since a column is a single
quantity on a single scale.

Usage
  python plot_ar_v2_2col.py --v2-dir . --out ar_spatial_2col
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cartopy.crs as ccrs       # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import (  # noqa: E402
    LV_DAYS, LV_DAYS_120, LV_DAYS_PUB, LV_FREQ, LV_FREQ_PUB, NYEAR, audit, disc,
    to_pm180)

PC = ccrs.PlateCarree()
INK = "#000000"          # black throughout; grey renders blurry in print
FS_TITLE, FS_ROW, FS_TICK, FS_CBAR = 14, 14, 10.5, 10.5


def draw(ax, field, src_lon, lat, levels):
    """One map panel. Returns the mappable so the column can share a colour bar."""
    lonp, fld = to_pm180(field, src_lon)
    cmap, norm = disc(levels)
    ax.set_global()
    m = ax.pcolormesh(lonp, lat, fld, cmap=cmap, norm=norm, transform=PC,
                      shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#555555")
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": FS_TICK, "color": INK}
    return m


def save(fig, out):
    for ext in ("png", "pdf"):
        p = f"{out}.{ext}"
        fig.savefig(p, dpi=250 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    plt.close(fig)


def days_only(rows, levels, lon, lat, out):
    """(a) MARS | (b) WNet, mean AR days per year, one shared colour bar."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.4), subplot_kw={"projection": PC})
    fig.subplots_adjust(left=0.045, right=0.99, top=0.92, bottom=0.24, wspace=0.10)
    for i, (ax, (label, _, days)) in enumerate(zip(axes, rows)):
        audit(f"{label} days per year", days, levels)
        m = draw(ax, days, lon, lat, levels)
        ax.set_title(f"({'ab'[i]}) {label}", fontsize=FS_TITLE, color=INK,
                     loc="left", pad=7)
    cax = fig.add_axes([0.27, 0.115, 0.46, 0.04])
    # the scale starts at 0, so there is nothing below it to extend towards
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="max", ticks=levels)
    cb.set_label("Mean AR days per year", fontsize=FS_TICK + 1, color=INK)
    cb.ax.tick_params(labelsize=FS_CBAR, colors=INK)
    cb.outline.set_edgecolor(INK)
    save(fig, out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "ar_spatial_2col"))
    # published breaks stop at 0.12 / 120, which saturates the v2 tropics but
    # gives the MARS row more contrast
    ap.add_argument("--levels", choices=("extended", "published"), default="extended")
    ap.add_argument("--days-only", action="store_true",
                    help="one row, MARS | WNet, mean AR days per year only")
    args = ap.parse_args()

    p = os.path.join(args.v2_dir, "ar_v2_merged.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run merge_ar_v2.py first")
    z = np.load(p)
    lat, lon, n_t = z["lat"], z["lon"], int(z["n_t"])

    rows = [("MARS", z["all_targ"] / max(n_t, 1),
             z["all_ardays_t"] / float(NYEAR)),
            ("WNet", z["all_pred"] / max(n_t, 1),
             z["all_ardays"] / float(NYEAR))]
    pub = args.levels == "published"
    cols = [("AR frequency", LV_FREQ_PUB if pub else LV_FREQ),
            ("Mean AR days per year", LV_DAYS_PUB if pub else LV_DAYS)]

    if args.days_only:
        return days_only(rows, LV_DAYS_120, lon, lat, args.out)   # paper scale, 0 to 120

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.0),
                             subplot_kw={"projection": PC})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.18,
                        hspace=0.16, wspace=0.10)

    mappables = [None, None]
    for r, (label, fq, dy) in enumerate(rows):
        for c, (ctitle, lv) in enumerate(cols):
            field = (fq, dy)[c]
            audit(f"{label} {ctitle}", field, lv)
            mappables[c] = draw(axes[r, c], field, lon, lat, lv)
            if r == 0:
                axes[r, c].set_title(ctitle, fontsize=FS_TITLE, color=INK, pad=8)
        # clear of the latitude labels, which sit just inside x = 0
        axes[r, 0].text(-0.155, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=FS_ROW, color=INK)

    # one horizontal colour bar per column, under the lower row
    for c, (_, lv) in enumerate(cols):
        b = axes[1, c].get_position()
        # below the lower row's longitude labels, not across them
        cax = fig.add_axes([b.x0 + 0.03, 0.055, b.width - 0.06, 0.022])
        cb = fig.colorbar(mappables[c], cax=cax, orientation="horizontal",
                          extend="both", ticks=lv)
        cb.ax.tick_params(labelsize=FS_CBAR, colors=INK)
        cb.outline.set_edgecolor(INK)

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
