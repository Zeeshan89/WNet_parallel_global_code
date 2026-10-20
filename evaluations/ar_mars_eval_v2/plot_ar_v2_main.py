#!/usr/bin/env python3
"""
plot_ar_v2_main.py — the three-row AR-versus-MARS figure, rebuilt.

Row 1  MARS (benchmark) : AR frequency | mean AR days/yr | mean AR duration profile
Row 2  WNet v1          : AR frequency | mean AR days/yr | per-pixel F1 vs MARS
Row 3  WNet v2 (Phase 2): AR frequency | mean AR days/yr | per-pixel F1 vs MARS

Four things differ from the first version of this figure:

  1. Every panel now comes from one code path. The previous version took the v1
     frequency and F1 from a cache built on a different model vintage
     (Percentile_ARs_detection/.../ETCs_enchanced_test_etc_21) while reading the
     v1 days/yr from the finalized product's CDO climatology, so row 2 mixed two
     provenances. Both v1 and v2 caches are now built by phase4_ar_vs_mars_v2.py.
  2. The colour breaks are extended. The published breaks stopped at 0.12 and
     120, which pushed about a fifth of every panel into the single over-range
     colour -- including the whole tropical over-detection.
  3. pcolormesh is rasterized, as it is in regen_ar_spatial_entire.py. Without
     it the vector PDF of a 721x1440 grid came out at 118 MB.
  4. The duration panel carries all three curves; the v2 curve was missing.

Usage
  python plot_ar_v2_main.py --v1-dir ../ar_mars_eval_v1_recomputed --v2-dir .
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
    EPS, LV_DAYS, LV_F1, LV_FREQ, NYEAR, audit, disc, to_pm180)

PC = ccrs.PlateCarree()


def load(d, label):
    p = os.path.join(d, f"ar_{label}_merged.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run merge_ar_v2.py --cache-dir {d} --label {label}")
    return np.load(p)


def draw(ax, field, src_lon, lat, levels, title):
    lonp, fld = to_pm180(field, src_lon)
    cmap, norm = disc(levels)
    ax.set_global()
    m = ax.pcolormesh(lonp, lat, fld, cmap=cmap, norm=norm, transform=PC,
                      shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#555555")
    ax.set_title(title, fontsize=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    plt.colorbar(m, ax=ax, extend="both", fraction=0.026, pad=0.015, ticks=levels)


def main():
    ap = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--v1-dir", default=os.path.join(HERE, "..", "ar_mars_eval_v1_recomputed"))
    ap.add_argument("--out", default=os.path.join(HERE, "ar_spatial_3row"))
    args = ap.parse_args()

    z2 = load(args.v2_dir, "v2")
    lat, lon = z2["lat"], z2["lon"]
    n2 = int(z2["n_t"])
    have_v1 = os.path.exists(os.path.join(args.v1_dir, "ar_v1_merged.npz"))
    if have_v1:
        z1 = load(args.v1_dir, "v1")
        n1 = int(z1["n_t"])
    else:
        print(f"note: no v1 cache in {args.v1_dir}; plotting MARS and v2 only")

    freq_mars = z2["all_targ"] / max(n2, 1)
    days_mars = z2["all_ardays_t"] / float(NYEAR)
    f_v2 = z2["all_pred"] / max(n2, 1)
    d_v2 = z2["all_ardays"] / float(NYEAR)
    f1_v2 = 2 * z2["all_tp"] / (2 * z2["all_tp"] + z2["all_fp"] + z2["all_fn"] + EPS)

    rows = [("MARS (benchmark)", freq_mars, days_mars, None)]
    if have_v1:
        f1_v1 = 2 * z1["all_tp"] / (2 * z1["all_tp"] + z1["all_fp"] + z1["all_fn"] + EPS)
        rows.append(("WNet v1", z1["all_pred"] / max(n1, 1),
                     z1["all_ardays"] / float(NYEAR), f1_v1))
    rows.append(("WNet v2 (Phase 2)", f_v2, d_v2, f1_v2))

    fig, axes = plt.subplots(len(rows), 3, figsize=(21, 3.5 * len(rows)),
                             subplot_kw={"projection": PC})
    fig.subplots_adjust(left=0.035, right=0.975, top=0.93, bottom=0.05,
                        hspace=0.30, wspace=0.12)
    for r, (label, fq, dy, f1) in enumerate(rows):
        audit(f"{label} freq", fq, LV_FREQ)
        audit(f"{label} days/yr", dy, LV_DAYS)
        draw(axes[r, 0], fq, lon, lat, LV_FREQ, f"{label} — AR frequency")
        draw(axes[r, 1], dy, lon, lat, LV_DAYS, f"{label} — mean AR days/yr")
        if f1 is not None:
            audit(f"{label} F1", f1, LV_F1)
            draw(axes[r, 2], f1, lon, lat, LV_F1, f"{label} — per-pixel F1 vs MARS")

    # the duration profile replaces the empty MARS F1 slot
    box = axes[0, 2].get_position()
    axes[0, 2].remove()
    axd = fig.add_axes([box.x0, box.y0 + 0.035, box.width * 0.80, box.height * 0.80])
    curves = [(os.path.join(args.v2_dir, "ar_duration_v2.npz"), "zon_mars",
               "MARS", "#2a78d6"),
              (os.path.join(args.v1_dir, "ar_duration_v1.npz"), "zon_wnet",
               "WNet v1", "#eb6834"),
              (os.path.join(args.v2_dir, "ar_duration_v2.npz"), "zon_wnet",
               "WNet v2", "#1baf7a")]
    for path, key, name, col in curves:
        if os.path.exists(path):
            z = np.load(path)
            axd.plot(z[key], z["lat"], color=col, lw=2, label=name)
    axd.set_title("Mean AR event duration", fontsize=11)
    axd.set_xlabel("Mean Duration (Days)", fontsize=9)
    axd.set_ylabel("Latitude", fontsize=9)
    axd.grid(alpha=0.3)
    axd.legend(fontsize=9, frameon=False, loc="upper right")
    axd.tick_params(labelsize=8)

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
