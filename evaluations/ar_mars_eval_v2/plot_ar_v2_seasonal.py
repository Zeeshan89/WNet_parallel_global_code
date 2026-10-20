#!/usr/bin/env python3
"""
plot_ar_v2_seasonal.py — AR frequency by season, benchmark against v2.

Four columns DJF / MAM / JJA / SON; rows MARS, WNet v2, and their difference.
Season months follow precip_framework/lib/pfa_common.py.

The seasonal split is what separates the two candidate readings of the tropical
over-detection: a genuine moisture-transport signal would migrate with the ITCZ
through the year, whereas a systematic threshold artefact would sit still.

Usage
  python plot_ar_v2_seasonal.py --v2-dir .
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import (  # noqa: E402
    LV_DIFF, LV_FREQ_SEA, SEASONS, audit, cos_w, disc, disc_diff, map_metrics, to_pm180)

PC = ccrs.PlateCarree()


def panel(ax, field, lon, lat, levels, title, diverging=False):
    lonp, fld = to_pm180(field, lon)
    cmap, norm = (disc_diff(levels) if diverging else disc(levels))
    ax.set_global()
    m = ax.pcolormesh(lonp, lat, fld, cmap=cmap, norm=norm, transform=PC,
                      shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#555555")
    ax.set_title(title, fontsize=10)
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return m


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "ar_v2_seasonal"))
    args = ap.parse_args()

    z = np.load(os.path.join(args.v2_dir, "ar_v2_merged.npz"))
    lat, lon = z["lat"], z["lon"]
    w = cos_w(lat)

    fig, axes = plt.subplots(3, 4, figsize=(23, 10.5), subplot_kw={"projection": PC})
    fig.subplots_adjust(left=0.03, right=0.94, top=0.93, bottom=0.04,
                        hspace=0.22, wspace=0.10)

    mf = md = None
    for c, s in enumerate(SEASONS):
        nt = int(z[f"scal_{s}"][4])
        fm = z[f"sea_{s}_targ"] / max(nt, 1)
        fp = z[f"sea_{s}_pred"] / max(nt, 1)
        m = map_metrics(z[f"sea_{s}_tp"], z[f"sea_{s}_fp"], z[f"sea_{s}_fn"], w=w)
        audit(f"MARS {s} freq", fm, LV_FREQ_SEA)
        audit(f"v2 {s} freq", fp, LV_FREQ_SEA)
        print(f"  {s}: n_t={nt}  IoU={m['IoU']:.4f}  POD={m['POD']:.4f}  "
              f"FAR={m['FAR']:.4f}  BIAS={m['BIAS']:.4f}  (cos-lat weighted)")
        mf = panel(axes[0, c], fm, lon, lat, LV_FREQ_SEA, f"MARS — {s}")
        panel(axes[1, c], fp, lon, lat, LV_FREQ_SEA, f"WNet v2 — {s}")
        md = panel(axes[2, c], fp - fm, lon, lat, LV_DIFF,
                   f"v2 − MARS — {s}   (IoU {m['IoU']:.3f}, BIAS {m['BIAS']:.2f})",
                   diverging=True)

    cax = fig.add_axes([0.955, 0.42, 0.010, 0.45])
    fig.colorbar(mf, cax=cax, extend="both", ticks=LV_FREQ_SEA).set_label(
        "AR frequency", fontsize=9)
    cax2 = fig.add_axes([0.955, 0.07, 0.010, 0.24])
    fig.colorbar(md, cax=cax2, extend="both", ticks=LV_DIFF).set_label(
        "difference", fontsize=9)

    for ext in ("png", "pdf"):
        out = f"{args.out}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
