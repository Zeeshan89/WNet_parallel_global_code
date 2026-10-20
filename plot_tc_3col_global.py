#!/usr/bin/env python3
"""
plot_tc_3col_global.py — WNet | TempestExtremes | IBTrACS, three columns side by
side at global scale, for the whole record.

Two modes:
  --mode all      one row, every year's tracks overlaid (the climatology view)
  --mode peryear  one row per year (the existing tracks_peryear layout)

Colour convention matches plot_tracks_peryear.py: each IBTrACS storm gets a colour,
a model track inherits the colour of the storm it matches, and unmatched model
tracks are grey — so grey is the false alarms.

IMPORTANT CAVEAT drawn on the figure: the TempestExtremes TC trajectory files are
NOT global. They span roughly lat 10..46, lon -135..-60 (N Atlantic / E Pacific).
On a global map its column is therefore empty almost everywhere, which is a
property of the reference file and not a failure of the tracker. Its box is drawn
so the empty area cannot be misread.

Environment: /ccs/home/zaslam/.conda/envs/my_env/bin/python
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_tracks_peryear import (          # noqa: E402
    parse_hurricane_txt, ibtracs_tracks, match_tracks, palette, GREY, INK,
)
from phase4_evaluate_tracks import parse_te_tc   # noqa: E402

TE_BOX = dict(lon=(-135.0, -60.0), lat=(10.0, 46.0))


def draw(ax, tracks, colours, title, show_te_box=False):
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    ax.coastlines(lw=0.3, color="0.55", zorder=1)
    order = sorted(range(len(tracks)), key=lambda i: colours[i] is not None)
    for i in order:
        t, c = tracks[i], colours[i]
        la = [p[1] for p in t]
        lo = [p[2] for p in t]
        col = GREY if c is None else c
        lw, al, z = (0.35, 0.40, 2) if c is None else (0.55, 0.85, 3)
        sla, slo = [la[0]], [lo[0]]
        for k in range(1, len(la)):
            if abs(lo[k] - lo[k - 1]) > 180:      # dateline wrap
                ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                        transform=ccrs.PlateCarree(), zorder=z)
                sla, slo = [], []
            sla.append(la[k]); slo.append(lo[k])
        if len(sla) > 1:
            ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                    transform=ccrs.PlateCarree(), zorder=z)
    if show_te_box:
        lo0, lo1 = TE_BOX["lon"]; la0, la1 = TE_BOX["lat"]
        ax.add_patch(mpatches.Rectangle(
            (lo0, la0), lo1 - lo0, la1 - la0, transform=ccrs.PlateCarree(),
            fill=False, ec="#c0392b", lw=1.1, ls="--", zorder=5))
        ax.text(lo0 + 2, la1 + 3, "TE file covers this box only",
                transform=ccrs.PlateCarree(), fontsize=7.5, color="#c0392b",
                zorder=6)
    gl = ax.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.25, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(title, fontsize=10, color=INK, pad=4)


def load_year(y, track_dir, wind_kt):
    new = parse_hurricane_txt(os.path.join(
        track_dir, "hurricane", f"CONUS_year_{y}",
        f"CONUS_hurricane_tracks_detailed_{y}.txt"))
    te = parse_te_tc(y)
    ref = ibtracs_tracks(y, wind_kt=wind_kt)
    return new, te, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=list(range(1980, 2025)))
    ap.add_argument("--mode", choices=["all", "peryear"], default="all")
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--outdir", default="phase4_figures")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    loaded = []
    for y in args.years:
        new, te, ref = load_year(y, args.track_dir, args.wind_kt)
        if not ref:
            continue
        loaded.append((y, new, te, ref))
        print(f"  {y}: WNet {len(new):4d} | TE {len(te):3d} | IBTrACS {len(ref):3d}",
              flush=True)
    if not loaded:
        raise SystemExit("no years loaded — check --track-dir")

    if args.mode == "all":
        # Colour every storm of every year, keeping a per-year palette so adjacent
        # storms stay distinguishable; matching is done within each year, since a
        # storm only exists in its own season.
        A_new, C_new, A_te, C_te, A_ref, C_ref = [], [], [], [], [], []
        for (y, new, te, ref) in loaded:
            cols = palette(len(ref))
            a_new = match_tracks(new, ref, args.radius_km)
            a_te = match_tracks(te, ref, args.radius_km)
            A_new += new; C_new += [None if j is None else cols[j] for j in a_new]
            A_te += te;   C_te += [None if j is None else cols[j] for j in a_te]
            A_ref += ref; C_ref += [cols[j] for j in range(len(ref))]
        h_new = sum(c is not None for c in C_new)
        h_te = sum(c is not None for c in C_te)
        y0, y1 = loaded[0][0], loaded[-1][0]

        fig, axes = plt.subplots(3, 1, figsize=(15, 12.6),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.03, hspace=0.16)
        draw(axes[0], A_new, C_new,
             f"WNet (IBTrACS-trained) — {len(A_new)} tracks, {h_new} matched")
        draw(axes[1], A_te, C_te,
             f"TempestExtremes — {len(A_te)} tracks, {h_te} matched",
             show_te_box=True)
        draw(axes[2], A_ref, C_ref, f"IBTrACS (observed) — {len(A_ref)} storms")
        fig.suptitle(
            f"Tropical cyclone tracks {y0}-{y1}, all years overlaid  —  "
            "colour = matched IBTrACS storm; grey = no match (false alarm)\n"
            "TempestExtremes trajectories are regional (boxed); its blank area is a "
            "property of that reference file, not a detection failure",
            fontsize=11.5, color=INK, x=0.05, ha="left", y=0.975)
        out = os.path.join(args.outdir, f"tc_3col_global_{args.tag}_{y0}_{y1}.png")
    else:
        n = len(loaded)
        fig, axes = plt.subplots(n, 3, figsize=(19, 2.75 * n),
                                 subplot_kw={"projection": ccrs.PlateCarree()},
                                 squeeze=False)
        fig.subplots_adjust(left=0.04, right=0.99, top=1 - 0.35 / n, bottom=0.01,
                            hspace=0.22, wspace=0.06)
        for r, (y, new, te, ref) in enumerate(loaded):
            cols = palette(len(ref))
            a_new = match_tracks(new, ref, args.radius_km)
            a_te = match_tracks(te, ref, args.radius_km)
            c_new = [None if j is None else cols[j] for j in a_new]
            c_te = [None if j is None else cols[j] for j in a_te]
            draw(axes[r][0], new, c_new,
                 f"WNet — {len(new)} tracks, "
                 f"{len({j for j in a_new if j is not None})}/{len(ref)} found")
            draw(axes[r][1], te, c_te,
                 f"TempestExtremes — {len(te)} tracks, "
                 f"{len({j for j in a_te if j is not None})}/{len(ref)} found",
                 show_te_box=(r == 0))
            draw(axes[r][2], ref, [cols[j] for j in range(len(ref))],
                 f"IBTrACS — {len(ref)} storms")
            axes[r][0].text(-0.05, 0.5, str(y), transform=axes[r][0].transAxes,
                            rotation=90, va="center", ha="center",
                            fontsize=12, fontweight="bold", color=INK)
        fig.suptitle("Tropical cyclone tracks by year — WNet | TempestExtremes | "
                     "IBTrACS", fontsize=13, color=INK, x=0.04, ha="left",
                     y=1 - 0.10 / n)
        out = os.path.join(args.outdir, f"tc_3col_peryear_{args.tag}.png")

    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
