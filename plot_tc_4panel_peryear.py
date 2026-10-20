#!/usr/bin/env python3
"""
plot_tc_4panel_peryear.py — ONE FIGURE PER YEAR, four panels each:

    WNet new (IBTrACS-trained)   |   WNet old (CyTRACK-trained)
    TempestExtremes              |   IBTrACS (observed)

all at global scale, 60S-60N.

Colour convention, same as plot_tracks_peryear.py: every IBTrACS storm of that year
gets its own colour; a model track inherits the colour of the storm it matches most
often (>= 2 corresponding points within --radius-km at the same 6-hourly stamp);
unmatched model tracks are grey. Grey is therefore the false alarms, and the same
real storm carries the same colour in all four panels.

The TempestExtremes trajectory files are NOT global — they span roughly lat 10..46,
lon -135..-60 (N Atlantic / E Pacific). Its box is drawn so the empty remainder is
not misread as a detection failure; it is a property of that reference file.

Environment: /ccs/home/zaslam/.conda/envs/my_env/bin/python

Usage
  plot_tc_4panel_peryear.py --new-dir <tracks_final> --years 1980 ... 2024
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_tracks_peryear import (           # noqa: E402
    parse_hurricane_txt, ibtracs_tracks, match_tracks, palette, GREY, INK, OLD,
)
from phase4_evaluate_tracks import parse_te_tc    # noqa: E402

TE_BOX = dict(lon=(-135.0, -60.0), lat=(10.0, 46.0))


def draw(ax, tracks, colours, title, te_box=False):
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    ax.coastlines(lw=0.35, color="0.5", zorder=1)
    order = sorted(range(len(tracks)), key=lambda i: colours[i] is not None)
    for i in order:
        t, c = tracks[i], colours[i]
        la = [p[1] for p in t]
        lo = [p[2] for p in t]
        col = GREY if c is None else c
        lw, al, z = (0.6, 0.5, 2) if c is None else (1.1, 0.92, 3)
        sla, slo = [la[0]], [lo[0]]
        for k in range(1, len(la)):
            if abs(lo[k] - lo[k - 1]) > 180:          # dateline wrap
                ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                        transform=ccrs.PlateCarree(), zorder=z)
                sla, slo = [], []
            sla.append(la[k]); slo.append(lo[k])
        if len(sla) > 1:
            ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                    transform=ccrs.PlateCarree(), zorder=z)
        ax.plot(lo[0], la[0], "o", color=col, ms=2.4 if c is not None else 1.2,
                alpha=al, transform=ccrs.PlateCarree(), zorder=z + 1)
    if te_box:
        lo0, lo1 = TE_BOX["lon"]; la0, la1 = TE_BOX["lat"]
        ax.add_patch(mpatches.Rectangle(
            (lo0, la0), lo1 - lo0, la1 - la0, transform=ccrs.PlateCarree(),
            fill=False, ec="#c0392b", lw=1.0, ls="--", zorder=5))
        ax.text(lo0 + 2, la1 + 3, "TE file covers this box only",
                transform=ccrs.PlateCarree(), fontsize=7, color="#c0392b", zorder=6)
    gl = ax.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.25, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(title, fontsize=10.5, color=INK, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True,
                    help="tracks dir of the IBTrACS-trained model "
                         "(contains hurricane/CONUS_year_YYYY/)")
    ap.add_argument("--old-dir", default=os.path.join(OLD, "Hurricanes_WNet_final"),
                    help="old WNet tracks; default is the finalized product")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(1980, 2025)))
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--outdir", default="phase4_figures/tc_peryear_4panel")
    ap.add_argument("--label", default=None,
                    help="product name stamped on the figure; defaults to the "
                         "basename of --new-dir")
    ap.add_argument("--gate-h", type=float,
                    default=float(os.environ.get("TC_MIN_DURATION_H", "48")),
                    help="minimum track duration the --new-dir tracks were built "
                         "with; stamped on the figure so the two products cannot "
                         "be confused")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Two TC track products now exist side by side (48 h `tracks_final` and 24 h
    # `tracks_final_dur24`), and the reference can be keyed two ways, so a figure
    # that does not say which it shows is unreadable a month from now. Both are
    # stamped on the panel and the caption.
    _gate_h = args.gate_h
    _ykey = os.environ.get("IBTRACS_YEAR_KEY", "calendar")
    _wsrc = os.environ.get("IBTRACS_WIND_SOURCE", "usa")
    # There is now more than one TC track product (cap-8 `tracks_final_dur24` and
    # cap-50 `tracks_pw50_dur24`), both built with the same 24 h gate. Without the
    # product name on the figure the two are indistinguishable, so stamp the track
    # directory's basename unless the caller overrides it.
    _prod = args.label or os.path.basename(args.new_dir.rstrip("/"))
    print(f"  new-dir : {args.new_dir}")
    print(f"  product : {_prod}")
    print(f"  config  : {_gate_h:g} h duration gate, IBTrACS keyed by {_ykey} "
          f"year, wind source '{_wsrc}', >= {args.wind_kt:g} kt\n")

    print(f"  {'year':<6}{'new':>6}{'old':>6}{'TE':>6}{'IBTrACS':>9}"
          f"{'new found':>11}{'old found':>11}{'TE found':>10}")
    print("  " + "-" * 66)
    for y in args.years:
        new = parse_hurricane_txt(os.path.join(
            args.new_dir, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        old = parse_hurricane_txt(os.path.join(
            args.old_dir, f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        te = parse_te_tc(y)
        ref = ibtracs_tracks(y, wind_kt=args.wind_kt)
        if not ref:
            print(f"  {y:<6} no IBTrACS reference — skipped")
            continue

        cols = palette(len(ref))
        a_new = match_tracks(new, ref, args.radius_km)
        a_old = match_tracks(old, ref, args.radius_km)
        a_te = match_tracks(te, ref, args.radius_km)
        c_new = [None if j is None else cols[j] for j in a_new]
        c_old = [None if j is None else cols[j] for j in a_old]
        c_te = [None if j is None else cols[j] for j in a_te]
        f_new = len({j for j in a_new if j is not None})
        f_old = len({j for j in a_old if j is not None})
        f_te = len({j for j in a_te if j is not None})

        fig, axes = plt.subplots(2, 2, figsize=(19, 7.6),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.035, right=0.99, top=0.86, bottom=0.04,
                            hspace=0.20, wspace=0.07)
        draw(axes[0][0], new, c_new,
             f"WNet NEW [{_prod}] — {len(new)} tracks, "
             f"{f_new}/{len(ref)} storms found")
        draw(axes[0][1], old, c_old,
             f"WNet OLD (CyTRACK-trained) — {len(old)} tracks, {f_old}/{len(ref)} storms found")
        draw(axes[1][0], te, c_te,
             f"TempestExtremes — {len(te)} tracks, {f_te}/{len(ref)} storms found",
             te_box=True)
        draw(axes[1][1], ref, [cols[j] for j in range(len(ref))],
             f"IBTrACS (observed) — {len(ref)} storms")
        fig.suptitle(
            f"Tropical cyclone tracks {y}  —  colour = matched IBTrACS storm "
            "(same colour in every panel); grey = no match, i.e. a false alarm\n"
            "The old model was trained on a mask with IBTrACS dilated into it, so its "
            "storms-found count is circular and not a fair competitor   |   "
            f"WNet NEW: {_prod}, {_gate_h:g} h duration gate, "
            f"reference keyed by {_ykey} year",
            fontsize=12, color=INK, x=0.035, ha="left", y=0.975)

        out = os.path.join(args.outdir, f"tc_4panel_{y}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {y:<6}{len(new):>6}{len(old):>6}{len(te):>6}{len(ref):>9}"
              f"{f'{f_new}/{len(ref)}':>11}{f'{f_old}/{len(ref)}':>11}"
              f"{f'{f_te}/{len(ref)}':>10}", flush=True)
    print(f"\n  wrote {len(os.listdir(args.outdir))} figures to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
