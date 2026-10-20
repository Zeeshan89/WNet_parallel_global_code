#!/usr/bin/env python3
"""
plot_tc_2panel_peryear.py — the two-panel cut of plot_tc_4panel_peryear.py:

    New WNet   |   IBTrACS

Same data, same matching, same colour convention as the four-panel figure — every
IBTrACS storm of the year gets its own colour, a model track inherits the colour of
the storm it matches, and unmatched model tracks are grey (i.e. false alarms). The
WNet-old and TempestExtremes panels are simply dropped, and the panel titles are the
bare product names, so the figure stands on its own next to the four-panel version.

The drawing, parsing and matching all come from the four-panel script; nothing is
reimplemented here.

Environment: /ccs/home/zaslam/.conda/envs/my_env/bin/python

Usage
  plot_tc_2panel_peryear.py --new-dir <tracks dir> --years 2005 --outdir <dir>
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_tracks_peryear import (           # noqa: E402
    parse_hurricane_txt, ibtracs_tracks, match_tracks, palette,
)
from plot_tc_4panel_peryear import draw     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True,
                    help="tracks dir of the new model (contains hurricane/CONUS_year_YYYY/)")
    ap.add_argument("--years", nargs="+", type=int, default=[2005])
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--outdir", default="phase4_figures/tc_peryear_4panel_pw50")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for y in args.years:
        new = parse_hurricane_txt(os.path.join(
            args.new_dir, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        ref = ibtracs_tracks(y, wind_kt=args.wind_kt)
        if not ref:
            print(f"  {y}: no IBTrACS reference — skipped")
            continue

        cols = palette(len(ref))
        a_new = match_tracks(new, ref, args.radius_km)
        c_new = [None if j is None else cols[j] for j in a_new]
        f_new = len({j for j in a_new if j is not None})

        fig, axes = plt.subplots(1, 2, figsize=(19, 3.9),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.035, right=0.99, top=0.90, bottom=0.06, wspace=0.07)
        draw(axes[0], new, c_new, "New WNet")
        draw(axes[1], ref, [cols[j] for j in range(len(ref))], "IBTrACS")

        out = os.path.join(args.outdir, f"tc_2panel_{y}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {y}: New WNet {len(new)} tracks, {f_new}/{len(ref)} storms found "
              f"| IBTrACS {len(ref)} storms")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
