#!/usr/bin/env python3
"""
plot_ar_regional_gap.py — where inside each regional box do MARS and WNet differ?

The regional time series (ar_regional_timeseries) follow each other closely from
year to year (r about 0.95) but sit at different levels. A constant offset is a
spatial question, not a temporal one, so this script resolves the region mean
along the two directions that matter:

  top row     mean AR days per year against latitude   (all pixels in the box)
  bottom row  mean AR days per year against longitude, over the surface the
              region set is about: ocean pixels for --set ocean (the approach to
              the coast, without the land effect), land pixels for --set land
              (the decay from the coast inland, without the adjacent seas)

It also prints how the region-mean gap splits into the other surface, the
equatorward, central and poleward thirds of the primary surface, and the strip
next to the coast, plus the WNet/MARS AR-area ratio by season.

Usage
  python plot_ar_regional_gap.py [--set ocean|land]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import NYEAR  # noqa: E402
import plot_ar_regional as R  # noqa: E402
from plot_ar_regional import (  # noqa: E402
    C_MARS, C_WNET, FS_LAB, FS_LEG, FS_TICK, FS_TITLE, INK, TAGS, Cache, save)

SEASONS = ("DJF", "MAM", "JJA", "SON")


def fmt_lat(v, _):
    return f"{abs(v):.0f}°{'N' if v >= 0 else 'S'}"


def fmt_lon(v, _):
    return f"{abs(v):.0f}°{'W' if v < 0 else 'E'}" if v else "0°"


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(R.REGION_SETS), default="ocean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    regions = R.REGION_SETS[args.set]
    out = args.out or os.path.join(HERE, R.SET_DIR[args.set])
    os.makedirs(out, exist_ok=True)
    on_land = args.set == "land"
    surf = "land" if on_land else "ocean"

    C = Cache(os.path.join(HERE, "ar_v2_merged.npz"))
    ma = C.get("all_ardays_t") / float(NYEAR)
    wn = C.get("all_ardays") / float(NYEAR)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08,
                        wspace=0.16, hspace=0.30)

    for i, reg in enumerate(regions):
        name, la0, la1, lo0, lo1 = reg
        box = C.box(la0, la1, lo0, lo1)
        isl = C.land(box)
        prim = box & (isl if on_land else ~isl)       # the surface this set is about
        rows = np.where(box.any(1))[0]
        cols = np.where(box.any(0))[0]
        w = C.w

        # ---- latitude profile, all pixels
        ax = axes[0, i]
        sub = np.ix_(rows, cols)
        ax.plot(ma[sub].mean(1), C.lat[rows], color=C_MARS, lw=2.2)
        ax.plot(wn[sub].mean(1), C.lat[rows], color=C_WNET, lw=2.2)
        ax.set_title(f"({TAGS[i]}) {name}", fontsize=FS_TITLE, color=INK, loc="left", pad=7)
        ax.set_xlabel("Mean AR days per year", fontsize=FS_LAB, color=INK)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_lat))
        ax.set_ylim(la0, la1)
        ax.set_xlim(0, 160)

        # ---- longitude profile over the primary surface; sparse columns are dropped
        ax = axes[1, i]
        pm = prim[sub]
        ww = w[sub] * pm
        ok = pm.mean(0) >= 0.3
        prof = lambda f: np.where(ok, (f[sub] * ww).sum(0) / np.maximum(ww.sum(0), 1e-9), np.nan)  # noqa: E731
        ax.plot(C.lon[cols], prof(ma), color=C_MARS, lw=2.2)
        ax.plot(C.lon[cols], prof(wn), color=C_WNET, lw=2.2)
        ax.set_title(f"({TAGS[i + 3]}) {name}", fontsize=FS_TITLE, color=INK, loc="left", pad=7)
        ax.set_xlabel("Longitude", fontsize=FS_LAB, color=INK)
        # explicit whole-degree ticks: automatic 2.5 deg ticks were rounded to 122, 118, ...
        ax.xaxis.set_major_locator(MultipleLocator(5 if (lo1 - lo0) <= 30 else 10))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_lon))
        ax.set_xlim(lo0, lo1)
        ax.set_ylim(0, 130)

        for a in axes[:, i]:
            a.tick_params(labelsize=FS_TICK, colors=INK)
            a.grid(alpha=0.3)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)

        # ---- where does the region-mean gap come from
        tot = (w * box).sum()
        print(f"\n{name}:  MARS {(ma * w * box).sum() / tot:.1f}  "
              f"WNet {(wn * w * box).sum() / tot:.1f}  "
              f"gap {((ma - wn) * w * box).sum() / tot:+.1f} days/yr")
        third = (la1 - la0) / 3.0
        alat = np.abs(C.lat)[:, None] * np.ones_like(w)
        lo_a = min(abs(la0), abs(la1))
        lon2 = C.lon[None, :] * np.ones_like(w)
        parts = [("ocean" if on_land else "land", box & ~prim)]
        for k, lab in enumerate(("equatorward third", "central third", "poleward third")):
            band = (alat >= lo_a + k * third) & (alat <= lo_a + (k + 1) * third + 1e-9)
            parts.append((f"{surf}, {lab}", prim & band))
        # all three are west coasts: the coast is the east edge of an ocean box and
        # the west side of a land box
        if on_land:
            parts.append((f"{surf}, within 5 deg of west edge", prim & (lon2 <= lo0 + 5)))
            parts.append((f"{surf}, more than 10 deg inland", prim & (lon2 >= lo0 + 10)))
        else:
            parts.append((f"{surf}, within 10 deg of east edge", prim & (lon2 >= lo1 - 10)))
        for lab, m in parts:
            c = ((ma - wn) * w * m).sum() / tot
            a = (w * m).sum() / tot * 100
            print(f"   {lab:<36} area {a:4.0f} %   MARS {(ma*w*m).sum()/(w*m).sum():6.1f}  "
                  f"WNet {(wn*w*m).sum()/(w*m).sum():6.1f}   contributes {c:+5.1f} days")
        rat = []
        for s in SEASONS:
            p, t = C.get(f"sea_{s}_pred"), C.get(f"sea_{s}_targ")
            rat.append(f"{s} {(p * w * box).sum() / (t * w * box).sum():.2f}")
        print("   WNet / MARS AR area by season:  " + "   ".join(rat))

    axes[0, 0].set_ylabel("Latitude", fontsize=FS_LAB, color=INK)
    axes[1, 0].set_ylabel("Mean AR days per year", fontsize=FS_LAB, color=INK)
    fig.legend(handles=[Line2D([], [], color=C_MARS, lw=2.2, label="MARS"),
                        Line2D([], [], color=C_WNET, lw=2.2, label="WNet")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=FS_LEG, labelcolor=INK, columnspacing=2.5)
    save(fig, os.path.join(out, "ar_regional_gap_profiles"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
