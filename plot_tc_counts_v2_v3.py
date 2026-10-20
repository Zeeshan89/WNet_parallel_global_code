#!/usr/bin/env python3
"""
plot_tc_counts_v2_v3.py — tropical cyclones detected per year by the v2 and the v3 WNet
products, side by side, to see whether v3 is missing storms.

Counting v2 against v3 alone would only say which of the two finds more. IBTrACS, the
observational archive, is therefore drawn in both panels as the number of storms that actually
occurred, so "missing" can be read against observations rather than against the other version.
IBTrACS storms are counted for the same window the tracker searches (a point within 45 degrees
of the equator) and at tropical-storm strength or above (maximum wind >= 34 kt), which is the
threshold the project's TC evaluation uses.

  python plot_tc_counts_v2_v3.py [--out <dir>]
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
CODE = os.path.join(NB, "WNet_parallel_global_code")
sys.path.insert(0, CODE)
V2 = os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_finalized_v", "Hurricanes_WNet_final")
V3 = os.path.join(CODE, "ARs_Hurricanes_ETCs_MCS", "Phase2", "H_final_e300", "tracks_v3_tc", "hurricane")
IBTRACS = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/StormType-Data/TCs/"
           "ibtracs/IBTrACS.ALL.v04r01.nc")
Y0, Y1 = 1980, 2024
INK = "#000000"


def wnet_counts(root):
    """{year: number of tracks} from the header line of each yearly detail file."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "CONUS_year_*", "*_detailed_*.txt"))):
        year = int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
        with open(p) as f:
            for line in f:
                if "Total number of tracks:" in line:
                    out[year] = int(line.split(":")[1].strip())
                    break
    return out


def ibtracs_counts(wind_kt=34.0, lat_max=45.0):
    """{year: storms reaching wind_kt with a point within lat_max of the equator}.

    Counted by the storm's first valid time, not by IBTrACS `season`, so the year means the
    same thing as it does for the WNet tracks (which are written per calendar year).
    """
    from netCDF4 import Dataset, num2date
    from collections import defaultdict
    counts = defaultdict(int)
    with Dataset(IBTRACS) as ds:
        # usa_wind is int16 in this file, so it must be cast before a NaN fill value is used
        lat = np.ma.filled(ds.variables["lat"][:].astype("f8"), np.nan)
        wind = np.ma.filled(ds.variables["usa_wind"][:].astype("f8"), np.nan)
        tv = ds.variables["time"]
        traw = np.ma.filled(tv[:], np.nan)
        units, cal = tv.units, getattr(tv, "calendar", "standard")
    for i in range(lat.shape[0]):
        ok = np.isfinite(traw[i])
        if not ok.any():
            continue
        if not (np.nanmax(wind[i]) >= wind_kt):
            continue
        if not (np.nanmin(np.abs(lat[i])) <= lat_max):
            continue
        t0 = num2date(traw[i][ok][0], units, cal, only_use_cftime_datetimes=False,
                      only_use_python_datetimes=True)
        counts[t0.year] += 1
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CODE, "phase2_figures"))
    args = ap.parse_args()
    years = np.arange(Y0, Y1 + 1)
    v2, v3 = wnet_counts(V2), wnet_counts(V3)
    ib = ibtracs_counts()
    a = np.array([v2.get(y, np.nan) for y in years], float)
    b = np.array([v3.get(y, np.nan) for y in years], float)
    o = np.array([ib.get(y, np.nan) for y in years], float)

    hdr = f"{'year':>6}{'v2':>7}{'v3':>7}{'v3-v2':>8}{'IBTrACS':>9}{'v2-IB':>7}{'v3-IB':>7}"
    lines = [hdr]
    for k, y in enumerate(years):
        lines.append(f"{y:>6}{a[k]:>7.0f}{b[k]:>7.0f}{b[k]-a[k]:>8.0f}{o[k]:>9.0f}"
                     f"{a[k]-o[k]:>7.0f}{b[k]-o[k]:>7.0f}")
    lines.append(f"{'mean':>6}{np.nanmean(a):>7.1f}{np.nanmean(b):>7.1f}{np.nanmean(b-a):>8.1f}"
                 f"{np.nanmean(o):>9.1f}{np.nanmean(a-o):>7.1f}{np.nanmean(b-o):>7.1f}")
    lines.append(f"{'total':>6}{np.nansum(a):>7.0f}{np.nansum(b):>7.0f}{np.nansum(b-a):>8.0f}"
                 f"{np.nansum(o):>9.0f}")
    print("\n".join(lines))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tc_counts_v2_vs_v3.txt"), "w") as f:
        f.write("Tropical cyclones per year: WNet v2, WNet v3 and IBTrACS (>= 34 kt, a point within\n"
                "45 deg of the equator). v2 = Global_ARs_Hurr_ETCs_MCS_finalized_v/Hurricanes_WNet_final,\n"
                "v3 = Phase2/H_final_e300/tracks_v3_tc/hurricane.\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(args.out, "tc_counts_v2_vs_v3.csv"), "w") as f:
        f.write("year,v2,v3,v3_minus_v2,ibtracs,v2_minus_ibtracs,v3_minus_ibtracs\n")
        for k, y in enumerate(years):
            f.write(f"{y},{a[k]:.0f},{b[k]:.0f},{b[k]-a[k]:.0f},{o[k]:.0f},"
                    f"{a[k]-o[k]:.0f},{b[k]-o[k]:.0f}\n")
    for lab, v in (("v2", a), ("v3", b)):
        r = np.corrcoef(v[np.isfinite(v) & np.isfinite(o)], o[np.isfinite(v) & np.isfinite(o)])[0, 1]
        print(f"  {lab}: interannual correlation with IBTrACS r = {r:.2f}; "
              f"years below IBTrACS: {int(np.nansum(v < o))}/{len(years)}")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.14, wspace=0.06)
    for ax, (lab, v, col) in zip(axes, (("(a) WNet v2", a, "#4c72b0"), ("(b) WNet v3", b, "#c44e52"))):
        ax.bar(years, v, color=col, width=0.75, label="WNet tracks")
        ax.plot(years, o, color=INK, lw=1.6, marker="o", ms=3.2, label="IBTrACS (>= 34 kt)")
        ax.set_title(f"{lab}   mean {np.nanmean(v):.0f}/yr", fontsize=12, color=INK, loc="left", pad=6)
        ax.set_xlabel("year", fontsize=11, color=INK)
        ax.set_xlim(Y0 - 1, Y1 + 1)
        ax.grid(alpha=0.3, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=9.5, colors=INK)
    axes[0].set_ylabel("tropical cyclones per year", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(args.out, f"tc_counts_v2_vs_v3.{ext}")
        fig.savefig(p, dpi=250, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
