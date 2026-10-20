#!/usr/bin/env python3
"""
plot_tracks_3col.py — old WNet, new WNet and an independent benchmark, side by side.

  --kind tc    old WNet | new WNet | IBTrACS         (observed storm positions)
  --kind etc   old WNet | new WNet | TempestExtremes (physics tracker, CONUS box)

Tracks are drawn as lines with a marker at genesis, so both the number of systems
and where they run are visible at a glance. Panel headers carry the track count,
which is usually the first thing that differs.

A note on the benchmarks. IBTrACS is observational and global, so the TC panel is a
fair three-way comparison. The TempestExtremes ETC files are CONUS-only
(lon -135 to -55, lat 10 to 55), so for ETC the benchmark column covers a much
smaller domain than the two WNet columns; the plot restricts every ETC column to
that box so the three are looking at the same region.
"""
import os
import csv
import glob
import argparse
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
OLD = os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_finalized")
NEW = os.path.join(NB, "WNet_parallel_global_code", "ARs_Hurricanes_ETCs_MCS",
                   "Phase2_detections", "tracks")
SD = "/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/StormType-Data"
IBTRACS = os.path.join(SD, "TCs", "ibtracs", "IBTrACS.ALL.v04r01.nc")
TE_ETC = os.path.join(SD, "ETCs", "TE")

TEST_YEARS = [1983, 1994, 2003, 2012, 2021]
ETC_BOX = dict(lon=(-135, -55), lat=(10, 55))

C_OLD, C_NEW, C_REF = "#eb6834", "#1baf7a", "#2a78d6"
INK, INK2 = "#131a1f", "#55636b"


# ── parsers ──────────────────────────────────────────────────────────────────
def parse_hurricane_txt(path):
    """CONUS_hurricane_tracks_detailed_YYYY.txt -> list of [(lat, lon), ...]."""
    tracks, cur = [], None
    if not os.path.exists(path):
        return tracks
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("TRACK "):
                if cur:
                    tracks.append(cur)
                cur = []
                continue
            m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
            if m and cur is not None:
                cur.append((float(m.group(1)), float(m.group(2))))
    if cur:
        tracks.append(cur)
    return [t for t in tracks if len(t) >= 2]


def parse_etc_txt(path):
    """ETC_tracks_YYYY_meteorological*.txt -> list of [(lat, lon), ...]."""
    if not os.path.exists(path):
        return []
    byid = defaultdict(list)
    with open(path, errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                tid = int(p[0])
                lon = float(p[2])
                lat = float(p[3])
            except ValueError:
                continue
            byid[tid].append((lat, lon))
    return [v for v in byid.values() if len(v) >= 2]


def parse_te_etc(year):
    """TempestExtremes ETC trajectories (CONUS) -> list of [(lat, lon), ...]."""
    cands = sorted(glob.glob(os.path.join(
        TE_ETC, f"ERA5_trajectories_ETC_conus_{year}_6hr_csv*.txt")))
    if not cands:
        return []
    byid = defaultdict(list)
    with open(cands[-1], errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(row) < 9:
                continue
            try:
                tid = int(row[0])
                lon = float(row[7])
                lat = float(row[8])
            except ValueError:
                continue
            if lon > 180:
                lon -= 360
            byid[tid].append((lat, lon))
    return [v for v in byid.values() if len(v) >= 2]


def ibtracs_tracks(years):
    """IBTrACS storm tracks for the given years -> list of [(lat, lon), ...]."""
    from netCDF4 import Dataset
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    wind = ds.variables["usa_wind"][:]
    ds.close()
    out = []
    for si in np.where(np.isin(season, years))[0]:
        la = np.ma.filled(lat[si], np.nan).astype(float)
        lo = np.ma.filled(lon[si], np.nan).astype(float)
        # usa_wind is int16, so cast before filling with NaN
        wd = np.ma.filled(wind[si].astype(float), np.nan)
        ok = np.isfinite(la) & np.isfinite(lo) & (np.nan_to_num(wd) >= 34.0)
        if ok.sum() < 2:
            continue
        la, lo = la[ok], lo[ok]
        lo = np.where(lo > 180, lo - 360, lo)
        out.append(list(zip(la, lo)))
    return out


# ── drawing ──────────────────────────────────────────────────────────────────
def clip_to_box(tracks, box):
    """Keep only tracks with at least one point inside the comparison box.

    Needed for ETC: the TempestExtremes files are CONUS-only, so counting global
    WNet tracks against a regional benchmark would compare different domains and
    the numbers would be meaningless.
    """
    if not box:
        return tracks
    lo0, lo1 = box["lon"]
    la0, la1 = box["lat"]
    out = []
    for t in tracks:
        if any(la0 <= la <= la1 and lo0 <= lo <= lo1 for la, lo in t):
            out.append(t)
    return out


def draw(ax, tracks, colour, title, box=None):
    tracks = clip_to_box(tracks, box)
    if box:
        ax.set_extent([box["lon"][0], box["lon"][1], box["lat"][0], box["lat"][1]],
                      crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2f2ef", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    ax.coastlines(lw=0.4, color="0.45", zorder=1)
    ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="0.7", zorder=1)
    n = 0
    for t in tracks:
        la = [p[0] for p in t]
        lo = [p[1] for p in t]
        # split where the track crosses the dateline so it does not streak across
        seg_la, seg_lo = [la[0]], [lo[0]]
        for i in range(1, len(la)):
            if abs(lo[i] - lo[i - 1]) > 180:
                ax.plot(seg_lo, seg_la, color=colour, lw=0.55, alpha=0.55,
                        transform=ccrs.PlateCarree(), zorder=2)
                seg_la, seg_lo = [], []
            seg_la.append(la[i])
            seg_lo.append(lo[i])
        if len(seg_la) > 1:
            ax.plot(seg_lo, seg_la, color=colour, lw=0.55, alpha=0.55,
                    transform=ccrs.PlateCarree(), zorder=2)
        ax.plot(lo[0], la[0], "o", color=colour, ms=1.6, alpha=0.8,
                transform=ccrs.PlateCarree(), zorder=3)
        n += 1
    gl = ax.gridlines(draw_labels=True, lw=0.25, color="gray", alpha=0.3, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(f"{title}\n{n} tracks", fontsize=11, color=INK, pad=5)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tc", "etc"], required=True)
    ap.add_argument("--years", nargs="+", type=int, default=TEST_YEARS)
    ap.add_argument("--outdir", default="phase4_figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    years = args.years
    if args.kind == "tc":
        old = [t for y in years for t in parse_hurricane_txt(os.path.join(
            OLD, "Hurricanes_WNet_final", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))]
        new = [t for y in years for t in parse_hurricane_txt(os.path.join(
            NEW, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))]
        ref = ibtracs_tracks(years)
        ref_name = "IBTrACS (observed)"
        box = None
    else:
        old = [t for y in years for t in parse_etc_txt(os.path.join(
            OLD, "ETCs_WNet_final", f"ETC_tracks_{y}_meteorological.txt"))]
        new = [t for y in years for t in parse_etc_txt(os.path.join(
            NEW, "etc", f"ETC_tracks_{y}_meteorological_v2.txt"))]
        ref = [t for y in years for t in parse_te_etc(y)]
        ref_name = "TempestExtremes"
        box = ETC_BOX

    have = [y for y in years
            if (args.kind == "tc"
                and os.path.exists(os.path.join(NEW, "hurricane", f"CONUS_year_{y}",
                                                f"CONUS_hurricane_tracks_detailed_{y}.txt")))
            or (args.kind == "etc"
                and os.path.exists(os.path.join(NEW, "etc",
                                                f"ETC_tracks_{y}_meteorological_v2.txt")))]
    print(f"[{args.kind}] years with new tracks: {have}")
    print(f"  old={len(old)}  new={len(new)}  {ref_name}={len(ref)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.985, top=0.84, bottom=0.06, wspace=0.09)
    counts = [
        draw(axes[0], old, C_OLD, "WNet old", box),
        draw(axes[1], new, C_NEW, "WNet new (Phase 2)", box),
        draw(axes[2], ref, C_REF, ref_name, box),
    ]
    label = "Tropical cyclone" if args.kind == "tc" else "Extratropical cyclone"
    sub = f"held-out years {', '.join(str(y) for y in have)}"
    if box:
        sub += f"  |  domain restricted to the TempestExtremes box"
    fig.suptitle(f"{label} tracks  —  {sub}", fontsize=13, color=INK,
                 x=0.03, ha="left", y=0.975)

    out = os.path.join(args.outdir, f"tracks_3col_{args.kind}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    plt.close(fig)

    print(f"\n  track counts: old={counts[0]}  new={counts[1]}  {ref_name}={counts[2]}")
    for nm, tr in (("old", old), ("new", new), (ref_name, ref)):
        if tr:
            lens = [len(t) for t in tr]
            print(f"    {nm:<20} n={len(tr):<6} median points/track={int(np.median(lens))}")


if __name__ == "__main__":
    main()
