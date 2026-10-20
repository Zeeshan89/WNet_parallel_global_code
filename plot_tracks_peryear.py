#!/usr/bin/env python3
"""
plot_tracks_peryear.py — one row per year, three columns, with colours that mean
the same thing in every panel.

  --kind tc    old WNet | new WNet | IBTrACS         (observed storm positions)
  --kind etc   old WNet | new WNet | TempestExtremes (physics tracker, CONUS box)

Colour convention
-----------------
Every benchmark track in a year gets its own colour. A model track is then matched
to a benchmark track and inherits that colour, so the same real storm carries the
same colour in all three panels and can be followed across them by eye. A model
track that matches no benchmark storm is drawn in light grey, which makes the
false alarms immediately visible as the grey background against the coloured hits.

Matching rule: a model point and a benchmark point correspond if they share the
same 6-hourly timestamp (within 3 hours) and lie within --radius-km of each other.
A model track is assigned to whichever benchmark track it corresponds to most
often, and only if it corresponds at least twice, so a single chance crossing does
not colour a spurious track.
"""
import os
import re
import csv
import glob
import argparse
from collections import defaultdict
from datetime import datetime, timedelta

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
GREY = "#b9bfc3"
INK = "#131a1f"
EARTH_R = 6371.0


def palette(n):
    """A long, repeating qualitative palette. Order is fixed, so a given benchmark
    index always gets the same colour in every panel and every year."""
    base = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    return [base[i % len(base)] for i in range(max(n, 1))]


# ── parsers, all returning list of tracks, each a list of (datetime, lat, lon) ──
def parse_hurricane_txt(path):
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
            m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                         line)
            if m and cur is not None:
                cur.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M"),
                            float(m.group(2)), float(m.group(3))))
    if cur:
        tracks.append(cur)
    return [t for t in tracks if len(t) >= 2]


def parse_etc_txt(path):
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
                t = datetime.strptime(p[1], "%Y-%m-%d_%H:%M:%S")
                byid[tid].append((t, float(p[3]), float(p[2])))
            except (ValueError, IndexError):
                continue
    return [v for v in byid.values() if len(v) >= 2]


def parse_te_etc(year):
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
                t = datetime(int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                lon = float(row[7])
                lat = float(row[8])
            except (ValueError, IndexError):
                continue
            if lon > 180:
                lon -= 360
            byid[int(row[0])].append((t, lat, lon))
    return [v for v in byid.values() if len(v) >= 2]


def ibtracs_tracks(year, wind_kt=34.0, wind_source=None, year_key=None):
    """IBTrACS storms to score a model's `year` track file against.

    wind_source: "usa" reproduces the original behaviour (usa_wind only); "any"
    takes the maximum over every agency that reported. Default follows the
    IBTRACS_WIND_SOURCE environment variable, then "usa", so existing figures and
    scores reproduce unchanged unless a caller opts in.

    Why "any" matters: usa_wind is US-agency-reported and its coverage outside the
    Atlantic and E Pacific is patchy — May 1995 has 156 storm positions worldwide
    and none with a finite usa_wind >= 34 kt. Scoring a model that was TRAINED on
    all-agency intensities against a usa_wind-only reference would count genuine
    W Pacific / N Indian detections as false alarms.

    year_key: "calendar" (default) selects storms by the calendar year their
    qualifying positions actually fall in; "season" restores the old
    `season == year` behaviour. Default follows IBTRACS_YEAR_KEY.

    Why "calendar" matters: IBTrACS `season` is a season LABEL, not a calendar
    year. The Southern Hemisphere 1983/84 season is labelled 1984, so a storm in
    Nov/Dec 1983 carries season 1984 and `season == year` scored it against the
    model's 1984 track file — a file in which it has no points at all, because the
    model writes one file per CALENDAR year. Measured over the five test years,
    that mis-pairing cost 27 storms, of which 19 are genuinely detected: they were
    charged twice, once as a miss and once as a false alarm. Pooled POD 0.662 ->
    0.700 and FAR 0.101 -> 0.035 from this change alone, with no retraining and no
    tracker change. phase4_build_ibtracs_label.py:76 already applies the same
    reasoning when BUILDING the training target, so until now the model was
    trained on the right storms and scored against the wrong year.
    """
    from netCDF4 import Dataset, num2date
    if wind_source is None:
        wind_source = os.environ.get("IBTRACS_WIND_SOURCE", "usa")
    if year_key is None:
        year_key = os.environ.get("IBTRACS_YEAR_KEY", "calendar")
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    if year_key == "season":
        idx = np.where(season == year)[0]
    else:
        # Over-select the neighbouring seasons, then keep only the positions that
        # fall inside the calendar year. A season never spans more than one New
        # Year, so +-1 is sufficient.
        idx = np.where((season >= year - 1) & (season <= year + 1))[0]
    # Read the covering block of storms once and select inside it. Identical values and
    # masks, but netCDF4 reads a fancy index list storm by storm (18 s for one variable of
    # one year; 79 s per call in total) against 0.06 s for the block.
    lo = int(idx.min()) if idx.size else 0
    hi = int(idx.max()) + 1 if idx.size else 0

    def _take(var):
        return var[lo:hi][idx - lo]

    lat = _take(ds.variables["lat"])
    lon = _take(ds.variables["lon"])
    if wind_source == "any":
        cols = [v for v in ds.variables
                if v.endswith("_wind") and ds.variables[v].ndim == 2]
        st = [np.ma.filled(_take(ds.variables[v]).astype(float), np.nan) for v in cols]
        with np.errstate(all="ignore"):
            wind = np.nanmax(np.dstack(st), axis=2) if st else \
                np.ma.filled(_take(ds.variables["usa_wind"]).astype(float), np.nan)
    else:
        wind = _take(ds.variables["usa_wind"])
    traw = _take(ds.variables["time"])
    units = ds.variables["time"].units
    cal = getattr(ds.variables["time"], "calendar", "standard")
    ds.close()
    out = []
    for i in range(len(idx)):
        la = np.ma.filled(lat[i].astype(float), np.nan)
        lo = np.ma.filled(lon[i].astype(float), np.nan)
        wd = np.asarray(wind[i], dtype=float) if wind_source == "any" \
            else np.ma.filled(wind[i].astype(float), np.nan)
        tt = np.ma.filled(traw[i].astype(float), np.nan)
        ok = (np.isfinite(la) & np.isfinite(lo) & np.isfinite(tt)
              & (np.nan_to_num(wd) >= wind_kt))
        if ok.sum() < 2:
            continue
        times = num2date(tt[ok], units, cal,
                         only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)
        lo_ok = np.where(lo[ok] > 180, lo[ok] - 360, lo[ok])
        pts = [(t.replace(minute=0, second=0, microsecond=0), a, b)
               for t, a, b in zip(times, la[ok], lo_ok)]
        if year_key != "season":
            pts = [p for p in pts if p[0].year == year]
            # Below min_hits the storm cannot be matched at all, so keeping it
            # would only inflate the denominator with unmatchable references.
            if len(pts) < 2:
                continue
        out.append(pts)
    return out


# ── matching ─────────────────────────────────────────────────────────────────
def haversine(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    dp = p2 - p1
    dl = np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_tracks(model, ref, radius_km, min_hits=2, tol_h=3):
    """For each model track, the index of the benchmark track it overlaps most."""
    stamps = defaultdict(list)          # rounded hour -> [(ref_idx, lat, lon)]
    for j, rt in enumerate(ref):
        for (t, la, lo) in rt:
            key = t.replace(minute=0, second=0, microsecond=0)
            stamps[key].append((j, la, lo))

    assign = []
    for mt in model:
        votes = defaultdict(int)
        for (t, la, lo) in mt:
            t0 = t.replace(minute=0, second=0, microsecond=0)
            cands = []
            for dh in range(-tol_h, tol_h + 1):
                cands.extend(stamps.get(t0 + timedelta(hours=dh), []))
            for (j, rla, rlo) in cands:
                if haversine(la, lo, rla, rlo) <= radius_km:
                    votes[j] += 1
        if votes:
            j, v = max(votes.items(), key=lambda kv: kv[1])
            assign.append(j if v >= min_hits else None)
        else:
            assign.append(None)
    return assign


# ── drawing ──────────────────────────────────────────────────────────────────
def clip_to_box(tracks, box):
    if not box:
        return tracks, list(range(len(tracks)))
    lo0, lo1 = box["lon"]
    la0, la1 = box["lat"]
    keep = [i for i, t in enumerate(tracks)
            if any(la0 <= la <= la1 and lo0 <= lo <= lo1 for _, la, lo in t)]
    return [tracks[i] for i in keep], keep


def draw(ax, tracks, colours, title, box):
    if box:
        ax.set_extent([box["lon"][0], box["lon"][1], box["lat"][0], box["lat"][1]],
                      crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-180, 180, -55, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    ax.coastlines(lw=0.35, color="0.5", zorder=1)
    n_col = 0
    # grey first so coloured (matched) tracks sit on top and stay readable
    order = sorted(range(len(tracks)), key=lambda i: colours[i] is not None)
    for i in order:
        t, c = tracks[i], colours[i]
        la = [p[1] for p in t]
        lo = [p[2] for p in t]
        col = GREY if c is None else c
        lw, al, z = (0.5, 0.45, 2) if c is None else (1.0, 0.9, 3)
        if c is not None:
            n_col += 1
        sla, slo = [la[0]], [lo[0]]
        for k in range(1, len(la)):
            if abs(lo[k] - lo[k - 1]) > 180:
                ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                        transform=ccrs.PlateCarree(), zorder=z)
                sla, slo = [], []
            sla.append(la[k])
            slo.append(lo[k])
        if len(sla) > 1:
            ax.plot(slo, sla, color=col, lw=lw, alpha=al,
                    transform=ccrs.PlateCarree(), zorder=z)
        ax.plot(lo[0], la[0], "o", color=col, ms=2.2 if c is not None else 1.2,
                alpha=al, transform=ccrs.PlateCarree(), zorder=z + 1)
    gl = ax.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.25, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    ax.set_title(title, fontsize=9.5, color=INK, pad=4)
    return n_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tc", "etc"], required=True)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--outdir", default="phase4_figures")
    ap.add_argument("--track-dir", default=NEW,
                    help="directory holding the new model's tracks; defaults to "
                         "the Phase2_detections set so existing behaviour is "
                         "unchanged. Point it at Phase2_C200/tracks to plot the "
                         "C_full200 tracks instead.")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename, so runs from different "
                         "track directories do not overwrite each other")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def load(y):
        if args.kind == "tc":
            o = parse_hurricane_txt(os.path.join(
                OLD, "Hurricanes_WNet_final", f"CONUS_year_{y}",
                f"CONUS_hurricane_tracks_detailed_{y}.txt"))
            n = parse_hurricane_txt(os.path.join(
                args.track_dir, "hurricane", f"CONUS_year_{y}",
                f"CONUS_hurricane_tracks_detailed_{y}.txt"))
            r = ibtracs_tracks(y)
            return o, n, r, "IBTrACS (observed)", None
        o = parse_etc_txt(os.path.join(
            OLD, "ETCs_WNet_final", f"ETC_tracks_{y}_meteorological.txt"))
        n = parse_etc_txt(os.path.join(
            args.track_dir, "etc", f"ETC_tracks_{y}_meteorological_v2.txt"))
        r = parse_te_etc(y)
        return o, n, r, "TempestExtremes", ETC_BOX

    years = args.years or TEST_YEARS
    years = [y for y in years if load(y)[1]]          # only years the new model has
    if not years:
        raise SystemExit("no years with new-model tracks")
    print(f"[{args.kind}] years: {years}")

    fig, axes = plt.subplots(len(years), 3, figsize=(19, 3.05 * len(years)),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.02,
                        hspace=0.22, wspace=0.07)

    summary = []
    for r, y in enumerate(years):
        old, new, ref, refname, box = load(y)
        old, _ = clip_to_box(old, box)
        new, _ = clip_to_box(new, box)
        ref, _ = clip_to_box(ref, box)

        cols = palette(len(ref))
        a_old = match_tracks(old, ref, args.radius_km)
        a_new = match_tracks(new, ref, args.radius_km)

        c_old = [None if j is None else cols[j] for j in a_old]
        c_new = [None if j is None else cols[j] for j in a_new]
        c_ref = [cols[j] for j in range(len(ref))]

        m_old = draw(axes[r][0], old, c_old,
                     f"WNet old — {len(old)} tracks, {sum(c is not None for c in c_old)} matched", box)
        m_new = draw(axes[r][1], new, c_new,
                     f"WNet new — {len(new)} tracks, {sum(c is not None for c in c_new)} matched", box)
        draw(axes[r][2], ref, c_ref, f"{refname} — {len(ref)} tracks", box)

        # how many benchmark storms were found by each model
        hit_old = len({j for j in a_old if j is not None})
        hit_new = len({j for j in a_new if j is not None})
        axes[r][0].text(-0.055, 0.5, str(y), transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold", color=INK)
        summary.append((y, len(old), len(new), len(ref), m_old, m_new,
                        hit_old, hit_new))

    label = "Tropical cyclone" if args.kind == "tc" else "Extratropical cyclone"
    note = ("colour = matched benchmark storm (same colour across all three panels); "
            "grey = no benchmark match")
    fig.suptitle(f"{label} tracks by year  —  {note}", fontsize=12,
                 color=INK, x=0.045, ha="left", y=0.985)

    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.outdir, f"tracks_peryear_{args.kind}{suffix}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    plt.close(fig)

    print(f"\n  {'year':<6}{'old':>7}{'new':>7}{'bench':>7}"
          f"{'old match':>11}{'new match':>11}{'bench found old':>17}{'bench found new':>17}")
    print("  " + "-" * 84)
    for (y, no, nn, nr, mo, mn, ho, hn) in summary:
        print(f"  {y:<6}{no:>7}{nn:>7}{nr:>7}{mo:>11}{mn:>11}"
              f"{f'{ho}/{nr}':>17}{f'{hn}/{nr}':>17}")


if __name__ == "__main__":
    main()
