#!/usr/bin/env python3
"""
phase4_tc_miss_taxonomy.py — why does the IBTrACS-target model miss the storms it
misses?

The headline number (POD 0.641 in the TempestExtremes box, 0.593 global) is a
single figure covering five failure modes that have nothing to do with each other
and completely different fixes.  This script assigns every missed IBTrACS storm to
exactly one of them, walking the pipeline in order so the FIRST stage that could
have killed the storm is the one it is charged to:

  5 bookkeeping     the storm could never have been matched by construction —
                    its qualifying points lie outside the calendar year whose
                    track file we read (IBTrACS `season` is a season LABEL, so a
                    S-hemisphere storm in Nov/Dec of year Y carries season Y+1),
                    or outside the tracker's +-45 deg latitude window, or fewer
                    than `min_hits` points survive both.
  1 segmentation    the network never fired: no detected pixel within the match
                    radius at any of the storm's own timestamps.
  2 sub-threshold   pixels exist but no tracker CENTRE does — the connected blob
                    fails the area gate (TC_MIN_AREA_PX / TC_MAX_AREA_PX) — or
                    centres exist but too few / too briefly to clear the hard
                    >=5 points AND >=48 h gate inside create_hurricane_tracks.
  4 fragmentation   two or more candidate tracks cover the storm, each too short
                    to pass, where one continuous track would have passed.
  3 tracker reject  a single candidate covered the storm and validate_hurricane_
                    track killed it; the criterion is named.

Notes on the instrumentation
----------------------------
`HurricaneTracker.REJECT` only counts candidates that already cleared
close_track's own >=5-points and >=48-h gates, so it cannot see the two commonest
kills.  TC_CAPTURE_CANDIDATES=1 (added alongside REJECT) records EVERY closed
candidate with the criterion that ended it; this script sets it.

The exported detection NetCDFs are BINARY (phase4_export_detections.py thresholds
at 0.5 before writing), so `max_intensity` is 1.0 everywhere and the TC_MIN_
INTENSITY / TC_FRAME_INTENSITY gates are no-ops.  That also means "fired but below
0.5" is NOT observable here — it needs a probability re-export.  Bucket 1 is
therefore "did not fire AT 0.5", and the unrecoverable count reported at the end
is an upper bound on what a threshold change could recover.

Usage
  phase4_tc_miss_taxonomy.py --years 1983 2003 2021
  phase4_tc_miss_taxonomy.py --years 1983 1994 2003 2012 2021 --te-domain
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
SD = "/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/StormType-Data"
IBTRACS = os.path.join(SD, "TCs", "ibtracs", "IBTrACS.ALL.v04r01.nc")

DET_DIR = os.environ.get("TC_DET_DIR") or os.path.join(NB, "Global_TC_IBTrACS_final", "Hurricanes_WNet_final")
TRACK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ARs_Hurricanes_ETCs_MCS", "Phase2_E_ibtracs", "tracks_final")

TE_BOX = dict(lon=(-135.0, -55.0), lat=(5.0, 60.0))
EARTH_R = 6371.0
LAT_LIM = 45.0          # HurricaneTracker.region_bounds
MIN_AREA_PX = int(os.environ.get("TC_MIN_AREA_PX", "3"))
MAX_AREA_PX = int(os.environ.get("TC_MAX_AREA_PX", "2000"))
MIN_BLOB_AREA = 3.5     # HurricaneTracker.min_hurricane_area


def haversine(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lo2) - np.asarray(lo1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── reference, with the metadata the plain loader throws away ────────────────
# IBTrACS `nature` codes: TS tropical, DS disturbance, SS subtropical,
# ET extratropical, MX mixture, NR not reported.
#   "all"     every point, whatever its nature  -- the historical behaviour, and
#             the one every published POD in this project was computed with
#   "label"   drop ET only. EXACTLY symmetric with the training target:
#             phase4_build_ibtracs_label.py:62 uses exclude_nature=("ET",), so a
#             point the label never painted is a point the model was trained NOT
#             to fire on. Scoring those as TC misses charges the model for
#             obeying its own target, and this project has a separate ETC class
#             that owns extratropical systems.
#   "tc_only" keep TS/DS only -- the strict reading of "tropical", also dropping
#             subtropical and mixed points. Stricter than the label, so it is a
#             sensitivity bound rather than a symmetric choice.
NATURE_SETS = {
    "all": None,
    "label": ("ET",),
    "tc_only": ("ET", "SS", "MX", "NR"),
}


def ibtracs_storms(year, wind_kt=34.0, wind_source="any", nature=None):
    """Every IBTrACS storm of one SEASON, with basin / name / per-point wind.

    Selection is identical to plot_tracks_peryear.ibtracs_tracks (season == year,
    >= wind_kt, >= 2 qualifying points) so the hit/miss counts here reconcile with
    the published POD; only the returned metadata is richer.

    nature: which nature codes to KEEP, see NATURE_SETS. Default follows the
    IBTRACS_NATURE environment variable, then "all", so every existing number
    reproduces unchanged unless a caller opts in.
    """
    from netCDF4 import Dataset, num2date
    if nature is None:
        nature = os.environ.get("IBTRACS_NATURE", "all")
    if nature not in NATURE_SETS:
        raise SystemExit(f"IBTRACS_NATURE must be one of {sorted(NATURE_SETS)}")
    drop = NATURE_SETS[nature]
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    idx = np.where(season == year)[0]
    lat = ds.variables["lat"][idx]
    lon = ds.variables["lon"][idx]
    if wind_source == "any":
        cols = [v for v in ds.variables
                if v.endswith("_wind") and ds.variables[v].ndim == 2]
        st = [np.ma.filled(ds.variables[v][idx].astype(float), np.nan) for v in cols]
        with np.errstate(all="ignore"):
            wind = np.nanmax(np.dstack(st), axis=2)
    else:
        wind = np.ma.filled(ds.variables["usa_wind"][idx].astype(float), np.nan)
    traw = ds.variables["time"][idx]
    units = ds.variables["time"].units
    cal = getattr(ds.variables["time"], "calendar", "standard")

    def _chars(var, i):
        raw = ds.variables[var][i]
        return b"".join(np.ma.filled(raw, b"").ravel()).decode(errors="ignore").strip()

    basin_raw = ds.variables["basin"][idx]
    nature_raw = ds.variables["nature"][idx] if "nature" in ds.variables else None
    sid_all = [_chars("sid", i) for i in idx]
    name_all = [_chars("name", i) for i in idx]
    ds.close()

    def _nat(k, j):
        if nature_raw is None:
            return ""
        return b"".join(np.ma.filled(nature_raw[k][j], b"")).decode(
            errors="ignore").strip()

    out = []
    for k, i in enumerate(idx):
        la = np.ma.filled(lat[k].astype(float), np.nan)
        lo = np.ma.filled(lon[k].astype(float), np.nan)
        wd = np.asarray(wind[k], dtype=float)
        tt = np.ma.filled(traw[k].astype(float), np.nan)
        ok = (np.isfinite(la) & np.isfinite(lo) & np.isfinite(tt)
              & (np.nan_to_num(wd) >= wind_kt))
        if drop is not None and ok.any():
            # Applied to the POINTS, not the storm: a storm that starts tropical
            # and later goes extratropical keeps its tropical half, which is the
            # half the label painted.
            ok = ok & np.array([(_nat(k, j) not in drop) if ok[j] else False
                                for j in range(len(ok))])
        if ok.sum() < 2:
            continue
        times = num2date(tt[ok], units, cal, only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)
        lo_ok = np.where(lo[ok] > 180, lo[ok] - 360, lo[ok])
        pts = [(t.replace(minute=0, second=0, microsecond=0), float(a), float(b),
                float(w))
               for t, a, b, w in zip(times, la[ok], lo_ok, wd[ok])]
        bs = b"".join(np.ma.filled(basin_raw[k][ok][0], b"")).decode(
            errors="ignore").strip() if ok.sum() else "??"
        nats = [_nat(k, j) for j in np.where(ok)[0]]
        out.append({
            "sid": sid_all[k], "name": name_all[k] or "UNNAMED",
            "season": year, "basin": bs or "??",
            "points": pts,
            "peak_kt": float(np.nanmax(wd[ok])),
            # fraction of the KEPT points that the training label would not have
            # painted; non-zero only when nature == "all"
            "et_frac": (sum(1 for n in nats if n in ("ET", "SS", "MX", "NR"))
                        / len(nats)) if nats else 0.0,
        })
    return out


def parse_hurricane_txt(path):
    """Model tracks; same parser as plot_tracks_peryear."""
    import re
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


def match_tracks(model, ref, radius_km=300.0, min_hits=2, tol_h=3):
    """Index of the reference track each model track overlaps most (or None)."""
    stamps = defaultdict(list)
    for j, rt in enumerate(ref):
        for p in rt:
            stamps[p[0].replace(minute=0, second=0, microsecond=0)].append(
                (j, p[1], p[2]))
    assign = []
    for mt in model:
        votes = defaultdict(int)
        for p in mt:
            t0 = p[0].replace(minute=0, second=0, microsecond=0)
            for dh in range(-tol_h, tol_h + 1):
                for (j, rla, rlo) in stamps.get(t0 + timedelta(hours=dh), []):
                    if haversine(p[1], p[2], rla, rlo) <= radius_km:
                        votes[j] += 1
        if votes:
            j, v = max(votes.items(), key=lambda kv: kv[1])
            assign.append(j if v >= min_hits else None)
        else:
            assign.append(None)
    return assign


def in_box(pts, box):
    if not box:
        return True
    lo0, lo1 = box["lon"]
    la0, la1 = box["lat"]
    return any(la0 <= p[1] <= la1 and lo0 <= p[2] <= lo1 for p in pts)


# ── detection-field probe ────────────────────────────────────────────────────
def probe_detections(year, wanted, radius_km=300.0):
    """For every (timestamp, lat, lon) the caller asks about, what did the model
    actually emit there?

    wanted : dict[timestamp] -> list of (key, lat, lon)
    Returns dict[key] -> list of per-point records:
        {"t", "pix": bool, "min_pix_km": float, "area": int, "centre": bool}
      pix        a detected pixel within radius_km        (network fired at all)
      area       size in ERA5 pixels of the blob owning the nearest such pixel
      centre     that blob would have produced a tracker centre (area gate)

    Also returns the per-timestep census of ALL blobs, which is what turns "the
    area gate dropped it" from an inference into a measurement.
    """
    import xarray as xr
    from scipy.ndimage import label

    found = defaultdict(list)
    blob_areas = []
    files = sorted(glob.glob(os.path.join(
        DET_DIR, f"era5dara_{year:04d}*_Hurricane_detection.nc")))
    if not files:
        raise SystemExit(f"no detection files for {year} in {DET_DIR}")

    for fp in files:
        ds = xr.open_dataset(fp)
        times = [np.datetime64(t, "s").astype("datetime64[s]").item()
                 for t in ds.time.values]
        need = [i for i, t in enumerate(times) if t in wanted]
        if not need:
            ds.close()
            continue
        lats = ds.latitude.values
        lons = ds.longitude.values
        keep_lat = np.where((lats >= -LAT_LIM) & (lats <= LAT_LIM))[0]
        rlat = lats[keep_lat]
        var = ds["Hurricane_detection"]
        for i in need:
            m = var.values[i][keep_lat] > 0.5
            lab, n = label(m)
            if n:
                areas = np.bincount(lab.ravel())
                blob_areas.extend(areas[1:].tolist())
            for (key, plat, plon) in wanted[times[i]]:
                # window around the storm; radius_km <= 300 => <= 3 deg lat
                dlat = radius_km / 111.0 + 0.5
                dlon = dlat / max(np.cos(np.radians(plat)), 0.15)
                jl = np.where(np.abs(rlat - plat) <= dlat)[0]
                # longitude window, wrapped
                dl = np.abs((lons - plon + 180.0) % 360.0 - 180.0)
                jo = np.where(dl <= dlon)[0]
                rec = {"t": times[i], "pix": False, "min_pix_km": np.inf,
                       "area": 0, "centre": False}
                if len(jl) and len(jo):
                    sub = m[np.ix_(jl, jo)]
                    if sub.any():
                        yy, xx = np.nonzero(sub)
                        d = haversine(plat, plon, rlat[jl][yy], lons[jo][xx])
                        k = int(np.argmin(d))
                        if d[k] <= radius_km:
                            rec["pix"] = True
                            rec["min_pix_km"] = float(d[k])
                            lb = lab[jl[yy[k]], jo[xx[k]]]
                            a = int((lab == lb).sum())
                            rec["area"] = a
                            rec["centre"] = (a >= MIN_BLOB_AREA
                                             and MIN_AREA_PX <= a <= MAX_AREA_PX)
                found[key].append(rec)
        ds.close()
    return dict(found), blob_areas


# ── candidate tracks, with the criterion that ended each one ─────────────────
def candidate_tracks(year):
    """Re-run detection + linking for one year, capturing every closed candidate.

    Uses the production HurricaneTracker so the linking rules are the real ones.
    Met fields (vorticity / MSLP / wind) are skipped: they are written to the
    output text files but never read by create_hurricane_tracks or
    validate_hurricane_track, so omitting them cannot change which tracks survive.
    """
    os.environ["TC_CAPTURE_CANDIDATES"] = "1"
    os.environ.setdefault("TC_MAX_AREA_PX", str(MAX_AREA_PX))
    import xarray as xr
    import pandas as pd
    from Hurricane_track import HurricaneTracker

    HurricaneTracker.CAPTURE = True
    HurricaneTracker.CANDIDATES = []
    HurricaneTracker.REJECT = {}

    tracker = HurricaneTracker(
        hurricane_data_dir=DET_DIR,
        era5_data_dir="/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/",
        vorticity_dir="", wind_speed_path="",
        output_dir="/tmp/tc_miss_taxonomy_unused")

    rows = []
    for fp in sorted(glob.glob(os.path.join(
            DET_DIR, f"era5dara_{year:04d}*_Hurricane_detection.nc"))):
        ds = xr.open_dataset(fp)
        lats = ds.latitude.values
        lons = ds.longitude.values
        var = ds["Hurricane_detection"]
        for i, t in enumerate(ds.time.values):
            for h in tracker.extract_hurricane_centers(var.values[i], lats, lons):
                rows.append({"timestamp": pd.Timestamp(t),
                             "center_lat": h["center_lat"],
                             "center_lon": h["center_lon"],
                             "area_pixels": h["area_pixels"],
                             "max_intensity": h["max_intensity"],
                             "mean_intensity": h["mean_intensity"]})
        ds.close()

    df = pd.DataFrame(rows)
    accepted = tracker.create_hurricane_tracks(df) if len(df) else []
    cands = list(HurricaneTracker.CANDIDATES)
    HurricaneTracker.CAPTURE = False
    return df, cands, accepted, dict(HurricaneTracker.REJECT)


# ── the ladder ───────────────────────────────────────────────────────────────
def classify(year, args):
    ref = ibtracs_storms(year, wind_kt=args.wind_kt, wind_source=args.wind_source)
    box = TE_BOX if args.te_domain else None
    if box:
        ref = [s for s in ref if in_box(s["points"], box)]

    model = parse_hurricane_txt(os.path.join(
        args.track_dir, "hurricane", f"CONUS_year_{year}",
        f"CONUS_hurricane_tracks_detailed_{year}.txt"))
    if box:
        model = [t for t in model if in_box(t, box)]

    ref_pts = [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref]
    assign = match_tracks(model, ref_pts, args.radius_km, args.min_hits, args.tol_h)
    hit_idx = {j for j in assign if j is not None}
    missed = [k for k in range(len(ref)) if k not in hit_idx]

    print(f"  {year}: {len(ref)} reference storms, {len(model)} model tracks, "
          f"{len(hit_idx)} hits, {len(missed)} misses "
          f"(POD {len(hit_idx)/max(len(ref),1):.3f})")

    # -- bucket 5, first, because it must be subtracted before anything else --
    verdicts = {}
    for k in missed:
        s = ref[k]
        pts = s["points"]
        in_year = [p for p in pts if p[0].year == year]
        eligible = [p for p in in_year if abs(p[1]) <= LAT_LIM]
        s["n_pts"] = len(pts)
        s["n_in_year"] = len(in_year)
        s["n_eligible"] = len(eligible)
        if len(in_year) < args.min_hits:
            verdicts[k] = ("5 bookkeeping", "season_vs_calendar_year")
        elif len(eligible) < args.min_hits:
            verdicts[k] = ("5 bookkeeping", "outside_lat45_window")

    todo = [k for k in missed if k not in verdicts]

    # -- detection probe for everything still unexplained --------------------
    wanted = defaultdict(list)
    for k in todo:
        for p in ref[k]["points"]:
            if p[0].year == year and abs(p[1]) <= LAT_LIM:
                wanted[p[0]].append((k, p[1], p[2]))
    # a control group: the same probe over storms we DID catch, so "the network
    # fired on 60% of points" can be read against what a hit looks like
    ctrl = defaultdict(list)
    for k in sorted(hit_idx):
        for p in ref[k]["points"]:
            if p[0].year == year and abs(p[1]) <= LAT_LIM:
                ctrl[p[0]].append((("hit", k), p[1], p[2]))
    allw = defaultdict(list)
    for d in (wanted, ctrl):
        for t, v in d.items():
            allw[t].extend(v)
    probe, blob_areas = probe_detections(year, dict(allw), args.radius_km)

    # -- candidate tracks ----------------------------------------------------
    df, cands, accepted, reject = candidate_tracks(year)
    cand_pts = [[(t, la, lo) for (t, la, lo, _a) in c["points"]] for c in cands]
    cassign = match_tracks(cand_pts, ref_pts, args.radius_km, args.min_hits,
                           args.tol_h)
    cand_by_ref = defaultdict(list)
    for ci, j in enumerate(cassign):
        if j is not None:
            cand_by_ref[j].append(ci)

    for k in todo:
        recs = probe.get(k, [])
        n = len(recs)
        n_pix = sum(1 for r in recs if r["pix"])
        n_ctr = sum(1 for r in recs if r["centre"])
        ref[k]["frac_pix"] = n_pix / n if n else 0.0
        ref[k]["frac_centre"] = n_ctr / n if n else 0.0
        ref[k]["n_probe"] = n
        ref[k]["areas"] = sorted(r["area"] for r in recs if r["pix"])

        mine = cand_by_ref.get(k, [])
        ref[k]["n_candidates"] = len(mine)
        ref[k]["cand_reasons"] = [cands[ci]["reason"] for ci in mine]

        if n_pix == 0:
            verdicts[k] = ("1 segmentation", "no_pixel_within_radius")
        elif n_ctr == 0:
            over = [r["area"] for r in recs if r["pix"] and r["area"] > MAX_AREA_PX]
            verdicts[k] = ("2 sub-threshold",
                           "blob_over_area_cap" if over else "blob_under_area_cap")
        elif not mine:
            # centres existed but no candidate track ever covered the storm:
            # the linker never chained them into anything 2 points long nearby
            verdicts[k] = ("2 sub-threshold", "centres_never_linked")
        elif len(mine) >= 2 and all(
                cands[ci]["reason"] in ("n_points_lt5", "duration_lt48")
                for ci in mine):
            verdicts[k] = ("4 fragmentation",
                           f"{len(mine)}_short_candidates")
        else:
            worst = sorted(
                (cands[ci]["reason"] for ci in mine),
                key=lambda r: (r == "accepted", r.startswith("rej:")))[-1]
            if worst in ("n_points_lt5", "duration_lt48"):
                verdicts[k] = ("2 sub-threshold", worst)
            elif worst == "accepted":
                # a candidate covering this storm WAS accepted, so the published
                # track file should contain it -- a matching artefact, not a miss
                verdicts[k] = ("5 bookkeeping", "accepted_but_unmatched")
            else:
                verdicts[k] = ("3 tracker reject", worst)

    return ref, missed, hit_idx, verdicts, probe, blob_areas, reject, cands, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=[1983, 1994, 2003, 2012, 2021])
    ap.add_argument("--track-dir", default=TRACK_DIR)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--tol-h", type=int, default=3)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--wind-source", default="any", choices=["any", "usa"])
    ap.add_argument("--te-domain", action="store_true")
    ap.add_argument("--json", default="phase4_results/tc_miss_taxonomy.json")
    args = ap.parse_args()

    print(f"detections : {DET_DIR}")
    print(f"tracks     : {args.track_dir}")
    print(f"domain     : {'TE box' if args.te_domain else 'GLOBAL'}   "
          f"area gate {MIN_AREA_PX}..{MAX_AREA_PX} px\n")

    tally = Counter()
    sub = Counter()
    by_basin = defaultdict(lambda: {"ref": 0, "hit": 0})
    by_basin_bucket = defaultdict(Counter)
    rows = []
    all_blob = []
    reject_tot = Counter()

    for y in args.years:
        ref, missed, hit_idx, verdicts, probe, blobs, reject, cands, df = \
            classify(y, args)
        all_blob.extend(blobs)
        reject_tot.update(reject)
        for k, s in enumerate(ref):
            by_basin[s["basin"]]["ref"] += 1
            if k in hit_idx:
                by_basin[s["basin"]]["hit"] += 1
        for k in missed:
            b, why = verdicts.get(k, ("? unclassified", "?"))
            tally[b] += 1
            sub[(b, why)] += 1
            s = ref[k]
            by_basin_bucket[s["basin"]][b] += 1
            rows.append({
                "year": y, "sid": s["sid"], "name": s["name"],
                "basin": s["basin"], "peak_kt": s["peak_kt"],
                "n_points": s["n_pts"], "n_in_year": s.get("n_in_year"),
                "n_eligible": s.get("n_eligible"),
                "frac_pix": round(s.get("frac_pix", -1), 3),
                "frac_centre": round(s.get("frac_centre", -1), 3),
                "n_candidates": s.get("n_candidates", 0),
                "cand_reasons": s.get("cand_reasons", []),
                "bucket": b, "why": why,
                "t0": s["points"][0][0].isoformat(),
                "lat0": round(s["points"][0][1], 2),
                "lon0": round(s["points"][0][2], 2),
            })

    n_ref = sum(v["ref"] for v in by_basin.values())
    n_hit = sum(v["hit"] for v in by_basin.values())
    n_miss = len(rows)

    print(f"\n{'='*72}\n  MISS TAXONOMY  —  {n_ref} reference storms, "
          f"{n_hit} hits (POD {n_hit/max(n_ref,1):.3f}), {n_miss} misses\n{'='*72}")
    for b, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<18} {c:>4}  ({c/max(n_miss,1):5.1%} of misses, "
              f"{c/max(n_ref,1):5.1%} of all storms)")
        for (bb, why), cc in sorted(sub.items(), key=lambda kv: -kv[1]):
            if bb == b:
                print(f"      {why:<28} {cc:>4}")

    print(f"\n  POD by basin")
    print(f"  {'basin':<8}{'ref':>6}{'hit':>6}{'POD':>8}   dominant miss bucket")
    for bs in sorted(by_basin, key=lambda b: -by_basin[b]["ref"]):
        v = by_basin[bs]
        dom = by_basin_bucket[bs].most_common(1)
        dtxt = f"{dom[0][0]} ({dom[0][1]})" if dom else "-"
        print(f"  {bs:<8}{v['ref']:>6}{v['hit']:>6}"
              f"{v['hit']/max(v['ref'],1):>8.3f}   {dtxt}")

    if all_blob:
        a = np.array(all_blob)
        print(f"\n  blob-area census ({len(a)} blobs over the probed timesteps): "
              f"median {np.median(a):.0f} px, p90 {np.percentile(a,90):.0f}, "
              f"max {a.max():.0f}")
        print(f"    below the {MIN_AREA_PX}px floor : "
              f"{100*np.mean(a < max(MIN_AREA_PX, MIN_BLOB_AREA)):.2f}%")
        print(f"    above the {MAX_AREA_PX}px cap   : "
              f"{100*np.mean(a > MAX_AREA_PX):.2f}%")
    if reject_tot:
        t = sum(reject_tot.values())
        print(f"\n  validate_hurricane_track rejections ({t}):")
        for k, v in reject_tot.most_common():
            print(f"    {k:<18}{v:>5}")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"years": args.years, "te_domain": args.te_domain,
                   "n_ref": n_ref, "n_hit": n_hit, "n_miss": n_miss,
                   "buckets": {k: v for k, v in tally.items()},
                   "sub_reasons": {f"{a}|{b}": c for (a, b), c in sub.items()},
                   "by_basin": {k: v for k, v in by_basin.items()},
                   "misses": rows}, f, indent=1, default=str)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
