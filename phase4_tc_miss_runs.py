#!/usr/bin/env python3
"""
phase4_tc_miss_runs.py — is the model failing to SEE the missed storms, or failing
to see them for long enough IN A ROW?

phase4_tc_miss_taxonomy.py showed the dominant miss bucket is "centres exist but
the candidate track has <5 points or spans <48 h".  That is compatible with two
very different causes, and they have opposite fixes:

  flicker      the network fires on a storm at scattered timesteps, never on
               enough consecutive ones.  Fix is in the loss / the model.
  linker       the network fires on a long unbroken run, but
               create_hurricane_tracks refuses to chain the centres (speed gate,
               430 km radius, 7 deg/6 h latitude gate, dateline).  Fix is in the
               tracker and costs nothing.

The discriminator is the RUN STRUCTURE of detection presence along each storm's
own 6-hourly stamps.  If the longest unbroken run (allowing the tracker's own
<=4-step bridging) already fails 5 points / 48 h, no tracker change can recover
the storm.  If it passes and the tracker still produced a short candidate, the
tracker is what lost it.

Also reports, for every storm stamp, the distance from the IBTrACS centre to the
nearest model blob CENTROID -- not just to the nearest lit pixel.  A large blob
whose centre of mass sits >300 km away is detected but unmatchable, which would
show up as a miss without any of the above being true.

Usage
  phase4_tc_miss_runs.py --years 2003 2021
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_miss_taxonomy import (      # noqa: E402
    ibtracs_storms, parse_hurricane_txt, match_tracks, haversine,
    DET_DIR, TRACK_DIR, LAT_LIM, MIN_AREA_PX, MAX_AREA_PX, MIN_BLOB_AREA,
)

MAX_GAP_STEPS = 4        # HurricaneTracker.create_hurricane_tracks default
MIN_POINTS = 5           # close_track
MIN_DURATION_H = 48.0    # close_track


def blob_centres(year, need_times):
    """dict[timestamp] -> array of (lat, lon, area) for every blob that would
    become a tracker centre, replicating extract_hurricane_centers' gates."""
    import xarray as xr
    from scipy.ndimage import label, center_of_mass

    out = {}
    for fp in sorted(glob.glob(os.path.join(
            DET_DIR, f"era5dara_{year:04d}*_Hurricane_detection.nc"))):
        ds = xr.open_dataset(fp)
        times = [np.datetime64(t, "s").astype("datetime64[s]").item()
                 for t in ds.time.values]
        need = [i for i, t in enumerate(times) if t in need_times]
        if not need:
            ds.close()
            continue
        lats = ds.latitude.values
        lons = ds.longitude.values
        keep = np.where((lats >= -LAT_LIM) & (lats <= LAT_LIM))[0]
        rlat = lats[keep]
        var = ds["Hurricane_detection"]
        for i in need:
            m = var.values[i][keep] > 0.5
            lab, n = label(m)
            rows = []
            if n:
                areas = np.bincount(lab.ravel())
                idx = [j for j in range(1, n + 1)
                       if areas[j] >= MIN_BLOB_AREA
                       and MIN_AREA_PX <= areas[j] <= MAX_AREA_PX]
                if idx:
                    coms = center_of_mass(m, lab, idx)
                    for j, (cy, cx) in zip(idx, coms):
                        cy, cx = int(round(cy)), int(round(cx))
                        if cy >= len(rlat) or cx >= len(lons):
                            continue
                        if abs(rlat[cy]) > 45.0:
                            continue
                        rows.append((float(rlat[cy]), float(lons[cx]),
                                     int(areas[j])))
            out[times[i]] = np.array(rows, dtype=float) if rows else \
                np.zeros((0, 3))
        ds.close()
    return out


def runs_of(flags):
    """Lengths of maximal True runs, and of the False gaps between them."""
    runs, gaps, cur, gap = [], [], 0, 0
    for f in flags:
        if f:
            if gap:
                gaps.append(gap)
                gap = 0
            cur += 1
        else:
            if cur:
                runs.append(cur)
                cur = 0
            gap += 1
    if cur:
        runs.append(cur)
    return runs, gaps


def bridged_segments(flags, max_gap=MAX_GAP_STEPS):
    """Merge True runs separated by <= max_gap False steps, as the tracker's
    ghost-position bridging does. Returns list of (i0, i1) inclusive index spans."""
    idx = [i for i, f in enumerate(flags) if f]
    if not idx:
        return []
    segs = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - segs[-1][1] - 1 <= max_gap:
            segs[-1][1] = i
        else:
            segs.append([i, i])
    return [tuple(s) for s in segs]


def analyse(year, args):
    ref = ibtracs_storms(year, wind_kt=args.wind_kt, wind_source=args.wind_source)
    model = parse_hurricane_txt(os.path.join(
        args.track_dir, "hurricane", f"CONUS_year_{year}",
        f"CONUS_hurricane_tracks_detailed_{year}.txt"))
    ref_pts = [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref]
    assign = match_tracks(model, ref_pts, args.radius_km, 2, 3)
    hit_idx = {j for j in assign if j is not None}

    need = set()
    for s in ref:
        for p in s["points"]:
            if p[0].year == year and abs(p[1]) <= LAT_LIM:
                need.add(p[0])
    centres = blob_centres(year, need)

    rows = []
    for k, s in enumerate(ref):
        pts = [p for p in s["points"]
               if p[0].year == year and abs(p[1]) <= LAT_LIM]
        if len(pts) < 2:
            continue
        flags, dists = [], []
        for (t, la, lo, _w) in pts:
            c = centres.get(t)
            if c is None or not len(c):
                flags.append(False)
                dists.append(np.inf)
                continue
            d = haversine(la, lo, c[:, 0], c[:, 1])
            dmin = float(d.min())
            dists.append(dmin)
            flags.append(dmin <= args.radius_km)
        segs = bridged_segments(flags)
        best = None
        for (i0, i1) in segs:
            npts = sum(flags[i0:i1 + 1])
            dur = (pts[i1][0] - pts[i0][0]).total_seconds() / 3600.0
            if best is None or (npts, dur) > (best[0], best[1]):
                best = (npts, dur)
        runs, gaps = runs_of(flags)
        finite = [d for d in dists if np.isfinite(d)]
        rows.append({
            "year": year, "k": k, "sid": s["sid"], "name": s["name"],
            "basin": s["basin"], "peak_kt": s["peak_kt"],
            "hit": k in hit_idx,
            "n_steps": len(pts),
            "cov": round(sum(flags) / len(pts), 3),
            "longest_run": max(runs) if runs else 0,
            "n_runs": len(runs),
            "max_gap": max(gaps) if gaps else 0,
            "best_seg_pts": best[0] if best else 0,
            "best_seg_dur_h": best[1] if best else 0.0,
            "recoverable": bool(best and best[0] >= MIN_POINTS
                                and best[1] >= MIN_DURATION_H),
            "median_centre_km": round(float(np.median(finite)), 1) if finite else None,
            "n_far_only": sum(1 for d in dists
                              if np.isfinite(d) and d > args.radius_km),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2003, 2021])
    ap.add_argument("--track-dir", default=TRACK_DIR)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--wind-source", default="any")
    ap.add_argument("--json", default="phase4_results/tc_miss_runs.json")
    args = ap.parse_args()

    rows = []
    for y in args.years:
        r = analyse(y, args)
        rows.extend(r)
        print(f"  {y}: {len(r)} storms analysed")

    hits = [r for r in rows if r["hit"]]
    miss = [r for r in rows if not r["hit"]]

    def summ(label, rs):
        if not rs:
            return
        cov = np.array([r["cov"] for r in rs])
        lr = np.array([r["longest_run"] for r in rs])
        bp = np.array([r["best_seg_pts"] for r in rs])
        bd = np.array([r["best_seg_dur_h"] for r in rs])
        print(f"  {label:<10}{len(rs):>5}"
              f"{cov.mean():>10.3f}{np.median(lr):>10.0f}{np.median(bp):>10.0f}"
              f"{np.median(bd):>11.0f}{100*np.mean([r['recoverable'] for r in rs]):>13.1f}")

    print(f"\n  {'group':<10}{'n':>5}{'mean cov':>10}{'med run':>10}"
          f"{'med seg pts':>10}{'med seg h':>11}{'% >=5pts&48h':>13}")
    summ("HIT", hits)
    summ("MISS", miss)

    print(f"\n  Of the {len(miss)} misses:")
    rec = [r for r in miss if r["recoverable"]]
    zero = [r for r in miss if r["cov"] == 0.0]
    thin = [r for r in miss if 0.0 < r["cov"] and not r["recoverable"]]
    print(f"    never detected (cov 0)                       {len(zero):>4}"
          f"  ({len(zero)/len(miss):5.1%})")
    print(f"    detected but no >=5pt/48h bridged segment    {len(thin):>4}"
          f"  ({len(thin)/len(miss):5.1%})   <- flicker")
    print(f"    detected WITH a >=5pt/48h bridged segment    {len(rec):>4}"
          f"  ({len(rec)/len(miss):5.1%})   <- tracker lost it")

    far = [r for r in miss if r["cov"] == 0.0 and r["n_far_only"] > 0]
    print(f"    (of the never-detected, {len(far)} had a blob centroid present "
          f"but always >{args.radius_km:g} km away)")

    print(f"\n  Coverage of the MISSED storms that were detected at all:")
    cv = sorted(r["cov"] for r in miss if r["cov"] > 0)
    if cv:
        cv = np.array(cv)
        print(f"    min {cv.min():.2f}  p25 {np.percentile(cv,25):.2f}  "
              f"median {np.median(cv):.2f}  p75 {np.percentile(cv,75):.2f}  "
              f"max {cv.max():.2f}")

    with open(args.json, "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
