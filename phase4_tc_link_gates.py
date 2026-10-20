#!/usr/bin/env python3
"""
phase4_tc_link_gates.py — which linking gate breaks the chain on a missed storm?

phase4_tc_miss_runs.py answers "was the storm visible for long enough in a row".
This answers the complementary question: given the centres the model DID emit
along a storm, which of create_hurricane_tracks' gates refuses to connect them.

For each storm it walks the storm's own 6-hourly stamps, takes the nearest model
centre at each stamp, and evaluates the consecutive pair against every condition
the linker applies, in the order the linker applies them:

  no_centre        one of the two stamps has no centre within --assoc-km
  speed_slow       < 0.5 m/s   (is_realistic_hurricane_movement)
  speed_fast       > 32 m/s
  dlat             latitude change > 7 deg per 6 h
  equator          sign change in latitude
  dlon             longitude change > 10 deg per 6 h
  radius           distance > 430 km * min(dt/6, 2)
  ok               the pair would have been linked

The speed gate is the one to watch: the model field is 192x384 (~0.94 deg) and
centres are snapped to the 0.25 deg output grid, so a storm whose blob does not
move between two steps produces a displacement of exactly zero, a speed of zero,
and a REFUSED link — the tracker breaks the chain precisely on the slow-moving
storms.  Whether that actually happens is measured here rather than assumed.

Usage
  phase4_tc_link_gates.py --years 2003 2021
"""
import os
import sys
import glob
import json
import argparse
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_miss_taxonomy import (      # noqa: E402
    ibtracs_storms, parse_hurricane_txt, match_tracks, haversine,
    DET_DIR, TRACK_DIR, LAT_LIM, MIN_AREA_PX, MAX_AREA_PX, MIN_BLOB_AREA,
)

MAX_DIST_KM = 430.0      # HurricaneTracker.max_distance_km
MAX_DLAT_6H = 7.0        # max_lat_change_per_6h
MIN_SPEED_MS = 0.5
MAX_SPEED_MS = 32.0
MAX_DLON_6H = 10.0


def centres_for(year, need):
    """dict[timestamp] -> (n,3) array of gated centres (lat, lon, area)."""
    import xarray as xr
    from scipy.ndimage import label, center_of_mass
    out = {}
    for fp in sorted(glob.glob(os.path.join(
            DET_DIR, f"era5dara_{year:04d}*_Hurricane_detection.nc"))):
        ds = xr.open_dataset(fp)
        times = [np.datetime64(t, "s").astype("datetime64[s]").item()
                 for t in ds.time.values]
        idx = [i for i, t in enumerate(times) if t in need]
        if not idx:
            ds.close()
            continue
        lats, lons = ds.latitude.values, ds.longitude.values
        keep = np.where((lats >= -LAT_LIM) & (lats <= LAT_LIM))[0]
        rlat = lats[keep]
        var = ds["Hurricane_detection"]
        for i in idx:
            m = var.values[i][keep] > 0.5
            lab, n = label(m)
            rows = []
            if n:
                areas = np.bincount(lab.ravel())
                sel = [j for j in range(1, n + 1)
                       if areas[j] >= MIN_BLOB_AREA
                       and MIN_AREA_PX <= areas[j] <= MAX_AREA_PX]
                if sel:
                    for j, (cy, cx) in zip(sel, center_of_mass(m, lab, sel)):
                        cy, cx = int(round(cy)), int(round(cx))
                        if cy < len(rlat) and cx < len(lons) \
                                and abs(rlat[cy]) <= 45.0:
                            rows.append((float(rlat[cy]), float(lons[cx]),
                                         float(areas[j])))
            out[times[i]] = np.array(rows) if rows else np.zeros((0, 3))
        ds.close()
    return out


def gate_of(p, q, dt_h):
    """Which gate a consecutive centre pair fails, in the linker's own order."""
    la1, lo1 = p
    la2, lo2 = q
    d = float(haversine(la1, lo1, la2, lo2))
    v = d * 1000.0 / (dt_h * 3600.0)
    if v < MIN_SPEED_MS:
        return "speed_slow", d, v
    if v > MAX_SPEED_MS:
        return "speed_fast", d, v
    if abs(la2 - la1) > MAX_DLAT_6H * (dt_h / 6.0):
        return "dlat", d, v
    if (la1 > 0) != (la2 > 0):
        return "equator", d, v
    dlon = abs(lo2 - lo1)
    if dlon > 180.0:
        dlon = 360.0 - dlon
    if dlon > MAX_DLON_6H * (dt_h / 6.0):
        return "dlon", d, v
    if d >= MAX_DIST_KM * min(dt_h / 6.0, 2.0):
        return "radius", d, v
    return "ok", d, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2003, 2021])
    ap.add_argument("--track-dir", default=TRACK_DIR)
    ap.add_argument("--assoc-km", type=float, default=300.0,
                    help="how close a centre must be to count as this storm's")
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--json", default="phase4_results/tc_link_gates.json")
    args = ap.parse_args()

    tally = {"hit": Counter(), "missed": Counter()}
    zero_disp = {"hit": 0, "missed": 0}
    pairs = {"hit": 0, "missed": 0}
    per_storm = []

    for year in args.years:
        ref = ibtracs_storms(year, wind_kt=args.wind_kt, wind_source="any")
        mtr = parse_hurricane_txt(os.path.join(
            args.track_dir, "hurricane", f"CONUS_year_{year}",
            f"CONUS_hurricane_tracks_detailed_{year}.txt"))
        ref_pts = [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref]
        hit = {j for j in match_tracks(mtr, ref_pts, 300.0, 2, 3) if j is not None}

        need = set()
        for s in ref:
            for p in s["points"]:
                if p[0].year == year and abs(p[1]) <= LAT_LIM:
                    need.add(p[0])
        cen = centres_for(year, need)

        for k, s in enumerate(ref):
            pts = [p for p in s["points"]
                   if p[0].year == year and abs(p[1]) <= LAT_LIM]
            if len(pts) < 2:
                continue
            grp = "hit" if k in hit else "missed"
            chain = []
            for (t, la, lo, _w) in pts:
                c = cen.get(t)
                if c is None or not len(c):
                    chain.append(None)
                    continue
                d = haversine(la, lo, c[:, 0], c[:, 1])
                j = int(np.argmin(d))
                chain.append((float(c[j, 0]), float(c[j, 1]))
                             if d[j] <= args.assoc_km else None)
            local = Counter()
            for i in range(len(chain) - 1):
                dt_h = (pts[i + 1][0] - pts[i][0]).total_seconds() / 3600.0
                if dt_h <= 0:
                    continue
                if chain[i] is None or chain[i + 1] is None:
                    local["no_centre"] += 1
                    continue
                g, d, v = gate_of(chain[i], chain[i + 1], dt_h)
                local[g] += 1
                if d == 0.0:
                    zero_disp[grp] += 1
                pairs[grp] += 1
            tally[grp].update(local)
            per_storm.append({"year": year, "sid": s["sid"], "name": s["name"],
                              "basin": s["basin"], "hit": grp == "hit",
                              "gates": dict(local)})
        print(f"  {year}: {len(ref)} storms, {len(hit)} hit", flush=True)

    print(f"\n  consecutive-stamp link verdicts along each storm "
          f"(association radius {args.assoc_km:g} km)")
    keys = ["ok", "no_centre", "speed_slow", "speed_fast", "radius", "dlat",
            "dlon", "equator"]
    print(f"  {'verdict':<12}{'hit storms':>14}{'missed storms':>16}")
    for kk in keys:
        h, m = tally["hit"][kk], tally["missed"][kk]
        th, tm = sum(tally["hit"].values()), sum(tally["missed"].values())
        print(f"  {kk:<12}{h:>8} {h/max(th,1):>5.1%}{m:>10} {m/max(tm,1):>5.1%}")

    print(f"\n  pairs where the two centres were byte-identical (zero "
          f"displacement -> speed 0 -> refused):")
    for g in ("hit", "missed"):
        print(f"    {g:<8}{zero_disp[g]:>6} of {pairs[g]} linkable pairs "
              f"({zero_disp[g]/max(pairs[g],1):.1%})")

    with open(args.json, "w") as f:
        json.dump({"tally": {g: dict(c) for g, c in tally.items()},
                   "zero_disp": zero_disp, "pairs": pairs,
                   "per_storm": per_storm}, f, indent=1)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
