#!/usr/bin/env python3
"""
phase4_tc_recover.py — how much POD is available WITHOUT retraining?

Two levers, measured separately and then together:

A. the calendar-year fix
   ibtracs_tracks() selects storms on `season == year` and matches them against
   the model's per-CALENDAR-year track file. IBTrACS `season` is a season LABEL,
   so a Southern-Hemisphere storm in Nov/Dec of year Y carries season Y+1 and is
   scored against year Y+1's tracks, in which it has no points at all. The label
   builder already handles this (phase4_build_ibtracs_label.py:76), so the model
   was TRAINED on these storms and is only being SCORED wrongly.

   Three ways of counting are compared:
     season      current behaviour, the 0.662 baseline
     calendar    reference storms keyed by the calendar year their points fall
                 in -- the correct pairing for a per-calendar-year track file
     pooled      season == year, but matched against tracks from Y-1, Y and Y+1
                 pooled, so a storm straddling New Year can be caught by either
                 file. Keeps the published denominator, so it isolates how many
                 of the 27 are genuinely detected rather than merely miscounted.

B. the tracker gates
   create_hurricane_tracks' >=5 points and >=48 h gates, its <=4-step bridging and
   its 430 km linking radius are the only knobs that do not need retraining.
   Sweeping them shows how much of the remaining miss list is reachable, and what
   it costs in false alarms.

Usage
  phase4_tc_recover.py --mode counting
  phase4_tc_recover.py --mode gates --sweep
"""
import os
import sys
import json
import argparse
import itertools
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_miss_taxonomy import (      # noqa: E402
    ibtracs_storms, parse_hurricane_txt, match_tracks, TRACK_DIR, LAT_LIM,
)

CENTRES = "phase4_results/centres"
YEARS = [1983, 1994, 2003, 2012, 2021]


def load_model_tracks(track_dir, year):
    return parse_hurricane_txt(os.path.join(
        track_dir, "hurricane", f"CONUS_year_{year}",
        f"CONUS_hurricane_tracks_detailed_{year}.txt"))


def score(model, ref_pts, radius_km=300.0, min_hits=2, tol_h=3):
    if not ref_pts:
        return None
    assign = match_tracks(model, ref_pts, radius_km, min_hits, tol_h)
    seen, dup, fa = set(), 0, 0
    for j in assign:
        if j is None:
            fa += 1
        elif j in seen:
            dup += 1
        else:
            seen.add(j)
    hits = len(seen)
    misses = len(ref_pts) - hits
    den = hits + misses + fa
    return {"n_model": len(model), "n_ref": len(ref_pts), "hits": hits,
            "misses": misses, "false_alarms": fa, "duplicates": dup,
            "pod": hits / len(ref_pts), "far": fa / len(model) if model else 0.0,
            "csi": hits / den if den else 0.0}


def pool(rows):
    t = defaultdict(int)
    for r in rows:
        if r is None:
            continue
        for k in ("n_model", "n_ref", "hits", "misses", "false_alarms",
                  "duplicates"):
            t[k] += r[k]
    den = t["hits"] + t["misses"] + t["false_alarms"]
    o = dict(t)
    o["pod"] = t["hits"] / t["n_ref"] if t["n_ref"] else 0.0
    o["far"] = t["false_alarms"] / t["n_model"] if t["n_model"] else 0.0
    o["csi"] = t["hits"] / den if den else 0.0
    return o


# ── A. counting conventions ──────────────────────────────────────────────────
_REF_CACHE = {}


def storms_of_season(y, wind_kt):
    if y not in _REF_CACHE:
        _REF_CACHE[y] = ibtracs_storms(y, wind_kt=wind_kt, wind_source="any")
    return _REF_CACHE[y]


def ref_for(year, how, wind_kt=34.0):
    """Reference storm point-lists for one model year under a counting rule."""
    if how == "season":
        ref = storms_of_season(year, wind_kt)
        return [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref], ref
    # calendar: any storm from seasons Y-1..Y+1 with >=2 qualifying points that
    # actually fall inside calendar year `year` and inside the tracker's window
    out, meta = [], []
    for s in itertools.chain(*(storms_of_season(y, wind_kt)
                               for y in (year - 1, year, year + 1))):
        pts = [(t, la, lo) for (t, la, lo, _w) in s["points"]
               if t.year == year and abs(la) <= LAT_LIM]
        if len(pts) >= 2:
            out.append(pts)
            meta.append(s)
    return out, meta


def mode_counting(args):
    rows = {"season": [], "calendar": [], "pooled": []}
    detail = []
    for y in args.years:
        mt = load_model_tracks(args.track_dir, y)
        rs, meta_s = ref_for(y, "season", args.wind_kt)
        rows["season"].append(score(mt, rs, args.radius_km))

        rc, _ = ref_for(y, "calendar", args.wind_kt)
        rows["calendar"].append(score(mt, rc, args.radius_km))

        pooled = []
        for yy in (y - 1, y, y + 1):
            pooled.extend(load_model_tracks(args.track_dir, yy))
        rp = score(pooled, rs, args.radius_km)
        rows["pooled"].append(rp)

        # which season-selected storms change verdict when the adjacent years'
        # track files are allowed to match them
        a = match_tracks(mt, rs, args.radius_km, 2, 3)
        b = match_tracks(pooled, rs, args.radius_km, 2, 3)
        ha = {j for j in a if j is not None}
        hb = {j for j in b if j is not None}
        for k in sorted(hb - ha):
            s = meta_s[k]
            n_in = sum(1 for p in s["points"] if p[0].year == y)
            detail.append({"year": y, "sid": s["sid"], "name": s["name"],
                           "basin": s["basin"], "peak_kt": s["peak_kt"],
                           "n_points": len(s["points"]), "n_in_year": n_in})
        print(f"  {y}: season POD {rows['season'][-1]['pod']:.3f}  "
              f"calendar POD {rows['calendar'][-1]['pod']:.3f}  "
              f"pooled POD {rp['pod']:.3f}", flush=True)

    print(f"\n  {'convention':<12}{'ref':>6}{'model':>7}{'hits':>6}{'miss':>6}"
          f"{'FA':>6}{'POD':>8}{'FAR':>8}{'CSI':>8}")
    print("  " + "-" * 67)
    for k in ("season", "calendar", "pooled"):
        t = pool(rows[k])
        print(f"  {k:<12}{t['n_ref']:>6}{t['n_model']:>7}{t['hits']:>6}"
              f"{t['misses']:>6}{t['false_alarms']:>6}{t['pod']:>8.3f}"
              f"{t['far']:>8.3f}{t['csi']:>8.3f}")

    if detail:
        print(f"\n  storms recovered as REAL hits by allowing the adjacent "
              f"year's track file ({len(detail)}):")
        print(f"  {'year':<6}{'name':<14}{'basin':>6}{'peak':>7}"
              f"{'pts':>5}{'in-yr':>7}")
        for d in detail:
            print(f"  {d['year']:<6}{d['name'][:13]:<14}{d['basin']:>6}"
                  f"{d['peak_kt']:>7.0f}{d['n_points']:>5}{d['n_in_year']:>7}")
    with open(args.json, "w") as f:
        json.dump({"totals": {k: pool(v) for k, v in rows.items()},
                   "recovered": detail}, f, indent=1)
    print(f"\n  wrote {args.json}")
    return 0


# ── B. tracker gates ─────────────────────────────────────────────────────────
def tracks_from_cache(year, env):
    """Re-link one year's cached centres under a given gate configuration."""
    import pandas as pd
    for k, v in env.items():
        os.environ[k] = str(v)
    for m in list(sys.modules):
        if m == "Hurricane_track":
            del sys.modules[m]
    from Hurricane_track import HurricaneTracker

    z = np.load(os.path.join(CENTRES, f"centres_{year}.npz"),
                allow_pickle=False)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(z["timestamp"]),
        "center_lat": z["center_lat"], "center_lon": z["center_lon"],
        "area_pixels": z["area_pixels"], "max_intensity": z["max_intensity"],
        "mean_intensity": z["max_intensity"],
    })
    tk = HurricaneTracker(
        hurricane_data_dir="", era5_data_dir="", vorticity_dir="",
        wind_speed_path="", output_dir="/tmp/tc_recover_unused")
    if "TC_MAX_LINK_KM" in env:
        tk.max_distance_km = float(env["TC_MAX_LINK_KM"])
    tr = tk.create_hurricane_tracks(df)
    return [[(d["timestamp"].to_pydatetime(), d["center_lat"], d["center_lon"])
             for d in t["detections"]] for t in tr]


# Production baseline as shipped: 5 points, 24 h, gap 4 steps, 430 km link,
# 2000 px area cap, 0.5 m/s speed floor.
PROD = {"TC_MIN_POINTS": 5, "TC_MIN_DURATION_H": 24.0, "TC_MAX_GAP_STEPS": 4,
        "TC_MAX_LINK_KM": 430, "TC_MAX_AREA_PX": 2000,
        "TC_MIN_SPEED_MS": 0.5, "TC_LAND_GAP_STEPS": 0}

# The two levers the landfall diagnosis pointed at. Detection coverage over land
# equals coverage over ocean (0.443 vs 0.445), so land truncation is a LINKING
# problem, not a detection one. Both are tested against the shipped baseline.
LAND_CONFIGS = [
    ("production            floor 0.5, land gap = ocean", {}),
    ("speed floor 0.5 -> 0.0  (disable)", {"TC_MIN_SPEED_MS": 0.0}),
    ("speed floor 0.5 -> 0.1", {"TC_MIN_SPEED_MS": 0.1}),
    ("land gap 4 -> 8", {"TC_LAND_GAP_STEPS": 8}),
    ("land gap 4 -> 12", {"TC_LAND_GAP_STEPS": 12}),
    ("land gap 4 -> 16", {"TC_LAND_GAP_STEPS": 16}),
    ("land gap 8  + floor 0.0", {"TC_LAND_GAP_STEPS": 8, "TC_MIN_SPEED_MS": 0.0}),
    ("land gap 12 + floor 0.0", {"TC_LAND_GAP_STEPS": 12, "TC_MIN_SPEED_MS": 0.0}),
    ("land gap 16 + floor 0.0", {"TC_LAND_GAP_STEPS": 16, "TC_MIN_SPEED_MS": 0.0}),
]


def mode_gates(args, land=False):
    configs = [
        ("production        5pts/48h/gap4/430km", {}),
        ("points 5->4       4pts/48h", {"TC_MIN_POINTS": 4}),
        ("points 5->3       3pts/48h", {"TC_MIN_POINTS": 3}),
        ("duration 48->36h  5pts/36h", {"TC_MIN_DURATION_H": 36}),
        ("duration 48->24h  5pts/24h", {"TC_MIN_DURATION_H": 24}),
        ("both  4pts/24h", {"TC_MIN_POINTS": 4, "TC_MIN_DURATION_H": 24}),
        ("both  3pts/24h", {"TC_MIN_POINTS": 3, "TC_MIN_DURATION_H": 24}),
        ("bridge gap4->8    5pts/48h/gap8", {"TC_MAX_GAP_STEPS": 8}),
        ("bridge gap4->12   5pts/48h/gap12", {"TC_MAX_GAP_STEPS": 12}),
        ("link 430->600km", {"TC_MAX_LINK_KM": 600}),
        ("gap8 + 600km", {"TC_MAX_GAP_STEPS": 8, "TC_MAX_LINK_KM": 600}),
        ("gap8 + 600km + 4pts/24h",
         {"TC_MAX_GAP_STEPS": 8, "TC_MAX_LINK_KM": 600,
          "TC_MIN_POINTS": 4, "TC_MIN_DURATION_H": 24}),
        ("gap12 + 600km + 3pts/24h",
         {"TC_MAX_GAP_STEPS": 12, "TC_MAX_LINK_KM": 600,
          "TC_MIN_POINTS": 3, "TC_MIN_DURATION_H": 24}),
    ]
    base = dict(PROD)
    if land:
        configs = LAND_CONFIGS
    else:
        base["TC_MIN_DURATION_H"] = 48.0

    print(f"  reference: calendar-year keyed (the fix from --mode counting)")
    print(f"\n  {'configuration':<40}{'tracks':>8}{'hits':>6}{'miss':>6}"
          f"{'FA':>6}{'POD':>8}{'FAR':>8}{'CSI':>8}")
    print("  " + "-" * 90)
    out = []
    for label, over in configs:
        env = dict(base)
        env.update(over)
        rows = []
        for y in args.years:
            mt = tracks_from_cache(y, env)
            rc, _ = ref_for(y, "calendar", args.wind_kt)
            rows.append(score(mt, rc, args.radius_km))
        t = pool(rows)
        print(f"  {label:<40}{t['n_model']:>8}{t['hits']:>6}{t['misses']:>6}"
              f"{t['false_alarms']:>6}{t['pod']:>8.3f}{t['far']:>8.3f}"
              f"{t['csi']:>8.3f}")
        out.append({"config": label, "env": over, **t})
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote {args.json}")
    return 0


def main():
    global CENTRES          # rebound below from --centres-dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["counting", "gates", "land"],
                    default="counting")
    ap.add_argument("--centres-dir", default=CENTRES,
                    help="cached centres (phase4_tc_centres_cache.py)")
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    ap.add_argument("--track-dir", default=TRACK_DIR)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.json is None:
        args.json = f"phase4_results/tc_recover_{args.mode}.json"
    CENTRES = args.centres_dir
    if args.mode == "counting":
        return mode_counting(args)
    return mode_gates(args, land=(args.mode == "land"))


if __name__ == "__main__":
    sys.exit(main())
