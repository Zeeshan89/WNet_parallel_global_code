#!/usr/bin/env python3
"""
phase4_tc_miss_learnability.py — at the storms the model MISSES, is the signature
present in the six channels the model actually reads?

This is the measurement that decides whether the remaining TC recall gap needs a new
input variable or a changed training parameter, and it has never been made. The repo
has the hit/miss machinery (phase4_tc_miss_runs.py, phase4_tc_prob_probe.py) and the
physical-field machinery (phase4_ibtracs_learnability.py) but has never joined them:
the existing learnability numbers are POOLED over all storm positions, so they are
dominated by the storms we already catch and cannot say anything about the misses.

Two deliberate differences from phase4_ibtracs_learnability.py:

1. It probes the SPLIT CACHE at the model's own 192x384 grid, not ERA5 at 721x1440.
   "Is it in ERA5" and "can the model see it" are different questions, and the second
   is the one that matters: the model is fed `split_{YYYYMM}.npy[:, 0:6]` and nothing
   else. Probing native resolution would credit the model with detail that the 0.94
   deg grid has already smoothed away. Channel order is fixed by
   dataset.py:PREDICTOR_SPECS -> 0 IVT, 1 MSLP, 2 vorticity850, 3 wind_speed850,
   4 BT, 5 IWV; labels follow at 6 AR, 7 Hurricane, 8 ETC, 9 MCS.

2. It reads the TC LABEL channel too (7). If the label is painted at a missed storm,
   the model was asked to detect it and did not -> a training problem. If the label
   is absent, the model was never asked -> a target problem. That distinction cannot
   be made from the predictors alone.

Note on the years: 2003 and 2021 are HELD-OUT, so the cache label at these positions
was never trained on. It is still the authoritative statement of what the target
construction would have asked for, which is what we want to check.

Thresholds are taken unchanged from phase4_ibtracs_learnability.py:137-186 so the HIT
group can be sanity-checked against the already published pooled fractions
(~0.99 localmin / depth300 / vort, 0.92 wind175 for Sept 2003). If the HIT group does
not land near those, the harness is wrong and not the model.

Usage
  phase4_tc_miss_learnability.py --years 2003 2021
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_miss_taxonomy import (      # noqa: E402
    ibtracs_storms, parse_hurricane_txt, match_tracks, TRACK_DIR, LAT_LIM,
)

CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
         "Storms_precipitation_notebooks/tensor_cache_v5_ibtracs")

H, W = 192, 384
GLAT = np.linspace(90.0, -90.0, H)
GLON = np.linspace(-180.0, 180.0 - 360.0 / W, W)
DLAT = 180.0 / (H - 1)                  # ~0.94 deg
CH = {"IVT": 0, "MSLP": 1, "VORT": 2, "WIND": 3, "BT": 4, "IWV": 5}
HURR_LABEL_CH = 7

# Same physical cuts as phase4_ibtracs_learnability.py
VORT_THRESH = 1.45e-5
DEPTH150, DEPTH300 = 150.0, 300.0
WIND10, WIND175 = 10.0, 17.5


def windows(plat, plon, half_deg, ring_deg):
    """Row/col index arrays for a local box and a wider background ring box."""
    def box(hd):
        dlon = hd / max(np.cos(np.radians(plat)), 0.15)
        rows = np.where(np.abs(GLAT - plat) <= hd)[0]
        d = np.abs((GLON - plon + 180.0) % 360.0 - 180.0)
        cols = np.where(d <= dlon)[0]
        return rows, cols
    return box(half_deg), box(ring_deg)


def probe_month(tag, wanted, half_deg, ring_deg):
    """wanted: dict[timestamp] -> [(key, lat, lon)]. Returns list of records."""
    f = os.path.join(CACHE, f"split_{tag}.npy")
    mf = os.path.join(CACHE, f"split_{tag}.meta.npy")
    if not (os.path.exists(f) and os.path.exists(mf)):
        return []
    arr = np.load(f, mmap_mode="r")
    times = np.load(mf, allow_pickle=True)
    out = []
    for i, tt in enumerate(times):
        t = tt.astype("datetime64[s]").astype(object).replace(
            minute=0, second=0, microsecond=0)
        if t not in wanted:
            continue
        frame = np.asarray(arr[i], dtype=np.float32)
        for (key, plat, plon) in wanted[t]:
            (lr, lc), (rr, rc) = windows(plat, plon, half_deg, ring_deg)
            if not len(lr) or not len(lc) or not len(rr) or not len(rc):
                continue
            loc = frame[:, lr][:, :, lc]
            ring = frame[:, rr][:, :, rc]

            mslp_min = float(loc[CH["MSLP"]].min())
            ring_max = float(ring[CH["MSLP"]].max())
            depth = ring_max - mslp_min
            v = loc[CH["VORT"]]
            vort_signed = float(v.max() if plat >= 0 else -v.min())
            out.append({
                "key": key, "t": t.isoformat(),
                "lat": round(float(plat), 2), "lon": round(float(plon), 2),
                "mslp_min_pa": round(mslp_min, 1),
                "depth_pa": round(depth, 1),
                "vort_signed": float(f"{vort_signed:.4g}"),
                "wind_max": round(float(loc[CH["WIND"]].max()), 2),
                "iwv_max": round(float(loc[CH["IWV"]].max()), 2),
                "bt_min": round(float(loc[CH["BT"]].min()), 1),
                "ivt_max": round(float(loc[CH["IVT"]].max()), 1),
                "label_max": round(float(loc[HURR_LABEL_CH].max()), 3),
                "label_px": int((loc[HURR_LABEL_CH] > 0.5).sum()),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2003, 2021])
    ap.add_argument("--track-dir", default=os.path.join(
        os.path.dirname(TRACK_DIR), "tracks_final_dur24"))
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--half-deg", type=float, default=2.8,
                    help="local box half-width in degrees (~3 model cells, 310 km)")
    ap.add_argument("--ring-deg", type=float, default=5.5,
                    help="background box half-width; matches the detector's "
                         "great_circle_deg=5.5")
    ap.add_argument("--json", default="phase4_results/tc_miss_learnability.json")
    args = ap.parse_args()

    print(f"  cache     : {CACHE}")
    print(f"  tracks    : {args.track_dir}")
    print(f"  local box : +-{args.half_deg:g} deg, background +-{args.ring_deg:g} deg\n")

    recs = []
    meta = {}
    for y in args.years:
        ref = [s for s in ibtracs_storms(y, args.wind_kt, "any")
               if sum(1 for p in s["points"]
                      if p[0].year == y and abs(p[1]) <= LAT_LIM) >= 2]
        rp = [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref]
        mt = parse_hurricane_txt(os.path.join(
            args.track_dir, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        hit = {j for j in match_tracks(mt, rp, args.radius_km, 2, 3)
               if j is not None}
        print(f"  {y}: {len(ref)} ref, {len(hit)} hit, {len(ref)-len(hit)} miss "
              f"(POD {len(hit)/len(ref):.3f})")

        wanted = defaultdict(list)
        for k, s in enumerate(ref):
            key = f"{y}:{s['sid']}"
            meta[key] = {"name": s["name"], "basin": s["basin"],
                         "peak_kt": s["peak_kt"], "hit": k in hit,
                         "n_points": len(s["points"]), "year": y}
            for (t, la, lo, _w) in s["points"]:
                if t.year == y and abs(la) <= LAT_LIM:
                    wanted[t].append((key, la, lo))

        for mo in range(1, 13):
            r = probe_month(f"{y:04d}{mo:02d}", wanted, args.half_deg,
                            args.ring_deg)
            recs.extend(r)
            print(f"    {y}{mo:02d}: {len(r)} positions", flush=True)

    if not recs:
        raise SystemExit("nothing probed")

    for r in recs:
        r["hit"] = meta[r["key"]]["hit"]
        r["basin"] = meta[r["key"]]["basin"]
    hitr = [r for r in recs if r["hit"]]
    misr = [r for r in recs if not r["hit"]]

    print(f"\n  {len(recs)} storm positions probed "
          f"({len(hitr)} on hit storms, {len(misr)} on missed storms)")

    def frac(rows, fn):
        return np.mean([fn(r) for r in rows]) if rows else float("nan")

    tests = [
        ("MSLP depth >= 150 Pa", lambda r: r["depth_pa"] >= DEPTH150),
        ("MSLP depth >= 300 Pa", lambda r: r["depth_pa"] >= DEPTH300),
        ("vorticity sign-correct >= 1.45e-5", lambda r: r["vort_signed"] >= VORT_THRESH),
        ("wind850 >= 10 m/s", lambda r: r["wind_max"] >= WIND10),
        ("wind850 >= 17.5 m/s (34 kt)", lambda r: r["wind_max"] >= WIND175),
        ("vort AND wind10", lambda r: r["vort_signed"] >= VORT_THRESH
                                      and r["wind_max"] >= WIND10),
        ("TC LABEL painted here", lambda r: r["label_max"] > 0.5),
    ]
    print(f"\n  fraction of positions where the signature is present "
          f"(model grid, 192x384)")
    print(f"  {'test':<38}{'HIT':>9}{'MISS':>9}{'delta':>9}")
    print("  " + "-" * 65)
    rows_out = []
    for name, fn in tests:
        h, m = frac(hitr, fn), frac(misr, fn)
        print(f"  {name:<38}{h:>9.3f}{m:>9.3f}{m-h:>+9.3f}")
        rows_out.append({"test": name, "hit": float(h), "miss": float(m)})

    print(f"\n  magnitude distributions (median [p10, p90])")
    print(f"  {'quantity':<20}{'HIT':>28}{'MISS':>28}")
    for q, unit in (("depth_pa", "Pa"), ("vort_signed", "1/s"),
                    ("wind_max", "m/s"), ("iwv_max", "mm"),
                    ("bt_min", "K"), ("ivt_max", "kg/m/s")):
        def d(rows):
            a = np.array([r[q] for r in rows], dtype=float)
            return f"{np.median(a):>9.3g} [{np.percentile(a,10):>8.3g},{np.percentile(a,90):>8.3g}]"
        print(f"  {q+' ('+unit+')':<20}{d(hitr):>28}{d(misr):>28}")

    # Per-storm rollup: which missed storms have NO signature anywhere
    per = defaultdict(list)
    for r in recs:
        per[r["key"]].append(r)
    print(f"\n  missed storms ranked by how often the MSLP+vort+wind signature "
          f"is present along them")
    print(f"  {'storm':<18}{'basin':>6}{'peak':>6}{'n':>5}{'sig frac':>10}"
          f"{'label frac':>12}")
    bad = []
    for key, rows in sorted(per.items(),
                            key=lambda kv: frac(kv[1], tests[5][1])):
        if meta[key]["hit"]:
            continue
        sf = frac(rows, tests[5][1])
        lf = frac(rows, tests[6][1])
        m = meta[key]
        print(f"  {m['name'][:17]:<18}{m['basin']:>6}{m['peak_kt']:>6.0f}"
              f"{len(rows):>5}{sf:>10.3f}{lf:>12.3f}")
        if sf < 0.5:
            bad.append((m["name"], sf))
    print(f"\n  missed storms whose signature is present at < 50% of their own "
          f"positions: {len(bad)}")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"tests": rows_out, "meta": meta, "records": recs}, f,
                  indent=1, default=str)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
