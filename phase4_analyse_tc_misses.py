#!/usr/bin/env python3
"""
phase4_analyse_tc_misses.py — WHICH IBTrACS storms are missed, and does a duration
filter throw away the short ones?

Two questions this answers, neither of which a POD/FAR/CSI table can:

  1. The storms the new model misses — does the old system find them? A miss that
     the old system also misses is a hard storm; a miss only the new system makes
     is a regression. Note the old TC output is the ch5 training mask with dilated
     IBTrACS positions unioned in, so it finds real storms almost by construction.
     Its column here is a reference, not a competitor.

  2. Does --min-duration-h 96 discard short-lived tropical cyclones? The filter
     applies to MODEL tracks, but a short real storm tends to produce a short model
     track, so the risk is real and specific. Stratifying detection rate by the
     REFERENCE storm's own duration shows directly whether short storms are being
     sacrificed, and how many such storms exist at all.

Reference duration is the span of the storm's >= --wind-kt points, i.e. the same
points that define a storm here, not its full IBTrACS lifetime.

Environment: /ccs/home/zaslam/.conda/envs/my_env/bin/python

Usage
  phase4_analyse_tc_misses.py
  phase4_analyse_tc_misses.py --min-duration-h 96 --min-straightness 0.4 --genesis-lat-max 20
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_tracks_peryear import (        # noqa: E402
    parse_hurricane_txt, ibtracs_tracks, match_tracks, clip_to_box,
    OLD, NEW, TEST_YEARS,
)
from phase4_evaluate_tracks import (     # noqa: E402
    parse_te_tc, filter_tracks, track_stats, TE_BOX,
)

DUR_BINS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 144), (144, 1e9)]


def bin_label(lo, hi):
    return f"<{hi:g}h" if lo == 0 else (f">={lo:g}h" if hi > 1e8 else f"{lo:g}-{hi:g}h")


def found_set(model, ref, radius_km):
    """Indices of reference tracks matched by at least one model track."""
    return {j for j in match_tracks(model, ref, radius_km) if j is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(TEST_YEARS))
    ap.add_argument("--track-dir", default=NEW)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--te-domain", action="store_true")
    # the candidate filter
    ap.add_argument("--min-points", type=int, default=None)
    ap.add_argument("--min-duration-h", type=float, default=96.0)
    ap.add_argument("--min-path-km", type=float, default=None)
    ap.add_argument("--min-straightness", type=float, default=0.4)
    ap.add_argument("--min-speed-ms", type=float, default=None)
    ap.add_argument("--max-speed-ms", type=float, default=None)
    ap.add_argument("--genesis-lat-max", type=float, default=20.0)
    ap.add_argument("--max-abs-lat", type=float, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    box = TE_BOX if args.te_domain else None

    rows = []            # one per reference storm
    per_year = {}
    for y in args.years:
        ref = ibtracs_tracks(y, wind_kt=args.wind_kt)
        new = parse_hurricane_txt(os.path.join(
            args.track_dir, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        old = parse_hurricane_txt(os.path.join(
            OLD, "Hurricanes_WNet_final", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        te = parse_te_tc(y)
        if box:
            ref, _ = clip_to_box(ref, box)
            new, _ = clip_to_box(new, box)
            old, _ = clip_to_box(old, box)
            te, _ = clip_to_box(te, box)
        if not ref:
            continue

        new_f, _ = filter_tracks(new, args)

        f_new = found_set(new, ref, args.radius_km)
        f_newf = found_set(new_f, ref, args.radius_km)
        f_old = found_set(old, ref, args.radius_km)
        f_te = found_set(te, ref, args.radius_km) if te else set()

        per_year[y] = dict(n_ref=len(ref), n_new=len(new), n_new_f=len(new_f),
                           n_old=len(old), n_te=len(te),
                           found_new=len(f_new), found_new_f=len(f_newf),
                           found_old=len(f_old), found_te=len(f_te))

        for j, rt in enumerate(ref):
            s = track_stats(rt)
            rows.append(dict(year=y, idx=j, duration_h=s["duration_h"],
                             n_points=s["n_points"], genesis_lat=s["genesis_lat"],
                             max_abs_lat=s["max_abs_lat"],
                             new=j in f_new, new_f=j in f_newf,
                             old=j in f_old, te=j in f_te))

    n = len(rows)
    dom = "TE box" if args.te_domain else "global"
    filt = (f"dur>={args.min_duration_h:g}h" if args.min_duration_h else "") + \
           (f", str>={args.min_straightness:g}" if args.min_straightness else "") + \
           (f", genesis|lat|<={args.genesis_lat_max:g}" if args.genesis_lat_max else "")
    print(f"\n  {n} IBTrACS storms (>= {args.wind_kt:g} kt), years {args.years}, {dom}")
    print(f"  filter applied to 'new filtered': {filt or 'none'}")

    # ── 1. who finds what ────────────────────────────────────────────────────
    print(f"\n  === detection rate ===")
    print(f"  {'detector':<16}{'found':>7}{'missed':>8}{'POD':>8}")
    print("  " + "-" * 39)
    for key, lab in [("new", "new (raw)"), ("new_f", "new (filtered)"),
                     ("old", "old*"), ("te", "TempestExtremes")]:
        f = sum(r[key] for r in rows)
        print(f"  {lab:<16}{f:>7}{n - f:>8}{f / n:>8.3f}")
    print("  * old TC output is the ch5 mask with IBTrACS unioned in — circular.")

    # ── 2. new's misses vs old ───────────────────────────────────────────────
    miss_new = [r for r in rows if not r["new"]]
    miss_nf = [r for r in rows if not r["new_f"]]
    print(f"\n  === the storms 'new' misses ===")
    a = sum(1 for r in miss_new if r["old"])
    b = sum(1 for r in miss_new if not r["old"])
    print(f"  missed by new (raw)        : {len(miss_new)}")
    print(f"    ...but FOUND by old      : {a}   (old has IBTrACS in its mask)")
    print(f"    ...missed by old too     : {b}   (genuinely hard)")
    a2 = sum(1 for r in miss_nf if r["old"])
    b2 = sum(1 for r in miss_nf if not r["old"])
    lost = sum(1 for r in rows if r["new"] and not r["new_f"])
    print(f"  missed by new (filtered)   : {len(miss_nf)}")
    print(f"    ...found by old          : {a2}")
    print(f"    ...missed by old too     : {b2}")
    print(f"  storms LOST TO THE FILTER  : {lost}  (found raw, missed after filtering)")

    # ── 3. the short-storm question ──────────────────────────────────────────
    print(f"\n  === detection rate by the REFERENCE storm's own duration ===")
    print(f"  (duration = span of its >= {args.wind_kt:g} kt points)")
    print(f"  {'ref duration':<14}{'storms':>8}{'new raw':>10}{'new filt':>10}"
          f"{'lost':>7}{'old':>8}{'TE':>8}")
    print("  " + "-" * 65)
    for lo, hi in DUR_BINS:
        sel = [r for r in rows if lo <= r["duration_h"] < hi]
        if not sel:
            continue
        k = len(sel)
        pn = sum(r["new"] for r in sel) / k
        pf = sum(r["new_f"] for r in sel) / k
        lo_n = sum(1 for r in sel if r["new"] and not r["new_f"])
        po = sum(r["old"] for r in sel) / k
        pt = sum(r["te"] for r in sel) / k
        print(f"  {bin_label(lo, hi):<14}{k:>8}{pn:>10.3f}{pf:>10.3f}"
              f"{lo_n:>7}{po:>8.3f}{pt:>8.3f}")

    short = [r for r in rows if r["duration_h"] < args.min_duration_h] \
        if args.min_duration_h else []
    if short:
        ks = len(short)
        print(f"\n  {ks} of {n} storms ({ks / n:.1%}) are themselves shorter than the "
              f"{args.min_duration_h:g} h filter.")
        print(f"    new raw finds {sum(r['new'] for r in short) / ks:.3f} of them, "
              f"filtered {sum(r['new_f'] for r in short) / ks:.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"per_year": per_year, "storms": rows,
                       "filter": {k: v for k, v in vars(args).items()
                                  if k.startswith(("min_", "max_", "genesis_"))}},
                      f, indent=1, default=str)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
