#!/usr/bin/env python3
"""
phase4_tc_fullrecord_compare.py — did the two no-retraining fixes actually improve
the product, over the WHOLE 1980-2024 record rather than the five test years?

Compares three states, all scored the same way:

  baseline   season-keyed reference   + tracks_final      (48 h gate)
  fix 1      calendar-keyed reference + tracks_final      (48 h gate)
  fix 1+2    calendar-keyed reference + tracks_final_dur24 (24 h gate)

so the two levers separate cleanly: the difference between rows 1 and 2 is pure
counting, and between 2 and 3 is pure tracker.

WHY THE HELD-OUT SPLIT IS PRINTED SEPARATELY, AND WHY IT IS THE ONLY HONEST NUMBER
---------------------------------------------------------------------------------
IBTrACS is this model's TRAINING TARGET (phase4_build_ibtracs_label.py wrote it
into tensor_cache_v5_ibtracs, replacing the CyTRACK heuristic). It is therefore
NOT an independent evaluator on the 40 years the model trained on. A POD computed
over all 45 years is a fit statistic, not a skill estimate, and it will look better
than the model deserves. Only 1983, 1994, 2003, 2012 and 2021 were held out.

So the full-record numbers answer "how many storms does the delivered product
now contain", which is a real and useful question about the artefact, while the
held-out numbers answer "is it actually better", which is the question about skill.
Both are printed; do not quote the 45-year POD as skill.

Usage
  phase4_tc_fullrecord_compare.py
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_recover import (            # noqa: E402
    load_model_tracks, score, pool, ref_for, _REF_CACHE,
)
import phase4_tc_recover as R              # noqa: E402

TEST_YEARS = [1983, 1994, 2003, 2012, 2021]
OLD_DIR = "ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/tracks_final"
NEW_DIR = "ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs/tracks_final_dur24"


def run_state(track_dir, how, years, wind_kt, radius_km):
    """Per-year contingency rows for one (track set, counting rule) state."""
    out = {}
    for y in years:
        mt = load_model_tracks(track_dir, y)
        ref, _ = ref_for(y, how, wind_kt)
        out[y] = score(mt, ref, radius_km)
    return out


def show(label, rows, years):
    t = pool([rows[y] for y in years if rows.get(y)])
    print(f"  {label:<26}{t['n_ref']:>6}{t['n_model']:>8}{t['hits']:>7}"
          f"{t['misses']:>7}{t['false_alarms']:>7}{t['pod']:>8.3f}"
          f"{t['far']:>8.3f}{t['csi']:>8.3f}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1980)
    ap.add_argument("--y1", type=int, default=2024)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--json", default="phase4_results/tc_fullrecord_compare.json")
    args = ap.parse_args()

    years = list(range(args.y0, args.y1 + 1))
    train = [y for y in years if y not in TEST_YEARS]
    held = [y for y in years if y in TEST_YEARS]

    print(f"  scoring {len(years)} years ({args.y0}-{args.y1}); "
          f"{len(held)} held out, {len(train)} used in training\n")

    states = {
        "baseline  season + 48h": run_state(OLD_DIR, "season", years,
                                            args.wind_kt, args.radius_km),
        "fix1      calendar+48h": run_state(OLD_DIR, "calendar", years,
                                            args.wind_kt, args.radius_km),
        "fix1+2    calendar+24h": run_state(NEW_DIR, "calendar", years,
                                            args.wind_kt, args.radius_km),
    }

    tot = {}
    for name, ys in (("ALL 45 YEARS (fit, not skill)", years),
                     ("HELD-OUT 5 YEARS (skill)", held),
                     ("TRAINING 40 YEARS", train)):
        print(f"\n  === {name} ===")
        print(f"  {'state':<26}{'ref':>6}{'tracks':>8}{'hits':>7}{'miss':>7}"
              f"{'FA':>7}{'POD':>8}{'FAR':>8}{'CSI':>8}")
        print("  " + "-" * 87)
        for label, rows in states.items():
            tot[(name, label)] = show(label, rows, ys)

    print("\n  === NET CHANGE, baseline -> both fixes ===")
    for name, ys in (("ALL 45 YEARS", years), ("HELD-OUT 5", held),
                     ("TRAINING 40", train)):
        key = name if name != "ALL 45 YEARS" else "ALL 45 YEARS (fit, not skill)"
        key = {"ALL 45 YEARS": "ALL 45 YEARS (fit, not skill)",
               "HELD-OUT 5": "HELD-OUT 5 YEARS (skill)",
               "TRAINING 40": "TRAINING 40 YEARS"}[name]
        b = tot[(key, "baseline  season + 48h")]
        n = tot[(key, "fix1+2    calendar+24h")]
        print(f"  {name:<16}hits {b['hits']:>5} -> {n['hits']:<5} "
              f"({n['hits']-b['hits']:+d})   "
              f"misses {b['misses']:>5} -> {n['misses']:<5} "
              f"({n['misses']-b['misses']:+d})   "
              f"POD {b['pod']:.3f} -> {n['pod']:.3f}   "
              f"CSI {b['csi']:.3f} -> {n['csi']:.3f}   "
              f"FAR {b['far']:.3f} -> {n['far']:.3f}")

    # Per-year regression check: did ANY year get worse?
    base = states["baseline  season + 48h"]
    new = states["fix1+2    calendar+24h"]
    worse = [y for y in years
             if base.get(y) and new.get(y) and new[y]["pod"] < base[y]["pod"]]
    print(f"\n  years where POD went DOWN: {len(worse)}/{len(years)}"
          + (f"  {worse}" if worse else ""))
    csi_worse = [y for y in years
                 if base.get(y) and new.get(y) and new[y]["csi"] < base[y]["csi"]]
    print(f"  years where CSI went DOWN: {len(csi_worse)}/{len(years)}"
          + (f"  {csi_worse}" if csi_worse else ""))

    with open(args.json, "w") as f:
        json.dump({"years": years, "held_out": held,
                   "per_year": {k: {str(y): v for y, v in rows.items()}
                                for k, rows in states.items()},
                   "totals": {f"{a}|{b}": v for (a, b), v in tot.items()}},
                  f, indent=1)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
