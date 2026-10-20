#!/usr/bin/env python3
"""
phase4_tc_peryear_table.py — one row per year, OLD vs NEW TC tracks, 1980-2024.

Both products are scored against the SAME reference (calendar-year keyed, the
corrected convention), so the per-year delta is attributable to the model and the
tracker alone and not to the counting fix. The counting fix was measured separately
and is not included here.

  OLD  tracks_final          cap-8 model  (scalar pos_weight), 48 h duration gate
  NEW  tracks_pw50_e200      cap-50 model (epoch 13),          24 h duration gate

CAVEAT ON READING THIS TABLE. IBTrACS is this model's TRAINING TARGET
(phase4_build_ibtracs_label.py wrote it into tensor_cache_v5_ibtracs), so only
1983, 1994, 2003, 2012 and 2021 are held out. Rows are flagged accordingly: the
40 unflagged years are a FIT statistic, not skill. Use the year-by-year numbers to
see where the gain is distributed, and the HELD-OUT subtotal as the skill estimate.

Usage
  phase4_tc_peryear_table.py
  phase4_tc_peryear_table.py --old <dir> --new <dir> --csv out.csv
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_recover import (            # noqa: E402
    load_model_tracks, score, pool, ref_for,
)

TEST_YEARS = {1983, 1994, 2003, 2012, 2021}
P2 = "ARs_Hurricanes_ETCs_MCS/Phase2_E_ibtracs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=f"{P2}/tracks_final")
    ap.add_argument("--new", default=f"{P2}/tracks_pw50_e200")
    ap.add_argument("--y0", type=int, default=1980)
    ap.add_argument("--y1", type=int, default=2024)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--csv", default="phase4_results/tc_peryear_old_vs_new.csv")
    args = ap.parse_args()

    years = list(range(args.y0, args.y1 + 1))
    print(f"  OLD : {args.old}")
    print(f"  NEW : {args.new}")
    print(f"  reference: calendar-year keyed, >= {args.wind_kt:g} kt, "
          f"match radius {args.radius_km:g} km\n")

    rows = []
    for y in years:
        ref, _ = ref_for(y, "calendar", args.wind_kt)
        o = score(load_model_tracks(args.old, y), ref, args.radius_km)
        n = score(load_model_tracks(args.new, y), ref, args.radius_km)
        if o is None or n is None:
            continue
        rows.append({"year": y, "held_out": y in TEST_YEARS,
                     "n_ref": n["n_ref"],
                     "old_tracks": o["n_model"], "old_hits": o["hits"],
                     "old_miss": o["misses"], "old_fa": o["false_alarms"],
                     "old_pod": o["pod"], "old_far": o["far"], "old_csi": o["csi"],
                     "new_tracks": n["n_model"], "new_hits": n["hits"],
                     "new_miss": n["misses"], "new_fa": n["false_alarms"],
                     "new_pod": n["pod"], "new_far": n["far"], "new_csi": n["csi"],
                     "d_hits": n["hits"] - o["hits"],
                     "d_pod": n["pod"] - o["pod"],
                     "d_csi": n["csi"] - o["csi"]},)
        if len(rows) == 1:
            print(f"  {'year':<6}{'ref':>5}{'OLDtrk':>8}{'OLDhit':>7}{'OLDpod':>8}"
                  f"{'NEWtrk':>8}{'NEWhit':>7}{'NEWpod':>8}"
                  f"{'+hits':>7}{'dPOD':>8}{'dCSI':>8}", flush=True)
        r = rows[-1]
        print(f"  {r['year']:<6}{r['n_ref']:>5}"
              f"{r['old_tracks']:>8}{r['old_hits']:>7}{r['old_pod']:>8.3f}"
              f"{r['new_tracks']:>8}{r['new_hits']:>7}{r['new_pod']:>8.3f}"
              f"{r['d_hits']:>+7d}{r['d_pod']:>+8.3f}{r['d_csi']:>+8.3f}"
              f"{' *' if r['held_out'] else ''}", flush=True)

    hdr = (f"\n  {'year':<6}{'ref':>5}"
           f"{'OLDtrk':>8}{'OLDhit':>7}{'OLDpod':>8}"
           f"{'NEWtrk':>8}{'NEWhit':>7}{'NEWpod':>8}"
           f"{'+hits':>7}{'dPOD':>8}{'dCSI':>8}  ")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for r in rows:
        flag = " *" if r["held_out"] else ""
        print(f"  {r['year']:<6}{r['n_ref']:>5}"
              f"{r['old_tracks']:>8}{r['old_hits']:>7}{r['old_pod']:>8.3f}"
              f"{r['new_tracks']:>8}{r['new_hits']:>7}{r['new_pod']:>8.3f}"
              f"{r['d_hits']:>+7d}{r['d_pod']:>+8.3f}{r['d_csi']:>+8.3f}{flag}")

    def block(label, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        o = pool([{k[4:]: r[k] for k in r if k.startswith("old_")}
                  | {"n_ref": r["n_ref"], "duplicates": 0} for r in rs])
        n = pool([{k[4:]: r[k] for k in r if k.startswith("new_")}
                  | {"n_ref": r["n_ref"], "duplicates": 0} for r in rs])
        print(f"  {label:<6}{n['n_ref']:>5}"
              f"{o['n_model']:>8}{o['hits']:>7}{o['pod']:>8.3f}"
              f"{n['n_model']:>8}{n['hits']:>7}{n['pod']:>8.3f}"
              f"{n['hits']-o['hits']:>+7d}{n['pod']-o['pod']:>+8.3f}"
              f"{n['csi']-o['csi']:>+8.3f}")
    print("  " + "-" * (len(hdr) - 3))
    block("ALL45", lambda r: True)
    block("HELD*", lambda r: r["held_out"])
    block("train", lambda r: not r["held_out"])
    print("\n  * = held out of training. The other 40 years had IBTrACS as their "
          "training target, so their numbers are fit, not skill.")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
