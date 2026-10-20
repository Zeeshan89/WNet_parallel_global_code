#!/usr/bin/env python3
"""
phase4_etc_tracker_design.py — choose an ETC tracker configuration against TWO
independent benchmarks on common ground.

Why two benchmarks and a common box
-----------------------------------
No ETC reference is global, so a tracker cannot be designed against one alone:

  TempestExtremes ETC   lon -135..-55, lat 10..55   physics tracker, CONUS only
  NAEC                  lon -180..0,   lat 20..80   observational, NH-western

They also disagree on how many ETCs exist — in the common window NAEC carries
roughly twice TE's count — because NAEC includes weaker, shorter systems. A
configuration that looks good against one and bad against the other is overfitted
to that reference, which is exactly what having both is for.

Everything is therefore scored inside COMMON_BOX (lon -135..-55, lat 20..55), the
intersection, and with the SYMMETRIC QC of etc_analysis/global_etc_analysis.py
(48 h, 8 points, 1000 km net displacement) applied to model and reference alike.
Without that QC, NAEC storms this tracker structurally cannot emit (it enforces
48 h and an 1800 km path) would be counted as misses it never had a chance at.

What is being decided
---------------------
The mountain mask, which acts DURING tracking and so needs one run per variant:

  A  frame ON,  frac 0.30   current default
  B  frame OFF, frac 0.30   allow genesis over terrain, keep the track rule
  C  frame OFF, frac 0.70   reject only lows that spend most of their life there
  D  frame OFF, frac 1.00   mask disabled entirely

and min_track_length, which is post-hoc and so is swept per variant for free.

The mask targets a real artefact (sea-level-reduced MSLP over high terrain, plus
an 850 mb level that is underground there), but it is geographic and cannot tell a
lee cyclone from an artefact. Against TE 2000-2010 it discards ~19% of real ETCs
and 18.6% of real ETCs have genesis inside a box. The tracker already carries
propagation filters (min_avg_propagation_ms, min_net_east_km, retrograde limits)
that target stationary lows directly, so the question is whether the mask still
earns its cost once those are doing their job.

Usage
  phase4_etc_tracker_design.py --years 2000 ... 2010
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase4_evaluate_tracks as E   # noqa: E402
from plot_tracks_peryear import parse_etc_txt, clip_to_box   # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "ARs_Hurricanes_ETCs_MCS", "Phase2_C200")
VARIANTS = [
    ("A frame1 frac0.30 (current)", os.path.join(BASE, "mask_sweep", "maskA_frame1_frac030")),
    ("B frame0 frac0.30",           os.path.join(BASE, "mask_sweep", "maskB_frame0_frac030")),
    ("C frame0 frac0.70",           os.path.join(BASE, "mask_sweep", "maskC_frame0_frac070")),
    ("D mask off",                  os.path.join(BASE, "mask_sweep", "maskD_off")),
]
MIN_LENS = [8, 10, 12, 14]


def score_variant(tdir, years, ref_name, min_points, qc=True):
    """Score one variant against one reference inside COMMON_BOX."""
    hits = misses = fa = dup = n_model = n_ref = 0
    for y in years:
        model = parse_etc_txt(os.path.join(
            tdir, "etc", f"ETC_tracks_{y}_meteorological_v2.txt"))
        ref = E.parse_naec(y) if ref_name == "naec" else E.parse_te_etc(y)
        if not model or not ref:
            continue
        model, _ = clip_to_box(model, E.COMMON_BOX)
        ref, _ = clip_to_box(ref, E.COMMON_BOX)
        if qc:
            model = E.qc_filter(model)
            ref = E.qc_filter(ref)
        if min_points:
            model = [t for t in model if len(t) >= min_points]
        if not ref:
            continue
        r = E.score(model, ref, 300.0, 2, 3)
        if r is None:
            continue
        hits += r["hits"]; misses += r["misses"]; fa += r["false_alarms"]
        dup += r["duplicates"]; n_model += r["n_model"]; n_ref += r["n_ref"]
    den = hits + misses + fa
    return dict(n_model=n_model, n_ref=n_ref, hits=hits, misses=misses,
                false_alarms=fa, duplicates=dup,
                pod=hits / n_ref if n_ref else 0.0,
                far=fa / n_model if n_model else 0.0,
                csi=hits / den if den else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=list(range(2000, 2011)))
    ap.add_argument("--min-lens", nargs="+", type=int, default=MIN_LENS)
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--json", default="phase4_results/etc_tracker_design.json")
    ap.add_argument("--sweep-dir",
                    help="score every sub-directory that holds an etc/ folder instead of the four "
                         "mountain-mask variants (Run_etc_tracker_sweep.sh); give --json as well, "
                         "or the mask-sweep results are overwritten")
    args = ap.parse_args()
    qc = not args.no_qc

    variants = VARIANTS
    if args.sweep_dir:
        variants = [(d, os.path.join(args.sweep_dir, d)) for d in sorted(os.listdir(args.sweep_dir))]
    avail = [(lab, d) for lab, d in variants
             if os.path.isdir(os.path.join(d, "etc"))]
    if not avail:
        raise SystemExit(f"no variant track dirs under {args.sweep_dir or BASE + '/mask_sweep'}")

    print(f"\n  years {args.years[0]}-{args.years[-1]}   "
          f"domain COMMON_BOX lon {E.COMMON_BOX['lon']} lat {E.COMMON_BOX['lat']}")
    print(f"  symmetric QC: {'ON (48 h, 8 pts, 1000 km net) applied to BOTH sides' if qc else 'OFF'}")

    out = {}
    for ref_name, ref_lab in (("te", "TempestExtremes"), ("naec", "NAEC")):
        print(f"\n  ===== reference: {ref_lab} =====")
        print(f"  {'variant':<28}{'minlen':>7}{'trk':>6}{'ref':>6}{'hits':>6}"
              f"{'FA':>6}{'dup':>5}{'POD':>7}{'FAR':>7}{'CSI':>7}")
        print("  " + "-" * 85)
        for lab, tdir in avail:
            for ml in args.min_lens:
                r = score_variant(tdir, args.years, ref_name, ml, qc=qc)
                out[f"{ref_name}|{lab}|{ml}"] = r
                print(f"  {lab if ml == args.min_lens[0] else '':<28}{ml:>7}"
                      f"{r['n_model']:>6}{r['n_ref']:>6}{r['hits']:>6}"
                      f"{r['false_alarms']:>6}{r['duplicates']:>5}"
                      f"{r['pod']:>7.3f}{r['far']:>7.3f}{r['csi']:>7.3f}")
            print()

    # Rank by the WORST of the two CSIs, so a config cannot win by overfitting one
    # reference. A tracker that only satisfies TE or only NAEC is not a tracker we
    # can trust outside the window that reference happens to cover.
    print("  ===== ranked by the WORSE of the two CSIs =====")
    print(f"  {'variant':<28}{'minlen':>7}{'CSI(TE)':>9}{'CSI(NAEC)':>11}"
          f"{'min':>7}{'POD(TE)':>9}{'POD(NAEC)':>11}")
    print("  " + "-" * 84)
    rows = []
    for lab, _ in avail:
        for ml in args.min_lens:
            t = out[f"te|{lab}|{ml}"]; n = out[f"naec|{lab}|{ml}"]
            rows.append((min(t["csi"], n["csi"]), lab, ml, t, n))
    for worst, lab, ml, t, n in sorted(rows, reverse=True):
        print(f"  {lab:<28}{ml:>7}{t['csi']:>9.3f}{n['csi']:>11.3f}"
              f"{worst:>7.3f}{t['pod']:>9.3f}{n['pod']:>11.3f}")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"years": args.years, "qc": qc,
                       "common_box": E.COMMON_BOX, "results": out}, f, indent=1)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
