#!/usr/bin/env python3
"""
plot_tracks_filtered.py — the tracks_peryear figure, with the post-tracker filter
applied, so the filter's effect is visible rather than only tabulated.

Same layout, colour convention and matching rule as plot_tracks_peryear.py, whose
drawing and parsing code this imports rather than duplicating: every benchmark
storm gets a colour, a model track inherits the colour of the benchmark storm it
matches most often, and unmatched model tracks are grey. Grey is therefore the
false alarms, and the filter is working when grey disappears and colour does not.

  --mode filter   new RAW | new FILTERED | IBTrACS      (what the filter removed)
  --mode compare  old      | new FILTERED | IBTrACS      (matches tracks_peryear_tc)

Filters are applied to the new model's tracks only. The benchmark and the old
system are drawn unfiltered — filtering the baseline too would flatter the
comparison.

Environment: /ccs/home/zaslam/.conda/envs/my_env/bin/python

Usage
  plot_tracks_filtered.py --mode filter \
      --min-duration-h 96 --min-straightness 0.4 --genesis-lat-max 20
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_tracks_peryear import (        # noqa: E402
    parse_hurricane_txt, ibtracs_tracks, match_tracks, clip_to_box,
    palette, draw, OLD, NEW, TEST_YEARS, INK,
)
from phase4_evaluate_tracks import filter_tracks, TE_BOX   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["filter", "compare"], default="filter")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--track-dir", default=NEW)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--te-domain", action="store_true")
    ap.add_argument("--outdir", default="phase4_figures")
    ap.add_argument("--tag", default=None, help="suffix for the output filename")

    ap.add_argument("--min-points", type=int, default=None)
    ap.add_argument("--min-duration-h", type=float, default=96.0)
    ap.add_argument("--min-path-km", type=float, default=None)
    ap.add_argument("--min-straightness", type=float, default=0.4)
    ap.add_argument("--min-speed-ms", type=float, default=None)
    ap.add_argument("--max-speed-ms", type=float, default=None)
    ap.add_argument("--genesis-lat-max", type=float, default=20.0)
    ap.add_argument("--max-abs-lat", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    box = TE_BOX if args.te_domain else None
    years = args.years or list(TEST_YEARS)

    bits = []
    if args.min_duration_h:
        bits.append(f"duration>={args.min_duration_h:g}h")
    if args.min_straightness:
        bits.append(f"straightness>={args.min_straightness:g}")
    if args.genesis_lat_max:
        bits.append(f"genesis|lat|<={args.genesis_lat_max:g}")
    if args.min_path_km:
        bits.append(f"path>={args.min_path_km:g}km")
    filt_txt = ", ".join(bits) if bits else "none"

    loaded = []
    for y in years:
        ref = ibtracs_tracks(y, wind_kt=args.wind_kt)
        new = parse_hurricane_txt(os.path.join(
            args.track_dir, "hurricane", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        old = parse_hurricane_txt(os.path.join(
            OLD, "Hurricanes_WNet_final", f"CONUS_year_{y}",
            f"CONUS_hurricane_tracks_detailed_{y}.txt"))
        if box:
            ref, _ = clip_to_box(ref, box)
            new, _ = clip_to_box(new, box)
            old, _ = clip_to_box(old, box)
        if not new or not ref:
            continue
        new_f, _ = filter_tracks(new, args)
        loaded.append((y, ref, new, new_f, old))

    if not loaded:
        raise SystemExit("no years with tracks — check --track-dir")
    years = [t[0] for t in loaded]
    print(f"[tc] years: {years}")

    fig, axes = plt.subplots(len(loaded), 3, figsize=(19, 3.05 * len(loaded)),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.945, bottom=0.02,
                        hspace=0.22, wspace=0.07)

    summary = []
    for r, (y, ref, new, new_f, old) in enumerate(loaded):
        cols = palette(len(ref))
        a_new = match_tracks(new, ref, args.radius_km)
        a_newf = match_tracks(new_f, ref, args.radius_km)
        a_old = match_tracks(old, ref, args.radius_km)

        c_new = [None if j is None else cols[j] for j in a_new]
        c_newf = [None if j is None else cols[j] for j in a_newf]
        c_old = [None if j is None else cols[j] for j in a_old]
        c_ref = [cols[j] for j in range(len(ref))]

        h_new = len({j for j in a_new if j is not None})
        h_newf = len({j for j in a_newf if j is not None})
        h_old = len({j for j in a_old if j is not None})

        if args.mode == "filter":
            left, c_left = new, c_new
            ltitle = (f"WNet new, RAW — {len(new)} tracks, "
                      f"{h_new}/{len(ref)} storms found")
        else:
            left, c_left = old, c_old
            ltitle = (f"WNet old — {len(old)} tracks, "
                      f"{h_old}/{len(ref)} storms found")

        draw(axes[r][0], left, c_left, ltitle, box)
        draw(axes[r][1], new_f, c_newf,
             f"WNet new, FILTERED — {len(new_f)} tracks, "
             f"{h_newf}/{len(ref)} storms found", box)
        draw(axes[r][2], ref, c_ref,
             f"IBTrACS (observed) — {len(ref)} storms", box)

        axes[r][0].text(-0.055, 0.5, str(y), transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold", color=INK)
        summary.append((y, len(ref), len(new), h_new, len(new_f), h_newf,
                        len(old), h_old))

    fig.suptitle(
        "Tropical cyclone tracks by year — colour = matched IBTrACS storm "
        "(same colour in every panel); grey = no match, i.e. a false alarm\n"
        f"post-tracker filter on the new model: {filt_txt}",
        fontsize=11.5, color=INK, x=0.045, ha="left", y=0.992)

    tag = args.tag or args.mode
    out = os.path.join(args.outdir, f"tracks_peryear_tc_{tag}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    plt.close(fig)

    print(f"\n  {'year':<7}{'storms':>8}{'raw trk':>9}{'raw found':>11}"
          f"{'filt trk':>10}{'filt found':>12}{'old trk':>9}{'old found':>11}")
    print("  " + "-" * 79)
    for (y, nr, nn, hn, nf, hf, no, ho) in summary:
        print(f"  {y:<7}{nr:>8}{nn:>9}{f'{hn}/{nr}':>11}{nf:>10}"
              f"{f'{hf}/{nr}':>12}{no:>9}{f'{ho}/{nr}':>11}")
    tot = [sum(s[i] for s in summary) for i in range(1, 8)]
    print("  " + "-" * 79)
    print(f"  {'ALL':<7}{tot[0]:>8}{tot[1]:>9}{f'{tot[2]}/{tot[0]}':>11}"
          f"{tot[3]:>10}{f'{tot[4]}/{tot[0]}':>12}{tot[5]:>9}"
          f"{f'{tot[6]}/{tot[0]}':>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
