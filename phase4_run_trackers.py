#!/usr/bin/env python3
"""
phase4_run_trackers.py — run the EXISTING trackers on Phase 2 detections.

Imports HurricaneTracker and ETCTracker from the production scripts rather than
copying or editing them. Their main() functions hardcode the old trial paths, so
this reproduces the same calls with the new input/output directories and the test
year range. The tracking rules themselves are untouched, so any difference in the
resulting tracks is attributable to the detections alone — which is the whole
point of the comparison.

Usage
  python3 phase4_run_trackers.py --det-dir <exported nc dir> --out-dir <tracks> \
      --years 1983 1994 2003 2012 2021 --which hurricane etc
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ERA5_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
VORTICITY_DIR = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                 "vor850_latlongrid/vor850_latlongrid/")
WIND_SPEED = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
              "wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc")


def run_hurricane(det_dir, out_dir, years):
    from Hurricane_track import HurricaneTracker
    os.makedirs(out_dir, exist_ok=True)
    tracker = HurricaneTracker(
        hurricane_data_dir=det_dir,
        era5_data_dir=ERA5_DIR,
        vorticity_dir=VORTICITY_DIR,
        wind_speed_path=WIND_SPEED,
        output_dir=out_dir,
    )
    # run_tracking takes a contiguous range; the test years are not contiguous,
    # so call it once per year.
    for y in years:
        print(f"\n===== Hurricane tracking {y} =====")
        try:
            tracker.run_tracking(start_year=y, end_year=y)
        except Exception as e:
            import traceback
            print(f"! hurricane {y} failed: {e}")
            traceback.print_exc()
    # Which criterion discarded how many candidate tracks. With the IBTrACS-target
    # model the detection field is ~24x cleaner than the one these thresholds were
    # tuned against, so this says directly where recall is being lost.
    rej = getattr(HurricaneTracker, "REJECT", {})
    if rej:
        tot = sum(rej.values())
        print(f"\n=== track rejections by criterion ({tot} total) ===")
        for k, v in sorted(rej.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<16} {v:>6}  ({v/tot:.1%})")


def run_etc(det_dir, out_dir, years):
    from updated_monthly_ETC_Tracker import ETCTracker
    os.makedirs(out_dir, exist_ok=True)
    # Same construction as updated_monthly_ETC_Tracker.main()
    tracker = ETCTracker(
        max_distance_km=800,
        min_track_length=8,
        # The shipped +-70 matches the latitude band of the ETC label (dataset.py:750-753):
        # widening it here only has an effect once the label itself reaches further.
        region_bounds={"lon_min": -180.0, "lon_max": 180.0,
                       "lat_min": float(os.environ.get("ETC_LAT_MIN", "-70.0")),
                       "lat_max": float(os.environ.get("ETC_LAT_MAX", "70.0"))},
    )
    files = sorted(glob.glob(os.path.join(det_dir, "era5dara_*_ETCs_detection.nc")))
    by_year = {}
    for f in files:
        try:
            yr = int(os.path.basename(f).split("_")[1][:4])
        except (ValueError, IndexError):
            continue
        if yr in years:
            by_year.setdefault(yr, []).append(f)

    for y in sorted(by_year):
        print(f"\n===== ETC tracking {y} ({len(by_year[y])} files) =====")
        out_path = os.path.join(out_dir, f"ETC_tracks_{y}_meteorological_v2.txt")
        try:
            n = tracker.process_multiple_months(sorted(by_year[y]), out_path)
            print(f"  {y}: {n} tracks -> {out_path}")
        except Exception as e:
            import traceback
            print(f"! etc {y} failed: {e}")
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--years", nargs="+", type=int,
                    default=[1983, 1994, 2003, 2012, 2021])
    ap.add_argument("--which", nargs="+", default=["hurricane", "etc"],
                    choices=["hurricane", "etc"])
    args = ap.parse_args()

    print(f"detections : {args.det_dir}")
    print(f"tracks out : {args.out_dir}")
    print(f"years      : {args.years}")

    if "hurricane" in args.which:
        run_hurricane(args.det_dir, os.path.join(args.out_dir, "hurricane"),
                      args.years)
    if "etc" in args.which:
        run_etc(args.det_dir, os.path.join(args.out_dir, "etc"), args.years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
