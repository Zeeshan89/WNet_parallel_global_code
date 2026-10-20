#!/usr/bin/env python3
"""
phase4_tc_centres_cache.py — dump one year's tracker CENTRES to a small npz.

Every gate sweep re-runs create_hurricane_tracks, and the expensive part is not
the linking (milliseconds) but re-reading twelve 721x1440 monthly NetCDFs to
rebuild the detection table (minutes). The centres depend only on the detection
files and the area gate, never on the linking or validation thresholds, so they
are computed once here and reused.

Produces exactly the columns create_hurricane_tracks consumes. Met fields are
omitted: they are written to the output text files but never read by the linker
or by validate_hurricane_track.

Usage
  phase4_tc_centres_cache.py --year 2003 [--outdir phase4_results/centres]
"""
import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase4_tc_miss_taxonomy as _tax        # noqa: E402

OUTDIR = "phase4_results/centres"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--det-dir", default=_tax.DET_DIR,
                    help="detection NetCDF dir; centres depend on the detections "
                         "and the area gate only, never on the linking gates, so "
                         "one cache serves any number of gate configurations")
    ap.add_argument("--max-area-px", type=int,
                    default=int(os.environ.get("TC_MAX_AREA_PX", "2000")))
    args = ap.parse_args()

    os.environ["TC_MAX_AREA_PX"] = str(args.max_area_px)
    DET_DIR = args.det_dir
    import xarray as xr
    from Hurricane_track import HurricaneTracker

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"centres_{args.year}.npz")
    if os.path.exists(out):
        print(f"  {args.year}: already cached")
        return 0

    tracker = HurricaneTracker(
        hurricane_data_dir=DET_DIR,
        era5_data_dir="/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/",
        vorticity_dir="", wind_speed_path="",
        output_dir="/tmp/tc_centres_cache_unused")

    ts, la, lo, ar, mx = [], [], [], [], []
    for fp in sorted(glob.glob(os.path.join(
            DET_DIR, f"era5dara_{args.year:04d}*_Hurricane_detection.nc"))):
        ds = xr.open_dataset(fp)
        lats = ds.latitude.values
        lons = ds.longitude.values
        var = ds["Hurricane_detection"]
        for i, t in enumerate(ds.time.values):
            for h in tracker.extract_hurricane_centers(var.values[i], lats, lons):
                ts.append(np.datetime64(t, "s"))
                la.append(h["center_lat"])
                lo.append(h["center_lon"])
                ar.append(h["area_pixels"])
                mx.append(h["max_intensity"])
        ds.close()
        print(f"    {os.path.basename(fp)}: {len(ts)} centres so far", flush=True)

    np.savez_compressed(out,
                        timestamp=np.array(ts, dtype="datetime64[s]"),
                        center_lat=np.array(la, dtype=np.float64),
                        center_lon=np.array(lo, dtype=np.float64),
                        area_pixels=np.array(ar, dtype=np.int32),
                        max_intensity=np.array(mx, dtype=np.float32))
    print(f"  {args.year}: {len(ts)} centres -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
