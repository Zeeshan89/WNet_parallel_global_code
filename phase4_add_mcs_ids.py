#!/usr/bin/env python3
"""
phase4_add_mcs_ids.py — add an `MCS_id` storm-track variable next to `MCS_detection` in the
exported MCS files, so one file carries both the mask and the identity of each system.

`MCS_detection` says whether a pixel belongs to a mesoscale convective system at that time; it
cannot say whether the system at 06 UTC is the same system as at 12 UTC. Storm-hours, lifetime
and any per-system statistic need that identity. `post_processing.track_mcs_timeseries` supplies
it: within one monthly file it labels the blobs of each frame (merging across the +-180 seam),
links a blob to the previous-frame blob it overlaps most when that overlap reaches
`link_overlap` of the smaller blob, and writes the resulting track number into every pixel of
the system (0 where there is none). This is Feng's `cloudtracknumber` convention, which is what
the shared MCS benchmark metric expects.

Parameters match `test.py`, the code behind the published MCS evaluation, so the IDs here mean
the same thing: overlap-only, with NO minimum-area and NO minimum-lifetime pruning
(`min_area_km2=None`, `link_overlap=0.5`, `min_lifetime_steps=1`). With those settings no blob
is ever discarded, so `MCS_id > 0` covers exactly the pixels where `MCS_detection > 0.5`;
`MCS_detection` itself is never modified.

IDs restart at 1 in every monthly file, so a system is identified by (file, id). A system
crossing a month boundary appears as a new id in the later file, which is what the monthly
storm-hours metric assumes.

  srun -n 96 phase4_add_mcs_ids.py --dir <.../MCS_WNet_final>
  phase4_add_mcs_ids.py --dir <...> --check          report which files have MCS_id
"""
import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
DEFAULT_DIR = os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_finalized_v3", "MCS_WNet_final")
LINK_OVERLAP = 0.5          # as in test.py
MIN_LIFETIME_STEPS = 1      # as in test.py: no lifetime pruning
MIN_AREA_KM2 = None         # as in test.py: raw model blobs, no area criterion


def add_ids(path, overwrite=False):
    """Append MCS_id to one monthly file. Returns a one-line report."""
    from netCDF4 import Dataset
    from post_processing import track_mcs_timeseries

    with Dataset(path) as ds:
        if "MCS_id" in ds.variables and not overwrite:
            return f"{os.path.basename(path)}: already has MCS_id"
        ds.set_auto_mask(False)
        det = ds.variables["MCS_detection"][:]
        lat = ds.variables["latitude"][:].astype(float)
        lon = ds.variables["longitude"][:].astype(float)

    binary, ids = track_mcs_timeseries(
        det > 0.5, lat, lon, min_area_km2=MIN_AREA_KM2, link_overlap=LINK_OVERLAP,
        min_lifetime_steps=MIN_LIFETIME_STEPS, verbose=False)

    # With these parameters nothing is filtered out, so the tracked mask must equal the input.
    # If that ever stops holding, the file would silently disagree with itself; fail instead.
    same = int(((binary > 0) != (det > 0.5)).sum())
    if same:
        raise SystemExit(f"{path}: tracking changed {same} pixels of the mask; refusing to write")

    with Dataset(path, "a") as ds:
        if "MCS_id" in ds.variables:
            v = ds.variables["MCS_id"]
        else:
            v = ds.createVariable("MCS_id", "i4", ("time", "latitude", "longitude"),
                                  zlib=True, complevel=4, fill_value=0)
            v.long_name = "MCS storm track identifier"
            v.description = ("track number of the mesoscale convective system occupying this "
                             "pixel; 0 where there is none. Overlap tracking within this monthly "
                             "file, so a system is identified by (file, id) and ids restart each "
                             "month. Nonzero exactly where MCS_detection > 0.5.")
            v.units = "1"
            v.link_overlap = LINK_OVERLAP
            v.min_lifetime_steps = MIN_LIFETIME_STEPS
            v.min_area_km2 = "none (raw model blobs)"
            v.convention = "Feng et al. cloudtracknumber"
            v.source = "phase4_add_mcs_ids.py / post_processing.track_mcs_timeseries"
        v[:] = ids
    n_tracks = int(ids.max())
    return (f"{os.path.basename(path)}: {n_tracks} tracks, "
            f"{100.0 * (ids > 0).mean():.3f}% of pixels labelled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--check", action="store_true", help="report coverage, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*_MCS_detection.nc")))
    if not files:
        raise SystemExit(f"no MCS files in {args.dir}")

    if args.check:
        from netCDF4 import Dataset
        have = 0
        for p in files:
            with Dataset(p) as ds:
                have += "MCS_id" in ds.variables
        print(f"{args.dir}: {have}/{len(files)} files have MCS_id")
        if have:
            with Dataset(next(p for p in files if "MCS_id" in Dataset(p).variables)) as ds:
                for n in ("MCS_detection", "MCS_id"):
                    v = ds.variables[n]
                    print(f"  {n:<16} {v.dtype} {v.shape} {getattr(v, 'long_name', '')}")
        return 0

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    for p in files[rank::world]:
        print("  " + add_ids(p, overwrite=args.overwrite), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
