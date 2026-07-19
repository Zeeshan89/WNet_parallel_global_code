#!/usr/bin/env python3
"""
DEPRECATED — do not run this on output from the precipitation-based MCS branch.

test.py now applies the full criteria inline (min_area_km2=60000.0 AND
min_lifetime_steps=2, in the correct order, inside track_mcs_timeseries), so its
output already satisfies the area criterion. This script is NOT equivalent to
that path: it applies the 60k area filter post-hoc but never applies the
minimum-lifetime criterion. Running it afterwards would be near-idempotent on
area while silently implying the two pipelines are comparable when their
lifetime criteria differ.

Kept only for reprocessing pre-existing BT-branch outputs in Output_new_IWV.
For anything new, point the metric pipeline at test.py's output directly.

---
Apply a 60,000 km^2 minimum-area criterion to existing IWV-gate MCS predictions
WITHOUT re-running the GPUs. Reads *_MCS_detection.nc from IN_DIR, drops per-timestep
tracked blobs (by MCS_id) whose latitude-weighted area < MIN_AREA_KM2, and writes the
filtered MCS files (MCS_detection + MCS_id) to OUT_DIR.

Usage:
  python apply_min_area.py                 # all files in IN_DIR
  python apply_min_area.py 2010 2011       # only these years
Then point the metric pipeline at OUT_DIR (see notes printed at the end).
"""
import os, sys, glob
import numpy as np
import xarray as xr
from scipy import ndimage

IN_DIR  = "ARs_Hurricanes_ETCs_MCS/Output_new_IWV"
OUT_DIR = "ARs_Hurricanes_ETCs_MCS/Output_new_IWV_60k"
MIN_AREA_KM2 = 60000.0
R = 6371.0

def pixel_area(lats, lons):
    dlat = np.deg2rad(abs(np.mean(np.diff(lats))))
    dlon = np.deg2rad(abs(np.mean(np.diff(lons))))
    cell = (R ** 2) * dlat * dlon * np.cos(np.deg2rad(lats))
    return np.repeat(np.clip(cell, 0, None)[:, None], lons.size, axis=1).astype(np.float32)

def main():
    years = set(sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, "*_MCS_detection.nc")))
    if years:
        files = [f for f in files if any(f"era5dara_{y}" in os.path.basename(f) for y in years)]
    print(f"{len(files)} MCS files -> {OUT_DIR}  (min area {MIN_AREA_KM2:.0f} km^2)")

    for f in files:
        ds = xr.open_dataset(f)
        la = "latitude" if "latitude" in ds.coords else "lat"
        lo = "longitude" if "longitude" in ds.coords else "lon"
        area2d = pixel_area(ds[la].values, ds[lo].values)
        ids = ds["MCS_id"].values.astype(np.int32).copy()
        for t in range(ids.shape[0]):
            u = np.unique(ids[t]); u = u[u > 0]
            if u.size:
                a = np.asarray(ndimage.sum(area2d, ids[t], u))
                drop = u[a < MIN_AREA_KM2]
                if drop.size:
                    ids[t][np.isin(ids[t], drop)] = 0
        out = ds.copy()
        out["MCS_id"] = (ds["MCS_id"].dims, ids)
        out["MCS_detection"] = (ds["MCS_detection"].dims, (ids > 0).astype(np.float32))
        out.to_netcdf(os.path.join(OUT_DIR, os.path.basename(f)))
        ds.close(); out.close()
        print("  wrote", os.path.basename(f))

    print(f"\nDone. Next:")
    print(f"  1) Edit mcs-wnet-newfinalcode.py:  MCS_DIR -> {OUT_DIR};  OUTPUT_DIR -> outputs_wnet_standard_IWV_60k")
    print(f"  2) python mcs-wnet-newfinalcode.py")
    print(f"  3) Edit plot_wnet_newcode_final.py base dir -> outputs_wnet_standard_IWV_60k ; run it")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
