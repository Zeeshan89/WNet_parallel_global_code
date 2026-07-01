#!/usr/bin/env python3
"""
compute_ar_duration.py — mean AR *event* duration (days) per cell + zonal profile,
for MARS and WNet AR_detection, from the daily AR-presence files.

mean duration = (total AR-days) / (number of AR events), where an event is a 0->1
transition in the daily binary AR-presence series. Saved small so the figure can
load it instead of re-reading the ~68 GB daily files.

Outputs: ar_duration.npz  (lat, dur_mars[lat,lon], dur_wnet[lat,lon],
                            zon_mars[lat], zon_wnet[lat])
"""
import os
import numpy as np
import xarray as xr
import dask

HERE = os.path.dirname(os.path.abspath(__file__))
# Daily AR-presence files: per-cell mean event duration in DAYS (option A).
# Each frame = 1 day, so an event lasts a whole number of days (>= 1).
MARS_DAILY = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/Analysis_MARS/MARS_daily.nc"
WNET_DAILY = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
              "Global_ARs_Hurr_ETCs_MCS_finalized/ARs_WNet_final/Analysis/AR_detection_daily.nc")


def mean_duration(path):
    ds = xr.open_dataset(path, chunks={"time": 300})
    var = next(v for v in ds.data_vars if v != "time_bnds")
    lat = ds["latitude"].values
    p = (ds[var] > 0.5)
    pi = p.astype("int8")
    total = p.sum("time")                        # total AR-days per cell
    rising = (pi.diff("time") == 1).sum("time")   # number of events (0->1 edges)
    total, rising = dask.compute(total, rising)   # single fused streaming pass
    ds.close()
    total = np.asarray(total.values, float)
    rising = np.asarray(rising.values, float)
    dur = np.where(rising > 0, total / rising, np.nan)  # days per event
    zon = np.nanmean(dur, axis=1)                        # zonal mean vs latitude
    return lat, dur.astype(np.float32), zon.astype(np.float32)


def main():
    print("MARS duration (daily, per-cell) ...", flush=True)
    lat, dur_m, zon_m = mean_duration(MARS_DAILY)
    print(f"  MARS zonal duration: min={np.nanmin(zon_m):.2f} max={np.nanmax(zon_m):.2f} d", flush=True)
    print("WNet duration (daily, per-cell) ...", flush=True)
    _, dur_w, zon_w = mean_duration(WNET_DAILY)
    print(f"  WNet zonal duration: min={np.nanmin(zon_w):.2f} max={np.nanmax(zon_w):.2f} d", flush=True)
    out = os.path.join(HERE, "ar_duration.npz")
    np.savez_compressed(out, lat=lat, dur_mars=dur_m, dur_wnet=dur_w,
                        zon_mars=zon_m, zon_wnet=zon_w)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
