#!/usr/bin/env python3
"""
Build IWV monthly climatology (12, lat, lon) for the MCS moisture-anomaly gate.

Output mirrors mslp_monthly_climatology.nc: dims (time=12, latitude, longitude),
longitude 0-360, variable name 'IWV', months ordered Jan..Dec. load_percentile_dataset
then converts longitude to -180..180 and feeds it to ARDatasetImproved(iwv_climatology=...).

Run on a compute node (streams a large merged file with dask):
  srun -n1 -c8 --mem=0 python build_iwv_climatology.py
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

SRC = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/global_stats/IWV_train_merged.nc"
OUT = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/global_stats/iwv_monthly_climatology.nc"
VARNAME = "IWV"

def main():
    print(f"Opening {SRC}")
    ds = xr.open_dataset(SRC, chunks={"time": 200})
    if VARNAME in ds:
        da = ds[VARNAME]
    else:
        da = ds[list(ds.data_vars)[0]]
        print(f"'{VARNAME}' not found; using '{da.name}'")

    # normalize time coord name
    if "valid_time" in da.dims and "time" not in da.dims:
        da = da.rename({"valid_time": "time"})

    print("Computing per-cell monthly climatology (groupby month mean)...")
    clim = da.groupby("time.month").mean("time")          # (month=12, lat, lon)
    clim = clim.sortby("month")                            # ensure Jan..Dec
    clim = clim.compute()

    # rename month -> time with 12 monthly timestamps (mirrors mslp clim layout)
    months = pd.to_datetime([f"2000-{m:02d}-01" for m in clim["month"].values])
    clim = clim.rename({"month": "time"}).assign_coords(time=months)
    clim.name = VARNAME
    clim.attrs["long_name"] = "IWV monthly climatology"
    clim.attrs["units"] = "mm"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    clim.to_dataset(name=VARNAME).to_netcdf(OUT)
    print(f"Saved {OUT}  shape={tuple(clim.shape)}  "
          f"global-mean IWV by month = {[round(float(clim.isel(time=i).mean()),1) for i in range(clim.sizes['time'])]}")

if __name__ == "__main__":
    main()
