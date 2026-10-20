#!/usr/bin/env python
"""
Build HOURLY brightness temperature from ERA5 hourly MTNLWRF (forecast format),
drop-in compatible with the 6-hourly BT files test.py reads:
  var 'bt' (Kelvin), coord 'valid_time' (datetime64 hourly), dims (valid_time,lat,lon).

Per calendar month M: open MTNLWRF month M + the PREVIOUS month (the forecast valid
stamps are shifted +7h, so month M's file spans 01T07:00..next-01T06:00 — the previous
month supplies M-01T00:00..06:00). Flatten (forecast_initial_time,forecast_hour)->valid_time,
concat, sort, dedupe, slice the calendar month, then bt = (|MTNLWRF|/sigma)**0.25.

Usage: build_hourly_bt.py --year 2004 --months 7 --out <path.nc>
"""
import argparse, os, numpy as np, xarray as xr

MT_DIR = "/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/mtnlwrf/monthly"
SIGMA = 5.67e-8

def mt_path(year, month):
    return os.path.join(MT_DIR, f"ERA5_global_hourly_mtnlwrf_{year:04d}{month:02d}.nc")

def flatten_month(year, month):
    """Return a DataArray MTNLWRF(valid_time,lat,lon) for one monthly forecast file."""
    ds = xr.open_dataset(mt_path(year, month), decode_timedelta=True)
    init = ds["forecast_initial_time"].values                      # (Ninit,) datetime64
    fhr  = ds["forecast_hour"].values.astype(int)                  # (12,) hours 1..12
    vt   = init[:, None] + fhr[None, :].astype("timedelta64[h]")   # (Ninit,12)
    mt   = ds["MTNLWRF"].values                                    # (Ninit,12,lat,lon)
    Ni, Nf, ny, nx = mt.shape
    mt   = mt.reshape(Ni * Nf, ny, nx)
    vt   = vt.reshape(Ni * Nf)
    da = xr.DataArray(mt, dims=("valid_time", "latitude", "longitude"),
                      coords={"valid_time": vt,
                              "latitude": ds["latitude"].values,
                              "longitude": ds["longitude"].values})
    ds.close()
    return da

def build_month(year, month):
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    parts = []
    if os.path.exists(mt_path(py, pm)):
        parts.append(flatten_month(py, pm))           # prev month closes the +7h boundary gap
    else:
        print(f"  [warn] prev month {py}-{pm:02d} absent (dataset start) — "
              f"first ~7h of {year}-{month:02d} will be missing")
    parts.append(flatten_month(year, month))
    da = xr.concat(parts, dim="valid_time").sortby("valid_time") if len(parts) > 1 else parts[0]
    # drop duplicate stamps (month boundary overlap), keep first
    _, idx = np.unique(da["valid_time"].values, return_index=True)
    da = da.isel(valid_time=np.sort(idx))
    # slice calendar month M: 01T00:00 .. last hour of month
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00:00")
    end   = (np.datetime64(f"{year:04d}-{month:02d}-01") + np.timedelta64(32, "D"))
    end   = np.datetime64(f"{str(end)[:7]}-01T00:00")            # first of next month
    da = da.sel(valid_time=slice(start, end - np.timedelta64(1, "h")))
    bt = (np.abs(da) / SIGMA) ** 0.25
    bt = bt.astype("float32").rename("bt")
    return bt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--months", type=str, required=True, help="comma list e.g. 6,7,8")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    months = [int(m) for m in a.months.split(",")]
    das = []
    for m in months:
        print(f"[build] {a.year}-{m:02d} ...", flush=True)
        bt = build_month(a.year, m)
        n = bt.sizes["valid_time"]
        t0, t1 = str(bt.valid_time.values[0])[:16], str(bt.valid_time.values[-1])[:16]
        # gap check
        dt = np.diff(bt.valid_time.values).astype("timedelta64[h]").astype(int)
        gaps = int((dt != 1).sum())
        print(f"        steps={n}  {t0}..{t1}  1h-gaps={gaps}  bt[min={float(bt.min()):.1f} max={float(bt.max()):.1f}]", flush=True)
        das.append(bt)
    out = xr.concat(das, dim="valid_time").sortby("valid_time")
    out = out.to_dataset(name="bt")
    out["bt"].attrs = {"units": "K", "long_name": "brightness temperature from hourly MTNLWRF",
                       "note": "bt=(|MTNLWRF|/5.67e-8)**0.25"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_netcdf(a.out, encoding={"bt": {"dtype": "float32"}})
    print(f"[write] {a.out}  total_steps={out.sizes['valid_time']}", flush=True)

if __name__ == "__main__":
    main()
