#!/usr/bin/env python
"""
Build an HOURLY moisture 'dara' file (era5dara_hourly_YYYYMM.nc) for the MCS-only hourly run.
  IWV  = real hourly TCWV (kg m-2 ~ mm)   -> drives the MCS moisture-anomaly gate
  all other dara vars = ZERO on the hourly time axis (ch0-10 are zeroed at inference anyway;
       kept present + right-shape so test.py's calculate_ivt / var extraction never crash).
Grid/time come from the TCWV file (721x1440, lon 0..359.75, 720/744 hourly stamps).
Var set + attrs mirror the existing 6-hourly dara so the loader sees an identical schema.

Usage: build_hourly_moisture.py --year 2004 --month 7 --out <path.nc>
"""
import argparse, glob, os, numpy as np, xarray as xr

TCWV_DIR = "/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/tcwv"
DARA_TMPL = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/era5dara_{year:04d}{month:02d}.nc"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    tcwv_glob = os.path.join(TCWV_DIR, f"{a.year:04d}{a.month:02d}", "*128_137_tcwv*.nc")
    tcwv_file = sorted(glob.glob(tcwv_glob))
    assert tcwv_file, f"no TCWV file at {tcwv_glob}"
    tc = xr.open_dataset(tcwv_file[0])
    tcwv = tc["TCWV"].astype("float32")                       # (time,lat,lon) hourly
    nt, ny, nx = tcwv.shape
    print(f"[tcwv] {tcwv_file[0].split('/')[-1]}  steps={nt}  "
          f"{str(tc.time.values[0])[:16]}..{str(tc.time.values[-1])[:16]}", flush=True)

    tmpl = xr.open_dataset(DARA_TMPL.format(year=a.year, month=a.month))
    varlist = list(tmpl.data_vars)                            # Qu,Qv,IWV,CRH,TPR,... schema
    print(f"[schema] dara vars: {varlist}", flush=True)

    coords = {"time": tc["time"].values,
              "latitude": tcwv["latitude"].values,
              "longitude": tcwv["longitude"].values}
    zeros = np.zeros((nt, ny, nx), dtype="float32")
    out = xr.Dataset(coords=coords)
    for v in varlist:
        if v.upper() == "IWV":
            da = xr.DataArray(tcwv.values, dims=("time", "latitude", "longitude"), coords=coords)
        else:
            da = xr.DataArray(zeros, dims=("time", "latitude", "longitude"), coords=coords)
        da.attrs = dict(tmpl[v].attrs)
        out[v] = da
    out["IWV"].attrs["note"] = "hourly ERA5 TCWV substituted for IWV (validated corr 0.9986 vs 6-hourly IWV)"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 1, "dtype": "float32"} for v in varlist}
    out.to_netcdf(a.out, encoding=enc)
    tc.close(); tmpl.close()
    print(f"[write] {a.out}  vars={len(varlist)} steps={nt}  (IWV real, rest zero)", flush=True)

if __name__ == "__main__":
    main()
