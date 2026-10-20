#!/usr/bin/env python3
"""
cache_to_netcdf.py — write the training cache back out as netCDF, one file per variable.

tensor_cache_v11_fullres/split_YYYYMM.npy is (T, 10, 721, 1440) float32 in RAW physical units
(standardization happens at training time, not in the cache), with split_YYYYMM.meta.npy holding
the T timestamps as datetime64[s]. The grid is implied, not stored: latitude 90 to -90 and
longitude -180 to 179.75, both at 0.25 deg. That is the ERA5 source grid rolled by half a turn;
verified against Storms_data/data/era5dara_YYYYMM.nc, where the same field matches to 0 after
np.roll(src, 720, axis=1). --lon-0-360 rolls it back to the source 0 to 360 convention.

    python cache_to_netcdf.py --month 200507 --out-dir /path/out
    python cache_to_netcdf.py --years 2000 2005 --vars IVT MSLP AR --out-dir /path/out
    python cache_to_netcdf.py --month 200507 --single --lon-0-360 --out-dir /path/out

NOTE on what the cache cannot give back: channel 0 is the IVT MAGNITUDE and channel 3 the 850 hPa
WIND SPEED. The components (Qu, Qv, u850, v850) are not stored, so anything direction dependent
has to go back to Storms_data/data/era5dara_YYYYMM.nc, which holds Qu and Qv.
"""
import os
import argparse
from datetime import datetime

import numpy as np
import netCDF4

NB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(NB, "tensor_cache_v11_fullres")
#  channel: (name, units, long_name)
CHANNELS = [
    ("IVT",      "kg m-1 s-1", "integrated vapor transport, magnitude, from Qu and Qv"),
    ("MSLP",     "Pa",         "mean sea level pressure"),
    ("vort850",  "s-1",        "relative vorticity at 850 hPa"),
    ("wind850",  "m s-1",      "wind speed at 850 hPa, magnitude"),
    ("BT",       "K",          "cloud-top brightness temperature"),
    ("IWV",      "kg m-2",     "integrated water vapor"),
    ("AR",       "1",          "atmospheric river label (L6 literature criteria)"),
    ("TC",       "1",          "tropical cyclone label (IBTrACS discs, full lifecycle)"),
    ("ETC",      "1",          "extratropical cyclone label (closed pressure contour)"),
    ("MCS",      "1",          "mesoscale convective system label (cold core and shield)"),
]
NAMES = [c[0] for c in CHANNELS]


def grid(lon_0_360=False):
    lat = 90.0 - 0.25 * np.arange(721)
    lon = -180.0 + 0.25 * np.arange(1440)
    if lon_0_360:
        lon = (lon + 360.0) % 360.0
    return lat.astype(np.float64), lon.astype(np.float64)


def write(path, varspecs, times, lat, lon, complevel=4):
    """varspecs: [(name, units, long_name, array (T,H,W))]."""
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        v = ds.createVariable("time", "i8", ("time",))
        v.units = "hours since 1900-01-01 00:00:00"
        v.calendar = "proleptic_gregorian"
        ref = np.datetime64("1900-01-01T00:00:00", "s")
        v[:] = ((times.astype("datetime64[s]") - ref).astype("int64") // 3600)
        for nm, arr in (("latitude", lat), ("longitude", lon)):
            c = ds.createVariable(nm, "f8", (nm,))
            c.units = "degrees_north" if nm == "latitude" else "degrees_east"
            c[:] = arr
        for name, units, long_name, a in varspecs:
            d = ds.createVariable(name, "f4", ("time", "latitude", "longitude"),
                                  zlib=True, complevel=complevel,
                                  chunksizes=(1, lat.size, lon.size))
            d.units = units
            d.long_name = long_name
            d[:] = a
        ds.source = "tensor_cache_v11_fullres (WNet 0.25 deg training cache), raw physical units"
        ds.history = f"{datetime.now():%Y-%m-%d} cache_to_netcdf.py"


def month_out(cache, month, out_dir, names, single, lon_0_360, complevel):
    a = np.load(os.path.join(cache, f"split_{month}.npy"), mmap_mode="r")
    times = np.load(os.path.join(cache, f"split_{month}.meta.npy"), allow_pickle=True)
    times = np.asarray(times, dtype="datetime64[s]")
    if a.shape[0] != times.size:
        raise SystemExit(f"{month}: {a.shape[0]} steps but {times.size} timestamps")
    lat, lon = grid(lon_0_360)
    roll = 720 if lon_0_360 else 0
    os.makedirs(out_dir, exist_ok=True)
    specs = []
    for name in names:
        i = NAMES.index(name)
        arr = np.asarray(a[:, i], dtype=np.float32)
        if roll:
            arr = np.roll(arr, roll, axis=2)
        specs.append((name, CHANNELS[i][1], CHANNELS[i][2], arr))
    if single:
        p = os.path.join(out_dir, f"wnet_cache_{month}.nc")
        write(p, specs, times, lat, lon, complevel)
        print(f"  wrote {p}  ({len(specs)} variables, {times.size} steps)", flush=True)
    else:
        for s in specs:
            p = os.path.join(out_dir, f"wnet_{s[0]}_{month}.nc")
            write(p, [s], times, lat, lon, complevel)
            print(f"  wrote {p}  ({times.size} steps)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--month", help="YYYYMM; or use --years")
    ap.add_argument("--years", nargs=2, type=int, metavar=("Y0", "Y1"))
    ap.add_argument("--vars", nargs="+", default=NAMES, choices=NAMES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--single", action="store_true", help="all variables in one file per month")
    ap.add_argument("--lon-0-360", action="store_true",
                    help="write on the ERA5 source convention instead of -180 to 180")
    ap.add_argument("--complevel", type=int, default=4)
    args = ap.parse_args()

    if args.month:
        months = [args.month]
    elif args.years:
        months = [f"{y}{m:02d}" for y in range(args.years[0], args.years[1] + 1)
                  for m in range(1, 13)]
    else:
        raise SystemExit("give --month or --years")
    for mo in months:
        if not os.path.exists(os.path.join(args.cache_dir, f"split_{mo}.npy")):
            print(f"  {mo}: not in the cache, skipped", flush=True)
            continue
        month_out(args.cache_dir, mo, args.out_dir, args.vars, args.single,
                  args.lon_0_360, args.complevel)


if __name__ == "__main__":
    main()
