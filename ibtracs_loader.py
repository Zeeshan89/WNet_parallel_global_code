# ibtracs_loader.py
"""
Loads IBTrACS v04 and builds a fast lookup:
    dict[ datetime(Y,M,D,H) ]  ->  list[ (lat, lon_norm, wind_ms, sshs) ]

Only keeps:
  - seasons 1980–2024
  - 6-hourly snapped timestamps  (3-h records are averaged to nearest 6-h slot)
  - lat in [-50, 50]  (your training domain)
  - usa_wind valid and >= wind_threshold_kt
  - snaps lon to [-180, 180)
"""

from netCDF4 import Dataset, num2date
import numpy as np
from collections import defaultdict
from datetime import datetime


def load_ibtracs_for_training(
    file_path,
    start_year       = 1980,
    end_year         = 2024,
    wind_threshold_kt= 33.0,    # 33 kt = tropical storm; 64 kt = hurricane
    lat_min          = -50.0,
    lat_max          =  50.0,
):
    """
    Returns
    -------
    lookup : dict[ datetime ] -> list[ (lat, lon, wind_ms, sshs) ]
        keyed by 6-hourly datetime objects (minute/second = 0)
        wind_ms  = wind speed in m/s  (converted from knots)
        sshs     = Saffir-Simpson category (-1=tropical storm, 0–5)
                   -9 if not available
    """
    ds            = Dataset(file_path)
    seasons       = np.array(ds.variables['season'][:], dtype=int)      # (13452,)
    time_raw      = ds.variables['time'][:]                              # (13452,360) masked
    lat_raw       = ds.variables['lat'][:]                               # (13452,360)
    lon_raw       = ds.variables['lon'][:]                               # (13452,360)
    wind_raw      = ds.variables['usa_wind'][:]                          # (13452,360) kts
    sshs_raw      = ds.variables['usa_sshs'][:]                          # (13452,360) int8
    time_units    = ds.variables['time'].units                           # days since 1858-11-17
    time_calendar = getattr(ds.variables['time'], 'calendar', 'standard')
    ds.close()

    # Pre-select storms in the year range (fast index filter)
    storm_indices = np.where((seasons >= start_year) & (seasons <= end_year))[0]
    print(f"  IBTrACS: {len(storm_indices)} storms in {start_year}–{end_year}")

    lookup = defaultdict(list)   # datetime -> [(lat, lon, wind_ms, sshs), ...]

    for si in storm_indices:
        # ── build per-storm valid mask ───────────────────────────────────────
        t_row    = time_raw[si, :]
        lat_row  = lat_raw[si, :]
        lon_row  = lon_raw[si, :]
        wind_row = wind_raw[si, :]
        sshs_row = sshs_raw[si, :]

        # getmaskarray always returns a bool array (safe for scalar mask)
        valid = (
            ~np.ma.getmaskarray(t_row)    &
            ~np.ma.getmaskarray(lat_row)  &
            ~np.ma.getmaskarray(lon_row)  &
            ~np.ma.getmaskarray(wind_row)
        )
        if valid.sum() == 0:
            continue

        # convert to plain numpy (no masked-array issues downstream)
        t_vals    = np.array(t_row[valid],    dtype=float)
        lat_vals  = np.array(lat_row[valid],  dtype=float)
        lon_vals  = np.array(lon_row[valid],  dtype=float)
        wind_vals = np.array(wind_row[valid], dtype=float)

        # sshs may still be masked at some points → replace with -9
        sshs_vals_raw = sshs_row[valid]
        sshs_mask     = np.ma.getmaskarray(sshs_vals_raw)
        sshs_vals     = np.where(sshs_mask, -9,
                                 np.array(sshs_vals_raw, dtype=int))

        # decode times
        decoded = num2date(t_vals, units=time_units, calendar=time_calendar)

        for i, dt in enumerate(decoded):
            lat  = float(lat_vals[i])
            lon  = float(lon_vals[i])
            wind = float(wind_vals[i])
            sshs = int(sshs_vals[i])

            # ── domain filter ───────────────────────────────────────────────
            if not (lat_min <= lat <= lat_max):
                continue

            # ── wind threshold filter ────────────────────────────────────────
            if wind < wind_threshold_kt:
                continue

            # ── normalise lon to [-180, 180) ─────────────────────────────────
            lon_norm = (lon + 180.0) % 360.0 - 180.0

            # ── snap timestamp to nearest 6-h slot ──────────────────────────
            # e.g. 03:00 → 00:00,  09:00 → 06:00 (round down)
            hour6 = (dt.hour // 6) * 6
            key   = datetime(dt.year, dt.month, dt.day, hour6)

            wind_ms = wind * 0.514444   # knots → m/s
            lookup[key].append((lat, lon_norm, wind_ms, sshs))

    print(f"  IBTrACS lookup: {len(lookup)} unique 6-hourly timesteps with active storms")

    # quick sanity: show a few entries near hurricane season
    sample_keys = sorted(k for k in lookup if k.year == 2005 and k.month == 8)[:4]
    for k in sample_keys:
        pts = lookup[k]
        print(f"    {k}  →  {len(pts)} storm point(s)  "
              f"e.g. lat={pts[0][0]:.1f} lon={pts[0][1]:.1f} "
              f"wind={pts[0][2]:.1f}m/s sshs={pts[0][3]}")

    return dict(lookup)   # plain dict, faster lookup than defaultdict