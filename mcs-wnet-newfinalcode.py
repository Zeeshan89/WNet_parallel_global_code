#!/usr/bin/env python3
"""
WNet (6-hourly) processing — standardized outputs (days + hours + monthly bars)

STANDARD DEFINITIONS:
  keep = (MCS_present) & (tp_mm_per_step > 0)

Where:
  MCS_present = (MCS_detection > MCS_GT)

Core annual maps:
  yearly_MCS_days.nc
  freq_hours.nc
  hours_per_active_day.nc

Precip (ERA5 tp):
  avg_precip.nc      : mean tp_mm_per_step during keep (mm/step)
  percent_precip.nc  : 100 * sum(tp during keep)/sum(tp total)

Monthly:
  monthly_domain_presence_hours.csv
  monthly_domain_cell_hours.csv
  monthly_mean_cell_hours.csv
  monthly_counts.csv  (alias -> presence hours)

IMPORTANT:
- WNet has no storm IDs. We DO NOT create "monthly_mcs_hours_by_id.csv" here.
  To keep your plotting scripts unified, we also write:
      monthly_mcs_hours_by_id.csv = monthly_domain_presence_hours
  This is a bounded, comparable surrogate (but not literally ID-based).

DASK NOTE:
- Avoid `.item()` on lazy arrays. Use `.compute()` before scalar extraction.
"""

import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import center_of_mass

# -----------------------
# USER SETTINGS
# -----------------------
START_YEAR = int(os.environ.get("MCS_START_YEAR", "2004"))
END_YEAR   = int(os.environ.get("MCS_END_YEAR", "2014"))

MCS_DIR    = os.environ.get("MCS_DIR", "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Output_new_IWV_60k/")
PRECIP_DIR = os.environ.get("PRECIP_DIR", "/lustre/orion/proj-shared/cli138/dr6/eventtype//Data/ERA5/input")  # expects ERA5_tp_input_{year}_6hourly_conus.nc
OUTPUT_DIR = os.environ.get("MCS_OUTPUT_DIR", "outputs_wnet_standard_IWV_60k")
# glob pattern for monthly detection files; hourly outputs need the 'hourly_' infix variant
MCS_PATTERN = os.environ.get("MCS_PATTERN", "era5dara_{year}{month:02d}_MCS_detection*.nc")

LAT_BOUNDS = (25.0, 50.99454)
LON_BOUNDS = (-109.9899, -70.0101)



# WNet binarization threshold
MCS_GT   = 0.0     # e.g. 0.5 if probability
TP_GT_MM = 0.0     # gate on tp_mm_per_step > 0

# -----------------------
# Helpers
# -----------------------
def _normalize_latlon_names(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:  ren["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords: ren["lon"] = "longitude"
    return ds.rename(ren) if ren else ds

def _to_minus180_180(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords:
        mx = float(ds.longitude.max().compute()) if hasattr(ds.longitude.data, "compute") else float(ds.longitude.max())
        if mx > 180:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    return ds

def subset_latlon(ds: xr.Dataset, lat_bounds, lon_bounds) -> xr.Dataset:
    ds = _normalize_latlon_names(ds)
    ds = _to_minus180_180(ds)

    lat0, lat1 = lat_bounds
    lon0, lon1 = lon_bounds
    lat = ds.latitude
    lon = ds.longitude
    lat_desc = bool(lat[0] > lat[-1])
    lon_desc = bool(lon[0] > lon[-1])

    lat_slice = slice(max(lat0, lat1), min(lat0, lat1)) if lat_desc else slice(min(lat0, lat1), max(lat0, lat1))
    lon_slice = slice(max(lon0, lon1), min(lon0, lon1)) if lon_desc else slice(min(lon0, lon1), max(lon0, lon1))

    out = ds.sel(latitude=lat_slice, longitude=lon_slice)
    if int(out.sizes.get("latitude", 0)) == 0 or int(out.sizes.get("longitude", 0)) == 0:
        raise ValueError("Spatial subset produced empty grid. Check bounds/coords.")
    return out

def infer_dt_hours(time_da: xr.DataArray) -> float:
    t = pd.to_datetime(time_da.values)
    if t.size < 2:
        return np.nan
    dt = pd.to_timedelta(np.median(np.diff(t)))
    return float(dt.total_seconds() / 3600.0)

def safe_scalar(x):
    try:
        return x.item()
    except Exception:
        return np.asarray(x.compute()).item()

# -----------------------
# Open WNet monthly -> yearly
# -----------------------
def open_wnet_year(mcs_dir: str, year: int,
                   pattern: str = MCS_PATTERN) -> xr.Dataset:
    """Assemble a year from monthly WNet MCS files, carrying the tracked MCS_id.

    MCS_id restarts at 1 in each monthly file, so we offset each month's nonzero
    IDs by the running max to make them UNIQUE across the year (needed for correct
    per-track summaries; the monthly storm-hours metric groups by month anyway).
    """
    monthly = []
    id_offset = 0
    for m in range(1, 13):
        files = sorted(glob.glob(os.path.join(mcs_dir, pattern.format(year=year, month=m))))
        if not files:
            continue
        dsm = xr.open_mfdataset(files, combine="by_coords", decode_times=True)
        if "MCS_detection" not in dsm:
            raise KeyError("Variable 'MCS_detection' not found in WNet files.")
        keepvars = ["MCS_detection"]
        if "MCS_id" in dsm:
            ids = dsm["MCS_id"]
            ids = ids.where(ids == 0, ids + id_offset)        # year-unique IDs
            mx = int(np.asarray(ids.max()))
            id_offset = max(id_offset, mx)
            dsm = dsm.assign(MCS_id=ids)
            keepvars.append("MCS_id")
        monthly.append(dsm[keepvars])
    if not monthly:
        raise FileNotFoundError(f"No WNet monthly files found for {year} in {mcs_dir}")
    ds = xr.concat(monthly, dim="time").sortby("time")
    return ds

# -----------------------
# Monthly metrics
# -----------------------
def monthly_domain_presence_hours(keep: xr.DataArray, dt_hours: float) -> pd.Series:
    present = keep.max(dim=keep.dims[1:]).astype("int8")
    s = (present.resample(time="MS").sum() * dt_hours).to_series()
    s.index = pd.to_datetime(s.index).to_period("M").astype(str)
    s.name = "hours"
    return s

def monthly_domain_cell_hours(keep: xr.DataArray, dt_hours: float) -> pd.Series:
    ncell = keep.sum(dim=keep.dims[1:]).astype("float32")
    s = (ncell.resample(time="MS").sum() * dt_hours).to_series()
    s.index = pd.to_datetime(s.index).to_period("M").astype(str)
    s.name = "cell_hours"
    return s

def monthly_mean_cell_hours(keep: xr.DataArray, dt_hours: float) -> pd.Series:
    hpc = keep.resample(time="MS").sum() * dt_hours
    s = hpc.mean(dim=hpc.dims[1:], skipna=True).to_series()
    s.index = pd.to_datetime(s.index).to_period("M").astype(str)
    s.name = "hours_per_cell"
    return s

# -----------------------
# ID-based metric — IDENTICAL definitions to the Feng benchmark (verbatim).
# WNet supplies its own MCS_id (overlap-tracked in test.py); the math below is
# the same as mcs-feng-newfinalcode.py so the comparison uses the same metric.
# -----------------------
def compute_dt_hours(time_values) -> float:
    t = pd.to_datetime(time_values)
    if t.size < 2:
        return np.nan
    dt = pd.to_timedelta(np.median(np.diff(t)))
    return float(dt.total_seconds() / 3600.0)

def frac_index_to_coord(coord_1d, idxf):
    n = coord_1d.size
    if idxf <= 0: return float(coord_1d[0])
    if idxf >= n - 1: return float(coord_1d[-1])
    i0 = int(np.floor(idxf)); i1 = i0 + 1; w = idxf - i0
    return float((1 - w) * coord_1d[i0] + w * coord_1d[i1])

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def monthly_storm_hours_by_id(mcs_ids: xr.DataArray, keep: xr.DataArray) -> pd.Series:
    """TE-style: per timestep, add dt_hours for each unique ID present in domain (gated)."""
    times = pd.to_datetime(mcs_ids.time.values)
    dt_h = int(round(compute_dt_hours(mcs_ids.time.values)))
    T = mcs_ids.sizes["time"]

    data_all = np.asarray(mcs_ids.values)   # load once (avoid T dask computes per timestep)
    keep_all = np.asarray(keep.values)
    ids_per_time = []
    for i in range(T):
        arr = data_all[i]
        km = keep_all[i]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        arr = np.where(km, arr, np.nan)
        ids = np.unique(arr[~np.isnan(arr)])
        ids = ids[ids > 0]
        ids_per_time.append(ids)

    df = pd.DataFrame({"time": times, "mcs_ids": ids_per_time})
    df["month"] = df["time"].dt.to_period("M")
    df = df.explode("mcs_ids").dropna(subset=["mcs_ids"])
    if df.empty:
        return pd.Series(dtype=float, name="hours")

    df["mcs_ids"] = df["mcs_ids"].astype(int)
    df["hours"] = dt_h

    out = df.groupby(["month", "mcs_ids"])["hours"].sum().groupby("month").sum()
    out.index = out.index.astype(str)
    out.name = "hours"
    return out

def extract_properties_per_id(mcs_ids_gated: xr.DataArray, lat1d: np.ndarray, lon1d: np.ndarray) -> pd.DataFrame:
    tvals = pd.to_datetime(mcs_ids_gated.time.values)
    dt_h = int(round(compute_dt_hours(mcs_ids_gated.time.values)))

    data = mcs_ids_gated.values  # (T, Y, X)
    T = data.shape[0]

    unique_ids = np.unique(data)
    unique_ids = unique_ids[unique_ids > 0]

    rows = []
    for uid in unique_ids:
        mask = (data == uid)
        present = mask.reshape(T, -1).any(axis=1)
        idx = np.where(present)[0]
        if idx.size == 0:
            continue

        areas, centroids = [], []
        for k in idx:
            blob = mask[k]
            area = int(blob.sum())
            areas.append(area)
            if area > 0:
                cy, cx = center_of_mass(blob)
                latc = frac_index_to_coord(lat1d, cy)
                lonc = frac_index_to_coord(lon1d, cx)
                centroids.append((latc, lonc))

        dist_km = 0.0
        if len(centroids) > 1:
            for (la1, lo1), (la2, lo2) in zip(centroids[:-1], centroids[1:]):
                dist_km += haversine_km(lo1, la1, lo2, la2)

        duration_h = idx.size * dt_h
        rows.append({
            "MCS_ID": int(uid),
            "Start_Time": str(tvals[idx[0]]),
            "End_Time": str(tvals[idx[-1]]),
            "Duration_hrs": float(duration_h),
            "Avg_Area_cells": float(np.mean(areas)) if areas else 0.0,
            "Max_Area_cells": float(np.max(areas)) if areas else 0.0,
            "Path_Length_km": float(dist_km),
            "Avg_Speed_kmph": float(dist_km / duration_h) if duration_h > 0 else 0.0
        })
    return pd.DataFrame(rows)

# -----------------------
# Load + align precip
# -----------------------
def load_align_wnet_and_precip(year: int):
    ds_m = open_wnet_year(MCS_DIR, year)
    ds_m = subset_latlon(ds_m, LAT_BOUNDS, LON_BOUNDS)
    ds_m = ds_m.sortby("time")
    mcs = ds_m["MCS_detection"]
    if "MCS_id" not in ds_m:
        raise KeyError("Variable 'MCS_id' not found — re-run test.py inference to add tracked IDs.")
    mcs_ids = ds_m["MCS_id"]

    dt_hours = infer_dt_hours(mcs.time)
    if not np.isfinite(dt_hours):
        raise ValueError("Could not infer dt_hours from WNet time axis.")
    if not np.isclose(dt_hours, 6.0, atol=0.2):
        print(f"  WARN: inferred dt_hours={dt_hours:.2f} (expected ~6). Continuing.")

    precip_file = os.path.join(PRECIP_DIR, f"ERA5_tp_input_{year}_6hourly_conus.nc")
    if not os.path.exists(precip_file):
        raise FileNotFoundError(precip_file)

    ds_p = xr.open_dataset(precip_file)
    ds_p = subset_latlon(ds_p, LAT_BOUNDS, LON_BOUNDS)
    if "tp" not in ds_p:
        raise KeyError(f"'tp' not found in {precip_file}")

    tp_mm = (ds_p["tp"] * 1000.0).astype("float32")  # m/step -> mm/step
    tp_mm.attrs["units"] = "mm"
    tp_mm.attrs["long_name"] = "Total precipitation per step (mm)"

    # nearest within half of the PRECIP step (precip is 6-hourly), so hourly MCS steps
    # each map to the nearest 6-hourly precip (round(dt/2)=0 for hourly would give all-NaN).
    tol = np.timedelta64(3, "h")
    tp_mm = tp_mm.reindex(time=mcs.time, method="nearest", tolerance=tol)

    mcs, mcs_ids, tp_mm = xr.align(mcs, mcs_ids, tp_mm, join="inner")
    return mcs, mcs_ids, tp_mm, dt_hours

# -----------------------
# Per-year processing
# -----------------------
_CONUS_MASK_CACHE = {}
def conus_mask_for(lat_da, lon_da):
    """Boolean DataArray (latitude,longitude): True only inside the CONUS landmask.
    Built from Natural Earth admin-1 US states (excl. Alaska/Hawaii) via cartopy.
    Cached per grid (Natural Earth data is cached under $HOME, so compute nodes reuse it)."""
    lat1d = np.asarray(lat_da.values); lon1d = np.asarray(lon_da.values)
    key = (float(lat1d[0]), float(lat1d[-1]), lat1d.size, float(lon1d[0]), float(lon1d[-1]), lon1d.size)
    if key in _CONUS_MASK_CACHE:
        return _CONUS_MASK_CACHE[key]
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_1_states_provinces_lakes")
    geoms = [r.geometry for r in shpreader.Reader(shp).records()
             if r.attributes.get("admin") == "United States of America"
             and r.attributes.get("name") not in ("Alaska", "Hawaii")]
    pc = prep(unary_union(geoms))
    m = np.zeros((lat1d.size, lon1d.size), dtype=bool)
    for i, la in enumerate(lat1d):
        for j, lo in enumerate(lon1d):
            m[i, j] = pc.contains(Point(float(lo), float(la)))
    da = xr.DataArray(m, dims=("latitude", "longitude"),
                      coords={"latitude": lat_da.values, "longitude": lon_da.values})
    print(f"  [CONUS mask] {int(m.sum())}/{m.size} cells inside CONUS", flush=True)
    _CONUS_MASK_CACHE[key] = da
    return da


def process_year(year: int):
    print(f"\n--- WNet year {year} --- {datetime.utcnow().isoformat()}")

    try:
        mcs, mcs_ids, tp_mm, dt_hours = load_align_wnet_and_precip(year)
    except Exception as e:
        print(f"  Skipping {year}: {e}")
        return

    # ERA5 tp is already mm/step; build a rate (mm/hr) to mirror the Feng precip handling.
    precip_step_mm = tp_mm.astype("float32")
    precip_rate = (precip_step_mm / dt_hours).astype("float32")
    lat1d = np.asarray(mcs_ids["latitude"].values)
    lon1d = np.asarray(mcs_ids["longitude"].values)

    # keep = (MCS_id > 0). ID-based storm-hours count CONCURRENT tracks (no longer a surrogate);
    # metric definitions are identical to the Feng benchmark above.
    # CONUS-only: drop all MCS presence outside the CONUS landmask, so every downstream
    # metric (maps + domain/monthly sums) is restricted to CONUS.
    conus = conus_mask_for(mcs_ids["latitude"], mcs_ids["longitude"])
    keep = (mcs_ids > 0) & conus

    # Annual maps
    keep_daily = keep.resample(time="1D").max().astype("int8")
    yearly_days = keep_daily.sum(dim="time").astype("float32").rename("mcs_days")

    freq_hours = (keep.sum(dim="time") * dt_hours).astype("float32").rename("freq")

    hours_per_active_day = xr.where(yearly_days > 0, freq_hours / yearly_days, np.nan).astype("float32")
    hours_per_active_day = hours_per_active_day.rename("hours_per_active_day")

    # Precip (avg_precip as mm/step, matching original WNet semantics)
    total_precip = precip_step_mm.sum(dim="time", skipna=True)
    precip_during = precip_step_mm.where(keep).sum(dim="time", skipna=True)
    avg_precip = precip_step_mm.where(keep).mean(dim="time", skipna=True)
    avg_precip.attrs["units"] = "mm/step"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        percent_precip = (100.0 * (precip_during / total_precip)).astype("float32")

    # Monthly metrics
    mon_presence = monthly_domain_presence_hours(keep, dt_hours)
    mon_cell     = monthly_domain_cell_hours(keep, dt_hours)
    mon_mean     = monthly_mean_cell_hours(keep, dt_hours)
    mon_idhours  = monthly_storm_hours_by_id(mcs_ids, keep)

    # Summary (per ID). This per-ID extraction is O(n_ids x T x cells) and becomes the
    # dominant cost at hourly cadence (8760 steps x thousands of IDs -> hours). Its output
    # (summary.csv) is NOT consumed by plot_wnet_newcode_final.py, so it is skippable:
    # set MCS_SKIP_SUMMARY=1 to bypass it (default keeps the original 6-hourly behavior).
    if os.environ.get("MCS_SKIP_SUMMARY", "0") == "1":
        summary_df = pd.DataFrame()
    else:
        mcs_ids_gated = mcs_ids.where(keep, other=0)
        summary_df = extract_properties_per_id(mcs_ids_gated, lat1d, lon1d)

    # Domain-wide days/year
    daily_any = keep_daily.max(dim=keep_daily.dims[1:])
    domain_days = int(safe_scalar(daily_any.sum()))

    # CONUS-only: blank (NaN) everything outside the CONUS landmask in the saved maps
    # (keep is already CONUS-gated for the sums; this makes the map cells outside CONUS
    # empty rather than zero, matching the benchmark look).
    yearly_days          = yearly_days.where(conus)
    freq_hours           = freq_hours.where(conus)
    hours_per_active_day = hours_per_active_day.where(conus)
    avg_precip           = avg_precip.where(conus)
    percent_precip       = percent_precip.where(conus)

    # Save
    ydir = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(ydir, exist_ok=True)

    yearly_days.to_netcdf(os.path.join(ydir, "yearly_MCS_days.nc"))
    freq_hours.to_netcdf(os.path.join(ydir, "freq_hours.nc"))
    hours_per_active_day.to_netcdf(os.path.join(ydir, "hours_per_active_day.nc"))

    avg_precip.to_netcdf(os.path.join(ydir, "avg_precip.nc"))
    percent_precip.to_netcdf(os.path.join(ydir, "percent_precip.nc"))

    mon_presence.to_csv(os.path.join(ydir, "monthly_domain_presence_hours.csv"))
    mon_cell.to_csv(os.path.join(ydir, "monthly_domain_cell_hours.csv"))
    mon_mean.to_csv(os.path.join(ydir, "monthly_mean_cell_hours.csv"))
    mon_presence.to_csv(os.path.join(ydir, "monthly_counts.csv"))

    if mon_idhours is None or mon_idhours.empty:
        pd.Series(dtype=float, name="hours").to_csv(os.path.join(ydir, "monthly_mcs_hours_by_id.csv"))
    else:
        mon_idhours.to_csv(os.path.join(ydir, "monthly_mcs_hours_by_id.csv"), header=True)

    summary_df.to_csv(os.path.join(ydir, "summary.csv"), index=False)
    pd.DataFrame([{"year": year, "domain_MCS_days": domain_days}]).to_csv(
        os.path.join(ydir, "domain_MCS_days_by_year.csv"), index=False
    )

    with open(os.path.join(ydir, "README_processing.txt"), "w") as f:
        f.write("STANDARDIZED WNet processing notes:\n")
        f.write(f"- dt_hours inferred = {dt_hours:.3f} (expected ~6)\n")
        f.write("- keep = (MCS_id > 0); MCS_id is overlap-tracked in test.py\n")
        f.write("- MCS criteria upstream: 60,000 km^2 area + 50% overlap link + 6h lifetime\n")
        f.write("- monthly_mcs_hours_by_id.csv is REAL ID-based storm-hours (same metric as Feng)\n")

    print(f"  Saved {ydir} | dt_hours={dt_hours:.2f} | domain_days={domain_days}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for y in range(START_YEAR, END_YEAR + 1):
        process_year(y)

if __name__ == "__main__":
    main()

