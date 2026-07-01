#!/usr/bin/env python3
"""
Quick test: take the IWV-gate MCS predictions (Output_new_IWV, overlap-only, NO area)
and see what applying a 60,000 km^2 minimum-area criterion does to the metrics —
to decide whether the area criterion is needed.

Compares, over CONUS for a couple of years:
  - NO area  (as-is, overlap-only)
  - +60k area (drop per-timestep tracked blobs < 60,000 km^2)
on MCS-days and ID-based storm-hours seasonality. Feng numbers printed for reference.
"""
import os, glob
import numpy as np, pandas as pd, xarray as xr
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MCS_DIR = "ARs_Hurricanes_ETCs_MCS/Output_new_IWV"
YEARS = [2010, 2011]
LAT0, LAT1, LON0, LON1 = 25.0, 51.0, -110.0, -70.0
MIN_AREA_KM2 = 60000.0
DT = 6.0
R = 6371.0
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def pixel_area(lats, lons):
    dlat = np.deg2rad(abs(np.mean(np.diff(lats)))); dlon = np.deg2rad(abs(np.mean(np.diff(lons))))
    cell = (R**2) * dlat * dlon * np.cos(np.deg2rad(lats))
    return np.repeat(np.clip(cell, 0, None)[:, None], lons.size, axis=1).astype(np.float32)

def area_filter_ids(ids2d, area2d, thr):
    out = ids2d.copy()
    u = np.unique(out); u = u[u > 0]
    if u.size:
        a = ndimage.sum(area2d, out, u)
        drop = u[np.asarray(a) < thr]
        if drop.size:
            out[np.isin(out, drop)] = 0
    return out

def run(apply_area):
    days_per_year = []
    monthly = {m: [] for m in range(1, 13)}
    for y in YEARS:
        fs = sorted(glob.glob(os.path.join(MCS_DIR, f"era5dara_{y}*_MCS_detection.nc")))
        if not fs: continue
        daily_any_year = []
        for f in fs:
            ds = xr.open_dataset(f)
            la = "latitude" if "latitude" in ds.coords else "lat"
            lo = "longitude" if "longitude" in ds.coords else "lon"
            ld = bool(ds[la][0] > ds[la][-1])
            ds = ds.sel({la: slice(LAT1, LAT0) if ld else slice(LAT0, LAT1), lo: slice(LON0, LON1)})
            ids = ds["MCS_id"].values.astype(np.int32)
            lats = ds[la].values; lons = ds[lo].values
            area2d = pixel_area(lats, lons)
            T = ids.shape[0]
            month = int(pd.Timestamp(ds.time.values[0]).month)
            present = np.zeros(ids.shape, dtype=bool)
            n_ids_step = 0
            for t in range(T):
                cur = area_filter_ids(ids[t], area2d, MIN_AREA_KM2) if apply_area else ids[t]
                present[t] = cur > 0
                n_ids_step += int(np.unique(cur[cur > 0]).size)
            monthly[month].append(n_ids_step * DT)
            # daily presence for MCS-days
            tt = pd.to_datetime(ds.time.values)
            da = xr.DataArray(present, coords={"time": tt}, dims=["time", "y", "x"])
            daily_any_year.append(da.resample(time="1D").max())
            ds.close()
        if daily_any_year:
            dd = xr.concat(daily_any_year, dim="time")
            days_per_year.append(float(dd.sum("time").mean()))
    mser = pd.Series({m: np.mean(v) for m, v in monthly.items() if v}).reindex(range(1, 13))
    return np.mean(days_per_year), mser

print(f"Testing min-area effect on {MCS_DIR}, years {YEARS}, CONUS\n")
d_no, m_no = run(False)
d_ar, m_ar = run(True)
print("                         MCS-days/yr   storm-hrs peak   min   ratio")
for name, d, m in [("NO area (as-is)", d_no, m_no), (f"+{int(MIN_AREA_KM2)} area", d_ar, m_ar)]:
    print(f"  {name:20s} {d:8.1f}     {m.max():9.0f}  {m.min():5.0f}  {m.max()/m.min():5.1f}")
print(f"  {'Feng (14-yr ref)':20s} {40.7:8.1f}     {1431:9.0f}  {279:5.0f}  {5.1:5.1f}")
print("\n  storm-hrs/mo NO area:", [int(round(m_no.get(i, np.nan))) for i in range(1, 13)])
print("  storm-hrs/mo +area  :", [int(round(m_ar.get(i, np.nan))) for i in range(1, 13)])

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(1, 13)
ax.bar(x - 0.2, m_no.values, 0.4, label="NO area (overlap-only)", color="tab:blue")
ax.bar(x + 0.2, m_ar.values, 0.4, label=f"+{int(MIN_AREA_KM2)} km² area", color="tab:orange")
ax.axhline(1431, ls="--", c="red", lw=1, label="Feng Jul peak (ref)")
ax.set_xticks(x); ax.set_xticklabels(MON); ax.set_ylabel("storm-hours/month")
ax.set_title(f"Effect of 60k area criterion on IWV-gate MCS storm-hours ({YEARS[0]}–{YEARS[-1]}, CONUS)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
out = "comparison_wnet_vs_feng_IWV/area_effect_test.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
print("\nSaved", out)
