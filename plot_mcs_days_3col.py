#!/usr/bin/env python
"""
Mean MCS-days/year (2004-2018) as a 3-column figure:
  col1 = OLD 6-hourly (full-channel, Output_new_IWV)
  col2 = NEW 6-hourly (MCS_ONLY baseline, Outputs_6hourly_mcsonly)
  col3 = HOURLY placeholder (left blank for now; replot later)
MCS-days/month per cell = resample('1D').max() on (MCS_detection>0.5), summed over days;
accumulated over all months, divided by #years.
"""
import glob, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS"
YEARS = list(range(2004, 2019))
OUT_PNG = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/mcs_days_3col.png"

def mcs_days_per_year(d, hourly=False):
    acc = None; lat = lon = None; nfiles = 0
    for y in YEARS:
        pat = f"{d}/*hourly_{y}??_*MCS_detection.nc" if hourly else f"{d}/era5dara_{y}??_MCS_detection.nc"
        for f in sorted(glob.glob(pat)):
            ds = xr.open_dataset(f)
            m = (ds["MCS_detection"] > 0.5)
            md = m.resample(time="1D").max().sum("time")   # MCS-days this month, per cell
            md = md.compute() if hasattr(md, "compute") else md
            acc = md.values.astype("float64") if acc is None else acc + md.values
            if lat is None: lat, lon = ds["latitude"].values, ds["longitude"].values
            nfiles += 1; ds.close()
    print(f"  {d.split('/')[-1]}: {nfiles} files, days/yr[max={np.nanmax(acc)/len(YEARS):.1f}]", flush=True)
    return acc / len(YEARS), lat, lon

print("computing OLD 6-hourly ...", flush=True)
old, lat, lon = mcs_days_per_year(f"{BASE}/Output_new_IWV")
print("computing NEW 6-hourly (MCS_ONLY) ...", flush=True)
new, _, _ = mcs_days_per_year(f"{BASE}/Outputs_6hourly_mcsonly")
print("computing HOURLY (MCS_ONLY) ...", flush=True)
hrly, _, _ = mcs_days_per_year(f"{BASE}/Outputs_hourly", hourly=True)

vmax = float(np.nanpercentile(np.concatenate([old.ravel(), new.ravel(), hrly.ravel()]), 99.5))
panels = [(old,  "OLD 6-hourly (full channels)"),
          (new,  "NEW 6-hourly (MCS_ONLY)"),
          (hrly, "HOURLY (MCS_ONLY)")]
fig, ax = plt.subplots(1, 3, figsize=(21, 4.6), constrained_layout=True)
for k, (dat, ttl) in enumerate(panels):
    a = ax[k]
    im = a.pcolormesh(lon, lat, dat, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=a, shrink=0.85, label="MCS days / year")
    a.set_xlabel("longitude"); a.set_ylabel("latitude")
    a.set_title(f"{ttl}\nglobal mean {np.nanmean(dat):.1f} d/yr", fontsize=11)
fig.suptitle("Mean MCS days per year, 2004-2018", fontsize=15)
fig.savefig(OUT_PNG, dpi=110)
print(f"[saved] {OUT_PNG}", flush=True)
