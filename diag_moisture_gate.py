#!/usr/bin/env python3
"""
Diagnostic: how many MCS (cold-cloud-shield) pixels does the CURRENT absolute-IWV
moisture gate erase in a winter month, vs an IWV-ANOMALY gate?

Faithfully replicates the relevant pieces of dataset._create_mcs_masks:
  - blob detection from BT (boundary<=241, core<=228 winter, min sizes)
  - moisture_score(IWV) and the winter gate score>=0.3  (== IWV>=33 mm)
Then compares to: keep blob pixels where IWV anomaly (vs Jan climatology) >= 0.
No model needed; reads BT + IWV directly.
"""
import os, glob
import numpy as np
import xarray as xr
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_YEAR, TARGET_MONTH = 2010, 1            # January 2010
CLIM_YEARS = range(2004, 2018)
BT_DIR   = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new"
DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
LAT0, LAT1, LON0, LON1 = 25.0, 51.0, -110.0, -70.0    # CONUS
# winter seasonal params (from _get_seasonal_mcs_parameters)
BT_BOUNDARY, BT_CORE = 241.0, 228.0
MIN_BLOB_AREA, MIN_CORE_PIX = 12, 8
WINTER_MOIST_THRESH = 0.3

def to180_conus(da, latn, lonn):
    da = da.assign_coords({lonn: (((da[lonn] + 180) % 360) - 180)}).sortby(lonn)
    la = da[latn]; lo = da[lonn]
    lat_slice = slice(LAT1, LAT0) if bool(la[0] > la[-1]) else slice(LAT0, LAT1)
    return da.sel({latn: lat_slice, lonn: slice(LON0, LON1)})

def moisture_score(iwv):
    s = np.zeros_like(iwv, dtype=np.float32)
    s[iwv >= 50] = 1.0
    m = (iwv >= 40) & (iwv < 50); s[m] = (iwv[m] - 40) / 10 + 0.5
    m = (iwv >= 30) & (iwv < 40); s[m] = (iwv[m] - 30) / 10
    m = iwv < 30;                 s[m] = np.maximum((iwv[m] - 20) / 10, 0.0)
    return s

def blob_mask(bt):
    bnd = bt <= BT_BOUNDARY
    cor = bt <= BT_CORE
    lab, n = ndimage.label(bnd, structure=np.ones((3, 3)))
    out = np.zeros_like(bnd, dtype=bool)
    if n:
        sizes = ndimage.sum(np.ones_like(bt), lab, range(1, n + 1))
        coresz = ndimage.sum(cor.astype(float), lab, range(1, n + 1))
        keep = np.nonzero((sizes >= MIN_BLOB_AREA) & (coresz >= MIN_CORE_PIX))[0] + 1
        if keep.size:
            out = np.isin(lab, keep)
    return out

def load_bt(year, month):
    ds = xr.open_dataset(os.path.join(BT_DIR, f"era5_bt_6hourly_{year}.nc"))
    if "valid_time" in ds.dims: ds = ds.rename({"valid_time": "time"})
    bt = to180_conus(ds["bt"], "latitude", "longitude")
    return bt.sel(time=bt["time"].dt.month == month)

def load_iwv(year, month):
    ds = xr.open_dataset(os.path.join(DATA_DIR, f"era5dara_{year}{month:02d}.nc"))
    return to180_conus(ds["IWV"], "latitude", "longitude")

def main():
    print(f"Diagnostic for {TARGET_YEAR}-{TARGET_MONTH:02d} over CONUS")
    bt  = load_bt(TARGET_YEAR, TARGET_MONTH).load()
    iwv = load_iwv(TARGET_YEAR, TARGET_MONTH).load()
    # align times
    t = np.intersect1d(bt.time.values, iwv.time.values)
    bt = bt.sel(time=t); iwv = iwv.sel(time=t)
    # regrid IWV to BT grid if needed
    if iwv.shape[1:] != bt.shape[1:]:
        iwv = iwv.interp(latitude=bt.latitude, longitude=bt.longitude)
    print(f"  timesteps={bt.sizes['time']}  grid={bt.shape[1:]}")

    # Jan IWV climatology (per cell, across years) on BT grid
    clim_list = []
    for y in CLIM_YEARS:
        try:
            iy = load_iwv(y, TARGET_MONTH).mean("time")
            if iy.shape != bt.shape[1:]:
                iy = iy.interp(latitude=bt.latitude, longitude=bt.longitude)
            clim_list.append(iy)
        except FileNotFoundError:
            pass
    clim = xr.concat(clim_list, dim="yr").mean("yr").values
    print(f"  climatology from {len(clim_list)} years; mean Jan IWV over CONUS = {np.nanmean(clim):.1f} mm")

    btv, iwvv = bt.values, iwv.values
    tot_blob = tot_abs = tot_anom = 0
    for k in range(btv.shape[0]):
        blob = blob_mask(btv[k])
        if not blob.any():
            continue
        sc = moisture_score(iwvv[k])
        abs_keep  = blob & (sc >= WINTER_MOIST_THRESH)          # current gate (IWV>=33)
        anom_keep = blob & ((iwvv[k] - clim) >= 0.0)            # anomaly gate (above-normal)
        tot_blob += int(blob.sum()); tot_abs += int(abs_keep.sum()); tot_anom += int(anom_keep.sum())

    print("\n=== January 2010 cold-cloud-shield (blob) pixels, summed over month ===")
    print(f"  blob pixels (pre-gate)           : {tot_blob}")
    print(f"  kept by CURRENT abs-IWV gate     : {tot_abs}  ({100*tot_abs/tot_blob:.1f}%)  -> ERASED {100*(1-tot_abs/tot_blob):.1f}%")
    print(f"  kept by ANOMALY gate (>=0)       : {tot_anom}  ({100*tot_anom/tot_blob:.1f}%)  -> ERASED {100*(1-tot_anom/tot_blob):.1f}%")
    print(f"  recovery factor (anom/abs)       : {tot_anom/max(tot_abs,1):.1f}x")

    # representative-timestep map
    sizes = [(k, int(blob_mask(btv[k]).sum())) for k in range(btv.shape[0])]
    k = max(sizes, key=lambda x: x[1])[0]
    blob = blob_mask(btv[k]); sc = moisture_score(iwvv[k])
    absk = blob & (sc >= WINTER_MOIST_THRESH); anomk = blob & ((iwvv[k] - clim) >= 0.0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, m, ti in zip(ax, [blob, absk, anomk],
                        [f"Cold-cloud blobs (pre-gate): {int(blob.sum())} px",
                         f"After CURRENT abs-IWV gate: {int(absk.sum())} px",
                         f"After ANOMALY gate: {int(anomk.sum())} px"]):
        a.imshow(m, origin="lower", cmap="Reds", aspect="auto"); a.set_title(ti, fontsize=10); a.axis("off")
    fig.suptitle(f"Winter MCS moisture-gate erasure — {str(bt.time.values[k])[:13]} (CONUS)")
    out = "comparison_wnet_vs_feng/moisture_gate_winter_diag.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print("Saved", out)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
