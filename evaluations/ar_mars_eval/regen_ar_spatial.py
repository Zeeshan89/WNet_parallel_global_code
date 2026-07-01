#!/usr/bin/env python3
"""Regenerate ar_spatial_{primary,oop}.{png,pdf} with the Mean Duration line plot
replacing the WNet-MARS bias panel. Reads the eval cache + days files + ar_duration.npz."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import xarray as xr

OUT_DIR = os.path.dirname(os.path.abspath(__file__)); EPS = 1e-12
PERIODS = {"oop": (1980, 2005), "primary": (2006, 2024)}
z = np.load(os.path.join(OUT_DIR, "ar_eval_cache.npz"), allow_pickle=True)
lat, lon = z["lat"], z["lon"]
acc, pix = {}, {}
for p in PERIODS:
    tp, fp, fn, tn, nt = z[f"{p}_scalar"]
    acc[p] = dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), n_t=int(nt))
    pix[p] = {k: z[f"{p}_pix_{k}"] for k in ("tp", "fp", "fn")}

MARS_DAYS_NC = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/Analysis_MARS/MARS_AvgARDaysPerYear.nc"
WNET_DAYS_NC = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
                "Global_ARs_Hurr_ETCs_MCS_finalized/ARs_WNet_final/Analysis/AR_detection_AvgARDaysPerYear.nc")
_md = xr.open_dataset(MARS_DAYS_NC)
mars_days = np.asarray(_md["MARS"].squeeze().values, float); mars_lon = _md.longitude.values.astype(float); _md.close()
_wd = xr.open_dataset(WNET_DAYS_NC)
wnet_days = np.asarray(_wd["AR_detection"].squeeze().values, float); wnet_lon = _wd.longitude.values.astype(float); _wd.close()
DAYS_VMAX = float(np.nanpercentile(np.concatenate([mars_days.ravel(), wnet_days.ravel()]), 99.5))

_dz = np.load(os.path.join(OUT_DIR, "ar_duration.npz"))
DUR_LAT, ZON_MARS, ZON_WNET = _dz["lat"], _dz["zon_mars"], _dz["zon_wnet"]


def to_pm180(field, src_lon):
    lp = np.asarray(src_lon, float)
    if lp.max() > 180:
        lp = ((lp + 180.0) % 360.0) - 180.0
    o = np.argsort(lp)
    return lp[o], np.asarray(field)[:, o]


def plot_period_maps(p):
    a = acc[p]; tp, fp, fn = pix[p]["tp"], pix[p]["fp"], pix[p]["fn"]; n_t = a["n_t"]
    tgt_freq = (tp + fn) / n_t; prd_freq = (tp + fp) / n_t
    f1_pix = 2*tp / (2*tp + fp + fn + EPS); f1_pix = np.where((tp + fn) > 0, f1_pix, -1.0)
    fv = np.nanpercentile(tgt_freq, 99.5)
    panel = {0: (tgt_freq, lon, "MARS AR frequency", "Blues", (0, fv)),
             1: (mars_days, mars_lon, "MARS mean AR days/yr", "YlGnBu", (0, DAYS_VMAX)),
             3: (prd_freq, lon, "WNet AR frequency", "Blues", (0, fv)),
             4: (wnet_days, wnet_lon, "WNet mean AR days/yr", "YlGnBu", (0, DAYS_VMAX)),
             5: (f1_pix, lon, "per-pixel F1", "viridis", (0, 1))}
    fig, axes = plt.subplots(2, 3, figsize=(21, 8),
                             subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
    for idx, ax in enumerate(axes.ravel()):
        if idx == 2:
            ax.remove()
            axd = fig.add_subplot(2, 3, 3)
            axd.plot(ZON_MARS, DUR_LAT, color="#1f77b4", lw=1.9, label="MARS")
            axd.plot(ZON_WNET, DUR_LAT, color="#d62728", lw=1.9, label="WNet")
            axd.set_xlim(0, 3.5); axd.set_ylim(-90, 90)
            axd.set_yticks([-90, -60, -30, 0, 30, 60, 90])
            axd.set_xlabel("Mean Duration (Days)", fontsize=12)
            axd.set_ylabel("Latitude", fontsize=12)
            axd.set_title("Mean AR duration", fontsize=12)
            axd.tick_params(labelsize=11); axd.grid(alpha=0.3)
            axd.legend(fontsize=11, loc="upper right")
            continue
        field, src_lon, title, cmap, (vmin, vmax) = panel[idx]
        lon_p, field_p = to_pm180(field, src_lon)
        cmap_obj = plt.get_cmap(cmap).copy()
        if title == "per-pixel F1":
            cmap_obj.set_under(cmap_obj(0.0))
        m = ax.pcolormesh(lon_p, lat, field_p, transform=ccrs.PlateCarree(),
                          cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.coastlines(lw=0.4, color="0.25")
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.45")
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 12}; gl.ylabel_style = {"size": 12}
        ax.set_title(title, fontsize=12)
        cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"ar_spatial_{p}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


for p in ("primary", "oop"):
    plot_period_maps(p)
