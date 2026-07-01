#!/usr/bin/env python3
"""
regen_ar_spatial_entire.py — AR vs MARS spatial figure for the ENTIRE period
(1980-2024), i.e. primary (2006-2024) + OOP (1980-2005) combined.

Reuses the cached pixel counts (../ar_mars_eval/ar_eval_cache.npz), the full-record
days/yr climatologies, and the full-record mean-duration profile (ar_duration.npz),
producing the same 2x3 panel as the per-period figures:

  MARS AR frequency | MARS mean AR days/yr | Mean Duration (Days)
  WNet AR frequency | WNet mean AR days/yr | per-pixel F1

Outputs (this folder): ar_spatial_entire.{png,pdf}, summary_ar_entire_1980_2024.txt
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "ar_mars_eval")
EPS = 1e-12

# ---- combine primary + OOP pixel counts -> entire period (1980-2024) ----
z = np.load(os.path.join(EVAL, "ar_eval_cache.npz"), allow_pickle=True)
lat, lon = z["lat"], z["lon"]
pix = {k: (z[f"primary_pix_{k}"].astype(np.int64) + z[f"oop_pix_{k}"].astype(np.int64))
       for k in ("tp", "fp", "fn")}
ps, op = z["primary_scalar"], z["oop_scalar"]
tp, fp, fn, tn, n_t = (int(ps[i] + op[i]) for i in range(5))
acc = dict(tp=tp, fp=fp, fn=fn, tn=tn, n_t=n_t)

# ---- full-record days/yr climatologies (own lon conventions) ----
MARS_DAYS = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/Analysis_MARS/MARS_AvgARDaysPerYear.nc"
WNET_DAYS = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
             "Global_ARs_Hurr_ETCs_MCS_finalized/ARs_WNet_final/Analysis/AR_detection_AvgARDaysPerYear.nc")
_m = xr.open_dataset(MARS_DAYS); mars_days = np.asarray(_m["MARS"].squeeze().values, float)
mars_lon = _m.longitude.values.astype(float); _m.close()
_w = xr.open_dataset(WNET_DAYS); wnet_days = np.asarray(_w["AR_detection"].squeeze().values, float)
wnet_lon = _w.longitude.values.astype(float); _w.close()
DAYS_VMAX = float(np.nanpercentile(np.concatenate([mars_days.ravel(), wnet_days.ravel()]), 99.5))

# ---- full-record mean-duration zonal profile ----
_dz = np.load(os.path.join(EVAL, "ar_duration.npz"))
DUR_LAT, ZON_MARS, ZON_WNET = _dz["lat"], _dz["zon_mars"], _dz["zon_wnet"]


def to_pm180(field, src_lon):
    lp = np.asarray(src_lon, float)
    if lp.max() > 180:
        lp = ((lp + 180.0) % 360.0) - 180.0
    o = np.argsort(lp)
    return lp[o], np.asarray(field)[:, o]


def metrics():
    out = dict(IoU=tp/(tp+fp+fn+EPS), Dice=2*tp/(2*tp+fp+fn+EPS),
               Precision=tp/(tp+fp+EPS), POD=tp/(tp+fn+EPS),
               FAR=fp/(tp+fp+EPS), BIAS=(tp+fp)/(tp+fn+EPS))
    out["F1"] = out["Dice"]
    num = 2*(tp*tn - fp*fn); den = (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) + EPS
    out["HSS"] = num/den
    return out


def plot():
    tgt_freq = (pix["tp"] + pix["fn"]) / n_t
    prd_freq = (pix["tp"] + pix["fp"]) / n_t
    f1_pix = 2*pix["tp"] / (2*pix["tp"] + pix["fp"] + pix["fn"] + EPS)
    f1_pix = np.where((pix["tp"] + pix["fn"]) > 0, f1_pix, -1.0)
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
        out = os.path.join(HERE, f"ar_spatial_entire.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


def write_summary():
    mm = metrics()
    p = os.path.join(HERE, "summary_ar_entire_1980_2024.txt")
    with open(p, "w") as f:
        f.write("="*70 + "\nWNet AR Detection vs MARS — pixel skill, ENTIRE period 1980-2024\n" + "="*70 + "\n")
        f.write("(primary 2006-2024 + OOP 1980-2005 pixel counts combined)\n\n")
        f.write("ar_pixel_entire:\n")
        for k in ("IoU", "Dice", "F1", "Precision", "POD", "FAR", "BIAS", "HSS"):
            f.write(f"  {k:<12} = {mm[k]:.6f}\n")
        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("n_t", n_t)):
            f.write(f"  {k:<12} = {v}\n")
    print("wrote", p)
    print(f"  entire IoU={mm['IoU']:.3f} F1={mm['F1']:.3f} POD={mm['POD']:.3f} "
          f"FAR={mm['FAR']:.3f} HSS={mm['HSS']:.3f}")


if __name__ == "__main__":
    plot()
    write_summary()
