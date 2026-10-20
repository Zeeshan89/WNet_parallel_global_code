#!/usr/bin/env python3
"""
plot_smooth_vs_nn_maps.py — climatology maps from the shipped (nearest-neighbour, blocky) and the
smooth (bilinear-probability) AR/MCS exports, side by side with their difference, 1980-2024 over
20-60N, 135-65W (the regional subset box).

  row 1  AR days/yr from the 6-hourly files: a day counts if ANY of its 4 steps is flagged
         (method and levels of ARs_WNet_final_subset/plot_subset_ardays.py)
  row 2  MCS days/yr from the hourly files as hours/24 (levels of plot_wnet_newcode_final.py's
         mean_annual_mcs_days_from_hours). This is the raw mask frequency, not the gated
         Regional_MCS pipeline map.
Printed and in the panel titles: box-mean, pattern r, mean |difference|.

Usage (1 CPU node):  python plot_smooth_vs_nn_maps.py --workers 48
"""
import os
import glob
import argparse
from multiprocessing import Pool

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, TwoSlopeNorm  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
LAT = np.linspace(90, -90, 721)
LON = np.linspace(-180, 179.75, 1440)
IY = np.flatnonzero((LAT >= 20) & (LAT <= 60))
IX = np.flatnonzero((LON >= -135) & (LON <= -65))
NYEARS = 45.0
WBGYR = LinearSegmentedColormap.from_list("wbgyr", [
    (0.00, "#ffffff"), (0.08, "#d8ecf9"), (0.18, "#a6d5ee"), (0.30, "#5fb0db"),
    (0.38, "#3f93c6"), (0.46, "#2fa08c"), (0.56, "#5cbf5a"), (0.64, "#a5d63f"),
    (0.72, "#f2e534"), (0.82, "#f6a623"), (0.91, "#e8551f"), (1.00, "#b00000"),
], N=256)
PRODUCTS = [  # (row title, var, dir blocky, dir smooth, reducer, levels, unit)
    ("AR days per year (6-hourly)", "AR_detection",
     f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final",
     f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2_smooth/ARs_WNet_final",
     "days4", [5, 20, 40, 60, 80, 100, 120], "days / year"),
    ("MCS days per year from hours/24 (hourly)", "MCS_detection",
     f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2_hourly/MCS_WNet_final",
     f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2_hourly_smooth/MCS_WNet_final",
     "hours", [2, 5, 10, 15, 20, 25, 30, 40, 50], "equivalent days / year"),
]

# MAPS_MODE=fullres: the shipped v2 (0.94 deg model, smooth-upsampled) against the model trained and run at
# 0.25 deg (Global_ARs_Hurr_ETCs_MCS_fullres025), both 6-hourly; MCS days = daily presence like AR days.
MODE = os.environ.get("MAPS_MODE", "smooth")
if MODE == "fullres":
    FR = f"{NB}/Global_ARs_Hurr_ETCs_MCS_fullres025"
    PRODUCTS = [
        ("AR days per year (6-hourly)", "AR_detection",
         f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final", f"{FR}/ARs_WNet_final",
         "days4", [5, 20, 40, 60, 80, 100, 120], "days / year"),
        ("MCS days per year (6-hourly)", "MCS_detection",
         f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2/MCS_WNet_final", f"{FR}/MCS_WNet_final",
         "days4", [2, 5, 10, 15, 20, 25, 30, 40, 50], "days / year"),
    ]
# MAPS_ONLY=AR / MCS: draw one row only (e.g. AR while the MCS files are still being rewritten)
ONLY = os.environ.get("MAPS_ONLY", "")
if ONLY:
    PRODUCTS = [p for p in PRODUCTS if p[1].startswith(ONLY)]
LAB0, LAB1 = (("WNet 0.94° (v2)", "WNet 0.25°") if MODE == "fullres" else ("Current (blocky)", "Smooth"))


def one(job):
    path, var, how = job
    with h5py.File(path, "r") as f:
        a = f[var][:, IY[0]:IY[-1] + 1, IX[0]:IX[-1] + 1] > 0.5
    if how == "days4":
        nd = a.shape[0] // 4
        return a[:nd * 4].reshape(nd, 4, *a.shape[1:]).any(1).sum(0, dtype=np.int64)
    return a.sum(0, dtype=np.int64)


def climatology(pool, d, var, how):
    files = sorted(glob.glob(os.path.join(d, f"era5dara_*_{var}.nc")))
    assert len(files) == 540, (d, len(files))
    acc = sum(pool.imap_unordered(one, [(f, var, how) for f in files], chunksize=2))
    acc = acc / NYEARS
    return acc / 24.0 if how == "hours" else acc


def disc(levels):
    cmap = ListedColormap(WBGYR(np.linspace(0.06, 1.0, len(levels) - 1)))
    cmap.set_under("white"); cmap.set_over("#ff00d4")
    return cmap, BoundaryNorm(levels, cmap.N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=(f"{NB}/Global_ARs_Hurr_ETCs_MCS_fullres025/fullres_vs_v2_maps" if MODE == "fullres"
                                      else f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2_smooth/smooth_vs_blocky_maps"))
    a = ap.parse_args()
    if ONLY:
        a.out += f"_{ONLY}"
    lat, lon = LAT[IY], LON[IX]
    cos = np.cos(np.radians(lat))[:, None] * np.ones((1, lon.size))
    fig, axs = plt.subplots(len(PRODUCTS), 3, figsize=(19, 4.3 * len(PRODUCTS)), squeeze=False,
                            subplot_kw={"projection": ccrs.PlateCarree()})
    with Pool(a.workers) as pool:
        for r, (title, var, d0, d1, how, lv, unit) in enumerate(PRODUCTS):
            B = climatology(pool, d0, var, how)
            S = climatology(pool, d1, var, how)
            wm = lambda z: float((z * cos).sum() / cos.sum())   # noqa: E731  area-weighted box mean
            rr = float(np.corrcoef(B.ravel(), S.ravel())[0, 1])
            mad = wm(np.abs(S - B))
            print(f"{title}: box mean {LAB0} {wm(B):.2f}  {LAB1} {wm(S):.2f}  (x{wm(S) / wm(B):.3f})  "
                  f"r {rr:.4f}  mean|diff| {mad:.2f}  max|diff| {np.abs(S - B).max():.1f}", flush=True)
            cmap, norm = disc(lv)
            lim = max(1.0, float(np.percentile(np.abs(S - B), 99.5)))
            for c, (Z, name) in enumerate([(B, f"{LAB0}, box mean {wm(B):.1f}"),
                                           (S, f"{LAB1}, box mean {wm(S):.1f}"),
                                           (S - B, f"{LAB1} minus {LAB0}, r = {rr:.3f}, mean |diff| {mad:.2f}")]):
                ax = axs[r, c]
                if c < 2:
                    m = ax.pcolormesh(lon, lat, Z, cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
                    if c == 1:
                        cb = fig.colorbar(m, ax=axs[r, :2], extend="both", ticks=lv, fraction=0.02, pad=0.02)
                        cb.set_label(unit)
                else:
                    m = ax.pcolormesh(lon, lat, Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                                      shading="auto", transform=ccrs.PlateCarree())
                    cb = fig.colorbar(m, ax=ax, extend="both", fraction=0.035, pad=0.02)
                    cb.set_label(unit)
                ax.set_extent([-135, -65, 20, 60], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor="#555555")
                ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#888888")
                ax.add_feature(cfeature.STATES, linewidth=0.15, edgecolor="#aaaaaa")
                ax.set_title(f"{title}\n{name}" if c == 0 else name, fontsize=10)
    fig.suptitle("WNet 1980-2024: 0.94° model (v2) vs model trained and run at 0.25°" if MODE == "fullres" else
                 "WNet 1980-2024: current nearest-neighbour export vs smooth (bilinear probability) export",
                 fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", f"{a.out}.{ext}")


if __name__ == "__main__":
    main()
