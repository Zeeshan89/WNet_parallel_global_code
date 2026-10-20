#!/usr/bin/env python
"""
Sanity check: are hourly and 6-hourly WNet MCS detections capturing the SAME
storms? Compares one month (default July 2004) between:
  Outputs_hourly/            (hourly inference, 744 steps/month)
  Outputs_6hourly_mcsonly/   (6-hourly baseline, 00/06/12/18Z, 124 steps/month)

Produces, in ./hourly_vs_6hourly_check/ :
  1) frequency_<YYYYMM>.png  - monthly MCS occurrence frequency maps
       hourly(all) | 6-hourly | hourly@6h(matched) | (6h - hourly@6h) diff
  2) snapshots_<YYYYMM>.png  - matched instantaneous detection masks, a few
       valid times, hourly vs 6-hourly side by side
  3) prints quantitative agreement at the matched times (IoU, pixel agreement,
       pattern correlation of the frequency maps).
"""
import os, glob, argparse
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception as e:
    print(f"[warn] cartopy unavailable ({e}); plotting without coastlines")
    HAVE_CARTOPY = False

BASE = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code"
HDIR = f"{BASE}/ARs_Hurricanes_ETCs_MCS/Outputs_hourly"
SDIR = f"{BASE}/ARs_Hurricanes_ETCs_MCS/Outputs_6hourly_mcsonly"
OUT  = f"{BASE}/hourly_vs_6hourly_check"
os.makedirs(OUT, exist_ok=True)

def find_file(d, yyyymm, hourly):
    pats = ([f"{d}/era5dara_hourly_{yyyymm}_MCS_detection.nc"] if hourly
            else [f"{d}/era5dara_{yyyymm}_MCS_detection.nc"])
    for p in pats:
        if os.path.exists(p):
            return p
    g = glob.glob(f"{d}/*{yyyymm}*_MCS_detection.nc")
    return g[0] if g else None


def base_ax(ax, title):
    if HAVE_CARTOPY:
        ax.coastlines(linewidth=0.4, color="0.3")
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="0.5")
        ax.set_global()
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="200407", help="YYYYMM to compare")
    args = ap.parse_args()
    ym = args.month

    fh = find_file(HDIR, ym, hourly=True)
    fs = find_file(SDIR, ym, hourly=False)
    if not fh or not fs:
        raise SystemExit(f"missing file: hourly={fh} 6hourly={fs}")
    print(f"hourly  : {fh}")
    print(f"6-hourly: {fs}")

    dh = xr.open_dataset(fh)["MCS_detection"]
    ds = xr.open_dataset(fs)["MCS_detection"]
    dh = (dh > 0).astype("float32")
    ds = (ds > 0).astype("float32")

    lat = dh.latitude.values
    lon = dh.longitude.values

    # --- match hourly to the 6-hourly valid times (00/06/12/18Z) ---
    common = np.intersect1d(dh.time.values, ds.time.values)
    dh_m = dh.sel(time=common)
    ds_m = ds.sel(time=common)
    print(f"matched valid times: {len(common)} (6h steps this month)")

    # --- frequency maps ---
    freq_h_all = dh.mean("time").values          # all hourly steps
    freq_h_6h  = dh_m.mean("time").values         # hourly sampled at 6h times
    freq_s     = ds_m.mean("time").values         # 6-hourly
    diff       = freq_s - freq_h_6h

    proj = dict(projection=ccrs.PlateCarree()) if HAVE_CARTOPY else {}
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    imopts = dict(origin="upper", extent=extent, transform=ccrs.PlateCarree()) if HAVE_CARTOPY \
             else dict(origin="upper", extent=extent)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8),
                            subplot_kw=proj, constrained_layout=True)
    vmax = max(np.nanpercentile(freq_h_all, 99.5), np.nanpercentile(freq_s, 99.5), 1e-3)
    p00 = axs[0, 0].imshow(freq_h_all, cmap="turbo", vmin=0, vmax=vmax, **imopts)
    base_ax(axs[0, 0], f"Hourly frequency (all {dh.sizes['time']} steps)")
    fig.colorbar(p00, ax=axs[0, 0], shrink=0.7, label="fraction of time MCS present")

    p01 = axs[0, 1].imshow(freq_s, cmap="turbo", vmin=0, vmax=vmax, **imopts)
    base_ax(axs[0, 1], f"6-hourly frequency ({ds_m.sizes['time']} steps)")
    fig.colorbar(p01, ax=axs[0, 1], shrink=0.7, label="fraction")

    p10 = axs[1, 0].imshow(freq_h_6h, cmap="turbo", vmin=0, vmax=vmax, **imopts)
    base_ax(axs[1, 0], f"Hourly sampled @6h times ({dh_m.sizes['time']} steps)")
    fig.colorbar(p10, ax=axs[1, 0], shrink=0.7, label="fraction")

    dmax = np.nanpercentile(np.abs(diff), 99.5) or 1e-3
    p11 = axs[1, 1].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, **imopts)
    base_ax(axs[1, 1], "Difference: 6-hourly − hourly@6h")
    fig.colorbar(p11, ax=axs[1, 1], shrink=0.7, label="fraction diff")

    fig.suptitle(f"WNet MCS detection: hourly vs 6-hourly  —  {ym}", fontsize=13)
    fpath = f"{OUT}/frequency_{ym}.png"
    fig.savefig(fpath, dpi=130)
    plt.close(fig)
    print(f"wrote {fpath}")

    # --- matched snapshots ---
    n = len(common)
    idx = sorted(set(np.linspace(int(0.15*n), int(0.85*n), 4).astype(int)))
    fig, axs = plt.subplots(len(idx), 2, figsize=(15, 3.2*len(idx)),
                            subplot_kw=proj, constrained_layout=True)
    if len(idx) == 1:
        axs = axs[None, :]
    for r, i in enumerate(idx):
        t = np.datetime_as_string(common[i], unit="h")
        a = dh_m.isel(time=i).values
        b = ds_m.isel(time=i).values
        axs[r, 0].imshow(np.ma.masked_where(a == 0, a), cmap="autumn_r",
                         vmin=0, vmax=1, **imopts)
        base_ax(axs[r, 0], f"HOURLY  {t}Z   (pix={int(a.sum())})")
        axs[r, 1].imshow(np.ma.masked_where(b == 0, b), cmap="winter",
                         vmin=0, vmax=1, **imopts)
        base_ax(axs[r, 1], f"6-HOURLY {t}Z   (pix={int(b.sum())})")
    fig.suptitle(f"Matched-time MCS detection masks  —  {ym}", fontsize=13)
    spath = f"{OUT}/snapshots_{ym}.png"
    fig.savefig(spath, dpi=130)
    plt.close(fig)
    print(f"wrote {spath}")

    # --- quantitative agreement at matched times ---
    A = dh_m.values.reshape(n, -1).astype(bool)
    B = ds_m.values.reshape(n, -1).astype(bool)
    inter = (A & B).sum(1)
    union = (A | B).sum(1)
    iou = np.where(union > 0, inter/np.maximum(union, 1), np.nan)
    agree = (A == B).mean(1)                      # pixel agreement incl. true-neg
    # pattern correlation of frequency maps
    fa, fb = freq_h_6h.ravel(), freq_s.ravel()
    corr = np.corrcoef(fa, fb)[0, 1]
    print("\n=== agreement at matched 6-hourly valid times ===")
    print(f"mean IoU (detected-pixel overlap)        : {np.nanmean(iou):.3f}")
    print(f"median IoU                                : {np.nanmedian(iou):.3f}")
    print(f"mean pixel agreement (incl. no-MCS)       : {agree.mean():.5f}")
    print(f"mean hourly@6h MCS pixels/step            : {A.sum(1).mean():.0f}")
    print(f"mean 6-hourly  MCS pixels/step            : {B.sum(1).mean():.0f}")
    print(f"freq-map pattern correlation (hourly@6h vs 6h): {corr:.3f}")

    with open(f"{OUT}/agreement_{ym}.txt", "w") as fo:
        fo.write(f"month {ym}\nmatched_times {n}\n")
        fo.write(f"mean_IoU {np.nanmean(iou):.4f}\nmedian_IoU {np.nanmedian(iou):.4f}\n")
        fo.write(f"mean_pixel_agreement {agree.mean():.6f}\n")
        fo.write(f"mean_pix_hourly6h {A.sum(1).mean():.1f}\nmean_pix_6hourly {B.sum(1).mean():.1f}\n")
        fo.write(f"freq_pattern_corr {corr:.4f}\n")
    print(f"wrote {OUT}/agreement_{ym}.txt")


if __name__ == "__main__":
    main()
