#!/usr/bin/env python
"""
Phase 3 — hourly vs 6-hourly MCS, July 2004 (both MCS_ONLY, 6 h lifetime).
Headline: the DIURNAL CYCLE. 6-hourly samples only 00/06/12/18 UTC; converted to local
solar time across longitudes that aliases the diurnal cycle, whereas hourly resolves all
24 local hours. Also: MCS frequency maps (hourly vs 6-hourly) + difference.

Outputs (PNG) to --outdir.
Usage: phase3_hourly_vs_6hourly.py --hourly <nc> --sixhr <nc> --outdir <dir>
"""
import argparse, os, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# land MCS hotspots active in NH summer (lat0,lat1, lon0E,lon1E) in 0..360 lon
REGIONS = {
    "US Great Plains": (35, 48, 258, 270),
    "Sahel/C. Africa": (5, 15, 0, 30),
    "India monsoon":   (15, 26, 74, 88),
    "Amazon":          (-10, 2, 290, 310),
}

def load(path):
    ds = xr.open_dataset(path)["MCS_detection"]
    # lon may be -180..180 (test.py writes it that way); normalize to 0..360 for boxing
    lon = ds["longitude"].values
    if lon.min() < 0:
        ds = ds.assign_coords(longitude=(ds["longitude"] % 360)).sortby("longitude")
    return (ds > 0.5).astype("float32")

def box_mean_series(mask, lat0, lat1, lon0, lon1):
    la = mask["latitude"]; lo = mask["longitude"]
    sub = mask.sel(latitude=slice(*( (lat1,lat0) if la[0]>la[-1] else (lat0,lat1) )),
                   longitude=slice(lon0, lon1))
    return sub.mean(dim=("latitude","longitude")).values, 0.5*(lon0+lon1)

def local_hour(times, lon_center):
    ut = np.array([int(str(t)[11:13]) for t in times])
    return (ut + lon_center/15.0) % 24

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", required=True)
    ap.add_argument("--sixhr", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    h = load(a.hourly); s = load(a.sixhr)
    print(f"hourly steps={h.sizes['time']}  6-hourly steps={s.sizes['time']}")

    # ---- 1. frequency maps + difference ----
    fh = h.mean("time"); fs = s.mean("time")
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    for k,(dat,ttl) in enumerate([(fs,"6-hourly freq"),(fh,"hourly freq"),(fh-fs,"hourly - 6hourly")]):
        cmap = "RdBu_r" if k==2 else "viridis"
        vm = np.nanpercentile(np.abs(dat.values),99) if k==2 else np.nanpercentile(dat.values,99)
        im = ax[k].pcolormesh(dat["longitude"], dat["latitude"], dat.values,
                              cmap=cmap, vmin=(-vm if k==2 else 0), vmax=vm, shading="auto")
        ax[k].set_title(f"MCS {ttl} — July 2004"); ax[k].set_xlabel("lon"); ax[k].set_ylabel("lat")
        plt.colorbar(im, ax=ax[k], shrink=0.8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig1_freq_maps.png", dpi=110); plt.close(fig)
    print(f"[fig] {a.outdir}/fig1_freq_maps.png  (domain freq hourly={float(fh.mean()):.4f} 6hr={float(fs.mean()):.4f})")

    # ---- 2. diurnal cycle by LOCAL solar hour, per region ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.ravel()
    for i,(name,box) in enumerate(REGIONS.items()):
        hs, lonc = box_mean_series(h, *box)
        ss, _    = box_mean_series(s, *box)
        lh_h = local_hour(h["time"].values, lonc)
        lh_s = local_hour(s["time"].values, lonc)
        # bin hourly into 24 local-hour bins; 6-hourly into its (few) local hours
        bins = np.arange(0,25)
        idx_h = np.digitize(lh_h, bins)-1
        dh = np.array([hs[idx_h==b].mean() if (idx_h==b).any() else np.nan for b in range(24)])
        # 6-hourly: group by rounded local hour
        lh_s_r = np.round(lh_s).astype(int)%24
        us = sorted(set(lh_s_r)); ds_ = [ss[lh_s_r==u].mean() for u in us]
        ax = axes[i]
        ax.plot(range(24), dh*100, "-o", ms=3, lw=2, label="hourly (24 local hrs)")
        ax.plot(us, np.array(ds_)*100, "s", ms=9, color="crimson", label="6-hourly (aliased)")
        ax.set_title(f"{name}  (lon~{lonc:.0f}E)"); ax.set_xlabel("local solar hour")
        ax.set_ylabel("MCS area frac (%)"); ax.set_xticks(range(0,24,3)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Diurnal cycle of MCS frequency — hourly resolves it, 6-hourly aliases (July 2004)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig2_diurnal_cycle.png", dpi=110); plt.close(fig)
    print(f"[fig] {a.outdir}/fig2_diurnal_cycle.png")

    # ---- 3. numeric summary: diurnal amplitude captured ----
    print("\nRegion            hourly peak-hr  hourly amp(%)   6hourly amp(%)")
    for name,box in REGIONS.items():
        hs, lonc = box_mean_series(h, *box); ss,_ = box_mean_series(s, *box)
        lh_h = local_hour(h["time"].values, lonc); idx_h = np.digitize(lh_h, np.arange(0,25))-1
        dh = np.array([hs[idx_h==b].mean() if (idx_h==b).any() else np.nan for b in range(24)])
        lh_s_r = np.round(local_hour(s["time"].values, lonc)).astype(int)%24
        ds_ = np.array([ss[lh_s_r==u].mean() for u in sorted(set(lh_s_r))])
        amp_h = (np.nanmax(dh)-np.nanmin(dh))*100; amp_s = (ds_.max()-ds_.min())*100
        print(f"  {name:16s}  {int(np.nanargmax(dh)):02d}L           {amp_h:6.3f}         {amp_s:6.3f}")

if __name__ == "__main__":
    main()
