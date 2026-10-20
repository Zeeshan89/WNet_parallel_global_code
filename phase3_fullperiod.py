#!/usr/bin/env python
"""
Phase 3 (full period 2004-2018) — hourly vs 6-hourly MCS sensitivity.
Memory-safe: reduces each monthly *_MCS_detection.nc file one at a time and accumulates.

Deliverables:
  fig1_mcs_days.png    mean MCS-days/yr: hourly | 6-hourly | (hourly - 6-hourly)
  fig2_diurnal.png     diurnal cycle of MCS freq by LOCAL solar hour, per region
                       (hourly resolves 24 local hrs; 6-hourly aliases to ~4)
  phase3_summary.txt   domain means, per-region diurnal amplitude + peak hour

Usage: phase3_fullperiod.py --hourly_dir <d> --sixhr_dir <d> --outdir <d>
"""
import argparse, glob, os, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REGIONS = {  # (lat0,lat1, lon0E,lon1E) 0..360
    "US Great Plains": (35, 48, 258, 270),
    "Sahel/C. Africa": (5, 15, 0, 30),
    "India monsoon":   (15, 26, 74, 88),
    "Amazon":          (-10, 2, 290, 310),
}
THR = 0.5

def month_stamp(f):  # era5dara_[hourly_]YYYYMM_MCS_detection.nc -> YYYYMM
    b = os.path.basename(f)
    return b.replace("_MCS_detection.nc", "").split("_")[-1]

def reduce_dir(d, label):
    """One pass over all monthly files -> freq_sum, nstep, daycount, nyears, diurnal accumulators."""
    files = sorted(glob.glob(os.path.join(d, "*_MCS_detection.nc")))
    print(f"[{label}] {len(files)} files", flush=True)
    freq_sum = None; nstep = 0; day_sum = None
    years = set()
    # diurnal: per region -> [sum over 24 local-hr bins], [count]
    dsum = {r: np.zeros(24) for r in REGIONS}; dcnt = {r: np.zeros(24) for r in REGIONS}
    latv = lonv = None
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        v = ds["MCS_detection"]
        lon = ds["longitude"].values
        if lon.min() < 0:  # normalize -180..180 -> 0..360
            ds = ds.assign_coords(longitude=(ds["longitude"] % 360)).sortby("longitude")
            v = ds["MCS_detection"]; lon = ds["longitude"].values
        lat = ds["latitude"].values
        if latv is None: latv, lonv = lat, lon
        m = (v.values > THR)                                   # (t,lat,lon) bool
        # frequency map
        if freq_sum is None: freq_sum = np.zeros(m.shape[1:], np.float64); day_sum = np.zeros(m.shape[1:], np.float64)
        freq_sum += m.sum(0); nstep += m.shape[0]
        # MCS-days: daily max then count days
        t = ds["time"].values
        yr = int(str(t[0])[:4]); years.add(yr)
        day = np.array([str(x)[:10] for x in t])
        for dd in np.unique(day):
            idx = np.where(day == dd)[0]
            day_sum += m[idx].max(0)
        # diurnal per region (box-mean freq per timestep, binned by local hour)
        ut = np.array([int(str(x)[11:13]) for x in t])
        for r,(la0,la1,lo0,lo1) in REGIONS.items():
            la_slice = slice(la1,la0) if lat[0] > lat[-1] else slice(la0,la1)
            sub = ds["MCS_detection"].sel(latitude=la_slice, longitude=slice(lo0,lo1))
            bm = (sub.values > THR).mean(axis=(1,2))           # (t,)
            lonc = 0.5*(lo0+lo1)
            lh = ((ut + lonc/15.0) % 24).astype(int)
            for b in range(24):
                sel = lh == b
                if sel.any(): dsum[r][b] += bm[sel].sum(); dcnt[r][b] += sel.sum()
        ds.close()
        if (i+1) % 24 == 0: print(f"  [{label}] {i+1}/{len(files)}", flush=True)
    nyears = max(1, len(years))
    freq = freq_sum / max(1, nstep)
    days_per_yr = day_sum / nyears
    diurnal = {r: (dsum[r]/np.where(dcnt[r]>0, dcnt[r], np.nan)) for r in REGIONS}
    return dict(freq=freq, days=days_per_yr, diurnal=diurnal, lat=latv, lon=lonv,
                nyears=nyears, nfiles=len(files))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly_dir", required=True)
    ap.add_argument("--sixhr_dir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    H = reduce_dir(a.hourly_dir, "hourly")
    S = reduce_dir(a.sixhr_dir, "6hourly")

    # fig1: MCS-days/yr maps
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.3))
    for k,(dat,ttl,cm) in enumerate([(S["days"],"6-hourly","viridis"),
                                     (H["days"],"hourly","viridis"),
                                     (H["days"]-S["days"],"hourly - 6-hourly","RdBu_r")]):
        vmax = np.nanpercentile(np.abs(dat),99.5)
        im=ax[k].pcolormesh(H["lon"],H["lat"],dat,cmap=cm,
                            vmin=(-vmax if k==2 else 0),vmax=vmax,shading="auto")
        ax[k].set_title(f"MCS days/yr — {ttl}"); plt.colorbar(im,ax=ax[k],shrink=.8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig1_mcs_days.png",dpi=110); plt.close(fig)

    # fig2: diurnal cycle per region
    fig, axes = plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel()
    for i,r in enumerate(REGIONS):
        axes[i].plot(range(24), H["diurnal"][r]*100, "-o", ms=3, lw=2, label="hourly (24 local hr)")
        s = S["diurnal"][r]*100; present = ~np.isnan(s)
        axes[i].plot(np.arange(24)[present], s[present], "s", ms=9, color="crimson", label="6-hourly (aliased)")
        axes[i].set_title(r); axes[i].set_xlabel("local solar hour"); axes[i].set_ylabel("MCS area frac (%)")
        axes[i].set_xticks(range(0,24,3)); axes[i].grid(alpha=.3); axes[i].legend(fontsize=8)
    fig.suptitle("Diurnal cycle of MCS frequency (2004-2018): hourly resolves it, 6-hourly aliases", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig2_diurnal.png",dpi=110); plt.close(fig)

    # summary
    with open(f"{a.outdir}/phase3_summary.txt","w") as fh:
        def w(s): print(s); fh.write(s+"\n")
        w(f"Phase 3 — hourly ({H['nfiles']} mo, {H['nyears']} yr) vs 6-hourly ({S['nfiles']} mo)")
        w(f"domain-mean MCS freq: hourly={np.nanmean(H['freq']):.5f}  6-hourly={np.nanmean(S['freq']):.5f}")
        w(f"domain-mean MCS days/yr: hourly={np.nanmean(H['days']):.3f}  6-hourly={np.nanmean(S['days']):.3f}")
        w("region            hourly_peakL  hourly_amp%  6hr_amp%")
        for r in REGIONS:
            hd=H["diurnal"][r]*100; sd=S["diurnal"][r]*100
            hamp=np.nanmax(hd)-np.nanmin(hd); samp=np.nanmax(sd)-np.nanmin(sd)
            w(f"  {r:16s}  {int(np.nanargmax(hd)):02d}L          {hamp:6.3f}      {samp:6.3f}")
    print(f"[done] figures + summary in {a.outdir}", flush=True)

if __name__ == "__main__":
    main()
