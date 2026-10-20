#!/usr/bin/env python3
"""
phase4_tc_label_screen.py — Tier B: screen tightened CyTRACK criteria at the LABEL
level against IBTrACS, without retraining.

Why this works
--------------
Sweeping the model's probability threshold from 0.5 to 0.99 moved TC skill only
0.097 -> 0.110, landing on the training mask's own CSI of 0.1103. The model
converges onto its teacher and stops. So the mask's score against IBTrACS is a
usable predictor of what a retrained model would reach, and candidate criteria can
be ranked for the cost of recomputing masks instead of the cost of retraining.
Only the winner is worth a cache rebuild plus a training run.

What is scored
--------------
The physics detection ONLY. The IBTrACS injection at dataset.py:1365-1419 is
deliberately NOT applied: it merely adds storms the physics missed, so it inflates
POD without touching false alarms and would mask exactly the effect being measured.
Every number here is therefore what the criteria find on their own.

Blob centroids of the `strict` mask are matched to IBTrACS positions within
--radius-km at the same 6-hourly stamp, the same convention phase4_evaluate_tc.py
uses, so these numbers are comparable with the model-level ones.

Fidelity
--------
Runs at native 721x1440 on the real source fields, and calls
ARDatasetImproved._detect_hurricane_centres / _dilate_mslp_centers directly through
a lightweight shim rather than reimplementing them, so a candidate cannot differ
from production by an accident of transcription. The detector's two previously
hardcoded relaxations (equatorial vorticity halving, wind-OR-vorticity) are now
parameters with defaults that reproduce current behaviour.

Usage
  phase4_tc_label_screen.py --year 2003 --month 8 --outdir tc_label_screen
"""
import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
VOR_DIR = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
           "vor850_latlongrid/vor850_latlongrid/")
WIND_FILE = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
             "wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc")
MSLP_CLIM = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
             "data/global_stats/mslp_monthly_climatology.nc")
IBTRACS = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
           "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc")
EARTH_R = 6371.0

# Candidate criteria. "baseline" is the values the production call site passes at
# dataset.py:1313-1324; every other entry changes one thing (or a stated few) so
# the effect is attributable. Defaults of _detect_hurricane_centres are the
# function's own signature values, which are STRICTER than the call site — the
# production mask relaxed every tunable one.
BASELINE = dict(min_slp_threshold=101500.0, mslp_anomaly_threshold=-500.0,
                great_circle_deg=5.5, dmslp_threshold=150.0,
                vorticity_threshold=1.45e-5, filter_center_km=400.0,
                max_wind_threshold=10.0, radius_msw_km=150.0,
                equatorial_vor_scale=0.5, require_both=False)

CANDIDATES = {
    "baseline (production)":      {},
    "revert to fn defaults":      dict(min_slp_threshold=101000.0,
                                       mslp_anomaly_threshold=-800.0,
                                       dmslp_threshold=300.0, radius_msw_km=100.0),
    "dmslp 300 only":             dict(dmslp_threshold=300.0),
    "dmslp 400":                  dict(dmslp_threshold=400.0),
    "anomaly -800 only":          dict(mslp_anomaly_threshold=-800.0),
    "wind 17.5 (34 kt) only":     dict(max_wind_threshold=17.5),
    "no equatorial relax":        dict(equatorial_vor_scale=1.0),
    "require wind AND vort":      dict(require_both=True),
    "defaults + wind 17.5":       dict(min_slp_threshold=101000.0,
                                       mslp_anomaly_threshold=-800.0,
                                       dmslp_threshold=300.0, radius_msw_km=100.0,
                                       max_wind_threshold=17.5),
    "defaults + AND":             dict(min_slp_threshold=101000.0,
                                       mslp_anomaly_threshold=-800.0,
                                       dmslp_threshold=300.0, radius_msw_km=100.0,
                                       require_both=True),
    "defaults + wind + AND":      dict(min_slp_threshold=101000.0,
                                       mslp_anomaly_threshold=-800.0,
                                       dmslp_threshold=300.0, radius_msw_km=100.0,
                                       max_wind_threshold=17.5, require_both=True),
    "strict all":                 dict(min_slp_threshold=101000.0,
                                       mslp_anomaly_threshold=-800.0,
                                       dmslp_threshold=400.0, radius_msw_km=100.0,
                                       max_wind_threshold=17.5, require_both=True,
                                       equatorial_vor_scale=1.0),
}


class _Shim:
    """Minimal stand-in exposing exactly what the two reused methods read."""
    def __init__(self, mslp, vort, wind, lats):
        self.mslp = mslp
        self.vorticity = vort
        self.wind_speed = wind
        self.lats = lats
        self.vor850_hurricane_threshold = 1.45e-5


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _roll(ds):
    """All four sources are stored on lon 0..359.75; the detector and the cache
    both work on -180..180, so every one gets the same roll. Skipping it on any
    single source would silently offset that field by half the globe."""
    return ds.assign_coords(
        longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")


def load_month(year, month):
    """MSLP / vorticity / wind_speed at native resolution, on lon -180..180."""
    import xarray as xr
    import pandas as pd

    src = _roll(xr.open_dataset(
        os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc")))
    mslp = src["MSLP"].values.astype(np.float32)
    times = pd.to_datetime(src["time"].values)
    lats = src["latitude"].values.astype(np.float32)
    src.close()

    vds = xr.open_dataset(os.path.join(
        VOR_DIR, f"ERA5_vor_6hourly_{year:04d}.850mb_latlon.nc"))
    vds = _roll(vds)
    vort = vds["var138"].sel(time=times, method="nearest").values.astype(np.float32)
    vds.close()

    wds = _roll(xr.open_dataset(WIND_FILE))
    wind = wds["wspd"].sel(time=times, method="nearest").values.astype(np.float32)
    wds.close()

    # Named explicitly: this file also carries time_bnds, and taking data_vars[0]
    # would pick that (12, 2) array instead of the (12, 721, 1440) field.
    cds = _roll(xr.open_dataset(MSLP_CLIM))
    clim = cds["MSLP"].values.astype(np.float32)
    cds.close()

    # vorticity and wind carry a singleton level dimension (plev / level).
    vort = np.squeeze(vort)
    wind = np.squeeze(wind)
    for name, arr in (("mslp", mslp), ("vorticity", vort), ("wind", wind)):
        if arr.shape != mslp.shape:
            raise ValueError(f"{name} shape {arr.shape} != mslp {mslp.shape}")
    if clim.shape[0] != 12 or clim.shape[1:] != mslp.shape[1:]:
        raise ValueError(f"climatology shape {clim.shape} incompatible")
    return mslp, vort, wind, lats, times, clim


def ibtracs_by_stamp(year, wind_kt=34.0):
    """dict[datetime] -> [(lat, lon)] for storms at or above wind_kt."""
    from netCDF4 import Dataset, num2date
    from collections import defaultdict
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    idx = np.where(season == year)[0]
    lat = ds.variables["lat"][idx]; lon = ds.variables["lon"][idx]
    wind = ds.variables["usa_wind"][idx]; traw = ds.variables["time"][idx]
    units = ds.variables["time"].units
    cal = getattr(ds.variables["time"], "calendar", "standard")
    ds.close()
    out = defaultdict(list)
    for i in range(len(idx)):
        la = np.ma.filled(lat[i].astype(float), np.nan)
        lo = np.ma.filled(lon[i].astype(float), np.nan)
        wd = np.ma.filled(wind[i].astype(float), np.nan)
        tt = np.ma.filled(traw[i].astype(float), np.nan)
        ok = (np.isfinite(la) & np.isfinite(lo) & np.isfinite(tt)
              & (np.nan_to_num(wd) >= wind_kt))
        if not ok.any():
            continue
        ts = num2date(tt[ok], units, cal, only_use_cftime_datetimes=False,
                      only_use_python_datetimes=True)
        lo_ok = np.where(lo[ok] > 180, lo[ok] - 360, lo[ok])
        for t, a, b in zip(ts, la[ok], lo_ok):
            out[t.replace(minute=0, second=0, microsecond=0)].append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--outdir", default="tc_label_screen")
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--min-pixels", type=int, default=2)
    args = ap.parse_args()

    from scipy import ndimage
    from dataset import ARDatasetImproved as DS

    tag = f"{args.year:04d}{args.month:02d}"
    os.makedirs(args.outdir, exist_ok=True)
    out_npz = os.path.join(args.outdir, f"screen_{tag}.npz")
    if os.path.exists(out_npz):
        print(f"  {tag} already done"); return 0

    mslp, vort, wind, lats, times, clim = load_month(args.year, args.month)
    nT, H, W = mslp.shape
    lons = np.linspace(-180.0, 180.0 - 360.0 / W, W)
    shim = _Shim(mslp, vort, wind, lats)
    truth = ibtracs_by_stamp(args.year, args.wind_kt)

    lat2d = np.broadcast_to(lats[:, None], (H, W))
    lat_mask = ((lat2d >= -45.0) & (lat2d <= 45.0)).astype(np.float32)
    anom_all = mslp - clim[args.month - 1][None, :, :]

    names = list(CANDIDATES)
    acc = {n: dict(hits=0, misses=0, fa=0, n_det=0, n_truth=0) for n in names}

    for i in range(nT):
        stamp = times[i].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        tp = truth.get(stamp, [])
        # vorticity / wind masks are candidate-independent
        nh = (lat2d >= 0) & (vort[i] >= 1.45e-5)
        sh = (lat2d < 0) & (vort[i] <= -1.45e-5)
        vor_m = (nh | sh).astype(np.float32)
        wind_m = (wind[i] >= 10.0).astype(np.float32)
        either = ((vor_m + wind_m) >= 1).astype(np.float32)

        for name in names:
            p = dict(BASELINE); p.update(CANDIDATES[name])
            centres = DS._detect_hurricane_centres(shim, i, mslp_anomaly=anom_all[i], **p)
            dil = DS._dilate_mslp_centers(shim, centres, dilation_km=200.0)
            strict = (mslp[i] < p["min_slp_threshold"]).astype(np.float32) * dil
            strict = strict * either * lat_mask

            dets = []
            if strict.any():
                lab, n = ndimage.label(strict > 0, structure=np.ones((3, 3)))
                for k in range(1, n + 1):
                    sel = lab == k
                    if sel.sum() < args.min_pixels:
                        continue
                    r, c = np.nonzero(sel)
                    ang = np.radians(lons[c])
                    dets.append((lats[r].mean(),
                                 np.degrees(np.arctan2(np.sin(ang).mean(),
                                                       np.cos(ang).mean()))))
            a = acc[name]
            a["n_det"] += len(dets); a["n_truth"] += len(tp)
            used = set()
            for (dla, dlo) in dets:
                best, bj = args.radius_km, None
                for j, (tla, tlo) in enumerate(tp):
                    if j in used:
                        continue
                    d = haversine(dla, dlo, tla, tlo)
                    if d <= best:
                        best, bj = d, j
                if bj is None:
                    a["fa"] += 1
                else:
                    used.add(bj); a["hits"] += 1
            a["misses"] += len(tp) - len(used)

    np.savez(out_npz, names=np.array(names, dtype=object),
             stats=np.array([[acc[n]["hits"], acc[n]["misses"], acc[n]["fa"],
                              acc[n]["n_det"], acc[n]["n_truth"]] for n in names]),
             n_steps=nT)
    print(f"  {tag}: {nT} steps, {len(names)} candidates -> {out_npz}", flush=True)
    for n in names:
        a = acc[n]
        pod = a["hits"] / max(a["hits"] + a["misses"], 1)
        far = a["fa"] / max(a["hits"] + a["fa"], 1)
        csi = a["hits"] / max(a["hits"] + a["misses"] + a["fa"], 1)
        print(f"    {n:<26} POD {pod:.3f} FAR {far:.3f} CSI {csi:.3f} "
              f"det {a['n_det']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
