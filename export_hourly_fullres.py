#!/usr/bin/env python3
"""
export_hourly_fullres.py — HOURLY MCS detections from the 0.25 deg WNet (J_fullres_e60/best_mcs.pth), 1980-2024,
built on the fly (a 0.25 deg hourly predictor cache would be ~10 TB).

Predictors per hour, exactly as build_hourly_split_cache.py does for 192x384 but without the resize:
  BT  (ch4)  Storms_data/OLR/brightness_temp_hourly/era5_bt_hourly_YYYY.nc     rolled to lon -180..180
  IWV (ch5)  Storms_data/data_hourly/era5dara_hourly_YYYYMM.nc (IWV)            rolled to lon -180..180
  IVT, MSLP, vort850, wind850 (ch0-3): linear in time between the 6-hourly frames of tensor_cache_v11_fullres
             (+ the next month's first frame); the MCS head depends on BT and IWV (leave-one-out IoU 0.000 / 0.001),
             not on these (>= 0.969).
Model on the 0.25 deg grid (rows padded 721 -> 728, output cropped), threshold 0.5, written like
phase4_export_detections.py (MCS_detection float32, lat 90..-90, lon -180..179.75, zlib 4) to
<outdir>/MCS_WNet_final/era5dara_YYYYMM_MCS_detection.nc. Months are given to ranks in CONTIGUOUS blocks so each
rank keeps one 36 GB yearly BT file open. Resumable (.tmp then rename).
  srun ... python export_hourly_fullres.py --years 1980 ... --outdir <dir>
"""
import os
import sys
import json
import time
import argparse

_rank = int(os.environ.get("SLURM_PROCID", 0))
_mc = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_mc, exist_ok=True)
os.environ.update(MIOPEN_USER_DB_PATH=_mc, MIOPEN_CUSTOM_CACHE_DIR=_mc, MIOPEN_FIND_MODE="2", MIOPEN_DISABLE_CACHE="0")

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hourly_split_cache as H          # noqa: E402  (month bounds, BT handle, dynamics interpolation)
from phase4_export_detections import load_model   # noqa: E402

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
CACHE = os.path.join(NB, "tensor_cache_v11_fullres")
H.SIXHR_CACHE = CACHE                          # load_dynamics now reads the 0.25 deg 6-hourly cache


def load_native(kind, year, month):
    """(times, (T,721,1440) float32) on lon -180..180 — load_hourly_field without the resize."""
    import xarray as xr
    start, end = H._month_bounds(year, month)
    if kind == "bt":
        ds = H._bt_dataset(year)
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        da = ds["bt"].sel({tname: slice(start, end - np.timedelta64(1, "h"))})
        close = None
    else:
        ds = xr.open_dataset(os.path.join(H.DARA_HOURLY_DIR, f"era5dara_hourly_{year:04d}{month:02d}.nc"))
        tname = "time" if "time" in ds.dims else "valid_time"
        da = ds["IWV"].sel({tname: slice(start, end - np.timedelta64(1, "h"))})
        close = ds
    times = da[tname].values.astype("datetime64[ns]")
    out = np.empty((len(times), 721, 1440), np.float32)
    for s in range(0, len(times), 48):
        a = np.asarray(da.isel({tname: slice(s, s + 48)}).values, np.float32)
        out[s:s + 48] = np.roll(a, a.shape[-1] // 2, axis=-1)
    if close is not None:
        close.close()
    return times, out


@torch.no_grad()
def month(model, dev, centre, scale, year, mon, outdir, bs=2):
    import xarray as xr
    tag = f"{year:04d}{mon:02d}"
    dst = os.path.join(outdir, "MCS_WNet_final", f"era5dara_{tag}_MCS_detection.nc")
    if os.path.exists(dst):
        return "skip"
    t0 = time.time()
    bt_t, bt = load_native("bt", year, mon)
    iw_t, iw = load_native("iwv", year, mon)
    times = np.intersect1d(bt_t, iw_t)
    bt, iw = bt[np.searchsorted(bt_t, times)], iw[np.searchsorted(iw_t, times)]
    src_t, dyn = H.load_dynamics(year, mon)          # (S,4,721,1440) incl. next month's first frame
    nT = len(times)
    mask = np.zeros((nT, 721, 1440), np.float32)
    ph = (-721) % 8
    for s in range(0, nT, bs):
        e = min(s + bs, nT)
        d = H.resample_dynamics(src_t, dyn, times[s:e])            # (b,4,H,W) linear in time
        x = np.concatenate([d, bt[s:e, None], iw[s:e, None]], axis=1)
        x = (x - centre[None, :, None, None]) / scale[None, :, None, None]
        xt = F.pad(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev), (0, 0, 0, ph), mode="replicate")
        _, p = model.encode(xt)
        mask[s:e] = (p[:, 3, :721] > 0.5).float().cpu().numpy()          # head 3 = MCS
    del bt, iw, dyn
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.linspace(-180.0, 180.0 - 0.25, 1440)
    ds = xr.Dataset({"MCS_detection": (("time", "latitude", "longitude"), mask)},
                    coords={"time": times, "latitude": lat, "longitude": lon})
    ds["MCS_detection"].attrs = {"long_name": "MCS detection (WNet Phase 2)",
                                 "source_grid": "native 0.25 deg model grid (721x1440), no resampling; hourly: BT/IWV "
                                                "hourly, IVT/MSLP/vort/wind linear in time from 6-hourly",
                                 "prob_threshold": 0.5, "domain_mask": "none",
                                 "checkpoint": "J_fullres_e60/best_mcs.pth"}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    ds.to_netcdf(tmp, encoding={"MCS_detection": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    os.replace(tmp, dst)
    return f"{nT} steps {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--outdir", default=os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_fullres025_hourly"))
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "ARs_Hurricanes_ETCs_MCS/Phase2/J_fullres_e60/best_mcs.pth"))
    ap.add_argument("--stats", default=os.path.join(HERE, "predictor_stats_fullres.json"))
    a = ap.parse_args()
    rank, size = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    ngpu = max(torch.cuda.device_count(), 1)
    dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0)) % ngpu}" if torch.cuda.is_available() else "cpu")
    st = json.load(open(a.stats))["stats"]
    from dataset import ARDatasetImproved as DS
    centre = np.array([st[n]["mean"] for n in DS.PREDICTOR_NAMES], np.float32)
    scale = np.array([st[n]["std"] for n in DS.PREDICTOR_NAMES], np.float32)
    model, ep = load_model(a.checkpoint, dev)
    tags = sorted((y, m) for y in a.years for m in range(1, 13))
    mine = tags[rank * len(tags) // size:(rank + 1) * len(tags) // size]
    print(f"rank {rank}/{size}: {len(mine)} months, checkpoint epoch {ep}", flush=True)
    for y, m in mine:
        try:
            print(f"  [{rank}] {y}{m:02d}: {month(model, dev, centre, scale, y, m, a.outdir)}", flush=True)
        except Exception as e:
            print(f"  [{rank}] {y}{m:02d}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
