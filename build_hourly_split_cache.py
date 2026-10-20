#!/usr/bin/env python3
"""
build_hourly_split_cache.py — hourly predictor cache for MCS inference, 1980-2024.

Why this is inference-only and needs no retraining
--------------------------------------------------
WNetMultiLabel is a stateless per-frame 2D CNN (WNet_model.py:178-190) — no Conv3d,
no recurrence, no time channel. Cadence is a property of how often the model is
called, not of its weights, and the 6-hourly training set is a strict subsample of
the hourly field distribution, so hourly frames are in-distribution. The four
heuristic label channels are never read at inference
(phase4_export_detections.py:160 slices `arr[sel, :6]`), so no hourly CyTRACK,
IBTrACS or wind-shear work is needed.

Why only two channels are native hourly
---------------------------------------
phase1_diagnostics/phase1_results.json, leave-one-out (zero one predictor, IoU of
the resulting output against the full-input output), MCS head:

    IVT 0.981 | MSLP 0.969 | vorticity 0.988 | wind_speed 0.976
    BT  0.000 | IWV  0.001

The MCS head collapses only when BT or IWV is removed and is >96% unchanged when
any dynamical channel is removed. BT and IWV are therefore taken from real hourly
sources; IVT, MSLP, vorticity and wind850 are linearly interpolated in time from
the existing 6-hourly cache, which keeps them physically plausible and in
distribution. That is also the only practical choice: hourly vorticity850 exists
on disk at 3-hourly cadence on a 640x1280 Gaussian grid, and hourly wind850 would
require an 850 mb subset of ~57 TB of full-level u/v.

The interpolation is not taken on faith. --dynamics nearest rebuilds the same month
holding the nearest 6-hourly frame instead; if the MCS output agrees between the
two, the fill method is demonstrably not load-bearing. Run both and compare.

Provenance, verified against the existing 6-hourly cache for 2004-07
--------------------------------------------------------------------
Both sources are on lat 90..-90, lon 0..359.75 and must be rolled to -180..180 to
match the cache, then bilinearly resampled to 192x384:

    BT  (from era5_bt_hourly_YYYY.nc)  corr 1.000000, MAE 0.0041 K, max|d| 0.008 K
    IWV (TCWV in era5dara_hourly_*.nc) corr 0.998800, MAE 0.46 mm

BT is exact to float rounding. IWV differs slightly because the 6-hourly cache's
IWV and hourly TCWV have different provenance; build_hourly_moisture.py records
the same 0.9986 agreement.

Output is byte-compatible with tensor_cache_v3_split apart from carrying 6 channels
instead of 10, so phase4_export_detections.py consumes it unmodified:

    phase4_export_detections.py --cache-dir <outdir> --classes MCS \
        --years 1980 ... 2024 --subdirs --outdir <...>

Values are stored RAW; normalisation happens at read time from predictor_stats.json,
whose climatological mean/std are cadence-insensitive and are reused unchanged.

Size: ~1.3 GB/month, ~650 GB for 1980-2024.

Usage
  # one month, to a scratch dir
  build_hourly_split_cache.py --years 2004 --months 7 --outdir /path/to/cache

  # full record, MPI-sharded (SLURM_PROCID / SLURM_NTASKS)
  srun build_hourly_split_cache.py --years 1980 ... 2024 --outdir /path/to/cache
"""
import os
import sys
import argparse
import time as _time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data"
SIXHR_CACHE = os.path.join(NB, "Storms_precipitation_notebooks",
                           "tensor_cache_v3_split")
BT_HOURLY_DIR = os.path.join(NB, "OLR", "brightness_temp_hourly")
DARA_HOURLY_DIR = os.path.join(NB, "data_hourly")

N_PRED = 6
TARGET = (192, 384)
# Channel order must match dataset.py PREDICTOR_SPECS exactly.
#   0 IVT  1 MSLP  2 vorticity  3 wind_speed  4 BT  5 IWV
DYN_CHANNELS = [0, 1, 2, 3]
BT_CH, IWV_CH = 4, 5


def _regrid(a):
    """(T,721,1440) on lon 0..360 -> (T,192,384) on lon -180..180.

    The roll is what makes this match the cache: cache_loader.py:139-141 rolls the
    source longitudes to -180..180 before anything else, and skipping it drops the
    BT correlation from 1.000000 to 0.65.
    """
    import torch
    import torch.nn.functional as F
    a = np.ascontiguousarray(a, dtype=np.float32)
    a = np.roll(a, a.shape[-1] // 2, axis=-1)
    t = torch.from_numpy(a).unsqueeze(1)                    # (T,1,H,W)
    t = F.interpolate(t, size=TARGET, mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()


def _month_bounds(year, month):
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00:00")
    nxt = np.datetime64(f"{year:04d}-{month:02d}-01") + np.timedelta64(32, "D")
    end = np.datetime64(f"{str(nxt)[:7]}-01T00:00")
    return start, end


# The hourly BT files are per-YEAR and 36 GB each, so reopening one per month is the
# single most expensive thing this script can do. Hold the current year's handle open
# and let consecutive months of that year reuse it (and the OS page cache).
_BT_HANDLE = {"year": None, "ds": None}


def _bt_dataset(year):
    import xarray as xr
    if _BT_HANDLE["year"] != year:
        if _BT_HANDLE["ds"] is not None:
            _BT_HANDLE["ds"].close()
        path = os.path.join(BT_HOURLY_DIR, f"era5_bt_hourly_{year:04d}.nc")
        _BT_HANDLE["ds"] = xr.open_dataset(path)
        _BT_HANDLE["year"] = year
    return _BT_HANDLE["ds"]


def load_hourly_field(kind, year, month, chunk=48):
    """Return (times, (T,192,384) float32) for 'bt' or 'iwv'."""
    import xarray as xr
    start, end = _month_bounds(year, month)

    if kind == "bt":
        ds = _bt_dataset(year)
        close_after = False
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        da = ds["bt"].sel({tname: slice(start, end - np.timedelta64(1, "h"))})
    else:
        path = os.path.join(DARA_HOURLY_DIR,
                            f"era5dara_hourly_{year:04d}{month:02d}.nc")
        ds = xr.open_dataset(path)
        close_after = True
        tname = "time" if "time" in ds.dims else "valid_time"
        da = ds["IWV"].sel({tname: slice(start, end - np.timedelta64(1, "h"))})

    times = da[tname].values.astype("datetime64[ns]")
    nT = len(times)
    out = np.empty((nT, TARGET[0], TARGET[1]), dtype=np.float32)
    # Chunked so peak memory stays ~200 MB instead of the 3 GB a whole month of
    # 721x1440 float32 would take.
    for s in range(0, nT, chunk):
        e = min(s + chunk, nT)
        out[s:e] = _regrid(da.isel({tname: slice(s, e)}).values)
    if close_after:
        ds.close()
    return times, out


def load_dynamics(year, month):
    """6-hourly ch0-3 plus the next month's first frame, and their timestamps.

    The extra frame closes the gap at the end of the month: the 6-hourly axis stops
    at 18:00 on the last day, so hours 19:00-23:00 have no right-hand bracket
    without it. At the end of the record it is simply absent and those hours clamp
    to the last available frame.
    """
    tag = f"{year:04d}{month:02d}"
    npy = os.path.join(SIXHR_CACHE, f"split_{tag}.npy")
    meta = os.path.join(SIXHR_CACHE, f"split_{tag}.meta.npy")
    if not (os.path.exists(npy) and os.path.exists(meta)):
        raise FileNotFoundError(f"6-hourly cache missing for {tag}")
    arr = np.load(npy, mmap_mode="r")
    times = np.load(meta, allow_pickle=True).astype("datetime64[ns]")
    dyn = np.asarray(arr[:, DYN_CHANNELS], dtype=np.float32)

    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    ntag = f"{ny:04d}{nm:02d}"
    nnpy = os.path.join(SIXHR_CACHE, f"split_{ntag}.npy")
    nmeta = os.path.join(SIXHR_CACHE, f"split_{ntag}.meta.npy")
    if os.path.exists(nnpy) and os.path.exists(nmeta):
        narr = np.load(nnpy, mmap_mode="r")
        ntimes = np.load(nmeta, allow_pickle=True).astype("datetime64[ns]")
        dyn = np.concatenate([dyn, np.asarray(narr[:1, DYN_CHANNELS],
                                              dtype=np.float32)], axis=0)
        times = np.concatenate([times, ntimes[:1]])
    return times, dyn


def resample_dynamics(src_times, src, dst_times, mode="interp"):
    """Map 6-hourly (S,4,H,W) onto the hourly axis -> (T,4,H,W)."""
    s_f = src_times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    d_f = dst_times.astype("datetime64[s]").astype(np.int64).astype(np.float64)

    if mode == "nearest":
        idx = np.abs(d_f[:, None] - s_f[None, :]).argmin(axis=1)
        return src[idx]

    # np.searchsorted gives the right-hand bracket; clip keeps the ends in range so
    # timestamps outside the source span clamp to the first/last frame rather than
    # extrapolating.
    hi = np.clip(np.searchsorted(s_f, d_f, side="left"), 1, len(s_f) - 1)
    lo = hi - 1
    span = s_f[hi] - s_f[lo]
    w = np.where(span > 0, (d_f - s_f[lo]) / np.where(span > 0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    # Chunked: src[lo] and src[hi] on the full hourly axis would each materialise a
    # (744,4,192,384) copy, so the whole month costs ~2.6 GB of temporaries on top
    # of the output array. 128 steps at a time keeps that under 500 MB.
    out = np.empty((len(d_f),) + src.shape[1:], dtype=np.float32)
    for s in range(0, len(d_f), 128):
        e = min(s + 128, len(d_f))
        ww = w[s:e, None, None, None]
        out[s:e] = (1.0 - ww) * src[lo[s:e]] + ww * src[hi[s:e]]
    return out


def build_month(year, month, outdir, mode, overwrite=False):
    tag = f"{year:04d}{month:02d}"
    npy_out = os.path.join(outdir, f"split_{tag}.npy")
    meta_out = os.path.join(outdir, f"split_{tag}.meta.npy")
    if not overwrite and os.path.exists(npy_out) and os.path.exists(meta_out):
        return "skip", 0

    t0 = _time.time()
    bt_times, bt = load_hourly_field("bt", year, month)
    iwv_times, iwv = load_hourly_field("iwv", year, month)

    # BT and IWV come from different products; align on the intersection rather
    # than assuming both are a complete month, so a short file is caught here
    # instead of producing a silently misaligned cache.
    if len(bt_times) != len(iwv_times) or not np.array_equal(bt_times, iwv_times):
        common = np.intersect1d(bt_times, iwv_times)
        if len(common) == 0:
            raise ValueError(f"{tag}: BT and IWV share no timestamps")
        bt = bt[np.searchsorted(bt_times, common)]
        iwv = iwv[np.searchsorted(iwv_times, common)]
        print(f"  [{tag}] BT {len(bt_times)} / IWV {len(iwv_times)} stamps -> "
              f"{len(common)} common", flush=True)
        times = common
    else:
        times = bt_times

    src_times, dyn_src = load_dynamics(year, month)
    dyn = resample_dynamics(src_times, dyn_src, times, mode=mode)

    nT = len(times)
    out = np.empty((nT, N_PRED, TARGET[0], TARGET[1]), dtype=np.float32)
    out[:, DYN_CHANNELS] = dyn
    out[:, BT_CH] = bt
    out[:, IWV_CH] = iwv

    os.makedirs(outdir, exist_ok=True)
    # Write-then-rename. The job is resumable by "skip if the file exists", so a
    # walltime kill part-way through a 1.3 GB np.save would otherwise leave a
    # truncated month that the next run silently accepts as complete.
    tmp_npy = f"{npy_out}.tmp{os.getpid()}"
    tmp_meta = f"{meta_out}.tmp{os.getpid()}"
    try:
        # Save through file handles: np.save(path) would append a second ".npy"
        # to these temp names and the rename below would then miss the file.
        with open(tmp_npy, "wb") as f:
            np.save(f, out)
        with open(tmp_meta, "wb") as f:
            np.save(f, times)
        # meta first, then the array: the array is what the skip test keys on, so
        # it must be the last thing to appear.
        os.replace(tmp_meta, meta_out)
        os.replace(tmp_npy, npy_out)
    finally:
        for p in (tmp_npy, tmp_meta):
            if os.path.exists(p):
                os.remove(p)
    dt = _time.time() - t0
    print(f"  [{tag}] {nT} steps, {out.nbytes / 1024**3:.2f} GB, {dt:.0f}s "
          f"(dynamics={mode})", flush=True)
    return "ok", nT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--months", nargs="+", type=int,
                    default=list(range(1, 13)))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dynamics", choices=["interp", "nearest"], default="interp",
                    help="how ch0-3 are mapped from 6-hourly. 'nearest' exists to "
                         "test whether the choice matters at all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    size = int(os.environ.get("SLURM_NTASKS", 1))

    tags = sorted((y, m) for y in args.years for m in args.months)
    # CONTIGUOUS blocks, not round-robin: a rank then works through consecutive
    # months of the same year and reuses one open 36 GB BT file (see _bt_dataset)
    # instead of reopening a different year's file every month.
    n = len(tags)
    lo = rank * n // size
    hi = (rank + 1) * n // size
    mine = tags[lo:hi]
    span = f"{mine[0][0]}-{mine[0][1]:02d}..{mine[-1][0]}-{mine[-1][1]:02d}" if mine else "none"
    print(f"rank {rank}/{size}: {len(mine)} of {n} months [{span}] -> {args.outdir}",
          flush=True)

    n_ok = n_skip = n_fail = 0
    for (y, m) in mine:
        try:
            status, _ = build_month(y, m, args.outdir, args.dynamics,
                                    overwrite=args.overwrite)
            n_ok += status == "ok"
            n_skip += status == "skip"
        except Exception as e:
            n_fail += 1
            print(f"  [{y:04d}{m:02d}] FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"rank {rank}: built {n_ok}, skipped {n_skip}, failed {n_fail}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
