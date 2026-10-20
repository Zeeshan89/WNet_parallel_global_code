#!/usr/bin/env python3
"""
phase4_build_etc_label.py — rebuild the ETC label (channel 8 of the split cache) with closed
MSLP lows as cyclone centres, as native 0.25 degree masks for testing and as a new cache.

Why. The shipped label (dataset.py:_create_etc_masks) accepts a local minimum of smoothed
MSLP when the MAXIMUM pressure in an 800 km box exceeds it by 1 hPa, which is not a closed
contour and passes reduction artefacts over high terrain; it then gates the disc on an
ABSOLUTE pressure (MSLP <= 1010 hPa NH, 1015 SH), which structurally excludes shallow
cyclones (NAECC's shallowest quintile, > 1001 hPa, is matched at 0.33 against TRACK's
0.85 even with the improved tracker); and it stops at 70 degrees. The tracker fix (S15,
closed 150 Pa contour within 6 degrees, etc_closed_contour.py) could not touch those,
because the mask never covered the cyclones.

The `cc` label keeps everything else of the shipped label (600 km NH / 700 km SH discs of
FIXED pixel radius around the centres, the gridded 75th-percentile vorticity and wind gates,
the 300 km "moderate" branch, the bilinear resize to 192 x 384) and changes three things:
  1. centres = closed lows of the smoothed MSLP (150 Pa within 6 degrees, merged within
     6 degrees), the S15 tracker's criterion, so label and tracker share one definition;
  2. no absolute pressure gate: the closed contour is the (relative) depth criterion;
  3. latitude band 20-85 instead of 20-70, and dilation wraps across the dateline.

  --mode cache --source legacy --verify   the legacy path must reproduce channel 8 of the
                                          existing cache (agreement about 1.0) BEFORE the new
                                          label is trusted (phase4_rebuild_hurricane_label.py
                                          protocol)
  --mode native --source cc legacy        writes <native-dir>/<source>/era5dara_YYYYMM_ETCs_detection.nc
                                          in the layout of the exported detections, so the
                                          tracker and phase4_etc_miss_attribution.py --det-dir
                                          can score the LABEL itself
  --mode cache --source cc --src-cache <v6> --outdir <v7>   new cache, channel 8 only

One (year, month) per task, round-robin over the ranks; finished months are skipped.
"""
import os
import sys
import argparse

import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import etc_closed_contour as cc                                   # noqa: E402
from phase4_tc_label_screen import load_month                     # noqa: E402  (MSLP, vort, wind on lon -180..180)

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
VOR_PCT = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/"
           "output/vor850_75th_percentile_global.nc")
WIND_PCT = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/wind_speed_850/output/windspeed850_75th_percentile_global.nc"
NATIVE_DIR = os.path.join(NB, "ETC_label_candidates")
SRC_CACHE = os.path.join(NB, "tensor_cache_v6_arlit")           # carries the new AR label already
ETC_CH = 8                                                       # 6 predictors then [AR, Hurricane, ETC, MCS]
TARGET = (192, 384)
GRID_KM = 27.5                                                   # the label's pixel scale (dataset.py)
CC = dict(depth_pa=150.0, radius_deg=6.0, sigma_km=75.0)         # = the S15 tracker
BAND = {"legacy": (20.0, 70.0), "cc": (20.0, 85.0)}


class _Shim:
    """What _create_etc_masks reads from the dataset object."""

    def __init__(self, mslp, vort, wind, lats, vthr, wthr):
        self.mslp, self.vorticity, self.wind_speed, self.lats = mslp, vort, wind, lats
        self.vor850_threshold, self.windspeed_threshold = vthr, wthr


def resize(a):
    """Bilinear to 192x384, matching _compute_single_sample's resize (dataset.py:703-718)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(a[:, None], dtype=np.float32))
    return F.interpolate(t, size=TARGET, mode="bilinear", align_corners=False)[:, 0].numpy()


def thresholds():
    """The gridded 75th-percentile vorticity / wind thresholds exactly as the dataset holds them."""
    from utils import load_percentile_dataset
    out = []
    for p in (VOR_PCT, WIND_PCT):
        a = load_percentile_dataset(p).values.astype(np.float32)
        out.append(np.nan_to_num(np.squeeze(a), nan=0.0, posinf=0.0, neginf=0.0))
    return out


def lat_mask(lats, band, shape):
    a = np.abs(lats)
    return np.broadcast_to(((a >= band[0]) & (a <= band[1]))[:, None], shape).astype(np.float32)


def legacy_label(shim, lats):
    import types
    from dataset import ARDatasetImproved as DS
    # _create_etc_masks reaches its helpers through self; give the shim the production ones
    for name in ("_smooth_mslp", "_detect_mslp_local_minima", "_dilate_mslp_centers"):
        setattr(shim, name, types.MethodType(getattr(DS, name), shim))
    m2 = lat_mask(lats, BAND["legacy"], shim.mslp.shape[1:])
    out = np.zeros(shim.mslp.shape, np.float32)
    for i in range(shim.mslp.shape[0]):
        out[i] = DS._create_etc_masks(shim, i, m2)["etc_enhanced"]
    return out


def dilate_wrap(mask, km):
    """The label's fixed-pixel disc dilation, periodic in longitude."""
    r = int(km / GRID_KM)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (x * x + y * y) <= r * r
    padded = np.pad(mask > 0, ((0, 0), (r, r)), mode="wrap")
    return binary_dilation(padded, structure=disc)[:, r:-r].astype(np.float32)


def cc_label(shim, lats, lons):
    nt, ny, nx = shim.mslp.shape
    lat2d = np.broadcast_to(lats[:, None], (ny, nx))
    m2 = lat_mask(lats, BAND["cc"], (ny, nx))
    vthr = shim.vor850_threshold if shim.vor850_threshold.ndim == 2 else np.broadcast_to(shim.vor850_threshold, (ny, nx))
    out = np.zeros((nt, ny, nx), np.float32)
    for i in range(nt):
        p = cc.smooth(shim.mslp[i], lats, CC["sigma_km"])
        centres = np.zeros((ny, nx), np.float32)
        for la, lo, _ in cc.closed_lows(p, lats, lons, depth_pa=CC["depth_pa"], radius_deg=CC["radius_deg"],
                                        merge_deg=CC["radius_deg"], sigma_km=0, max_abs_lat=BAND["cc"][1]):
            if abs(la) < BAND["cc"][0]:
                continue
            centres[int(np.argmin(np.abs(lats - la))), int(np.argmin(np.abs(lons - lo)))] = 1.0
        region = np.maximum(dilate_wrap(centres * (lat2d >= 0), 600.0), dilate_wrap(centres * (lat2d < 0), 700.0))
        vort = ((lat2d >= 0) & (shim.vorticity[i] >= vthr)) | ((lat2d < 0) & (shim.vorticity[i] <= -vthr))
        wind = shim.wind_speed[i] >= shim.windspeed_threshold
        vort, wind = vort.astype(np.float32) * m2, wind.astype(np.float32) * m2
        high = region * m2 * vort * wind
        moderate = dilate_wrap(centres * np.maximum(vort, wind), 300.0)
        out[i] = np.maximum(high, moderate)
    return out


def build(year, month, sources, thr):
    mslp, vort, wind, lats, times, _ = load_month(year, month)
    lons = np.linspace(-180.0, 180.0 - 360.0 / mslp.shape[2], mslp.shape[2])
    shim = _Shim(mslp, vort, wind, lats, *thr)
    out = {}
    if "legacy" in sources:
        out["legacy"] = legacy_label(shim, lats)
    if "cc" in sources:
        out["cc"] = cc_label(shim, lats, lons)
    return out, lats, lons, times


def write_native(path, mask, lats, lons, times, note):
    from netCDF4 import Dataset
    t0 = times[0].to_pydatetime().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    tmp = f"{path}.tmp{os.getpid()}"
    with Dataset(tmp, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("latitude", lats.size)
        ds.createDimension("longitude", lons.size)
        tv = ds.createVariable("time", "i8", ("time",))
        tv.units = f"hours since {t0:%Y-%m-%d %H:%M:%S}"
        tv.calendar = "standard"
        tv[:] = [(t.to_pydatetime() - t0).total_seconds() / 3600.0 for t in times]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lats
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lons
        v = ds.createVariable("ETCs_detection", "f4", ("time", "latitude", "longitude"), zlib=True,
                              complevel=1, chunksizes=(1, lats.size, lons.size))
        v[:] = mask
        v.long_name = "ETC label " + note
        ds.source = "phase4_build_etc_label.py"
    os.replace(tmp, path)


def do_native(year, month, args, thr):
    tag = f"{year:04d}{month:02d}"
    paths = {s: os.path.join(args.native_dir, s, f"era5dara_{tag}_ETCs_detection.nc") for s in args.source}
    todo = [s for s in args.source if args.overwrite or not os.path.exists(paths[s])]
    if not todo:
        print(f"  {tag}: done", flush=True)
        return
    masks, lats, lons, times = build(year, month, todo, thr)
    w = np.cos(np.radians(lats))[:, None]
    for s in todo:
        os.makedirs(os.path.dirname(paths[s]), exist_ok=True)
        write_native(paths[s], masks[s], lats, lons, times, s)
    print(f"  {tag}: " + "  ".join(f"{s} {100 * (masks[s].mean(0) * w).sum() / (w.sum() * lons.size):.2f}%" for s in todo),
          flush=True)


def do_cache(year, month, args, thr):
    import shutil
    tag = f"{year:04d}{month:02d}"
    src_npy = os.path.join(args.src_cache, f"split_{tag}.npy")
    src_meta = os.path.join(args.src_cache, f"split_{tag}.meta.npy")
    if not os.path.exists(src_npy):
        print(f"  {tag}: MISSING {src_npy}", flush=True)
        return
    dst = os.path.join(args.outdir, f"split_{tag}.npy") if args.outdir else None
    if dst and os.path.exists(dst) and not (args.verify or args.overwrite):
        print(f"  {tag}: done", flush=True)
        return
    source = args.source[0]
    masks, lats, lons, times = build(year, month, [source], thr)
    meta = np.load(src_meta, allow_pickle=True).astype("datetime64[h]")
    got = np.array([np.datetime64(t.to_pydatetime(), "h") for t in times])
    if meta.shape != got.shape or not (meta == got).all():
        raise SystemExit(f"{tag}: time axis of the source files differs from the cache meta")
    lab = resize(masks[source])
    old = np.load(src_npy, mmap_mode="r")
    a, b = lab > 0.5, np.asarray(old[:, ETC_CH]) > 0.5
    print(f"  {tag}: {source} positive rate {b.mean():.5f} -> {a.mean():.5f}; IoU with existing channel {ETC_CH}: "
          f"{(a & b).sum() / max((a | b).sum(), 1):.5f}; agreement {(a == b).mean():.6f}; "
          f"MAE {np.abs(lab - np.asarray(old[:, ETC_CH])).mean():.2e}", flush=True)
    if args.verify:
        return
    arr = np.array(old, dtype=np.float32)
    arr[:, ETC_CH] = lab
    os.makedirs(args.outdir, exist_ok=True)
    shutil.copyfile(src_meta, os.path.join(args.outdir, f"split_{tag}.meta.npy"))   # meta first (see AR builder)
    tmp = f"{dst}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("native", "cache"))
    ap.add_argument("--source", nargs="+", default=["cc"], choices=("legacy", "cc"))
    ap.add_argument("--years", nargs=2, type=int, default=[1980, 2024])
    ap.add_argument("--year-list", nargs="+", type=int)
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--native-dir", default=NATIVE_DIR)
    ap.add_argument("--src-cache", default=SRC_CACHE)
    ap.add_argument("--outdir")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.mode == "cache" and len(args.source) != 1:
        ap.error("--mode cache takes exactly one --source")
    if args.mode == "cache" and not (args.outdir or args.verify):
        ap.error("--mode cache needs --outdir or --verify")
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    thr = thresholds()
    years = args.year_list or range(args.years[0], args.years[1] + 1)
    for y, m in [(y, m) for y in years for m in args.months][rank::world]:
        (do_native if args.mode == "native" else do_cache)(y, m, args, thr)


if __name__ == "__main__":
    sys.exit(main())
