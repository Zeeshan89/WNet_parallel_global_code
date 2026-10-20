#!/usr/bin/env python3
"""
phase4_build_ar_label.py — build the literature-based AR label (ar_label_lib.py) from ERA5
Qu and Qv, as native 0.25 degree masks for scoring and as channel 6 of a new training cache.

The model reproduces its label (AR IoU 0.97 against its own label), so a candidate label can
be scored against the established AR methods BEFORE any training: what the label shows is
what the retrained model will show. Candidates:

  L0  control: IVT > 400 (the shipped label; must reproduce channel 6 of the cache)
  L1  Guan-Waliser style: per-cell 5-month percentile, floor 100, geometry + transport tests
  L2  hybrid floor: 250 equatorward of 55, tapering to 100 at 65 and poleward
  L3  L2 + the object must reach |latitude| >= 23.5 (objects confined to the tropics dropped)
  L4  L3 + width <= 1500 km (a too-wide object is thresholded again, not dropped)

Modes
  --mode native   writes <native-dir>/<Lx>/era5dara_YYYYMM_AR_detection.nc for every
                  candidate, in the layout of the exported detections (variable AR_detection,
                  latitude 90..-90, longitude -180..179.75), so phase4_ar_artmip_v2.py and
                  phase4_ar_vs_mars_v2.py score a label exactly as they score the model.
  --mode cache    --source Lx: resizes the label to 192x384 the way the existing label was
                  (bilinear, align_corners=False, kept as float; training thresholds at 0.5)
                  and writes a NEW cache in which only channel 6 differs from --src-cache.
                  --verify compares with the existing channel 6 instead of writing (use with
                  --source L0: agreement must be about 1.0 before any new label is trusted).

One (year, month) per task, round-robin over the ranks; finished months are skipped.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ar_label_lib as L   # noqa: E402

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PCTL_NC = os.path.join(NB, "ivt_only", "ivt_pctl_monthly_5mo.nc")
NATIVE_DIR = os.path.join(NB, "AR_label_candidates")
SRC_CACHE = os.path.join(NB, "tensor_cache_v5_ibtracs")     # v3 would revert the IBTrACS TC label
AR_CH = 6                                                    # 6 predictors then [AR, Hurricane, ETC, MCS]
TARGET = (192, 384)
MAX_WIDTH_KM = 1500.0
# name: (floor profile, object must reach |lat| 23.5, max width km); L0 is the shipped scalar cut.
# L5 and L6 fill the land gap between L1 (floor 100: land ARs like Guan-Waliser, IoU 0.93 with
# it) and L3 (floor 250: land ARs like Mundhenk), both with the tropical rule of L3.
CAND_SPEC = {"L1": ("gw", False, None), "L2": ("hybrid", False, None), "L3": ("hybrid", True, None),
             "L4": ("hybrid", True, MAX_WIDTH_KM), "L5": ("gw", True, None), "L6": ("hybrid175", True, None)}
CANDIDATES = ("L0",) + tuple(CAND_SPEC)


def resize(a):
    """Bilinear to 192x384, matching _compute_single_sample's resize (dataset.py:703-718)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(a[:, None], dtype=np.float32))
    return F.interpolate(t, size=TARGET, mode="bilinear", align_corners=False)[:, 0].numpy()


def read_month(year, month):
    """Qu, Qv (nt, ny, nx), lat, lon, times (datetime64[h]). Whole variables in one read: the
    2024-08..12 files are chunked deep in time."""
    from netCDF4 import Dataset, num2date
    with Dataset(os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc")) as ds:
        ds.set_auto_mask(False)
        qu = np.nan_to_num(ds.variables["Qu"][:].astype(np.float32))
        qv = np.nan_to_num(ds.variables["Qv"][:].astype(np.float32))
        lat = ds.variables["latitude"][:].astype(np.float64)
        lon = ds.variables["longitude"][:].astype(np.float64)
        tv = ds.variables["time"]
        t = num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return qu, qv, lat, lon, np.array([np.datetime64(x, "h") for x in t])


def thresholds(month, lat, kinds=("gw", "hybrid")):
    """{floor kind: (levels, ny, nx)} intensity thresholds for a calendar month."""
    from netCDF4 import Dataset
    with Dataset(PCTL_NC) as ds:
        ds.set_auto_mask(False)
        p = ds.variables["ivt_pctl"][:, month - 1].astype(np.float32)
        plat = ds.variables["latitude"][:]
    assert np.allclose(plat, lat), "percentile grid differs from the data grid"
    return {k: np.maximum(p, L.floor_profile(lat, k)[None]) for k in kinds}


def build(year, month, which):
    """{candidate: (nt, ny, nx) bool} on the native grid (longitude 0..359.75), and times."""
    qu, qv, lat, lon, times = read_month(year, month)
    out = {c: np.zeros(qu.shape, bool) for c in which}
    if "L0" in which:
        out["L0"] = np.hypot(qu, qv) > 400.0
    specs = {c: CAND_SPEC[c] for c in which if c != "L0"}
    if specs:
        runs = sorted({(f, w) for f, _, w in specs.values()}, key=str)   # one detect per (floor, width)
        thr = thresholds(month, lat, {f for f, _ in runs})
        area = L.cell_area(lat, lon)
        for i in range(qu.shape[0]):
            for f, w in runs:
                a, r = L.detect(qu[i], qv[i], thr[f], lat, lon, area, params=dict(max_width_km=w))
                for c, (cf, reach, cw) in specs.items():
                    if (cf, cw) == (f, w):
                        out[c][i] = r if reach else a
    return out, lat, lon, times


def write_native(path, mask, lat, lon, times, note):
    from netCDF4 import Dataset
    half = lon.size // 2
    lon180 = np.roll(((lon + 180.0) % 360.0) - 180.0, half)
    tmp = f"{path}.tmp{os.getpid()}"
    with Dataset(tmp, "w") as ds:
        ds.createDimension("time", times.size)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        tv = ds.createVariable("time", "i8", ("time",))
        tv.units = f"hours since {str(times[0])[:7]}-01 00:00:00"
        tv.calendar = "standard"
        tv[:] = (times - np.datetime64(str(times[0])[:7] + "-01T00", "h")).astype(np.int64)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon180
        v = ds.createVariable("AR_detection", "i1", ("time", "latitude", "longitude"), zlib=True,
                              complevel=1, chunksizes=(1, lat.size, lon.size))
        v[:] = np.roll(mask, half, axis=2).astype(np.int8)
        v.long_name = "AR label candidate " + note
        ds.source = "phase4_build_ar_label.py / ar_label_lib.py"
    os.replace(tmp, path)


def do_native(year, month, args):
    tag = f"{year:04d}{month:02d}"
    paths = {c: os.path.join(args.native_dir, c, f"era5dara_{tag}_AR_detection.nc") for c in args.candidates}
    todo = [c for c in args.candidates if args.overwrite or not os.path.exists(paths[c])]
    if not todo:
        print(f"  {tag}: done", flush=True)
        return
    masks, lat, lon, times = build(year, month, todo)
    w = np.cos(np.radians(lat))[:, None]
    for c in todo:
        os.makedirs(os.path.dirname(paths[c]), exist_ok=True)
        write_native(paths[c], masks[c], lat, lon, times, c)
    print(f"  {tag}: " + "  ".join(
        f"{c} {100 * (masks[c].mean(0) * w).sum() / (w.sum() * lon.size):.2f}%" for c in todo), flush=True)


def do_cache(year, month, args):
    import shutil
    tag = f"{year:04d}{month:02d}"
    src_npy = os.path.join(args.src_cache, f"split_{tag}.npy")
    src_meta = os.path.join(args.src_cache, f"split_{tag}.meta.npy")
    if not os.path.exists(src_npy):
        print(f"  {tag}: MISSING {src_npy}", flush=True)
        return
    dst = os.path.join(args.outdir, f"split_{tag}.npy") if args.outdir else None
    if dst and os.path.exists(dst) and not (args.verify or args.stats_only or args.overwrite):
        print(f"  {tag}: done", flush=True)
        return
    masks, lat, lon, times = build(year, month, [args.source])
    meta = np.load(src_meta, allow_pickle=True).astype("datetime64[h]")
    if meta.shape != times.shape or not (meta == times).all():
        raise SystemExit(f"{tag}: time axis of era5dara differs from the cache meta")
    native = np.roll(masks[args.source], lon.size // 2, axis=2)          # cache longitude is -180..180
    lab = resize(native.astype(np.float32))
    old = np.load(src_npy, mmap_mode="r")
    a, b = lab > 0.5, np.asarray(old[:, AR_CH]) > 0.5
    # what the 192x384 grid can hold of the native label: nearest-neighbour back to 0.25 deg
    iy = np.minimum(np.arange(native.shape[1]) * TARGET[0] // native.shape[1], TARGET[0] - 1)
    ix = np.minimum(np.arange(native.shape[2]) * TARGET[1] // native.shape[2], TARGET[1] - 1)
    up = a[:, iy][:, :, ix]
    ceil = (up & native).sum() / max((up | native).sum(), 1)
    print(f"  {tag}: {args.source} positive rate {b.mean():.5f} -> {a.mean():.5f}; IoU with existing "
          f"channel {AR_CH}: {(a & b).sum() / max((a | b).sum(), 1):.5f}; agreement {(a == b).mean():.6f}; "
          f"MAE {np.abs(lab - np.asarray(old[:, AR_CH])).mean():.2e}; native-vs-192x384 IoU {ceil:.3f}", flush=True)
    if args.verify or args.stats_only:
        return
    arr = np.array(old, dtype=np.float32)
    arr[:, AR_CH] = lab
    os.makedirs(args.outdir, exist_ok=True)
    shutil.copyfile(src_meta, os.path.join(args.outdir, f"split_{tag}.meta.npy"))   # meta first: an npy
    tmp = f"{dst}.tmp{os.getpid()}"                                                   # without meta is skipped forever
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("native", "cache"))
    ap.add_argument("--years", nargs=2, type=int, default=[1980, 2024])
    ap.add_argument("--year-list", nargs="+", type=int, help="explicit years instead of a range")
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    ap.add_argument("--native-dir", default=NATIVE_DIR)
    ap.add_argument("--source", choices=CANDIDATES, help="cache mode: the label to write")
    ap.add_argument("--src-cache", default=SRC_CACHE)
    ap.add_argument("--outdir", help="cache mode: the new cache directory")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.mode == "cache" and not args.source:
        ap.error("--mode cache needs --source")
    if args.mode == "cache" and not (args.outdir or args.verify or args.stats_only):
        ap.error("--mode cache needs --outdir, --verify or --stats-only")
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    years = args.year_list or range(args.years[0], args.years[1] + 1)
    tasks = [(y, m) for y in years for m in args.months][rank::world]
    for y, m in tasks:
        (do_native if args.mode == "native" else do_cache)(y, m, args)


if __name__ == "__main__":
    sys.exit(main())
