#!/usr/bin/env python3
"""
phase4_export_detections.py — write Phase 2 model detections as monthly NetCDFs
in exactly the layout the existing trackers consume.

Hurricane_track.py reads   era5dara_YYYYMM_Hurricane_detection.nc  var Hurricane_detection
updated_monthly_ETC_Tracker.py reads
                           era5dara_YYYYMM_ETCs_detection.nc       var ETCs_detection

Both were written against the old pipeline's full-resolution output, so the model
maps (192x384) are upsampled back to the ERA5 grid (721x1440, lat 90..-90,
lon -180..179.75) and written with the same coordinate names. That keeps the
trackers byte-compatible — no tracker edits, so any difference in the resulting
tracks comes from the detections and not from a changed tracking rule.

One exception, 2026-09-23: Hurricane_track.py's movement check had its haversine
arguments swapped (lon, lat instead of lat, lon), which cut tracks at landfalls
near 90W and 90E. That is fixed. TC_HAVERSINE_LEGACY=1 restores the old call, so
tracks made before the fix can be reproduced byte-for-byte.

Nearest-neighbour upsampling is used on purpose: these are binary masks, and
bilinear would produce fractional edge values that a downstream `> 0` test turns
into a fattened blob (the same failure post_processing.py:406-413 documents).

--upsample bilinear (2026-09-25, for AR and MCS): interpolate the model PROBABILITY to
the output grid and threshold there. The mask stays binary, so the fattening above
cannot happen, and the 0.94 deg staircase edges go away. Against nearest on 2003
(compare_upsampling.py): same systems (centroid shift ~20 km), area x0.98 AR / x0.94
MCS, monthly frequency maps r >= 0.99 at 1 deg. Nearest stays the default, so the
TC/ETC tracks and the RegCM exports are unchanged.

Usage
  python3 phase4_export_detections.py --checkpoint <best.pth> --years 1983 1994 2003 2012 2021
"""

import os

_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_DISABLE_CACHE"] = "0"

import sys
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_model import WNetMultiLabel
from dataset import ARDatasetImproved as DS
from train import TEST_YEARS

N_PRED = DS.N_PREDICTORS
PRED_NAMES = DS.PREDICTOR_NAMES
ERA5_SHAPE = (721, 1440)

# class -> (head index, output filename suffix, NetCDF variable name)
EXPORTS = OrderedDict([
    ("Hurricane", (1, "Hurricane_detection", "Hurricane_detection")),
    ("ETC",       (2, "ETCs_detection",      "ETCs_detection")),
    ("AR",        (0, "AR_detection",        "AR_detection")),
    ("MCS",       (3, "MCS_detection",       "MCS_detection")),
])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = WNetMultiLabel(in_channels=N_PRED, num_classes=4,
                       discretize=ckpt.get("discretize", "ste")).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    m.load_state_dict(state)
    m.eval()
    return m, ckpt.get("epoch", "?")


def upsample_nn(mask_tw_hw, shape=ERA5_SHAPE):
    """(T,H,W) binary -> (T,*shape) nearest neighbour.

    Nearest, not bilinear: these are binary masks and any interpolation invents
    fractional detections. For ERA5 this is an UPsample (192x384 -> 721x1440); for
    RegCM it is a DOWNsample back to the native 156x328, where nearest is doubly
    correct because it cannot manufacture a detection that the model did not make.
    """
    t = torch.from_numpy(mask_tw_hw.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=tuple(shape), mode="nearest")
    return t.squeeze(1).numpy().astype(np.float32)


def upsample_bilinear_prob(prob_tw_hw, thr, shape=ERA5_SHAPE, device="cpu", chunk=62):
    """(T,H,W) probability -> (T,*shape) binary: bilinear on the probability, then > thr."""
    out = np.zeros((prob_tw_hw.shape[0],) + tuple(shape), np.float32)
    for s in range(0, prob_tw_hw.shape[0], chunk):
        t = torch.from_numpy(prob_tw_hw[s:s + chunk]).unsqueeze(1).to(device)
        t = F.interpolate(t, size=tuple(shape), mode="bilinear", align_corners=False)
        out[s:s + chunk] = (t.squeeze(1) > thr).cpu().numpy()
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--years", nargs="+", type=int, default=list(TEST_YEARS))
    ap.add_argument("--classes", nargs="+", default=["Hurricane", "ETC"],
                    help="which classes to export")
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--subdirs", action="store_true",
                    help="write each class into its own subdirectory, mirroring "
                         "the layout of Global_ARs_Hurr_ETCs_MCS_finalized")
    # ── dataset-agnostic output geometry (added for RegCM) ──────────────────
    # Defaults reproduce the ERA5 behaviour exactly: 721x1440, lat 90..-90.
    ap.add_argument("--file-prefix", default="era5dara_",
                    help="output filename prefix; RegCM trackers expect "
                         "'<MODEL>dara_' (Hurricane_track.py file_prefix)")
    ap.add_argument("--out-lat", default=None,
                    help="1-D .npy of output latitudes. RegCM must export on its "
                         "NATIVE 156x328 0.25deg grid with ASCENDING latitude — "
                         "that is the grid the paired <M>dara_ met fields live on "
                         "and the one mask_to_domain derives the domain from.")
    ap.add_argument("--out-lon", default=None, help="1-D .npy of output longitudes")
    ap.add_argument("--out-mask", default=None,
                    help="boolean .npy on the OUTPUT grid; detections are zeroed "
                         "outside it. REQUIRED whenever training used --valid-mask: "
                         "the loss never penalised those pixels, so the model is "
                         "unconstrained there and emits arbitrary output. Measured "
                         "without it on RegCM: 84%% of NOR and 89%% of MPI hurricane "
                         "detections, and 36-65%% of ETC, lay outside the domain.")
    ap.add_argument("--upsample", choices=["nearest", "bilinear", "native"], default="nearest",
                    help="model grid -> output grid: nearest on the thresholded mask (default, "
                         "the shipped TC/ETC path), bilinear on the probability then threshold, or "
                         "native: the model runs ON the output grid (0.25 deg model + 0.25 deg cache, "
                         "rows padded to a multiple of 8, cropped back), no resampling at all")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite months that already exist (default: skip, so a "
                         "timed-out job can simply be resubmitted)")
    args = ap.parse_args()

    # Round-robin month sharding across ranks. Inference is trivial here (~20 s
    # for the whole record on 64 GPUs); the cost is writing 2160 NetCDFs, so the
    # parallelism is really about spreading the I/O.
    shard_rank = int(os.environ.get("SLURM_PROCID", 0))
    shard_size = int(os.environ.get("SLURM_NTASKS", 1))

    import xarray as xr

    # several ranks may share a node's GPUs (the write is CPU-bound): rank -> GPU round-robin
    device = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0)) % torch.cuda.device_count()}"
                          if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.stats) as f:
        stats = json.load(f)["stats"]
    centre = np.array([stats[n]["mean"] for n in PRED_NAMES], dtype=np.float32)
    scale = np.array([stats[n]["std"] for n in PRED_NAMES], dtype=np.float32)

    model, epoch = load_model(args.checkpoint, device)
    print(f"Exporting {args.classes} for years {args.years}")
    print(f"  checkpoint {args.checkpoint} (epoch {epoch}) -> {args.outdir}")

    if bool(args.out_lat) != bool(args.out_lon):
        raise SystemExit("--out-lat and --out-lon must be given together")
    if args.out_lat:
        lat = np.load(args.out_lat).astype(np.float64)
        lon = np.load(args.out_lon).astype(np.float64)
        out_shape = (len(lat), len(lon))
        print(f"  output grid : {out_shape} lat {lat[0]:.3f}..{lat[-1]:.3f} "
              f"({'ascending' if lat[0] < lat[-1] else 'descending'}) "
              f"lon {lon[0]:.3f}..{lon[-1]:.3f}")
    else:
        lat = np.linspace(90.0, -90.0, ERA5_SHAPE[0])
        lon = np.linspace(-180.0, 180.0 - 360.0 / ERA5_SHAPE[1], ERA5_SHAPE[1])
        out_shape = ERA5_SHAPE

    out_mask = None
    if args.out_mask:
        out_mask = np.load(args.out_mask).astype(bool)
        if out_mask.shape != tuple(out_shape):
            raise SystemExit(f"--out-mask shape {out_mask.shape} != output grid "
                             f"{tuple(out_shape)}")
        print(f"  output mask : {args.out_mask}  "
              f"({100*out_mask.mean():.1f}% of frame kept)")

    # class -> destination directory
    dest = {}
    for cname in args.classes:
        if args.subdirs:
            sub = {"AR": "ARs_WNet_final", "Hurricane": "Hurricanes_WNet_final",
                   "ETC": "ETCs_WNet_final", "MCS": "MCS_WNet_final"}[cname]
            dest[cname] = os.path.join(args.outdir, sub)
        else:
            dest[cname] = args.outdir
        os.makedirs(dest[cname], exist_ok=True)

    all_tags = [f"{y}{m:02d}" for y in args.years for m in range(1, 13)]
    my_tags = all_tags[shard_rank::shard_size]
    print(f"  rank {shard_rank}/{shard_size}: {len(my_tags)} of {len(all_tags)} months")

    n_written = 0
    n_skipped = 0
    for tag in my_tags:
            year = int(tag[:4])
            split = os.path.join(args.cache_dir, f"split_{tag}.npy")
            meta = os.path.join(args.cache_dir, f"split_{tag}.meta.npy")
            if not (os.path.exists(split) and os.path.exists(meta)):
                continue

            todo = [c for c in args.classes
                    if args.overwrite or not os.path.exists(os.path.join(
                        dest[c], f"{args.file_prefix}{tag}_{EXPORTS[c][1]}.nc"))]
            if not todo:
                n_skipped += len(args.classes)
                continue
            arr = np.load(split, mmap_mode="r")
            times = np.load(meta, allow_pickle=True)
            nT = arr.shape[0]

            smooth = args.upsample == "bilinear"
            native = args.upsample == "native"
            H, W = arr.shape[2:]
            if native and (H, W) != tuple(out_shape):
                raise SystemExit(f"--upsample native needs the cache on the output grid: {(H, W)} vs {tuple(out_shape)}")
            ph, pw = (-H) % 8, (-W) % 8            # the U-Net pools 3x
            preds = np.zeros((nT, 4, H, W), dtype=np.float32 if smooth else bool)
            for s in range(0, nT, args.batch_size):
                sel = list(range(s, min(s + args.batch_size, nT)))
                blk = np.asarray(arr[sel, :N_PRED], dtype=np.float32)
                x = (blk - centre[None, :, None, None]) / scale[None, :, None, None]
                xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
                if ph or pw:
                    xt = F.pad(xt, (0, pw, 0, ph), mode="replicate")
                _, p = model.encode(xt)
                p = p[:, :, :H, :W]
                preds[sel] = (p if smooth else p > args.prob_threshold).cpu().numpy()

            for cname in todo:
                head_i, suffix, varname = EXPORTS[cname]
                if native:
                    big = preds[:, head_i].astype(np.float32)
                elif smooth:
                    big = upsample_bilinear_prob(np.ascontiguousarray(preds[:, head_i]),
                                                 args.prob_threshold, out_shape, device)
                else:
                    big = upsample_nn(preds[:, head_i], out_shape)
                if out_mask is not None:
                    big *= out_mask[None].astype(np.float32)
                ds = xr.Dataset(
                    {varname: (("time", "latitude", "longitude"), big)},
                    coords={"time": times[:nT], "latitude": lat, "longitude": lon},
                )
                ds[varname].attrs = {
                    "long_name": f"{cname} detection (WNet Phase 2)",
                    "source_grid": ("native 0.25 deg model grid (721x1440), no resampling" if native else
                                    "192x384 model grid, probability bilinear-interpolated "
                                    "then thresholded" if smooth else
                                    "192x384 model grid, nearest-neighbour resampled"),
                    "prob_threshold": args.prob_threshold,
                    # NetCDF attributes cannot be Python bools — netCDF4 rejects
                    # dtype b1 with "illegal data type for attribute". Store the
                    # mask PATH when masked (more useful than a flag anyway).
                    "domain_mask": (os.path.basename(args.out_mask)
                                    if out_mask is not None else "none"),
                    "checkpoint": os.path.join(os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint))),
                                               os.path.basename(args.checkpoint)),
                }
                out = os.path.join(dest[cname], f"{args.file_prefix}{tag}_{suffix}.nc")
                # dtype and structure match the old outputs exactly; only zlib is
                # added. The originals were uncompressed float32 at 515 MB each
                # (558 GB for AR alone) — compression is transparent to xarray and
                # netCDF4, so the trackers and any comparison code are unaffected.
                ds.to_netcdf(out, encoding={varname: {
                    "zlib": True, "complevel": 4, "dtype": "float32"}})
                ds.close()
                del big, ds
                n_written += 1
            print(f"  [{shard_rank}] {tag}: {nT} steps -> {len(todo)} files", flush=True)

    print(f"\nrank {shard_rank}: wrote {n_written}, skipped {n_skipped} -> {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
