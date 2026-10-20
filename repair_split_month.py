#!/usr/bin/env python3
"""
repair_split_month.py — build a split-cache month when its preprocessed HDF5
cache is partially corrupt.

Motivating case: era5_preprocessed_201309.h5 has three unreadable datasets
(`ivt`, `ivt_grad`, `mslp` — "filter returned failure during read"). The normal
loader then falls back to NetCDF, and that fallback merges a 0..360 ERA5 grid
with the -180..180 auxiliary grids WITHOUT rolling longitudes first, so xarray
takes the coordinate UNION and the arrays come out with longitude length 2160
instead of 1440. The split builder's shape guard catches that and refuses the
month rather than writing a corrupted tensor.

This script sidesteps both problems by assembling the month from the sources
that are known good, per field:

  predictors  IVT        <- sqrt(Qu^2 + Qv^2) from the source ERA5 file (raw; the
                            cached `ivt` is thresholded at 400 and unusable anyway)
              MSLP       <- source ERA5 file if the cache copy is unreadable
              vorticity  \
              wind_speed  }  from the preprocessed HDF5 (these read fine)
              BT, IWV    /
  labels      <- the legacy 14-channel tensor cache, channels [1, 5, 10, 13],
                 so the masks are byte-identical to every other month

Everything is rolled to the cache's -180..180 longitude convention and resized
with the same bilinear interpolation the dataset uses, so the output is
indistinguishable from a normally-built month.

Usage
  python3 repair_split_month.py --month 201309
  python3 repair_split_month.py --month 201309 --dry-run
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARDatasetImproved as DS

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
H5_DIR = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
          "Storms_precipitation_notebooks/Percentile_ARs_detection/"
          "cache/preprocessed_v3")
TARGET_SIZE = (192, 384)

# predictor name -> key in the preprocessed HDF5 (None = only from source ERA5)
H5_KEY = {
    "IVT": None,               # always recomputed raw from Qu/Qv
    "MSLP": "mslp",
    "vorticity": "vorticity",
    "wind_speed": "wind_speed",
    "BT": "BT_ds",
    "IWV": "IWV",
}
# fallback variable name in the source ERA5 NetCDF
SRC_VAR = {"MSLP": "MSLP", "IWV": "IWV"}


def roll_to_pm180(ds):
    """Re-index a 0..360 dataset onto the cache's -180..180 sorted convention."""
    return ds.assign_coords(
        longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")


def resize_stack(data, target_size):
    """Bilinear resize a (C,H,W) stack — identical to ARDatasetImproved._resize_stack."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.shape[1:] == tuple(target_size):
        return data
    try:
        t = torch.from_numpy(data).unsqueeze(0)
    except (TypeError, RuntimeError):
        t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    t = F.interpolate(t, size=tuple(target_size), mode="bilinear",
                      align_corners=False).squeeze(0)
    return t.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYYMM")
    ap.add_argument("--out-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--legacy-dir", default=DS.TENSOR_CACHE_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import h5py
    import xarray as xr

    m = args.month
    h5_path = os.path.join(H5_DIR, f"era5_preprocessed_{m}.h5")
    src_path = os.path.join(DATA_DIR, f"era5dara_{m}.nc")
    legacy_path = os.path.join(args.legacy_dir, f"tensors_{m}.npy")

    for p in (h5_path, src_path, legacy_path):
        if not os.path.exists(p):
            print(f"missing required input: {p}")
            return 1

    print(f"Repairing split month {m}")

    # ── labels from the legacy cache ─────────────────────────────────────────
    legacy = np.load(legacy_path, mmap_mode="r")
    n_t = legacy.shape[0]
    labels = np.asarray(legacy[:, DS.LEGACY_LABEL_CHANNELS], dtype=np.float32)
    print(f"  labels   : {labels.shape} from legacy ch{DS.LEGACY_LABEL_CHANNELS}")

    # ── source ERA5, rolled to the cache convention ──────────────────────────
    src = roll_to_pm180(xr.open_dataset(src_path))
    ivt_raw = np.sqrt(src["Qu"].values ** 2 + src["Qv"].values ** 2).astype(np.float32)
    print(f"  IVT      : {ivt_raw.shape} raw from Qu/Qv "
          f"({100*(ivt_raw == 0).mean():.2f}% zeros)")

    # ── predictors, preferring the HDF5 and falling back per-field ───────────
    fields = {"IVT": ivt_raw}
    with h5py.File(h5_path, "r") as h:
        for name in DS.PREDICTOR_NAMES:
            if name == "IVT":
                continue
            key = H5_KEY[name]
            try:
                fields[name] = np.asarray(h[key][:], dtype=np.float32)
                print(f"  {name:<9}: {fields[name].shape} from HDF5")
            except Exception as e:
                var = SRC_VAR.get(name)
                if var is None or var not in src:
                    print(f"  {name:<9}: HDF5 unreadable ({str(e)[:40]}) and no "
                          f"source fallback — cannot repair")
                    return 1
                fields[name] = src[var].values.astype(np.float32)
                print(f"  {name:<9}: {fields[name].shape} from SOURCE "
                      f"(HDF5 copy corrupt)")
    src.close()

    # ── shape agreement ──────────────────────────────────────────────────────
    ref = fields["IVT"].shape
    for name, arr in fields.items():
        if arr.shape != ref:
            print(f"  shape mismatch: {name} {arr.shape} vs IVT {ref}")
            return 1
    if ref[0] != n_t:
        print(f"  timestep mismatch: predictors {ref[0]} vs legacy labels {n_t}")
        return 1
    print(f"  all predictors agree on {ref}")

    # ── assemble ─────────────────────────────────────────────────────────────
    out = np.zeros((n_t, DS.N_PREDICTORS + DS.N_LABELS, *TARGET_SIZE),
                   dtype=np.float32)
    for i in range(n_t):
        stack = np.stack([fields[n][i] for n in DS.PREDICTOR_NAMES], axis=0)
        out[i, :DS.N_PREDICTORS] = resize_stack(stack, TARGET_SIZE)
        out[i, DS.N_PREDICTORS:] = labels[i]
    print(f"  assembled: {out.shape} ({out.nbytes/1024**2:.0f} MB)")

    for c, name in enumerate(DS.PREDICTOR_NAMES):
        ch = out[:, c]
        print(f"    {name:<11} min={ch.min():11.4g} max={ch.max():11.4g} "
              f"mean={ch.mean():11.4g}")

    if args.dry_run:
        print("  dry run — nothing written")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    npy = os.path.join(args.out_dir, f"split_{m}.npy")
    meta = os.path.join(args.out_dir, f"split_{m}.meta.npy")
    np.save(npy, out)
    meta_src = os.path.join(args.legacy_dir, f"tensors_{m}.meta.npy")
    if os.path.exists(meta_src):
        np.save(meta, np.load(meta_src, allow_pickle=True))
    else:
        with h5py.File(h5_path, "r") as h:
            np.save(meta, np.asarray(h["time"]))
    print(f"  wrote {npy}")
    print(f"  wrote {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
