#!/usr/bin/env python3
"""
compute_predictor_stats.py — per-channel standardisation stats for the Phase 1
split cache.

The split cache stores RAW physical values (IVT in kg/m/s, MSLP in Pa, vorticity
in 1/s, ...) precisely so the normalisation can be fixed without rebuilding
~200 GB of tensors. This script derives that normalisation and writes it to JSON;
the dataset applies it in __getitem__.

Why not reuse the existing `global_stats` in main.py / precompute_tensors.py:
those numbers are wrong for this purpose. They pair a sensible mean with a far
too small std (IVT mean 170.566, "std" 11.6209 — that is the spread of the
global-mean time series, not the pixel-level spread). Standardising with them
gave the legacy ch0 a range of about -14 to +107 instead of roughly -3 to +3,
i.e. a ~10x inflated dynamic range on the model's only continuous input.

Robust option (default): centre on the median and scale by a normal-consistent
IQR. IVT, vorticity and wind speed are heavy-tailed — a handful of storm pixels
drag the mean and inflate the std, which squashes the ordinary range where most
of the discrimination has to happen. Pass --moments for plain mean/std.

Usage
  python3 compute_predictor_stats.py                      # every 6th month, every 8th step
  python3 compute_predictor_stats.py --month-stride 1 --time-stride 4
  python3 compute_predictor_stats.py --moments
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARDatasetImproved

PREDICTOR_NAMES = ARDatasetImproved.PREDICTOR_NAMES
N_PRED = ARDatasetImproved.N_PREDICTORS
N_LAB = ARDatasetImproved.N_LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=ARDatasetImproved.SPLIT_CACHE_DIR)
    ap.add_argument("--month-stride", type=int, default=6,
                    help="use every Nth month found in the cache")
    ap.add_argument("--time-stride", type=int, default=8,
                    help="use every Nth timestep within a month")
    ap.add_argument("--moments", action="store_true",
                    help="plain mean/std instead of the robust median/IQR")
    ap.add_argument("--out", default="predictor_stats.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "split_*.npy")))
    files = [f for f in files if not f.endswith(".meta.npy")]
    if not files:
        print(f"No split_*.npy found in {args.cache_dir}")
        return 1
    files = files[::args.month_stride]

    print(f"Sampling {len(files)} months (every {args.month_stride}), "
          f"every {args.time_stride} timesteps")

    # Streaming moments, plus a reservoir of values for the robust path.
    n = np.zeros(N_PRED, dtype=np.float64)
    s = np.zeros(N_PRED, dtype=np.float64)
    ss = np.zeros(N_PRED, dtype=np.float64)
    vmin = np.full(N_PRED, np.inf)
    vmax = np.full(N_PRED, -np.inf)
    samples = [[] for _ in range(N_PRED)]

    expected_c = N_PRED + N_LAB
    for k, f in enumerate(files):
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception as e:
            print(f"  ! {os.path.basename(f)} unreadable ({e}) — skipping")
            continue
        if arr.ndim != 4 or arr.shape[1] != expected_c:
            print(f"  ! {os.path.basename(f)} has shape {arr.shape}, "
                  f"expected (N, {expected_c}, H, W) — skipping")
            continue

        idx = range(0, arr.shape[0], args.time_stride)
        block = np.asarray(arr[list(idx), :N_PRED], dtype=np.float64)  # (t, C, H, W)

        finite = np.isfinite(block)
        for c in range(N_PRED):
            v = block[:, c][finite[:, c]]
            if v.size == 0:
                continue
            n[c] += v.size
            s[c] += v.sum()
            ss[c] += (v * v).sum()
            vmin[c] = min(vmin[c], float(v.min()))
            vmax[c] = max(vmax[c], float(v.max()))
            # subsample for quantiles — keep memory bounded
            step = max(1, v.size // 20000)
            samples[c].append(v[::step].astype(np.float32))

        if (k + 1) % 10 == 0 or k == len(files) - 1:
            print(f"  {k+1}/{len(files)} months")

    mean = s / np.maximum(n, 1)
    var = ss / np.maximum(n, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    stats = {}
    print()
    print(f"  {'channel':<12} {'median':>13} {'IQR-sigma':>12} {'mean':>13} "
          f"{'std':>12} {'min':>12} {'max':>12}")
    print("  " + "-" * 92)
    for c, name in enumerate(PREDICTOR_NAMES):
        pool = np.concatenate(samples[c]) if samples[c] else np.array([0.0], dtype=np.float32)
        q25, q50, q75 = np.percentile(pool, [25, 50, 75])
        # 1.349 = IQR of a standard normal, so this matches std for Gaussian data
        iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else float(std[c])

        if args.moments:
            centre, scale = float(mean[c]), float(std[c])
        else:
            centre, scale = float(q50), iqr_sigma
        if not np.isfinite(scale) or scale <= 0:
            scale = float(std[c]) if std[c] > 0 else 1.0

        stats[name] = {
            "mean": centre,          # what __getitem__ subtracts
            "std": scale,            # what __getitem__ divides by
            "method": "moments" if args.moments else "median_iqr",
            "raw_mean": float(mean[c]),
            "raw_std": float(std[c]),
            "median": float(q50),
            "q25": float(q25),
            "q75": float(q75),
            "min": float(vmin[c]),
            "max": float(vmax[c]),
            "n_pixels": float(n[c]),
        }
        print(f"  {name:<12} {q50:>13.5g} {iqr_sigma:>12.5g} {mean[c]:>13.5g} "
              f"{std[c]:>12.5g} {vmin[c]:>12.5g} {vmax[c]:>12.5g}")

    # Report the standardised dynamic range each choice produces — this is the
    # number that was silently broken before.
    print()
    print("  standardised range with the selected centre/scale:")
    for name in PREDICTOR_NAMES:
        st = stats[name]
        lo = (st["min"] - st["mean"]) / st["std"]
        hi = (st["max"] - st["mean"]) / st["std"]
        print(f"    {name:<12} [{lo:>9.2f}, {hi:>9.2f}]")

    with open(args.out, "w") as f:
        json.dump({"predictors": PREDICTOR_NAMES, "stats": stats}, f, indent=2)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
