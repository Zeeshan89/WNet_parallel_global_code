#!/usr/bin/env python
"""Build the AR-vs-MARS evaluation cache (heavy streaming pass over all months).

Writes, into this script's directory:
  ar_eval_cache.npz   - per-period scalar contingency + per-pixel tp/fp/fn + lat/lon
  ar_monthly.csv      - per-month scalar metrics (for the time series)

The notebook AR_MARS_evaluation.ipynb loads these instead of recomputing.
Run with the my_env python:
  /ccs/home/zaslam/.conda/envs/my_env/bin/python build_ar_cache.py [MONTH_STRIDE]
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PRED_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
            "Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/"
            "ETCs_enchanced_test_etc_21")
TARGET_VAR = "MARS"          # modified tARget-v3 AR shapes = benchmark
PRED_VAR = "AR_detection"    # WNet AR mask
PERIODS = {"oop": (1980, 2005), "primary": (2006, 2024)}
HERE = os.path.dirname(os.path.abspath(__file__))
MONTH_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def period_of(year):
    for name, (y0, y1) in PERIODS.items():
        if y0 <= year <= y1:
            return name
    return None


def load_target(year, month):
    f = os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc")
    if not os.path.exists(f):
        return None, None
    ds = xr.open_dataset(f)
    if TARGET_VAR not in ds.data_vars:
        ds.close(); return None, None
    arr = (ds[TARGET_VAR].values > 0)
    coords = (ds.latitude.values, ds.longitude.values)
    ds.close()
    return arr, coords


def load_pred(year, month):
    f = os.path.join(PRED_DIR, f"era5dara_{year:04d}{month:02d}_{PRED_VAR}.nc")
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    da = ds[PRED_VAR].assign_coords(longitude=(ds[PRED_VAR].longitude % 360)).sortby("longitude")
    arr = (da.values > 0.5)
    ds.close()
    return arr


def main():
    H = W = None
    lat = lon = None
    acc = {p: dict(tp=0, fp=0, fn=0, tn=0, n_t=0) for p in PERIODS}
    pix = {}
    monthly = []
    eps = 1e-12

    months = [(y, m) for y in range(1980, 2025) for m in range(1, 13)][::MONTH_STRIDE]
    print(f"stride={MONTH_STRIDE}, {len(months)} months", flush=True)

    for i, (y, m) in enumerate(months):
        p = period_of(y)
        if p is None:
            continue
        tgt, coords = load_target(y, m)
        prd = load_pred(y, m)
        if tgt is None or prd is None:
            print(f"  SKIP {y}-{m:02d} (missing)", flush=True); continue
        if tgt.shape != prd.shape:
            print(f"  SHAPE MISMATCH {y}-{m:02d}: {tgt.shape} vs {prd.shape}", flush=True); continue
        if H is None:
            lat, lon = coords
            H, W = tgt.shape[1], tgt.shape[2]
            for pp in PERIODS:
                pix[pp] = dict(tp=np.zeros((H, W), np.int64),
                               fp=np.zeros((H, W), np.int64),
                               fn=np.zeros((H, W), np.int64))
        tp_pix = np.logical_and(prd, tgt).sum(axis=0)
        fp_pix = np.logical_and(prd, ~tgt).sum(axis=0)
        fn_pix = np.logical_and(~prd, tgt).sum(axis=0)
        nt = tgt.shape[0]
        tp = int(tp_pix.sum()); fp = int(fp_pix.sum()); fn = int(fn_pix.sum())
        tn = nt * H * W - tp - fp - fn
        acc[p]["tp"] += tp; acc[p]["fp"] += fp; acc[p]["fn"] += fn
        acc[p]["tn"] += tn; acc[p]["n_t"] += nt
        pix[p]["tp"] += tp_pix; pix[p]["fp"] += fp_pix; pix[p]["fn"] += fn_pix
        monthly.append(dict(
            year=y, month=m, period=p, n_t=nt, tp=tp, fp=fp, fn=fn,
            IoU=tp / (tp + fp + fn + eps),
            F1=2 * tp / (2 * tp + fp + fn + eps),
            Precision=tp / (tp + fp + eps),
            POD=tp / (tp + fn + eps),
            FAR=fp / (tp + fp + eps),
            BIAS=(tp + fp) / (tp + fn + eps),
        ))
        if i % 12 == 0:
            print(f"  {y}-{m:02d} [{p}] IoU={monthly[-1]['IoU']:.3f} "
                  f"POD={monthly[-1]['POD']:.3f} FAR={monthly[-1]['FAR']:.3f}", flush=True)

    # save cache
    out = dict(lat=lat, lon=lon,
               periods=np.array(list(PERIODS.keys())),
               stride=np.array(MONTH_STRIDE))
    for p in PERIODS:
        a = acc[p]
        out[f"{p}_scalar"] = np.array([a["tp"], a["fp"], a["fn"], a["tn"], a["n_t"]], dtype=np.int64)
        out[f"{p}_pix_tp"] = pix[p]["tp"]
        out[f"{p}_pix_fp"] = pix[p]["fp"]
        out[f"{p}_pix_fn"] = pix[p]["fn"]
    npz = os.path.join(HERE, "ar_eval_cache.npz")
    np.savez_compressed(npz, **out)
    pd.DataFrame(monthly).to_csv(os.path.join(HERE, "ar_monthly.csv"), index=False)
    print(f"\nWROTE {npz}")
    print(f"WROTE {os.path.join(HERE, 'ar_monthly.csv')}")
    for p in PERIODS:
        tp, fp, fn, tn, ntt = out[f"{p}_scalar"]
        print(f"  {p:8s} IoU={tp/(tp+fp+fn+eps):.4f} POD={tp/(tp+fn+eps):.4f} "
              f"FAR={fp/(tp+fp+eps):.4f} BIAS={(tp+fp)/(tp+fn+eps):.4f} n_t={ntt}")


if __name__ == "__main__":
    main()
