#!/usr/bin/env python
"""Object-overlap AR detection metrics (ARTMIP-style) — WNet vs MARS.

Per timestep: label connected AR objects (longitude-wrap aware) in the MARS
target and in the WNet prediction, then:
  POD = fraction of MARS objects covered by >=FRAC of their area by ANY WNet AR pixel
  FAR = fraction of WNet objects NOT covered by >=FRAC of their area by ANY MARS AR pixel
Reported at several overlap fractions. Counts give an object-count bias too.

This is the lenient, physically meaningful complement to strict pixel IoU —
forgives filament width/displacement that pixel IoU punishes.

Run with the my_env python:
  python build_ar_object_metrics.py [MONTH_STRIDE]
"""
import os, sys, json
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PRED_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
            "Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/"
            "ETCs_enchanced_test_etc_21")
TARGET_VAR = "MARS"
PRED_VAR = "AR_detection"
PERIODS = {"oop": (1980, 2005), "primary": (2006, 2024)}
FRACS = [0.10, 0.25, 0.50]      # overlap fractions to report
MINPX = 10                      # drop objects smaller than this (noise specks)
HERE = os.path.dirname(os.path.abspath(__file__))
MONTH_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def period_of(year):
    for name, (y0, y1) in PERIODS.items():
        if y0 <= year <= y1:
            return name
    return None


def load_target_month(year, month):
    f = os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc")
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    arr = (ds[TARGET_VAR].values > 0) if TARGET_VAR in ds.data_vars else None
    ds.close()
    return arr


def load_pred_month(year, month):
    f = os.path.join(PRED_DIR, f"era5dara_{year:04d}{month:02d}_{PRED_VAR}.nc")
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    da = ds[PRED_VAR].assign_coords(longitude=(ds[PRED_VAR].longitude % 360)).sortby("longitude")
    arr = (da.values > 0.5)
    ds.close()
    return arr


def label_wrap(mask):
    """4-connected labeling with longitude (last-axis) wrap merging."""
    lab, n = ndi.label(mask)
    if n <= 1:
        return lab, n
    left, right = lab[:, 0], lab[:, -1]
    rows = np.where((left > 0) & (right > 0))[0]
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for r in rows:
        a, b = find(int(left[r])), find(int(right[r]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    if not len(rows):
        return lab, n
    remap = np.array([find(i) for i in range(n + 1)])
    lab = remap[lab]
    # vectorized consecutive relabel (0 -> 0)
    uniq = np.unique(lab)
    comp = np.zeros(int(uniq.max()) + 1, dtype=np.int64)
    comp[uniq] = np.arange(len(uniq))
    out = comp[lab]
    n_out = len(uniq) - (1 if uniq[0] == 0 else 0)
    return out, n_out


def object_coverage(lab, n, other_mask):
    """Return per-object (area, covered_fraction) for labels 1..n."""
    if n == 0:
        return np.array([]), np.array([])
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n + 1)[1:].astype(float)
    cov = np.bincount(flat, weights=other_mask.ravel().astype(float), minlength=n + 1)[1:]
    frac = np.divide(cov, area, out=np.zeros_like(cov), where=area > 0)
    return area, frac


def main():
    acc = {p: dict(n_obs=0, n_pred=0,
                   obs_hit={f: 0 for f in FRACS},
                   pred_match={f: 0 for f in FRACS}) for p in PERIODS}
    monthly = []
    months = [(y, m) for y in range(1980, 2025) for m in range(1, 13)][::MONTH_STRIDE]
    print(f"stride={MONTH_STRIDE}, {len(months)} months, fracs={FRACS}, minpx={MINPX}", flush=True)

    for i, (y, m) in enumerate(months):
        p = period_of(y)
        if p is None:
            continue
        tgt = load_target_month(y, m); prd = load_pred_month(y, m)
        if tgt is None or prd is None or tgt.shape != prd.shape:
            print(f"  SKIP {y}-{m:02d}", flush=True); continue
        mn = dict(year=y, month=m, period=p, n_obs=0, n_pred=0)
        for f in FRACS:
            mn[f"obs_hit_{f}"] = 0; mn[f"pred_match_{f}"] = 0
        for t in range(tgt.shape[0]):
            om, pm = tgt[t], prd[t]
            olab, on = label_wrap(om)
            plab, pn = label_wrap(pm)
            oarea, ofrac = object_coverage(olab, on, pm)   # obs covered by pred
            parea, pfrac = object_coverage(plab, pn, om)   # pred covered by obs
            okeep = oarea >= MINPX
            pkeep = parea >= MINPX
            n_obs = int(okeep.sum()); n_pred = int(pkeep.sum())
            acc[p]["n_obs"] += n_obs; acc[p]["n_pred"] += n_pred
            mn["n_obs"] += n_obs; mn["n_pred"] += n_pred
            for f in FRACS:
                oh = int(((ofrac >= f) & okeep).sum())
                pmt = int(((pfrac >= f) & pkeep).sum())
                acc[p]["obs_hit"][f] += oh
                acc[p]["pred_match"][f] += pmt
                mn[f"obs_hit_{f}"] += oh; mn[f"pred_match_{f}"] += pmt
        monthly.append(mn)
        if i % 12 == 0:
            f = 0.25
            pod = mn[f"obs_hit_{f}"] / max(mn["n_obs"], 1)
            far = 1 - mn[f"pred_match_{f}"] / max(mn["n_pred"], 1)
            print(f"  {y}-{m:02d} [{p}] objs obs={mn['n_obs']} pred={mn['n_pred']} "
                  f"POD@.25={pod:.3f} FAR@.25={far:.3f}", flush=True)

    # summarize
    summ = {}
    for p in PERIODS:
        a = acc[p]
        d = dict(n_obs=a["n_obs"], n_pred=a["n_pred"],
                 obj_count_bias=a["n_pred"] / max(a["n_obs"], 1))
        for f in FRACS:
            d[f"POD_{f}"] = a["obs_hit"][f] / max(a["n_obs"], 1)
            d[f"FAR_{f}"] = 1 - a["pred_match"][f] / max(a["n_pred"], 1)
        summ[p] = d
    with open(os.path.join(HERE, "ar_object_metrics.json"), "w") as fh:
        json.dump(dict(fracs=FRACS, minpx=MINPX, periods=summ), fh, indent=2)
    pd.DataFrame(monthly).to_csv(os.path.join(HERE, "ar_object_monthly.csv"), index=False)
    print("\n=== OBJECT-OVERLAP SUMMARY ===")
    for p in PERIODS:
        d = summ[p]
        print(f"{p:8s} n_obs={d['n_obs']} n_pred={d['n_pred']} "
              f"count_bias={d['obj_count_bias']:.3f}")
        for f in FRACS:
            print(f"         @frac{f}: POD={d[f'POD_{f}']:.4f}  FAR={d[f'FAR_{f}']:.4f}")
    print("WROTE ar_object_metrics.json, ar_object_monthly.csv")


if __name__ == "__main__":
    main()
