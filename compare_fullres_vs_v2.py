#!/usr/bin/env python3
"""
compare_fullres_vs_v2.py — held-out-year pixel skill of the 0.25 deg WNet export against the shipped v2 product,
both scored against the SAME native 0.25 deg labels (tensor_cache_v11_fullres ch 6-9).

  fullres : Global_ARs_Hurr_ETCs_MCS_fullres025[_test]/<class>_WNet_final   (model run on 0.25 deg)
  v2      : Global_ARs_Hurr_ETCs_MCS_finalized_v2/<class>_WNet_final        (0.94 deg model, AR/MCS smooth-
            upsampled, TC/ETC nearest; TC masks are H_final_e300 best.pth, trained on the >=34 kt label, so the
            v9 lifecycle label below favours the new model for TC — read the TC row with that in mind)
Per class: IoU, precision, recall, area ratio (detection / label), and share of mask edge cells lying on the
0.94 deg grid lines (blockiness; ~0.26 for an unrelated 0.25 deg mask).
  python compare_fullres_vs_v2.py --year 2003 [--fullres-dir ...]
"""
import os
import json
import argparse

import numpy as np
import h5py

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
CLS = [("AR", "ARs_WNet_final", "AR_detection", 6), ("Hurricane", "Hurricanes_WNet_final", "Hurricane_detection", 7),
       ("ETC", "ETCs_WNet_final", "ETCs_detection", 8), ("MCS", "MCS_WNet_final", "MCS_detection", 9)]


def edge_on_grid(m):
    """share of horizontal mask edges that sit on a 192x384 cell boundary (0.94 deg lattice)."""
    e = m[..., :, 1:] != m[..., :, :-1]
    if not e.any():
        return float("nan")
    xb = np.floor(np.arange(1, m.shape[-1]) * 384 / 1440) != np.floor(np.arange(0, m.shape[-1] - 1) * 384 / 1440)
    return float(e[..., xb].sum() / e.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2003)
    ap.add_argument("--fullres-dir", default=f"{NB}/Global_ARs_Hurr_ETCs_MCS_fullres025_test")
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    prods = {"fullres": a.fullres_dir, "v2": f"{NB}/Global_ARs_Hurr_ETCs_MCS_finalized_v2"}
    acc = {(p, c[0]): np.zeros(4) for p in prods for c in CLS}      # tp fp fn  edge-sum
    edges = {(p, c[0]): [] for p in prods for c in CLS}
    for m in range(1, 13):
        tag = f"{a.year}{m:02d}"
        lab = np.load(f"{NB}/tensor_cache_v11_fullres/split_{tag}.npy", mmap_mode="r")
        for name, sub, var, ch in CLS:
            y = np.asarray(lab[::a.stride, ch]) > 0.5
            for p, d in prods.items():
                f = f"{d}/{sub}/era5dara_{tag}_{var}.nc"
                if not os.path.exists(f):
                    continue
                with h5py.File(f, "r") as h:
                    x = h[var][::a.stride] > 0.5
                n = min(len(x), len(y))
                xx, yy = x[:n], y[:n]
                acc[(p, name)][:3] += [(xx & yy).sum(), (xx & ~yy).sum(), (~xx & yy).sum()]
                if m in (1, 7):
                    edges[(p, name)].append(edge_on_grid(xx[::4]))
        print(f"  {tag} done", flush=True)
    res = {}
    print(f"\n{a.year} (held-out), vs native 0.25 deg labels; every {a.stride}th 6-hourly step")
    print(f"{'class':<10}{'product':<9}{'IoU':>7}{'P':>7}{'R':>7}{'area':>7}{'edge-on-0.94grid':>18}")
    for name, *_ in CLS:
        for p in prods:
            tp, fp, fn, _ = acc[(p, name)]
            if tp + fp + fn == 0:
                continue
            r = dict(iou=tp / (tp + fp + fn), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                     area_ratio=(tp + fp) / max(tp + fn, 1), edge_on_grid=float(np.nanmean(edges[(p, name)] or [np.nan])))
            res[f"{name}_{p}"] = r
            print(f"{name:<10}{p:<9}{r['iou']:7.3f}{r['precision']:7.3f}{r['recall']:7.3f}{r['area_ratio']:7.2f}{r['edge_on_grid']:18.2f}")
    os.makedirs("phase4_results", exist_ok=True)
    with open(f"phase4_results/compare_fullres_vs_v2_{a.year}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
