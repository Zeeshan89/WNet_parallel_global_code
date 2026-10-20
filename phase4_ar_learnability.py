#!/usr/bin/env python3
"""
phase4_ar_learnability.py — how well does the retrained model reproduce its OWN AR label, and
where does it not? The network sees |IVT| (and five other fields) on a 192 x 384 grid, but the
label was built at 0.25 degrees from IVT direction, object length and local climatology, none
of which the network is given. If the model misses the label somewhere (say over land, or in
one latitude band), that is the place for the post-filter or the extra predictor of the plan's
contingency, and the scores against the other methods there are not to be read as label defects.

Compares the exported AR masks (era5dara_YYYYMM_AR_detection.nc, 0.25 deg, nearest-neighbour
from 192 x 384) with the native label (AR_label_candidates/<Lx>/, the validation years) by
latitude band and by land / ocean (era5_land_sea_mask.nc), area weighted. The same numbers for
the shipped product against the shipped label (L0) give the resolution ceiling both share
(native vs 192 x 384 IoU is about 0.86 for L0).

  srun -n 48 phase4_ar_learnability.py --pred-dir <product>/ARs_WNet_final --outdir <dir>
  phase4_ar_learnability.py --summarize --outdir <dir>        -> learnability.csv / .txt
"""
import os
import sys
import glob
import argparse

import numpy as np

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
LABEL_DIR = os.path.join(NB, "AR_label_candidates")
LSM = os.path.join(NB, "era5_land_sea_mask.nc")
YEARS = (1987, 1998, 2007, 2016)                      # the validation years: native labels exist
BANDS = (("tropics |lat|<15", 0, 15), ("subtropics 15-30", 15, 30), ("midlat 30-60", 30, 60),
         ("polar >60", 60, 90), ("global", 0, 90))
SURFACES = ("ocean", "land", "all")


def read_mask(path, var):
    """(time, lat, lon) bool with latitude 90..-90 and longitude starting at -180."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        a = np.squeeze(ds.variables[var][:]) > 0.5
        lat = ds.variables["latitude"][:].astype(float)
        lon = ds.variables["longitude"][:].astype(float)
    if a.ndim == 2:
        a = a[None]
    if lat[0] < lat[-1]:
        a, lat = a[:, ::-1], lat[::-1]
    sh = int(np.argmin(np.abs(((lon + 180.0) % 360.0) - 180.0)))       # column of longitude -180
    return np.roll(a, -sh, axis=2), lat


def accumulate(pred, lab, land):
    """(surface, quantity, lat): TP, FP, FN, label cells, prediction cells, surface cells."""
    rows = np.zeros((2, 6, pred.shape[1]), np.int64)
    for k, sel in enumerate((~land, land)):
        rows[k] = [(pred & lab & sel).sum((0, 2)), (pred & ~lab & sel).sum((0, 2)),
                   (~pred & lab & sel).sum((0, 2)), (lab & sel).sum((0, 2)),
                   (pred & sel).sum((0, 2)), pred.shape[0] * sel.sum(1)]
    return rows


def table(rows, lat, header):
    """Area-weighted scores by latitude band and surface from summed rows."""
    w = np.cos(np.radians(lat))
    lines = [header, f"{'band':<20}{'surface':<8}{'IoU':>7}{'POD':>7}{'FAR':>7}{'BIAS':>7}{'label%':>8}{'pred%':>8}"]
    csv = ["band,surface,iou,pod,far,bias,label_freq,pred_freq"]
    for name, a0, a1 in BANDS:
        sel = (np.abs(lat) >= a0) & (np.abs(lat) <= a1)
        for k, surf in enumerate(SURFACES):
            r = rows.sum(0) if surf == "all" else rows[k]
            tp, fp, fn, nl, npd, nc = [float((r[i] * w)[sel].sum()) for i in range(6)]
            iou, pod = tp / max(tp + fp + fn, 1.0), tp / max(tp + fn, 1.0)
            far, bias = fp / max(tp + fp, 1.0), npd / max(nl, 1.0)
            fl, fp_ = nl / max(nc, 1.0), npd / max(nc, 1.0)
            lines.append(f"{name:<20}{surf:<8}{iou:>7.3f}{pod:>7.3f}{far:>7.3f}{bias:>7.2f}{100 * fl:>8.3f}{100 * fp_:>8.3f}")
            csv.append(f"{name},{surf},{iou:.4f},{pod:.4f},{far:.4f},{bias:.4f},{fl:.6f},{fp_:.6f}")
    return "\n".join(lines), "\n".join(csv)


def land_mask(lat_ref, nlon):
    m, lat = read_mask(LSM, "lsm")
    assert m.shape[1:] == (lat_ref.size, nlon) and np.allclose(lat, lat_ref, atol=1e-3), "land mask grid differs"
    return m[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir")
    ap.add_argument("--label", default="L6", help="candidate directory under AR_label_candidates")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.summarize:
        return summarize(args)
    if not args.pred_dir:
        ap.error("--pred-dir is needed unless --summarize")
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    months = [f"{y}{m:02d}" for y in YEARS for m in range(1, 13)][rank::world]
    land = None
    for tag in months:
        out = os.path.join(args.outdir, f"part_{tag}.npz")
        if os.path.exists(out):
            continue
        pred, lat = read_mask(os.path.join(args.pred_dir, f"era5dara_{tag}_AR_detection.nc"), "AR_detection")
        lab, lat2 = read_mask(os.path.join(LABEL_DIR, args.label, f"era5dara_{tag}_AR_detection.nc"), "AR_detection")
        assert pred.shape == lab.shape and np.allclose(lat, lat2), tag
        if land is None:
            land = land_mask(lat, pred.shape[2])
        np.savez(out, rows=accumulate(pred, lab, land), lat=lat, n_t=pred.shape[0], label=args.label)
        print(f"  {tag}: label {lab.mean():.4f} pred {pred.mean():.4f} "
              f"IoU {(pred & lab).sum() / max((pred | lab).sum(), 1):.3f}", flush=True)


def summarize(args):
    parts = sorted(glob.glob(os.path.join(args.outdir, "part_*.npz")))
    if not parts:
        raise SystemExit(f"no part files in {args.outdir}")
    rows, n_t, label = None, 0, args.label
    for p in parts:
        z = np.load(p)
        rows = z["rows"].astype(np.float64) if rows is None else rows + z["rows"]
        lat, n_t = z["lat"], n_t + int(z["n_t"])
        label = str(z["label"]) if "label" in z else label
    txt, csv = table(rows, lat, f"model vs its own label ({label}), {len(parts)} months of {YEARS}, "
                                f"{n_t} time steps, area weighted")
    print(txt)
    open(os.path.join(args.outdir, "learnability.txt"), "w").write(txt + "\n")
    open(os.path.join(args.outdir, "learnability.csv"), "w").write(csv + "\n")


if __name__ == "__main__":
    sys.exit(main())
