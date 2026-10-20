#!/usr/bin/env python
"""Phase-0 de-risk gate for the hourly MCS evaluation.

Question: can we run the trained WNet with the 11 non-MCS input channels (0-10:
AR/Hurricane/ETC) zeroed, keeping only the 3 MCS masks (ch11-13)? The hourly run
has no hourly IVT/MSLP/vorticity/wind, so those channels cannot be built. This
checks — on EXISTING 6-hourly data — whether zeroing them changes the MCS output.

Compares two test.py runs of the SAME month/checkpoint that differ ONLY in
MCS_ONLY (0 = full channels, 1 = ch0-10 zeroed):

    python verify_mcs_only.py <full_dir> <mcsonly_dir> [YYYYMM]

Gate: if the MCS fields agree closely (high correlation, small domain-total
difference), zeroing is safe and the hourly result is trustworthy. If they
diverge, fall back to the algorithmic ch13 masks (no net).
"""
import sys
import glob
import os
import numpy as np
import xarray as xr

if len(sys.argv) < 3:
    sys.exit(__doc__)
FULL_DIR, MCSONLY_DIR = sys.argv[1], sys.argv[2]
STAMP = sys.argv[3] if len(sys.argv) > 3 else "200407"


def _load(d):
    fs = sorted(glob.glob(os.path.join(d, f"*{STAMP}*_MCS_detection.nc")))
    if not fs:
        sys.exit(f"No *{STAMP}*_MCS_detection.nc in {d}")
    ds = xr.open_dataset(fs[0])
    return ds["MCS_detection"]


full = _load(FULL_DIR)
mcs = _load(MCSONLY_DIR)
print(f"\n=== MCS_ONLY de-risk: {STAMP} ===")
print(f"  full   : {FULL_DIR}  ({full.sizes.get('time')} steps)")
print(f"  mcsonly: {MCSONLY_DIR}  ({mcs.sizes.get('time')} steps)")

fb = (full.values > 0.5)
mb = (mcs.values > 0.5)
n = fb.size

# per-timestep-cell agreement
both = (fb & mb).sum()
only_full = (fb & ~mb).sum()
only_mcs = (~fb & mb).sum()
union = (fb | mb).sum()
iou = both / max(union, 1)

# per-cell MCS frequency maps (fraction of steps flagged)
ff = fb.mean(axis=0).ravel()
mf = mb.mean(axis=0).ravel()
mask = (ff > 0) | (mf > 0)
corr = np.corrcoef(ff[mask], mf[mask])[0, 1] if mask.sum() > 2 else float("nan")

print(f"\n  domain MCS pixels: full={int(fb.sum())}  mcsonly={int(mb.sum())}  "
      f"ratio={mb.sum()/max(fb.sum(),1):.3f}")
print(f"  pixel agreement  : IoU={iou:.3f}  both={int(both)}  "
      f"only_full={int(only_full)}  only_mcsonly={int(only_mcs)}")
print(f"  freq-map corr    : r={corr:.4f}")

# ── verdict ──────────────────────────────────────────────────────────────────
ratio = mb.sum() / max(fb.sum(), 1)
ok = (iou >= 0.85) and (corr >= 0.95) and (0.85 <= ratio <= 1.15)
print(f"\n  {'PASS' if ok else 'REVIEW'}: "
      + ("zeroing ch0-10 leaves MCS ~unchanged — hourly zero-fill is validated."
         if ok else
         "MCS shifted materially — inspect before trusting the hourly run; "
         "consider the algorithmic ch13 masks instead."))
print("  (thresholds: IoU>=0.85, freq-corr>=0.95, total ratio 0.85-1.15)\n")
sys.exit(0 if ok else 1)
