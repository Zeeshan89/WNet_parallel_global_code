#!/usr/bin/env python3
"""
phase4_rebuild_hurricane_label.py — rebuild ONLY the Hurricane label channel of the
split cache under the Tier B decision (wind AND vorticity), and write a patched
copy of the month.

Why only one channel
--------------------
The split cache is (N, 10, 192, 384) = 6 raw predictors + 4 labels
[AR, Hurricane, ETC, MCS]. `require_both` touches the hurricane mask alone, so the
predictors and the other three labels are copied through byte-for-byte. That turns
a ~23 min/month full scipy mask rebuild into a hurricane-only recompute, and it
guarantees the untouched channels cannot drift.

What changed and why
--------------------
dataset.py now passes require_both=True. Tier B screening over 2003+2012 (2924
steps, physics only, no IBTrACS injection) measured the old OR relaxation at
105,604 extra false objects for 2 extra storms out of 3,271:
  POD 0.888 -> 0.887 | FAR 0.986 -> 0.973 | CSI 0.014 -> 0.027
  detections 72.2 -> 36.1 per step, against ~2.3 real storms present.

Fidelity
--------
The mask is rebuilt through the production methods (_detect_hurricane_centres,
_dilate_mslp_centers) at native 721x1440, with the IBTrACS injection applied —
unlike the screening, which excluded it deliberately. The injection is part of the
production label, so it must be present here. The result is bilinearly resized to
192x384, matching how _compute_single_sample resizes the 14-channel stack.

--verify rebuilds with require_both=False and compares against the EXISTING cache
channel. That must match closely before any month is regenerated; if it does not,
the rebuild path differs from production and the output cannot be trusted.

Usage
  phase4_rebuild_hurricane_label.py --year 2003 --month 9 --verify
  phase4_rebuild_hurricane_label.py --year 2003 --month 9 --outdir <cache_v4>
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_label_screen import load_month, _Shim, IBTRACS   # noqa: E402

SPLIT_CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
               "Storms_precipitation_notebooks/tensor_cache_v3_split")
HURR_CH = 7                      # 6 predictors then [AR, Hurricane, ETC, MCS]
TARGET = (192, 384)


def resize(a):
    """Bilinear to 192x384, matching _compute_single_sample's resize."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(a[:, None], dtype=np.float32))
    return F.interpolate(t, size=TARGET, mode="bilinear",
                         align_corners=False)[:, 0].numpy()


def build_hurricane_mask(year, month, require_both, times, mslp, vort, wind,
                         lats, clim):
    """The production `strict` mask at native resolution, incl. IBTrACS injection."""
    from scipy.ndimage import binary_dilation
    from dataset import ARDatasetImproved as DS
    from ibtracs_loader import load_ibtracs_for_training

    nT, H, W = mslp.shape
    shim = _Shim(mslp, vort, wind, lats)
    lat2d = np.broadcast_to(lats[:, None], (H, W))
    lat_mask = ((lat2d >= -45.0) & (lat2d <= 45.0)).astype(np.float32)
    anom = mslp - clim[month - 1][None, :, :]
    lons = np.linspace(-180.0, 180.0 - 360.0 / W, W)

    look = load_ibtracs_for_training(IBTRACS, start_year=year, end_year=year,
                                     wind_threshold_kt=0.0,
                                     lat_min=-50.0, lat_max=50.0)

    dil_px = max(1, int(round(200.0 / (0.25 * 111.0))))
    chk_px = dil_px
    yy, xx = np.ogrid[-dil_px:dil_px + 1, -dil_px:dil_px + 1]
    disc = (yy ** 2 + xx ** 2 <= dil_px ** 2)

    out = np.zeros((nT, H, W), dtype=np.float32)
    for i in range(nT):
        centres = DS._detect_hurricane_centres(
            shim, i, mslp_anomaly=anom[i],
            min_slp_threshold=101500.0, mslp_anomaly_threshold=-500.0,
            great_circle_deg=5.5, dmslp_threshold=150.0,
            vorticity_threshold=1.45e-5, filter_center_km=400.0,
            max_wind_threshold=8.0, radius_msw_km=150.0,
            require_both=require_both)
        dil = DS._dilate_mslp_centers(shim, centres, dilation_km=200.0)
        low = (mslp[i] < 101500.0).astype(np.float32) * dil
        nh = (lat2d >= 0) & (vort[i] >= 1.45e-5)
        sh = (lat2d < 0) & (vort[i] <= -1.45e-5)
        vor_m = (nh | sh).astype(np.float32) * lat_mask
        wind_m = (wind[i] >= 10.0).astype(np.float32) * lat_mask
        strict = (low * lat_mask) * ((vor_m + wind_m) >= 1).astype(np.float32)

        # IBTrACS injection: only where the physics missed, exactly as production.
        key = times[i].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        pts = look.get(key, []) if look else []
        if pts:
            spot = np.zeros((H, W), dtype=np.float32)
            for (slat, slon, _w, _s) in pts:
                r = int(np.argmin(np.abs(lats - slat)))
                c = int(np.argmin(np.abs(lons - slon)))
                r0, r1 = max(0, r - chk_px), min(H, r + chk_px + 1)
                c0, c1 = max(0, c - chk_px), min(W, c + chk_px + 1)
                if strict[r0:r1, c0:c1].sum() > 0:
                    continue
                spot[r, c] = 1.0
            if spot.sum() > 0:
                inj = binary_dilation(spot > 0, structure=disc).astype(np.float32)
                strict = np.maximum(strict, inj * lat_mask)
        out[i] = strict
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="rebuild with require_both=False and compare against the "
                         "existing cache channel; proves the path matches production")
    args = ap.parse_args()

    tag = f"{args.year:04d}{args.month:02d}"
    src_npy = os.path.join(SPLIT_CACHE, f"split_{tag}.npy")
    src_meta = os.path.join(SPLIT_CACHE, f"split_{tag}.meta.npy")
    if not os.path.exists(src_npy):
        raise SystemExit(f"missing {src_npy}")

    if args.outdir and not args.verify:
        dst = os.path.join(args.outdir, f"split_{tag}.npy")
        if os.path.exists(dst):
            print(f"  {tag} already done"); return 0

    mslp, vort, wind, lats, times, clim = load_month(args.year, args.month)
    old = np.load(src_npy, mmap_mode="r")

    if args.verify:
        ref = build_hurricane_mask(args.year, args.month, False, times, mslp,
                                   vort, wind, lats, clim)
        got = resize(ref)
        cur = np.asarray(old[:, HURR_CH], dtype=np.float32)
        n = min(len(got), len(cur))
        d = np.abs(got[:n] - cur[:n])
        agree = float(((got[:n] > 0.5) == (cur[:n] > 0.5)).mean())
        print(f"  VERIFY {tag}: MAE {d.mean():.6f}  max {d.max():.4f}  "
              f"binary agreement {agree:.5f}")
        print(f"    cache positive rate {float((cur[:n]>0.5).mean()):.5f}  "
              f"rebuilt {float((got[:n]>0.5).mean()):.5f}")
        return 0

    new = build_hurricane_mask(args.year, args.month, True, times, mslp, vort,
                               wind, lats, clim)
    new_small = resize(new)
    arr = np.array(old, dtype=np.float32)
    n = min(len(arr), len(new_small))
    before = float((arr[:n, HURR_CH] > 0.5).mean())
    arr[:n, HURR_CH] = new_small[:n]
    after = float((arr[:n, HURR_CH] > 0.5).mean())

    os.makedirs(args.outdir, exist_ok=True)
    dst = os.path.join(args.outdir, f"split_{tag}.npy")
    tmp = f"{dst}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)
    import shutil
    shutil.copyfile(src_meta, os.path.join(args.outdir, f"split_{tag}.meta.npy"))
    print(f"  {tag}: hurricane positive rate {before:.5f} -> {after:.5f} "
          f"({100*(after-before)/max(before,1e-9):+.1f}%)  -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
