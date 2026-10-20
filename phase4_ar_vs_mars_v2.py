#!/usr/bin/env python3
"""
phase4_ar_vs_mars_v2.py — build the AR-versus-MARS evaluation cache for the NEW
(Phase 2) AR detections, using exactly the same conventions as the original
evaluations/ar_mars_eval/build_ar_cache.py so the two are directly comparable.

Matched conventions, taken from the original script:
  * target  = MARS variable inside era5dara_YYYYMM.nc, thresholded at > 0
  * pred    = AR_detection, longitude re-indexed to 0..360 and sorted, > 0.5
  * periods = oop 1980-2005, primary 2006-2024
  * per-pixel tp/fp/fn accumulated as int64 on the native 721x1440 grid

Added for the 3-row figure:
  * pred_count / targ_count, giving the AR frequency maps
  * AR-days per pixel, counting a day when ANY of its four 6-hourly steps is an
    AR, which is what "mean AR days per year" means in the existing figure

Added for the v2 rebuild (2026-09-09):
  * per-PERIOD pixel maps, so cos-lat-weighted metrics can be computed per period
    rather than only for the whole record
  * per-SEASON pixel maps and scalars (DJF/MAM/JJA/SON, months as in
    precip_framework/lib/pfa_common.py)
  * per-MONTH scalar contingency counts, for the monthly time series
  * AR days per pixel per YEAR, for interannual trends
  * event-duration accumulators (within-month 0->1 edges plus the first/last day
    masks needed to stitch edges across month boundaries at merge time)
  * all of the above for the TARGET (MARS) as well as the prediction, so the
    benchmark and the model come off one code path

CDO is not available on Frontier, so the quantities the v1 analysis obtained with
`mergetime -> daymax -> yearsum -> timmean` are computed here instead. They are
algebraically the same: a day counts as an AR day if ANY of its four 6-hourly
steps is flagged, and mean AR days per year is the 45-year total over 45.

Run this once per product; point --pred-dir at the v1 finalized detections to get
a v1 cache built by exactly this code, which is what makes the figure rows
comparable (the old ar_mars_eval cache came from a different vintage).

The legacy keys (pix_*, oop_scalar, primary_scalar, lat, lon, n_days) are still
written unchanged, so merge_and_plot_ar_3row.py keeps working as-is.

Months are sharded round-robin across ranks; each rank writes a partial npz and
merge_and_plot_ar_3row.py combines them.

Usage
  srun -n 32 python3 phase4_ar_vs_mars_v2.py --outdir evaluations/ar_mars_eval_v2
"""

import os
import sys
import argparse

import numpy as np

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PRED_DIR_DEFAULT = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                    "Storms_precipitation_notebooks/"
                    "Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final")
TARGET_VAR = "MARS"
PRED_VAR = "AR_detection"
PERIODS = {"oop": (1980, 2005), "primary": (2006, 2024)}

# same definition as precip_framework/lib/pfa_common.py
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_OF = {m: s for s, months in SEASONS.items() for m in months}

PIX_KEYS = ("tp", "fp", "fn", "pred", "targ", "ardays", "ardays_t")
SEA_KEYS = ("tp", "fp", "fn", "pred", "targ")
YEAR0, YEAR1 = 1980, 2024
NYEAR = YEAR1 - YEAR0 + 1


def period_of(year):
    for name, (y0, y1) in PERIODS.items():
        if y0 <= year <= y1:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=PRED_DIR_DEFAULT)
    ap.add_argument("--outdir", default="evaluations/ar_mars_eval_v2")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    import xarray as xr

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.outdir, exist_ok=True)

    months = [(y, m) for y in range(1980, 2025) for m in range(1, 13)][::args.stride]
    mine = months[rank::world]
    if rank == 0:
        print(f"{len(months)} months over {world} ranks", flush=True)
    print(f"  rank {rank}: {len(mine)} months", flush=True)

    acc = {p: dict(tp=0, fp=0, fn=0, tn=0, n_t=0) for p in PERIODS}
    sacc = {s: dict(tp=0, fp=0, fn=0, tn=0, n_t=0) for s in SEASONS}
    pix_per = None          # per-period pixel maps
    pix_sea = None          # per-season pixel maps
    yr_days = None          # (NYEAR, ny, nx) AR days per year, pred and target
    yr_days_t = None
    rise = None             # within-month 0->1 daily edges, pred and target
    rise_t = None
    edge = {}               # (y, m) -> (first_day_p, last_day_p, first_t, last_t)
    month_rows = []         # (year, month, tp, fp, fn, tn, n_t)
    lat = lon = None
    n_days_total = 0

    for (y, m) in mine:
        per = period_of(y)
        if per is None:
            continue
        sea = SEASON_OF[m]
        tf = os.path.join(DATA_DIR, f"era5dara_{y:04d}{m:02d}.nc")
        pf = os.path.join(args.pred_dir, f"era5dara_{y:04d}{m:02d}_{PRED_VAR}.nc")
        if not (os.path.exists(tf) and os.path.exists(pf)):
            print(f"  rank {rank}: {y}{m:02d} MISSING FILE", flush=True)
            continue

        try:
            td = xr.open_dataset(tf)
            if TARGET_VAR not in td.data_vars:
                td.close()
                print(f"  rank {rank}: {y}{m:02d} no {TARGET_VAR}", flush=True)
                continue
            tgt = td[TARGET_VAR].values > 0
            if lat is None:
                lat = td.latitude.values.astype(np.float32)
                lon = td.longitude.values.astype(np.float32)
            td.close()

            pd_ = xr.open_dataset(pf)
            # same re-indexing the original evaluation used
            da = pd_[PRED_VAR].assign_coords(
                longitude=(pd_[PRED_VAR].longitude % 360)).sortby("longitude")
            prd = da.values > 0.5
            pd_.close()
        except Exception as e:
            print(f"  rank {rank}: {y}{m:02d} failed ({e})", flush=True)
            continue

        n = min(tgt.shape[0], prd.shape[0])
        tgt, prd = tgt[:n], prd[:n]
        if pix_per is None:
            shp = tgt.shape[1:]
            pix_per = {p: {k: np.zeros(shp, np.int64) for k in PIX_KEYS} for p in PERIODS}
            pix_sea = {s: {k: np.zeros(shp, np.int64) for k in SEA_KEYS} for s in SEASONS}
            yr_days = np.zeros((NYEAR,) + shp, np.int16)
            yr_days_t = np.zeros((NYEAR,) + shp, np.int16)
            rise = np.zeros(shp, np.int32)
            rise_t = np.zeros(shp, np.int32)

        tp = np.logical_and(prd, tgt)
        fp = np.logical_and(prd, ~tgt)
        fn = np.logical_and(~prd, tgt)

        tp_m = tp.sum(0, dtype=np.int64)
        fp_m = fp.sum(0, dtype=np.int64)
        fn_m = fn.sum(0, dtype=np.int64)
        pred_m = prd.sum(0, dtype=np.int64)
        targ_m = tgt.sum(0, dtype=np.int64)

        P, S = pix_per[per], pix_sea[sea]
        P["tp"] += tp_m; P["fp"] += fp_m; P["fn"] += fn_m
        P["pred"] += pred_m; P["targ"] += targ_m
        S["tp"] += tp_m; S["fp"] += fp_m; S["fn"] += fn_m
        S["pred"] += pred_m; S["targ"] += targ_m

        # AR days: a day counts if any of its four 6-hourly steps is flagged.
        # This is the python equivalent of CDO daymax, which is unavailable here.
        nd = n // 4
        if nd > 0:
            dly = prd[:nd * 4].reshape(nd, 4, *prd.shape[1:]).any(1)
            dly_t = tgt[:nd * 4].reshape(nd, 4, *tgt.shape[1:]).any(1)
            P["ardays"] += dly.sum(0, dtype=np.int64)
            P["ardays_t"] += dly_t.sum(0, dtype=np.int64)
            yi = y - YEAR0
            yr_days[yi] += dly.sum(0, dtype=np.int16)
            yr_days_t[yi] += dly_t.sum(0, dtype=np.int16)
            # event edges inside this month; the month boundary is stitched at
            # merge time from the stored first/last day masks
            if nd > 1:
                rise += (dly[1:] & ~dly[:-1]).sum(0, dtype=np.int32)
                rise_t += (dly_t[1:] & ~dly_t[:-1]).sum(0, dtype=np.int32)
            edge[(y, m)] = (dly[0].copy(), dly[-1].copy(),
                            dly_t[0].copy(), dly_t[-1].copy())
            n_days_total += nd
            del dly, dly_t

        m_tp, m_fp, m_fn = int(tp_m.sum()), int(fp_m.sum()), int(fn_m.sum())
        m_tn = int(np.logical_and(~prd, ~tgt).sum())
        for a in (acc[per], sacc[sea]):
            a["tp"] += m_tp; a["fp"] += m_fp; a["fn"] += m_fn
            a["tn"] += m_tn; a["n_t"] += n
        month_rows.append((y, m, m_tp, m_fp, m_fn, m_tn, n))
        del tgt, prd, tp, fp, fn

    if pix_per is None:
        print(f"  rank {rank}: nothing processed", flush=True)
        return 0

    # legacy keys: totals over all periods, exactly as before
    legacy = {k: sum(pix_per[p][k] for p in PERIODS) for k in PIX_KEYS}

    out = os.path.join(args.outdir, f"part_{rank:04d}.npz")
    payload = dict(lat=lat, lon=lon, n_days=n_days_total)
    payload.update({f"pix_{k}": v for k, v in legacy.items()})
    payload.update({f"{p}_scalar": np.array(
        [acc[p]["tp"], acc[p]["fp"], acc[p]["fn"], acc[p]["tn"], acc[p]["n_t"]],
        np.int64) for p in PERIODS})
    # new: per-period pixel maps
    payload.update({f"pix_{p}_{k}": pix_per[p][k] for p in PERIODS for k in PIX_KEYS})
    # new: per-season pixel maps and scalars
    payload.update({f"sea_{s}_{k}": pix_sea[s][k] for s in SEASONS for k in SEA_KEYS})
    payload.update({f"{s}_scalar": np.array(
        [sacc[s]["tp"], sacc[s]["fp"], sacc[s]["fn"], sacc[s]["tn"], sacc[s]["n_t"]],
        np.int64) for s in SEASONS})
    # new: per-month scalars -> monthly time series
    payload["month_rows"] = (np.array(month_rows, np.int64) if month_rows
                             else np.zeros((0, 7), np.int64))
    # new: AR days per year, for interannual trends
    payload["yr_days"] = yr_days
    payload["yr_days_t"] = yr_days_t
    # new: event-duration pieces. rise/rise_t are the within-month 0->1 edges;
    # the boundary edges are added at merge time from these first/last day masks.
    payload["rise"] = rise
    payload["rise_t"] = rise_t
    # The masks are bit-packed: 540 months x 4 masks would otherwise be 2.2 GB
    # of bool to hold in memory at merge time, versus 280 MB packed.
    ekeys = sorted(edge)
    payload["edge_keys"] = (np.array(ekeys, np.int64) if ekeys
                            else np.zeros((0, 2), np.int64))
    npix = int(np.prod(rise.shape))
    for name, idx in (("edge_first", 0), ("edge_last", 1),
                      ("edge_first_t", 2), ("edge_last_t", 3)):
        payload[name] = (
            np.packbits(np.stack([edge[k][idx] for k in ekeys]).reshape(len(ekeys), -1),
                        axis=1) if ekeys else np.zeros((0, (npix + 7) // 8), np.uint8))

    np.savez_compressed(out, **payload)
    print(f"  rank {rank}: wrote {out} ({len(month_rows)} months)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
