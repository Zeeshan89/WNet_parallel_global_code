#!/usr/bin/env python3
"""
phase4_ar_objects_v2.py — ARTMIP-style object-overlap AR metrics, MPI-sharded.

Same definitions as evaluations/ar_mars_eval/build_ar_object_metrics.py, whose
label_wrap and object_coverage are imported rather than re-implemented:

  POD = fraction of MARS objects covered by >= FRAC of their area by ANY
        predicted AR pixel
  FAR = fraction of predicted objects NOT covered by >= FRAC of their area by
        ANY MARS pixel

This is the lenient complement to strict pixel IoU: it forgives the filament
width and displacement errors that pixel overlap punishes hardest. It matters
for the v2 comparison because v2 trades precision for recall relative to v1, and
a pixel score cannot separate "same objects, fatter" from "more objects".

The v1 run took about 68 minutes serially. Sharding months round-robin over 32
ranks brings it to a couple of minutes, matching phase4_ar_vs_mars_v2.py.

Usage
  srun -n 32 python3 phase4_ar_objects_v2.py \
      --pred-dir .../Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final \
      --outdir evaluations/ar_mars_eval_v2
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "evaluations", "ar_mars_eval"))
from build_ar_object_metrics import (  # noqa: E402
    FRACS, MINPX, PERIODS, label_wrap, object_coverage, period_of)

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
TARGET_VAR = "MARS"
PRED_VAR = "AR_detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    import xarray as xr

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.outdir, exist_ok=True)

    months = [(y, m) for y in range(1980, 2025) for m in range(1, 13)][::args.stride]
    mine = months[rank::world]
    if rank == 0:
        print(f"{len(months)} months over {world} ranks, fracs={FRACS}, minpx={MINPX}",
              flush=True)

    acc = {p: dict(n_obs=0, n_pred=0,
                   obs_hit={f: 0 for f in FRACS},
                   pred_match={f: 0 for f in FRACS}) for p in PERIODS}
    monthly = []

    for (y, m) in mine:
        p = period_of(y)
        if p is None:
            continue
        tf = os.path.join(DATA_DIR, f"era5dara_{y:04d}{m:02d}.nc")
        pf = os.path.join(args.pred_dir, f"era5dara_{y:04d}{m:02d}_{PRED_VAR}.nc")
        if not (os.path.exists(tf) and os.path.exists(pf)):
            print(f"  rank {rank}: SKIP {y}-{m:02d} (missing)", flush=True)
            continue
        try:
            td = xr.open_dataset(tf)
            tgt = td[TARGET_VAR].values > 0
            td.close()
            pdd = xr.open_dataset(pf)
            da = pdd[PRED_VAR].assign_coords(
                longitude=(pdd[PRED_VAR].longitude % 360)).sortby("longitude")
            prd = da.values > 0.5
            pdd.close()
        except Exception as e:
            print(f"  rank {rank}: {y}-{m:02d} failed ({e})", flush=True)
            continue
        if tgt.shape != prd.shape:
            print(f"  rank {rank}: SKIP {y}-{m:02d} shape {tgt.shape} vs {prd.shape}",
                  flush=True)
            continue

        mn = dict(year=y, month=m, period=p, n_obs=0, n_pred=0)
        for f in FRACS:
            mn[f"obs_hit_{f}"] = 0
            mn[f"pred_match_{f}"] = 0
        for t in range(tgt.shape[0]):
            om, pm = tgt[t], prd[t]
            olab, on = label_wrap(om)
            plab, pn = label_wrap(pm)
            oarea, ofrac = object_coverage(olab, on, pm)   # obs covered by pred
            parea, pfrac = object_coverage(plab, pn, om)   # pred covered by obs
            okeep, pkeep = oarea >= MINPX, parea >= MINPX
            n_obs, n_pred = int(okeep.sum()), int(pkeep.sum())
            acc[p]["n_obs"] += n_obs
            acc[p]["n_pred"] += n_pred
            mn["n_obs"] += n_obs
            mn["n_pred"] += n_pred
            for f in FRACS:
                oh = int(((ofrac >= f) & okeep).sum())
                pmt = int(((pfrac >= f) & pkeep).sum())
                acc[p]["obs_hit"][f] += oh
                acc[p]["pred_match"][f] += pmt
                mn[f"obs_hit_{f}"] += oh
                mn[f"pred_match_{f}"] += pmt
        monthly.append(mn)

    out = os.path.join(args.outdir, f"objpart_{rank:04d}.json")
    with open(out, "w") as fh:
        json.dump(dict(fracs=FRACS, minpx=MINPX, monthly=monthly,
                       acc={p: {"n_obs": acc[p]["n_obs"], "n_pred": acc[p]["n_pred"],
                                "obs_hit": {str(f): acc[p]["obs_hit"][f] for f in FRACS},
                                "pred_match": {str(f): acc[p]["pred_match"][f]
                                               for f in FRACS}}
                            for p in PERIODS}), fh)
    print(f"  rank {rank}: wrote {out} ({len(monthly)} months)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
