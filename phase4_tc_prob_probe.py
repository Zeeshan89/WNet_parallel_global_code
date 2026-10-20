#!/usr/bin/env python3
"""
phase4_tc_prob_probe.py — how confident is the TC head where it misses?

The exported detection NetCDFs are BINARY: phase4_export_detections.py applies
`p > 0.5` before writing, so nothing downstream can tell "the network saw nothing"
apart from "the network saw it at 0.45".  That distinction decides whether the
recall loss is fixable by a threshold (free) or needs retraining (expensive), so
this script goes back to the checkpoint and reads the PROBABILITY at every
IBTrACS storm position.

For each storm position (>= --wind-kt) it records the maximum TC probability
within --radius-km on the model's own 192x384 grid, then reports the distribution
split by whether the storm was hit or missed by the tracker, and what fraction of
missed-storm positions would light up at a range of thresholds.

Runs on CPU; one month is ~120 timesteps of a 192x384 UNet.

Usage
  phase4_tc_prob_probe.py --years 2003 --months 7 8 9 10
"""
import os
import sys
import json
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_tc_miss_taxonomy import (      # noqa: E402
    ibtracs_storms, parse_hurricane_txt, match_tracks, TRACK_DIR, LAT_LIM,
)

CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
         "Storms_precipitation_notebooks/tensor_cache_v5_ibtracs")
CKPT = "ARs_Hurricanes_ETCs_MCS/Phase2/E_ibtracs200/best.pth"
STATS = "predictor_stats.json"
HURR_HEAD = 1
H, W = 192, 384


def load_model(ckpt_path):
    from WNet_model import WNetMultiLabel
    from dataset import ARDatasetImproved as DS
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = WNetMultiLabel(in_channels=DS.N_PREDICTORS, num_classes=4,
                       discretize=ckpt.get("discretize", "ste"))
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v)
                            for k, v in state.items())
    m.load_state_dict(state)
    m.eval()
    return m, ckpt.get("epoch", "?")


@torch.no_grad()
def month_probs(model, centre, scale, tag, n_pred):
    """(T, 192, 384) TC probability + the month's timestamps."""
    f = os.path.join(CACHE, f"split_{tag}.npy")
    mf = os.path.join(CACHE, f"split_{tag}.meta.npy")
    if not (os.path.exists(f) and os.path.exists(mf)):
        return None, None
    arr = np.load(f, mmap_mode="r")
    times = np.load(mf, allow_pickle=True)
    out = np.zeros((arr.shape[0], H, W), dtype=np.float32)
    for s in range(0, arr.shape[0], 8):
        sel = slice(s, min(s + 8, arr.shape[0]))
        blk = np.asarray(arr[sel, :n_pred], dtype=np.float32)
        x = (blk - centre[None, :, None, None]) / scale[None, :, None, None]
        _, p = model.encode(torch.from_numpy(np.ascontiguousarray(x)))
        out[sel] = p[:, HURR_HEAD].numpy()
    return out, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2003])
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--track-dir", default=TRACK_DIR)
    ap.add_argument("--json", default="phase4_results/tc_prob_probe.json")
    args = ap.parse_args()

    from dataset import ARDatasetImproved as DS
    with open(STATS) as f:
        stats = json.load(f)["stats"]
    centre = np.array([stats[n]["mean"] for n in DS.PREDICTOR_NAMES],
                      dtype=np.float32)
    scale = np.array([stats[n]["std"] for n in DS.PREDICTOR_NAMES],
                     dtype=np.float32)
    model, epoch = load_model(args.checkpoint)
    print(f"  checkpoint {args.checkpoint} (epoch {epoch})")

    glat = np.linspace(90.0, -90.0, H)
    glon = np.linspace(-180.0, 180.0 - 360.0 / W, W)

    recs = []
    for year in args.years:
        ref = ibtracs_storms(year, wind_kt=args.wind_kt, wind_source="any")
        model_tr = parse_hurricane_txt(os.path.join(
            args.track_dir, "hurricane", f"CONUS_year_{year}",
            f"CONUS_hurricane_tracks_detailed_{year}.txt"))
        ref_pts = [[(t, la, lo) for (t, la, lo, _w) in s["points"]] for s in ref]
        hit = {j for j in match_tracks(model_tr, ref_pts, 300.0, 2, 3)
               if j is not None}

        want = defaultdict(list)
        for k, s in enumerate(ref):
            for (t, la, lo, w) in s["points"]:
                if t.year == year and abs(la) <= LAT_LIM:
                    want[t].append((k, la, lo, w))

        for mo in args.months:
            tag = f"{year:04d}{mo:02d}"
            probs, times = month_probs(model, centre, scale, tag,
                                       DS.N_PREDICTORS)
            if probs is None:
                print(f"    {tag}: no cache, skipped")
                continue
            n = 0
            for i, tt in enumerate(times):
                t = tt.astype("datetime64[s]").astype(object).replace(
                    minute=0, second=0, microsecond=0)
                if t not in want:
                    continue
                pm = probs[i]
                for (k, la, lo, w) in want[t]:
                    dy = (glat - la) * 111.0
                    dlon = (glon - lo + 180.0) % 360.0 - 180.0
                    dx = dlon * 111.0 * np.cos(np.radians(la))
                    jl = np.where(np.abs(dy) <= args.radius_km)[0]
                    jo = np.where(np.abs(dx) <= args.radius_km)[0]
                    if not len(jl) or not len(jo):
                        continue
                    sub = pm[np.ix_(jl, jo)]
                    recs.append({"year": year, "k": k, "sid": ref[k]["sid"],
                                 "basin": ref[k]["basin"], "hit": k in hit,
                                 "wind_kt": float(w),
                                 "pmax": float(sub.max()),
                                 "pmean": float(sub.mean())})
                    n += 1
            print(f"    {tag}: {n} storm positions probed", flush=True)

    if not recs:
        raise SystemExit("nothing probed")
    p = np.array([r["pmax"] for r in recs])
    ish = np.array([r["hit"] for r in recs])
    print(f"\n  {len(recs)} storm positions   "
          f"({ish.sum()} on hit storms, {(~ish).sum()} on missed storms)")
    print(f"\n  max TC probability within {args.radius_km:g} km of the storm centre")
    print(f"  {'group':<10}{'n':>7}{'p10':>8}{'p25':>8}{'median':>8}{'p75':>8}"
          f"{'p90':>8}")
    for lab, sel in (("hit", ish), ("missed", ~ish)):
        q = np.percentile(p[sel], [10, 25, 50, 75, 90]) if sel.any() else [0] * 5
        print(f"  {lab:<10}{sel.sum():>7}" + "".join(f"{v:>8.3f}" for v in q))

    print(f"\n  fraction of storm positions above a probability threshold")
    print(f"  {'thr':>6}{'hit storms':>13}{'missed storms':>15}")
    for thr in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        print(f"  {thr:>6.2f}{np.mean(p[ish] > thr):>13.3f}"
              f"{np.mean(p[~ish] > thr):>15.3f}")

    print(f"\n  missed-storm positions with essentially NO signal (pmax < 0.05): "
          f"{np.mean(p[~ish] < 0.05):.3f}")

    by_basin = defaultdict(list)
    for r in recs:
        if not r["hit"]:
            by_basin[r["basin"]].append(r["pmax"])
    print(f"\n  missed-storm positions by basin: median pmax / frac > 0.5 / "
          f"frac < 0.05")
    for b in sorted(by_basin, key=lambda b: -len(by_basin[b])):
        v = np.array(by_basin[b])
        print(f"    {b:<5}{len(v):>6}   {np.median(v):>6.3f}   "
              f"{np.mean(v > 0.5):>6.3f}   {np.mean(v < 0.05):>6.3f}")

    with open(args.json, "w") as f:
        json.dump(recs, f)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
