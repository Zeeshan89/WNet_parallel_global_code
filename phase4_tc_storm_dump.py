#!/usr/bin/env python3
"""
phase4_tc_storm_dump.py — why does WNet miss or cut short a specific storm?

Writes one row per IBTrACS 6-hourly position, for the WHOLE best track at any wind
(depression, tropical-storm and post-landfall stages), for every storm that
enters --box in --years:

  time, lat, lon, any-agency wind, ts (>=34 kt & not ET), et, land
  label_px     pixels of the TC label (cache ch7) within 2.8 deg (what the model was taught)
  pmax/pmean   TC probability within 300 km, re-run from --checkpoint on the model grid
  depth_pa, vort_signed, wind850   the predictor signature (phase4_tc_miss_learnability)
  d_<prod>     km to the nearest point of each track product at the same time

and one row per storm with a miss category:

  A  out of definition: fewer than 2 tropical-storm positions (never in the label/reference)
  HIT  detected under the paper rule; `cov` gives how much of its life is covered
  B  model silent: median pmax < 0.05 over the storm's TS positions
  C  below threshold: the model responds but < 20% of TS positions reach p > 0.5
  D  detected, lost by the tracker: >= 20% of TS positions reach p > 0.5, yet no track

Reuses probe_month (phase4_tc_miss_learnability.py), load_model (phase4_tc_prob_probe.py)
and the IBTrACS loader / matching of hurricane_analysis/tc_paper_analysis.py.

Usage (GPU node, ranks split years):
  srun ... python phase4_tc_storm_dump.py --years 1980 1982 --checkpoint <best_hurricane.pth> \
      --tracks "v2=/.../Hurricanes_WNet_final" --outdir phase4_results/storm_dump
"""
import os

# Per-rank MIOpen cache, set before torch loads (same as phase4_export_detections.py):
# ranks sharing one cache DB fail with miopenStatusInternalError.
_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_DISABLE_CACHE"] = "0"

import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "hurricane_analysis"))
from phase4_tc_miss_learnability import probe_month, GLAT, GLON  # noqa: E402
from phase4_tc_prob_probe import load_model, HURR_HEAD  # noqa: E402
import tc_paper_analysis as tpa  # noqa: E402

CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
         "Storms_precipitation_notebooks/tensor_cache_v5_ibtracs")
LSM = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/era5_land_sea_mask.nc"
USBOX = (-105.0, -65.0, 23.0, 45.0)
R_KM = 300.0


def land_fn():
    from netCDF4 import Dataset
    d = Dataset(LSM)
    m = np.asarray(d["lsm"][0], float)
    d.close()
    return lambda la, lo: m[np.clip(np.rint((90 - la) / 0.25).astype(int), 0, 720),
                            np.rint(np.mod(lo, 360) / 0.25).astype(int) % 1440] >= 0.5


@torch.no_grad()
def probs(model, dev, centre, scale, tag, n_pred):
    f, mf = (os.path.join(CACHE, f"split_{tag}{s}.npy") for s in ("", ".meta"))
    if not os.path.exists(f):
        return None, None
    arr = np.load(f, mmap_mode="r")
    times = np.load(mf, allow_pickle=True)
    out = np.zeros((arr.shape[0], len(GLAT), len(GLON)), np.float32)
    for s in range(0, arr.shape[0], 16):
        sel = slice(s, min(s + 16, arr.shape[0]))
        x = (np.asarray(arr[sel, :n_pred], np.float32) - centre[None, :, None, None]) \
            / scale[None, :, None, None]
        _, p = model.encode(torch.from_numpy(np.ascontiguousarray(x)).to(dev))
        out[sel] = p[:, HURR_HEAD].float().cpu().numpy()
    return out, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--box", nargs=4, type=float, default=list(USBOX), help="lon0 lon1 lat0 lat1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--tracks", nargs="+", default=[], help="label=dir with CONUS_year_Y")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    rank, nr = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    years = a.years[rank::nr]
    os.makedirs(a.outdir, exist_ok=True)
    years = [y for y in years if not os.path.exists(os.path.join(a.outdir, f"storms_{y}.json"))]
    if not years:
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ep = load_model(a.checkpoint)
    model.to(dev)
    from dataset import ARDatasetImproved as DS
    st = json.load(open(a.stats))["stats"]
    centre = np.array([st[n]["mean"] for n in DS.PREDICTOR_NAMES], np.float32)
    scale = np.array([st[n]["std"] for n in DS.PREDICTOR_NAMES], np.float32)
    n_pred = centre.size
    is_land = land_fn()
    box = tuple(a.box)

    # IBTrACS through the paper loader, all positions, with ts/et/name/sid
    tpa.Y0, tpa.Y1 = min(years), max(years)
    ib_all, _ = tpa.load_ibtracs()
    prods = {}
    # Keep every track the tracker wrote (the paper's 48 h rule is applied at scoring
    # time, not here): a 30 h track that stops at landfall is exactly what the map shows.
    tpa.MIN_LIFE_H = 0
    for spec in a.tracks:
        lab, d = spec.split("=", 1)
        tpa.WNET_PAT = os.path.join(d, "CONUS_year_{y}", "CONUS_hurricane_tracks_detailed_{y}.txt")
        prods[lab] = tpa.load_wnet()
    flats = {k: tpa.flatten(v) for k, v in prods.items() if v}

    glat2, glon2 = np.meshgrid(GLAT, GLON, indexing="ij")
    for y in years:
        storms = [s for s in ib_all
                  if tpa.inside(s["lat"], s["lon"], box).any()
                  and y in {tpa.ymd(t)[0] for t in s["t"][tpa.inside(s["lat"], s["lon"], box)]}]
        rows, summ = [], []
        by_month = {}
        for si, s in enumerate(storms):
            for k in np.where(tpa.inside(s["lat"], s["lon"], box))[0]:
                yy, mm = tpa.ymd(s["t"][k])
                if yy == y:
                    by_month.setdefault(mm, []).append((si, k))
        for mm, items in sorted(by_month.items()):
            tag = f"{y:04d}{mm:02d}"
            P, times = probs(model, dev, centre, scale, tag, n_pred)
            if P is None:
                continue
            tidx = {tt.astype("datetime64[h]").astype(np.int64): i for i, tt in enumerate(times)}
            wanted = {}
            for si, k in items:
                dt = datetime.utcfromtimestamp(int(storms[si]["t"][k]) * 3600)
                wanted.setdefault(dt, []).append(((si, int(k)), float(storms[si]["lat"][k]),
                                                  float(storms[si]["lon"][k])))
            pred = {r["key"]: r for r in probe_month(tag, wanted, 2.8, 5.5)}
            for si, k in items:
                s = storms[si]
                la, lo, t = float(s["lat"][k]), float(s["lon"][k]), int(s["t"][k])
                i = tidx.get(t)
                pm = pn = np.nan
                if i is not None:
                    near = tpa.hav(la, lo, glat2, glon2) <= R_KM
                    pm, pn = float(P[i][near].max()), float(P[i][near].mean())
                r = pred.get((si, int(k)), {})
                row = dict(sid=s["sid"], name=s["name"], t=int(t), lat=la, lon=lo,
                           wind=float(s["wind"][k]) if np.isfinite(s["wind"][k]) else None,
                           ts=bool(s["ts"][k]), et=bool(s["et"][k]),
                           land=bool(is_land(np.array(la), np.array(lo))),
                           label_px=r.get("label_px"), pmax=pm, pmean=pn,
                           depth_pa=r.get("depth_pa"), vort_signed=r.get("vort_signed"),
                           wind850=r.get("wind_max"))
                for lab, (T, LA, LO, _) in flats.items():
                    i0, i1 = np.searchsorted(T, t, "left"), np.searchsorted(T, t, "right")
                    row[f"d_{lab}"] = float(tpa.hav(la, lo, LA[i0:i1], LO[i0:i1]).min()) if i1 > i0 else None
                rows.append(row)
        # per-storm category
        for sid in dict.fromkeys(r["sid"] for r in rows):
            R = [r for r in rows if r["sid"] == sid]
            ts = [r for r in R if r["ts"]]
            p_ts = np.array([r["pmax"] for r in ts], float)
            e = dict(year=y, sid=sid, name=R[0]["name"], n=len(R), n_ts=len(ts),
                     n_land=sum(r["land"] for r in R),
                     vmax=max((r["wind"] or 0) for r in R),
                     pmax_med_ts=float(np.nanmedian(p_ts)) if p_ts.size else None,
                     frac_p50_ts=float(np.nanmean(p_ts > 0.5)) if p_ts.size else None,
                     frac_p50_all=float(np.nanmean(np.array([r["pmax"] for r in R], float) > 0.5)),
                     frac_p50_land=(float(np.nanmean(np.array([r["pmax"] for r in R if r["land"]], float) > 0.5))
                                    if any(r["land"] for r in R) else None))
            for lab in flats:
                dd = [r[f"d_{lab}"] for r in ts]
                hits = sum(1 for d in dd if d is not None and d <= R_KM)
                cov = np.mean([1.0 if (r[f"d_{lab}"] is not None and r[f"d_{lab}"] <= R_KM) else 0.0
                               for r in R if not r["et"]]) if R else 0.0
                if len(ts) < 2:
                    cat = "A"
                elif hits >= 2:
                    cat = "HIT"
                elif e["pmax_med_ts"] is not None and e["pmax_med_ts"] < 0.05:
                    cat = "B"
                elif (e["frac_p50_ts"] or 0) < 0.2:
                    cat = "C"
                else:
                    cat = "D"
                e[f"cat_{lab}"], e[f"cov_{lab}"] = cat, float(cov)
            summ.append(e)
        with open(os.path.join(a.outdir, f"storms_{y}.json"), "w") as fh:
            json.dump(dict(checkpoint=a.checkpoint, epoch=ep, box=box, storms=summ, positions=rows),
                      fh, default=float)
        print(f"[rank {rank}] {y}: {len(summ)} storms, {len(rows)} positions", flush=True)


if __name__ == "__main__":
    main()
