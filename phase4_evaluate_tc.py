#!/usr/bin/env python3
"""
phase4_evaluate_tc.py — evaluate TC detection against IBTrACS on the HELD-OUT
test years, alongside TempestExtremes as an independent competitor.

Why this is the test that matters
---------------------------------
Everything up to Phase 3 scored the model against its own heuristic masks. That
can only ever measure how well it reproduces a hand-built rule. IBTrACS is
observational and the model has never seen it, so this is the first number that
says whether the network detects real storms.

Three detectors are scored against the same IBTrACS truth:

  WNet        our model's hurricane head. Never saw IBTrACS, never saw a label
              at inference — only IVT/MSLP/vorticity/wind/BT/IWV.
  TempestExtremes  an established physics-based tracker run on the same ERA5.
              Also does not use IBTrACS, so this is the fair competitor and the
              number that decides whether the model is actually good.
  heuristic   the ch5 training mask. Scored as a REFERENCE, not a competitor.
              Its actual definition (dataset.py:1279-1416) is

                  (CyTRACK MSLP-minimum centres, dilated, AND (vorticity OR wind))
                  OR (dilated IBTrACS positions)

              so its POD is circular — IBTrACS is unioned straight into it and it
              scores near 1.0 against its own source. Its FAR is NOT circular,
              however, and that is the informative half: the CyTRACK term fires
              on far more objects than there are real TCs, which is the flaw the
              model inherits by learning the mask faithfully.

Matching is point-based: a detection counts as a hit if it falls within
--radius-km of an IBTrACS centre at the same 6-hourly timestamp.

  POD = hits / IBTrACS centres            (did we find the real storms?)
  FAR = false alarms / detections         (how much of what we flag is spurious?)
  CSI = hits / (hits + misses + false alarms)

Usage
  python3 phase4_evaluate_tc.py --checkpoint <best.pth>
  python3 phase4_evaluate_tc.py --checkpoint <best.pth> --years 2003 2012
"""

import os

_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_DISABLE_CACHE"] = "0"

import sys
import csv
import json
import glob
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_model import WNetMultiLabel
from dataset import ARDatasetImproved as DS
from train import TEST_YEARS

IBTRACS = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
           "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc")
TE_DIR = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
          "StormType-Data/TCs/TE")

PRED_NAMES = DS.PREDICTOR_NAMES
N_PRED = DS.N_PREDICTORS
HURR_LABEL_IDX = 1          # (AR, Hurricane, ETC, MCS)
HURR_HEAD_IDX = 1
EARTH_R = 6371.0


def grid_coords(nlat=192, nlon=384):
    """Cell centres of the model grid.

    The cache was bilinearly resampled from 721x1440 (lat 90..-90,
    lon -180..179.75), so this is accurate to under half a cell (~0.5 deg,
    ~50 km) — small against the 300 km matching radius.
    """
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(-180.0, 180.0 - 360.0 / nlon, nlon)
    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Arrays broadcast."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def blob_centroids(mask, lat, lon, min_pixels=2):
    """Connected components of a binary map -> (lat, lon) centroids."""
    if not mask.any():
        return []
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() < min_pixels:
            continue
        rows, cols = np.nonzero(sel)
        # Longitude is periodic; average on the circle so a blob straddling the
        # dateline does not produce a centroid in the middle of the Pacific.
        la = lat[rows].mean()
        ang = np.radians(lon[cols])
        lo = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        out.append((la, lo))
    return out


def load_te_year(year):
    """TempestExtremes TC trajectories -> dict[datetime] -> [(lat, lon)]."""
    path = os.path.join(TE_DIR, f"ERA5_trajectories_hur_{year}_6hr_csv_v14.txt")
    if not os.path.exists(path):
        return None
    out = defaultdict(list)
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        for row in rdr:
            if len(row) < 9:
                continue
            try:
                y, mo, d, h = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                lon = float(row[7])
                lat = float(row[8])
            except (ValueError, IndexError):
                continue
            if lon > 180.0:          # TE writes 0..360
                lon -= 360.0
            out[datetime(y, mo, d, h)].append((lat, lon))
    return out


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = WNetMultiLabel(in_channels=N_PRED, num_classes=4,
                           discretize=ckpt.get("discretize", "ste")).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get("epoch", "?")


class Score:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.n_detections = 0

    def pod(self):
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    def far(self):
        return self.false_alarms / self.n_detections if self.n_detections else float("nan")

    def csi(self):
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")


def score_timestep(dets, truth, radius_km, sc):
    """Match detections to truth centres within radius_km (greedy, symmetric)."""
    if not truth and not dets:
        return
    if not truth:
        sc.n_detections += len(dets)
        sc.false_alarms += len(dets)
        return
    if not dets:
        sc.misses += len(truth)
        return

    tl = np.array([[t[0], t[1]] for t in truth], dtype=float)
    dl = np.array([[d[0], d[1]] for d in dets], dtype=float)
    D = haversine(tl[:, 0][:, None], tl[:, 1][:, None],
                  dl[:, 0][None, :], dl[:, 1][None, :])
    near = D <= radius_km

    sc.n_detections += len(dets)
    sc.hits += int(near.any(axis=1).sum())          # truth centres that were found
    sc.misses += int((~near.any(axis=1)).sum())
    sc.false_alarms += int((~near.any(axis=0)).sum())  # detections matching nothing


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--years", nargs="+", type=int, default=list(TEST_YEARS))
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--wind-kt", type=float, default=34.0,
                    help="IBTrACS intensity floor; 34 kt = named storm, 64 = hurricane")
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    # A single operating point cannot say whether the model is a poor detector or
    # merely cut in the wrong place. Sweeping the probability threshold (and the
    # minimum blob size) separates those two explanations: if CSI peaks far above
    # the p=0.5 value, the model is fine and the decision rule was wrong.
    ap.add_argument("--sweep", nargs="+", type=float, default=None,
                    help="probability thresholds to score in one pass")
    ap.add_argument("--min-pixels", type=int, default=2,
                    help="drop detected blobs smaller than this")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", default="phase4_results")
    # DOMAIN RESTRICTION — required for a fair TempestExtremes comparison.
    # The TE trajectory files are NOT global: 2003 spans lat 10..54.5,
    # lon 225..300 (i.e. N Atlantic / E Pacific only). Scoring a regional
    # tracker against global IBTrACS counts every storm outside its box as a
    # miss and makes its POD meaningless. All three detectors must be scored
    # inside the same window.
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    ap.add_argument("--te-domain", action="store_true",
                    help="shorthand for the TE box: lat 5..60, lon -135..-55")
    args = ap.parse_args()

    if args.te_domain:
        args.lat_min, args.lat_max = 5.0, 60.0
        args.lon_min, args.lon_max = -135.0, -55.0

    def in_domain(pts):
        if args.lat_min is None:
            return list(pts)
        return [(la, lo) for (la, lo) in pts
                if args.lat_min <= la <= args.lat_max
                and args.lon_min <= lo <= args.lon_max]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print(" Phase 4 — TC detection vs IBTrACS on HELD-OUT test years")
    print("=" * 78)
    print(f"  test years   : {args.years}")
    print(f"  match radius : {args.radius_km:.0f} km")
    print(f"  IBTrACS floor: {args.wind_kt:.0f} kt")
    print(f"  checkpoint   : {args.checkpoint}")

    with open(args.stats) as f:
        stats = json.load(f)["stats"]
    centre = np.array([stats[n]["mean"] for n in PRED_NAMES], dtype=np.float32)
    scale = np.array([stats[n]["std"] for n in PRED_NAMES], dtype=np.float32)

    model, epoch = load_model(args.checkpoint, device)
    print(f"  ckpt epoch   : {epoch}")

    from ibtracs_loader import load_ibtracs_for_training
    ib = load_ibtracs_for_training(IBTRACS, start_year=min(args.years),
                                   end_year=max(args.years),
                                   wind_threshold_kt=args.wind_kt,
                                   lat_min=-50.0, lat_max=50.0)

    lat, lon = grid_coords()
    sweep = args.sweep or [args.prob_threshold]
    names = [f"WNet@p{t:g}" for t in sweep] + ["TempestExtremes", "heuristic"]
    scores = {k: Score() for k in names}
    n_steps = 0
    n_truth = 0

    for year in args.years:
        te = load_te_year(year)
        if te is None:
            print(f"  ! no TempestExtremes file for {year}")
            te = {}

        months = sorted(glob.glob(os.path.join(args.cache_dir, f"split_{year}*.npy")))
        months = [m for m in months if not m.endswith(".meta.npy")]
        if not months:
            print(f"  ! no split cache for {year}")
            continue

        for mpath in months:
            meta = mpath.replace(".npy", ".meta.npy")
            if not os.path.exists(meta):
                continue
            times = np.load(meta, allow_pickle=True)
            arr = np.load(mpath, mmap_mode="r")

            for start in range(0, arr.shape[0], args.batch_size):
                sel = list(range(start, min(start + args.batch_size, arr.shape[0])))
                block = np.asarray(arr[sel], dtype=np.float32)
                x = (block[:, :N_PRED] - centre[None, :, None, None]) / scale[None, :, None, None]
                xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
                _, probs = model.encode(xt)
                ph = probs[:, HURR_HEAD_IDX].cpu().numpy()
                preds = {t: ph > t for t in sweep}
                heur = block[:, N_PRED + HURR_LABEL_IDX] > 0.5

                for k, t in enumerate(sel):
                    ts = times[t]
                    try:
                        dt = ts.astype("datetime64[s]").astype(datetime)
                    except Exception:
                        dt = ts
                    if not isinstance(dt, datetime):
                        continue
                    dt = dt.replace(minute=0, second=0, microsecond=0)

                    truth = in_domain([(a, b) for (a, b, *_rest) in ib.get(dt, [])])
                    n_truth += len(truth)
                    n_steps += 1

                    for t in sweep:
                        score_timestep(in_domain(blob_centroids(
                            preds[t][k], lat, lon, args.min_pixels)), truth,
                            args.radius_km, scores[f"WNet@p{t:g}"])
                    score_timestep(in_domain(blob_centroids(
                        heur[k], lat, lon, args.min_pixels)), truth,
                        args.radius_km, scores["heuristic"])
                    score_timestep(in_domain(te.get(dt, [])), truth,
                                   args.radius_km, scores["TempestExtremes"])

        print(f"  {year}: done ({n_steps} timesteps cumulative)")

    print()
    print("=" * 78)
    print(f" RESULTS — {n_steps} timesteps, {n_truth} IBTrACS storm positions")
    print("=" * 78)
    print()
    print(f"  {'detector':<18} {'POD':>8} {'FAR':>8} {'CSI':>8} {'hits':>8} "
          f"{'miss':>8} {'false':>8}")
    print("  " + "-" * 70)
    out = {}
    for name in names:
        s = scores[name]
        out[name] = {"pod": s.pod(), "far": s.far(), "csi": s.csi(),
                     "hits": s.hits, "misses": s.misses,
                     "false_alarms": s.false_alarms,
                     "n_detections": s.n_detections}
        print(f"  {name:<18} {s.pod():>8.4f} {s.far():>8.4f} {s.csi():>8.4f} "
              f"{s.hits:>8d} {s.misses:>8d} {s.false_alarms:>8d}")
    print()
    print("  NOTE: the heuristic is NOT a competitor. Its mask is built from")
    print("  dilated IBTrACS centres, so it is scored against its own source and")
    print("  its POD is circular. TempestExtremes is the fair comparison.")

    with open(os.path.join(args.outdir, "phase4_tc.json"), "w") as f:
        json.dump({"years": args.years, "radius_km": args.radius_km,
                   "wind_kt": args.wind_kt, "n_timesteps": n_steps,
                   "n_ibtracs_positions": n_truth,
                   "checkpoint": args.checkpoint, "results": out}, f,
                  indent=2, default=float)
    print(f"\n  wrote {os.path.join(args.outdir, 'phase4_tc.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
