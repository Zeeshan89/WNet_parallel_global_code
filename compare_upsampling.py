#!/usr/bin/env python3
"""
compare_upsampling.py — how different is the smooth-upsampled 0.25 deg product from the shipped
blocky one? Same model, same 192x384 inputs, same 0.5 threshold; only the step from the model grid to
0.25 deg differs:
    blocky   threshold on 192x384, then nearest neighbour to 721x1440 (phase4_export_detections today)
    smooth   bilinear interpolation of the PROBABILITY to 721x1440, then threshold

Reported per class and month:
  pixel      area ratio smooth/blocky, IoU, agreement over all cells, Pearson r of the binary masks
  blobs      8-connected objects per step (longitude periodic): count, share of blocky blobs with a
             one-to-one partner (IoU >= 0.5), matched-blob area ratio, centroid shift (km)
  frequency  per-cell occurrence frequency over the month (the field behind AR-days / MCS-days maps):
             Pearson r, mean |difference| in % of time steps, and r after both are averaged to 1 deg
Writes phase4_results/compare_upsampling.json
"""
import os
import sys
import json
import argparse
from collections import OrderedDict

_rank = int(os.environ.get("SLURM_PROCID", 0))
_mc = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_mc, exist_ok=True)
os.environ.update(MIOPEN_USER_DB_PATH=_mc, MIOPEN_CUSTOM_CACHE_DIR=_mc, MIOPEN_FIND_MODE="2",
                  MIOPEN_DISABLE_CACHE="0")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
CK = os.path.join(HERE, "ARs_Hurricanes_ETCs_MCS", "Phase2", "H_final_e300")
CASES = [  # (label, cache dir, year, month, class, head, checkpoint)
    ("AR 6-hourly Jan", os.path.join(NB, "tensor_cache_v5_ibtracs"), 2003, 1, "AR", 0, "best_ar.pth"),
    ("AR 6-hourly Jul", os.path.join(NB, "tensor_cache_v5_ibtracs"), 2003, 7, "AR", 0, "best_ar.pth"),
    ("MCS 6-hourly Jan", os.path.join(NB, "tensor_cache_v5_ibtracs"), 2003, 1, "MCS", 3, "best.pth"),
    ("MCS 6-hourly Jul", os.path.join(NB, "tensor_cache_v5_ibtracs"), 2003, 7, "MCS", 3, "best.pth"),
    ("MCS hourly Jul", os.path.join(NB, "tensor_cache_hourly"), 2003, 7, "MCS", 3, "best.pth"),
]
LAT = np.linspace(90, -90, 721)
R_KM = 6371.0


def load(ckpt, dev):
    from WNet_model import WNetMultiLabel
    from dataset import ARDatasetImproved as DS
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = WNetMultiLabel(in_channels=DS.N_PREDICTORS, num_classes=4, discretize=c.get("discretize", "ste"))
    st = OrderedDict((k.replace("module.", "", 1), v) for k, v in c.get("model_state_dict", c).items())
    m.load_state_dict(st)
    return m.to(dev).eval()


def nearest(m):
    yi = np.clip(np.floor(np.arange(721) * 192 / 721).astype(int), 0, 191)
    xi = np.clip(np.floor(np.arange(1440) * 384 / 1440).astype(int), 0, 383)
    return m[..., yi[:, None], xi[None, :]]


def label_wrap(b):
    lab, n = ndimage.label(b, structure=np.ones((3, 3), bool))
    if n > 1:
        a, z = lab[:, 0], lab[:, -1]
        pairs = {(int(x), int(y)) for x, y in zip(a, z) if x and y and x != y}
        for x, y in pairs:
            lab[lab == y] = x
        u = np.unique(lab); u = u[u > 0]
        remap = np.zeros(lab.max() + 1, int); remap[u] = np.arange(1, u.size + 1)
        lab = remap[lab]; n = u.size
    return lab, n


def centroid_km(la, lo):
    """great-circle centroid of cell coordinates (degrees) -> (lat, lon)."""
    x = np.cos(np.radians(la)) * np.cos(np.radians(lo)); y = np.cos(np.radians(la)) * np.sin(np.radians(lo))
    z = np.sin(np.radians(la))
    X, Y, Z = x.mean(), y.mean(), z.mean()
    return np.degrees(np.arctan2(Z, np.hypot(X, Y))), np.degrees(np.arctan2(Y, X))


def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(min(1.0, h)))


def _centroids(lab, n, xyz):
    """(n+1, 2) lat/lon centroid of every label, from unit-sphere coordinates."""
    f = lab.ravel()
    c = np.stack([np.bincount(f, weights=v, minlength=n + 1) for v in xyz], 1)
    return np.degrees(np.arctan2(c[:, 2], np.hypot(c[:, 0], c[:, 1]))), np.degrees(np.arctan2(c[:, 1], c[:, 0]))


def blobs(Bk, Sm, lo_grid, la_grid, min_cells=16):
    n_b = n_s = matched = 0
    ratios, shifts = [], []
    rl, ro = np.radians(la_grid).ravel(), np.radians(lo_grid).ravel()
    xyz = (np.cos(rl) * np.cos(ro), np.cos(rl) * np.sin(ro), np.sin(rl))
    for t in range(Bk.shape[0]):
        lb, nb = label_wrap(Bk[t]); ls, ns = label_wrap(Sm[t])
        sz_b = np.bincount(lb.ravel(), minlength=nb + 1); sz_s = np.bincount(ls.ravel(), minlength=ns + 1)
        keep_b = np.flatnonzero(sz_b >= min_cells); keep_b = keep_b[keep_b > 0]
        n_b += keep_b.size; n_s += int((sz_s[1:] >= min_cells).sum())
        if keep_b.size == 0:
            continue
        both = (lb > 0) & (ls > 0)                                   # contingency table of overlaps
        ct = np.bincount(lb[both].astype(np.int64) * (ns + 1) + ls[both], minlength=(nb + 1) * (ns + 1))
        ct = ct.reshape(nb + 1, ns + 1)
        cbl, cbo = _centroids(lb, nb, xyz); csl, cso = _centroids(ls, ns, xyz)
        for i in keep_b:
            j = int(ct[i].argmax())
            inter = ct[i, j]
            if j == 0 or inter == 0:
                continue
            if inter / (sz_b[i] + sz_s[j] - inter) >= 0.5:
                matched += 1
                ratios.append(sz_s[j] / sz_b[i])
                shifts.append(hav((cbl[i], cbo[i]), (csl[j], cso[j])))
    return dict(blobs_blocky=n_b, blobs_smooth=n_s, matched_share=matched / max(n_b, 1),
                matched_area_ratio_median=float(np.median(ratios)) if ratios else float("nan"),
                centroid_shift_km_median=float(np.median(shifts)) if shifts else float("nan"),
                centroid_shift_km_p90=float(np.percentile(shifts, 90)) if shifts else float("nan"))


@torch.no_grad()
def probs(model, head, arr, mu, sd, dev, step=8):
    out = np.zeros((arr.shape[0], 192, 384), np.float32)
    for s in range(0, arr.shape[0], step):
        x = (np.asarray(arr[s:s + step, :6], np.float32) - mu) / sd
        _, p = model.encode(torch.from_numpy(np.ascontiguousarray(x)).to(dev))
        out[s:s + step] = p[:, head].float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2, help="use every n-th step for pixel/blob stats")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dataset import ARDatasetImproved as DS
    st = json.load(open(os.path.join(HERE, "predictor_stats.json")))["stats"]
    mu = np.array([st[n]["mean"] for n in DS.PREDICTOR_NAMES], np.float32)[None, :, None, None]
    sd = np.array([st[n]["std"] for n in DS.PREDICTOR_NAMES], np.float32)[None, :, None, None]
    lo_grid, la_grid = np.meshgrid(np.linspace(-180, 179.75, 1440), LAT)
    models = {}
    res = OrderedDict()
    for lab, cdir, y, m, cls, head, ck in CASES:
        f = os.path.join(cdir, f"split_{y:04d}{m:02d}.npy")
        if not os.path.exists(f):
            print(f"  {lab}: missing {f}", flush=True); continue
        if ck not in models:
            models[ck] = load(os.path.join(CK, ck), dev)
        arr = np.load(f, mmap_mode="r")
        P = probs(models[ck], head, arr, mu, sd, dev)
        Bk = nearest(P > 0.5)                                            # shipped
        Sm = F.interpolate(torch.from_numpy(P[:, None]), size=(721, 1440), mode="bilinear",
                           align_corners=False)[:, 0].numpy() > 0.5      # smooth
        k = slice(None, None, a.stride)
        b, s = Bk[k], Sm[k]
        inter, uni = (b & s).sum(), (b | s).sum()
        r = dict(steps=int(P.shape[0]),
                 area_ratio=float(s.sum() / max(b.sum(), 1)), iou=float(inter / max(uni, 1)),
                 agreement=float((b == s).mean()),
                 pearson_r=float(np.corrcoef(b.ravel().astype(np.float32), s.ravel().astype(np.float32))[0, 1]))
        r.update(blobs(b, s, lo_grid, la_grid))
        fb, fs = Bk.mean(0), Sm.mean(0)                                  # occurrence frequency per cell
        r["freq_r"] = float(np.corrcoef(fb.ravel(), fs.ravel())[0, 1])
        r["freq_mean_abs_diff_pct"] = float(np.abs(fb - fs).mean() * 100)
        c1 = lambda z: z[:720].reshape(180, 4, 360, 4).mean((1, 3))      # noqa: E731  (1 deg)
        r["freq_r_1deg"] = float(np.corrcoef(c1(fb).ravel(), c1(fs).ravel())[0, 1])
        res[lab] = r
        print(f"{lab:<17} area x{r['area_ratio']:.3f}  IoU {r['iou']:.3f}  agree {r['agreement']:.4f}  r {r['pearson_r']:.3f} | "
              f"blobs {r['blobs_blocky']}->{r['blobs_smooth']} matched {r['matched_share']:.2f} "
              f"area x{r['matched_area_ratio_median']:.2f} shift {r['centroid_shift_km_median']:.0f} km (p90 {r['centroid_shift_km_p90']:.0f}) | "
              f"freq r {r['freq_r']:.3f} (1deg {r['freq_r_1deg']:.3f}) |diff| {r['freq_mean_abs_diff_pct']:.2f}%",
              flush=True)
    os.makedirs(os.path.join(HERE, "phase4_results"), exist_ok=True)
    with open(os.path.join(HERE, "phase4_results", "compare_upsampling.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
