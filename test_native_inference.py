#!/usr/bin/env python3
"""
test_native_inference.py — can WNet be run at full 0.25 deg resolution at inference time?

WNetMultiLabel is fully convolutional (conv / maxpool / x2 upsample, no fixed-size layer), so it
accepts any grid whose size is a multiple of 16. It was TRAINED on 192x384 (~0.94 deg), so on a finer
grid every learned filter sees a 2-3.75x smaller physical patch. This script measures whether that
scale change is harmless, for one month, before any production export.

For --year/--month:
  1. native predictors (721x1440, lon -180..179.75): MSLP, vorticity850, wind850, BT, IWV from the
     same preprocessed monthly cache precompute_tensors.py uses, IVT recomputed from Qu/Qv.
     CHECK: bilinearly resized to 192x384 (F.interpolate, align_corners=False, as the cache builder)
     they must reproduce the training cache (tensor_cache_v5_ibtracs) channels 0-5.
  2. the unchanged checkpoints of the shipped v2 AR (best_ar.pth) and MCS (best.pth) products at
     192x384 (= the product), 384x768 (~0.47 deg) and 721x1440 (native; padded to 736 rows).
  3. per class and resolution, on every --stride-th time step:
       area ratio vs the 192x384 output (nearest-neighbour to 0.25 deg, as shipped)
       IoU vs the 192x384 output
       AR : IoU vs the NATIVE-resolution L6 AR label (phase4_build_ar_label.build), the label's own
            0.25 deg definition before it was reduced to 192x384 for training
       MCS: share of MCS pixels that are cold cloud (BT <= 241 K), the MCS label's physical signature
  4. a North America panel figure for one time step.
Writes phase4_results/native_inference_<YYYYMM>.json and phase4_figures/native_inference_<YYYYMM>.png
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
CACHE5 = os.path.join(NB, "tensor_cache_v5_ibtracs")
CK = os.path.join(HERE, "ARs_Hurricanes_ETCs_MCS", "Phase2", "H_final_e300")
AR_HEAD, MCS_HEAD = 0, 3
SIZES = OrderedDict([("192x384", (192, 384)), ("384x768", (384, 768)), ("721x1440", (721, 1440))])


def native_predictors(year, month):
    """(T, 6, 721, 1440) raw predictors in PREDICTOR_SPECS order, times (datetime64)."""
    import xarray as xr
    from precompute_tensors import _load_aux_data, CACHE_DIR
    from cache_loader import CacheAwareDataLoader
    import glob
    vfiles = sorted(glob.glob(os.path.join(
        "/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/", "*.nc")))
    aux = _load_aux_data(year, month, vfiles, wind_speed_full=None)
    d = CacheAwareDataLoader(cache_dir=os.path.join(HERE, CACHE_DIR)).load_data(
        year=year, month=month, data_dir=DATA_DIR, use_cache=True, fallback_to_netcdf=True,
        vorticity=aux["vorticity"], wind_speed=aux["wind_speed"], temp_500mb=aux["temp_500mb"],
        BT_ds=aux["BT_ds"], windhshear_ds=aux["windhshear_ds"], ivt_percentile=400)
    src = xr.open_dataset(os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc"))
    src = src.assign_coords(longitude=(((src.longitude + 180) % 360) - 180)).sortby("longitude")
    ivt = np.sqrt(src["Qu"].values ** 2 + src["Qv"].values ** 2).astype(np.float32)
    times = src["time"].values
    src.close()

    def get(*keys):
        for k in keys:
            if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 3:
                return np.asarray(d[k], np.float32)
        raise KeyError(keys)
    fields = [ivt, get("mslp"), get("vorticity"), get("wind_speed"), get("BT_ds", "BT"), get("IWV", "iwv")]
    return np.stack(fields, axis=1), times


def resize(x, hw):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False).numpy()


def load(ckpt, dev):
    from WNet_model import WNetMultiLabel
    from dataset import ARDatasetImproved as DS
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = WNetMultiLabel(in_channels=DS.N_PREDICTORS, num_classes=4, discretize=c.get("discretize", "ste"))
    st = c.get("model_state_dict", c)
    st = OrderedDict((k.replace("module.", "", 1), v) for k, v in st.items())
    m.load_state_dict(st)
    return m.to(dev).eval(), c.get("epoch", "?")


@torch.no_grad()
def infer(model, x, dev):
    """x (T,6,H,W) normalised -> (T,4,H,W) probabilities; pads H,W up to a multiple of 16."""
    T, C, H, W = x.shape
    ph, pw = (-H) % 16, (-W) % 16
    out = np.zeros((T, 4, H, W), np.float32)
    for i in range(T):
        t = torch.from_numpy(x[i:i + 1]).to(dev)
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        _, p = model.encode(t)
        out[i] = p[0, :, :H, :W].float().cpu().numpy()
    return out


def to_native(m, hw_native=(721, 1440)):
    """binary mask on a model grid -> 0.25 deg by nearest neighbour (as the shipped export)."""
    H, W = m.shape[-2:]
    yi = np.clip(np.floor(np.arange(hw_native[0]) * H / hw_native[0]).astype(int), 0, H - 1)
    xi = np.clip(np.floor(np.arange(hw_native[1]) * W / hw_native[1]).astype(int), 0, W - 1)
    return m[..., yi[:, None], xi[None, :]]


def bilinear_prob(p192):
    """(T,192,384) probability -> (T,721,1440) by bilinear interpolation (no thresholding first)."""
    return resize(p192[:, None], (721, 1440))[:, 0]


def shift_ensemble(model, head, Xn, dev, n=4):
    """Shifted-grid ensemble: roll the 0.25 deg input by (dy, dx) in 0..n-1 fine cells, resize to
    192x384 exactly as the cache builder does, infer at the TRAINED scale, bring each probability map
    back to 0.25 deg (nearest, then un-shift) and average the n*n maps. Every 0.25 deg cell gets n*n
    votes from differently aligned coarse grids, so its value is the model's, not a physics field's."""
    T = Xn.shape[0]
    acc = np.zeros((T, 721, 1440), np.float32)
    for dy in range(n):
        for dx in range(n):
            xs = np.roll(Xn, -dx, axis=3)                       # longitude is periodic
            if dy:
                xs = np.concatenate([xs[:, :, dy:], np.repeat(xs[:, :, -1:], dy, axis=2)], axis=2)
            p = infer(model, resize(xs, (192, 384)), dev)[:, head]
            up = to_native(p)                                    # (T,721,1440) float
            up = np.roll(up, dx, axis=2)
            if dy:
                up = np.concatenate([np.repeat(up[:, :1], dy, axis=1), up[:, :-dy]], axis=1)
            acc += up
    return acc / (n * n)


def edge_on_grid(m):
    """share of longitude-direction value changes that sit on a 192x384 cell boundary (1.0 = blocky)."""
    b = np.diff(np.floor(np.arange(1440) * 384 / 1440).astype(int)) != 0
    c = np.diff(m.astype(np.int8), axis=-1) != 0
    tot = c.sum()
    return float(c[..., b].sum() / tot) if tot else float("nan")


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2003)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--figstep", type=int, default=56)
    a = ap.parse_args()
    tag = f"{a.year:04d}{a.month:02d}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dataset import ARDatasetImproved as DS
    stats = json.load(open(os.path.join(HERE, "predictor_stats.json")))["stats"]
    mu = np.array([stats[n]["mean"] for n in DS.PREDICTOR_NAMES], np.float32)[None, :, None, None]
    sd = np.array([stats[n]["std"] for n in DS.PREDICTOR_NAMES], np.float32)[None, :, None, None]

    X, times = native_predictors(a.year, a.month)
    print(f"native predictors {X.shape}", flush=True)
    # ---- check 1: the native fields reproduce the training cache when resized
    cache = np.load(os.path.join(CACHE5, f"split_{tag}.npy"), mmap_mode="r")
    idx = list(range(0, X.shape[0], a.stride))
    chk = {}
    small = resize(X[idx], (192, 384))
    for c, n in enumerate(DS.PREDICTOR_NAMES):
        ref = np.asarray(cache[idx, c], np.float32)
        chk[n] = dict(corr=float(np.corrcoef(small[:, c].ravel(), ref.ravel())[0, 1]),
                      mae=float(np.abs(small[:, c] - ref).mean()), ref_std=float(ref.std()))
        print(f"  check {n:<11} corr {chk[n]['corr']:.6f}  MAE {chk[n]['mae']:.4g} (std {chk[n]['ref_std']:.4g})",
              flush=True)

    ar_m, ep_ar = load(os.path.join(CK, "best_ar.pth"), dev)
    mcs_m, ep_mcs = load(os.path.join(CK, "best.pth"), dev)
    Xs = X[idx]
    Xn = (Xs - mu) / sd
    res = {"checks": chk, "epochs": {"ar": ep_ar, "mcs": ep_mcs}, "steps": len(idx)}
    masks = {}
    for name, hw in SIZES.items():
        xin = Xn if hw == (721, 1440) else resize(Xn, hw)
        pa = infer(ar_m, xin, dev)[:, AR_HEAD] > 0.5
        pm = infer(mcs_m, xin, dev)[:, MCS_HEAD] > 0.5
        if hw != (721, 1440):
            pa, pm = to_native(pa), to_native(pm)
        masks[name] = (pa, pm)
        print(f"  ran {name}", flush=True)
    base_a, base_m = masks["192x384"]

    # AR reference: native L6 label for the same steps
    from phase4_build_ar_label import build
    lab, lat, lon, ltimes = build(a.year, a.month, ["L6"])
    lab = np.roll(lab["L6"], lab["L6"].shape[2] // 2, axis=2)[idx]     # 0..360 -> -180..180
    bt = Xs[:, 4]
    for name, (pa, pm) in masks.items():
        res[name] = dict(
            ar_area_ratio=float(pa.sum() / max(base_a.sum(), 1)), ar_iou_vs_192=iou(pa, base_a),
            ar_iou_vs_native_label=iou(pa, lab),
            mcs_area_ratio=float(pm.sum() / max(base_m.sum(), 1)), mcs_iou_vs_192=iou(pm, base_m),
            mcs_cold_share=float((bt[pm] <= 241.0).mean()) if pm.any() else float("nan"))
        print(f"  {name:<9} AR: area x{res[name]['ar_area_ratio']:.2f}  IoU vs 192 {res[name]['ar_iou_vs_192']:.3f}  "
              f"IoU vs native L6 label {res[name]['ar_iou_vs_native_label']:.3f} | MCS: area x{res[name]['mcs_area_ratio']:.2f}  "
              f"IoU vs 192 {res[name]['mcs_iou_vs_192']:.3f}  cold-cloud share {res[name]['mcs_cold_share']:.3f}",
              flush=True)
    res["native_label_ar_area_ratio_vs_192"] = float(lab.sum() / max(base_a.sum(), 1))

    # ---- inference-level 0.25 deg options that keep the TRAINED scale
    x192 = resize(Xn, (192, 384))
    pa192 = infer(ar_m, x192, dev)[:, AR_HEAD]
    pm192 = infer(mcs_m, x192, dev)[:, MCS_HEAD]
    alt = {"bilinear_prob": (bilinear_prob(pa192[:, None][:, 0]) > 0.5, bilinear_prob(pm192) > 0.5),
           "shift_ensemble16": (shift_ensemble(ar_m, AR_HEAD, Xn, dev) > 0.5,
                                shift_ensemble(mcs_m, MCS_HEAD, Xn, dev) > 0.5)}
    # physics-refined AR product (refine_fullres.py, grow 1) for the same steps, if present
    import netCDF4 as nc
    rf = os.path.join(NB, "Refined_fullres_candidates", "ARs_6h", f"era5dara_{tag}_AR_detection.nc")
    if os.path.exists(rf):
        with nc.Dataset(rf) as f:
            alt["physics_refined_AR"] = (np.asarray(f["AR_detection"][idx]) > 0.5, None)
    alt["192x384 (shipped)"] = (base_a, base_m)
    for name, (pa, pm) in alt.items():
        r = dict(ar_iou_vs_native_label=iou(pa, lab), ar_area_ratio=float(pa.sum() / max(base_a.sum(), 1)),
                 ar_edge_on_coarse_grid=edge_on_grid(pa))
        if pm is not None:
            r.update(mcs_area_ratio=float(pm.sum() / max(base_m.sum(), 1)), mcs_iou_vs_192=iou(pm, base_m),
                     mcs_cold_share=float((bt[pm] <= 241.0).mean()) if pm.any() else float("nan"),
                     mcs_edge_on_coarse_grid=edge_on_grid(pm))
        res["alt:" + name] = r
        print(f"  {name:<20} AR IoU vs native label {r['ar_iou_vs_native_label']:.3f}  area x{r['ar_area_ratio']:.2f}  "
              f"edges on coarse grid {r['ar_edge_on_coarse_grid']:.2f}"
              + (f" | MCS area x{r['mcs_area_ratio']:.2f} IoU vs 192 {r['mcs_iou_vs_192']:.3f} "
                 f"cold {r['mcs_cold_share']:.3f} edges on coarse grid {r['mcs_edge_on_coarse_grid']:.2f}" if pm is not None else ""),
              flush=True)
    masks.update({k: v for k, v in alt.items() if v[1] is not None and k in ("bilinear_prob", "shift_ensemble16")})
    os.makedirs(os.path.join(HERE, "phase4_results"), exist_ok=True)
    with open(os.path.join(HERE, "phase4_results", f"native_inference_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)

    # figure: one step over North America
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = min(len(idx) - 1, a.figstep // a.stride)
    la = np.linspace(90, -90, 721); lo = np.linspace(-180, 179.75, 1440)
    jy = (la >= 10) & (la <= 60); jx = (lo >= -140) & (lo <= -50)
    fig, axs = plt.subplots(2, 4, figsize=(22, 8))
    cols = [("native L6 label / BT<=241", lab[k], bt[k] <= 241)] + [(n, *masks[n]) for n in
                                                                  ("192x384", "bilinear_prob", "shift_ensemble16")]
    for c, (n, A, M) in enumerate(cols):
        for r, (Z, t) in enumerate(((A, "AR"), (M, "MCS"))):
            ax = axs[r, c]
            ax.pcolormesh(lo[jx], la[jy], np.asarray(Z[k] if Z.ndim == 3 else Z)[jy][:, jx], cmap="Blues",
                          vmin=0, vmax=1.3, shading="auto")
            ax.set_title(f"{t}: {n}", fontsize=10); ax.set_aspect("equal")
    fig.suptitle(f"WNet 0.25 deg output options (model always run at its trained 192x384 scale), {str(times[idx[k]])[:13]}")
    os.makedirs(os.path.join(HERE, "phase4_figures"), exist_ok=True)
    p = os.path.join(HERE, "phase4_figures", f"native_inference_{tag}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
