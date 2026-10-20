#!/usr/bin/env python3
"""
phase4_compare_old_new.py — old (Trail9) vs new (Phase 2) detections on the
held-out test years, for the two classes that exist as grids in both.

The old production outputs live in Global_ARs_Hurr_ETCs_MCS_finalized/ as
monthly binary NetCDFs at full ERA5 resolution (721x1440). The new model works
on the 192x384 model grid. Both are compared here against the same heuristic
label, on the same grid, over the same timesteps.

What to look for
----------------
Phase 0 established that the OLD pipeline's AR output never used the network at
all — post_processing.py:448-453 overwrites the model's AR map with a plain
`IVT > percentile` threshold — and that its MCS output was the raw model class,
which was itself a pass-through of input channel 13. So the expected result is
that OLD sits almost exactly on top of the heuristic label, while NEW departs
from it. Departure is not automatically an improvement: this script quantifies
the difference and draws the maps, and whether the difference is *better* is
what the independent references (IBTrACS, Feng, TempestExtremes) decide.

Outputs
  phase4_figures/compare_old_new_<class>.png   frequency maps + difference
  phase4_results/compare_old_new.json          IoU / frequency table

Usage
  python3 phase4_compare_old_new.py --checkpoint <best.pth> --years 1983 2003 2012
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
import json
import glob
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_model import WNetMultiLabel
from dataset import ARDatasetImproved as DS
from train import TEST_YEARS

OLD_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
           "Storms_precipitation_notebooks/Global_ARs_Hurr_ETCs_MCS_finalized")

# class -> (old subdir, old filename suffix, old variable, label index, head index)
GRID_CLASSES = OrderedDict([
    ("AR",  ("ARs_WNet_final",  "AR_detection",  "AR_detection",  0, 0)),
    ("MCS", ("MCS_WNet_final",  "MCS_detection", "MCS_detection", 3, 3)),
])

N_PRED = DS.N_PREDICTORS
PRED_NAMES = DS.PREDICTOR_NAMES
TARGET = (192, 384)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"


def regrid(a, target=TARGET):
    """Bilinear-resample (T,H,W) onto the model grid — same op the cache used."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=target, mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = WNetMultiLabel(in_channels=N_PRED, num_classes=4,
                       discretize=ckpt.get("discretize", "ste")).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    m.load_state_dict(state)
    m.eval()
    return m, ckpt.get("epoch", "?")


class Agg:
    """Accumulate per-pixel detection frequency and pairwise overlap counts."""

    def __init__(self, shape):
        self.n = 0
        self.freq = {k: np.zeros(shape, np.float64) for k in ("old", "new", "label")}
        self.inter = {k: 0.0 for k in ("old_label", "new_label", "old_new")}
        self.tot = {k: 0.0 for k in ("old", "new", "label")}

    def update(self, old, new, lab):
        self.n += old.shape[0]
        for k, v in (("old", old), ("new", new), ("label", lab)):
            self.freq[k] += v.sum(axis=0)
            self.tot[k] += float(v.sum())
        self.inter["old_label"] += float(np.logical_and(old, lab).sum())
        self.inter["new_label"] += float(np.logical_and(new, lab).sum())
        self.inter["old_new"] += float(np.logical_and(old, new).sum())

    def iou(self, a, b):
        key = f"{a}_{b}"
        i = self.inter[key]
        u = self.tot[a] + self.tot[b] - i
        return i / u if u > 0 else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--years", nargs="+", type=int, default=list(TEST_YEARS))
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--figdir", default="phase4_figures")
    ap.add_argument("--outdir", default="phase4_results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    import xarray as xr

    with open(args.stats) as f:
        stats = json.load(f)["stats"]
    centre = np.array([stats[n]["mean"] for n in PRED_NAMES], dtype=np.float32)
    scale = np.array([stats[n]["std"] for n in PRED_NAMES], dtype=np.float32)

    model, epoch = load_model(args.checkpoint, device)
    print("=" * 78)
    print(" Phase 4 — old (Trail9) vs new (Phase 2) on held-out test years")
    print("=" * 78)
    print(f"  years      : {args.years}")
    print(f"  checkpoint : {args.checkpoint} (epoch {epoch})")

    aggs = {c: Agg(TARGET) for c in GRID_CLASSES}

    for year in args.years:
        for month in range(1, 13):
            tag = f"{year}{month:02d}"
            split = os.path.join(args.cache_dir, f"split_{tag}.npy")
            if not os.path.exists(split):
                continue
            arr = np.load(split, mmap_mode="r")
            nT = arr.shape[0]

            # model prediction for the whole month
            preds = np.zeros((nT, 4, *TARGET), dtype=bool)
            for s in range(0, nT, args.batch_size):
                sel = list(range(s, min(s + args.batch_size, nT)))
                blk = np.asarray(arr[sel, :N_PRED], dtype=np.float32)
                x = (blk - centre[None, :, None, None]) / scale[None, :, None, None]
                xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
                _, p = model.encode(xt)
                preds[sel] = (p > args.prob_threshold).cpu().numpy()

            labels = np.asarray(arr[:, N_PRED:], dtype=np.float32) > 0.5

            for cname, (sub, suffix, var, lab_i, head_i) in GRID_CLASSES.items():
                oldf = os.path.join(OLD_DIR, sub, f"era5dara_{tag}_{suffix}.nc")
                if not os.path.exists(oldf):
                    continue
                try:
                    d = xr.open_dataset(oldf)
                    old = d[var].values
                    d.close()
                except Exception as e:
                    print(f"  ! {oldf}: {e}")
                    continue
                if old.shape[0] != nT:
                    m = min(old.shape[0], nT)
                    old = old[:m]
                    o_new = preds[:m, head_i]
                    o_lab = labels[:m, lab_i]
                else:
                    o_new = preds[:, head_i]
                    o_lab = labels[:, lab_i]
                # old is full ERA5 resolution -> put it on the model grid
                old_g = regrid(old.astype(np.float32)) > 0.5
                aggs[cname].update(old_g, o_new, o_lab)
        print(f"  {year}: done")

    # ── report ───────────────────────────────────────────────────────────────
    results = {}
    print()
    print(f"  {'class':<6} {'n steps':>8} {'freq old':>10} {'freq new':>10} "
          f"{'freq label':>11} {'IoU old/lab':>12} {'IoU new/lab':>12} {'IoU old/new':>12}")
    print("  " + "-" * 88)
    for cname, ag in aggs.items():
        if ag.n == 0:
            continue
        npix = ag.n * TARGET[0] * TARGET[1]
        r = {"n_timesteps": ag.n,
             "freq_old": ag.tot["old"] / npix,
             "freq_new": ag.tot["new"] / npix,
             "freq_label": ag.tot["label"] / npix,
             "iou_old_label": ag.iou("old", "label"),
             "iou_new_label": ag.iou("new", "label"),
             "iou_old_new": ag.iou("old", "new")}
        results[cname] = r
        print(f"  {cname:<6} {ag.n:>8d} {r['freq_old']:>10.4f} {r['freq_new']:>10.4f} "
              f"{r['freq_label']:>11.4f} {r['iou_old_label']:>12.4f} "
              f"{r['iou_new_label']:>12.4f} {r['iou_old_new']:>12.4f}")

        # ── figure ──────────────────────────────────────────────────────────
        fq = {k: ag.freq[k] / max(ag.n, 1) for k in ag.freq}
        fig, axes = plt.subplots(1, 4, figsize=(19, 3.4), facecolor=SURFACE)
        fig.subplots_adjust(left=0.03, right=0.98, top=0.80, bottom=0.10, wspace=0.13)
        vmax = max(np.percentile(fq["old"], 99.5),
                   np.percentile(fq["new"], 99.5),
                   np.percentile(fq["label"], 99.5)) or 1.0
        ext = [-180, 180, -90, 90]
        for ax, (key, ttl) in zip(axes[:3], [
                ("old", "Old (Trail9 production)"),
                ("new", "New (Phase 2)"),
                ("label", "Heuristic label")]):
            im = ax.imshow(fq[key], cmap="Blues", vmin=0, vmax=vmax,
                           extent=ext, aspect="auto")
            ax.set_title(ttl, color=INK, fontsize=10, loc="left", pad=6)
            plt.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
        # difference: diverging, neutral midpoint at zero
        dmax = np.percentile(np.abs(fq["new"] - fq["old"]), 99.5) or 1.0
        im = axes[3].imshow(fq["new"] - fq["old"], cmap="RdBu_r",
                            vmin=-dmax, vmax=dmax, extent=ext, aspect="auto")
        axes[3].set_title("New − Old", color=INK, fontsize=10, loc="left", pad=6)
        plt.colorbar(im, ax=axes[3], fraction=0.028, pad=0.01)
        for ax in axes:
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-60, -30, 0, 30, 60])
            ax.tick_params(colors=INK_2, labelsize=8)
        fig.suptitle(f"{cname} detection frequency — held-out test years "
                     f"{args.years}   ({ag.n} timesteps)",
                     color=INK, fontsize=12, x=0.03, ha="left", y=0.96)
        out = os.path.join(args.figdir, f"compare_old_new_{cname}.png")
        fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"         -> {out}")

    with open(os.path.join(args.outdir, "compare_old_new.json"), "w") as f:
        json.dump({"years": args.years, "checkpoint": args.checkpoint,
                   "results": results}, f, indent=2, default=float)
    print(f"\n  wrote {os.path.join(args.outdir, 'compare_old_new.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
