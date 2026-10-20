#!/usr/bin/env python3
"""
phase1_predictor_sensitivity.py — verification for the Phase 1 split contract.

Phase 0 (phase0_shortcut_diagnostic.py) proved every class was a copy of one of
its own input channels. Phase 1 removes those channels from the model's input
entirely, so the copy test no longer applies — there is nothing left to copy.
This is its successor, and it answers the questions that become askable once the
shortcut is gone:

  T0 leak test    Zero ALL 6 predictors. The model then has no information at
                  all, so every class must collapse to a constant map. If any
                  class still tracks its label, a label is reaching the model
                  and the split is not actually clean.

  T1 sensitivity  Zero one predictor at a time, per class. This is the honest
                  successor to the copy test: it shows which PHYSICAL field each
                  class relies on. AR should lean on IVT, MCS on BT/IWV,
                  Hurricane and ETC on MSLP/vorticity/wind. A class insensitive
                  to every predictor has not learned anything.

  T2 solo         Keep one predictor, zero the other 5. How far can each class
                  get from a single field?

  T3 agreement    IoU/Dice against the heuristic label. Under the split contract
                  this is a real (if mask-referenced) skill number rather than
                  the copy metric it was in Phase 0. Expect it to be LOWER than
                  Phase 0's 0.71-0.99 — that drop is the shortcut going away.

  T4 reconstruction / T4b one-hot / T6 peakedness — as in Phase 0, but now over
                  the 6 physical channels, where reconstruction is a genuine
                  bottleneck objective rather than a mask pass-through.

Usage
  python3 phase1_predictor_sensitivity.py --checkpoint <split-trained.pth>
"""

import os

# MIOpen per-process cache — MUST precede `import torch` on Frontier or every
# Conv2d fails with miopenStatusInternalError (see test.py:1-15).
_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"]     = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"]        = "2"
os.environ["MIOPEN_DISABLE_CACHE"]    = "0"

import sys
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_model import WNetMultiLabel
from dataset import ARDatasetImproved as DS

PRED_NAMES = DS.PREDICTOR_NAMES          # IVT, MSLP, vorticity, wind_speed, BT, IWV
N_PRED = DS.N_PREDICTORS
LABEL_NAMES = DS.LABEL_NAMES             # AR, Hurricane, ETC, MCS
MASK_BIN_THRESH = 0.5


class PairCounter:
    """Pixel-weighted intersection / union counts accumulated over all samples."""

    def __init__(self):
        self.inter = 0.0
        self.a_only = 0.0
        self.b_only = 0.0

    def update(self, a, b):
        both = (a & b).sum().item()
        self.inter += both
        self.a_only += a.sum().item() - both
        self.b_only += b.sum().item() - both

    @property
    def a_total(self):
        return self.inter + self.a_only

    @property
    def b_total(self):
        return self.inter + self.b_only

    def iou(self):
        u = self.inter + self.a_only + self.b_only
        return self.inter / u if u > 0 else float("nan")

    def dice(self):
        d = self.a_total + self.b_total
        return 2.0 * self.inter / d if d > 0 else float("nan")

    def recall_of_a(self):
        return self.inter / self.a_total if self.a_total > 0 else float("nan")


def load_model(ckpt_path, device):
    # Phase 2 model: 4 independent sigmoid heads, no background class.
    ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    disc = ckpt_peek.get("discretize", "ste")
    model = WNetMultiLabel(in_channels=N_PRED, num_classes=len(LABEL_NAMES),
                           discretize=disc).to(device)
    ckpt = ckpt_peek
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get("epoch", "unknown")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--months", nargs="+", default=["201401", "201404", "201407", "201410"])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--outdir", default="phase1_diagnostics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.stats) as f:
        stats = json.load(f)["stats"]
    centre = np.array([stats[n]["mean"] for n in PRED_NAMES], dtype=np.float32)
    scale = np.array([stats[n]["std"] for n in PRED_NAMES], dtype=np.float32)

    print("=" * 78)
    print(" Phase 1 — predictor sensitivity (split contract)")
    print("=" * 78)
    print(f"  device     : {device}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  predictors : {', '.join(PRED_NAMES)}")
    print(f"  months     : {', '.join(args.months)} (stride {args.stride})")

    model, epoch = load_model(args.checkpoint, device)
    print(f"  ckpt epoch : {epoch}")

    configs = OrderedDict()
    configs["baseline"] = []
    configs["ZERO_ALL"] = list(range(N_PRED))                      # T0
    for c, name in enumerate(PRED_NAMES):
        configs[f"LOO_{name}"] = [c]                               # T1
        configs[f"SOLO_{name}"] = [k for k in range(N_PRED) if k != c]  # T2

    pair = {cfg: {n: PairCounter() for n in LABEL_NAMES} for cfg in configs}
    agree = {n: PairCounter() for n in LABEL_NAMES}
    leak = {n: PairCounter() for n in LABEL_NAMES}   # ZERO_ALL prediction vs label

    rec_sse = np.zeros(N_PRED, dtype=np.float64)
    rec_hard_sse = np.zeros(N_PRED, dtype=np.float64)
    rec_sum = np.zeros(N_PRED, dtype=np.float64)
    rec_sqsum = np.zeros(N_PRED, dtype=np.float64)
    rec_n = 0
    peak_maxprob = 0.0
    peak_entropy = 0.0
    peak_n = 0
    n_total = 0

    for month in args.months:
        npy = os.path.join(args.cache_dir, f"split_{month}.npy")
        if not os.path.exists(npy):
            print(f"  ! missing {npy} — skipping")
            continue
        arr = np.load(npy, mmap_mode="r")
        idx = list(range(0, arr.shape[0], args.stride))
        print(f"\n  {month}: {arr.shape[0]} timesteps -> using {len(idx)}")

        for start in range(0, len(idx), args.batch_size):
            sel = idx[start:start + args.batch_size]
            block = np.asarray(arr[sel], dtype=np.float32)
            raw_pred = block[:, :N_PRED]
            labels = block[:, N_PRED:]

            # standardise exactly as the dataset does
            norm = (raw_pred - centre[None, :, None, None]) / scale[None, :, None, None]
            x = torch.from_numpy(np.ascontiguousarray(norm)).to(device)
            y = torch.from_numpy(np.ascontiguousarray(labels)).to(device)
            n_total += x.shape[0]

            tgt_bin = {n: (y[:, i] > MASK_BIN_THRESH) for i, n in enumerate(LABEL_NAMES)}

            probs, rec, _ = model(x)
            base_masks = {n: (probs[:, i] > 0.5) for i, n in enumerate(LABEL_NAMES)}

            for n in LABEL_NAMES:
                agree[n].update(base_masks[n], tgt_bin[n])
                pair["baseline"][n].update(base_masks[n], base_masks[n])

            err = (rec - x).float()
            rec_sse += (err ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_sum += x.float().sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_sqsum += (x.float() ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_n += x.shape[0] * x.shape[2] * x.shape[3]

            hard = (probs > 0.5).float()      # the code the STE already feeds
            rec_h = model.unet2(hard)
            rec_hard_sse += ((rec_h - x).float() ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()

            # Independent sigmoids: "peakedness" is distance from 0.5, and the
            # per-head binary entropy, not a softmax entropy.
            p = probs.float().clamp(1e-6, 1 - 1e-6)
            peak_maxprob += (2 * (p - 0.5).abs()).mean(dim=1).sum().item()
            peak_entropy += (-(p * p.log() + (1 - p) * (1 - p).log())
                             ).mean(dim=1).sum().item()
            peak_n += x.shape[0] * x.shape[2] * x.shape[3]
            del probs, rec, err, p, hard, rec_h

            for cfg, zero_channels in configs.items():
                if cfg == "baseline":
                    continue
                xa = x.clone()
                for c in zero_channels:
                    xa[:, c] = 0.0
                _, probs_a = model.encode(xa)
                for i, n in enumerate(LABEL_NAMES):
                    pair[cfg][n].update(base_masks[n], (probs_a[:, i] > 0.5))
                    if cfg == "ZERO_ALL":
                        leak[n].update((probs_a[:, i] > 0.5), tgt_bin[n])
                del xa, probs_a

            del x, y, base_masks, tgt_bin
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n_total == 0:
        print("No samples processed — is the split cache built?")
        return 1

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit(f" RESULTS  ({n_total} timesteps, checkpoint epoch {epoch})")
    emit("=" * 78)

    # T0 ---------------------------------------------------------------------
    emit()
    emit("T0  LEAK TEST — zero ALL predictors; the model now knows nothing")
    emit("    Any class still matching its label means a label reached the model.")
    emit()
    emit(f"    {'class':<10} {'pred px':>12} {'IoU vs label':>14}  status")
    emit("    " + "-" * 52)
    leak_clean = True
    for n in LABEL_NAMES:
        i = leak[n].iou()
        px = leak[n].a_total
        bad = (not np.isnan(i)) and i > 0.05
        leak_clean = leak_clean and not bad
        emit(f"    {n:<10} {px:>12,.0f} {i:>14.4f}  {'LEAK!' if bad else 'clean'}")
    emit()
    emit(f"    verdict: {'NO LABEL LEAKAGE — split is clean' if leak_clean else 'LEAKAGE DETECTED'}")

    # T3 ---------------------------------------------------------------------
    emit()
    emit("T3  AGREEMENT — prediction vs heuristic label (now a real skill number)")
    emit()
    emit(f"    {'class':<10} {'pred px':>12} {'label px':>12} {'IoU':>8} {'Dice':>8}")
    emit("    " + "-" * 54)
    for n in LABEL_NAMES:
        p = agree[n]
        emit(f"    {n:<10} {p.a_total:>12,.0f} {p.b_total:>12,.0f} "
             f"{p.iou():>8.4f} {p.dice():>8.4f}")

    # T1 ---------------------------------------------------------------------
    emit()
    emit("T1  SENSITIVITY — zero one predictor, % of baseline prediction surviving")
    emit("    low value = this class depends on that field")
    emit()
    header = f"    {'zeroed':<12}" + "".join(f"{n:>12}" for n in LABEL_NAMES)
    emit(header)
    emit("    " + "-" * (12 + 12 * len(LABEL_NAMES)))
    sens = {}
    for name in PRED_NAMES:
        row = f"    {name:<12}"
        sens[name] = {}
        for n in LABEL_NAMES:
            v = pair[f"LOO_{name}"][n].recall_of_a()
            sens[name][n] = v
            row += f"{100*v:>11.1f}%"
        emit(row)

    # T2 ---------------------------------------------------------------------
    emit()
    emit("T2  SOLO — keep one predictor, zero the rest; % of baseline recovered")
    emit()
    emit(header)
    emit("    " + "-" * (12 + 12 * len(LABEL_NAMES)))
    solo = {}
    for name in PRED_NAMES:
        row = f"    {name:<12}"
        solo[name] = {}
        for n in LABEL_NAMES:
            v = pair[f"SOLO_{name}"][n].recall_of_a()
            solo[name][n] = v
            row += f"{100*v:>11.1f}%"
        emit(row)

    # T4 ---------------------------------------------------------------------
    mean = rec_sum / rec_n
    var = rec_sqsum / rec_n - mean ** 2
    mse = rec_sse / rec_n
    skill = 1.0 - mse / np.maximum(var, 1e-12)
    mse_h = rec_hard_sse / rec_n
    skill_h = 1.0 - mse_h / np.maximum(var, 1e-12)

    emit()
    emit("T4  RECONSTRUCTION of the 6 physical fields (soft vs one-hot code)")
    emit("    A large soft-to-one-hot drop means the bottleneck is leaking through")
    emit("    the softmax residue instead of through the class layout.")
    emit()
    emit(f"    {'ch':>3} {'name':<12} {'soft skill':>11} {'one-hot skill':>14} {'drop':>9}")
    emit("    " + "-" * 54)
    for c, name in enumerate(PRED_NAMES):
        emit(f"    {c:>3} {name:<12} {skill[c]:>11.4f} {skill_h[c]:>14.4f} "
             f"{skill[c]-skill_h[c]:>9.4f}")

    # T6 ---------------------------------------------------------------------
    mmp = peak_maxprob / peak_n
    ment = peak_entropy / peak_n
    emit()
    emit("T6  SOFTMAX PEAKEDNESS")
    emit(f"    mean |2p-1|          : {mmp:.4f}   (1.0 = saturated, 0 = all at 0.5)")
    emit(f"    mean binary entropy  : {ment/float(np.log(2)):.4f} bits/head  (0 = hard, 1 = 0.5)")
    emit()

    txt = os.path.join(args.outdir, "phase1_report.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    js = os.path.join(args.outdir, "phase1_results.json")
    with open(js, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "checkpoint_epoch": str(epoch),
            "months": args.months, "n_timesteps": n_total,
            "leak_clean": bool(leak_clean),
            "leak": {n: {"iou": leak[n].iou(), "pred_px": leak[n].a_total}
                     for n in LABEL_NAMES},
            "agreement": {n: {"iou": agree[n].iou(), "dice": agree[n].dice(),
                              "pred_px": agree[n].a_total,
                              "label_px": agree[n].b_total} for n in LABEL_NAMES},
            "sensitivity_leave_one_out": sens,
            "sensitivity_solo": solo,
            "reconstruction": {n: {"skill": skill[c], "skill_one_hot": skill_h[c],
                                   "mse": mse[c], "var": var[c]}
                               for c, n in enumerate(PRED_NAMES)},
            "sigmoid_saturation": {"mean_abs_2p_minus_1": mmp,
                                   "mean_binary_entropy_bits": ment / float(np.log(2))},
        }, f, indent=2, default=float)
    print(f"  wrote {txt}")
    print(f"  wrote {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
