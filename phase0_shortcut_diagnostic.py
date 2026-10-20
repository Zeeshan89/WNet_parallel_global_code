#!/usr/bin/env python3
"""
phase0_shortcut_diagnostic.py — Phase 0 of the "make the WNet really learn" plan.

Quantifies how much of the frozen model's segmentation output is a COPY of one of
its own input channels rather than a learned function of the physical fields.

Context: each class's training target is also one of the 14 input channels
(dataset.py:436-450) —

    class 1 AR        <- ch1  ivt_binary
    class 2 Hurricane <- ch5  hurricane strict eye
    class 3 ETC       <- ch10 etc_enhanced (adaptive)
    class 4 MCS       <- ch13 mcs_core

so a single convolution weight can solve the loss by passing the channel through.
These tests measure whether that is what happened. No training, no gradients —
just forward passes on a frozen checkpoint.

TESTS
  T1 leave-one-out  Zero class c's target channel. Measure what fraction of the
                    baseline class-c prediction survives.
                    survival -> 0%   ==> the class is a copy of that channel.

  T2 leave-one-in   Zero EVERY channel except class c's target. Measure what
                    fraction of the baseline prediction is reproduced.
                    reproduction -> 100% ==> the class uses that one channel alone
                    and ignores all physical fields.

  T3 agreement      IoU / Dice between the baseline prediction and the target
                    channel itself. -> 1.0 means output == label.

  T4 reconstruction Per-channel skill of the decoder (UNet2) output vs the input,
                    scored as 1 - MSE/Var so it is comparable across channels.
                    High skill on the binary mask channels + low skill on the
                    continuous IVT channel ==> the decoder is a mask pass-through,
                    not a learned generative bottleneck.

  T1b group LOO     Zero a class's WHOLE channel family, not just its target.
                    Needed because ch1 (AR target) is a deterministic function of
                    ch0 (IVT): killing ch1 alone lets the model re-derive it, so
                    the single-channel test understates AR's copy dependence.

  T5 control        Zero an auxiliary channel that is NO class's target (default
                    ch3, hurricane vorticity). If nothing changes, that channel is
                    inert and its science criteria cannot matter to the model.

  T6 peakedness     How discrete is the softmax the decoder receives? A W-Net's
                    reconstruction term only constrains the segmentation if
                    seg_probs is close to a hard assignment. Soft, continuous
                    probability maps at full resolution can carry the inputs
                    through almost losslessly, making T4's skill meaningless as
                    evidence of learning.

Usage
  python3 phase0_shortcut_diagnostic.py                       # default months
  python3 phase0_shortcut_diagnostic.py --months 201407 201401 --stride 4
"""

import os

# ── MIOpen: per-process SQLite cache + skip kernel search ────────────────────
# MUST happen before `import torch`. Without this, MIOpen's default per-user
# SQLite db on the home/lustre filesystem hits "disk I/O error" on Frontier and
# every Conv2d fails with miopenStatusInternalError. Mirrors test.py:1-15.
_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"]     = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"]        = "2"   # FAST: no search, use defaults
os.environ["MIOPEN_DISABLE_CACHE"]    = "0"
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_model import ImprovedWNet


# ── Configuration ────────────────────────────────────────────────────────────
TENSOR_CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                "Storms_precipitation_notebooks/tensor_cache_v2")

DEFAULT_CKPT = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                "Storms_precipitation_notebooks/WNet_parallel_global_code/"
                "ARs_Hurricanes_ETCs_MCS/Trail9/final_checkpoint_2006_2024.pth")

# (class name, argmax index, target channel in the 14-channel input)
CLASS_INFO = [
    ("AR",        1,  1),
    ("Hurricane", 2,  5),
    ("ETC",       3, 10),
    ("MCS",       4, 13),
]

CHANNEL_NAMES = [
    "ivt_normalized",          # 0   continuous
    "ivt_binary",              # 1   <- AR target
    "hurr_low_pressure",       # 2
    "hurr_high_vorticity",     # 3
    "hurr_high_wind_speed",    # 4
    "hurr_strict_eye",         # 5   <- Hurricane target
    "wind_speed_binary",       # 6
    "etc_low_pressure",        # 7
    "etc_vorticity",           # 8
    "etc_wind_speed",          # 9
    "etc_enhanced",            # 10  <- ETC target
    "mcs_boundary",            # 11
    "mcs_blobs",               # 12
    "mcs_core",                # 13  <- MCS target
]

MASK_BIN_THRESH = 0.5   # cached masks are fractional after the 721x1440 -> 192x384 resize

# T1b — whole-family ablations. label -> (class to report, channels to zero, why)
GROUP_ABLATIONS = [
    ("AR",        [0, 1],           "IVT pair: target + the field it is thresholded from"),
    ("AR",        [0],              "continuous IVT only, target ch1 left intact"),
    ("Hurricane", [2, 3, 4, 5, 6],  "all hurricane criterion channels"),
    ("ETC",       [7, 8, 9, 10],    "all ETC criterion channels"),
    ("MCS",       [11, 12, 13],     "all MCS criterion channels"),
]


# ── Accumulators ─────────────────────────────────────────────────────────────
class PairCounter:
    """Pixel-weighted intersection / union counts accumulated over all samples."""

    def __init__(self):
        self.inter = 0.0
        self.a_only = 0.0
        self.b_only = 0.0

    def update(self, a: torch.Tensor, b: torch.Tensor):
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
        """Fraction of A that is also in B — 'how much of A survives in B'."""
        return self.inter / self.a_total if self.a_total > 0 else float("nan")


def load_model(ckpt_path, device):
    model = ImprovedWNet(in_channels=14, num_classes=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # strip a DDP "module." prefix if present
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    return model, epoch


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--months", nargs="+", default=["201407", "201401"],
                    help="YYYYMM keys into the tensor cache")
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample timesteps within each month")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--control-channel", type=int, default=3,
                    help="an auxiliary channel that is no class's target")
    ap.add_argument("--outdir", default="phase0_diagnostics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print(" Phase 0 — input-ablation / shortcut diagnostic")
    print("=" * 78)
    print(f"  device      : {device}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  months      : {', '.join(args.months)}  (stride {args.stride})")

    model, epoch = load_model(args.checkpoint, device)
    print(f"  ckpt epoch  : {epoch}")

    # ── build the ablation configurations ────────────────────────────────────
    # name -> list of channels to zero
    configs = OrderedDict()
    configs["baseline"] = []
    for name, _, tgt in CLASS_INFO:
        configs[f"LOO_{name}"] = [tgt]                                   # T1
        configs[f"LOI_{name}"] = [c for c in range(14) if c != tgt]      # T2
    group_keys = []                                                      # T1b
    for cls_name, chans, why in GROUP_ABLATIONS:
        key = f"GRP_{cls_name}_" + "-".join(str(c) for c in chans)
        configs[key] = chans
        group_keys.append((key, cls_name, chans, why))
    cc = args.control_channel
    configs[f"CTRL_ch{cc}"] = [cc]                                       # T5

    # ── accumulators ─────────────────────────────────────────────────────────
    # pred-vs-pred comparisons, per class per config
    pair = {cfg: {name: PairCounter() for name, _, _ in CLASS_INFO} for cfg in configs}
    # pred-vs-target-mask agreement (T3), baseline only
    agree = {name: PairCounter() for name, _, _ in CLASS_INFO}
    # reconstruction (T4)
    rec_sse = np.zeros(14, dtype=np.float64)   # sum sq error
    rec_sum = np.zeros(14, dtype=np.float64)   # sum of x   (for variance)
    rec_sqsum = np.zeros(14, dtype=np.float64) # sum of x^2
    rec_n = 0
    # T4b: same decoder, fed a strictly one-hot version of its own input
    rec_hard_sse = np.zeros(14, dtype=np.float64)
    # softmax peakedness (T6)
    peak_maxprob = 0.0     # sum of max_c seg_probs[c]
    peak_entropy = 0.0     # sum of -sum_c p log p  (nats)
    peak_n = 0

    n_samples_total = 0

    for month in args.months:
        npy = os.path.join(TENSOR_CACHE, f"tensors_{month}.npy")
        if not os.path.exists(npy):
            print(f"  ! missing {npy} — skipping")
            continue
        arr = np.load(npy, mmap_mode="r")
        idx = list(range(0, arr.shape[0], args.stride))
        print(f"\n  {month}: {arr.shape[0]} timesteps -> using {len(idx)}")

        for start in range(0, len(idx), args.batch_size):
            sel = idx[start:start + args.batch_size]
            x = torch.from_numpy(np.asarray(arr[sel], dtype=np.float32)).to(device)
            n_samples_total += x.shape[0]

            # target masks, binarised
            tgt_bin = {name: (x[:, ch] > MASK_BIN_THRESH)
                       for name, _, ch in CLASS_INFO}

            # --- baseline forward (also gives us the reconstruction) ---------
            seg_probs, rec = model(x)
            base_pred = seg_probs.argmax(dim=1)
            base_masks = {name: (base_pred == ci) for name, ci, _ in CLASS_INFO}

            # T3 agreement + baseline self-pair
            for name, _, _ in CLASS_INFO:
                agree[name].update(base_masks[name], tgt_bin[name])
                pair["baseline"][name].update(base_masks[name], base_masks[name])

            # T4 reconstruction stats
            err = (rec - x).float()
            rec_sse += (err ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_sum += x.float().sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_sqsum += (x.float() ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            rec_n += x.shape[0] * x.shape[2] * x.shape[3]

            # T4b feed the SAME decoder a strictly one-hot map. If the skill
            # collapses, the reconstruction was riding on the sub-1% soft
            # residue of the softmax, not on the class layout itself.
            hard = F.one_hot(base_pred, num_classes=5).permute(0, 3, 1, 2).float()
            rec_h = model.unet2(hard)
            rec_hard_sse += ((rec_h - x).float() ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            del hard, rec_h

            # T6 how hard is the assignment the decoder actually sees?
            p = seg_probs.float().clamp_min(1e-12)
            peak_maxprob += p.max(dim=1).values.sum().item()
            peak_entropy += (-(p * p.log()).sum(dim=1)).sum().item()
            peak_n += x.shape[0] * x.shape[2] * x.shape[3]

            del seg_probs, rec, err, p

            # --- ablated forwards --------------------------------------------
            for cfg, zero_channels in configs.items():
                if cfg == "baseline":
                    continue
                xa = x.clone()
                for c in zero_channels:
                    xa[:, c] = 0.0
                seg_a, _ = model(xa)
                pred_a = seg_a.argmax(dim=1)
                for name, ci, _ in CLASS_INFO:
                    # A = baseline prediction, B = ablated prediction
                    pair[cfg][name].update(base_masks[name], (pred_a == ci))
                del xa, seg_a, pred_a

            del x, base_pred, base_masks, tgt_bin
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n_samples_total == 0:
        print("No samples processed — check the tensor cache path.")
        return 1

    # ── report ───────────────────────────────────────────────────────────────
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit(f" RESULTS  ({n_samples_total} timesteps, checkpoint epoch {epoch})")
    emit("=" * 78)

    # T3 --------------------------------------------------------------------
    emit()
    emit("T3  AGREEMENT — baseline prediction vs its own target channel")
    emit("    (IoU -> 1.0 means the output IS the label)")
    emit()
    emit(f"    {'class':<10} {'tgt ch':>6} {'pred px':>12} {'target px':>12} "
         f"{'IoU':>8} {'Dice':>8}")
    emit("    " + "-" * 62)
    for name, _, ch in CLASS_INFO:
        p = agree[name]
        emit(f"    {name:<10} {ch:>6} {p.a_total:>12,.0f} {p.b_total:>12,.0f} "
             f"{p.iou():>8.4f} {p.dice():>8.4f}")

    # T1 --------------------------------------------------------------------
    emit()
    emit("T1  LEAVE-ONE-OUT — zero the class's own target channel")
    emit("    survival = fraction of the baseline prediction that remains")
    emit("    survival near 0%  ==>  the class is a copy of that channel")
    emit()
    emit(f"    {'class':<10} {'zeroed':>8} {'base px':>12} {'ablated px':>12} "
         f"{'survival':>10}")
    emit("    " + "-" * 56)
    for name, _, ch in CLASS_INFO:
        p = pair[f"LOO_{name}"][name]
        emit(f"    {name:<10} {'ch'+str(ch):>8} {p.a_total:>12,.0f} "
             f"{p.b_total:>12,.0f} {100*p.recall_of_a():>9.2f}%")

    # T1b -------------------------------------------------------------------
    emit()
    emit("T1b GROUP LEAVE-ONE-OUT — zero a class's whole channel family")
    emit("    catches redundant encodings the single-channel test misses")
    emit()
    emit(f"    {'class':<10} {'zeroed':<14} {'base px':>12} {'ablated px':>12} "
         f"{'survival':>10}  why")
    emit("    " + "-" * 92)
    for key, cls_name, chans, why in group_keys:
        p = pair[key][cls_name]
        chan_str = "ch" + ",".join(str(c) for c in chans)
        emit(f"    {cls_name:<10} {chan_str:<14} {p.a_total:>12,.0f} "
             f"{p.b_total:>12,.0f} {100*p.recall_of_a():>9.2f}%  {why}")

    # T2 --------------------------------------------------------------------
    emit()
    emit("T2  LEAVE-ONE-IN — zero everything EXCEPT the class's target channel")
    emit("    reproduction = fraction of the baseline prediction recovered")
    emit("    reproduction near 100%  ==>  the class ignores all other inputs")
    emit()
    emit(f"    {'class':<10} {'kept':>8} {'base px':>12} {'ablated px':>12} "
         f"{'reproduced':>11} {'IoU':>8}")
    emit("    " + "-" * 66)
    for name, _, ch in CLASS_INFO:
        p = pair[f"LOI_{name}"][name]
        emit(f"    {name:<10} {'ch'+str(ch):>8} {p.a_total:>12,.0f} "
             f"{p.b_total:>12,.0f} {100*p.recall_of_a():>10.2f}% {p.iou():>8.4f}")

    # T5 --------------------------------------------------------------------
    emit()
    emit(f"T5  CONTROL — zero ch{cc} ({CHANNEL_NAMES[cc]}), which is no class's target")
    emit("    survival near 100% for every class ==> that channel is inert")
    emit()
    emit(f"    {'class':<10} {'base px':>12} {'ablated px':>12} {'survival':>10} {'IoU':>8}")
    emit("    " + "-" * 56)
    for name, _, _ in CLASS_INFO:
        p = pair[f"CTRL_ch{cc}"][name]
        emit(f"    {name:<10} {p.a_total:>12,.0f} {p.b_total:>12,.0f} "
             f"{100*p.recall_of_a():>9.2f}% {p.iou():>8.4f}")

    # T4 --------------------------------------------------------------------
    mean = rec_sum / rec_n
    var = rec_sqsum / rec_n - mean ** 2
    mse = rec_sse / rec_n
    skill = 1.0 - mse / np.maximum(var, 1e-12)

    emit()
    emit("T4  RECONSTRUCTION — decoder (UNet2) output vs the 14-channel input")
    emit("    skill = 1 - MSE/Var.  1.0 = perfect, 0.0 = no better than the mean")
    emit()
    emit(f"    {'ch':>3} {'name':<22} {'role':<12} {'MSE':>11} {'Var':>11} {'skill':>8}")
    emit("    " + "-" * 70)
    target_channels = {ch: name for name, _, ch in CLASS_INFO}
    for c in range(14):
        role = f"{target_channels[c]} tgt" if c in target_channels else (
            "continuous" if c == 0 else "aux")
        emit(f"    {c:>3} {CHANNEL_NAMES[c]:<22} {role:<12} {mse[c]:>11.5f} "
             f"{var[c]:>11.5f} {skill[c]:>8.4f}")

    # T4b -------------------------------------------------------------------
    mse_hard = rec_hard_sse / rec_n
    skill_hard = 1.0 - mse_hard / np.maximum(var, 1e-12)
    emit()
    emit("T4b RECONSTRUCTION FROM A STRICTLY ONE-HOT MAP")
    emit("    Same decoder, but seg_probs replaced by its own argmax one-hot.")
    emit("    skill collapsing here ==> T4 was riding on the soft residue of the")
    emit("    softmax, so the 'bottleneck' never forced the classes to mean anything.")
    emit()
    emit(f"    {'ch':>3} {'name':<22} {'soft skill':>11} {'one-hot skill':>14} {'drop':>9}")
    emit("    " + "-" * 64)
    for c in range(14):
        emit(f"    {c:>3} {CHANNEL_NAMES[c]:<22} {skill[c]:>11.4f} "
             f"{skill_hard[c]:>14.4f} {skill[c]-skill_hard[c]:>9.4f}")

    # T6 --------------------------------------------------------------------
    mean_maxprob = peak_maxprob / peak_n
    mean_entropy = peak_entropy / peak_n
    max_entropy = float(np.log(5))
    emit()
    emit("T6  SOFTMAX PEAKEDNESS — how discrete is the code the decoder receives?")
    emit("    A near-hard assignment makes T4's reconstruction skill meaningful.")
    emit("    A soft code can smuggle the inputs through, so T4 proves nothing.")
    emit()
    emit(f"    mean max-probability   : {mean_maxprob:.4f}   (1.0 = hard assignment, 0.2 = uniform)")
    emit(f"    mean entropy           : {mean_entropy:.4f} nats  of max {max_entropy:.4f}")
    emit(f"    normalised entropy     : {mean_entropy/max_entropy:.4f}   (0 = hard, 1 = uniform)")

    # ── verdict ──────────────────────────────────────────────────────────────
    emit()
    emit("=" * 78)
    emit(" COPY-DEPENDENCE SUMMARY  (the Phase 0 baseline number)")
    emit("=" * 78)
    emit()
    emit("    LOO  = survival after zeroing the target channel alone")
    emit("    GRP  = survival after zeroing the class's whole channel family")
    emit("    LOI  = baseline recovered from the target channel alone")
    emit()
    emit(f"    {'class':<10} {'agree IoU':>10} {'LOO surv':>10} {'GRP surv':>10} "
         f"{'LOI repro':>11} {'verdict':>22}")
    emit("    " + "-" * 79)
    # the family ablation that best represents each class
    grp_for_class = {}
    for key, cls_name, chans, _ in group_keys:
        # prefer the widest family ablation for the class
        if cls_name not in grp_for_class or len(chans) > len(grp_for_class[cls_name][1]):
            grp_for_class[cls_name] = (key, chans)

    summary = {}
    for name, _, ch in CLASS_INFO:
        a = agree[name].iou()
        loo = pair[f"LOO_{name}"][name].recall_of_a()
        loi = pair[f"LOI_{name}"][name].recall_of_a()
        gkey, gchans = grp_for_class.get(name, (None, []))
        grp = pair[gkey][name].recall_of_a() if gkey else float("nan")

        # the decisive number is the family ablation when we have one
        decisive = grp if not np.isnan(grp) else loo
        if decisive < 0.10 and (not np.isnan(loi)) and loi > 0.90:
            verdict = "PURE COPY"
        elif decisive < 0.35:
            verdict = "dominated by copy"
        elif decisive < 0.70:
            verdict = "partial copy"
        else:
            verdict = "uses other inputs"
        summary[name] = dict(target_channel=ch, agreement_iou=a,
                             loo_survival=loo, group_survival=grp,
                             group_channels=gchans, loi_reproduction=loi,
                             verdict=verdict)
        emit(f"    {name:<10} {a:>10.4f} {100*loo:>9.2f}% {100*grp:>9.2f}% "
             f"{100*loi:>10.2f}% {verdict:>22}")
    emit()

    # ── persist ──────────────────────────────────────────────────────────────
    txt_path = os.path.join(args.outdir, "phase0_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_path = os.path.join(args.outdir, "phase0_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": str(epoch),
            "months": args.months,
            "stride": args.stride,
            "n_timesteps": n_samples_total,
            "summary": summary,
            "agreement": {n: {"iou": agree[n].iou(), "dice": agree[n].dice(),
                              "pred_px": agree[n].a_total,
                              "target_px": agree[n].b_total}
                          for n, _, _ in CLASS_INFO},
            "leave_one_out": {n: {"survival": pair[f"LOO_{n}"][n].recall_of_a(),
                                  "base_px": pair[f"LOO_{n}"][n].a_total,
                                  "ablated_px": pair[f"LOO_{n}"][n].b_total}
                              for n, _, _ in CLASS_INFO},
            "leave_one_in": {n: {"reproduction": pair[f"LOI_{n}"][n].recall_of_a(),
                                 "iou": pair[f"LOI_{n}"][n].iou(),
                                 "base_px": pair[f"LOI_{n}"][n].a_total,
                                 "ablated_px": pair[f"LOI_{n}"][n].b_total}
                             for n, _, _ in CLASS_INFO},
            "group_leave_one_out": [
                {"class": cls_name, "channels": chans, "why": why,
                 "survival": pair[key][cls_name].recall_of_a(),
                 "base_px": pair[key][cls_name].a_total,
                 "ablated_px": pair[key][cls_name].b_total}
                for key, cls_name, chans, why in group_keys
            ],
            "softmax_peakedness": {"mean_max_prob": mean_maxprob,
                                   "mean_entropy_nats": mean_entropy,
                                   "max_entropy_nats": max_entropy,
                                   "normalised_entropy": mean_entropy / max_entropy},
            "control": {"channel": cc, "name": CHANNEL_NAMES[cc],
                        "per_class": {n: {"survival": pair[f"CTRL_ch{cc}"][n].recall_of_a(),
                                          "iou": pair[f"CTRL_ch{cc}"][n].iou()}
                                      for n, _, _ in CLASS_INFO}},
            "reconstruction": {CHANNEL_NAMES[c]: {"channel": c, "mse": mse[c],
                                                  "var": var[c], "skill": skill[c],
                                                  "skill_one_hot": skill_hard[c],
                                                  "mse_one_hot": mse_hard[c]}
                               for c in range(14)},
        }, f, indent=2, default=float)

    print(f"  wrote {txt_path}")
    print(f"  wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
