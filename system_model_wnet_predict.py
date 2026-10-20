"""
One-off helper for the system-model figure: run the selected Phase 2 checkpoint(s) on a single
cached time step and save the four class maps (plus the reconstruction) as a small .npz that
system_model_wnet.py reads for its Output-band thumbnail.

    conda activate /lustre/orion/proj-shared/cli138/z7a/envs/my_ddp_env
    python system_model_wnet_predict.py            # one 720 x 1440 step; minutes on CPU

Defaults: tensor_cache_v11_fullres/split_202107.npy (0.25 deg), index 123 (2021-07-31 18 UTC),
predictor_stats.json, and the four per-class checkpoints of the 0.25 deg run that produced the
shipped product, J_fullres_e60: AR from best_ar.pth, Hurricane from best_hurricane.pth, ETC from
best_etc.pth, MCS from best_mcs.pth (the pairing of Run_export_fullres.sh). Passing --checkpoint
runs all four classes from that single file instead.

The cache has 721 latitude rows and the U-Net pools three times, so the -90 row is dropped
(--crop-rows 720), exactly as in training. The shipped export instead reflect-pads 721 to 728 and
crops back, which differs only in the bottom row.
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CLASS_ORDER = ("AR", "TC", "ETC", "MCS")          # the model's four output channels
PER_CLASS = {"AR": "best_ar.pth", "TC": "best_hurricane.pth",
             "ETC": "best_etc.pth", "MCS": "best_mcs.pth"}


def load_model(path, torch, WNetMultiLabel):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model_state_dict"].items()}
    model = WNetMultiLabel(in_channels=6, num_classes=4, discretize=ck.get("discretize", "ste"))
    model.load_state_dict(sd)
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(NB, "tensor_cache_v11_fullres"))
    ap.add_argument("--month", default="202107")
    ap.add_argument("--index", type=int, default=123)
    ap.add_argument("--run-dir", default=os.path.join(
        HERE, "ARs_Hurricanes_ETCs_MCS", "Phase2", "J_fullres_e60"),
        help="directory holding the four per-class checkpoints")
    ap.add_argument("--checkpoint", default=None,
                    help="single checkpoint for all four classes, instead of --run-dir")
    ap.add_argument("--crop-rows", type=int, default=720,
                    help="keep the first N latitude rows; the U-Net needs a multiple of 8")
    ap.add_argument("--stats", default=os.path.join(HERE, "predictor_stats.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "system_model_wnet_pred_2021073118.npz"))
    args = ap.parse_args()

    import torch
    from WNet_model import WNetMultiLabel

    arr = np.load(os.path.join(args.cache_dir, f"split_{args.month}.npy"), mmap_mode="r")
    step = np.array(arr[args.index], dtype=np.float32)            # (10, 721, 1440) at 0.25 deg
    if args.crop_rows:
        step = step[:, :args.crop_rows]
    meta = os.path.join(args.cache_dir, f"split_{args.month}.meta.npy")
    ts = str(np.load(meta, allow_pickle=True)[args.index]) if os.path.exists(meta) else "unknown"
    x_raw, labels = step[:6], step[6:10] > 0.5
    print(f"step {args.month} index {args.index}: grid {x_raw.shape[1]} x {x_raw.shape[2]}", flush=True)

    st = json.load(open(args.stats))
    mean = np.array([st["stats"][k]["mean"] for k in st["predictors"]], dtype=np.float32)
    std = np.array([st["stats"][k]["std"] for k in st["predictors"]], dtype=np.float32)
    x = torch.from_numpy((x_raw - mean[:, None, None]) / std[:, None, None])[None]

    # one pass per distinct checkpoint; each class is taken from the checkpoint it was shipped from
    if args.checkpoint:
        sources = {c: args.checkpoint for c in CLASS_ORDER}
    else:
        sources = {c: os.path.join(args.run_dir, PER_CLASS[c]) for c in CLASS_ORDER}
    probs = np.zeros((4, x.shape[2], x.shape[3]), np.float32)
    rec = None
    epochs = {}
    for path in sorted(set(sources.values())):
        model, ck = load_model(path, torch, WNetMultiLabel)
        with torch.no_grad():
            p, r, _ = model(x)
        p = p[0].numpy()
        for i, c in enumerate(CLASS_ORDER):
            if sources[c] == path:
                probs[i] = p[i]
                epochs[c] = int(ck.get("epoch", -1))
        if rec is None or sources["AR"] == path:      # reconstruction from the AR pass
            rec = r[0].numpy()
        print(f"  {os.path.relpath(path, HERE)}  epoch {ck.get('epoch')}", flush=True)
    preds = probs > 0.5

    tag = (os.path.relpath(args.checkpoint, HERE) if args.checkpoint
           else os.path.relpath(args.run_dir, HERE) + " (per class)")
    np.savez_compressed(args.out, preds=preds, probs=probs.astype(np.float16),
                        rec=rec.astype(np.float16), labels=labels,
                        mslp=x_raw[1], timestamp=np.array(ts),
                        checkpoint=np.array(tag),
                        epoch=np.array(epochs.get("AR", -1)))
    print("wrote", args.out, "timestamp", ts)
    for i, c in enumerate(CLASS_ORDER):
        print(f"  {c:4s} predicted cells {int(preds[i].sum()):8d}   "
              f"label cells {int(labels[i].sum()):8d}   epoch {epochs.get(c)}")


if __name__ == "__main__":
    main()
