#!/usr/bin/env python3
"""
phase2_score_split.py — score a Phase-2 checkpoint on one data split with the
trainer's own validation machinery, and write the per-class metrics to JSON.

Why this exists: main_phase2.py computes IoU / precision / recall only on the
validation years each epoch. Nothing was ever computed on the training years, so a
train-vs-validation comparison (fig 6 of the paper set) needs the final checkpoint
re-run over the training split. The same call on the validation split must
reproduce the last history.json epoch — that is the correctness check — and on the
test split it is the first look at years the model has never seen.

Everything is reused, nothing re-implemented: dataset.SplitCacheDataset and
split_cache_files build the loader exactly as training did; train.validate_multilabel
runs the forward pass; metrics.SegmentationMetricAccumulator produces the same keys
that history.json carries (ar_iou, hurricane_precision, ...).

Distributed exactly like the trainer: one rank per GPU under srun, NCCL, counts
all-reduced before compute().

Usage (inside an srun with N ranks)
  python phase2_score_split.py --checkpoint <run>/final.pth --split train \
         --stride 4 --out <run>/score_train_stride4.json
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import SplitCacheDataset, split_cache_files          # noqa: E402
from metrics import SegmentationMetricAccumulator                # noqa: E402
from train import validate_multilabel                            # noqa: E402
from phase4_export_detections import load_model                  # noqa: E402
try:
    from train import split_years                                # noqa: E402
except ImportError:                                              # pragma: no cover
    from dataset import split_years                              # noqa: E402

CLASSES = ("ar", "hurricane", "etc", "mcs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--stride", type=int, default=4,
                    help="take every k-th 6-hourly frame; 4 matches the trainer's val_stride")
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    distributed = world > 1
    if distributed:
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=rank, world_size=world)
    local = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    is_main = rank == 0

    train_years, val_years, test_years = split_years()
    years = {"train": train_years, "val": val_years, "test": test_years}[args.split]
    cache_dir = os.environ.get("SPLIT_CACHE_DIR") or None
    files = split_cache_files(cache_dir=cache_dir, years=years)
    with open(args.stats) as f:
        stats = json.load(f)["stats"]
    ds = SplitCacheDataset(files, predictor_stats=stats, target_threshold=0.5, stride=args.stride)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False) if distributed else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=False, drop_last=False)

    model, epoch = load_model(args.checkpoint, device)
    if is_main:
        print(f"  split={args.split}  years={years}  months={len(files)}  samples={len(ds)}  "
              f"stride={args.stride}  ckpt epoch={epoch}  world={world}", flush=True)

    t0 = time.time()
    acc = SegmentationMetricAccumulator(device, prob_threshold=0.5, valid_mask=None)
    n_batches = validate_multilabel(model, loader, device, acc,
                                    use_amp=torch.cuda.is_available(), rank=rank)
    if distributed:
        acc.reduce(world)
    m = acc.compute()

    if is_main:
        out = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_epoch": epoch,
               "split": args.split, "years": list(map(int, years)), "n_months": len(files),
               "n_samples": len(ds), "stride": args.stride, "world_size": world,
               "seconds": round(time.time() - t0, 1),
               "metrics": {k: float(v) for k, v in m.items() if not k.endswith("_thresh")}}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  [{args.split}] " + "  ".join(
            f"{c}: IoU {m[c+'_iou']:.4f} P {m[c+'_precision']:.3f} R {m[c+'_recall']:.3f}"
            for c in CLASSES), flush=True)
        print(f"  wrote {args.out}  ({out['seconds']} s)", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
