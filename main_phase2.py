#!/usr/bin/env python3
"""
main_phase2.py — training entry point for the Phase 2 multi-label W-Net.

Separate from main.py on purpose. main.py exists to assemble a month from ERA5,
IBTrACS, percentile files and climatologies, then train on that one month before
moving to the next. The Phase 1 split cache already contains everything the model
needs, so all of that disappears and what is left is an ordinary training script:
index every timestep across every month, shuffle globally, train.

That also fixes the structural problem in the old loop. Training file-by-file
meant the model saw one month of weather at a time and the loss curve tracked
which file was loaded rather than learning progress. Here one Dataset spans the
whole record and the sampler shuffles across it.

What is being trained (Phase 2 decisions):
  * four INDEPENDENT sigmoid heads, no softmax, no overlap penalty — ARs and
    ETCs genuinely co-occur
  * BCE(pos_weight) + Tversky(beta > alpha) per class, for the rare-class imbalance
  * reconstruction from a straight-through DISCRETISED code, so the bottleneck
    actually binds (Phase 0: soft code reconstructed IVT at 0.996, one-hot at 0.369)
  * N-cut as a regulariser, using the corrected multi-channel affinity
  * held-out YEARS for val/test — random frame splits leak, because consecutive
    6-hourly frames are near-duplicates

Usage
  python3 main_phase2.py --epochs 20 --batch-size 8
  python3 main_phase2.py --epochs 2 --limit-train-months 12   # quick probe
"""

import os

# MIOpen per-process cache — MUST precede `import torch` on Frontier.
_rank_env = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank_env}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_DISABLE_CACHE"] = "0"

import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import SplitCacheDataset, split_cache_files
from WNet_model import WNetMultiLabel
from loss import StormSegLossV2, NCutLoss2DFixed
from metrics import SegmentationMetricAccumulator
from train import (train_wnet_multilabel, validate_multilabel,
                   split_years, VAL_YEARS, TEST_YEARS, load_init_weights)

CKPT_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
            "Storms_precipitation_notebooks/WNet_parallel_global_code/"
            "ARs_Hurricanes_ETCs_MCS/Phase2")

# Best IoU obtainable by thresholding ONE raw predictor at a single global value,
# measured in phase1_verify_split.py. This is the bar the network must clear, and
# it differs by an order of magnitude across classes — AR's label is essentially
# `IVT > percentile`, so a constant threshold already reproduces it at 0.94.
# Stored here so every run records the reference it is being judged against.
TRIVIAL_BASELINE = {"ar": 0.9371, "hurricane": 0.0333,
                    "etc": 0.0898, "mcs": 0.2124}


def setup_distributed():
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ["SLURM_NTASKS"])
        local = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(local)
        return True, world, rank, local, torch.device(f"cuda:{local}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 1, 0, 0, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="base LR; scaled by sqrt(world_size) when distributed")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--ncut-weight", type=float, default=0.1)
    # NCut neighbourhood in PIXELS (radius, spatial sigma). Defaults are the 192x384 values; on the
    # 0.25 deg cache use 15 / 15 so the kernel covers the same ~3.75 deg.
    ap.add_argument("--ncut-radius", type=int, default=4)
    ap.add_argument("--ncut-sigma1", type=float, default=4.0)
    # 0.25 deg cache: 721 rows, the U-Net needs a multiple of 8 -> --crop-rows 720 drops the -90 row
    ap.add_argument("--crop-rows", type=int, default=None)
    # None = the warmup/cosine schedule in _dynamic_weights. A number pins the
    # reconstruction weight, and 0 disables the decoder branch entirely, which
    # turns the W-Net into a plain supervised U-Net — the ablation that says
    # whether the second half of the architecture earns its cost.
    ap.add_argument("--rec-weight", type=float, default=None)
    ap.add_argument("--discretize", default="ste", choices=["ste", "soft"])
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--limit-train-months", type=int, default=None,
                    help="use only the first N training months (quick probes)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=4,
                    help="subsample validation timesteps; 6-hourly frames are "
                         "near-duplicates so this is nearly free accuracy-wise")
    ap.add_argument("--train-stride", type=int, default=1)
    # Training-set metrics. The trainer used to log losses on the training split
    # but IoU/precision/recall only on validation, so a train-vs-validation
    # comparison was impossible after the fact. Each epoch now also scores the model
    # (eval mode, same accumulator as validation) on every k-th training frame and
    # writes train_<class>_<metric> next to the val_ keys. Stride 4 mirrors
    # --val-stride and costs ~2 s per epoch; 0 disables the pass.
    ap.add_argument("--train-eval-stride", type=int, default=4,
                    help="subsample of the TRAINING years scored each epoch for "
                         "train_* metrics (0 = off)")
    # Rare-class knobs. Both pos_weight and Tversky beta push recall, so they
    # interact — the first smoke run came out at recall ~1.0 with low precision.
    # Each accepts either one value (shared) or four, ordered AR,Hurricane,ETC,MCS.
    ap.add_argument("--max-pos-weight", default="50",
                    help="scalar or 4 comma-separated values")
    ap.add_argument("--tversky-beta", default="0.7",
                    help="scalar or 4 comma-separated values (FN penalty)")
    ap.add_argument("--tversky-alpha", default="0.3",
                    help="scalar or 4 comma-separated values (FP penalty)")
    ap.add_argument("--tag", default="", help="label for this run in the ckpt dir")
    # Split a long run across several short jobs without distorting it. The LR
    # schedule is built from --epochs (cosine, T_max = epochs - warmup), so the
    # ONLY correct way to chunk a run is to keep --epochs at the full target and
    # limit how many epochs THIS job executes. Chunking by lowering --epochs
    # instead would compress the cosine and make the pieces non-equivalent.
    # Exits 0 after the budget, so an afterok dependency still fires.
    ap.add_argument("--epochs-this-run", type=int, default=0,
                    help="stop cleanly after this many epochs in this job "
                         "(0 = no limit); resume the rest with --resume last.pth")
    # best.pth is selected on skill averaged over ALL FOUR classes, which is the
    # right criterion for a multi-task product but the wrong one for any single
    # head. Measured on E_ibtracs200: TC IoU peaks at epoch 12 (0.5298, recall
    # 0.697) while the four-class skill keeps improving to epoch 130 (TC IoU
    # 0.5084, recall 0.648) because AR/ETC/MCS gains outweigh the TC loss. The
    # shipped TC model was therefore 118 epochs past its own optimum. This writes
    # an ADDITIONAL best_<class>.pth on that class's own IoU; best.pth is
    # untouched, so nothing about the existing selection changes.
    # Accepts SEVERAL classes, so one run can ship a properly selected head for every
    # storm type. Selecting on only one class is what left the v3 product without a
    # usable TC head: that run passed --select-class ar, so the TC class had to be
    # exported from best.pth (4-class), whose TC recall is 0.61 against 0.75 for a
    # TC-selected checkpoint, and the tracker then found 8 fewer storms per year.
    ap.add_argument("--select-class", nargs="+", default=None,
                    choices=["ar", "hurricane", "etc", "mcs"],
                    help="also checkpoint on each of these classes, to best_<class>.pth")
    # IoU is the WRONG criterion for TC and this was measured, not assumed. On the
    # 200-epoch cap-50 run, max TC IoU picked epoch 36 (recall 0.670) over epoch 13
    # of the short run (recall 0.754); tracked, those gave POD 0.808 vs 0.850. Track
    # POD follows RECALL, because recall is what buys the temporal continuity a
    # track needs -- IoU moved 0.008 while POD moved 0.070. A bare recall maximum is
    # no good either (this run's recall peak, epoch 17, had precision 0.385), so
    # recall selection is gated by a precision floor.
    # One value applies to every selected class; a list must match --select-class,
    # because the right criterion is class-dependent: IoU for AR, ETC and MCS, but
    # recall at a precision floor for TC, per the measurement described above.
    ap.add_argument("--select-metric", nargs="+", default=["iou"], choices=["iou", "recall"],
                    help="what --select-class maximises, one value or one per class")
    # Accepts a LIST. The right floor is not knowable in advance and is not portable
    # between datasets: ERA5 shipped 0.60, but on RegCM MPI's four-class best.pth
    # already sits at precision 0.568, so a 0.60 floor there LOSES recall. Since
    # per-epoch weights are not kept, finding the floor by experiment would otherwise
    # cost one full training run per candidate. Passing "0.45,0.50,0.55,0.60" writes
    # best_<class>_p45.pth ... _p60.pth from a SINGLE run, and the floor is then
    # chosen on track POD/FAR downstream, which is the only criterion that matters.
    # A single value keeps the original behaviour and the original filename.
    ap.add_argument("--select-min-precision", default="0.0",
                    help="with --select-metric recall, ignore epochs below this "
                         "precision. Scalar -> best_<class>.pth; comma-separated "
                         "list -> one best_<class>_p<NN>.pth per floor")
    # ── Dataset-agnostic knobs (added for the RegCM transfer) ────────────────────
    # Every default below reproduces the previous ERA5 behaviour exactly.
    ap.add_argument("--init-weights", default=os.environ.get("INIT_WEIGHTS") or None,
                    help="checkpoint to initialise from (transfer, NOT resume): "
                         "model weights only, fresh optimizer/scheduler/EMA, epoch 0")
    ap.add_argument("--allow-partial-init", action="store_true",
                    help="permit an --init-weights load matching <95%% of tensors")
    ap.add_argument("--baseline-json", default=None,
                    help="per-dataset trivial single-threshold baseline from "
                         "phase1_verify_split.py. Model selection is skill relative "
                         "to it, and each RegCM model has its OWN AR threshold "
                         "(287-357 kg/m/s), so the ERA5 dict is wrong for them.")
    ap.add_argument("--year-range", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="dataset year span; RegCM is 1981 2014. VAL/TEST years "
                         "outside the span drop out automatically.")
    ap.add_argument("--valid-mask", default=None,
                    help="(H,W) or (4,H,W) .npy; pixels outside are excluded from "
                         "the loss AND the metrics")
    ap.add_argument("--seed", type=int, default=0,
                    help="REQUIRED for the fine-tune vs scratch comparison to be "
                         "honest: without a fixed seed the two arms differ in data "
                         "order as well as initialisation")
    ap.add_argument("--resume", default=None,
                    help="resume a killed run (model+optimizer+scheduler+scaler+"
                         "EMA+epoch), unlike --init-weights")
    args = ap.parse_args()

    def _nums(s):
        v = [float(x) for x in str(s).split(",")]
        return v[0] if len(v) == 1 else v

    args.max_pos_weight = _nums(args.max_pos_weight)
    args.tversky_beta = _nums(args.tversky_beta)
    args.tversky_alpha = _nums(args.tversky_alpha)

    # Precision floors: always a list internally. A single value keeps the original
    # filename (best_<class>.pth) so existing callers are unaffected.
    _floors = [float(x) for x in str(args.select_min_precision).split(",") if x != ""]
    if not _floors:
        _floors = [0.0]
    args.select_min_precision = sorted(set(_floors))
    args.select_multi_floor = len(args.select_min_precision) > 1
    # One metric per selected class. A single value is broadcast, so the previous
    # single-class callers behave exactly as before.
    if args.select_class:
        if len(args.select_metric) == 1:
            args.select_metric = args.select_metric * len(args.select_class)
        elif len(args.select_metric) != len(args.select_class):
            raise SystemExit(f"--select-metric takes 1 value or one per --select-class "
                             f"({len(args.select_class)}); got {len(args.select_metric)}")
    if args.tag:
        args.ckpt_dir = os.path.join(args.ckpt_dir, args.tag)

    distributed, world, rank, local, device = setup_distributed()
    is_main = rank == 0

    # Seed every source of order/initialisation. DistributedSampler is seeded below.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    baseline = dict(TRIVIAL_BASELINE)
    if args.baseline_json:
        with open(args.baseline_json) as f:
            _b = json.load(f)
        _b = _b.get("baseline", _b)
        baseline = {k: float(_b[k]) for k in ("ar", "hurricane", "etc", "mcs")}
        if is_main:
            print(f"  trivial baseline override: {args.baseline_json} -> {baseline}")

    valid_mask = np.load(args.valid_mask) if args.valid_mask else None
    if valid_mask is not None and is_main:
        print(f"  valid mask: {args.valid_mask} shape {valid_mask.shape} "
              f"({100*np.mean(valid_mask):.1f}% of frame)")

    with open(args.stats) as f:
        stats = json.load(f)["stats"]

    train_years, val_years, test_years = (
        split_years(*args.year_range) if args.year_range else split_years())
    # SPLIT_CACHE_DIR lets a run target an alternative label build without editing
    # the module default, so v3 (original hurricane label) and v4 (require_both=True,
    # wind AND vorticity) are both reachable and the run records which it used.
    cache_dir = os.environ.get("SPLIT_CACHE_DIR") or None
    if cache_dir:
        print(f"  split cache override: {cache_dir}", flush=True)
    train_files = split_cache_files(cache_dir=cache_dir, years=train_years)
    val_files = split_cache_files(cache_dir=cache_dir, years=val_years)

    if args.limit_train_months:
        train_files = train_files[:args.limit_train_months]

    if is_main:
        print("=" * 78)
        print(" Phase 2 — multi-label W-Net training")
        print("=" * 78)
        print(f"  device        : {device}  (distributed={distributed}, world={world})")
        print(f"  train years   : {len(train_years)}  ({len(train_files)} months)")
        print(f"  val   years   : {val_years}  ({len(val_files)} months)")
        print(f"  test  years   : {test_years}  (held out, never touched here)")
        print(f"  discretize    : {args.discretize}")
        print(f"  ncut weight   : {args.ncut_weight}")
        print(f"  batch/accum   : {args.batch_size} x {args.grad_accum} x {world} GPUs")
        print(f"  tversky a/b   : {args.tversky_alpha} / {args.tversky_beta}")
        print(f"  max_pos_weight: {args.max_pos_weight}")
        print(f"  ckpt dir      : {args.ckpt_dir}")
        print("=" * 78)

    train_ds = SplitCacheDataset(train_files, predictor_stats=stats,
                                 target_threshold=0.5, stride=args.train_stride, crop_rows=args.crop_rows)
    val_ds = SplitCacheDataset(val_files, predictor_stats=stats,
                               target_threshold=0.5, stride=args.val_stride, crop_rows=args.crop_rows)

    train_sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                        shuffle=True, seed=args.seed)
                     if distributed else None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=False,
        drop_last=True, persistent_workers=args.num_workers > 0)
    # Shard validation too. Without a sampler every rank walked the WHOLE val set
    # and acc.reduce() then summed the counts — IoU is a ratio so the number came
    # out right, but the work was done world_size times over. DistributedSampler
    # pads the last shard by repeating a handful of samples; on ~1.5k val samples
    # that is under half a percent and does not move the metric.
    val_sampler = (DistributedSampler(val_ds, num_replicas=world, rank=rank,
                                      shuffle=False) if distributed else None)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        persistent_workers=args.num_workers > 0)

    # Strided, unshuffled view of the TRAINING years, scored in eval mode each epoch
    # exactly like validation, so train_* and val_* metrics are directly comparable.
    train_eval_loader = None
    if args.train_eval_stride > 0:
        train_eval_ds = SplitCacheDataset(train_files, predictor_stats=stats,
                                          target_threshold=0.5, stride=args.train_eval_stride, crop_rows=args.crop_rows)
        train_eval_sampler = (DistributedSampler(train_eval_ds, num_replicas=world, rank=rank,
                                                 shuffle=False) if distributed else None)
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=args.batch_size, shuffle=False, sampler=train_eval_sampler,
            num_workers=args.num_workers, pin_memory=False, drop_last=False,
            persistent_workers=args.num_workers > 0)
        if is_main:
            print(f"  train-eval    : every {args.train_eval_stride}th training frame "
                  f"({len(train_eval_ds)} samples) scored per epoch")

    model = WNetMultiLabel(in_channels=6, num_classes=4,
                           discretize=args.discretize).to(device)
    init_report = {}
    if args.init_weights:
        init_report = load_init_weights(model, args.init_weights, device=device,
                                        allow_partial=args.allow_partial_init,
                                        verbose=is_main)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local], output_device=local,
                    find_unused_parameters=False)

    lr = args.lr * (world ** 0.5) if distributed else args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Linear warmup then cosine. Without warmup the sqrt-scaled LR is applied at
    # full value on step 1, which is survivable at 8 GPUs (the pilot ran at
    # 2.8e-3) but not at 64, where the same rule gives 8e-3 on an effective batch
    # of 512. Large-batch runs need the ramp; small ones are unaffected by it.
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1))
    if warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[warmup_epochs])
    else:
        scheduler = cosine
    scaler = GradScaler() if torch.cuda.is_available() else None
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    seg_loss = StormSegLossV2(tversky_alpha=args.tversky_alpha,
                              tversky_beta=args.tversky_beta,
                              max_pos_weight=args.max_pos_weight,
                              valid_mask=valid_mask).to(device)
    ncut_loss = NCutLoss2DFixed(radius=args.ncut_radius, sigma_1=args.ncut_sigma1).to(device)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    history = []
    best_mean_iou = -1.0
    best_skill = -1.0
    # One running best per precision floor, keyed by the floor. Was a bare float.
    best_class_iou = {}
    start_epoch = 0

    # --resume restores the FULL training state. --init-weights deliberately does
    # not: that is transfer, and carrying an optimizer across datasets would apply
    # ERA5 moment estimates to RegCM gradients.
    if args.resume and os.path.exists(args.resume):
        _ck = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(
            _ck["model_state_dict"])
        optimizer.load_state_dict(_ck["optimizer_state_dict"])
        scheduler.load_state_dict(_ck["scheduler_state_dict"])
        if scaler is not None and _ck.get("scaler_state_dict"):
            scaler.load_state_dict(_ck["scaler_state_dict"])
        if _ck.get("ema_state_dict"):
            ema.load_state_dict(_ck["ema_state_dict"])
        start_epoch = int(_ck.get("epoch", -1)) + 1
        best_skill = float(_ck.get("best_skill", -1.0))
        best_mean_iou = float(_ck.get("best_mean_iou", -1.0))
        # Was a bare float before multi-floor selection; accept either so a run
        # started under the old code can still be resumed under the new.
        # Keys are (class, floor) since selection became multi-class. Older checkpoints
        # hold a floor-keyed dict (single class) or a bare float, so both are accepted
        # and attributed to the first selected class.
        _bci = _ck.get("best_class_iou", {})
        _c0 = args.select_class[0] if args.select_class else "ar"
        if isinstance(_bci, dict):
            best_class_iou = {(k if isinstance(k, tuple) else (_c0, float(k))): float(v)
                              for k, v in _bci.items()}
        else:
            best_class_iou = {(_c0, args.select_min_precision[0]): float(_bci)}
        history = list(_ck.get("history", []))
        if is_main:
            print(f"  resumed {args.resume} at epoch {start_epoch}")

    # Everything needed to reproduce or caption this run, written once up front
    # so a history file is self-describing when it turns up months later.
    run_config = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lr": lr, "base_lr": args.lr,
        "world_size": world, "discretize": args.discretize,
        "ncut_weight": args.ncut_weight, "ncut_radius": args.ncut_radius,
        "ncut_sigma1": args.ncut_sigma1, "crop_rows": args.crop_rows,
        "split_cache_dir": cache_dir, "stats_file": args.stats,
        "tversky_alpha": args.tversky_alpha, "tversky_beta": args.tversky_beta,
        "max_pos_weight": args.max_pos_weight,
        "train_years": train_years, "val_years": val_years,
        "test_years": test_years,
        "n_train_months": len(train_files), "n_val_months": len(val_files),
        "n_train_samples": len(train_ds), "n_val_samples": len(val_ds),
        "train_stride": args.train_stride, "val_stride": args.val_stride,
        "train_eval_stride": args.train_eval_stride,
        "trivial_baseline": baseline,
        "seed": args.seed,
        "valid_mask": args.valid_mask,
        "baseline_json": args.baseline_json,
        "year_range": args.year_range,
        # Recorded so a history.json says which checkpoints the run wrote and on
        # what criterion. Without this, "which epoch got exported and why" is only
        # recoverable from the job script — and it was a silent scalar/list
        # discrepancy of exactly that kind that mis-weighted the TC head before.
        "select_class": args.select_class,
        "select_metric": args.select_metric,
        "select_min_precision": args.select_min_precision,
        **init_report,
        "predictors": ["IVT", "MSLP", "vorticity", "wind_speed", "BT", "IWV"],
        "classes": ["AR", "Hurricane", "ETC", "MCS"],
        "tag": args.tag,
    }

    def _persist():
        """Write history as JSON (self-describing) and CSV (straight to plots)."""
        out = {"config": run_config, "epochs": history}
        with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        if history:
            import csv
            keys = sorted({k for r in history for k in r})
            keys = ["epoch"] + [k for k in keys if k != "epoch"]
            with open(os.path.join(args.ckpt_dir, "history.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in history:
                    w.writerow(r)

    stop_epoch = args.epochs
    if args.epochs_this_run > 0:
        stop_epoch = min(args.epochs, start_epoch + args.epochs_this_run)
        if is_main:
            print(f"  this job runs epochs {start_epoch}..{stop_epoch - 1} "
                  f"of {args.epochs} (budget {args.epochs_this_run})")
    for epoch in range(start_epoch, stop_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()

        losses, details = train_wnet_multilabel(
            model, train_loader, optimizer, scheduler, device,
            num_epochs=1,
            gradient_accumulation_steps=args.grad_accum,
            scaler=scaler, distributed=distributed, rank=rank,
            world_size=world, ema=ema, global_epoch_offset=epoch,
            seg_loss=seg_loss, ncut_loss=ncut_loss,
            ncut_weight=args.ncut_weight,
            rec_weight_fn=(None if args.rec_weight is None
                           else (lambda e, w=args.rec_weight: (w, 1.0))))

        rec = {"epoch": epoch, "train_loss": losses[-1] if losses else None,
               "lr": float(optimizer.param_groups[0]["lr"]),
               **{f"train_{k}": v[-1] for k, v in details.items() if v}}

        if (epoch + 1) % args.val_every == 0:
            acc = SegmentationMetricAccumulator(device, prob_threshold=0.5,
                                                valid_mask=valid_mask)
            n = validate_multilabel(model, val_loader, device, acc,
                                    use_amp=scaler is not None, rank=rank)
            acc.reduce(world)
            m = acc.compute()
            # Keep EVERY metric, including dice and the raw tp/fp/fn counts.
            # Counts are what let any other metric (or a confidence interval) be
            # recomputed afterwards without re-running the model — they cannot be
            # recovered from IoU alone, so not storing them is unrecoverable.
            # "_thresh" duplicates are dropped: under independent heads the
            # accumulator fills both variants identically (see metrics.py).
            rec.update({f"val_{k}": v for k, v in m.items()
                        if not k.endswith("_thresh")})
            if is_main:
                print(f"  [val] {n} batches | "
                      + " ".join(f"{c}: IoU {m[c+'_iou']:.4f} "
                                 f"P {m[c+'_precision']:.3f} R {m[c+'_recall']:.3f}"
                                 for c in ("ar", "hurricane", "etc", "mcs")))
            classes = ("ar", "hurricane", "etc", "mcs")
            mean_iou = float(np.mean([m[c + "_iou"] for c in classes]))
            rec["val_mean_iou"] = mean_iou

            # Model selection: mean skill RELATIVE to each class's trivial
            # single-threshold baseline, not raw mean IoU. The Phase 2 sweep
            # showed why — config B took the best raw mean (0.4909) while
            # collapsing hurricane from 0.237 to 0.079, because AR and MCS are
            # numerically large enough to hide the rarest class dying. Scoring
            # each class against what one global threshold already achieves puts
            # the classes on comparable footing.
            skill = float(np.mean([m[c + "_iou"] / baseline[c]
                                   for c in classes]))
            rec["val_skill_vs_baseline"] = skill
            rec["val_min_skill"] = float(min(m[c + "_iou"] / baseline[c]
                                             for c in classes))

            # Same pass over the training subset. Keys are train_<class>_<metric>
            # (train_ar_iou, ...), which do not collide with the per-class LOSS keys
            # train_ar / train_etc / ... written above. Model selection still uses
            # the validation numbers only.
            if train_eval_loader is not None:
                tacc = SegmentationMetricAccumulator(device, prob_threshold=0.5,
                                                     valid_mask=valid_mask)
                tn = validate_multilabel(model, train_eval_loader, device, tacc,
                                         use_amp=scaler is not None, rank=rank)
                tacc.reduce(world)
                tm = tacc.compute()
                rec.update({f"train_{k}": v for k, v in tm.items()
                            if not k.endswith("_thresh")})
                rec["train_mean_iou"] = float(np.mean([tm[c + "_iou"] for c in classes]))
                rec["train_skill_vs_baseline"] = float(np.mean(
                    [tm[c + "_iou"] / baseline[c] for c in classes]))
                rec["train_min_skill"] = float(min(tm[c + "_iou"] / baseline[c]
                                                   for c in classes))
                if is_main:
                    print(f"  [train-eval] {tn} batches | "
                          + " ".join(f"{c}: IoU {tm[c+'_iou']:.4f} "
                                     f"P {tm[c+'_precision']:.3f} R {tm[c+'_recall']:.3f}"
                                     for c in classes))

            if is_main and skill > best_skill:
                best_skill = skill
                best_mean_iou = mean_iou
                inner = model.module if hasattr(model, "module") else model
                torch.save({"model_state_dict": inner.state_dict(),
                            "epoch": epoch, "val_mean_iou": mean_iou,
                            "val_skill_vs_baseline": skill,
                            "discretize": args.discretize},
                           os.path.join(args.ckpt_dir, "best.pth"))
                print(f"  [ckpt] new best skill-vs-baseline {skill:.3f}x "
                      f"(mean IoU {mean_iou:.4f}) -> best.pth")

            if is_main and args.select_class:
              # Each selected class is checkpointed independently, on its own metric.
              for _c, _metric in zip(args.select_class, args.select_metric):
                _rec = float(m[_c + "_recall"])
                _prec = float(m[_c + "_precision"])
                # One independent selection per precision floor. With a single floor
                # this is exactly the original behaviour, writing best_<class>.pth.
                for _fl in args.select_min_precision:
                    if _metric == "recall":
                        _score = _rec if _prec >= _fl else -1.0
                    else:
                        # IoU selection ignores the floor entirely; with several
                        # floors every file would be identical, so only write one.
                        if _fl != args.select_min_precision[0]:
                            continue
                        _score = float(m[_c + "_iou"])
                    if _score <= best_class_iou.get((_c, _fl), -1.0):
                        continue
                    best_class_iou[(_c, _fl)] = _score
                    _name = (f"best_{_c}_p{int(round(_fl * 100)):02d}.pth"
                             if args.select_multi_floor else f"best_{_c}.pth")
                    inner = model.module if hasattr(model, "module") else model
                    torch.save({"model_state_dict": inner.state_dict(),
                                "epoch": epoch,
                                "select_class": _c,
                                "select_metric": _metric,
                                "select_min_precision": _fl,
                                f"val_{_c}_iou": float(m[_c + "_iou"]),
                                f"val_{_c}_recall": _rec,
                                f"val_{_c}_precision": _prec,
                                "val_mean_iou": mean_iou,
                                "val_skill_vs_baseline": skill,
                                "discretize": args.discretize},
                               os.path.join(args.ckpt_dir, _name))
                    print(f"  [ckpt] new best {_c} {_metric} "
                          f"{_score:.4f} at P>={_fl:.2f} "
                          f"(R {_rec:.3f} P {_prec:.3f}) -> {_name}")

        rec["epoch_seconds"] = float(time.time() - t0)
        history.append(rec)
        if is_main:
            print(f"  epoch {epoch+1}/{args.epochs} took {rec['epoch_seconds']:.0f}s")
            _persist()   # rewritten every epoch so a killed job keeps its curve
            # last.pth every epoch: these runs were previously unresumable, and a
            # walltime kill already destroyed the CNR/MRI RegCM runs once.
            _inner = model.module if hasattr(model, "module") else model
            _tmp = os.path.join(args.ckpt_dir, "last.pth.tmp")
            torch.save({"model_state_dict": _inner.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": (scaler.state_dict() if scaler else None),
                        "ema_state_dict": ema.state_dict(),
                        "epoch": epoch, "best_skill": best_skill,
                        "best_mean_iou": best_mean_iou,
                        "best_class_iou": best_class_iou, "history": history,
                        "discretize": args.discretize}, _tmp)
            os.replace(_tmp, os.path.join(args.ckpt_dir, "last.pth"))

    if is_main:
        inner = model.module if hasattr(model, "module") else model
        torch.save({"model_state_dict": inner.state_dict(),
                    "epoch": args.epochs - 1, "discretize": args.discretize},
                   os.path.join(args.ckpt_dir, "final.pth"))
        print(f"\nBest val mean IoU: {best_mean_iou:.4f}")
        print(f"Checkpoints in {args.ckpt_dir}")

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
