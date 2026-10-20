"""
train.py — Fixed for true distributed file-sharding training.

Key fixes vs. original:
  1. EMA is passed in from main (not re-created here) so it persists
     across monthly files and checkpoints correctly.
  2. Timing variables are always defined before the final print.
  3. _t_opt / _t_data reset logic is consistent for both scaler and
     non-scaler paths.
  4. No DistributedSampler needed here — files are already sharded.
"""

import numpy as np
import pandas as pd
import xarray as xr
import warnings
from warnings import filterwarnings
import os
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import time
from torch_ema import ExponentialMovingAverage

from loss import ARSegmentationLoss, StormSegLoss, clear_gpu_cache
from utils import print_memory_usage

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from itertools import cycle, islice


def _repeat(loader):
    """Endless pass over a DataLoader WITHOUT itertools.cycle's copy of every batch.

    cycle() stores each element of its first pass so it can replay them; for a DataLoader that pins
    every batch of the epoch in (shared) memory. Harmless at 192x384 (~3 GB per rank), fatal at
    0.25 deg (83 MB batches, ~34 GB per rank: /dev/shm exhausted at the end of epoch 1).
    """
    while True:
        for b in loader:
            yield b

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reduce_tensor(tensor, world_size, op=dist.ReduceOp.SUM):
    """Average a tensor across all DDP ranks."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    if op == dist.ReduceOp.SUM:
        rt /= world_size
    return rt


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_wnet_improved(
    model, train_loader, optimizer, scheduler, device,
    num_epochs=100,
    gradient_accumulation_steps=4,
    percentile_threshold=None,
    vorticity_threshold=None,
    use_step_scheduler=True,
    scaler=None,
    distributed=False,
    rank=0,
    world_size=1,
    ema=None,           # ← passed in from main so it persists across files
    file_idx=0,              # ← NEW: which file are we on (0-indexed)
    global_epoch_offset=0,   # ← NEW: total epochs already trained before this file
    split=False,             # ← Phase 1: loader yields (predictors, labels)
):
    """
    Training loop for one monthly file (called once per file per rank).

    Changes from original:
      - EMA accepted as argument (not re-created) so state accumulates
        correctly across all monthly files.
      - Timing variables always initialised before use.
      - _t_opt / _t_data reset is consistent in both AMP and non-AMP paths.
    """
    model.train()
    losses       = []
    loss_details = {"ar": [], "hurricane": [], "etc": [],
                    "mcs": [], "overlap": [], "total": []}

    is_distributed  = distributed and world_size > 1
    is_main_process = rank == 0

    # Loss functions
    # split=True  -> StormSegLoss reads targets from a separate label tensor, so
    #                the model cannot satisfy it by copying an input channel.
    # split=False -> legacy ARSegmentationLoss, which pulls its targets out of
    #                the input itself (the Phase 0 shortcut).
    ar_loss  = (StormSegLoss() if split else ARSegmentationLoss()).to(device)
    rec_loss = nn.MSELoss().to(device)

    # Create EMA locally only if not provided (single-file / non-persistent use)
    _local_ema = ema is None
    if _local_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    warmup_epochs = 5  # used only by _dynamic_weights


    if scaler is None and torch.cuda.is_available():
        scaler = GradScaler()

    if is_main_process:
        print(f"  train_wnet: device={device}, AMP={scaler is not None}, "
              f"distributed={is_distributed}, world_size={world_size}, "
              f"grad_accum={gradient_accumulation_steps}, epochs={num_epochs}")

    start_time    = time.time()
    total_batches = 0

    # ── Sync batch count across ranks so DDP doesn't deadlock ──────────
    # Different months have different timesteps (112-124 samples),
    # so different ranks may have different batch counts.
    # DDP requires all ranks to call forward/backward the same number of times.
    if is_distributed:
        my_batches = torch.tensor(len(train_loader), dtype=torch.int32, device=device)
        dist.all_reduce(my_batches, op=dist.ReduceOp.MAX)
        max_batches = int(my_batches.item())

        # Safety clamp — if something went wrong, cap at a sane value
        if max_batches > 10000 or max_batches <= 0:
            print(f"  [Rank {rank}] WARNING: max_batches={max_batches} is insane, "
                f"clamping to local {len(train_loader)}")
            max_batches = len(train_loader)

        if is_main_process:
            print(f"  DDP batch sync: this rank has {len(train_loader)} batches, "
                  f"using max={max_batches} across all ranks")
    else:
        max_batches = len(train_loader)

    try:
        for epoch in range(num_epochs):
            # ── Per-epoch timing accumulators (always initialised) ────────────
            _epoch_data_time     = 0.0
            _epoch_forward_time  = 0.0
            _epoch_backward_time = 0.0
            _epoch_optim_time    = 0.0
            _t_data = time.time()

            epoch_start_time   = time.time()
            epoch_loss         = 0.0
            epoch_ar_loss      = 0.0
            epoch_hurricane_loss = 0.0
            epoch_etc_loss     = 0.0
            epoch_mcs_loss     = 0.0
            epoch_overlap_loss = 0.0
            batch_count        = 0

            # No DistributedSampler → no set_epoch needed
            optimizer.zero_grad()

            # REPLACE WITH this one line:
            for batch_idx, batch in enumerate(islice(_repeat(train_loader), max_batches)):

                _epoch_data_time += time.time() - _t_data
                _t_fwd = time.time()

                # Periodic GPU cache clear (every 10 batches)
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

                # Split contract: the loader yields (predictors, labels). `x` is
                # all the model ever sees; `y` goes only to the loss.
                if split:
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                else:
                    x = batch.to(device, non_blocking=True)
                    y = x            # legacy: targets live inside the input
                batch_start_time = time.time()

                # ── Forward + loss ────────────────────────────────────────────
                if scaler is not None:
                    with autocast():
                        seg_probs, rec = model(x)
                        _epoch_forward_time += time.time() - _t_fwd
                        _t_bwd = time.time()

                        loss_dict          = ar_loss(seg_probs, y)
                        segmentation_loss  = loss_dict["total_loss"]
                        # Reconstruct the PHYSICAL fields. In split mode this is a
                        # real bottleneck objective; in legacy mode 13 of the 14
                        # channels were masks the decoder could pass straight through.
                        reconstruction_loss = rec_loss(rec, x)

                        rec_weight, seg_weight = _dynamic_weights(epoch, warmup_epochs, global_epoch_offset)

                        total_loss = (
                            seg_weight * segmentation_loss +
                            rec_weight * reconstruction_loss
                        ) / gradient_accumulation_steps

                    scaler.scale(total_loss).backward()
                    _epoch_backward_time += time.time() - _t_bwd
                    _t_opt = time.time()

                else:
                    seg_probs, rec = model(x)
                    _epoch_forward_time += time.time() - _t_fwd
                    _t_bwd = time.time()

                    loss_dict           = ar_loss(seg_probs, y)
                    segmentation_loss   = loss_dict["total_loss"]
                    reconstruction_loss = rec_loss(rec, x)

                    rec_weight, seg_weight = _dynamic_weights(epoch, warmup_epochs, global_epoch_offset)
                    total_loss = (
                        seg_weight * segmentation_loss +
                        rec_weight * reconstruction_loss
                    ) / gradient_accumulation_steps

                    total_loss.backward()
                    _epoch_backward_time += time.time() - _t_bwd
                    _t_opt = time.time()

                # ── Gradient accumulation step ────────────────────────────────
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    _epoch_optim_time += time.time() - _t_opt
                    ema.update()
                    optimizer.zero_grad()

                _t_data = time.time()   # reset for next batch's data-load timing

                # ── Logging ───────────────────────────────────────────────────
                seg_loss_val   = float(segmentation_loss.detach().cpu())
                rec_loss_val   = float(reconstruction_loss.detach().cpu())
                total_loss_val = float(total_loss.detach().cpu()) * gradient_accumulation_steps

                ar_loss_val        = float(loss_dict["ar_loss"].cpu())
                hurricane_loss_val = float(loss_dict["hurricane_loss"].cpu())
                etc_loss_val       = float(loss_dict["etc_loss"].cpu())
                mcs_loss_val       = float(loss_dict["mcs_loss"].cpu())
                overlap_loss_val   = float(loss_dict["overlap_loss"].cpu())

                epoch_loss         += total_loss_val
                epoch_ar_loss      += ar_loss_val
                epoch_hurricane_loss += hurricane_loss_val
                epoch_etc_loss     += etc_loss_val
                epoch_mcs_loss     += mcs_loss_val
                epoch_overlap_loss += overlap_loss_val
                batch_count        += 1
                total_batches      += 1

                if is_main_process and batch_idx % 10 == 0:
                    current_lr  = optimizer.param_groups[0]["lr"]
                    batch_time  = time.time() - batch_start_time
                    print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss {total_loss_val:.4f} | Seg {seg_loss_val:.4f} | "
                          f"Rec {rec_loss_val:.4f} | LR {current_lr:.2e} | "
                          f"t {batch_time:.2f}s")
                    print(f"    AR:{ar_loss_val:.4f} Hurr:{hurricane_loss_val:.4f} "
                          f"ETC:{etc_loss_val:.4f} MCS:{mcs_loss_val:.4f} "
                          f"Overlap:{overlap_loss_val:.4f}")
                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                        mem_res   = torch.cuda.memory_reserved(device)  / 1024**3
                        print(f"    GPU mem: {mem_alloc:.2f}GB alloc / {mem_res:.2f}GB reserved")
                    print_memory_usage()
                    clear_gpu_cache()

                del batch, x, y, seg_probs, rec, segmentation_loss, reconstruction_loss, total_loss

            # ── End of epoch ──────────────────────────────────────────────────
            n = batch_count if batch_count > 0 else 1
            avg_epoch_loss      = epoch_loss         / n
            avg_ar_loss         = epoch_ar_loss      / n
            avg_hurricane_loss  = epoch_hurricane_loss / n
            avg_etc_loss        = epoch_etc_loss     / n
            avg_mcs_loss        = epoch_mcs_loss     / n
            avg_overlap_loss    = epoch_overlap_loss / n

            # Reduce losses across all DDP ranks
            if is_distributed:
                loss_tensor = torch.tensor(
                    [avg_epoch_loss, avg_ar_loss, avg_hurricane_loss,
                     avg_etc_loss, avg_mcs_loss, avg_overlap_loss],
                    device=device
                )
                loss_tensor     = reduce_tensor(loss_tensor, world_size)
                avg_epoch_loss  = float(loss_tensor[0].cpu())
                avg_ar_loss     = float(loss_tensor[1].cpu())
                avg_hurricane_loss = float(loss_tensor[2].cpu())
                avg_etc_loss    = float(loss_tensor[3].cpu())
                avg_mcs_loss    = float(loss_tensor[4].cpu())
                avg_overlap_loss = float(loss_tensor[5].cpu())

            losses.append(avg_epoch_loss)
            loss_details["ar"].append(avg_ar_loss)
            loss_details["hurricane"].append(avg_hurricane_loss)
            loss_details["etc"].append(avg_etc_loss)
            loss_details["mcs"].append(avg_mcs_loss)
            loss_details["overlap"].append(avg_overlap_loss)
            loss_details["total"].append(avg_epoch_loss)

            if scheduler:
                scheduler.step()

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            if is_main_process:
                print(f"  Epoch {epoch+1}/{num_epochs} done | "
                      f"avg_loss={avg_epoch_loss:.6f} | "
                      f"epoch_time={epoch_time:.1f}s | total={total_time:.1f}s")
                print(f"    AR:{avg_ar_loss:.6f} Hurr:{avg_hurricane_loss:.6f} "
                      f"ETC:{avg_etc_loss:.6f} MCS:{avg_mcs_loss:.6f} "
                      f"Overlap:{avg_overlap_loss:.6f}")
                if epoch > 0:
                    remaining = (total_time / (epoch + 1)) * (num_epochs - epoch - 1)
                    print(f"    ETA: {remaining:.0f}s ({remaining/3600:.2f}h)")
                print(f"  [TIMING-Train] Epoch {epoch+1} — "
                      f"Data:{_epoch_data_time:.1f}s | "
                      f"Fwd:{_epoch_forward_time:.1f}s | "
                      f"Bwd:{_epoch_backward_time:.1f}s | "
                      f"Opt:{_epoch_optim_time:.1f}s")

            # Checkpoint at the last epoch of each file
            if is_main_process and (epoch + 1) == num_epochs:
                save_checkpoint(model, optimizer, scheduler, scaler, ema,
                                global_epoch_offset + epoch, avg_epoch_loss, is_distributed)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if is_distributed:
                dist.barrier()

    except Exception as e:
        if is_main_process:
            print(f"  Training interrupted: {e}")
            save_emergency_checkpoint(
                model, optimizer, scheduler, scaler, ema,
                epoch if "epoch" in dir() else 0,
                avg_epoch_loss if "avg_epoch_loss" in dir() else None,
                is_distributed,
            )
        raise

    if is_main_process:
        total_training_time = time.time() - start_time
        print(f"  Training done in {total_training_time:.0f}s "
              f"({total_training_time/3600:.2f}h), "
              f"{total_training_time/num_epochs:.1f}s/epoch avg")

    return losses, loss_details


# ─────────────────────────────────────────────────────────────────────────────
# Forward-only, seg-head-only validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_on_loader(
    model,
    val_loader,
    device,
    accumulator,
    use_amp: bool = True,
    rank: int = 0,
    split: bool = False,
):
    """
    Forward-only validation pass — runs only UNet1 (segmentation head),
    skipping UNet2 (reconstruction). Updates `accumulator` in place with
    per-class TP/FP/FN counts. Returns number of batches processed.

    DDP-safe: no gradient ops, no DDP collectives during forward — the
    only cross-rank communication is the SUM all_reduce performed by
    `accumulator.reduce()` outside this function.
    """
    inner   = model.module if hasattr(model, "module") else model
    seg_net = inner.unet1   # segmentation head only — skip reconstruction UNet2

    was_training = model.training
    model.eval()

    n_batches = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                if split:
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                else:
                    x = batch.to(device, non_blocking=True)
                    y = x
                if use_amp and torch.cuda.is_available():
                    with autocast():
                        seg_logits = seg_net(x)
                        seg_probs  = F.softmax(seg_logits, dim=1)
                else:
                    seg_logits = seg_net(x)
                    seg_probs  = F.softmax(seg_logits, dim=1)

                accumulator.update(seg_probs, y, split=split)
                n_batches += 1

                del batch, x, y, seg_logits, seg_probs
    finally:
        if was_training:
            model.train()

    return n_batches


def _dynamic_weights(epoch: int, warmup_epochs: int, global_epoch_offset: int = 0):
    """Return (rec_weight, seg_weight) based on GLOBAL epoch position."""
    global_epoch = epoch + global_epoch_offset
    if global_epoch < warmup_epochs:
        return 0.8, 0.2
    rec_weight = max(0.3, 0.8 - 0.05 * (global_epoch - warmup_epochs))
    return rec_weight, 1.0 - rec_weight

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                  "Storms_precipitation_notebooks/WNet_parallel_global_code/"
                  "ARs_Hurricanes_ETCs_MCS/Trail11")


def _model_state(model, is_distributed):
    if is_distributed:
        return model.module.state_dict()
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_checkpoint(model, optimizer, scheduler, scaler, ema, epoch, loss, is_distributed):
    try:
        ema.store()
        ema.copy_to()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"ema_checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     _model_state(model, is_distributed),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "loss":                 loss,
            "ema_state_dict":       ema.state_dict(),
        }, path)
        ema.restore()
        print(f"  Checkpoint saved → {path}")
    except Exception as e:
        print(f"  Checkpoint save failed: {e}")
        ema.restore()


def save_emergency_checkpoint(model, optimizer, scheduler, scaler, ema,
                               epoch, loss, is_distributed):
    try:
        ema.store()
        ema.copy_to()
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     _model_state(model, is_distributed),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "loss":                 loss,
            "ema_state_dict":       ema.state_dict(),
        }, "emergency_checkpoint_gpu.pth")
        print("  Emergency checkpoint saved.")
        ema.restore()
    except Exception as e:
        print(f"  Emergency checkpoint failed: {e}")
        ema.restore()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def calculate_model_flops(model, input_size=(1, 14, 192, 384)):
    try:
        from thop import profile
        dummy = torch.randn(input_size)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops, params
    except ImportError:
        return None, None


def print_training_stats(model, train_loader, device):
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_p:,}")
    print(f"Trainable params : {trainable_p:,}")
    print(f"Model size       : {total_p * 4 / 1024**2:.2f} MB")
    print(f"Batches/epoch    : {len(train_loader)}")
    print(f"Batch size       : {train_loader.batch_size}")
    print(f"Samples/epoch    : {len(train_loader.dataset)}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU memory       : {gpu_mem:.2f} GB")
    flops, _ = calculate_model_flops(model)
    if flops:
        print(f"FLOPs            : {flops/1e9:.2f} GFLOPs")
    print("="*60 + "\n")



def run_zombie_training(model, optimizer, scheduler, device, num_epochs,
                        gradient_accumulation_steps, world_size, rank):
    """
    Called by ranks that have no more files to process.
    Participates in every collective op that train_wnet_improved makes
    so DDP does not hang. Contributes zero gradients — weights unchanged.

    Must mirror train_wnet_improved's collective call pattern EXACTLY:
      1. one all_reduce  → max_batches sync
      2. per grad-accum step → DDP backward (implicit all_reduce on grads)
      3. per epoch → one all_reduce for loss aggregation
      4. per epoch → scheduler.step() to keep LR in sync
      5. per epoch → one dist.barrier()
    """
    # ── 1. Mirror: max_batches all_reduce ─────────────────────────────────────
    my_batches = torch.tensor(0, dtype=torch.int32, device=device)
    dist.all_reduce(my_batches, op=dist.ReduceOp.MAX)
    max_batches = int(my_batches.item())

    print(f"  [Rank {rank} ZOMBIE] {num_epochs} epochs × {max_batches} batches — no real data")

    # Dummy input matching ImprovedWNet: (B=1, C=14, H=192, W=384)
    dummy_batch = torch.zeros(1, 14, 192, 384, device=device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # ── 2. Mirror: per-batch DDP grad sync ────────────────────────────────
        for batch_idx in range(max_batches):
            seg_probs, rec = model(dummy_batch)
            # Loss is always exactly zero → grads are zero → weights never change
            dummy_loss = seg_probs.sum() * 0.0 + rec.sum() * 0.0
            (dummy_loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()   # steps with all-zero grads → no weight change
                optimizer.zero_grad()

        # ── 3. Mirror: epoch-end loss all_reduce ──────────────────────────────
        loss_tensor = torch.zeros(6, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        # ── 4. Mirror: scheduler step to keep LR in sync across ranks ─────────
        if scheduler:
            scheduler.step()

        # ── 5. Mirror: epoch-end barrier ──────────────────────────────────────
        dist.barrier()

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — multi-label training
# ═════════════════════════════════════════════════════════════════════════════

# Held-out years. Consecutive 6-hourly frames are near-duplicates, so a random
# frame split leaks the test set into training and every score comes out
# flattering. Splitting by YEAR is the cheapest honest option. These are spread
# across the record rather than taken from one end, so neither split is a single
# climate epoch, and they include both ENSO phases.
VAL_YEARS  = (1987, 1998, 2007, 2016)
TEST_YEARS = (1983, 1994, 2003, 2012, 2021)


def split_years(start_year=1980, end_year=2024):
    """Return (train_years, val_years, test_years) as sorted lists."""
    allx = list(range(start_year, end_year + 1))
    val = [y for y in allx if y in VAL_YEARS]
    test = [y for y in allx if y in TEST_YEARS]
    train = [y for y in allx if y not in VAL_YEARS and y not in TEST_YEARS]
    return train, val, test


def year_of_file(name):
    """Extract YYYY from era5dara_YYYYMM.nc / split_YYYYMM.npy style names."""
    import re
    m = re.search(r"(\d{4})(\d{2})", os.path.basename(str(name)))
    return int(m.group(1)) if m else None


def filter_files_by_years(files, years):
    years = set(years)
    return [f for f in files if year_of_file(f) in years]


def train_wnet_multilabel(
    model, train_loader, optimizer, scheduler, device,
    num_epochs=3,
    gradient_accumulation_steps=4,
    scaler=None,
    distributed=False,
    rank=0,
    world_size=1,
    ema=None,
    global_epoch_offset=0,
    seg_loss=None,
    ncut_loss=None,
    ncut_weight=0.1,
    rec_weight_fn=None,
):
    """Training loop for WNetMultiLabel on the Phase 1 split contract.

    Differences from train_wnet_improved that made a separate loop cleaner than
    another flag: the model returns (probs, rec, logits) rather than
    (seg_probs, rec); the loss consumes LOGITS not probabilities; and there is an
    N-cut term which needs the predictors as well as the predictions.

    total = seg_weight * segmentation + rec_weight * reconstruction
            + ncut_weight * ncut
    """
    from loss import StormSegLossV2, NCutLoss2DFixed

    model.train()
    is_distributed = distributed and world_size > 1
    is_main = rank == 0

    seg_loss = (seg_loss or StormSegLossV2()).to(device)
    ncut_loss = ncut_loss if ncut_loss is not None else NCutLoss2DFixed().to(device)
    rec_criterion = nn.MSELoss().to(device)
    rec_weight_fn = rec_weight_fn or (lambda e: _dynamic_weights(e, 5, global_epoch_offset))

    if scaler is None and torch.cuda.is_available():
        scaler = GradScaler()

    _local_ema = ema is None
    if _local_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    if is_distributed:
        nb = torch.tensor(len(train_loader), dtype=torch.int32, device=device)
        dist.all_reduce(nb, op=dist.ReduceOp.MAX)
        max_batches = int(nb.item())
        if max_batches > 10000 or max_batches <= 0:
            max_batches = len(train_loader)
    else:
        max_batches = len(train_loader)

    losses = []
    details = {"ar": [], "hurricane": [], "etc": [], "mcs": [],
               "ncut": [], "rec": [], "total": []}

    for epoch in range(num_epochs):
        acc = {k: 0.0 for k in ("total", "ar", "hurricane", "etc", "mcs", "ncut", "rec")}
        n_batches = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        for batch_idx, batch in enumerate(islice(_repeat(train_loader), max_batches)):
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            rec_w, seg_w = rec_weight_fn(epoch)

            if scaler is not None:
                with autocast():
                    probs, rec, logits = model(x)
                    ld = seg_loss(logits, y)
                    seg_term = ld["total_loss"]
                    rec_term = rec_criterion(rec, x)
                    ncut_term = (ncut_loss(probs, x) if ncut_weight > 0
                                 else probs.new_zeros(()))
                    total = (seg_w * seg_term + rec_w * rec_term
                             + ncut_weight * ncut_term) / gradient_accumulation_steps
                scaler.scale(total).backward()
            else:
                probs, rec, logits = model(x)
                ld = seg_loss(logits, y)
                seg_term = ld["total_loss"]
                rec_term = rec_criterion(rec, x)
                ncut_term = (ncut_loss(probs, x) if ncut_weight > 0
                             else probs.new_zeros(()))
                total = (seg_w * seg_term + rec_w * rec_term
                         + ncut_weight * ncut_term) / gradient_accumulation_steps
                total.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                ema.update()
                optimizer.zero_grad()

            acc["total"] += float(total.detach().cpu()) * gradient_accumulation_steps
            acc["rec"] += float(rec_term.detach().cpu())
            acc["ncut"] += float(ncut_term.detach().cpu())
            for c in ("ar", "hurricane", "etc", "mcs"):
                acc[c] += float(ld[f"{c}_loss"].cpu())
            n_batches += 1

            if is_main and batch_idx % 20 == 0:
                print(f"  ep {epoch+1}/{num_epochs} b {batch_idx}/{max_batches} | "
                      f"tot {acc['total']/max(n_batches,1):.4f} | "
                      f"seg {float(seg_term.detach().cpu()):.4f} | "
                      f"rec {float(rec_term.detach().cpu()):.4f} | "
                      f"ncut {float(ncut_term.detach().cpu()):.4f} | "
                      f"w(seg {seg_w:.2f} rec {rec_w:.2f})")

            del batch, x, y, probs, rec, logits, seg_term, rec_term, ncut_term, total

        n = max(n_batches, 1)
        avg = {k: v / n for k, v in acc.items()}

        if is_distributed:
            t = torch.tensor([avg[k] for k in ("total", "ar", "hurricane",
                                               "etc", "mcs", "ncut", "rec")],
                             device=device)
            t = reduce_tensor(t, world_size)
            for i, k in enumerate(("total", "ar", "hurricane", "etc",
                                   "mcs", "ncut", "rec")):
                avg[k] = float(t[i].cpu())

        losses.append(avg["total"])
        for k in details:
            details[k].append(avg[k])

        if scheduler:
            scheduler.step()

        if is_main:
            print(f"  Epoch {epoch+1}/{num_epochs} done | total={avg['total']:.6f} "
                  f"| AR {avg['ar']:.4f} Hurr {avg['hurricane']:.4f} "
                  f"ETC {avg['etc']:.4f} MCS {avg['mcs']:.4f} "
                  f"| rec {avg['rec']:.4f} ncut {avg['ncut']:.4f} "
                  f"| {time.time()-t_epoch:.0f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_distributed:
            dist.barrier()

    return losses, details


@torch.no_grad()
def validate_multilabel(model, val_loader, device, accumulator,
                        use_amp=True, rank=0):
    """Forward-only validation for WNetMultiLabel — segmentation head only."""
    inner = model.module if hasattr(model, "module") else model
    was_training = model.training
    model.eval()
    n = 0
    try:
        for batch in val_loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp and torch.cuda.is_available():
                with autocast():
                    _, probs = inner.encode(x)
            else:
                _, probs = inner.encode(x)
            accumulator.update(probs.float(), y, multilabel=True)
            n += 1
            del batch, x, y, probs
    finally:
        if was_training:
            model.train()
    return n


def load_init_weights(model, init_path, device=None, allow_partial=False,
                      verbose=True):
    """Initialise `model` from a checkpoint, reporting what actually matched.

    Transfer, not resume: model weights only, no optimizer/scheduler/EMA/epoch.

    WHY THIS RECONCILES KEYS INSTEAD OF JUST CALLING load_state_dict(strict=True)
    ---------------------------------------------------------------------------
    The RegCM loader printed only `len(state)` and relied on strict=True to catch
    problems. That is not a verification. `ImprovedWNet(14, 5)` and
    `WNetMultiLabel(6, 4)` BOTH have exactly 228 tensors and identical key names —
    only 6 of the 228 differ in shape — so a tensor count is the same number for
    the right architecture and the wrong one. A silent partial load is likewise
    indistinguishable from a successful fine-tune unless the keys are compared.

    Returns a dict recorded into the run config so history.json proves which arm
    produced a checkpoint.
    """
    import torch as _t

    ckpt = _t.load(init_path, map_location=device or "cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Checkpoints written from a DDP-wrapped model carry a "module." prefix.
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}

    tgt = model.state_dict()
    matched, shape_mismatch = [], []
    for k, v in state.items():
        if k in tgt:
            (matched if tuple(tgt[k].shape) == tuple(v.shape)
             else shape_mismatch).append(k)
    missing = [k for k in tgt if k not in state]
    unexpected = [k for k in state if k not in tgt]

    loadable = {k: state[k] for k in matched}
    model.load_state_dict(loadable, strict=False)

    report = {
        "init_weights": str(init_path),
        "init_matched": len(matched),
        "init_total_model_tensors": len(tgt),
        "init_shape_mismatch": len(shape_mismatch),
        "init_missing": len(missing),
        "init_unexpected": len(unexpected),
        "init_src_epoch": (ckpt.get("epoch") if isinstance(ckpt, dict) else None),
        "init_src_skill": (ckpt.get("val_skill_vs_baseline")
                           if isinstance(ckpt, dict) else None),
    }

    if verbose:
        print(f"  [init-weights] {init_path}")
        print(f"    matched        : {len(matched)}/{len(tgt)}")
        print(f"    shape mismatch : {len(shape_mismatch)}  {shape_mismatch[:10]}")
        print(f"    missing        : {len(missing)}  {missing[:10]}")
        print(f"    unexpected     : {len(unexpected)}  {unexpected[:10]}")
        print(f"    source epoch   : {report['init_src_epoch']}  "
              f"skill {report['init_src_skill']}")

    if not matched:
        raise SystemExit(
            f"--init-weights {init_path}: ZERO tensors matched. The checkpoint does "
            f"not correspond to this architecture; refusing to 'fine-tune' from a "
            f"random initialisation while reporting success.")
    # ANY shape mismatch is fatal, and a match FRACTION is not a sufficient guard.
    # Measured: loading the ERA5 WNetMultiLabel(6,4) checkpoint into a (14,5) model
    # matches 222/228 = 97.4% — comfortably past any sane percentage gate — while
    # the 6 skipped tensors are exactly unet1.enc1.double_conv.0.weight,
    # unet1.dec1.{weight,bias} and their unet2 counterparts: the input projection
    # and the output heads. Those would stay randomly initialised and the run would
    # still look like a successful fine-tune. Mismatched shapes mean the
    # architectures differ; refuse unless that is stated explicitly.
    if shape_mismatch and not allow_partial:
        raise SystemExit(
            f"--init-weights {init_path}: {len(shape_mismatch)} tensor(s) have the "
            f"right NAME but the wrong SHAPE, so the architectures differ:\n    "
            + "\n    ".join(f"{k}: ckpt {tuple(state[k].shape)} vs model "
                            f"{tuple(tgt[k].shape)}" for k in shape_mismatch[:8])
            + "\n  These are typically the input projection and the output heads — "
              "loading around them leaves exactly the layers that define the task "
              "randomly initialised. Pass --allow-partial-init only if deliberate.")

    frac = len(matched) / max(len(tgt), 1)
    if frac < 0.95 and not allow_partial:
        raise SystemExit(
            f"--init-weights {init_path}: only {len(matched)}/{len(tgt)} tensors "
            f"({100*frac:.1f}%) matched. Pass --allow-partial-init if that is "
            f"deliberate.")
    return report
