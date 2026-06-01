"""
main.py — True multi-node / multi-GPU distributed training on Frontier.

FILE-LEVEL SHARDING STRATEGY
─────────────────────────────
  Old (wrong):  every GPU loads the SAME monthly file → DistributedSampler
                splits samples within that file.  8× redundant I/O.

  New (correct): round-robin assigns disjoint monthly files to each rank.
                 rank 0 → months [0, W, 2W, …]
                 rank 1 → months [1, W+1, 2W+1, …]
                 DDP all_reduce keeps weights in sync after every step.
                 No DistributedSampler needed — each rank owns its data.

SCALING GUIDE (Frontier MI250X, 64 GB HBM per GCD)
────────────────────────────────────────────────────
  ~3.3 GB/file on disk → ~640 MB in memory after precompute.
  300 monthly files (1980-2005 70%) ÷ world_size GPUs:

  world_size │ nodes │ files/GPU │ effective batch
  ───────────┼───────┼───────────┼────────────────
       8     │   1   │    ~37    │  2×8×3  =  48
      16     │   2   │    ~19    │  2×16×3 =  96
      32     │   4   │    ~10    │  2×32×3 = 192
      64     │   8   │    ~5     │  2×64×3 = 384

  Recommended: start with 2 nodes (16 GPUs), scale to 4 if wall-time allows.
"""

# ── MIOpen: skip kernel search, use defaults instantly ────────────────────────
# Must happen before 'import torch' to take effect
import os
_rank = int(os.environ.get("SLURM_PROCID", 0))
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)

os.environ["MIOPEN_USER_DB_PATH"]     = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"]        = "2"   # FAST: no search, use defaults
os.environ["MIOPEN_DISABLE_CACHE"]    = "0"
# ─────────────────────────────────────────────────────────────────────────────


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import warnings
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from scipy.ndimage import grey_opening
import dask.array as da
import gc
import psutil
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import GPUtil
from torch.nn import functional as F
from scipy import ndimage
from torch_ema import ExponentialMovingAverage
from skimage.measure import regionprops
import math
import glob
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from WNet_model import ImprovedWNet
from dataset import ARDatasetImproved, MemoryEfficientDataLoader, preprocess_data_improved
from train import train_wnet_improved, run_zombie_training, validate_on_loader
from loss import DiceLoss, FocalLoss, NCutLoss2D, ARSegmentationLoss
from metrics import SegmentationMetricAccumulator, format_metrics
from utils import (setup_gpu, calculate_ivt, print_system_info,
                   find_matching_vorticity, find_latest_checkpoint,
                   load_checkpoint, load_percentile_dataset)
from visualization import visualize_four_class_segmentation, plot_training_history
from loss import optimize_batch_size_for_gpu
from cache_loader import CacheAwareDataLoader, get_cache_info
from datetime import timedelta
import argparse
import pickle
from ibtracs_loader import load_ibtracs_for_training
from pympler import asizeof

# ── File-level sharding helpers ───────────────────────────────────────────────
from distributed_dataset import (
    get_rank_files,
    log_file_assignment,
    build_train_val_split,
    log_train_val_split,
)

# ── MPI (optional but more robust on Frontier) ────────────────────────────────
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except Exception as e:
    MPI_AVAILABLE = False
    print(f"Warning: mpi4py unavailable ({e}). Using SLURM env vars.")


# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    """
    Setup distributed training for ROCm/AMD GPUs on Frontier.

    Supports both MPI-based and SLURM env-var based initialisation.
    Works for 1 node (8 GPUs) up to N nodes (8N GPUs).
    """
    def _select_device(rank, local_rank, gpu_count):
        visible_devices = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            device_id = 0
            reason = "ROCR_VISIBLE_DEVICES is set"
        else:
            if gpu_count >= 8 and local_rank < 4:
                device_id = (local_rank * 2) % gpu_count
                reason = "Frontier 4-GPU map (0,2,4,6)"
            else:
                device_id = int(local_rank) % int(gpu_count)
                reason = "local_rank % gpu_count"
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        print(f"Rank {rank}: Using GPU {device_id} ({reason})")
        return device_id, device

    # ── Method 1: MPI ────────────────────────────────────────────────────────
    if MPI_AVAILABLE:
        try:
            comm       = MPI.COMM_WORLD
            world_size = comm.Get_size()
            rank       = comm.Get_rank()
            gpu_count  = torch.cuda.device_count()

            slurm_localid = os.environ.get("SLURM_LOCALID")
            if slurm_localid is not None:
                local_rank = int(slurm_localid)
            else:
                tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", world_size))
                local_rank = int(rank) % max(1, tasks_per_node)

            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "hsn0")

            os.environ.update({
                "WORLD_SIZE": str(world_size), "RANK": str(rank),
                "LOCAL_RANK": str(local_rank), "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,    "NCCL_SOCKET_IFNAME": nccl_ifname,
            })

            if gpu_count == 0 or not torch.cuda.is_available():
                raise RuntimeError(f"Rank {rank}: No GPUs detected.")

            device_id, device = _select_device(rank, local_rank, gpu_count)

            dist.init_process_group(
                backend="nccl", init_method="env://",
                world_size=world_size, rank=rank,
                timeout=timedelta(hours=4),
                device_id=torch.device(f"cuda:{device_id}")  
            )
            print(f"Rank {rank}/{world_size}: Ready on {device} (MPI init)")
            return True, world_size, rank, local_rank, device

        except Exception as e:
            print(f"MPI init failed: {e}. Falling back to SLURM env vars.")

    # ── Method 2: SLURM env vars ─────────────────────────────────────────────
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env       = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if not world_size_env and "SLURM_NTASKS" in os.environ:
        world_size_env = os.environ.get("SLURM_NTASKS")
        rank_env       = os.environ.get("SLURM_PROCID")
        local_rank_env = os.environ.get("SLURM_LOCALID")

    if world_size_env and rank_env is not None and local_rank_env is not None:
        world_size = int(world_size_env)
        rank       = int(rank_env)
        local_rank = int(local_rank_env)

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "hsn0")

        os.environ.update({
            "WORLD_SIZE": str(world_size), "RANK": str(rank),
            "LOCAL_RANK": str(local_rank), "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,    "NCCL_SOCKET_IFNAME": nccl_ifname,
        })

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0 or not torch.cuda.is_available():
            raise RuntimeError(f"Rank {rank}: No GPUs detected.")

        device_id, device = _select_device(rank, local_rank, gpu_count)

        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank,
            timeout=timedelta(seconds=1800)
        )
        print(f"Rank {rank}/{world_size}: Ready on {device} (SLURM env init)")
        return True, world_size, rank, local_rank, device

    # ── Single-GPU fallback ───────────────────────────────────────────────────
    print("No distributed env detected — running in single-GPU mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 1, 0, 0, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Global hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

batch_size                  = 2
gradient_accumulation_steps = 3
chunk_size                  = 12
target_size                 = (192, 384)
epochs_per_file             = 3
num_passes                  = 20
in_channels                 = 14

model        = None
dataset      = None
train_loader = None
optimizer    = None
scheduler    = None
losses       = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: process one monthly file  (extracted from the old monolithic loop)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_file(
    nc_file, file_idx, total_files,
    data_dir, cache_loader, vorticity_files, wind_speed_full,
    temp_500mb_dir, windshear_dir, Bright_temp,
    vor850_results_per, windspeed_results_per, mslp_anom_result_glob,
    ibtracs_lookup, interp_percentile,
    model, optimizer, scheduler, scaler, ema,
    device, distributed, world_size, rank,
    start_epoch, loss_history,
    global_epoch_offset=0,
    epochs_per_file=3,
    mode="train",
    val_accumulator=None,
):
    """
    Load one monthly file, build the dataset/loader, then either:
      * mode="train" (default): run train_wnet_improved → returns (epoch_losses, loss_details)
      * mode="val": forward-only seg-only pass that updates `val_accumulator`
                    in place → returns (None, None)

    Because files are already sharded across ranks, NO DistributedSampler
    is used — each rank owns its data exclusively. DDP gradient sync happens
    inside train_wnet_improved via all_reduce. In val mode no DDP collectives
    are made by the forward pass; the only cross-rank sync is the SUM
    all_reduce performed by accumulator.reduce() outside this function.
    """
    is_val = (mode == "val")
    timing = {}
    _t = time.time()

    if rank == 0:
        tag = "VAL" if is_val else "TRAIN"
        print(f"\n{'─'*60}")
        print(f"[Rank {rank}] [{tag}] File {file_idx+1}/{total_files}: {nc_file}")

    try:
        # ── Extract year / month ──────────────────────────────────────────────
        date_str = nc_file.split("_")[1].split(".")[0]   # "era5dara_198003.nc" → "198003"
        year     = int(date_str[:4])
        month    = int(date_str[4:6])

        # ── Auxiliary data ────────────────────────────────────────────────────
        # IMPORTANT: aux loaders now LOG LOUDLY on missing-file or open-exception
        # (was: silent fallback to None → silent zero-fill in dataset → degenerate
        #  target masks → catastrophic-forgetting on otherwise-real training files).
        # Every failure prints rank + file + error so concurrent-access races
        # (e.g. multiple ranks opening the same yearly BT NetCDF) become visible.
        date_str_temp = f"{year:04d}{month:02d}"
        tag = "VAL" if is_val else "TRAIN"

        # 500 mb temperature
        temp_500mb_data = None
        temp_500mb_file = os.path.join(temp_500mb_dir, f"ERA5_t_500mb_{date_str_temp}.nc")
        if os.path.exists(temp_500mb_file):
            try:
                temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level="500")
                temp_500mb_ds = temp_500mb_ds.assign_coords(
                    longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
                ).sortby("longitude")
                temp_500mb_data = temp_500mb_ds["T"]
            except Exception as e:
                print(f"  [Rank {rank}] [{tag}] AUX LOAD FAILED [temp_500mb] for {nc_file}: "
                      f"{type(e).__name__}: {str(e)[:200]}")
                temp_500mb_data = None
        else:
            print(f"  [Rank {rank}] [{tag}] AUX MISSING [temp_500mb] for {nc_file}: "
                  f"{temp_500mb_file}")

        # Brightness temperature
        # Schema note: newer ERA5 BT files (re-downloaded via the updated CDS API)
        # have a 'valid_time' dimension instead of 'time'. Older files use 'time'.
        # We handle both via a conditional rename so the downstream sel(time=...)
        # works either way.
        bt_data  = None
        bt_file  = os.path.join(Bright_temp, f"era5_bt_6hourly_{year:04d}.nc")
        if os.path.exists(bt_file):
            try:
                bt_ds_tmp = xr.open_dataset(bt_file)
                if "valid_time" in bt_ds_tmp.dims:
                    bt_ds_tmp = bt_ds_tmp.rename({"valid_time": "time"})
                # Newer xarray rejects `method` together with a slice indexer
                # (slices do range selection and don't need nearest matching).
                # Use the actual last-day-of-month — "-31" fails for Apr/Jun/Sep/Nov
                # (30 days) and Feb (28/29) on strict DatetimeIndex parsing.
                _last_day = pd.Timestamp(year=year, month=month, day=1).days_in_month
                bt_data   = bt_ds_tmp.sel(
                    time=slice(f"{year:04d}-{month:02d}-01",
                               f"{year:04d}-{month:02d}-{_last_day:02d}")
                )
            except Exception as e:
                print(f"  [Rank {rank}] [{tag}] AUX LOAD FAILED [BT] for {nc_file}: "
                      f"{type(e).__name__}: {str(e)[:200]}")
                bt_data = None
        else:
            print(f"  [Rank {rank}] [{tag}] AUX MISSING [BT] for {nc_file}: {bt_file}")

        # Wind shear
        # Schema note: newer wind-shear NetCDFs ship the variable as 'windshear';
        # older files used 'shear_mag'. Try the canonical name, fall back to the
        # legacy name so both vintages work.
        windshear_data = None
        windshear_file = os.path.join(windshear_dir, f"6hourly_ERA5_windshear_mag_{date_str_temp}.nc")
        if os.path.exists(windshear_file):
            try:
                _ws_ds  = xr.open_dataset(windshear_file)
                _ws_var = "windshear" if "windshear" in _ws_ds.data_vars else "shear_mag"
                windshear_data = _ws_ds[_ws_var].squeeze()
            except Exception as e:
                print(f"  [Rank {rank}] [{tag}] AUX LOAD FAILED [windshear] for {nc_file}: "
                      f"{type(e).__name__}: {str(e)[:200]}")
                windshear_data = None
        else:
            print(f"  [Rank {rank}] [{tag}] AUX MISSING [windshear] for {nc_file}: "
                  f"{windshear_file}")

        # Vorticity
        vorticity_month_data = None
        vorticity_file = find_matching_vorticity(
            pd.Timestamp(year=year, month=month, day=1), vorticity_files
        )
        if vorticity_file is not None:
            try:
                vorticity_ds = xr.open_dataset(vorticity_file).squeeze("plev")
                vorticity_ds = vorticity_ds.assign_coords(
                    longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
                ).sortby("longitude")
                vorticity_month_data = vorticity_ds["var138"]
            except Exception as e:
                print(f"  [Rank {rank}] [{tag}] AUX LOAD FAILED [vorticity] for {nc_file}: "
                      f"{type(e).__name__}: {str(e)[:200]}")
                vorticity_month_data = None
        else:
            print(f"  [Rank {rank}] [{tag}] AUX MISSING [vorticity] for {nc_file}: "
                  f"no matching vorticity file for {year}-{month:02d}")

        timing["aux_load"] = time.time() - _t; _t = time.time()

        # ── Critical-aux gate: HARD SKIP files that would yield degenerate targets ──
        # If any critical aux is None, the dataset silently zero-fills the
        # corresponding channels, which produces all-zero target masks for
        # entire storm classes — pulling the model toward "predict background
        # everywhere" and tanking val IoU. Better to skip the file entirely.
        #
        # BT        → required for MCS target masks (channels 11, 12, 13)
        # vorticity → required for ETC + Hurricane vorticity components
        #             (channels 3, 8, 10) and CyTRACK-style hurricane centre detection
        #
        # temp_500mb and wind_shear are intentionally NOT critical: their absence
        # only relaxes the hurricane warm-core / shear filter rather than zeroing
        # an entire class's target — the model can still learn from such files.
        missing_critical = []
        if bt_data is None:
            missing_critical.append("BT")
        if vorticity_month_data is None:
            missing_critical.append("vorticity")

        if missing_critical:
            print(f"  [Rank {rank}] [{tag}] HARD SKIP {nc_file}: missing critical aux "
                  f"{missing_critical} → would produce degenerate target masks. "
                  f"Skipping this file for this pass (no gradient contribution).")

            if is_val:
                # Val: no DDP collectives inside the forward, so just skip locally.
                # Other ranks will hit the per-pass accumulator.reduce() and this
                # rank simply contributes zero counts → metric stays correct.
                return None, None

            # Train: peers will enter train_wnet_improved with its DDP collective
            # sequence (max_batches all_reduce, per-batch grad sync, per-epoch loss
            # all_reduce, scheduler.step, per-epoch barrier). To avoid a deadlock,
            # this rank must mirror that exact sequence — that's what
            # run_zombie_training does. Same path used for None-padded slots.
            if distributed:
                try:
                    dist.barrier()
                except Exception as _e:
                    print(f"[Rank {rank}] Pre-zombie barrier warning: {_e}")
            run_zombie_training(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=epochs_per_file,
                gradient_accumulation_steps=gradient_accumulation_steps,
                world_size=world_size,
                rank=rank,
            )
            if distributed:
                try:
                    dist.barrier()
                except Exception as _e:
                    print(f"[Rank {rank}] Post-zombie barrier warning: {_e}")
            return None, None

        # ── HDF5 cache / NetCDF fallback ──────────────────────────────────────
        preprocessed_data = cache_loader.load_data(
            year=year, month=month, data_dir=data_dir,
            use_cache=True, fallback_to_netcdf=True,
            vorticity=vorticity_month_data,
            wind_speed=wind_speed_full,
            temp_500mb=temp_500mb_data,
            BT_ds=bt_data,
            windhshear_ds=windshear_data,
            ivt_percentile=interp_percentile,
        )

        timing["cache_load"] = time.time() - _t; _t = time.time()
        if rank == 0:
            print(f"  [TIMING] Cache/NetCDF load: {timing['cache_load']:.2f}s")

        if preprocessed_data is None:
            if rank == 0:
                print(f"  Skipping {nc_file}: could not load data.")
            return None, None

        # ── Reconstruct time axis ─────────────────────────────────────────────
        if "time" in preprocessed_data:
            time_values = preprocessed_data["time"]
            try:
                if np.issubdtype(time_values.dtype, np.integer):
                    time_values = time_values.astype("datetime64[s]")
                elif time_values.dtype.type is np.bytes_ or time_values.dtype.char == "S":
                    str_times   = [t.decode("utf-8") if isinstance(t, (bytes, np.bytes_))
                                   else str(t) for t in time_values]
                    time_values = np.array(str_times, dtype="datetime64[s]")
            except Exception:
                ERA5_tmp    = xr.open_dataset(os.path.join(data_dir, nc_file))
                time_values = ERA5_tmp["time"].values
        else:
            ERA5_tmp    = xr.open_dataset(os.path.join(data_dir, nc_file))
            time_values = ERA5_tmp["time"].values

        # ── Reconstruct lat/lon ───────────────────────────────────────────────
        if "latitude" in preprocessed_data and "longitude" in preprocessed_data:
            lat_values = preprocessed_data["latitude"]
            lon_values = preprocessed_data["longitude"]
        else:
            _sample_3d = next(
                (v for k, v in preprocessed_data.items()
                 if k != "time" and isinstance(v, np.ndarray) and v.ndim == 3), None
            )
            if _sample_3d is not None:
                _n_lat, _n_lon = _sample_3d.shape[1], _sample_3d.shape[2]
                lat_values = np.linspace(90, -90, _n_lat)
                lon_values = np.linspace(-180, 180 - 360 / _n_lon, _n_lon)
            else:
                lat_values = lon_values = None

        # ── Build xarray Dataset ──────────────────────────────────────────────
        ERA5_6hrly_ARs = xr.Dataset()
        _skip_keys     = {"time", "latitude", "longitude"}
        for key, value in preprocessed_data.items():
            if key not in _skip_keys and isinstance(value, np.ndarray) and value.ndim == 3:
                coords_dict = {"time": time_values}
                if lat_values is not None: coords_dict["latitude"]  = lat_values
                if lon_values is not None: coords_dict["longitude"] = lon_values
                ERA5_6hrly_ARs[key] = xr.DataArray(
                    value, dims=["time", "latitude", "longitude"], coords=coords_dict
                )

        timing["reconstruct"] = time.time() - _t; _t = time.time()

        # ── IBTrACS slice for this month ──────────────────────────────────────
        era5_6h_keys = [
            pd.Timestamp(t).to_pydatetime().replace(minute=0, second=0, microsecond=0)
            for t in time_values
        ]
        ibtracs_this_month = {dt: ibtracs_lookup.get(dt, []) for dt in era5_6h_keys}

        if rank == 0:
            active = [(dt, pts) for dt, pts in ibtracs_this_month.items() if pts]
            print(f"  IBTrACS {year}-{month:02d}: "
                  f"{len(active)}/{len(era5_6h_keys)} timesteps have active storms")

        # ── Thresholds & stats ────────────────────────────────────────────────
        vorticity_threshold = 1.45e-5

        global_percentiles = {
            "vor850":          vor850_results_per,
            "windspeed":       windspeed_results_per,
            "BT_boundary":     241.0,
            "BT_core":         225.0,
            "Temp500":         270.0,
            "vor850_per_hurr": 1.45e-5,
            "windspeed_per_hurr": 8.0,
        }

        global_stats = {
            "IVT":        {"mean": 170.566,    "std": 11.6209},
            "IWV":        {"mean": 24.1535,    "std": 1.12097},
            "BT":         {"mean": 254.675,    "std": 0.648636},
            "MSLP":       {"mean": 101138.0,   "std": 15.6262},
            "vorticity":  {"mean": -4.09177e-10, "std": 4.02875e-05},
            "wind_speed": {"mean": 8.35723,    "std": 0.306227},
            "temp_500mb": {"mean": 258.428,    "std": 0.832617},
        }

        # ── Build ARDataset ───────────────────────────────────────────────────
        dataset = ARDatasetImproved(
            ERA5_6hrly_ARs,
            ivt_percentile=interp_percentile,
            vorticity_threshold=vorticity_threshold,
            target_size=target_size,
            global_percentiles=global_percentiles,
            global_stats=global_stats,
            precompute=True,
            mslp_climatology=mslp_anom_result_glob,
            ibtracs_lookup=ibtracs_this_month,
            year=year, 
            month=month
        )

        timing["dataset_init"] = time.time() - _t; _t = time.time()
        if rank == 0:
            print(f"  [TIMING] Dataset init (precompute): {timing['dataset_init']:.2f}s")

        # ── DataLoader — NO DistributedSampler (files already sharded) ────────
        # Each rank owns its own files exclusively, so we just shuffle within
        # the file.  DDP gradient all_reduce handles weight synchronisation.
        # Val mode: shuffle off and keep every sample (drop_last=False).
        train_loader = MemoryEfficientDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=False,          # ROCm pin_memory unreliable on Frontier
            persistent_workers=True,
            sampler=None,              # ← KEY: no DistributedSampler
            shuffle=(not is_val),      # shuffle for train; off for val
            drop_last=(not is_val),    # keep every val sample for accurate metrics
            prefetch_factor=1,
            timeout=300,
        )

        timing["loader_init"] = time.time() - _t; _t = time.time()

        if rank == 0:
            kind = "val" if is_val else "train"
            print(f"  [Rank {rank}] {len(dataset)} samples → "
                  f"{len(train_loader)} {kind} batches")

        if is_val:
            # ── Validate ────────────────────────────────────────────────────
            # Forward-only, seg-head-only — no DDP collectives during forward.
            # `val_accumulator` is shared across all val files in this pass.
            n_val_batches = validate_on_loader(
                model, train_loader, device,
                accumulator=val_accumulator,
                use_amp=(scaler is not None),
                rank=rank,
            )
            timing["val"] = time.time() - _t
            if rank == 0:
                total_file_time = sum(timing.values())
                print(f"  [TIMING-VAL] aux:{timing['aux_load']:.0f}s | "
                      f"cache:{timing['cache_load']:.0f}s | "
                      f"reconstruct:{timing['reconstruct']:.0f}s | "
                      f"dataset:{timing['dataset_init']:.0f}s | "
                      f"val:{timing['val']:.0f}s ({n_val_batches} batches) | "
                      f"TOTAL:{total_file_time:.0f}s")
            return None, None

        # ── Train ─────────────────────────────────────────────────────────────────
        # Sync all ranks here so every rank has finished dataset init
        # before the all_reduce inside train_wnet_improved fires.
        if distributed:
            dist.barrier()

        epoch_losses, loss_details = train_wnet_improved(
            model, train_loader, optimizer, scheduler, device,
            num_epochs=epochs_per_file,
            percentile_threshold=interp_percentile,
            vorticity_threshold=vorticity_threshold,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_step_scheduler=False,
            scaler=scaler,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            ema=ema,
            file_idx=file_idx,
            global_epoch_offset=global_epoch_offset,
        )

        timing["train"] = time.time() - _t
        if rank == 0:
            total_file_time = sum(timing.values())
            print(f"  [TIMING] aux:{timing['aux_load']:.0f}s | "
                  f"cache:{timing['cache_load']:.0f}s | "
                  f"reconstruct:{timing['reconstruct']:.0f}s | "
                  f"dataset:{timing['dataset_init']:.0f}s | "
                  f"train:{timing['train']:.0f}s | "
                  f"TOTAL:{total_file_time:.0f}s ({total_file_time/3600:.2f}h)")

        return epoch_losses, loss_details

    except Exception as e:
        if rank == 0:
            print(f"  ERROR processing {nc_file}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        return None, None

    finally:
        # Always release memory after each file
        for var in ["ERA5_6hrly_ARs", "dataset", "train_loader",
                    "preprocessed_data", "bt_data", "windshear_data",
                    "vorticity_month_data", "temp_500mb_data"]:
            try:
                del locals()[var]
            except KeyError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def main_improved(
    data_dir, checkpoint_path=None,
    start_year=1980, end_year=2005,
    start_month=1,   end_month=12,
    distributed=False, world_size=1, rank=0, local_rank=0, device=None,
):
    global model, dataset, train_loader, optimizer, scheduler, losses

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_gpu()
    scaler = GradScaler() if torch.cuda.is_available() else None

    if rank == 0:
        print_system_info()

    # ── Percentile / climatology datasets ────────────────────────────────────
    vor850_results_per_path    = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                                  "vor850_latlongrid/vor850_latlongrid/output/"
                                  "vor850_75th_percentile_global.nc")
    windspeed_results_per_path = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                                  "wind_speed_850/output/windspeed850_75th_percentile_global.nc")
    mslp_anomaly_glob          = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                                  "data/global_stats/mslp_monthly_climatology.nc")

    vor850_results_per    = load_percentile_dataset(vor850_results_per_path,    rank)
    windspeed_results_per = load_percentile_dataset(windspeed_results_per_path, rank)
    mslp_anom_result_glob = load_percentile_dataset(mslp_anomaly_glob,          rank)

    # ── Auxiliary data paths ──────────────────────────────────────────────────
    vorticity_dir  = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                      "vor850_latlongrid/vor850_latlongrid/")
    vorticity_files = sorted(glob.glob(
        os.path.join(vorticity_dir, "ERA5_vor_6hourly_*.850mb_latlon.nc")
    ))

    wind_speed_full = xr.open_dataset(
        "/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
        "wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc"
    ).squeeze("level")
    wind_speed_full = wind_speed_full.assign_coords(
        longitude=(((wind_speed_full.longitude + 180) % 360) - 180)
    ).sortby("longitude")

    temp_500mb_dir = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/"
                      "GLOBAL/from-ncar-server/t/monthly_output/6_hourly")
    windshear_dir  = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                      "wind_shear_mag_sqrtu_v/")
    Bright_temp    = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                      "OLR/brightness_temp_new/")

    # ── IBTrACS ───────────────────────────────────────────────────────────────
    ibtracs_file = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
                    "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc")
    if rank == 0:
        print("Loading IBTrACS …")
    ibtracs_lookup = load_ibtracs_for_training(
        ibtracs_file, start_year=1980, end_year=2024,
        wind_threshold_kt=0.0, lat_min=-50.0, lat_max=50.0,
    )
    if rank == 0:
        print(f"IBTrACS ready: {len(ibtracs_lookup)} active timesteps")

    # ── Collect ALL monthly files in the requested range ──────────────────────
    def _in_range(fname):
        date = fname.split("_")[1].split(".")[0]
        y, m = int(date[:4]), int(date[4:6])
        return (start_year, start_month) <= (y, m) <= (end_year, end_month)

    all_nc_files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("era5dara_") and f.endswith(".nc") and _in_range(f)
    )

    if rank == 0:
        print(f"\nFound {len(all_nc_files)} files for "
              f"{start_year}/{start_month:02d} – {end_year}/{end_month:02d}")
        if not all_nc_files:
            print("No files found — exiting.")
            return

    # ── Global-stratified TRAIN / VAL split (whole-month holdout) ────────────
    # ~13.5% of months held out as validation, stratified by calendar month so
    # every season (incl. hurricane season) has val positives. Identical split
    # on every rank (seed-controlled).
    train_files_all, val_files_all = build_train_val_split(
        all_nc_files, val_fraction=0.135, seed=42,
    )
    log_train_val_split(rank, all_nc_files, train_files_all, val_files_all)

    # ── FILE-LEVEL SHARDING (train + val sharded independently) ──────────────
    # Each rank gets a disjoint subset of monthly files via round-robin.
    my_files     = get_rank_files(train_files_all, rank, world_size, shuffle=False)
    my_val_files = get_rank_files(val_files_all,   rank, world_size, shuffle=False)
    log_file_assignment(rank, world_size, my_files, len(train_files_all))

    # ── Pad TRAIN file list so all ranks have equal count (prevents barrier timeout) ──
    # Ranks with fewer files get dummy None entries — _process_one_file skips them.
    if distributed:
        # Find max files across all ranks
        my_count = torch.tensor(len(my_files), device=device)
        max_count = my_count.clone()
        dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
        max_files = int(max_count.item())

        # Pad with None so all ranks iterate the same number of times
        while len(my_files) < max_files:
            my_files.append(None)

        if rank == 0:
            print(f"[Rank 0] Train file list padded to {max_files} entries per rank")

        # ── Pad VAL list too (no zombies needed — val forward has no DDP collectives;
        # the only val sync is the SUM all_reduce of TP/FP/FN counts afterwards) ──
        my_val_count  = torch.tensor(len(my_val_files), device=device)
        max_val_count = my_val_count.clone()
        dist.all_reduce(max_val_count, op=dist.ReduceOp.MAX)
        max_val_files = int(max_val_count.item())
        while len(my_val_files) < max_val_files:
            my_val_files.append(None)
        if rank == 0:
            print(f"[Rank 0] Val file list padded to {max_val_files} entries per rank")

    if not my_files:
        print(f"[Rank {rank}] No train files assigned — exiting (more GPUs than files?).")
        return

    # ── Count real files (excluding None pads) for scheduler calc ───────────
    my_real_files     = [f for f in my_files     if f is not None]
    my_real_val_files = [f for f in my_val_files if f is not None]
    if rank == 0:
        print(f"[Rank {rank}] {len(my_real_files)} real train files, "
              f"{len(my_real_val_files)} real val files (this rank)")
    total_training_epochs = num_passes * len(my_real_files) * epochs_per_file
    # warmup_epochs = min(10, total_training_epochs // 10)  # ~10% warmup
    warmup_epochs = min(5, total_training_epochs // 4)  # 5 global epochs w
    # ── Model ─────────────────────────────────────────────────────────────────
    model = ImprovedWNet(in_channels=in_channels, num_classes=5).to(device)

    base_lr      = 0.001
    # learning_rate = base_lr * world_size if distributed else base_lr
    # Sqrt scaling — much more stable than linear for large world_size
    import math as _math
    learning_rate = base_lr * _math.sqrt(world_size) if distributed else base_lr
    optimizer    = optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=1e-5, eps=1e-8, betas=(0.9, 0.999)
    )

    # Global cosine LR schedule with linear warmup — no per-file resets
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_training_epochs - warmup_epochs),
        eta_min=learning_rate * 1e-3,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    if rank == 0:
        print(f"LR schedule: {warmup_epochs} warmup + "
              f"{total_training_epochs - warmup_epochs} cosine epochs "
              f"(total {total_training_epochs})")

    start_epoch, last_loss, loss_history = load_checkpoint(
            model, optimizer, scheduler, scaler, ema, checkpoint_path, device
        )
    # Back-compat: ensure val_metrics list exists in history dict
    loss_history.setdefault("val_metrics", [])


    if distributed:
        device_id = device.index if device.index is not None else 0
        model = DDP(model, device_ids=[device_id], output_device=device_id,
                    find_unused_parameters=False)
        if rank == 0:
            print("Model wrapped with DDP")

    # ── Cache loader ──────────────────────────────────────────────────────────
    cache_loader   = CacheAwareDataLoader(
        cache_dir="../Percentile_ARs_detection/cache/preprocessed_v3"
    )
    interp_percentile = 400

    checkpoint_dir = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                      "Storms_precipitation_notebooks/WNet_parallel_global_code/"
                      "ARs_Hurricanes_ETCs_MCS/Trail11")
    loss_details_dir = os.path.join(checkpoint_dir, "loss_details")
    if rank == 0:
        os.makedirs(loss_details_dir, exist_ok=True)

    # ── Main training loop — multi-pass cycling over files ──────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting training: {len(my_real_files)} real files × "
              f"{num_passes} passes × {epochs_per_file} epochs/file on rank {rank}")
        print(f"{'='*60}")

    start_time = time.time()
    total_epochs_so_far = 0
    import random

    for pass_idx in range(num_passes):

        # Shuffle file order each pass (all ranks use same seed → same order)
        pass_files = list(my_files)  # includes None pads
        real_indices = [i for i, f in enumerate(pass_files) if f is not None]
        real_files_this_pass = [pass_files[i] for i in real_indices]
        random.Random(pass_idx + 42).shuffle(real_files_this_pass)
        for j, idx in enumerate(real_indices):
            pass_files[idx] = real_files_this_pass[j]

        if rank == 0:
            print(f"\n{'━'*60}")
            print(f"PASS {pass_idx+1}/{num_passes} | "
                  f"Global epoch {total_epochs_so_far} | "
                  f"LR {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'━'*60}")

        for file_idx, nc_file in enumerate(pass_files):

            # Ranks with padded None slots → zombie mode
            if nc_file is None:
                if rank == 0 or True:
                    print(f"[Rank {rank}] Pass {pass_idx+1} File slot "
                          f"{file_idx+1}/{len(pass_files)}: ZOMBIE MODE")

                if distributed:
                    dist.barrier()

                run_zombie_training(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=epochs_per_file,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    world_size=world_size,
                    rank=rank,
                )

                if distributed:
                    try:
                        dist.barrier()
                    except Exception as e:
                        print(f"[Rank {rank}] Post-zombie barrier warning: {e}")
                continue

            result = _process_one_file(
                nc_file=nc_file,
                file_idx=file_idx,
                total_files=len(pass_files),
                data_dir=data_dir,
                cache_loader=cache_loader,
                vorticity_files=vorticity_files,
                wind_speed_full=wind_speed_full,
                temp_500mb_dir=temp_500mb_dir,
                windshear_dir=windshear_dir,
                Bright_temp=Bright_temp,
                vor850_results_per=vor850_results_per,
                windspeed_results_per=windspeed_results_per,
                mslp_anom_result_glob=mslp_anom_result_glob,
                ibtracs_lookup=ibtracs_lookup,
                interp_percentile=interp_percentile,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                device=device,
                distributed=distributed,
                world_size=world_size,
                rank=rank,
                start_epoch=start_epoch,
                loss_history=loss_history,
                global_epoch_offset=total_epochs_so_far,
                epochs_per_file=epochs_per_file,
            )

            if result is not None:
                epoch_losses, loss_det = result
            else:
                epoch_losses, loss_det = None, None

            # Increment AFTER the file completes
            if epoch_losses is not None:
                total_epochs_so_far += epochs_per_file

            if epoch_losses is not None and rank == 0:
                loss_history["file_boundaries"].append(len(loss_history["total"]))
                loss_history["file_names"].append(nc_file)

                loss_history["total"].extend(loss_det["total"])
                loss_history["ar"].extend(loss_det["ar"])
                loss_history["hurricane"].extend(loss_det["hurricane"])
                loss_history["etc"].extend(loss_det["etc"])
                loss_history["mcs"].extend(loss_det["mcs"])
                loss_history["overlap"].extend(loss_det["overlap"])

                # Save history after every file
                history_path = os.path.join(checkpoint_dir, "loss_history.pkl")
                with open(history_path, "wb") as f:
                    pickle.dump(loss_history, f)

            # No inter-file LR decay — cosine scheduler handles it globally

            # Synchronise all ranks before moving to the next file
            if distributed:
                try:
                    dist.barrier()
                except Exception as e:
                    print(f"[Rank {rank}] Barrier timeout after file {file_idx}: {e}")

        # ── Per-pass VALIDATION (forward-only, seg-only) ────────────────────
        # Held-out months never seen during training. Each rank validates its
        # own val files (no DDP collectives in forward); per-class TP/FP/FN
        # counts are then SUM all_reduced and converted to IoU/Dice/F1/P/R.
        if rank == 0:
            print(f"\n{'═'*60}")
            print(f"VALIDATION after PASS {pass_idx+1}/{num_passes}  "
                  f"(epoch {total_epochs_so_far})")
            print(f"{'═'*60}")

        val_accumulator = SegmentationMetricAccumulator(device=device)
        val_start = time.time()

        for vf_idx, vnc in enumerate(my_val_files):
            if vnc is None:
                # padding slot for this rank — contributes zeros to accumulator
                continue
            try:
                _process_one_file(
                    nc_file=vnc,
                    file_idx=vf_idx,
                    total_files=len(my_val_files),
                    data_dir=data_dir,
                    cache_loader=cache_loader,
                    vorticity_files=vorticity_files,
                    wind_speed_full=wind_speed_full,
                    temp_500mb_dir=temp_500mb_dir,
                    windshear_dir=windshear_dir,
                    Bright_temp=Bright_temp,
                    vor850_results_per=vor850_results_per,
                    windspeed_results_per=windspeed_results_per,
                    mslp_anom_result_glob=mslp_anom_result_glob,
                    ibtracs_lookup=ibtracs_lookup,
                    interp_percentile=interp_percentile,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    device=device,
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank,
                    start_epoch=start_epoch,
                    loss_history=loss_history,
                    global_epoch_offset=total_epochs_so_far,
                    epochs_per_file=epochs_per_file,
                    mode="val",
                    val_accumulator=val_accumulator,
                )
            except Exception as e:
                print(f"[Rank {rank}] [VAL] file {vnc} failed: "
                      f"{type(e).__name__}: {e}")

        # SUM-reduce per-class counts across all ranks → per-class metrics
        val_accumulator.reduce(world_size)
        metrics = val_accumulator.compute()
        val_time = time.time() - val_start

        if rank == 0:
            print(f"\n[VAL PASS {pass_idx+1}/{num_passes}] "
                  f"time={val_time:.0f}s")
            print(f"  {format_metrics(metrics)}")
            loss_history["val_metrics"].append({
                "pass_idx":     pass_idx,
                "global_epoch": total_epochs_so_far,
                "val_time_sec": val_time,
                "metrics":      metrics,
            })
            # Persist updated history (now including this pass's val metrics)
            history_path = os.path.join(checkpoint_dir, "loss_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(loss_history, f)

        if distributed:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[Rank {rank}] Post-val barrier warning: {e}")

    # ── Final checkpoint (rank 0 only) ────────────────────────────────────────
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nAll passes done in {total_time:.0f}s ({total_time/3600:.2f}h)")

        model_to_save = model.module if distributed else model
        final_ckpt = {
            "epoch":              total_epochs_so_far,
            "model_state_dict":   model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":  scaler.state_dict() if scaler else None,
            "losses":             loss_history,
            "training_time":      total_time,
            "completed":          True,
            "ema_state_dict":     ema.state_dict(),
        }
        final_path = os.path.join(
            checkpoint_dir, f"final_checkpoint_{start_year}_{end_year}.pth"
        )
        torch.save(final_ckpt, final_path)
        print(f"Final checkpoint saved → {final_path}")
        plot_training_history(loss_history)
        print("Training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WNet AR/Hurricane/ETC/MCS Training")
    parser.add_argument("--start_year",  type=int, default=1980)
    parser.add_argument("--end_year",    type=int, default=2024)
    parser.add_argument("--start_month", type=int, default=1)
    parser.add_argument("--end_month",   type=int, default=12)
    parser.add_argument("--epochs_per_file", type=int, default=None,
                        help="Epochs per file per pass (default: use global)")
    parser.add_argument("--num_passes", type=int, default=None,
                        help="Number of passes over all files (default: use global)")
    args = parser.parse_args()

    assert 1 <= args.start_month <= 12
    assert 1 <= args.end_month   <= 12

    global epochs_per_file, num_passes
    if args.epochs_per_file is not None:
        epochs_per_file = args.epochs_per_file
    if args.num_passes is not None:
        num_passes = args.num_passes

    distributed, world_size, rank, local_rank, device = setup_distributed()
    
    effective_batch = batch_size * world_size * gradient_accumulation_steps


    if distributed:
        dist.barrier()

    data_directory = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
    checkpoint_dir = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                      "Storms_precipitation_notebooks/WNet_parallel_global_code/"
                      "ARs_Hurricanes_ETCs_MCS/Trail11")
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Year/Month range  : {args.start_year}/{args.start_month:02d} – "
              f"{args.end_year}/{args.end_month:02d}")
        print(f"  Distributed       : {distributed}")
        print(f"  World size        : {world_size} GPUs")
        print(f"  Per-GPU batch     : {batch_size}")
        print(f"  Effective batch   : {effective_batch} "
              f"(batch×GPUs×grad_accum = {batch_size}×{world_size}×{gradient_accumulation_steps})")
        print(f"  Epochs per file   : {epochs_per_file}")
        print(f"  Num passes        : {num_passes}")
        print(f"  Device            : {device}")
        print(f"  Checkpoint        : {checkpoint_path or 'None (fresh start)'}")
        print(f"{'='*70}\n")

    try:
        main_improved(
            data_directory,
            checkpoint_path=checkpoint_path,
            start_year=args.start_year,
            end_year=args.end_year,
            start_month=args.start_month,
            end_month=args.end_month,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
        )
    except Exception as e:
        print(f"\nERROR Rank {rank}: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        if distributed:
            try:
                dist.barrier()
                cleanup_distributed()
            except Exception as ce:
                print(f"Cleanup warning rank {rank}: {ce}")


if __name__ == "__main__":
    main()
