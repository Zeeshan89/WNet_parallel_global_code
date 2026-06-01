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
# For plotting with geographical coordinates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
# GPU and Distributed Training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
# Reading components
from WNet_model import ImprovedWNet
from dataset import ARDatasetImproved, MemoryEfficientDataLoader, preprocess_data_improved
from train import train_wnet_improved
from loss import DiceLoss, FocalLoss, NCutLoss2D, ARSegmentationLoss
from utils import setup_gpu, calculate_ivt, print_system_info, find_matching_vorticity, find_latest_checkpoint, load_checkpoint, load_percentile_dataset
from visualization import visualize_four_class_segmentation, plot_training_history
from loss import optimize_batch_size_for_gpu
from cache_loader import CacheAwareDataLoader, get_cache_info
from datetime import timedelta

# GPU and Distributed Training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
import argparse
from ibtracs_loader import load_ibtracs_for_training
from pympler import asizeof
from distributed_dataset import make_distributed_loaders


# MPI for distributed training (optional but more robust on HPC systems)
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except Exception as e:
    MPI_AVAILABLE = False
    print(f"Warning: mpi4py unavailable ({e}). Will use SLURM environment variables for distributed setup.")

def setup_distributed():
    """
    Setup distributed training environment for ROCm/AMD GPUs on Frontier.

    Follows S2S_Tak_Workflow pattern:
    - All 8 GCDs are visible to every process (--gres=gpu:8 in SLURM)
    - Each process selects its GPU via: device = local_rank % gpu_count
    - No manual ROCR_VISIBLE_DEVICES filtering needed
    - MASTER_ADDR and MASTER_PORT set by bash script
    - NCCL_SOCKET_IFNAME=hsn0 for Frontier Slingshot network
    """
    def _select_device(rank, local_rank, gpu_count):
        visible_devices = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            print(f"Rank {rank}: ROCR_VISIBLE_DEVICES={visible_devices}")
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

    # Method 1: Try MPI first (more reliable on HPC systems like Frontier)
    if MPI_AVAILABLE:
        try:
            comm = MPI.COMM_WORLD
            world_size = comm.Get_size()
            rank = comm.Get_rank()
            # Get GPU count to compute local_rank
            gpu_count = torch.cuda.device_count()
            slurm_localid = os.environ.get("SLURM_LOCALID")
            if slurm_localid is not None:
                local_rank = int(slurm_localid)
            else:
                tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", world_size))
                local_rank = int(rank) % max(1, tasks_per_node)
            
            print(f"Using MPI for rank detection: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            
            # Set environment variables for PyTorch DDP
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            nccl_ifname = os.environ.get('NCCL_SOCKET_IFNAME', 'hsn0')
            
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['NCCL_SOCKET_IFNAME'] = nccl_ifname
            
            # Proceed with device setup and initialization
            print(f"\nRank {rank}/{world_size} (LOCAL_RANK={local_rank}): Distributed Environment Check (MPI)")
            print(f"  MASTER_ADDR={master_addr}")
            print(f"  MASTER_PORT={master_port}")
            print(f"  NCCL_SOCKET_IFNAME={nccl_ifname}\n")
            
            # Select GPU
            if gpu_count == 0 or not torch.cuda.is_available():
                raise RuntimeError(
                    f"Rank {rank}: No GPUs detected (device_count={gpu_count}). "
                    f"Verify ROCm modules are loaded and GPUs are available."
                )
            
            print(f"Rank {rank}: Found {gpu_count} GPU(s) visible to this process")
            device_id, device = _select_device(rank, local_rank, gpu_count)
            
            # Initialize process group
            if rank == 0:
                print(f"\nInitializing NCCL backend...")
                print(f"  World size: {world_size}")
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=1800)
            )
            
            if rank == 0:
                print(f"Successfully initialized NCCL backend")
            
            print(f"Rank {rank}/{world_size}: Ready on {device}")
            return True, world_size, rank, local_rank, device
            
        except Exception as e:
            print(f"MPI initialization failed: {e}. Falling back to SLURM environment variables.")
    
    # Method 2: Check if running under srun/distributed launcher using SLURM env vars
    world_size_env = os.environ.get('WORLD_SIZE')
    rank_env = os.environ.get('RANK')
    local_rank_env = os.environ.get('LOCAL_RANK')

    # Try to use SLURM variables as fallback
    if not world_size_env and 'SLURM_NTASKS' in os.environ:
        print("WORLD_SIZE not found but SLURM_NTASKS detected - using SLURM variables")
        world_size_env = os.environ.get('SLURM_NTASKS')
        rank_env = os.environ.get('SLURM_PROCID')
        local_rank_env = os.environ.get('SLURM_LOCALID')

    if world_size_env and rank_env is not None and local_rank_env is not None:
        world_size = int(world_size_env)
        rank = int(rank_env)
        local_rank = int(local_rank_env)

        # Retrieve environment variables set by bash script
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        nccl_ifname = os.environ.get('NCCL_SOCKET_IFNAME', 'hsn0')

        # Ensure env vars are set for init_process_group with init_method='env://'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['NCCL_SOCKET_IFNAME'] = nccl_ifname

        # Print environment info for debugging
        print(f"\nRank {rank}/{world_size} (LOCAL_RANK={local_rank}): Distributed Environment Check")
        print(f"  MASTER_ADDR={master_addr}")
        print(f"  MASTER_PORT={master_port}")
        print(f"  NCCL_SOCKET_IFNAME={nccl_ifname}\n")

        # Verify GPUs are available
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0 or not torch.cuda.is_available():
            raise RuntimeError(
                f"Rank {rank}: No GPUs detected (device_count={gpu_count}). "
                f"Verify ROCm modules are loaded and GPUs are available."
            )

        print(f"Rank {rank}: Found {gpu_count} GPU(s) visible to this process")

        # Select GPU using visibility-aware mapping
        device_id, device = _select_device(rank, local_rank, gpu_count)

        # Initialize process group using env:// (S2S pattern)
        try:
            if rank == 0:
                print(f"\nInitializing NCCL backend...")
                print(f"  World size: {world_size}")

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=1800)
            )

        except Exception as e:
            print(f"ERROR Rank {rank}: NCCL init failed: {e}")
            raise RuntimeError(f"Failed to initialize distributed training: {e}")

        if rank == 0:
            print(f"Successfully initialized NCCL backend")

        print(f"Rank {rank}/{world_size}: Ready on {device}")

        return True, world_size, rank, local_rank, device
    else:
        # Single GPU mode (no distributed training)
        print(f"\nNo WORLD_SIZE in environment - running in single-GPU mode")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Single GPU detected: {torch.cuda.get_device_name(0)}")
        return False, 1, 0, 0, device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Global Variables ---
# The effective batch size across all GPUs is:
# ```
# Effective Batch Size = batch_size × num_gpus × gradient_accumulation_steps
#                      = 6 × 4 × 1 = 24 (for 4 GPUs)
#                      = 6 × 8 × 1 = 48 (for 8 GPUs)
# ```

# Manual batch size configuration
# OPTIMIZATION: Balanced settings to avoid OOM while improving GPU utilization
# batch_size=3 is safer than 4 for 8 GPUs with pin_memory + prefetching
batch_size = 2  # Original working value
gradient_accumulation_steps = 3  # Original working value
chunk_size = 12  # Increased for GPU processing
target_size = (192, 384) # Must be divisible by 8 (encoder stride 2^3); closest safe size to (180,360)
# target_size = (360, 720)  # Keep resolution high for GPU
epochs = 40
in_channels = 14


# Initialize global variables
model = None
dataset = None
train_loader = None
optimizer = None
scheduler = None
losses = None



def main_improved(data_dir, checkpoint_path=None, start_year=1980, end_year=2024,
                 start_month=1, end_month=12,
                 distributed=False, world_size=1, rank=0, local_rank=0, device=None):
    """
    Improved main function with GPU optimization, distributed training support, and checkpoint loading
    
    Args:
        data_dir (str): Directory containing the data files
        checkpoint_path (str): Path to checkpoint file to resume from (optional)
        start_year (int): Start year for training data
        end_year (int): End year for training data
        distributed (bool): Whether using distributed training
        world_size (int): Number of processes
        rank (int): Process rank
        local_rank (int): Local rank
        device: Device to use
    """

    global model, dataset, train_loader, optimizer, scheduler, losses
    
    timing = {}
    _t = time.time()

    # Print system information only on main process
    if rank == 0:
        print_system_info()
        if checkpoint_path:
            print(f"Checkpoint path provided: {checkpoint_path}")
        else:
            print("No checkpoint path provided")
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup GPU optimization
    setup_gpu()
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Load the 90th percentile IVT threshold data
    if rank == 0:
        print("Loading 90th percentile IVT threshold data...")
    
    # IVT_90th_percentile_path = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/ivt_only/IVT_90th_percentile_cdo.nc'
    vor850_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/output/vor850_75th_percentile_global.nc'
    # vor500_results_per_path = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/vor/nc_files/vor500_latlongrid/output/vor500_75th_percentile_global.nc'
    windspeed_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/wind_speed_850/output/windspeed850_75th_percentile_global.nc'
    # Tpr_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/tpr_percentiles/tpr_nonzero_85th_percentile_global.nc'
    # BT10_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new/output/bt_10th_percentile_global.nc'
    # temp500_results_per_path = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/monthly_output/6_hourly/output/t500_80th_percentile_global.nc'
    # vor850_results_per_hurr = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/output/vor850_85th_percentile_global.nc'
    # windspeed_results_per_hurr = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/wind_speed_850/output/windspeed850_85th_percentile_global.nc'
    mslp_anomaly_glob = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/global_stats/mslp_monthly_climatology.nc'

    # percentile_threshold   = load_percentile_dataset(IVT_90th_percentile_path, rank)
    vor850_results_per    = load_percentile_dataset(vor850_results_per_path, rank)
    # vor500_results_per    = load_percentile_dataset(vor500_results_per_path, rank)
    windspeed_results_per = load_percentile_dataset(windspeed_results_per_path, rank)
    # Tpr_results_per       = load_percentile_dataset(Tpr_results_per_path, rank)
    # BT10_results_per       = load_percentile_dataset(BT10_results_per_path, rank)
    # temp500_results_per       = load_percentile_dataset(temp500_results_per_path, rank)
    # vor850_results_per_hurr    = load_percentile_dataset(vor850_results_per_hurr, rank)
    # windspeed_results_hurr = load_percentile_dataset(windspeed_results_per_hurr, rank)
    mslp_anom_result_glob = load_percentile_dataset(mslp_anomaly_glob, rank)
    # print(windspeed_results_per)  # verify shape is (12, H, W)


    timing['percentile_loading'] = time.time() - _t; _t = time.time()
    if rank == 0: print(f"[TIMING] Percentile loading: {timing['percentile_loading']:.2f}s")

    
    datasets = {
        "vor850": vor850_results_per,
        # "vor500": vor500_results_per,
        "windspeed": windspeed_results_per,
        # "TPR": Tpr_results_per,
        # "BT10": BT10_results_per,
        # "Temp500": temp500_results_per,
        # "vor850_results_per_hurr": vor850_results_per_hurr,
        # "windspeed_results_hurr": windspeed_results_hurr
    }

    for name, ds in datasets.items():
        print(f"{name:12s} -> min: {ds.min().item():.3e}, max: {ds.max().item():.3e}")
    
    if rank == 0:
        print("Done reading percentiles:")

        
    # Load vorticity data for hurricane detection
    if rank == 0:
        print("Loading vorticity data for hurricane detection...")
    
    ## vorticity 850mb
    vorticity_dir = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/'
    vorticity_files = sorted(glob.glob(os.path.join(vorticity_dir, 'ERA5_vor_6hourly_*.850mb_latlon.nc')))
    
    
    ## vorticity 500mb
    # vorticity_500_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/vor/nc_files/vor500_latlongrid/'
    # vorticity_500_files = sorted(glob.glob(os.path.join(vorticity_500_dir, 'ERA5_vor_6hourly_*.500mb_latlon.nc')))
    
    # Load wind speed data ONCE outside the file loop
    wind_speed_path = os.path.join(
        '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc'
    )
    wind_speed_full = xr.open_dataset(wind_speed_path).squeeze('level')
    wind_speed_full = wind_speed_full.assign_coords(longitude=(((wind_speed_full.longitude + 180) % 360) - 180)).sortby('longitude')

    timing['wind_data_loading'] = time.time() - _t; _t = time.time()
    if rank == 0: print(f"[TIMING] Wind data loading: {timing['wind_data_loading']:.2f}s")

    # Load 500mb temperature data directory (t300-t500)
    # temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/Hurricane_warmcore/monthly_output_6hourly'
    
    ### need to change file name in temp_500mb_dir to t500_6hourly_198001.nc, t500_6hourly_198002.nc, etc. to match the pattern in vorticity_files and wind_speed_files
    temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/monthly_output/6_hourly'
    
    # Load windshear data directory # 6hourly_ERA5_windshear_198001.nc
    # Calculate magnitude: shear_mag = sqrt(u_shear^2 + v_shear^2) in windshear_dir
    windshear_dir = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_shear_mag_sqrtu_v/'
    
    Bright_temp = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new/' ##era5_bt_6hourly_1981.nc
    
    # Load SST data directory # era5_sst_1980.nc
    # SST_dir = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/sst/celsius/'
    

    ibtracs_file = (
        "/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
        "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc"
    )

    if rank == 0:
        print("Loading IBTrACS hurricane lookup …")

    
    ibtracs_lookup = load_ibtracs_for_training(
    ibtracs_file,
    start_year        = 1980,
    end_year          = 2024,
    wind_threshold_kt = 0.0,    # Include ALL IBTrACS points (TD and above) for complete track injection
    lat_min           = -50.0,
    lat_max           =  50.0,
    )
    # ibtracs_lookup : dict[ datetime ] -> list[ (lat, lon, wind_ms, sshs) ]

    if rank == 0:
        print(f"IBTrACS lookup ready: {len(ibtracs_lookup)} active timesteps\n")
    
    timing['ibtracs_loading'] = time.time() - _t; _t = time.time()
    if rank == 0: print(f"[TIMING] IBTrACS loading: {timing['ibtracs_loading']:.2f}s")

    # Collect .nc files based on year range
    nc_files = [f for f in os.listdir(data_dir) if f.startswith('era5dara_') and f.endswith('.nc')]

    def _in_range(fname):
        date = fname.split('_')[1].split('.')[0]  # e.g. "198003"
        y, m = int(date[:4]), int(date[4:6])
        start = (start_year, start_month)
        end   = (end_year,   end_month)
        return start <= (y, m) <= end

    nc_files = [f for f in nc_files if _in_range(f)]
    nc_files.sort()

    if rank == 0:
        print(f"Found {len(nc_files)} files for {start_year}/{start_month:02d} - {end_year}/{end_month:02d}")
        if len(nc_files) == 0:
            print("No files found for the specified year range!")
            return
    
    # Initialize model
    model = ImprovedWNet(in_channels=in_channels, num_classes=5).to(device)

    # Scale learning rate linearly with world size for multi-GPU training
    base_lr = 0.001
    learning_rate = base_lr * world_size if distributed else base_lr
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-5,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    if rank == 0:
        print(f"Learning rate: {learning_rate} (base_lr={base_lr} * world_size={world_size})")
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # EMA of model parameters for stability
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
    # Load checkpoint if provided
    start_epoch, last_loss, all_losses = load_checkpoint(
        model, optimizer, scheduler, scaler, ema, checkpoint_path, device
    )
    
    
    # Each process uses its own GPU selected by local_rank % gpu_count in setup_distributed()
    if distributed:
        device_id = device.index if device.index is not None else 0
        model = DDP(model, device_ids=[device_id], output_device=device_id, find_unused_parameters=False)
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel")
    
    if rank == 0:
        print("Stage 1: Training with percentile-based thresholds")
        print(f"Training on {len(nc_files)} files with {world_size} GPUs")
        print(f"Total accumulated losses from previous training: {len(all_losses)}")
    
    # Initialize cache loader
    cache_loader = CacheAwareDataLoader(cache_dir='../Percentile_ARs_detection/cache/preprocessed')
    
    # Show cache status
    if rank == 0:
        cache_info = get_cache_info('../Percentile_ARs_detection/cache/preprocessed')
        print(f"\nHDF5 Cache Status: {cache_info['total_files']} files, {cache_info['total_size_mb']:.1f} MB total")
        print("Tip: Use 'python preprocess_cache.py --year YYYY --month MM' to pre-cache months")
    # Default IVT percentile threshold (used when cache supplies preprocessed arrays)
    interp_percentile = 400
    
    # Training loop
    start_time = time.time()
    
    # Process all files in the specified year range
    for file_idx, nc_file in enumerate(nc_files):
        if rank == 0:
            print(f"\nProcessing file {file_idx+1}/{len(nc_files)}: {nc_file}")
        
        _file_t = time.time()
        _stage_t = time.time()
        
        try:
            # Initialize auxiliary variables to None (may not be available)
            vorticity = None
            wind_speed_selected = None
            BT_ds = None
            # TPR = None
            IWV = None
            
            # Extract year and month from filename
            date_str = nc_file.split('_')[1].split('.')[0]  # Gets "201604" from "era5dara_201604.nc"
            year = int(date_str[:4])
            month = int(date_str[4:6])
            
            # Try to load from HDF5 cache (10-20x faster if available)
            file_path = os.path.join(data_dir, nc_file)
            
            # Load auxiliary data for this month (temperature, BT, wind shear, etc.)
            temp_500mb_data = None
            date_str_temp = f"{year:04d}{month:02d}"            

            temp_500mb_file = os.path.join(temp_500mb_dir, f'ERA5_t_500mb_{date_str_temp}.nc')
            if os.path.exists(temp_500mb_file):
                try:
                    temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level='500')
                    temp_500mb_ds = temp_500mb_ds.assign_coords(
                        longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
                    ).sortby('longitude')
                    temp_500mb_interp = temp_500mb_ds["T"]
                    common_times = np.intersect1d(temp_500mb_interp.time.values, time_values)
                    temp_500mb_data = temp_500mb_interp.sel(time=common_times)
                except Exception:
                    temp_500mb_data = None
                
            bt_data = None
            bt_file = os.path.join(Bright_temp, f'era5_bt_6hourly_{year:04d}.nc')
            if os.path.exists(bt_file):
                try:
                    bt_ds_tmp = xr.open_dataset(bt_file)
                    # Select only this month's data
                    bt_data = bt_ds_tmp.sel(time=slice(f'{year:04d}-{month:02d}-01', f'{year:04d}-{month:02d}-31'), method='nearest')
                except Exception:
                    bt_data = None
            
            windshear_data = None
            windshear_file = os.path.join(windshear_dir, f'6hourly_ERA5_windshear_mag_{date_str_temp}.nc')
            if os.path.exists(windshear_file):
                try:
                    windshear_data = xr.open_dataset(windshear_file)['shear_mag'].squeeze()
                except Exception:
                    windshear_data = None
            
            # Find vorticity file for this month using proper date matching
            vorticity_month_data = None
            vorticity_file = find_matching_vorticity(pd.Timestamp(year=year, month=month, day=1), vorticity_files)
            if vorticity_file is not None:
                try:
                    vorticity_ds = xr.open_dataset(vorticity_file).squeeze('plev')
                    vorticity_ds = vorticity_ds.assign_coords(
                        longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
                    ).sortby('longitude')
                    vorticity_month_data = vorticity_ds['var138']
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Could not load vorticity from {vorticity_file}: {e}")
                    vorticity_month_data = None
            
            # Assign loaded data to the initialized variables
            vorticity = vorticity_month_data
            wind_speed_selected = wind_speed_full
            BT_ds = bt_data
            
            preprocessed_data = cache_loader.load_data(
                year=year,
                month=month,
                data_dir=data_dir,
                use_cache=True,
                fallback_to_netcdf=True,  # Will preprocess from NetCDF if cache not found
                # Pass auxiliary data for NetCDF fallback preprocessing
                vorticity=vorticity_month_data,
                wind_speed=wind_speed_full,
                temp_500mb=temp_500mb_data,
                BT_ds=bt_data,
                windhshear_ds=windshear_data,
                ivt_percentile=interp_percentile
            )
            
            _cache_time = time.time() - _stage_t; _stage_t = time.time()
            if rank == 0: print(f"[TIMING] Cache/NetCDF load: {_cache_time:.2f}s")

            if preprocessed_data is None:
                if rank == 0:
                    print(f"Error loading data from cache or NetCDF. Skipping file.")
                continue
            
            # Reconstruct xarray Dataset from cached numpy arrays for compatibility
            # Extract time from cache or original file and convert to datetime64
            if 'time' in preprocessed_data:
                time_values = preprocessed_data['time']
                # If stored as integer seconds since epoch, convert back
                try:
                    if np.issubdtype(time_values.dtype, np.integer):
                        time_values = time_values.astype('datetime64[s]')
                    # If stored as byte strings (ISO), decode and convert
                    elif time_values.dtype.type is np.bytes_ or time_values.dtype.char == 'S':
                        str_times = [t.decode('utf-8') if isinstance(t, (bytes, np.bytes_)) else str(t) for t in time_values]
                        time_values = np.array(str_times, dtype='datetime64[s]')
                except Exception:
                    # Fallback: try reading original file times
                    ERA5_tmp = xr.open_dataset(file_path)
                    time_values = ERA5_tmp['time'].values
            else:
                ERA5_tmp = xr.open_dataset(file_path)
                time_values = ERA5_tmp['time'].values
            
            # Create minimal xarray Dataset for compatibility with existing code
            # Reconstruct proper latitude/longitude coordinates from cached grid shape
            # (ERA5 0.25° global grid, longitude already shifted to -180..180 during preprocessing)
            ERA5_6hrly_ARs = xr.Dataset()


            _reconstruct_time = time.time() - _stage_t; _stage_t = time.time()
            if rank == 0: print(f"[TIMING] Dataset reconstruction: {_reconstruct_time:.2f}s")

            # Prefer lat/lon stored in the cache (added by updated preprocess_cache.py)
            if 'latitude' in preprocessed_data and 'longitude' in preprocessed_data:
                lat_values = preprocessed_data['latitude']
                lon_values = preprocessed_data['longitude']
            else:
                # Fallback: reconstruct from grid shape (for older cache files)
                _sample_3d = next((v for k, v in preprocessed_data.items()
                                   if k != 'time' and isinstance(v, np.ndarray) and v.ndim == 3), None)
                if _sample_3d is not None:
                    _n_lat, _n_lon = _sample_3d.shape[1], _sample_3d.shape[2]
                    lat_values = np.linspace(90, -90, _n_lat)          # N→S
                    lon_values = np.linspace(-180, 180 - 360 / _n_lon, _n_lon)  # -180→179.75
                else:
                    lat_values = None
                    lon_values = None

            _skip_keys = {'time', 'latitude', 'longitude'}
            for key, value in preprocessed_data.items():
                if key not in _skip_keys and isinstance(value, np.ndarray):
                    # Reconstruct the dimension structure (assuming time x lat x lon for 3D data)
                    if len(value.shape) == 3:
                        coords_dict = {'time': time_values}
                        if lat_values is not None:
                            coords_dict['latitude'] = lat_values
                        if lon_values is not None:
                            coords_dict['longitude'] = lon_values
                        ERA5_6hrly_ARs[key] = xr.DataArray(
                            value,
                            dims=['time', 'latitude', 'longitude'],
                            coords=coords_dict
                        )
            
            # Prepare data
            ivt = ERA5_6hrly_ARs.get('ivt')
            mslp = ERA5_6hrly_ARs.get('mslp')
            # TPR = preprocessed_data.get('TPR')
            IWV = preprocessed_data.get('IWV')
            # print("MSLP", mslp)
            # Data is now already preprocessed from cache (all gradients, normalizations computed)
            

            # ── IBTrACS slice for this month  ◄─── ADD HERE, after time_values is built
            # time_values is now fully resolved from preprocessed_data above
            era5_6h_keys = [
                pd.Timestamp(t).to_pydatetime().replace(minute=0, second=0, microsecond=0)
                for t in time_values
            ]

            ibtracs_this_month = {
                dt: ibtracs_lookup.get(dt, [])
                for dt in era5_6h_keys
            }
            print(f"Accurate deep size: {asizeof.asizeof(ibtracs_this_month) / 1024:.2f} KB")

            if rank == 0:
                active = [(dt, pts) for dt, pts in ibtracs_this_month.items() if pts]
                print(f"  IBTrACS {year}-{month:02d}: "
                    f"{len(active)}/{len(era5_6h_keys)} timesteps have active storms")
                for dt, pts in active[:3]:
                    for lat_s, lon_s, wind_s, sshs_s in pts:
                        print(f"    {dt}  lat={lat_s:.1f}  lon={lon_s:.1f}  "
                            f"wind={wind_s:.1f}m/s  sshs={sshs_s}")


            vorticity_threshold = 1.45e-5

            global_percentiles = {
                'vor850': vor850_results_per,        
                'windspeed': windspeed_results_per, 
                 
                ##MCS
                'BT_boundary': 241.0,
                'BT_core': 225.0,
                
                # UPDATED HURRICANE THRESHOLDS - Literature-based
                "Temp500": 270.0,  # temp500_results_per, K
                "vor850_per_hurr": 1.45e-5,            # CHANGED: Was 4.6e-05, now 1.5e-05 (Walsh et al. 2007)
                "windspeed_per_hurr": 8.0,            # CHANGED: Was 19.197, now 17.0 m/s (NHC tropical storm)
            }
            

            global_stats = {
                'IVT':  {'mean': 170.566,  'std': 11.6209},
                'IWV':  {'mean': 24.1535,  'std': 1.12097},
                'BT':   {'mean': 254.675,  'std': 0.648636},
                'MSLP': {'mean': 101138.0, 'std': 15.6262},
                'vorticity':  {'mean': -4.09177e-10, 'std': 4.02875e-05},  # ✓ fixed
                'wind_speed': {'mean': 8.35723,   'std': 0.306227},
                'temp_500mb': {'mean': 258.428,   'std': 0.832617},
            }


            # Use reconstructed xarray Dataset (ERA5_6hrly_ARs) so Dataset class
            # receives xarray objects with coordinates and .time attribute.
            # Only rank 0 pre-computes channels to avoid 8x redundant computation in distributed training

            dataset = ARDatasetImproved(
                ERA5_6hrly_ARs,
                ivt_percentile=interp_percentile,
                vorticity_threshold=vorticity_threshold,
                target_size=target_size,
                global_percentiles=global_percentiles,
                global_stats=global_stats,          # ← ADD THIS
                precompute=True,
                mslp_climatology=mslp_anom_result_glob,
                ibtracs_lookup=ibtracs_this_month,   # ← ADD THIS
            )
            
            _dataset_time = time.time() - _stage_t; _stage_t = time.time()
            if rank == 0: print(f"[TIMING] ARDataset init (incl. precompute): {_dataset_time:.2f}s")


            # Create distributed sampler if needed
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
            
            # Create dataloader with optimized settings
            # OPTIMIZATION: Balanced for both speed AND memory safety
            # 2 workers + prefetch=1 still gives parallelism without OOM risk

            num_loader_workers = 2  # per GPU process = 2×8 = 16 total workers on node (watch RAM)
            train_loader = MemoryEfficientDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_loader_workers,
                pin_memory=False,       # ROCm pin_memory is unreliable, keep False on Frontier
                persistent_workers=True,  # KEEP: avoids expensive worker restart each file
                sampler=sampler,
                drop_last=True,
                prefetch_factor=1,      # KEEP: only 1 batch ahead to limit memory
                timeout=300             # KEEP: safety net for Lustre slowness
            )
            
            _loader_time = time.time() - _stage_t; _stage_t = time.time()
            if rank == 0: print(f"[TIMING] DataLoader creation: {_loader_time:.2f}s")

            # Train the model
            if rank == 0:
                print(f"Starting training on {len(train_loader)} batches...")
            
            epoch_losses, loss_details = train_wnet_improved(
                model, 
                train_loader, 
                optimizer, 
                scheduler, 
                device,
                num_epochs=epochs,
                percentile_threshold=interp_percentile,
                vorticity_threshold=vorticity_threshold,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_step_scheduler=False,
                scaler=scaler,  # Pass scaler for mixed precision
                distributed=distributed,
                rank=rank,
                world_size=world_size  # Pass world_size for proper loss reduction
            )
            
            _train_time = time.time() - _stage_t
            _total_file_time = time.time() - _file_t
            if rank == 0:
                print(f"[TIMING] Training: {_train_time:.2f}s ({_train_time/3600:.2f}h)")
                print(f"[TIMING] Total for {nc_file}: {_total_file_time:.2f}s")
                print(f"[TIMING] Breakdown — Cache:{_cache_time:.1f}s | Reconstruct:{_reconstruct_time:.1f}s | "
                    f"Dataset:{_dataset_time:.1f}s | Loader:{_loader_time:.1f}s | Train:{_train_time:.1f}s")
    
            
            all_losses.extend(epoch_losses)
            
            # Save loss details for this month for aggregation
            if rank == 0:
                loss_details_dir = os.path.join(
                    '/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial49',
                    'loss_details'
                )
                os.makedirs(loss_details_dir, exist_ok=True)
                loss_details_path = os.path.join(
                    loss_details_dir,
                    f'loss_details_{year:04d}{month:02d}.pkl'
                )
                import pickle
                with open(loss_details_path, 'wb') as f:
                    pickle.dump(loss_details, f)
                print(f"Saved loss details to {loss_details_path}")
            
            # Visualize current results (only on main process)
            # if rank == 0:
            #     try:
            #         # Use safe indices that exist in the dataset
            #         dataset_len = len(dataset)
            #         if dataset_len == 0:
            #             print(f"Warning: Empty dataset for {nc_file}, skipping visualization")
            #         else:
            #             # Choose valid indices based on dataset size
            #             safe_indices = []
            #             for idx in [69, 35, 40]:
            #                 if idx < dataset_len:
            #                     safe_indices.append(idx)
                        
            #             # If not enough safe indices, use what's available
            #             if not safe_indices:
            #                 safe_indices = [min(i, dataset_len-1) for i in range(min(3, dataset_len))]
                        
            #             if safe_indices:
            #                 # Visualize if vorticity and wind_speed data are available
            #                 if vorticity is not None and wind_speed_selected is not None:
            #                     try:
            #                         visualize_four_class_segmentation(
            #                             model, dataset, ERA5_6hrly_ARs, vorticity, wind_speed_selected, nc_file, 
            #                             percentile_threshold=interp_percentile, mslp_data=mslp,
            #                             num_samples=len(safe_indices), indices=safe_indices
            #                         )
            #                         print(f"Successfully saved visualization for {nc_file}")
            #                     except Exception as viz_error:
            #                         print(f"Warning: Visualization failed for {nc_file}: {type(viz_error).__name__}: {viz_error}")
            #                         import traceback
            #                         traceback.print_exc()
            #                 else:
            #                     print(f"Note: Skipping visualization for {nc_file} - auxiliary data not available")
            #                     print(f"  Vorticity available: {vorticity is not None}")
            #                     print(f"  Wind speed available: {wind_speed_selected is not None}")
            #     except Exception as e:
            #         print(f"Error during visualization process for {nc_file}: {type(e).__name__}: {e}")
            #         import traceback
            #         traceback.print_exc()
            
            # Clean up memory - delete variables that were created
            vars_to_delete = [
                'ERA5_6hrly_ARs', 'ivt', 'mslp', 'vorticity', 'preprocessed_data', 
                'dataset', 'train_loader', 'interp_percentile', 'wind_speed_selected', 
                'BT_ds',  'IWV', 'global_percentiles', 'windhshear_ds', 'sampler'
            ]
            
            # Delete variables safely using try-except for each
            for var_name in vars_to_delete:
                try:
                    if var_name in locals():
                        exec(f'del {var_name}')
                except (NameError, UnboundLocalError):
                    pass  # Variable doesn't exist, skip
            
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            if rank == 0:
                print(f"Error processing {nc_file}: {e}")
                # Save checkpoint before failing
                model_to_save = model.module if distributed else model
                torch.save({
                    'epoch': start_epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'losses': all_losses,
                    'ema_state_dict': ema.state_dict()
                }, f'failed_model_checkpoint_rank{rank}.pth')
                
            raise
    
    # Save the final model after training (only on main process)
    if rank == 0:
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        model_to_save = model.module if distributed else model
        final_checkpoint = {
            'epoch': start_epoch + epochs,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'losses': all_losses,
            'training_time': training_time,
            'completed': True,
            'ema_state_dict': ema.state_dict()
        }
        
        # Save final checkpoint
        checkpoint_dir = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Trail1"
        final_checkpoint_path = os.path.join(checkpoint_dir, f'final_checkpoint_{start_year}_{end_year}.pth')
        torch.save(final_checkpoint, final_checkpoint_path)
        
        plot_training_history(all_losses)
        print("Training completed successfully!")

def main():
    """Main entry point that handles distributed training setup"""
    
    parser = argparse.ArgumentParser(description='WNet AR/Hurricane/ETC/MCS Training')
    parser.add_argument('--start_year',  type=int, default=1980, help='Start year (e.g. 1980)')
    parser.add_argument('--end_year',    type=int, default=2024, help='End year (e.g. 2024)')
    parser.add_argument('--start_month', type=int, default=1,    help='Start month 1-12 (e.g. 1)')
    parser.add_argument('--end_month',   type=int, default=12,   help='End month 1-12 (e.g. 12)')
    args = parser.parse_args()

    # Validate month range
    assert 1 <= args.start_month <= 12, "start_month must be between 1 and 12"
    assert 1 <= args.end_month   <= 12, "end_month must be between 1 and 12"

    start_year  = args.start_year
    end_year    = args.end_year
    start_month = args.start_month
    end_month   = args.end_month

    # Setup distributed training
    distributed, world_size, rank, local_rank, device = setup_distributed()
    effective_batch_size = batch_size * world_size * gradient_accumulation_steps
    
    # Synchronize all ranks before proceeding (important for multi-GPU)
    if distributed:
        dist.barrier()
    
    # Configuration - MODIFY THESE PARAMETERS AS NEEDED
    data_directory = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
    
    # Checkpoint configuration
    checkpoint_dir = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Trail1"
    
    # Try to find the latest checkpoint automatically
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    # Or specify a specific checkpoint path:
    # checkpoint_path = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/Checkpointpaths/ema_checkpoint_epoch_40.pth"
    
    # Set to None to start from scratch: 
    # checkpoint_path = None

    # Year/Month range configuration
    # start_year = 1980   # Modify this to set start year
    # end_year = 2024     # Modify this to set end year
    # start_month = 1     # Modify this to set start month (1-12)
    # end_month = 12      # Modify this to set end month (1-12)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Data directory: {data_directory}")
        print(f"  Checkpoint: {checkpoint_path if checkpoint_path else 'None (starting from scratch)'}")
        print(f"  Year/Month range: {start_year}/{start_month:02d} - {end_year}/{end_month:02d}")
        print(f"  Distributed training: {distributed}")
        print(f"  World size: {world_size} GPUs")
        print(f"  Per-GPU batch size: {batch_size}")
        print(f"  Effective batch size: {effective_batch_size} (batch_size × {world_size} GPUs × {gradient_accumulation_steps} grad_accum)")
        print(f"  Device: {device}")
        print(f"{'='*70}\n")
    
    try:
        main_improved(
            data_directory, 
            checkpoint_path=checkpoint_path,
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
            distributed=distributed, 
            world_size=world_size, 
            rank=rank, 
            local_rank=local_rank, 
            device=device
        )
    
    except Exception as e:
        print(f"\nERROR Rank {rank}: Training failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if distributed:
            try:
                dist.barrier()  # Synchronize before cleanup
                cleanup_distributed()
            except Exception as cleanup_error:
                print(f"Warning Rank {rank}: Error during cleanup: {cleanup_error}")


# python main.py --start_year 1980 --end_year 1980 --start_month 3 --end_month 4

if __name__ == "__main__":
    main()


