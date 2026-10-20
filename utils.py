import numpy as np
import pandas as pd
import xarray as xr
import warnings
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
import psutil
import os
import glob
## IVT calculation function
def calculate_ivt(test_new_ar):
    # Calculate IVT magnitude
    ivt = np.sqrt(test_new_ar['Qu']**2 + test_new_ar['Qv']**2)

    # Create IVT DataArray with metadata
    ivt_dataarray = xr.DataArray(
        data=ivt,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': test_new_ar['time'],
            'latitude': test_new_ar['latitude'],
            'longitude': test_new_ar['longitude']
        },
        name='IVT',
        attrs={
            'long_name': 'Integrated Vapor Transport',
            'units': 'kg m**-1 s**-1',  # Typical units for moisture flux
            'integration_limit': 'From surface to 200 hPa'
        }
    )

    # Add to original Dataset
    test_new_ar['IVT'] = ivt_dataarray

    return test_new_ar

# --- Helper Functions ---
def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def print_system_info():
    """Print comprehensive system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Check available RAM
    ram = psutil.virtual_memory()
    print(f"Available RAM: {ram.available / (1024 * 1024 * 1024):.2f} GB")
    print(f"Total RAM: {ram.total / (1024 * 1024 * 1024):.2f} GB")
    
    # Check number of CPUs
    cpu_count = os.cpu_count()
    print(f"Number of CPUs: {cpu_count}")
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / (1024**3):.2f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Check if we're in a distributed environment
    if 'WORLD_SIZE' in os.environ:
        print(f"Distributed Training: Yes")
        print(f"World Size: {os.environ['WORLD_SIZE']}")
        print(f"Rank: {os.environ.get('RANK', 'N/A')}")
        print(f"Local Rank: {os.environ.get('LOCAL_RANK', 'N/A')}")
    else:
        print(f"Distributed Training: No")
    
    print("=" * 60)



def setup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False   # Disable MIOpen kernel search
        # Set memory growth
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.9, i)

# Method 1: Simple text representation with torchinfo
def model_summary_to_file(model, input_size=(1, 6, 256, 256), filename="model_summary.txt"):
    """
    Generate a detailed text summary of model architecture and save to file.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        filename: Output file name
    """
    # Get summary object
    model_summary = summary(model, input_size=input_size, verbose=0, 
                           depth=10, col_names=["input_size", "output_size", 
                                              "kernel_size", "num_params"])
    
    # Print to console
    print(model_summary)
    
    # Save to file
    # with open(filename, 'w') as f:
    #     print(model_summary, file=f)
    
    # print(f"Model summary saved to {filename}")
    return model_summary


# Function to find matching vorticity file for a given date
def find_matching_vorticity(era5_date, vorticity_files):
        year = pd.Timestamp(era5_date).year
        for vfile in vorticity_files:
            file_year = int(os.path.basename(vfile).split('_')[3][:4])
            if file_year == year:
                return vfile
        return None
    

def load_checkpoint(model, optimizer, scheduler, scaler, ema, checkpoint_path, device):
    """
    Load training checkpoint
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):    
        # At the top (no checkpoint found):
        empty_history = {
            "total": [], "ar": [], "hurricane": [],
            "etc": [], "mcs": [], "overlap": [],
            "file_boundaries": [], "file_names": [], "lr": []
        }
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            if checkpoint_path is not None:
                print(f"Checkpoint file not found: {checkpoint_path}")
            print("Starting training from scratch")
            return 0, None, empty_history
    
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load EMA state
        if ema and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        # Handle both old (plain list) and new (dict) loss history formats
        raw_losses = checkpoint.get('losses', None)
        if raw_losses is None:
            loss_history = {
                "total": [], "ar": [], "hurricane": [],
                "etc": [], "mcs": [], "overlap": [],
                "file_boundaries": [], "file_names": [], "lr": []
            }
        elif isinstance(raw_losses, list):
            # Old format — wrap it so main.py gets a dict
            loss_history = {
                "total": raw_losses, "ar": [], "hurricane": [],
                "etc": [], "mcs": [], "overlap": [],
                "file_boundaries": [], "file_names": [], "lr": []
            }
        else:
            # Already new dict format
            loss_history = raw_losses
        
        print(f"Checkpoint loaded successfully")
        print(f"Resuming from epoch {epoch+1}, loss: {loss}")
        
        return epoch, loss, loss_history
        
    
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Starting training from scratch")
        return 0, None, empty_history

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint file in the directory
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ema_checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoint_files[-1]


def load_percentile_dataset(path, rank=0, lat_name="lat", lon_name="lon"):
    """
    Opens a NetCDF file, fixes longitude (0–360 → -180–180), 
    extracts the main variable, renames coordinates, 
    and squeezes singleton dimensions.
    """
    ds = xr.open_dataset(path)

    # Handle bounds if present
    if "bnds" in ds.dims:
        try:
            ds = ds.sel(bnds=1)
        except Exception:
            pass

    # Rename coordinates if necessary to match ERA5 conventions
    if lon_name in ds.dims and "longitude" not in ds.dims:
        ds = ds.rename({lon_name: "longitude", lat_name: "latitude"})

    # Rename time if stored as valid_time
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Fix longitude range if present
    for lon_key in ["lon", "longitude"]:
        if lon_key in ds.coords:
            ds = ds.assign_coords({lon_key: ((ds[lon_key] + 180) % 360) - 180}).sortby(lon_key)

    # Detect the main variable
    if "IVT" in ds:
        var = ds["IVT"]
    elif "ivt" in ds:
        var = ds["ivt"]
    elif "TPR" in ds:
        var = ds["TPR"]
    elif "wspd" in ds:
        var = ds["wspd"]
    elif "var138" in ds:
        var = ds["var138"]
    elif "bt" in ds:
        var = ds["bt"]
    elif "MSLP" in ds:
        var = ds["MSLP"]
    else:
        main_var = list(ds.data_vars)[0]
        var = ds[main_var]
        if rank == 0:
            print(f"Using {main_var} as threshold variable from {path}")

    # Squeeze out singleton dims (time, plev, level, etc.)
    var = var.squeeze()

    # Drop unneeded coords if they are scalar (plev, level, etc.)
    for coord in ["plev", "level", "bnds"]:
        if coord in var.coords and var[coord].size == 1:
            var = var.drop_vars(coord)

    return var
