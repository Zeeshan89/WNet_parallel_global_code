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


def open_wnet_dataset(path):
    """
    Open a RegCM regridded WNet file and normalise its time to a STANDARD
    (datetime64) calendar.

    RegCM CMIP runs use different calendars: ACC/CNR/MPI/MRI are gregorian, but
    BCC and NOR use a 365-day (noleap) calendar → xarray decodes their time as
    cftime.DatetimeNoLeap. That breaks (a) pd.Timestamp(...) used for IBTrACS keys
    and (b) time alignment/merge with the BT files (which are datetime64). Convert
    noleap → standard (align_on='date' just relabels existing steps; no Feb-29 is
    ever present in noleap data, so nothing is added/dropped). Gregorian files are
    a no-op.
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    if 'time' in ds.coords or 'time' in ds.dims:
        try:
            ds = ds.convert_calendar('standard', align_on='date')
        except Exception as e:
            print(f"[open_wnet_dataset] calendar normalise skipped for {path}: {e}")
    return ds


def regcm_checkpoint_dir(model=None):
    """THE single source of truth for the RegCM checkpoint directory.

    main.py (final_checkpoint, loss_history) and train.py (ema_checkpoint_epoch_*)
    both write checkpoints. They used to hardcode this path SEPARATELY, and the two
    copies drifted: main.py was repointed to the precip-MCS run while train.py kept
    writing ema_checkpoint_epoch_*.pth into the BT-era Trail_RegCM_<MODEL>/, silently
    overwriting the A/B baseline. Import this from both instead of re-deriving it.

    `Trail_RegCM_PR_<MODEL>`  — precipitation-based MCS (current).
    `Trail_RegCM_<MODEL>`     — the BT-era baseline: the verified AR/Hurricane/ETC
                                weights and the only reference to A/B against.
                                Never write there.
    """
    m = model or os.environ.get("REGCM_MODEL", "ACC")
    return ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
            "Storms_precipitation_notebooks/WNet_parallel_global_RegCM/"
            f"ARs_Hurricanes_ETCs_MCS/Trail_RegCM_PR_{m}")


def load_regcm_global_stats(model, gs_dir):
    """
    Read RegCM per-model normalization stats from the CDO-produced .txt files in
    <gs_dir> (e.g. ACC_IVT_mean.txt, ACC_IVT_std.txt, …). Returns a dict shaped
    exactly like the legacy hardcoded ERA5 `global_stats`, so it is a drop-in
    replacement in main.py / precompute_tensors.py.

    Keys IVT/MSLP/IWV/BT/vorticity/wind_speed match dataset.py::_compute_statistics.
    temp_500mb is omitted (absent in RegCM → zero-filled; only feeds the inert
    hurricane warm-core filter). BT is included (MCS re-enabled).
    """
    def rd(name):
        with open(os.path.join(gs_dir, f"{model}_{name}.txt")) as fh:
            return float(fh.read().split()[0])

    return {
        k: {"mean": rd(f"{k}_mean"), "std": rd(f"{k}_std")}
        for k in ["IVT", "MSLP", "IWV", "BT", "vorticity", "wind_speed"]
    }


def load_regcm_bt_thresholds(model, gs_dir, default=(241.0, 225.0), prefer_2d=True):
    """
    Per-model MCS brightness-temperature thresholds (shield/core), computed by
    compute_bt_thresholds.py. The fixed 241/225 K literals are calibrated for ERA5
    satellite Tb, but the RegCM cloud-top-temperature proxy is a different field
    whose cold bias is LATITUDE-DEPENDENT (~0 K in the tropics, ~15 K at 40-45 N).

    Resolution order per threshold:
      1. <MODEL>_BT_{boundary,core}_2d.nc — a per-CELL 2-D field matched to ERA5's
         per-cell cold-cloud frequency. PREFERRED: a single scalar cannot serve both
         the tropics and the continental mid-latitudes (a domain-wide scalar skews
         detection ~2x toward the tropics and starves the Great Plains).
      2. <MODEL>_BT_{boundary,core}.txt — domain-wide scalar (legacy fallback).
      3. the ERA5 literals (so the ERA5 pipeline is unaffected).

    Returns {'BT_boundary': <2-D DataArray | float>, 'BT_core': <2-D DataArray | float>}.
    dataset.py accepts either (see _as_threshold); `bt <= thr` broadcasts over (lat, lon).
    """
    def rd(name, fallback):
        if prefer_2d:
            p2 = os.path.join(gs_dir, f"{model}_{name}_2d.nc")
            if os.path.exists(p2):
                try:
                    return load_percentile_dataset(p2)
                except Exception as e:
                    print(f"[load_regcm_bt_thresholds] {p2} unreadable ({e}) → trying scalar")
        p = os.path.join(gs_dir, f"{model}_{name}.txt")
        try:
            with open(p) as fh:
                return float(fh.read().split()[0])
        except (OSError, ValueError):
            print(f"[load_regcm_bt_thresholds] {p} missing → using ERA5 literal {fallback}")
            return float(fallback)

    return {"BT_boundary": rd("BT_boundary", default[0]),
            "BT_core":     rd("BT_core",     default[1])}


def load_regcm_pr_thresholds(model, gs_dir, default=(0.1, 1.0, 0.7), prefer_2d=True):
    """
    MCS precipitation thresholds: convective-precip shield / core / convective
    fraction, all in mm/h except the dimensionless fraction. Returns
    {'PR_shield': …, 'PR_core': …, 'PR_convfrac': …}; dataset.py accepts a scalar
    or a 2-D field (see _as_threshold) and `prc >= thr` broadcasts over (lat, lon).

    THESE ARE FIXED PHYSICAL CONSTANTS BY DESIGN — there is deliberately no
    compute_pr_thresholds.py analogue of compute_bt_thresholds.py. Please read
    this before "improving" it into a per-cell calibration:

      * compute_bt_thresholds.py frequency-matches per-cell against ERA5 because
        RegCM's `T_cloud_top` is a *proxy*, not a measurement, with a
        latitude-dependent cold bias (~0 K tropics, ~15 K at 40-45 N) that is a
        pure artifact of the diagnostic. There, matching is bias-correcting an
        artifact, and it is justified.
      * `pr`/`prc` are real prognostic variables. 1.0 mm/h means the same thing
        in RegCM, in ERA5 and at a rain gauge — there is no instrument-vs-
        diagnostic mismatch to correct.
      * Per-cell ERA5 frequency-matching would be CIRCULAR: it imposes ERA5's
        spatial pattern on the detector by construction, and the model is then
        evaluated against ERA5. The known PNW/Cascades excess would vanish not
        because it was fixed but because it was defined away.
      * Fixed thresholds keep the six-model comparison meaningful. Six per-model
        threshold fields would make inter-model differences calibration residue.

    Defaults (validated on ACC 1990/1995/2000, 6-hourly, area >= 60,000 km^2):
      shield prc >= 0.1 mm/h AND convective fraction >= 0.7; core prc >= 1.0 mm/h
      → Great Plains 21.8, Midwest 23.4, Southeast 34.2 MCS days/yr, with the
      correct seasonality (Great Plains DJF 0.2 vs JJA 12.3) — the thing the
      brightness-temperature branch could never reproduce.

    The 2-D lookup is wired but unused: an override file only exists if someone
    deliberately creates one, and the resolution order keeps that option free.
    """
    def rd(name, fallback):
        if prefer_2d:
            p2 = os.path.join(gs_dir, f"{model}_{name}_2d.nc")
            if os.path.exists(p2):
                try:
                    return load_percentile_dataset(p2)
                except Exception as e:
                    print(f"[load_regcm_pr_thresholds] {p2} unreadable ({e}) → trying scalar")
        p = os.path.join(gs_dir, f"{model}_{name}.txt")
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    return float(fh.read().split()[0])
            except (OSError, ValueError):
                print(f"[load_regcm_pr_thresholds] {p} unreadable → using default {fallback}")
        return float(fallback)

    return {"PR_shield":   rd("PR_shield",   default[0]),
            "PR_core":     rd("PR_core",     default[1]),
            "PR_convfrac": rd("PR_convfrac", default[2])}

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

    # Detect the main variable.
    # NOTE: RegCM CDO outputs name the fields vorticity / wind_speed / IWV / MSLP
    # and also carry a leading `time_bnds` (datetime) variable — the old fallback
    # `list(ds.data_vars)[0]` grabbed `time_bnds`, producing a garbage (all-zero)
    # threshold. Recognise the RegCM names explicitly and skip bounds vars.
    if "IVT" in ds:
        var = ds["IVT"]
    elif "ivt" in ds:
        var = ds["ivt"]
    elif "TPR" in ds:
        var = ds["TPR"]
    elif "wspd" in ds:
        var = ds["wspd"]
    elif "wind_speed" in ds:
        var = ds["wind_speed"]
    elif "var138" in ds:
        var = ds["var138"]
    elif "vorticity" in ds:
        var = ds["vorticity"]
    elif "bt" in ds:
        var = ds["bt"]
    elif "MSLP" in ds:
        var = ds["MSLP"]
    elif "IWV" in ds:
        var = ds["IWV"]
    else:
        # Fallback: first real field, skipping coordinate-bounds variables.
        _skip = ("time_bnds", "lon_bnds", "lat_bnds", "longitude_bnds",
                 "latitude_bnds", "time_bounds", "bnds")
        candidates = [v for v in ds.data_vars
                      if v not in _skip and ds[v].ndim >= 2]
        main_var = candidates[0] if candidates else list(ds.data_vars)[0]
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
