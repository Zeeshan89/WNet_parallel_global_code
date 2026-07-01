import os

# ── MIOpen: per-process SQLite cache + skip kernel search ────────────────────
# MUST happen before `import torch`. Without this, MIOpen's default per-user
# SQLite db on the home/lustre filesystem hits "disk I/O error" on Frontier and
# every Conv2d fails with miopenStatusInternalError. Mirrors main.py's block.
_rank = int(os.environ.get("SLURM_PROCID", 0))
_world = int(os.environ.get("SLURM_NTASKS", 1))   # for multi-GPU file sharding
_miopen_cache = f"/tmp/miopen-cache-rank-{_rank}"
os.makedirs(_miopen_cache, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"]     = _miopen_cache
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _miopen_cache
os.environ["MIOPEN_FIND_MODE"]        = "2"   # FAST: no search, use defaults
os.environ["MIOPEN_DISABLE_CACHE"]    = "0"
# ─────────────────────────────────────────────────────────────────────────────

import gc
import glob
import numpy as np
import torch
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature9
from torch import nn, optim
from skimage.transform import resize
import argparse
from pathlib import Path

# Add missing imports for post-processing
import math
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_fill_holes, zoom, binary_dilation as scipy_binary_dilation
from skimage.measure import regionprops

# Import your model and helper functions - make sure these imports match your project structure
# You might need to adjust these imports based on your actual file structure
from WNet_model import ImprovedWNet
from dataset import ARDatasetImproved, preprocess_data_improved
from post_processing import enhanced_post_process_ar_segmentation, track_mcs_timeseries
from utils import setup_gpu, calculate_ivt, find_matching_vorticity, load_percentile_dataset
from scipy.ndimage import binary_opening, binary_closing
import numpy as np
import torch
import xarray as xr
import os
import glob
import gc
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib
from ibtracs_loader import load_ibtracs_for_training
from distributed_dataset import get_rank_files

# Set matplotlib backend for non-interactive environments
matplotlib.use('Agg')

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_size = (360, 720)  # Default size, can be adjusted


def clean_edge_artifacts(mask, seam_cols=2, pole_rows=2):
    """Remove CNN zero-padding edge stripes from a 2-D detection map (lat, lon).

    The WNet uses zero-padding convolutions, so on a global periodic field the
    network sees zeros past the antimeridian (±180°) and the poles and can emit a
    thin stripe of spurious/dropped detections in the outermost columns/rows.

    Longitude seam (±180°) = first/last columns; poles = first/last rows.
    Each edge band is overwritten with its nearest *interior* column/row, so a
    spurious edge-only line (interior neighbour = 0) is erased, while a genuine
    storm that extends inward (interior neighbour = 1) is preserved.
    """
    if mask is None or getattr(mask, 'ndim', None) != 2:
        return mask
    out = mask.copy()
    nlat, nlon = out.shape
    # Longitude seam — vertical lines
    if seam_cols > 0 and nlon > 2 * seam_cols:
        out[:, :seam_cols]  = out[:, seam_cols][:, None]
        out[:, -seam_cols:] = out[:, -seam_cols - 1][:, None]
    # Poles — horizontal lines
    if pole_rows > 0 and nlat > 2 * pole_rows:
        out[:pole_rows, :]  = out[pole_rows, :][None, :]
        out[-pole_rows:, :] = out[-pole_rows - 1, :][None, :]
    return out


def save_results_to_netcdf(ar_results, hurricane_results, etcs_results, mcs_results, original_dataset, output_dir, nc_filename, mcs_id_results=None):
    """
    Save post-processed AR, hurricane, ETC, and MCS results to separate NetCDF files.
    
    Args:
        ar_results: numpy array of shape (time, lat, lon) with AR detection results
        hurricane_results: numpy array of shape (time, lat, lon) with hurricane detection results
        etcs_results: numpy array of shape (time, lat, lon) with ETC detection results
        mcs_results: numpy array of shape (time, lat, lon) with MCS detection results (NEW)
        original_dataset: original ERA5 dataset with coordinates and time information
        output_dir: directory to save the results
        nc_filename: original filename to base the output names on
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinates from original dataset
    time_coords = original_dataset.time
    lat_coords = original_dataset.latitude
    lon_coords = original_dataset.longitude
    
    # Create AR dataset
    ar_ds = xr.Dataset(
        {
            'AR_detection': (['time', 'latitude', 'longitude'], ar_results),
        },
        coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        },
        attrs={
            'title': 'Atmospheric River Detection Results',
            'description': 'Post-processed AR detection from WNet model',
            'model': 'ImprovedWNet',
            'source': 'WNet-based storm detection system',
            'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    
    # Add variable attributes
    ar_ds['AR_detection'].attrs = {
        'long_name': 'Atmospheric River Detection Probability',
        'units': 'probability',
        'valid_range': [0.0, 1.0],
        'description': 'Post-processed atmospheric river detection probability from WNet segmentation'
    }
    
    
    # Create Hurricane dataset
    hurricane_ds = xr.Dataset(
        {
            'Hurricane_detection': (['time', 'latitude', 'longitude'], hurricane_results),
        },
        coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        },
        attrs={
            'title': 'Hurricane Detection Results',
            'description': 'Post-processed hurricane detection from WNet model',
            'model': 'ImprovedWNet',
            'source': 'WNet-based storm detection system',
            'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    
    # Add variable attributes
    hurricane_ds['Hurricane_detection'].attrs = {
        'long_name': 'Hurricane Detection Probability',
        'units': 'probability',
        'valid_range': [0.0, 1.0],
        'description': 'Post-processed hurricane detection probability from WNet segmentation'
    }
    
    # Create ETCs dataset
    etcs_ds = xr.Dataset(
        {
            'ETCs_detection': (['time', 'latitude', 'longitude'], etcs_results),
        },
        coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        },
        attrs={
            'title': 'ETCs Detection Results',
            'description': 'Post-processed ETC detection from WNet model',
            'model': 'ImprovedWNet',
            'source': 'WNet-based storm detection system',
            'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    
    # Add variable attributes
    etcs_ds['ETCs_detection'].attrs = {
        'long_name': 'ETCs Detection Probability',
        'units': 'probability',
        'valid_range': [0.0, 1.0],
        'description': 'Post-processed ETCs detection probability from WNet segmentation'
    }
    
    
    # Create MCS dataset
    mcs_ds = xr.Dataset(
        {
            'MCS_detection': (['time', 'latitude', 'longitude'], mcs_results),
        },
        coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        },
        attrs={
            'title': 'MCS Detection Results',
            'description': 'Post-processed MCS detection from WNet model',
            'model': 'ImprovedWNet',
            'source': 'WNet-based storm detection system',
            'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    # Add variable attributes
    mcs_ds['MCS_detection'].attrs = {
        'long_name': 'Mesoscale Convective System Detection Probability',
        'units': 'probability',
        'valid_range': [0.0, 1.0],
        'description': 'Post-processed MCS detection probability from WNet segmentation'
    }

    # Tracked storm-ID field (analogous to Feng's cloudtracknumber) so the shared
    # benchmark metric code can compute true ID-based storm-hours for WNet.
    if mcs_id_results is not None:
        mcs_ds['MCS_id'] = (['time', 'latitude', 'longitude'], mcs_id_results)
        mcs_ds['MCS_id'].attrs = {
            'long_name': 'MCS storm track ID',
            'units': 'id',
            'description': 'Overlap-tracked MCS storm ID per pixel (0 = none); '
                           'IDs are per monthly file. Analogous to FLEXTRKR/Feng cloudtracknumber.'
        }

    # Generate output filenames
    base_name = nc_filename.replace('.nc', '')
    ar_filename = f"{base_name}_AR_detection.nc"
    hurricane_filename = f"{base_name}_Hurricane_detection.nc"
    etc_filename = f"{base_name}_ETCs_detection.nc"
    mcs_filename = f"{base_name}_MCS_detection.nc"
    
    ar_filepath = os.path.join(output_dir, ar_filename)
    hurricane_filepath = os.path.join(output_dir, hurricane_filename)
    etc_filepath = os.path.join(output_dir, etc_filename)
    mcs_filepath = os.path.join(output_dir, mcs_filename)
    
    # Save datasets
    print(f"Saving AR results to: {ar_filepath}")
    ar_ds.to_netcdf(ar_filepath, format='NETCDF4', engine='netcdf4')
    
    print(f"Saving Hurricane results to: {hurricane_filepath}")
    hurricane_ds.to_netcdf(hurricane_filepath, format='NETCDF4', engine='netcdf4')
    
    print(f"Saving ETCs results to: {etc_filepath}")
    etcs_ds.to_netcdf(etc_filepath, format='NETCDF4', engine='netcdf4')
    
    print(f"Saving MCS results to: {mcs_filepath}")
    mcs_ds.to_netcdf(mcs_filepath, format='NETCDF4', engine='netcdf4')
    
    # Clean up
    ar_ds.close()
    hurricane_ds.close()
    etcs_ds.close()
    mcs_ds.close()
    
    
    return ar_filepath, hurricane_filepath, etc_filepath, mcs_filepath

def convert_xarray_to_numpy(data):
    """
    Convert xarray DataArray to numpy array, handling the conversion properly.
    """
    if isinstance(data, xr.DataArray):
        return data.values
    elif isinstance(data, xr.Dataset):
        # If it's a dataset, try to get the main variable
        main_vars = list(data.data_vars)
        if main_vars:
            return data[main_vars[0]].values
        else:
            raise ValueError("Dataset has no data variables")
    else:
        return data

def evaluate_and_save_results(model, dataset, era5_dataset, vorticity_data, wind_speed_data, temp_500mb, nc_file, 
                            percentile_threshold=None, output_dir=None):
    """
    Evaluate model on all samples in the dataset and save results to NetCDF files.
    Now includes MCS detection as the 4th storm class.
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get time values and coordinates
    time_values = era5_dataset.time.values
    lats = era5_dataset.latitude.values
    lons = era5_dataset.longitude.values
    
    # Check if latitude is descending
    lat_is_descending = lats[0] > lats[-1]
    
    # Initialize result arrays
    num_samples = len(dataset)
    ar_results = np.zeros((num_samples, len(lats), len(lons)), dtype=np.float32)
    hurricane_results = np.zeros((num_samples, len(lats), len(lons)), dtype=np.float32)
    etcs_results = np.zeros((num_samples, len(lats), len(lons)), dtype=np.float32)
    mcs_results = np.zeros((num_samples, len(lats), len(lons)), dtype=np.float32)
    
    print(f"Processing {num_samples} samples from {nc_file}")
    
    # Convert percentile threshold to numpy once outside the loop
    percentile_threshold_numpy = None
    if percentile_threshold is not None:
        percentile_threshold_numpy = convert_xarray_to_numpy(percentile_threshold)
        # Keep scalar thresholds as scalars. If array, collapse time dim and resize spatially if needed.
        if np.isscalar(percentile_threshold_numpy):
            percentile_threshold_numpy = float(percentile_threshold_numpy)
        else:
            arr = np.asarray(percentile_threshold_numpy)
            if arr.ndim == 3:
                arr = np.mean(arr, axis=0)
            if arr.shape != (len(lats), len(lons)):
                try:
                    arr = resize(arr, (len(lats), len(lons)), order=1, mode='reflect', anti_aliasing=False)
                except Exception:
                    arr = arr.astype(np.float32)
                    arr = np.resize(arr, (len(lats), len(lons)))
            percentile_threshold_numpy = arr.astype(np.float32)
    
    with torch.no_grad():
        for index in range(num_samples):
            if index % 100 == 0:  # Progress indicator
                print(f"Processing sample {index}/{num_samples}")
            
            # Get sample and model prediction
            sample = dataset[index]
            data = sample.unsqueeze(0).to(device)
            seg_probs, rec = model(data)
            
            # Get original data for this timestep
            if isinstance(dataset, torch.utils.data.Subset):
                actual_index = dataset.indices[index]
            else:
                actual_index = index
            
            # Get the segmentation map
            seg_map = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
            
            # Debug: Print segmentation map statistics
            unique_values, counts = np.unique(seg_map, return_counts=True)
            print(f"Sample {index}: Segmentation classes found: {dict(zip(unique_values, counts))}")
        
            # Get original data for post-processing
            sample_np = sample.cpu().numpy()
            hurricane_eye_mask = sample_np[5]
            etc_eye_mask = sample_np[10]
            
            
            # Get original data from ERA5 dataset
            ivt_data_orig = era5_dataset.IVT.values[actual_index]
            
            vor_data_orig = None
            if vorticity_data is not None:
                if 0 <= actual_index < vorticity_data.shape[0]:
                    vor_data_orig = vorticity_data.values[actual_index]
                else:
                    print(
                        f"Skipping vorticity: actual_index={actual_index}, "
                        f"size={vorticity_data.shape[0]}"
                    )
            
            wind_speed_data_orig = None
            if wind_speed_data is not None:
                wind_speed_data_orig = wind_speed_data.values[actual_index]
            
            mslp_data_orig = era5_dataset.MSLP.values[actual_index] if 'MSLP' in era5_dataset else None
            
            # Get target size from the dataset (same as model output)
            target_size = seg_map.shape
            
            # Resize original data to match model output size for processing
            if ivt_data_orig.shape != target_size:
                ivt_data_resized = resize(ivt_data_orig, target_size, order=1, mode='reflect', anti_aliasing=False)
            else:
                ivt_data_resized = ivt_data_orig
                
            if vor_data_orig is not None and vor_data_orig.shape != target_size:
                vor_data_resized = resize(vor_data_orig, target_size, order=1, mode='reflect', anti_aliasing=False)
            else:
                vor_data_resized = vor_data_orig
                
            if wind_speed_data_orig is not None and wind_speed_data_orig.shape != target_size:
                wind_speed_data_resized = resize(wind_speed_data_orig, target_size, order=1, mode='reflect', anti_aliasing=False)
            else:
                wind_speed_data_resized = wind_speed_data_orig
                
            if mslp_data_orig is not None and mslp_data_orig.shape != target_size:
                mslp_data_resized = resize(mslp_data_orig, target_size, order=1, mode='reflect', anti_aliasing=False)
            else:
                mslp_data_resized = mslp_data_orig
            

            seg_map_for_processing = seg_map
            ivt_data_processing = ivt_data_resized
            vor_data_processing = vor_data_resized
            wind_speed_data_processing = wind_speed_data_resized
            mslp_data_processing = mslp_data_resized
            hurricane_eye_mask_processing = hurricane_eye_mask
            etcs_eye_mask_processing = etc_eye_mask
            
            # Handle percentile threshold (allow scalar or array)
            if percentile_threshold_numpy is not None:
                if np.isscalar(percentile_threshold_numpy):
                    percentile_threshold_processing = percentile_threshold_numpy
                else:
                    if percentile_threshold_numpy.shape == target_size:
                        percentile_threshold_processing = percentile_threshold_numpy
                    else:
                        percentile_threshold_processing = resize(percentile_threshold_numpy, target_size, 
                                                                   order=1, mode='reflect', anti_aliasing=False)
            else:
                percentile_threshold_processing = None
            
            # Extract individual class maps
            ar_map = np.zeros_like(seg_map_for_processing)
            hurricane_map = np.zeros_like(seg_map_for_processing)
            etcs_map = np.zeros_like(seg_map_for_processing)
            
            
            # Class 1 = AR, Class 2 = Hurricane
            ar_map[seg_map_for_processing == 1] = 1
            hurricane_map[seg_map_for_processing == 2] = 1
            etcs_map[seg_map_for_processing == 3] = 1
            
            mcs_map = np.zeros_like(seg_map_for_processing)
            mcs_map[seg_map_for_processing == 4] = 1  # Class 4 = MCS
            
            print(f"Before post-processing - AR pixels: {np.sum(ar_map)}, ETCs pixels: {np.sum(etcs_map)}, Hurricane pixels: {np.sum(hurricane_map)}, MCS pixels: {np.sum(mcs_map)}")
            
            
            # Post-process AR detection
            try:
                post_processed_ar_map = enhanced_post_process_ar_segmentation(
                    ar_map, 
                    storm_class=1,
                    ivt_data=ivt_data_processing,
                    vor_data=None,
                    mslp_data=None,
                    percentile_threshold=percentile_threshold_processing,  # Pass numpy array
                    min_size=50,
                    min_aspect_ratio=2.0,
                    slope_threshold=0.3,
                    equatorial_band_size=0.3,
                    km_per_pixel=50,
                    min_length_km=2000,
                    min_span_km=1000,
                    sample_data=sample_np
                )
                
                            
                print(f"AR post-processing successful - pixels: {np.sum(post_processed_ar_map)}")
            except Exception as e:
                print(f"Error in AR post-processing: {e}")
                post_processed_ar_map = ar_map  # Use original if post-processing fails
            
            # Post-process Hurricane detection with improved parameters
            try:
                post_processed_hurricane_map = enhanced_post_process_ar_segmentation(
                        hurricane_map,
                        storm_class=2,
                        ivt_data=None,
                        vor_data=vor_data_processing,
                        mslp_data=mslp_data_processing,
                        percentile_threshold=None,
                        min_size=1,  # Very small minimum for hurricane eyes
                        km_per_pixel=27.75,  # Adjusted for 0.25 degree resolution
                        vor_threshold=4.6e-05,
                        mslp_eye_threshold=101000,  # 1010 hPa in Pa
                        lat_bounds=None,  # Not applied since already in channels
                        gradient_threshold=0.5,
                        sample_data=sample_np  # This contains all the enhanced channels
                )
                
               
                print(f"Hurricane post-processing successful - pixels: {np.sum(post_processed_hurricane_map)}")
            except Exception as e:
                print(f"Error in Hurricane post-processing: {e}")
                # Instead of using raw segmentation, apply basic filtering
                post_processed_hurricane_map = hurricane_map.copy()
                # Apply basic morphological operations to clean up hurricane detection
                if np.sum(post_processed_hurricane_map) > 0:
                    # Remove small isolated pixels
                
                    kernel = np.ones((3, 3))
                    post_processed_hurricane_map = binary_opening(post_processed_hurricane_map, kernel)
                    post_processed_hurricane_map = binary_closing(post_processed_hurricane_map, kernel)
                    post_processed_hurricane_map = post_processed_hurricane_map.astype(np.float32)
                
                print(f"Using basic filtered hurricane segmentation - pixels: {np.sum(post_processed_hurricane_map)}")

            # Post-process ETCs detection
            try:
                post_processed_etc_map = enhanced_post_process_ar_segmentation(
                    etcs_map,
                    storm_class=3,  # New ETC class
                    ivt_data=None,
                    vor_data=vor_data_processing,
                    mslp_data=mslp_data_processing,
                    percentile_threshold=None,
                    min_size=10,  # Moderate minimum size for ETCs
                    km_per_pixel=27.75,
                    vor_threshold=1e-5,  # Slightly lower vorticity threshold for ETCs
                    mslp_eye_threshold=101500,  # 1000 hPa threshold for ETCs
                    lat_bounds=None,
                    gradient_threshold=0.4,
                    sample_data=sample_np
                )

                            
                print(f"ETCs post-processing successful - pixels: {np.sum(post_processed_etc_map)}")
            except Exception as e:
                print(f"Error in AR post-processing: {e}")
                post_processed_etc_map = etcs_map  # Use original if post-processing fails
            
            # Flip back to original orientation if needed and resize to original grid
            # if lat_is_descending:
            #     post_processed_ar_map_final = np.flipud(post_processed_ar_map)
            #     post_processed_hurricane_map_final = np.flipud(post_processed_hurricane_map)
            # else:
            
            # MCS: keep RAW model blobs — no per-frame model/physics filtering.
            # Area (60,000 km²) + temporal-overlap (50–90%) are applied to the full
            # month time series after this loop (post_process_mcs_timeseries).
            post_processed_mcs_map = mcs_map.copy()
            print(f"MCS raw blobs kept (no per-frame filter) - pixels: {np.sum(post_processed_mcs_map)}")

            
            post_processed_ar_map_final = post_processed_ar_map
            
            post_processed_hurricane_map_final = post_processed_hurricane_map
            post_processed_etc_map_final = post_processed_etc_map
            post_processed_mcs_map_final = post_processed_mcs_map
            
            # Resize back to original grid size
            if post_processed_ar_map_final.shape != (len(lats), len(lons)):
                post_processed_ar_map_final = resize(post_processed_ar_map_final, (len(lats), len(lons)), 
                                                   order=0, mode='reflect', anti_aliasing=False)
            
            if post_processed_hurricane_map_final.shape != (len(lats), len(lons)):
                post_processed_hurricane_map_final = resize(post_processed_hurricane_map_final, (len(lats), len(lons)), 
                                                          order=0, mode='reflect', anti_aliasing=False)
            
            
            if post_processed_etc_map_final.shape != (len(lats), len(lons)):
                post_processed_etc_map_final = resize(post_processed_etc_map_final, (len(lats), len(lons)), 
                                                          order=0, mode='reflect', anti_aliasing=False)
                
                
            # Resize back to original grid size
            if post_processed_mcs_map_final.shape != (len(lats), len(lons)):
                post_processed_mcs_map_final = resize(post_processed_mcs_map_final, (len(lats), len(lons)), 
                                                    order=0, mode='reflect', anti_aliasing=False)


            # Clean CNN zero-padding edge stripes (vertical ±180° seam, horizontal poles)
            post_processed_ar_map_final        = clean_edge_artifacts(post_processed_ar_map_final)
            post_processed_hurricane_map_final = clean_edge_artifacts(post_processed_hurricane_map_final)
            post_processed_etc_map_final       = clean_edge_artifacts(post_processed_etc_map_final)
            post_processed_mcs_map_final       = clean_edge_artifacts(post_processed_mcs_map_final)

            # Store results
            ar_results[index] = post_processed_ar_map_final
            hurricane_results[index] = post_processed_hurricane_map_final
            etcs_results[index] = post_processed_etc_map_final
            mcs_results[index] = post_processed_mcs_map_final
            
            print(f"Final results - AR pixels: {np.sum(post_processed_ar_map_final)}, ETCs pixels: {np.sum(post_processed_etc_map_final)}, Hurricane pixels: {np.sum(post_processed_hurricane_map_final)}, MCS pixels: {np.sum(post_processed_mcs_map_final)}")
    
    # MCS: RAW model blobs + overlap tracking → storm-ID field (MCS_id), so the
    # benchmark metric can compute true ID-based storm-hours for WNet.
    # OVERLAP-ONLY: no minimum-area criterion, no minimum-lifetime pruning.
    mcs_results, mcs_id_results = track_mcs_timeseries(
        mcs_results, lats, lons,
        min_area_km2=None, link_overlap=0.5, min_lifetime_steps=1,
    )

    # Save results to NetCDF files
    ar_filepath, hurricane_filepath, etc_filepath, mcs_filepath = save_results_to_netcdf(
        ar_results, hurricane_results, etcs_results, mcs_results, era5_dataset, output_dir, nc_file,
        mcs_id_results=mcs_id_results,
    )
    
    print(f"Completed processing {nc_file}")
    print(f"AR detection results saved to: {ar_filepath}")
    print(f"Hurricane detection results saved to: {hurricane_filepath}")
    print(f"ETCs detection results saved to: {etc_filepath}")
    print(f"MCS detection results saved to: {mcs_filepath}")
    
    
    return ar_filepath, hurricane_filepath, etc_filepath, mcs_filepath


def visualize_segmentation_improved(model, dataset, era5_dataset, vorticity, wind_speed_selected, nc_file, vor_dataset=None, 
                                   mslp_data=None, percentile_threshold=None, num_samples=3, output_dir=None):
    """
    Enhanced visualization for three-class storm detection: ARs, Hurricanes, and ETCs.
    Updated to match the new visualization function with 15 plots including ETC-related channels.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure we don't try to sample more than available
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    time_values = era5_dataset.time.values
    lats = era5_dataset.latitude.values
    lons = era5_dataset.longitude.values
    target_size = dataset[0].shape[-2:]
    lat_is_descending = lats[0] > lats[-1]
    
    if lat_is_descending:
        lats_ds = np.linspace(lats.max(), lats.min(), target_size[0])
    else:
        lats_ds = np.linspace(lats.min(), lats.max(), target_size[0])
    lons_ds = np.linspace(lons.min(), lons.max(), target_size[1])
    lon, lat = np.meshgrid(lons_ds, lats_ds)
    
    # Three-class colormap: white=background, blue=AR, red=Hurricane, green=ETC
    four_class_cmap = ListedColormap(['white', 'blue', 'red', 'green', 'orange'])
    
    # Create larger figure for 14 subplots (5 rows x 3 columns per sample)
    fig, axes = plt.subplots(num_samples, 16, figsize=(75, 5 * num_samples), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
    
    with torch.no_grad():
        for i, index in enumerate(indices):
            try:
                sample = dataset[index]
                data = sample.unsqueeze(0).to(device)
                seg_probs, rec = model(data)
                
                if isinstance(dataset, torch.utils.data.Subset):
                    actual_index = dataset.indices[index]
                else:
                    actual_index = index
                
                # Extract all channels including new ETC channels
                sample_np = sample.cpu().numpy()
                
                # Original channels (0-11)
                # ivt_sample = sample_np[0]
                # mslp_sample = sample_np[1]
                # ivt_grad_sample = sample_np[2]
                # ivt_binary = sample_np[3]
                # vorticity_sample = sample_np[4]
                # vorticity_grad_sample = sample_np[5]
                # wind_speed_sample = sample_np[6]
                # hurricane_low_pressure = sample_np[7]
                # hurricane_high_vorticity = sample_np[8]
                # hurricane_high_wind_speed = sample_np[9]
                hurricane_strict = sample_np[5] if sample_np.shape[0] > 5 else np.zeros(target_size)
                # wind_speed_binary = sample_np[11]

                # New ETC channels (12-16) - handle case where they might not exist
                etc_low_pressure = sample_np[7] if sample_np.shape[0] > 7 else np.zeros(target_size)
                etc_vorticity = sample_np[8] if sample_np.shape[0] > 8 else np.zeros(target_size)
                etc_wind_speed = sample_np[9] if sample_np.shape[0] > 9 else np.zeros(target_size)
                etc_strict = sample_np[10] if sample_np.shape[0] > 10 else np.zeros(target_size)
                mcs_strict = sample_np[13] if sample_np.shape[0] > 13 else np.zeros(target_size)



                ivt_data_orig = era5_dataset.IVT.values[actual_index]
                vor_data_orig = None
                if vorticity is not None:
                    vor_data_orig = vorticity.values[actual_index]
                elif vor_dataset is not None:
                    timestamp = pd.Timestamp(time_values[actual_index])
                    vor_time_idx = np.abs(pd.to_datetime(vor_dataset.time.values) - timestamp).argmin()
                    vor_data_orig = vor_dataset.var138.values[vor_time_idx]
                    # Debug vorticity timestamp alignment
                    print(f"  Vorticity timestamp: {pd.to_datetime(vor_dataset.time.values)[vor_time_idx]}")
                    print(f"  Time difference: {abs(pd.to_datetime(vor_dataset.time.values)[vor_time_idx] - timestamp)}")
                
                mslp_data_orig = None
                if mslp_data is not None:
                    if hasattr(mslp_data, 'values'):
                        mslp_data_orig = mslp_data.values[actual_index]
                    else:
                        mslp_data_orig = mslp_data[actual_index]
                elif 'MSLP' in era5_dataset:
                    mslp_data_orig = era5_dataset.MSLP.values[actual_index]
                        
                # Process wind data with proper error handling
                wind_data_orig = None
                if wind_speed_selected is not None:
                    try:
                        wind_data_orig = wind_speed_selected.values[actual_index]
                    except (IndexError, KeyError):
                        # Fallback if indexing fails
                        timestamp = pd.Timestamp(time_values[actual_index])
                        wind_time_idx = np.abs(pd.to_datetime(wind_speed_selected.time.values) - timestamp).argmin()
                        wind_data_orig = wind_speed_selected.values[wind_time_idx]
                        
                # Resize data to target size if needed
                def resize_data(data, target_size):
                    if data is not None and data.shape != target_size:
                        return resize(data, target_size, order=1, mode='reflect', anti_aliasing=False)
                    return data

                ivt_plot_orig = resize_data(ivt_data_orig, target_size)
                vor_plot_orig = resize_data(vor_data_orig, target_size)
                mslp_plot_orig = resize_data(mslp_data_orig, target_size)
                wind_plot_orig = resize_data(wind_data_orig, target_size)
                    
                # Ground truth processing
                gt_plot = np.zeros(target_size, dtype=np.float32)
                if 'ARS' in era5_dataset:
                    gt = era5_dataset['ARS'].values[actual_index]
                    if gt.shape != target_size:
                        gt_plot = resize(gt, target_size, order=0, mode='reflect', anti_aliasing=False)
                    else:
                        gt_plot = gt
                else:
                    print("  No ground truth data available")
                
                # Get raw segmentation
                seg_map = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
                if seg_map.shape != target_size:
                    seg_map_plot = resize(seg_map, target_size, order=0, mode='reflect', anti_aliasing=False)
                else:
                    seg_map_plot = seg_map
                
                # Separate storm classes for post-processing
                ar_map = np.zeros_like(seg_map_plot)
                hurricane_map = np.zeros_like(seg_map_plot)
                etc_map = np.zeros_like(seg_map_plot)
                
                ar_map[seg_map_plot == 1] = 1
                hurricane_map[seg_map_plot == 2] = 1
                etc_map[seg_map_plot == 3] = 1  # New ETC class
                mcs_map = np.zeros_like(seg_map_plot)
                mcs_map[seg_map_plot == 4] = 1  # New MCS class

                # Post-process with error handling
                try:
                    post_processed_ar_map = enhanced_post_process_ar_segmentation(
                        ar_map, 
                        storm_class=1,
                        ivt_data=ivt_plot_orig,
                        vor_data=None,
                        mslp_data=None,
                        percentile_threshold=percentile_threshold,
                        min_size=50,
                        min_aspect_ratio=2.0,
                        slope_threshold=0.3,
                        equatorial_band_size=0.3,
                        km_per_pixel=50,
                        min_length_km=2000,
                        min_span_km=1000,
                        sample_data=sample_np
                    )
                except Exception as e:
                    print(f"Error in AR post-processing for visualization: {e}")
                    post_processed_ar_map = ar_map
                
                # Post-process Hurricane detection with enhanced channels
                try:
                    post_processed_hurricane_map = enhanced_post_process_ar_segmentation(
                        hurricane_map,
                        storm_class=2,
                        ivt_data=None,
                        vor_data=vor_plot_orig,
                        mslp_data=mslp_plot_orig,
                        percentile_threshold=None,
                        min_size=1,  # Very small minimum for hurricane eyes
                        km_per_pixel=27.75,  # Adjusted for 0.25 degree resolution
                        vor_threshold=4.6e-05,
                        mslp_eye_threshold=101000,  # 1010 hPa in Pa
                        lat_bounds=None,  # Not applied since already in channels
                        gradient_threshold=0.5,
                        sample_data=sample_np  # This contains all the enhanced channels
                    )
                    
                except Exception as e:
                    print(f"Error in Hurricane post-processing for visualization: {e}")
                    post_processed_hurricane_map = hurricane_map

                # Post-process ETC detection
                try:
                    post_processed_etc_map = enhanced_post_process_ar_segmentation(
                        etc_map,
                        storm_class=3,  # New ETC class
                        ivt_data=None,
                        vor_data=vor_plot_orig,
                        mslp_data=mslp_plot_orig,
                        percentile_threshold=None,
                        min_size=10,  # Moderate minimum size for ETCs
                        km_per_pixel=27.75,
                        vor_threshold=1e-5,  # Slightly lower vorticity threshold for ETCs
                        mslp_eye_threshold=101500,  # 1000 hPa threshold for ETCs
                        lat_bounds=None,
                        gradient_threshold=0.4,
                        sample_data=sample_np
                    )
                except Exception as e:
                    print(f"Error in ETC post-processing for visualization: {e}")
                    post_processed_etc_map = etc_map
                
                # Post-process MCS detection
                try:
                    post_processed_mcs_map = enhanced_post_process_ar_segmentation(
                        mcs_map,
                        storm_class=4,  # New MCS class
                        ivt_data=None,
                        vor_data=vor_plot_orig,
                        mslp_data=mslp_plot_orig,
                        percentile_threshold=None,
                        min_size=1,  # Moderate minimum size for ETCs
                        km_per_pixel=27.75,
                        vor_threshold=1e-5,  # Slightly lower vorticity threshold for ETCs
                        mslp_eye_threshold=101500,  # 1000 hPa threshold for ETCs
                        lat_bounds=None,
                        gradient_threshold=0.3,
                        sample_data=sample_np
                    )
                except Exception as e:
                    print(f"Error in MCS post-processing for visualization: {e}")
                    post_processed_mcs_map = mcs_map

            
                # -------------------- PLOTTING SECTION --------------------
                # Get timestamp
                timestamp = pd.Timestamp(time_values[actual_index]).strftime('%Y-%m-%d %H:%M')
                
                # Plot layout (14 columns):
                # 0: IVT, 1: Vorticity, 2: MSLP, 3: Wind Speed
                # 4: ETC Low Pressure, 5: ETC Cold Core, 6: ETC Vorticity, 7: ETC Wind Speed
                # 8: Hurricane Strict Mask, 9: ETC Strict Mask
                # 10: Raw Segmentation, 11: AR Processed, 12: Hurricane Processed, 13: ETC Processed, 14: Ground Truth

                # Basic meteorological fields
                ax = axes[i, 0]
                im = ax.pcolormesh(lon, lat, ivt_plot_orig, cmap='viridis', shading='auto')
                ax.set_title(f'IVT - {timestamp}')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='IVT (kg m⁻¹ s⁻¹)')
                
                # Vorticity (fix: mask invalids, set colormap)
                ax = axes[i, 1]
                if vor_plot_orig is not None:
                    vor_data = np.copy(vor_plot_orig)
                    vor_data = np.ma.masked_invalid(vor_data)
                    im = ax.pcolormesh(lon, lat, vor_data, cmap='RdBu_r', shading='auto')
                    ax.set_title(f'Vorticity - {timestamp}')
                    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='Vorticity (1/s)')
                else:
                    ax.set_visible(False)
                
                # MSLP
                ax = axes[i, 2]
                if mslp_plot_orig is not None:
                    im = ax.pcolormesh(lon, lat, mslp_plot_orig, cmap='coolwarm_r', shading='auto')
                    ax.set_title(f'MSLP - {timestamp}')
                    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='MSLP (hPa)')
                else:
                    ax.set_visible(False)
                    
                # Wind-speed
                ax = axes[i, 3]
                if wind_plot_orig is not None:
                    wind_data = np.copy(wind_plot_orig)
                    wind_data = np.ma.masked_invalid(wind_data)
                    im = ax.pcolormesh(lon, lat, wind_data, cmap='viridis', shading='auto')
                    ax.set_title('Wind Speed')
                    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='Wind Speed (m/s)')
                else:
                    ax.set_visible(False)

                # ETC-specific masks
                ax = axes[i, 4]
                im = ax.pcolormesh(lon, lat, etc_low_pressure, cmap='Blues', shading='auto', vmin=0, vmax=1)
                ax.set_title('ETC Low Pressure Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 5]
                im = ax.pcolormesh(lon, lat, etc_vorticity, cmap='Blues', shading='auto', vmin=0, vmax=1)
                ax.set_title('ETC Vorticity Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 6]
                im = ax.pcolormesh(lon, lat, etc_wind_speed, cmap='Blues', shading='auto', vmin=0, vmax=1)
                ax.set_title('ETC Wind Speed Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                # Strict detection masks
                ax = axes[i, 7]
                im = ax.pcolormesh(lon, lat, hurricane_strict, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('Hurricane  Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 8]
                im = ax.pcolormesh(lon, lat, etc_strict, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('ETC  Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                # Segmentation results
                ax = axes[i, 9]
                im = ax.pcolormesh(lon, lat, seg_map_plot, cmap=four_class_cmap, shading='auto', vmin=0, vmax=4)
                ax.set_title('WNet Raw Segmentation')
                cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, 
                                label='Storm Class', ticks=[0, 1, 2, 3, 4])
                cbar.ax.set_xticklabels(['Background', 'AR', 'Hurricane', 'ETCs', 'MCS'])


                ax = axes[i, 10]
                im = ax.pcolormesh(lon, lat, post_processed_ar_map, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('AR Processed')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 11]
                im = ax.pcolormesh(lon, lat, post_processed_hurricane_map, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('Hurricane Processed')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 12]
                im = ax.pcolormesh(lon, lat, post_processed_etc_map, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('ETC Processed')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 13]
                im = ax.pcolormesh(lon, lat, gt_plot, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('Ground Truth')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 14]
                im = ax.pcolormesh(lon, lat, mcs_strict, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('MCS Mask')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                ax = axes[i, 15]
                im = ax.pcolormesh(lon, lat, post_processed_mcs_map, cmap='jet', shading='auto', vmin=0, vmax=1)
                ax.set_title('MCS Processed')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

                # Add coastlines to all plots
                for j in range(16):
                    if axes[i, j].axison:
                        axes[i, j].coastlines()
                        
            except Exception as e:
                print(f"Error processing sample {index} for visualization: {e}")
                # Fill with empty plots in case of error
                for j in range(14):
                    axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                                   transform=axes[i, j].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Enhanced filename generation with timestamp information
    first_timestamp = pd.Timestamp(time_values[indices[0]]).strftime('%Y%m%d_%H%M')
    output_type = "three_class_storm_detection"
    filename = nc_file.replace('.nc', f'_{output_type}_{first_timestamp}.png')
    
    if output_dir is None:
        output_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial28"
    
    filepath = os.path.join(output_dir, filename)
    print(f"Saving three-class visualization to: {filepath}")
    
    try:
        plt.savefig(filepath, format='png', bbox_inches='tight', dpi=300)
        print(f"Visualization saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close(fig)
        
    return filepath


def find_latest_checkpoint(checkpoint_dir='.'):
    """Find the latest checkpoint file in the given directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ema_checkpoint_epoch_*.pth'))
    
    if not checkpoint_files:
        # Try looking for emergency checkpoints or generic checkpoints
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*checkpoint*.pth'))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_files[0]


def _month_outputs_exist(output_dir, nc_file):
    """True iff ALL four per-class detection NetCDFs for this month already exist.

    A month is 'done' only when AR, Hurricane, ETCs and MCS files are all present.
    save_results_to_netcdf writes MCS last, so requiring all four guards against a
    job that was killed mid-save (partial month → reprocessed, not falsely skipped).
    """
    base = nc_file.replace('.nc', '')
    suffixes = ('_AR_detection.nc', '_Hurricane_detection.nc',
                '_ETCs_detection.nc', '_MCS_detection.nc')
    return all(os.path.isfile(os.path.join(output_dir, base + s)) for s in suffixes)


def test_model_batch(data_dir, checkpoint_path=None, start_year=1980, end_year=2024,
                     output_dir=None, skip_existing=True, start_month=1, end_month=12):
    """
    Main function to test the model on all data files from start_year to end_year and save results.
    
    Args:
        data_dir: Directory containing ERA5 data files
        checkpoint_path: Path to checkpoint file (if None, will find the latest one)
        start_year: Start year for evaluation (default: 1980)
        end_year: End year for evaluation (default: 2024)
        output_dir: Directory to save output files
    """
    # Setup GPU and device
    setup_gpu()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint files found in the current directory.")
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load the 90th percentile IVT threshold data
    print("Loading 90th percentile IVT threshold data...")
    # IVT_90th_percentile_path = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/ivt_only/IVT_90th_percentile_cdo.nc'
    vor850_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/output/vor850_75th_percentile_global.nc'
    # vor500_results_per_path = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/vor/nc_files/vor500_latlongrid/output/vor500_75th_percentile_global.nc'
    windspeed_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/wind_speed_850/output/windspeed850_75th_percentile_global.nc'
    # Tpr_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/tpr_percentiles/tpr_nonzero_85th_percentile_global.nc'
    # BT10_results_per_path = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new/output/bt_10th_percentile_global.nc'
    mslp_anomaly_glob = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/global_stats/mslp_monthly_climatology.nc'
    # IWV monthly climatology (12, lat, lon) for the MCS moisture-anomaly gate (Option 1).
    iwv_climatology_glob = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/global_stats/iwv_monthly_climatology.nc'


    # percentile_threshold   = load_percentile_dataset(IVT_90th_percentile_path)
    vor850_results_per    = load_percentile_dataset(vor850_results_per_path)
    # vor500_results_per    = load_percentile_dataset(vor500_results_per_path)
    windspeed_results_per = load_percentile_dataset(windspeed_results_per_path)
    # Tpr_results_per       = load_percentile_dataset(Tpr_results_per_path)
    # BT10_results_per       = load_percentile_dataset(BT10_results_per_path)
    mslp_anom_result_glob = load_percentile_dataset(mslp_anomaly_glob)
    # Falls back to absolute-IWV gating if the climatology file is absent.
    iwv_clim_result_glob = (load_percentile_dataset(iwv_climatology_glob)
                            if os.path.exists(iwv_climatology_glob) else None)
    if iwv_clim_result_glob is None:
        print(f"⚠ IWV climatology not found at {iwv_climatology_glob} — MCS uses absolute-IWV gate")
    
    # Load SST data directory # era5_sst_1980.nc
    # SST_dir = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/sst/celsius/'
    # SST_files = sorted(glob.glob(os.path.join(SST_dir, 'era5_sst_*_6hourly_celsius.nc')))
    Bright_temp = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new/' 
    
    
    # Load vorticity data for hurricane detection
    print("Loading vorticity data for hurricane detection...")
    vorticity_dir = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/'
    vorticity_files = sorted(glob.glob(os.path.join(vorticity_dir, 'ERA5_vor_6hourly_*.850mb_latlon.nc')))
    
    ## vorticity 500mb
    # vorticity_500_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/vor/nc_files/vor500_latlongrid/'
    # vorticity_500_files = sorted(glob.glob(os.path.join(vorticity_500_dir, 'ERA5_vor_6hourly_*.500mb_latlon.nc')))
    
    
    # Load wind speed data ONCE outside the file loop
    print("Loading wind speed data...")
    wind_speed_path = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc'
    
    if os.path.exists(wind_speed_path):
        wind_speed_full = xr.open_dataset(wind_speed_path).squeeze('level')
        wind_speed_full = wind_speed_full.assign_coords(
            longitude=(((wind_speed_full.longitude + 180) % 360) - 180)
        ).sortby('longitude')
        print(f"Wind speed data loaded successfully. Shape: {wind_speed_full['wspd'].shape}")
    else:
        print(f"Warning: Could not find wind speed file at {wind_speed_path}")
        print("Wind speed data will be disabled.")
        wind_speed_full = None
    
    # Load 500mb temperature data directory (t300-t500)
    # temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/Hurricane_warmcore/monthly_output_6hourly'
    temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/monthly_output/6_hourly'
    

    # Load windshear data directory # 6hourly_ERA5_windshear_198001.nc
    # Calculate magnitude: shear_mag = sqrt(u_shear^2 + v_shear^2) in windshear_dir
    windshear_dir = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_shear_mag_sqrtu_v/'
    
    ibtracs_file = (
        "/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
        "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc"
    )

    ibtracs_lookup = load_ibtracs_for_training(
    ibtracs_file,
    start_year        = 1980,
    end_year          = 2024,
    wind_threshold_kt = 0.0,     # No wind filter — match notebook's read_ibtracs (keeps all valid usa_wind)
    lat_min           = -50.0,
    lat_max           =  50.0,
    )
    # ibtracs_lookup : dict[ datetime ] -> list[ (lat, lon, wind_ms, sshs) ]

    
    # Initialize model with the same architecture as during training
    # Updated to 12 input channels to account for wind speed
    model = ImprovedWNet(in_channels=14, num_classes=5).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    print(f"Month range: {start_month:02d}-{end_month:02d}")

    # Get all ERA5 files in the specified year range
    if os.path.isdir(data_dir):
        all_nc_files = [f for f in os.listdir(data_dir) if f.startswith('era5dara_') and f.endswith('.nc')]
        
        # Filter by year range
        nc_files = []
        for f in all_nc_files:
            try:
                file_year = int(f[9:13])  # Extract year from filename
                file_month = int(f[13:15])  # e.g. "04"
                if start_year <= file_year <= end_year and start_month <= file_month <= end_month:
                    nc_files.append(f)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse year from filename {f}")
                continue



        # Sort the files
        nc_files = sorted(nc_files)

        if not nc_files:
            print(f"No files found for year range {start_year}-{end_year}")
            return

        # ── Resume: drop months whose outputs already exist ───────────────────────────
        # Decided up-front on the GLOBAL list (before sharding / before any GPU work),
        # so no rank ever processes a month then skips it at save time. All ranks see
        # the same output dir and compute the SAME remaining list deterministically,
        # so the round-robin shard below stays disjoint and balanced.
        if skip_existing and output_dir is not None:
            before = len(nc_files)
            nc_files = [f for f in nc_files if not _month_outputs_exist(output_dir, f)]
            if _rank == 0:
                print(f"[resume] {before - len(nc_files)}/{before} months already complete "
                      f"in {output_dir} — skipping; {len(nc_files)} remaining to process")

        if not nc_files:
            if _rank == 0:
                print("[resume] All requested months already complete — nothing to do.")
            return

        # ── File-level sharding for multi-GPU inference (round-robin, matches training) ──
        # Shards the REMAINING (not-yet-done) files, so work stays balanced across GPUs
        # even on a partially-completed run.
        if _world > 1:
            all_count = len(nc_files)
            nc_files = get_rank_files(nc_files, _rank, _world, shuffle=False)
            print(f"[rank {_rank}/{_world}] handling {len(nc_files)}/{all_count} remaining files: "
                  f"{[f for f in nc_files]}")
    else:
        raise ValueError("data_dir must be a directory containing ERA5 files")
    
    print(f"Found {len(nc_files)} files to process for years {start_year}-{end_year}")
    
    # Process all files
    processed_files = []
    failed_files = []
    
    for file_idx, nc_file in enumerate(nc_files):
        print(f"\n[{file_idx+1}/{len(nc_files)}] Processing: {nc_file}")
        try:
            # Load the dataset
            file_path = os.path.join(data_dir, nc_file)
            ERA5_6hrly_ARs = xr.open_dataset(file_path)
            ERA5_6hrly_ARs = calculate_ivt(ERA5_6hrly_ARs)
            # Convert longitude from 0-360 to -180 to 180 and sort
            ERA5_6hrly_ARs = ERA5_6hrly_ARs.assign_coords(
                longitude=(((ERA5_6hrly_ARs.longitude + 180) % 360) - 180)
            ).sortby('longitude')
            
            # Prepare data
            ivt = ERA5_6hrly_ARs['IVT']
            mslp = ERA5_6hrly_ARs['MSLP']
            # TPR = ERA5_6hrly_ARs['TPR']
            IWV = ERA5_6hrly_ARs['IWV']
            
            # Find and load the matching vorticity file for the current year
            era5_dates = ERA5_6hrly_ARs['time'].values
            
            # Load matching 500mb temperature file
            date_part = nc_file.split('_')[1].split('.')[0]  # Gets "201604" from "era5dara_201604.nc"
            year_part = date_part[:4]  # '2016'
            
            # ### SST data loading/reading
            # sst_file = os.path.join(SST_dir, f'era5_sst_{year_part}_6hourly_celsius.nc')
            # # print("sst_file",sst_file)
            # if sst_file is not None:
            #     SST_ds = xr.open_dataset(sst_file).rename({'valid_time':'time'})
            #     SST_ds = SST_ds.assign_coords(
            #         longitude=(((SST_ds.longitude + 180) % 360) - 180)).sortby('longitude')
                
            #     SST_ds = SST_ds['sst']
            #     SST_ds = SST_ds.sel(time=ERA5_6hrly_ARs['time'])
            # else:
            #     print("No matching SST file found for this year.")
            #     SST_ds = None
            
            ### BT data loading/reading
            BT_file = os.path.join(Bright_temp, f'era5_bt_6hourly_{year_part}.nc')
            # print("sst_file",sst_file)
            if BT_file is not None:
                BT_ds = xr.open_dataset(BT_file).rename({'valid_time':'time'})
                BT_ds = BT_ds.assign_coords(
                    longitude=(((BT_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                
                BT_ds = BT_ds['bt']
                # BT_ds = BT_ds.sel(time=ERA5_6hrly_ARs['time'])
                common_times = np.intersect1d(BT_ds.time.values, ERA5_6hrly_ARs.time.values)
                BT_ds = BT_ds.sel(time=common_times)

            else:
                print("No matching BT file found for this year.")
                BT_ds = None
            
            # temp_500mb_file = os.path.join(temp_500mb_dir, f'ERA5_warmcore_{date_part}_6hourly.nc')
            windshear_file = os.path.join(windshear_dir, f'6hourly_ERA5_windshear_mag_{date_part}.nc') 
            
            # print("temp_500mb_file",temp_500mb_file)
            
            # if os.path.exists(temp_500mb_file):
            #     temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level='300')
            #     temp_500mb_ds = temp_500mb_ds.assign_coords(longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
            #     ).sortby('longitude')
                
            #     # Get the temperature variable (assuming it's 't' or similar)
            #     temp_500mb_interp = temp_500mb_ds["T"]
            #     common_times = np.intersect1d(temp_500mb_interp.time.values, ERA5_6hrly_ARs.time.values)
            #     temp_500mb = temp_500mb_interp.sel(time=common_times)

            #     # temp_500mb = temp_500mb_interp.sel(time=ERA5_6hrly_ARs['time'])
                
            #     print(f"Successfully loaded T300-500mb temperature data: {temp_500mb.shape}")
            # else:
            #     print(f"Temperature T300-500mb file not found: {temp_500mb_file}. Using default values.")
            #     temp_500mb = None

            temp_500mb_file = os.path.join(temp_500mb_dir, f'ERA5_t_500mb_{date_part}.nc')
            
            if os.path.exists(temp_500mb_file):
                temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level='500')
                temp_500mb_ds = temp_500mb_ds.assign_coords(
                    longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                
                # Get the temperature variable (assuming it's 'T' or similar)
                temp_500mb_interp = temp_500mb_ds["T"]
                # temp_500mb = temp_500mb_interp.sel(time=ERA5_6hrly_ARs['time'])
                common_times = np.intersect1d(temp_500mb_interp.time.values, ERA5_6hrly_ARs.time.values)
                temp_500mb = temp_500mb_interp.sel(time=common_times)
                
                print(f"Successfully loaded 500mb temperature data: {temp_500mb.shape}")
            else:
                print(f"Temperature file not found: {temp_500mb_file}. Using default values.")
                temp_500mb = None


            ## windshear magnitude dataset
            if os.path.exists(windshear_file):
                windhshear_ds = xr.open_dataset(windshear_file).sel(level='200')
                windhshear_ds = windhshear_ds.assign_coords(
                    longitude=(((windhshear_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                
                # Get the temperature variable (assuming it's 't' or similar)
                windshear_ds_interp = windhshear_ds["windshear"]
                common_times = np.intersect1d(windshear_ds_interp.time.values, ERA5_6hrly_ARs.time.values)
                windhshear_ds = windshear_ds_interp.sel(time=common_times)

                print(f"Successfully loaded windhshear_ds data: {windhshear_ds.shape}")
            else:
                print(f"windhshear file not found: {windshear_file}. Using default values.")
                windhshear_ds = None

            # Find and load vorticity data 850mb
            vorticity_file = find_matching_vorticity(era5_dates[0], vorticity_files)
            
            if vorticity_file is not None:
                vorticity_ds = xr.open_dataset(vorticity_file).squeeze('plev')
                vorticity_ds = vorticity_ds.assign_coords(
                    longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                
                vorticity_interp = vorticity_ds['var138']
                common_times = np.intersect1d(vorticity_interp.time.values, ERA5_6hrly_ARs.time.values)
                vorticity = vorticity_interp.sel(time=common_times)

            else:
                print("No matching vorticity file found for this year.")
                vorticity = None
                
            ## vorticity 500mb
            # vorticity_500_file = find_matching_vorticity(era5_dates[0], vorticity_500_files)
            # if vorticity_500_file is not None:
            #     vorticity_500_ds = xr.open_dataset(vorticity_500_file).squeeze('plev')
            #     vorticity_500_ds = vorticity_500_ds.assign_coords(
            #         longitude=(((vorticity_500_ds.longitude + 180) % 360) - 180)
            #     ).sortby('longitude')
                
            #     vorticity_500_interp = vorticity_500_ds['var138']
            #     common_times = np.intersect1d(vorticity_500_interp.time.values, ERA5_6hrly_ARs.time.values)
            #     vorticity_500_mb = vorticity_500_interp.sel(time=common_times)

            # else:
            #     print("No matching vorticity 500mb file found for this year. Skipping hurricane detection for this file.")
            #     vorticity_500_mb = None
            
            # Process wind speed data
            wind_speed_selected = None
            if wind_speed_full is not None:
                wind_speed_interp = wind_speed_full['wspd']
                wind_speed_selected = wind_speed_interp.sel(time=era5_dates)
                # wind_speed_selected = wind_speed_selected.where(wind_speed_selected >= 12, 12)
            
            # Interpolate percentile threshold to match dataset grid if needed
            interp_percentile = 400
            # if isinstance(percentile_threshold, (xr.DataArray, xr.Dataset)):
            #     interp_percentile = percentile_threshold
            # else:
            #     # If percentile_threshold is a scalar value
            #     print(f"Using fixed threshold value: {percentile_threshold}")
            #     interp_percentile = percentile_threshold
            
            print("Preprocessing data...")
            
            # Preprocess data using the same function as training
            
            preprocessed_data = preprocess_data_improved(
                ivt=ivt,
                mslp=mslp,
                vorticity=vorticity,
                wind_speed=wind_speed_selected,
                ivt_percentile=interp_percentile,
                temp_500mb=temp_500mb,
                BT_ds=BT_ds,
                # TPR=TPR,
                IWV=IWV,
                windhshear_ds=windhshear_ds
            )

            vorticity_threshold = 1.45e-5  # Match training threshold
            
            global_percentiles = {
                'vor850': vor850_results_per,       
                'windspeed': windspeed_results_per, 
                 
                ##MCS
                'BT_boundary': 241.0,
                'BT_core': 225.0,
                
                # UPDATED HURRICANE THRESHOLDS - Literature-based
                "Temp500": 270.0,  # temp500_results_per, K
                "vor850_per_hurr": 1.45e-5,            # CHANGED: Was 4.6e-05, now 1.5e-05 (Walsh et al. 2007)
                "windspeed_per_hurr": 10.0,            # CHANGED: Was 19.197, now 17.0 m/s (NHC tropical storm)
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

            # ── IBTrACS slice for this month  ◄─── ADD HERE, after time_values is built
            # time_values is now fully resolved from preprocessed_data above
            time_values = ERA5_6hrly_ARs.time.values   # ← ADD THIS

            era5_6h_keys = [
                pd.Timestamp(t).to_pydatetime().replace(minute=0, second=0, microsecond=0)
                for t in time_values
            ]

            ibtracs_this_month = {
                dt: ibtracs_lookup.get(dt, [])
                for dt in era5_6h_keys
            }

            # Create dataset with the same parameters as training
            print("Creating dataset...")
            dataset = ARDatasetImproved(
                preprocessed_data, 
                ivt_percentile=interp_percentile, 
                vorticity_threshold=vorticity_threshold, 
                target_size=(360, 720),  # Match training target_size
                global_percentiles=global_percentiles,
                global_stats=global_stats,          # ← ADD THIS
                precompute=False,
                mslp_climatology=mslp_anom_result_glob,
                iwv_climatology=iwv_clim_result_glob,     # IWV monthly clim → anomaly gate
                mcs_moisture_mode="anomaly",              # "anomaly" | "anomaly_soft" | "absolute"
                ibtracs_lookup=ibtracs_this_month,   # ← ADD THIS
            )
            
            print(f"Dataset size: {len(dataset)} samples")
            
            # Evaluate model and save results
            ar_filepath, hurricane_filepath, etc_filepath, mcs_filepath = evaluate_and_save_results(
                model, dataset, ERA5_6hrly_ARs, vorticity, wind_speed_selected,temp_500mb, nc_file,
                percentile_threshold=interp_percentile, output_dir=output_dir
            )
            
        
            # Visualization (optional - can be commented out for faster processing)
            # visualize_segmentation_improved(
            #     model, dataset, ERA5_6hrly_ARs, vorticity, wind_speed_selected, nc_file, 
            #     output_dir=output_dir, percentile_threshold=interp_percentile, mslp_data=mslp,
            #     num_samples=3
            # )
            
            processed_files.append({
                'input_file': nc_file,
                'ar_output': ar_filepath,
                'hurricane_output': hurricane_filepath,
                'ETC_output': etc_filepath,
                'MCS_output': mcs_filepath,  # NEW
            })
            
            # Clean up
            del ERA5_6hrly_ARs, ivt, mslp, vorticity, preprocessed_data, dataset,  interp_percentile, wind_speed_selected, BT_ds, IWV, windhshear_ds

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(nc_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Failed files: {len(failed_files)}")
    
    if processed_files:
        print(f"\nSuccessfully processed files:")
        for item in processed_files:
            print(f"  {item['input_file']}")
            print(f"    -> AR: {os.path.basename(item['ar_output'])}")
            print(f"    -> Hurricane: {os.path.basename(item['hurricane_output'])}")
            print(f"    -> ETC: {os.path.basename(item['ETC_output'])}")
            print(f"    -> MCS: {os.path.basename(item['MCS_output'])}")  # NEW
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  {f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Batch evaluation completed!")


if __name__ == "__main__":
    # CLI: any unspecified arg falls back to the historical defaults below.
    _parser = argparse.ArgumentParser(
        description="Forward-only inference: load checkpoint, run over a year range, "
                    "write per-class detection NetCDFs to --output-dir.",
    )
    _parser.add_argument(
        "--data-dir",
        default="/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/",
        help="ERA5 monthly NetCDF dir (default: project's data/)",
    )
    _parser.add_argument(
        "--checkpoint",
        default=("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Trail9/final_checkpoint_2006_2024.pth"),
        help="Path to .pth checkpoint (default: legacy MCS_enhanced_trial60 epoch 40)",
    )
    _parser.add_argument(
        "--output-dir",
        default=("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code/ARs_Hurricanes_ETCs_MCS/Outputs"),
        help="Where to write per-class detection NetCDFs",
    )
    _parser.add_argument("--start-year", type=int, default=2004)
    _parser.add_argument("--end-year",   type=int, default=2017)
    _parser.add_argument("--start-month", type=int, default=1, help="First month to process (1-12)")
    _parser.add_argument("--end-month",   type=int, default=12, help="Last month to process (1-12)")
    _parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess months even if their output NetCDFs already exist "
             "(default: skip already-complete months for resume).",
    )
    _args = _parser.parse_args()

    print(f"[test.py] checkpoint = {_args.checkpoint}")
    print(f"[test.py] data-dir   = {_args.data_dir}")
    print(f"[test.py] output-dir = {_args.output_dir}")
    print(f"[test.py] years      = {_args.start_year}-{_args.end_year}")

    test_model_batch(
        data_dir=_args.data_dir,
        checkpoint_path=_args.checkpoint,
        start_year=_args.start_year,
        end_year=_args.end_year,
        output_dir=_args.output_dir,
        skip_existing=not _args.force_reprocess,
        start_month=_args.start_month,
        end_month=_args.end_month,
    )