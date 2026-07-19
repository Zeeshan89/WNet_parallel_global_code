from skimage.transform import resize
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import os
import sys
from scipy import ndimage
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

import torch.optim as optim
import matplotlib.pyplot as plt

# For plotting with geographical coordinates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors

from post_processing import enhanced_post_process_ar_segmentation

from matplotlib.colors import ListedColormap


def visualize_four_class_segmentation(model, dataset, era5_dataset, vorticity, wind_speed_selected, nc_file, vor_dataset=None, 
                                     mslp_data=None, percentile_threshold=None, num_samples=3, indices=None):
    """
    Enhanced visualization for four-class storm detection: ARs, Hurricanes, ETCs, and MCS.
    Displays all relevant channels and post-processed results for each storm type.
    """
    
    model.eval()
    device = next(model.parameters()).device

    # Decide which indices to use
    if indices is None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        num_samples = len(indices)   # override num_samples if indices provided

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


    # Four-class colormap: white=background, blue=AR, red=Hurricane, green=ETC, orange=MCS
    four_class_cmap = ListedColormap(['white', 'blue', 'red', 'green', 'orange'])

    # Create larger figure for 15 subplots (5 rows x 3 columns per sample)
    fig, axes = plt.subplots(num_samples, 19, figsize=(100, 5 * num_samples), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
        
        
    with torch.no_grad():
        for i, index in enumerate(indices):
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
            ivt_binary = sample_np[1]
            # vorticity_sample = sample_np[4]
            # vorticity_grad_sample = sample_np[5]
            # wind_speed_sample = sample_np[6]
            # hurricane_low_pressure = sample_np[7]
            # hurricane_high_vorticity = sample_np[8]
            # hurricane_high_wind_speed = sample_np[9]
            hurricane_strict = sample_np[5]
            # wind_speed_binary = sample_np[11]
                
            # ETC channels (12-15) - CORRECTED
            etc_low_pressure = sample_np[7]
            etc_vorticity = sample_np[8]
            etc_wind_speed = sample_np[9]
            etc_strict = sample_np[10]

            # MCS channels (16-20) - CORRECTED
            # bt_sample = sample_np[16]
            # tpr_sample = sample_np[17]
            mcs_deep_convection = sample_np[11]
            mcs_heavy_precip = sample_np[12]
            mcs_core = sample_np[13]


            
            # Get original data for post-processing
            # Try to get IVT from era5_dataset (could be 'IVT', 'ivt', or 'ivt_enhanced')
            ivt_data_orig = None
            if hasattr(era5_dataset, 'IVT'):
                ivt_data_orig = era5_dataset.IVT.values[actual_index]
            elif hasattr(era5_dataset, 'ivt'):
                ivt_data_orig = era5_dataset.ivt.values[actual_index]
            elif hasattr(era5_dataset, 'ivt_enhanced'):
                ivt_data_orig = era5_dataset.ivt_enhanced.values[actual_index]
            else:
                print(f"Warning: No IVT data found in era5_dataset. Available vars: {list(era5_dataset.data_vars)}")
                ivt_data_orig = np.zeros(target_size)
            
            # Use vorticity from function parameter
            vor_data_orig = None
            if vorticity is not None:
                try:
                    vor_data_orig = vorticity.values[actual_index]
                except (IndexError, AttributeError, TypeError):
                    vor_data_orig = None
            
            # Use mslp_data from function parameter
            mslp_data_orig = None
            if mslp_data is not None:
                try:
                    if hasattr(mslp_data, 'values'):
                        mslp_data_orig = mslp_data.values[actual_index]
                    else:
                        mslp_data_orig = mslp_data[actual_index]
                except (IndexError, TypeError):
                    mslp_data_orig = None
                    
            # Use wind_speed_selected from function parameter
            wind_data_orig = None
            if wind_speed_selected is not None:
                try:
                    if hasattr(wind_speed_selected, 'values'):
                        wind_data_orig = wind_speed_selected.values[actual_index]
                    else:
                        wind_data_orig = wind_speed_selected[actual_index]
                except (IndexError, KeyError, TypeError):
                    wind_data_orig = None

            # Resize data to target size if needed
            def resize_data(data, target_size):
                if data is not None and data.shape != target_size:
                    return resize(data, target_size, order=1, mode='reflect', anti_aliasing=False)
                return data

            ivt_plot_orig = resize_data(ivt_data_orig, target_size)
            vor_plot_orig = resize_data(vor_data_orig, target_size)
            mslp_plot_orig = resize_data(mslp_data_orig, target_size)
            wind_plot_orig = resize_data(wind_data_orig, target_size)

            # Ground truth
            gt_plot = np.zeros(target_size, dtype=np.float32)
            if 'ARS' in era5_dataset:
                gt = era5_dataset['ARS'].values[actual_index]
                if gt.shape != target_size:
                    gt_plot = resize(gt, target_size, order=0, mode='reflect', anti_aliasing=False)
                else:
                    gt_plot = gt

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
            
            # Post-process each storm type
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

            post_processed_hurricane_map = enhanced_post_process_ar_segmentation(
                hurricane_map,
                storm_class=2,
                ivt_data=None,
                vor_data=vor_plot_orig,
                mslp_data=mslp_plot_orig,
                percentile_threshold=None,
                min_size=1,
                km_per_pixel=27.75,
                vor_threshold=5.0e-05,
                mslp_eye_threshold=102000,
                lat_bounds=None,
                gradient_threshold=0.5,
                sample_data=sample_np
            )

            # Post-process ETC detection
            post_processed_etc_map = enhanced_post_process_ar_segmentation(
                etc_map,
                storm_class=3,  # New ETC class
                ivt_data=None,
                vor_data=vor_plot_orig,
                mslp_data=mslp_plot_orig,
                percentile_threshold=None,
                min_size=20,  # Moderate minimum size for ETCs
                km_per_pixel=27.75,
                vor_threshold=3.0e-05,  # Slightly lower vorticity threshold for ETCs
                mslp_eye_threshold=101500,  # 1000 hPa threshold for ETCs
                lat_bounds=None,
                gradient_threshold=0.5,
                sample_data=sample_np
            )
            
            # Post-process MCS detection
            post_processed_mcs_map = enhanced_post_process_ar_segmentation(
                mcs_map,
                storm_class=4,
                ivt_data=None,
                vor_data=vor_plot_orig,
                mslp_data=mslp_plot_orig,
                percentile_threshold=None,
                min_size=10,  # Much smaller minimum size
                km_per_pixel=27.75,
                vor_threshold=1.0e-05,
                mslp_eye_threshold=101200,
                lat_bounds=None,
                gradient_threshold=0.3,
                sample_data=sample_np)
            
            # -------------------- PLOTTING SECTION --------------------
            # Get timestamp
            timestamp = pd.Timestamp(time_values[actual_index]).strftime('%Y-%m-%d %H:%M')
            
            
            # Plot layout (20 columns):
            # 0: IVT, 1: Vorticity, 2: MSLP, 3: Wind Speed
            # 4: ETC Low Pressure, 5: ETC Cold Core, 6: ETC Vorticity, 7: ETC Wind Speed
            # 8: Hurricane Strict Mask, 9: ETC Strict Mask
            # 10: MCS BT, 11: MCS TPR, 12: MCS Deep Convection, 13: MCS Heavy Precip, 14: MCS Core
            # 15: Raw Segmentation, 16: AR Processed, 17: Hurricane Processed, 18: ETC Processed, 19: MCS Processed


            # Basic meteorological fields
            ax = axes[i, 0]
            im = ax.pcolormesh(lon, lat, ivt_plot_orig, cmap='viridis', shading='auto')
            ax.set_title(f'IVT - {timestamp}')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='IVT (kg m⁻¹ s⁻¹)')

            ax = axes[i, 1]
            if vor_plot_orig is not None:
                vor_data = np.copy(vor_plot_orig)
                vor_data = np.ma.masked_invalid(vor_data)
                im = ax.pcolormesh(lon, lat, vor_data, cmap='RdBu_r', shading='auto')
                ax.set_title(f'Vorticity - {timestamp}')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='Vorticity (1/s)')
            else:
                ax.set_visible(False)

            ax = axes[i, 2]
            if mslp_plot_orig is not None:
                im = ax.pcolormesh(lon, lat, mslp_plot_orig, cmap='coolwarm_r', shading='auto')
                ax.set_title(f'MSLP - {timestamp}')
                fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='MSLP (hPa)')
            else:
                ax.set_visible(False)

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
            ax.set_title('Hurricane Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 8]
            im = ax.pcolormesh(lon, lat, etc_strict, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('ETC Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            # Segmentation results
            ax = axes[i, 9]
            im = ax.pcolormesh(lon, lat, seg_map_plot, cmap=four_class_cmap, shading='auto', vmin=0, vmax=4)
            ax.set_title('WNet Raw Segmentation')
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, 
                               label='Storm Class', ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_xticklabels(['Background', 'AR', 'Hurricane', 'ETCs', 'MCS'])

            ax = axes[i, 10]
            im = ax.pcolormesh(lon, lat, ivt_binary, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('AR Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 11]
            im = ax.pcolormesh(lon, lat, post_processed_ar_map, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('AR Processed')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 12]
            im = ax.pcolormesh(lon, lat, post_processed_hurricane_map, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('Hurricane Processed')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 13]
            im = ax.pcolormesh(lon, lat, post_processed_etc_map, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('ETC Processed')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 14]
            im = ax.pcolormesh(lon, lat, gt_plot, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('Ground Truth')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            # MCS-specific plots
            # ax = axes[i, 15]
            # im = ax.pcolormesh(lon, lat, bt_sample, cmap='coolwarm_r', shading='auto')
            # ax.set_title('MCS Brightness Temp (Norm)')
            # fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            # ax = axes[i, 16]
            # im = ax.pcolormesh(lon, lat, tpr_sample, cmap='Blues', shading='auto')
            # ax.set_title('MCS Total Precip Rate (Norm)')
            # fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
            ax = axes[i, 15]
            im = ax.pcolormesh(lon, lat, mcs_deep_convection, cmap='Blues', shading='auto', vmin=0, vmax=1)
            ax.set_title('MCS Boundry 241 Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 16]
            im = ax.pcolormesh(lon, lat, mcs_heavy_precip, cmap='Blues', shading='auto', vmin=0, vmax=1)
            ax.set_title('MCS (225K in 241K) Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

            ax = axes[i, 17]
            im = ax.pcolormesh(lon, lat, mcs_core, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('MCS Core Mask')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            
            ax = axes[i, 18]
            im = ax.pcolormesh(lon, lat, post_processed_mcs_map, cmap='jet', shading='auto', vmin=0, vmax=1)
            ax.set_title('MCS Processed')
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            
            # Add coastlines to all plots
            for j in range(19):
                if axes[i, j].axison:
                    axes[i, j].coastlines()

    plt.tight_layout()
    
    # Enhanced filename generation
    first_timestamp = pd.Timestamp(time_values[indices[0]]).strftime('%Y%m%d_%H%M')
    output_type = "three_class_storm_detection"
    filename = nc_file.replace('.nc', f'_{output_type}_{first_timestamp}.png')
    from utils import regcm_checkpoint_dir
    output_directory = regcm_checkpoint_dir()
    os.makedirs(output_directory, exist_ok=True)
    filepath = os.path.join(output_directory, filename)
    print(f"Saving three-class visualization to: {filepath}")
    plt.savefig(filepath, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_training_history(losses, save_path=None):
    # Save beside this run's own checkpoints. Uses the shared definition rather
    # than re-deriving the path: a private copy here kept writing the precip-MCS
    # run's plot into the BT-era Trail_RegCM_<MODEL>/, mislabelling the baseline.
    if save_path is None:
        from utils import regcm_checkpoint_dir
        save_path = os.path.join(regcm_checkpoint_dir(), "training_history.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(losses, list):
        # Legacy: plain list of total losses
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(losses, color="steelblue", linewidth=0.8)
        ax.set_xlabel("Epoch (all files concatenated)")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    # Structured history dict
    keys    = ["total", "ar", "hurricane", "etc", "mcs", "overlap"]
    colors  = ["black", "steelblue", "firebrick", "darkorange", "forestgreen", "purple"]
    labels  = ["Total", "AR", "Hurricane", "ETC", "MCS", "Overlap"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    boundaries = losses.get("file_boundaries", [])
    file_names = losses.get("file_names", [])

    for ax, key, color, label in zip(axes, keys, colors, labels):
        values = losses.get(key, [])
        if not values:
            ax.set_visible(False)
            continue

        x = np.arange(len(values))
        ax.plot(x, values, color=color, linewidth=0.8, alpha=0.9)

        # Shade alternating file blocks and add file labels
        for i, b_start in enumerate(boundaries):
            b_end = boundaries[i+1] if i+1 < len(boundaries) else len(values)
            if i % 2 == 0:
                ax.axvspan(b_start, b_end, alpha=0.06, color="gray")
            # Label each file block at the top
            mid = (b_start + b_end) / 2
            fname = file_names[i] if i < len(file_names) else ""
            date_str = fname.split("_")[1].split(".")[0] if "_" in fname else str(i)
            ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    date_str, fontsize=5, ha="center", va="bottom",
                    rotation=90, color="gray")

        # Smoothed trend line
        if len(values) > 10:
            window = max(5, len(values) // 50)
            smoothed = np.convolve(values, np.ones(window)/window, mode="valid")
            ax.plot(np.arange(window-1, len(values)), smoothed,
                    color=color, linewidth=2.0, alpha=0.6, linestyle="--",
                    label=f"Smoothed (w={window})")

        ax.set_xlabel("Epoch (all files concatenated)")
        ax.set_ylabel("Loss")
        ax.set_title(f"{label} Loss")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Loss History — All Files", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved → {save_path}")