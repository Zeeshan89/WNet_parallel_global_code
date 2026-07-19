from builtins import float, int
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from scipy.ndimage import zoom
from scipy.stats import norm
from scipy import ndimage
from skimage.measure import regionprops, label
import math
import gc
import os
import psutil
import time
import warnings
from typing import Optional, Tuple, Dict, Any
from scipy.ndimage import gaussian_filter, label, median_filter
from scipy.spatial.distance import cdist
from typing import Tuple  # Add to existing typing imports
from scipy.ndimage import minimum_filter, maximum_filter, binary_dilation
import time as _time
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




def _fmt_thr(v):
    """Format a threshold that may be a scalar OR a 2-D per-cell field, for logging."""
    try:
        if hasattr(v, 'ndim') and getattr(v, 'ndim', 0) >= 2:
            return (f"PER-CELL FIELD {tuple(v.shape)} "
                    f"(range {float(np.nanmin(v)):.2f}–{float(np.nanmax(v)):.2f} K, "
                    f"median {float(np.nanmedian(v)):.2f} K)")
        return f"GLOBAL SCALAR ({float(v):.2f} K)"
    except Exception:
        return str(v)


class ARDatasetImproved(Dataset):
    """
    Enhanced AR Dataset with GPU optimization, distributed training support, temperature-based hurricane detection,
    wind shear integration, and improved ETC detection with spatial coherence and intensity clustering
    """
    # Precip-MCS tensors live in their OWN dir, separate from the BT-branch cache:
    # changing _create_mcs_masks invalidates every cached tensor, and a separate
    # dir keeps the BT baseline available for A/B and makes rollback a path change.
    # MUST stay in sync with precompute_tensors.TENSOR_DIR.
    TENSOR_CACHE_DIR = f"/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_RegCM_PR_{os.environ.get('REGCM_MODEL', 'ACC')}"

    def __init__(self, 
                data: xr.Dataset, 
                ivt_percentile: Optional[xr.DataArray] = None, 
                vorticity_threshold: Optional[float] = None, 
                ars: Optional[xr.DataArray] = None, 
                target_size: Tuple[int, int] = (720, 1400), 
                chunk_size: int = 4,
                device: str = 'cuda',
                dtype: torch.dtype = torch.float32,
                pin_memory: bool = True,
                preload_data: bool = False,
                global_percentiles: Optional[Dict[str, xr.DataArray]] = None,
                global_stats: Optional[Dict[str, Dict[str, float]]] = None,  
                precompute: bool = True,
                mslp_climatology: Optional[np.ndarray] = None,
                iwv_climatology: Optional[np.ndarray] = None,
                mcs_moisture_mode: str = "anomaly",
                mcs_source: str = "precip",
                iwv_anom_threshold: float = 0.0,
                iwv_anom_soft_band: float = 5.0,
                ibtracs_lookup: Optional[dict] = None,
                year=None,
                month=None
                ):
        """
        Initialize the enhanced AR dataset with improved ETC detection using multi-level vorticity
        
        Args:
            data: xarray Dataset containing weather data including vorticity_500_mb and vorticity_500_grad
            ... (other args same as before)
        """
        super().__init__()
        
        _init_t = time.time()

        # GPU optimization settings
        self.device = device
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.preload_data = preload_data
        self.year  = year
        self.month = month

        # Store global percentiles for threshold calculation
        self.global_percentiles = global_percentiles or {}
        self.global_stats = global_stats or {}
        self.ibtracs_lookup = ibtracs_lookup  # dict[datetime] -> [(lat,lon,wind,sshs)]
        print(f"Initializing ARDatasetImproved with enhanced multi-level ETC detection for {device}...")
        
        self.total_samples = len(data.time)
        self.time = data.time.values
        self.target_size = target_size
        self.chunk_size = chunk_size
        self.ivt_percentile = ivt_percentile
        self.vorticity_threshold = vorticity_threshold

        # Memory optimization: convert to numpy arrays with optimal data types
        print("Converting data to optimized numpy arrays...")
        
        # Helper function to safely extract variables from Dataset or dict-like objects
        def extract_var(data_obj, var_names, default=None):
            """Extract variable from Dataset, trying multiple names"""
            if isinstance(var_names, str):
                var_names = [var_names]
            
            for name in var_names:
                try:
                    if isinstance(data_obj, xr.Dataset):
                        if name in data_obj.data_vars:
                            return data_obj[name]
                    elif hasattr(data_obj, name):
                        return getattr(data_obj, name)
                except Exception:
                    continue
            return default
        
        # Core data arrays - only what's needed
        self.ivt = self._convert_to_numpy(extract_var(data, 'ivt'), dtype=np.float32)
        self.mslp = self._convert_to_numpy(extract_var(data, 'mslp'), dtype=np.float32)
        self.ivt_grad = self._convert_to_numpy(extract_var(data, 'ivt_grad'), dtype=np.float32)
        self.vorticity_grad = self._convert_to_numpy(extract_var(data, 'vorticity_grad'), dtype=np.float32)
        self.vorticity = self._convert_to_numpy(extract_var(data, 'vorticity'), dtype=np.float32)  # 850mb
        self.wind_speed = self._convert_to_numpy(extract_var(data, 'wind_speed'), dtype=np.float32)
        
        # Temperature, wind shear, and SST data - with graceful fallback for missing auxiliary data
        self.temp_500mb = self._convert_to_numpy(extract_var(data, ['temp_500mb', 't', 'T', 'temperature']), dtype=np.float32)
        if self.temp_500mb is None:
            print("⚠ Missing temperature (temp_500mb) data - using zeros")
            self.temp_500mb = np.zeros_like(self.ivt)

        # MCS-specific data - handle missing auxiliary data with fallback
        self.bt_ds = self._convert_to_numpy(extract_var(data, ['BT_ds', 'BT']), dtype=np.float32)
        if self.bt_ds is None:
            print("⚠ Missing BT (Brightness Temperature) data - using zeros")
            self.bt_ds = np.zeros_like(self.ivt)
        
        # Precipitation (mm/h) — the MCS source. `pr` total, `prc` convective.
        # A zero-fill here would make every MCS mask empty, so the loaders gate
        # on precip being present rather than letting this fall back silently.
        self.pr = self._convert_to_numpy(extract_var(data, ['PR', 'pr']), dtype=np.float32)
        if self.pr is None:
            print("⚠ Missing pr (total precipitation) data - using zeros → MCS will be EMPTY")
            self.pr = np.zeros_like(self.ivt)

        self.prc = self._convert_to_numpy(extract_var(data, ['PRC', 'prc']), dtype=np.float32)
        if self.prc is None:
            print("⚠ Missing prc (convective precipitation) data - using zeros → MCS will be EMPTY")
            self.prc = np.zeros_like(self.ivt)


        self.iwv = self._convert_to_numpy(extract_var(data, ['IWV', 'iwv']), dtype=np.float32)
        if self.iwv is None:
            print("⚠ Missing IWV (Integrated Water Vapor) data - using zeros")
            self.iwv = np.zeros_like(self.ivt)
        
        self.windhshear = self._convert_to_numpy(extract_var(data, ['windhshear', 'wind_shear', 'shear_mag']), dtype=np.float32)
        if self.windhshear is None:
            print("⚠ Missing wind shear data - using zeros")
            self.windhshear = np.zeros_like(self.ivt)

        # Convert global percentile datasets to numpy arrays
        if 'vor850' in self.global_percentiles:
            self.vor850_threshold = self._convert_to_numpy(self.global_percentiles['vor850'], dtype=np.float32)
        else:
            self.vor850_threshold = None

        # if 'vor500' in self.global_percentiles:
        #     self.vor500_threshold = self._convert_to_numpy(self.global_percentiles['vor500'], dtype=np.float32)
        # else:
        #     self.vor500_threshold = None

        if 'windspeed' in self.global_percentiles:
            self.windspeed_threshold = self._convert_to_numpy(self.global_percentiles['windspeed'], dtype=np.float32)
        else:
            self.windspeed_threshold = None

        
        # MCS thresholds - TWO thresholds for boundary and core.
        # Accept EITHER a scalar (ERA5-style absolute 241/225 K) OR a 2-D per-cell
        # field (RegCM). The RegCM cloud-top proxy's cold bias is latitude-dependent
        # (~0 K in the tropics, ~15 K at 40-45 N), so a single scalar over-detects the
        # tropics and starves the continental mid-latitudes. A per-cell threshold field
        # (same idea as the vor850/windspeed 75th-percentile fields) removes that skew.
        # Comparisons like `bt <= thr` broadcast fine against a (lat, lon) array.
        def _as_threshold(val):
            if val is None:
                return None
            # Short-circuit plain numbers. _convert_to_numpy does `data.values`,
            # which raises on a float and logs a scary "Error converting data to
            # numpy" before the float(val) fallback below quietly does the right
            # thing — misleading noise, once per scalar threshold per file.
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            arr = self._convert_to_numpy(val, dtype=np.float32)
            if arr is not None and getattr(arr, 'ndim', 0) >= 2:
                return np.squeeze(arr)          # 2-D per-cell field
            return float(val)                    # scalar

        self.bt_boundary_threshold = (_as_threshold(self.global_percentiles['BT_boundary'])
                                      if 'BT_boundary' in self.global_percentiles else None)
        self.bt_core_threshold = (_as_threshold(self.global_percentiles['BT_core'])
                                  if 'BT_core' in self.global_percentiles else None)

        # MCS precipitation thresholds (mcs_source="precip", the default). Same
        # scalar-or-2D plumbing as BT: `prc >= thr` broadcasts over (lat, lon)
        # exactly as `bt <= thr` does. These are fixed physical constants by
        # design — see utils.load_regcm_pr_thresholds for why there is
        # deliberately no per-cell calibration analogue of compute_bt_thresholds.
        self.pr_shield_threshold = _as_threshold(self.global_percentiles.get('PR_shield'))
        self.pr_core_threshold = _as_threshold(self.global_percentiles.get('PR_core'))
        self.pr_convfrac_threshold = _as_threshold(self.global_percentiles.get('PR_convfrac'))

        
            
        # Add after the existing global percentile conversions (around line 80-95)
        if 'Temp500' in self.global_percentiles:
            self.temp500_threshold = float(self.global_percentiles['Temp500'])
        else:
            self.temp500_threshold = None

        if 'vor850_per_hurr' in self.global_percentiles:
            self.vor850_hurricane_threshold = float(self.global_percentiles['vor850_per_hurr'])
        else:
            self.vor850_hurricane_threshold = None

        if 'windspeed_per_hurr' in self.global_percentiles:
            self.windspeed_hurricane_threshold = float(self.global_percentiles['windspeed_per_hurr'])
        else:
            self.windspeed_hurricane_threshold = None

        if mslp_climatology is not None:
            print("Using pre-aligned MSLP climatology")
            self.mslp_clim = self._convert_to_numpy(mslp_climatology, dtype=np.float32)
        else:
            print("⚠ No MSLP climatology provided — Condition ii disabled")
            self.mslp_clim = None

        # IWV monthly climatology → anomaly-based MCS moisture gate (Option 1).
        # mirrors the MSLP climatology plumbing above. (12, H, W) on the data grid.
        self.mcs_moisture_mode = mcs_moisture_mode      # "anomaly" | "anomaly_soft" | "absolute"
        self.iwv_anom_threshold = float(iwv_anom_threshold)
        self.iwv_anom_soft_band = float(iwv_anom_soft_band)

        # MCS detection source. "precip" (default) uses convective precipitation;
        # "bt" is the legacy cloud-top-proxy branch, kept only for A/B comparison.
        # RegCM's BT proxy peaks in WINTER over the Great Plains and bottoms out in
        # July — anti-correlated with the MCS season — so "bt" cannot recover the
        # continental signal at any threshold.
        if mcs_source not in ("precip", "bt"):
            raise ValueError(f"mcs_source must be 'precip' or 'bt', got {mcs_source!r}")
        self.mcs_source = mcs_source
        print(f"MCS detection source: {self.mcs_source}"
              + (" (legacy cloud-top proxy — A/B only)" if self.mcs_source == "bt" else ""))
        if iwv_climatology is not None:
            print(f"Using pre-aligned IWV monthly climatology — MCS moisture gate = '{mcs_moisture_mode}'")
            self.iwv_clim = self._convert_to_numpy(iwv_climatology, dtype=np.float32)
        else:
            print("⚠ No IWV climatology provided — MCS moisture gate falls back to ABSOLUTE IWV")
            self.iwv_clim = None

        # Ground truth handling
        self.has_ground_truth = ars is not None
        if self.has_ground_truth:
            self.ars = self._convert_to_numpy(ars, dtype=np.float32)
        
        # Setup geographical information for storm detection
        self._setup_geographical_info(data)
        
        print(f"[TIMING-Dataset] geo_info: {time.time()-_init_t:.2f}s"); _init_t = time.time()

        # Compute statistics with GPU optimization
        self._compute_statistics()
        
        print(f"[TIMING-Dataset] statistics: {time.time()-_init_t:.2f}s"); _init_t = time.time()

        # Setup thresholds for hurricane and ETC detection
        self._setup_storm_thresholds()
        
        print(f"[TIMING-Dataset] storm_thresholds: {time.time()-_init_t:.2f}s"); _init_t = time.time()
        
        # Process threshold data
        self._process_threshold_data()
        
        print(f"[TIMING-Dataset] threshold_processing: {time.time()-_init_t:.2f}s"); _init_t = time.time()

        # PRE-COMPUTE all 14-channel tensors for every sample so __getitem__
        # becomes a cheap array lookup instead of expensive scipy operations.
        if precompute:
            # self._precompute_all_channels()
            self._precompute_all_channels_cached(year=year, month=month)

        print(f"[TIMING-Dataset] precompute_channels: {time.time()-_init_t:.2f}s")
        
        # Optional: preload data for faster access
        if self.preload_data:
            self._preload_tensors()
        
        print(f"Dataset initialization complete - {self.total_samples} samples ready for multi-level multi-storm detection")

    def _get_tensor_cache_path(self, year, month):
        """Return (npy_path, meta_path) for this year/month."""
        date_str  = f"{year:04d}{month:02d}"
        npy_path  = os.path.join(self.TENSOR_CACHE_DIR, f"tensors_{date_str}.npy")
        meta_path = os.path.join(self.TENSOR_CACHE_DIR, f"tensors_{date_str}.meta.npy")
        return npy_path, meta_path

    def _precompute_all_channels_cached(self, year=None, month=None):
        """
        Smart precompute: load from disk cache if available, else compute + save.

        Fast path  (~5 sec) : tensors_{YYYYMM}.npy exists → np.load()
        Slow path  (~23 min): .npy missing → _precompute_all_channels() → np.save()

        Args:
            year  : int, e.g. 1980  (used to find cache file)
            month : int, e.g. 1     (used to find cache file)
        """
        t0 = _time.time()

        # ── Try fast path first ───────────────────────────────────────────────────
        if year is not None and month is not None:
            npy_path, meta_path = self._get_tensor_cache_path(year, month)

            if os.path.exists(npy_path):
                try:
                    loaded = np.load(npy_path, mmap_mode='r')   # memory-mapped: no RAM copy

                    # Validate shape matches current dataset (14 channels, MCS on)
                    expected_shape = (self.total_samples, 14,
                                    self.target_size[0], self.target_size[1])

                    if loaded.shape == expected_shape:
                        # ✅ Cache hit — copy to contiguous array for fast indexing
                        self.precomputed_data = np.array(loaded, dtype=np.float32)
                        elapsed = _time.time() - t0
                        print(f"  ✅ [TENSOR CACHE HIT] Loaded {npy_path} "
                            f"in {elapsed:.1f}s  "
                            f"({self.precomputed_data.nbytes/1024**2:.0f} MB)")
                        return

                    else:
                        # Shape mismatch — cache is stale, recompute
                        print(f"  ⚠️  Cache shape mismatch: "
                            f"expected {expected_shape}, got {loaded.shape}. "
                            f"Recomputing...")

                except Exception as e:
                    print(f"  ⚠️  Cache load failed ({e}). Recomputing...")

        # ── Slow path: full precompute ────────────────────────────────────────────
        print(f"  🔄 [TENSOR CACHE MISS] Running full precompute "
            f"(~23 min) for {year:04d}-{month:02d}...")
        self._precompute_all_channels()   # original method — unchanged

        # ── Save to disk for next time ────────────────────────────────────────────
        if year is not None and month is not None:
            try:
                os.makedirs(self.TENSOR_CACHE_DIR, exist_ok=True)
                npy_path, meta_path = self._get_tensor_cache_path(year, month)

                np.save(npy_path, self.precomputed_data)

                # Save time metadata alongside tensors
                np.save(meta_path, self.time)

                elapsed = _time.time() - t0
                print(f"  💾 [TENSOR CACHE SAVED] {npy_path} "
                    f"({self.precomputed_data.nbytes/1024**2:.0f} MB, "
                    f"{elapsed/60:.1f} min total)")

            except Exception as e:
                # Don't crash training if save fails — just warn
                print(f"  ⚠️  Could not save tensor cache: {e}")


    def _precompute_all_channels(self):
        """
        Pre-compute all 14-channel tensors for every sample at init time.
        This moves the expensive scipy/numpy operations (gaussian_filter,
        minimum_filter, label, median_filter, distance_transform_edt, etc.)
        out of __getitem__ which is called per-batch by DataLoader workers.
        
        After this, __getitem__ is just an array lookup → near-zero CPU cost.
        """
        import time as _time
        t0 = _time.time()
        N = self.total_samples
        H_target, W_target = self.target_size
        
        # Allocate storage for all precomputed tensors (14 channels, MCS on)
        self.precomputed_data = np.empty(
            (N, 14, H_target, W_target), dtype=np.float32
        )

        print(f"Pre-computing 14-channel tensors for {N} samples "
            f"(target size {H_target}×{W_target})...")

        for idx in range(N):
            try:
                tensor = self._compute_single_sample(idx)
                self.precomputed_data[idx] = tensor
            except Exception as e:
                print(f"  Warning: Error precomputing sample {idx}: {e}")
                self.precomputed_data[idx] = np.zeros(
                    (14, H_target, W_target), dtype=np.float32
                )
            
            if (idx + 1) % 20 == 0 or idx == N - 1:
                elapsed = _time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (N - idx - 1) / rate if rate > 0 else 0
                print(f"  Pre-computed {idx+1}/{N} samples "
                    f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")
        
        total_time = _time.time() - t0
        mem_mb = self.precomputed_data.nbytes / 1024**2
        print(f"Pre-computation complete: {total_time:.1f}s, "
            f"{mem_mb:.0f} MB for {N} samples")
    
    def _compute_single_sample(self, idx: int) -> np.ndarray:
        """
        Compute the 14-channel tensor for a single sample.
        This is the same logic as the original __getitem__ but returns
        a numpy array of shape (14, H_target, W_target).
        """
        # Normalize
        if self.preload_data and hasattr(self, 'ivt_normalized'):
            ivt_sample = self.ivt_normalized[idx]
            mslp_sample = self.mslp_normalized[idx]
        else:
            ivt_sample = (self.ivt[idx] - self.ivt_mean) / self.ivt_std
            mslp_sample = (self.mslp[idx] - self.mslp_mean) / self.mslp_std
        
        # AR detection mask
        if self.percentile_threshold is not None:
            ivt_binary = (self.ivt[idx] > self.percentile_threshold).astype(np.float32)
        else:
            ivt_binary = (self.ivt[idx] > 250).astype(np.float32)
        
        # Geographical masks
        height, width = self.ivt[idx].shape
        
        hurricane_lat_mask_2d = np.zeros((height, width), dtype=np.float32)
        if len(self.hurricane_lat_mask.shape) == 1:
            for i in range(min(height, len(self.hurricane_lat_mask))):
                if self.hurricane_lat_mask[i]:
                    hurricane_lat_mask_2d[i, :] = 1.0
        else:
            hurricane_lat_mask_2d = self.hurricane_lat_mask.astype(np.float32)
        
        etc_lat_mask_2d = np.zeros((height, width), dtype=np.float32)
        if len(self.etc_lat_mask.shape) == 1:
            for i in range(min(height, len(self.etc_lat_mask))):
                if self.etc_lat_mask[i]:
                    etc_lat_mask_2d[i, :] = 1.0
        else:
            etc_lat_mask_2d = self.etc_lat_mask.astype(np.float32)
        
        # ── MCS re-enabled (RegCM BT now available) ────────────────────────────
        mcs_lat_mask_2d = np.zeros((height, width), dtype=np.float32)
        if len(self.mcs_lat_mask.shape) == 1:
            for i in range(min(height, len(self.mcs_lat_mask))):
                if self.mcs_lat_mask[i]:
                    mcs_lat_mask_2d[i, :] = 1.0
        else:
            mcs_lat_mask_2d = self.mcs_lat_mask.astype(np.float32)

        # Storm masks
        hurricane_masks = self._create_hurricane_masks(idx, hurricane_lat_mask_2d)
        etc_masks = self._create_etc_masks(idx, etc_lat_mask_2d)
        mcs_masks = self._create_mcs_masks(idx, mcs_lat_mask_2d)
        
        # Wind speed binary mask
        if self.windspeed_hurricane_threshold is not None:
            wind_speed_binary = (self.wind_speed[idx] >= self.windspeed_hurricane_threshold).astype(np.float32)
        else:
            wind_speed_binary = (self.wind_speed[idx] >= self.wind_speed_hurricane_threshold).astype(np.float32)
        wind_speed_binary *= hurricane_lat_mask_2d
        
        # Stack channels
        channels = [
            ivt_sample,
            ivt_binary,
            hurricane_masks['low_pressure'],
            hurricane_masks['high_vorticity'],
            hurricane_masks['high_wind_speed'],
            hurricane_masks['strict'],
            wind_speed_binary,
            etc_masks['etc_low_pressure'],
            etc_masks['etc_vorticity'],
            etc_masks['etc_wind_speed'],
            etc_masks['etc_enhanced'],
            mcs_masks['mcs_boundary'],
            mcs_masks['mcs_blobs'],
            mcs_masks['mcs_core'],
        ]

        
        target_shape = channels[0].shape
        cleaned_channels = []
        for ch in channels:
            if not isinstance(ch, np.ndarray):
                ch = np.asarray(ch, dtype=np.float32)
            else:
                ch = np.asarray(ch, dtype=np.float32)
            if ch.shape != target_shape:
                ch = np.broadcast_to(ch, target_shape)
            cleaned_channels.append(np.ascontiguousarray(ch, dtype=np.float32))
        
        data = np.stack(cleaned_channels, axis=0)  # (14, H, W)
        
        # Resize to target if necessary
        if data.shape[1:] != (self.target_size[0], self.target_size[1]):
            data = np.ascontiguousarray(data, dtype=np.float32)
            try:
                tensor = torch.from_numpy(data).unsqueeze(0)
            except (TypeError, RuntimeError):
                # Fallback for numpy/torch build mismatch (e.g. ROCm on Frontier)
                tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            tensor = F.interpolate(
                tensor, size=self.target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            data = tensor.numpy()
        
        return data

    def _convert_to_numpy(self, data: xr.DataArray, dtype: np.dtype = np.float32) -> np.ndarray:
        """Convert xarray to numpy with memory optimization"""
        if data is None:
            return None
        
        try:
            # Convert to numpy with specified dtype
            numpy_data = data.values.astype(dtype)
            
            # Handle NaN values
            if np.isnan(numpy_data).any():
                numpy_data = np.nan_to_num(numpy_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            return numpy_data
        except Exception as e:
            print(f"Warning: Error converting data to numpy: {e}")
            return None
    
    def _setup_geographical_info(self, data: xr.Dataset):
        """Setup geographical information for storm detection"""
        if hasattr(data, 'latitude'):
            self.latitudes = data.latitude.values
            self.lats = self.latitudes
            
            # Hurricane detection zone (tropical/subtropical regions)
            self.hurricane_lat_mask = np.logical_and(self.latitudes >= -45, self.latitudes <= 45)
            
            # ETC detection zone (mid-to-high latitudes, overlapping with hurricane zone)
            # ETCs typically form between 30°-70° latitude in both hemispheres
            self.etc_lat_mask = np.logical_or(
                np.logical_and(self.latitudes >= 20, self.latitudes <= 70),  # Northern hemisphere
                np.logical_and(self.latitudes >= -70, self.latitudes <= -20)  # Southern hemisphere
            )
            
            valid_hurricane_lats = np.sum(self.hurricane_lat_mask)
            valid_etc_lats = np.sum(self.etc_lat_mask)
            
            
            # MCS detection zone (primarily in mid-latitudes where convection is strong)
            self.mcs_lat_mask = np.logical_and(self.latitudes >= -60, self.latitudes <= 60)

            valid_mcs_lats = np.sum(self.mcs_lat_mask)
            
            print(f"Hurricane detection zone: {valid_hurricane_lats} latitude points between -45° and 45°")
            print(f"ETC detection zone: {valid_etc_lats} latitude points between 20°-70° and -70°--20°")
            print(f"MCS detection zone: {valid_mcs_lats} latitude points between -60° and 60°")
            
        else:
        
            print(f"Hurricane detection zone missing")
            print(f"ETC detection zone missing")
            print(f"MCS detection zone missing")
            
    def _compute_statistics(self):
        """Compute statistics for normalization - uses global stats if provided, else falls back to local monthly stats"""
        print("Computing optimized statistics...")
        
        # Use vectorized operations for better performance
        def safe_stats(data, name):
            if data is None:
                return 0.0, 1.0
            
            # Filter out invalid values
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0:
                print(f"Warning: No valid data for {name}")
                return 0.0, 1.0
            
            # For IVT, only use positive values
            if name == 'ivt':
                valid_data = valid_data[valid_data > 0]
                if len(valid_data) == 0:
                    return 0.0, 1.0
            
            mean_val = float(np.mean(valid_data))
            std_val = float(np.std(valid_data) + 1e-8)
            
            return mean_val, std_val
        
        # ── Compute local statistics as fallback ──
        self.ivt_mean, self.ivt_std                 = safe_stats(self.ivt,        'ivt')
        self.mslp_mean, self.mslp_std               = safe_stats(self.mslp,       'mslp')
        self.ivt_grad_mean, self.ivt_grad_std       = safe_stats(self.ivt_grad,   'ivt_grad')
        self.vorticity_grad_mean, self.vorticity_grad_std = safe_stats(self.vorticity_grad, 'vorticity_grad')
        self.vorticity_mean, self.vorticity_std     = safe_stats(self.vorticity,  'vorticity')
        self.wind_speed_mean, self.wind_speed_std   = safe_stats(self.wind_speed, 'wind_speed')
        self.temp_500mb_mean, self.temp_500mb_std   = safe_stats(self.temp_500mb, 'temp_500mb')
        self.iwv_mean, self.iwv_std                 = safe_stats(self.iwv,        'iwv')
        self.windhshear_mean, self.windhshear_std   = safe_stats(self.windhshear, 'windhshear')
        self.bt_ds_mean, self.bt_ds_std             = safe_stats(self.bt_ds,      'bt_ds')

        print(f"Local statistics computed - IVT: {self.ivt_mean:.2f}±{self.ivt_std:.2f}, "
            f"MSLP: {self.mslp_mean:.1f}±{self.mslp_std:.1f}")

        # ── Override with global stats where available ──
        print("\nApplying global statistics overrides...")

        if 'IVT' in self.global_stats:
            self.ivt_mean = float(self.global_stats['IVT']['mean'])
            self.ivt_std  = float(self.global_stats['IVT']['std'])
            print(f"✓ GLOBAL IVT stats:        mean={self.ivt_mean:.4f}, std={self.ivt_std:.4f}")
        else:
            print(f"✗ LOCAL  IVT stats:        mean={self.ivt_mean:.4f}, std={self.ivt_std:.4f}")

        if 'MSLP' in self.global_stats:
            self.mslp_mean = float(self.global_stats['MSLP']['mean'])
            self.mslp_std  = float(self.global_stats['MSLP']['std'])
            print(f"✓ GLOBAL MSLP stats:       mean={self.mslp_mean:.2f}, std={self.mslp_std:.4f}")
        else:
            print(f"✗ LOCAL  MSLP stats:       mean={self.mslp_mean:.2f}, std={self.mslp_std:.4f}")

        if 'IWV' in self.global_stats:
            self.iwv_mean = float(self.global_stats['IWV']['mean'])
            self.iwv_std  = float(self.global_stats['IWV']['std'])
            print(f"✓ GLOBAL IWV stats:        mean={self.iwv_mean:.4f}, std={self.iwv_std:.4f}")
        else:
            print(f"✗ LOCAL  IWV stats:        mean={self.iwv_mean:.4f}, std={self.iwv_std:.4f}")

        if 'BT' in self.global_stats:
            self.bt_ds_mean = float(self.global_stats['BT']['mean'])
            self.bt_ds_std  = float(self.global_stats['BT']['std'])
            print(f"✓ GLOBAL BT stats:         mean={self.bt_ds_mean:.4f}, std={self.bt_ds_std:.4f}")
        else:
            print(f"✗ LOCAL  BT stats:         mean={self.bt_ds_mean:.4f}, std={self.bt_ds_std:.4f}")

        if 'vorticity' in self.global_stats:
            self.vorticity_mean = float(self.global_stats['vorticity']['mean'])
            self.vorticity_std  = float(self.global_stats['vorticity']['std'])
            print(f"✓ GLOBAL vorticity stats:  mean={self.vorticity_mean:.4e}, std={self.vorticity_std:.4e}")
        else:
            print(f"✗ LOCAL  vorticity stats:  mean={self.vorticity_mean:.4e}, std={self.vorticity_std:.4e}")

        if 'wind_speed' in self.global_stats:
            self.wind_speed_mean = float(self.global_stats['wind_speed']['mean'])
            self.wind_speed_std  = float(self.global_stats['wind_speed']['std'])
            print(f"✓ GLOBAL wind_speed stats: mean={self.wind_speed_mean:.4f}, std={self.wind_speed_std:.4f}")
        else:
            print(f"✗ LOCAL  wind_speed stats: mean={self.wind_speed_mean:.4f}, std={self.wind_speed_std:.4f}")

        if 'temp_500mb' in self.global_stats:
            self.temp_500mb_mean = float(self.global_stats['temp_500mb']['mean'])
            self.temp_500mb_std  = float(self.global_stats['temp_500mb']['std'])
            print(f"✓ GLOBAL temp_500mb stats: mean={self.temp_500mb_mean:.4f}, std={self.temp_500mb_std:.4f}")
        else:
            print(f"✗ LOCAL  temp_500mb stats: mean={self.temp_500mb_mean:.4f}, std={self.temp_500mb_std:.4f}")

        # windhshear has no global stats available — always local
        print(f"✗ LOCAL  windhshear stats: mean={self.windhshear_mean:.4f}, std={self.windhshear_std:.4f}")

        # ── Final summary ──
        print("\n" + "="*60)
        print("FINAL NORMALIZATION STATISTICS")
        print("="*60)
        print(f"  IVT:        {self.ivt_mean:.4f} ± {self.ivt_std:.4f}")
        print(f"  MSLP:       {self.mslp_mean:.2f} ± {self.mslp_std:.4f}")
        print(f"  IWV:        {self.iwv_mean:.4f} ± {self.iwv_std:.4f}")
        print(f"  BT:         {self.bt_ds_mean:.4f} ± {self.bt_ds_std:.4f}")
        print(f"  Vorticity:  {self.vorticity_mean:.4e} ± {self.vorticity_std:.4e}")
        print(f"  Wind speed: {self.wind_speed_mean:.4f} ± {self.wind_speed_std:.4f}")
        print(f"  Temp 500mb: {self.temp_500mb_mean:.4f} ± {self.temp_500mb_std:.4f}")
        print(f"  Windhshear: {self.windhshear_mean:.4f} ± {self.windhshear_std:.4f}")
        print("="*60 + "\n")
        
    def _setup_storm_thresholds(self):
        """Setup adaptive thresholds using global percentiles where available"""
        print("Computing adaptive storm detection thresholds using global percentiles...")
        
        self.mslp_hurricane_threshold = 100200  # Keep this local as it's meteorologically consistent

        if self.temp500_threshold is not None:
            print("✓ Using fixed global 500mb temperature threshold for hurricanes")
            self.temp500_threshold = float(self.temp500_threshold)
            # REMOVE THIS LINE: self.temp_warm_core_threshold = None
        else:
            self.temp_warm_core_threshold = np.percentile(self.temp_500mb, 85)
            print("✗ Fallback to local 500mb temperature threshold")
        
        if self.vor850_hurricane_threshold is not None:
            print("✓ Using fixed global 850mb vorticity threshold for hurricanes")
            self.vor850_hurricane_threshold = float(self.vor850_hurricane_threshold)
            # REMOVE THIS LINE: self.vorticity_hurricane_threshold = None
        else:
            if self.vorticity_threshold is not None:
                self.vorticity_hurricane_threshold = self.vorticity_threshold
            else:
                self.vorticity_hurricane_threshold = np.percentile(self.vorticity, 85)
            print("✗ Fallback to local vorticity threshold")
        
        if self.windspeed_hurricane_threshold is not None:
            print("✓ Using fixed global wind speed threshold for hurricanes")
            self.windspeed_hurricane_threshold = max(float(self.windspeed_hurricane_threshold), 10.0)
            # REMOVE THIS LINE: self.wind_speed_hurricane_threshold = None
        else:
            self.wind_speed_hurricane_threshold = max(np.percentile(self.wind_speed, 90), 15.0)
            print("✗ Fallback to local wind speed threshold")

        # WARM CORE THRESHOLDS (T300-T500 difference)
        print("Setting up warm core thresholds...")

        # if self.warmcore_threshold is not None:
        #     print(f"✓ Using global warm core threshold: {self.warmcore_threshold:.2f} K")
        #     # No fallback needed
        # else:
        #     # Fallback: Use anomaly-based approach
        #     self.warmcore_threshold = -15.0  # Literature default
        #     print(f"✗ Fallback to literature warm core threshold: {self.warmcore_threshold:.2f} K")

        # if self.warmcore_anomaly_threshold is not None:
        #     print(f"✓ Using global warm core anomaly threshold: {self.warmcore_anomaly_threshold:.2f} K")
        # else:
        #     self.warmcore_anomaly_threshold = 3.5  # Literature default
        #     print(f"✗ Fallback to literature warm core anomaly threshold: {self.warmcore_anomaly_threshold:.2f} K")

        # Compute warm core climatology for anomaly-based detection (RECOMMENDED)
        # if not hasattr(self, 'warmcore_climatology'):
        #     self.warmcore_climatology = np.nanmean(self.temp_500mb, axis=0)
        #     print("✓ Warm core climatology computed")

        # # MCS thresholds
        # print("Setting up MCS thresholds using global single values...")

        # if self.bt_threshold is not None:
        #     print(f"✓ Using global brightness temperature threshold: {self.bt_threshold:.2f} K")
        #     self.bt_deep_convection_threshold = None  # Signal to use global threshold
        # else:
        #     self.bt_deep_convection_threshold = max(np.percentile(self.bt_ds, 5), 220.0)
        #     print(f"✗ Fallback to local brightness temperature threshold: {self.bt_deep_convection_threshold:.2f} K")

        # if self.tpr_threshold is not None:
        #     print(f"✓ Using global TPR threshold: {self.tpr_threshold:.4f} mm/h")
        #     self.tpr_heavy_threshold = None  # Signal to use global threshold
        # else:
        #     tpr_nonzero = self.tpr[self.tpr > 0.01]
        #     if len(tpr_nonzero) > 100:
        #         self.tpr_heavy_threshold = np.percentile(tpr_nonzero, 85)
        #     else:
        #         self.tpr_heavy_threshold = np.percentile(self.tpr, 95)
        #     print(f"✗ Fallback to local TPR threshold: {self.tpr_heavy_threshold:.4f} mm/h")


        # MCS thresholds - TWO THRESHOLDS APPROACH (BT only)
        print("Setting up MCS thresholds using two-threshold blob detection (BT only)...")

        # Boundary threshold (241K)
        if self.bt_boundary_threshold is not None:
            print(f"✓ Using global boundary brightness temperature threshold: {_fmt_thr(self.bt_boundary_threshold)}")
            self.bt_boundary_threshold_local = None
        else:
            self.bt_boundary_threshold_local = 241.0  # Fixed meteorological threshold
            print(f"✗ Fallback to literature boundary brightness temperature threshold: {self.bt_boundary_threshold_local:.2f} K")

        # Core threshold (225K)
        if self.bt_core_threshold is not None:
            print(f"✓ Using global core brightness temperature threshold: {_fmt_thr(self.bt_core_threshold)}")
            self.bt_core_threshold_local = None
        else:
            self.bt_core_threshold_local = 225.0  # Fixed meteorological threshold
            print(f"✗ Fallback to literature core brightness temperature threshold: {self.bt_core_threshold_local:.2f} K")


        # ETC thresholds
        self.mslp_etc_threshold = 101000  # Keep this local as it's meteorologically consistent

        # Print comprehensive threshold summary
        print("\n" + "="*60)
        print("STORM DETECTION THRESHOLD SUMMARY")
        print("="*60)
        
        print("\nHURRICANE THRESHOLDS:")
        print(f"  MSLP: {self.mslp_hurricane_threshold:.1f} Pa")
        
        if self.temp500_threshold is not None:
            print(f"  Temperature warm core: GLOBAL SCALAR ({self.temp500_threshold:.2f} K)")
        else:
            print(f"  Temperature warm core: LOCAL FIXED ({self.temp_warm_core_threshold:.2f} K)")
        
        if self.vor850_hurricane_threshold is not None:
            print(f"  Vorticity: GLOBAL SCALAR ({self.vor850_hurricane_threshold:.2e} s⁻¹)")
        else:
            print(f"  Vorticity: LOCAL FIXED ({self.vorticity_hurricane_threshold:.2e} s⁻¹)")
        
        if self.windspeed_hurricane_threshold is not None:
            print(f"  Wind speed: GLOBAL SCALAR ({self.windspeed_hurricane_threshold:.2f} m/s)")
        else:
            print(f"  Wind speed: LOCAL FIXED ({self.wind_speed_hurricane_threshold:.2f} m/s)")
        
        print("\nETC THRESHOLDS:")
        print(f"  MSLP: {self.mslp_etc_threshold:.1f} Pa")
        
        if self.vor850_threshold is not None:
            print(f"  Vorticity 850mb: GLOBAL SPATIAL (range: {self.vor850_threshold.min():.2e} to {self.vor850_threshold.max():.2e} s⁻¹)")
        else:
            print(f"  Vorticity 850mb: LOCAL FIXED (threshold would be set)")
        
        if self.windspeed_threshold is not None:
            print(f"  Wind speed: GLOBAL SPATIAL (range: {self.windspeed_threshold.min():.2f} to {self.windspeed_threshold.max():.2f} m/s)")
        else:
            print(f"  Wind speed: LOCAL FIXED (threshold would be set)")

        print("\nMCS THRESHOLDS:")

        # if self.tpr_threshold is not None:
        #     print(f"  Total Precipitation Rate: GLOBAL SCALAR ({self.tpr_threshold:.4f} mm/h)")
        # else:
        #     print(f"  Total Precipitation Rate: LOCAL FIXED ({self.tpr_heavy_threshold:.4f} mm/h)")

        if self.bt_boundary_threshold is not None:
            print(f"  Brightness Temperature (Boundary): {_fmt_thr(self.bt_boundary_threshold)}")
        else:
            print(f"  Brightness Temperature (Boundary): LITERATURE FIXED ({self.bt_boundary_threshold_local:.1f} K)")

        if self.bt_core_threshold is not None:
            print(f"  Brightness Temperature (Core): {_fmt_thr(self.bt_core_threshold)}")
        else:
            print(f"  Brightness Temperature (Core): LITERATURE FIXED ({self.bt_core_threshold_local:.1f} K)")
        
        print("="*60 + "\n")

    def _process_threshold_data(self):
        """Process percentile and vorticity threshold data"""
        print("Processing threshold data...")
        

        print("Using fixed IVT threshold of 400 kg/m/s")
        self.percentile_threshold = self.ivt_percentile  # Fixed threshold for IVT
        print("Using fixed IVT threshold of", self.percentile_threshold)
        
        # Process vorticity threshold
        if self.vorticity_threshold is not None:
            try:
                if isinstance(self.vorticity_threshold, (xr.Dataset, xr.DataArray)):
                    self.vort_threshold = self.vorticity_threshold.values.astype(np.float32)
                else:
                    self.vort_threshold = float(self.vorticity_threshold)
                
                print("Vorticity threshold processed successfully")
            except Exception as e:
                print(f"Warning: Error processing vorticity threshold: {e}")
                self.vort_threshold = None
        else:
            self.vort_threshold = None


    ##   Hurricane detection helper functions   ────────────────────────────────────────
    def _smooth_mslp(self, mslp_field: np.ndarray, sigma_km: float = 200.0) -> np.ndarray:
        """Smooth MSLP field with Gaussian filter"""
        grid_resolution_km = 27.5
        sigma_pixels = sigma_km / grid_resolution_km
        smoothed = gaussian_filter(mslp_field, sigma=sigma_pixels, mode='wrap')
        return smoothed
        
    def _compute_mslp_anomaly(self, idx: int) -> np.ndarray:
        """
        MSLP anomaly relative to long-term monthly climatology.
        
        Replicates CyTRACK's 14-day local mean for a global domain.
        Each grid point's anomaly is relative to its own climatological
        mean for that calendar month — removing the meridional gradient
        that makes zonal/global means useless as a reference.
        
        Falls back to zeros (disabling Condition ii) if no climatology.
        """
        if self.mslp_clim is None:
            return np.zeros_like(self.mslp[idx])
        
        # Get calendar month for this sample (0-indexed)
        month_idx = int(pd.Timestamp(self.time[idx]).month) - 1  # 0-indexed
        
        clim_field = self.mslp_clim[month_idx]       # (H, W)
        anomaly    = self.mslp[idx] - clim_field      # negative = below climatology
        return anomaly

    def _compute_iwv_anomaly(self, idx: int) -> np.ndarray:
        """IWV anomaly (mm) vs the monthly climatology for this sample's calendar month.

        Each monthly file shares one month, so the climatology slice is constant
        across the file. Returns zeros if no climatology was provided.
        """
        if self.iwv_clim is None:
            return np.zeros_like(self.iwv[idx])
        month_idx = int(pd.Timestamp(self.time[idx]).month) - 1   # 0-indexed
        clim_field = self.iwv_clim[month_idx]
        if clim_field.shape != self.iwv[idx].shape:
            from scipy.ndimage import zoom
            zf = (self.iwv[idx].shape[0] / clim_field.shape[0],
                  self.iwv[idx].shape[1] / clim_field.shape[1])
            clim_field = zoom(clim_field, zf, order=1)
        return self.iwv[idx] - clim_field

    def _detect_mslp_local_minima(self, mslp_field: np.ndarray, 
                                    pressure_diff_threshold: float = 150.0,
                                    ring_radius_km: float = 500.0) -> np.ndarray:
        """
        Detect MSLP local minima with surrounding pressure gradient check.
        Hemisphere-aware with relaxed thresholds.
        """
        grid_resolution_km = 27.5
        ring_radius_pixels = int(ring_radius_km / grid_resolution_km)
        
        local_min = minimum_filter(mslp_field, size=3, mode='wrap')
        minima_mask = (mslp_field == local_min).astype(np.float32)
        
        ring_max = maximum_filter(mslp_field, size=2*ring_radius_pixels+1, mode='wrap')
        pressure_diff = ring_max - mslp_field
        
        lat_2d = np.broadcast_to(self.lats[:, np.newaxis], mslp_field.shape)
        
        nh_mask = (lat_2d >= 20) & (lat_2d <= 70)
        sh_mask = (lat_2d <= -20) & (lat_2d >= -70)
        
        nh_threshold = pressure_diff_threshold
        sh_threshold = pressure_diff_threshold * 0.7
        
        nh_valid = (minima_mask > 0) & nh_mask & (pressure_diff >= nh_threshold)
        sh_valid = (minima_mask > 0) & sh_mask & (pressure_diff >= sh_threshold)
        valid_minima = nh_valid | sh_valid
        
        very_low_pressure = mslp_field < 101000
        weak_gradient_but_intense = (minima_mask > 0) & very_low_pressure & (pressure_diff >= pressure_diff_threshold * 0.5)
        valid_minima = valid_minima | weak_gradient_but_intense
        
        return valid_minima.astype(np.float32)

    def _detect_hurricane_centres(
        self,
        idx: int,
        mslp_anomaly: np.ndarray,
        min_slp_threshold: float      = 101000.0,  # Pa (1010 hPa)
        mslp_anomaly_threshold: float = -800.0,    # Pa (-8 hPa zonal anomaly)
        great_circle_deg: float       = 5.5,
        dmslp_threshold: float        = 300.0,     # Pa
        vorticity_threshold: float    = 1.45e-5,
        filter_center_km: float       = 400.0,
        max_wind_threshold: float     = 8.0,       # m/s — CyTRACK Condition iv
        radius_msw_km: float          = 100.0,     # km — CyTRACK radius_for_msw
    ) -> np.ndarray:
        """
        CyTRACK-style TC centre detection.
            i   — MSLP < min_slp_threshold (1010 hPa)
            ii  — zonal MSLP anomaly < mslp_anomaly_threshold (-5 hPa)
            iii — MSLP rises >= dmslp_threshold within great_circle_deg
            iv  — max wind within radius_msw_km >= max_wind_threshold
            v   — surface vorticity >= vorticity_threshold (NH) / <= -threshold (SH)
            vii — proximity dedup: keep lowest-MSLP within filter_center_km
        """
        mslp_field = self.mslp[idx]
        H, W       = mslp_field.shape
        grid_deg   = 0.25
        grid_km    = grid_deg * 111.0

        # ── Conditions i + ii + local minimum pre-filter ──────────────────────
        cond_i       = mslp_field < min_slp_threshold
        # Use size=5 to catch broader tropical lows near equator
        local_min    = minimum_filter(mslp_field, size=5, mode='wrap')
        is_local_min = (mslp_field == local_min)

        # Condition ii — apply ONLY in hurricane latitude band (-45° to +45°)
        # Outside tropics, mid-latitude anomalies are huge and meaningless for TC detection
        if mslp_anomaly is not None and np.any(mslp_anomaly != 0):
            lat_2d      = np.broadcast_to(self.lats[:, np.newaxis], mslp_field.shape)
            tc_lat_band = (lat_2d >= -45.0) & (lat_2d <= 45.0)
            
            # Inside TC band: require anomaly < threshold
            # Outside TC band: reject entirely (no TCs there anyway)
            cond_ii = np.where(
                tc_lat_band,
                mslp_anomaly < mslp_anomaly_threshold,
                False   # outside ±45°: never a TC centre
            )
        else:
            # No climatology: fall back to lat band only
            lat_2d  = np.broadcast_to(self.lats[:, np.newaxis], mslp_field.shape)
            cond_ii = (lat_2d >= -45.0) & (lat_2d <= 45.0)

        candidates = np.argwhere(cond_i & cond_ii & is_local_min)

        if len(candidates) == 0:
            return np.zeros((H, W), dtype=np.float32)

        # ── Condition iii — precompute radial sdiff ─────────────────────────────
        ring_px     = max(1, int(round(great_circle_deg / grid_deg)))
        ring_max    = maximum_filter(mslp_field, size=2 * ring_px + 1, mode='wrap')
        radial_diff = ring_max - mslp_field

        # ── Condition v — vorticity masks ─────────────────────────────────────
        # Near equator (|lat| < 10°), Coriolis → 0 so vorticity is naturally
        # weaker; use 50% reduced threshold to avoid missing near-equatorial TCs
        lat_2d    = np.broadcast_to(self.lats[:, np.newaxis], (H, W))
        vor_field = self.vorticity[idx]
        vor_thresh = self.vor850_hurricane_threshold if self.vor850_hurricane_threshold is not None \
                    else vorticity_threshold
        equatorial_band = np.abs(lat_2d) < 10.0
        vor_thresh_2d = np.where(equatorial_band, vor_thresh * 0.5, vor_thresh)
        nh_vor = (lat_2d >= 0) & (vor_field >=  vor_thresh_2d)
        sh_vor = (lat_2d <  0) & (vor_field <= -vor_thresh_2d)
        cond_v = nh_vor | sh_vor

        # ── Condition iv — precompute local wind max via max_filter ───────────
        radius_msw_px  = max(1, int(round(radius_msw_km / grid_km)))
        local_wind_max = maximum_filter(
            self.wind_speed[idx], size=2 * radius_msw_px + 1, mode='nearest'
        )

        # ── Per-candidate loop (conditions iii, iv, v) ────────────────────────
        # Relaxed logic: condition iii required, then EITHER iv OR v must pass.
        # Landfalling storms near Gulf/Atlantic coast can lose vorticity (surface
        # friction) while retaining wind, or vice versa — requiring both misses them.
        valid = np.zeros((H, W), dtype=np.float32)
        for (r, c) in candidates:
            if radial_diff[r, c] < dmslp_threshold:          # iii — always required
                continue
            has_wind = local_wind_max[r, c] >= max_wind_threshold   # iv
            has_vor  = cond_v[r, c]                                 # v
            if not (has_wind or has_vor):                    # need at least one
                continue
            valid[r, c] = 1.0

        # ── Condition vii — proximity deduplication ────────────────────────────
        centres = np.argwhere(valid > 0)
        if len(centres) == 0:
            return valid

        if len(centres) > 1:
            dists = np.sqrt(
                ((centres[:, 0:1] - centres[:, 0]) ** 2 +
                (centres[:, 1:2] - centres[:, 1]) ** 2).astype(np.float32)
            )
            filter_px   = max(1, int(round(filter_center_km / grid_km)))
            suppressed  = np.zeros(len(centres), dtype=bool)
            mslp_vals   = mslp_field[centres[:, 0], centres[:, 1]]
            order       = np.argsort(mslp_vals)

            for i_ord in order:
                if suppressed[i_ord]:
                    continue
                nearby = (dists[i_ord] < filter_px) & (~suppressed)
                nearby[i_ord] = False
                suppressed[nearby] = True

            dedup = np.zeros((H, W), dtype=np.float32)
            for k, (r, c) in enumerate(centres):
                if not suppressed[k]:
                    dedup[r, c] = 1.0
            return dedup

        return valid
    
    def _dilate_mslp_centers(self, minima_mask: np.ndarray, dilation_km: float = 300.0) -> np.ndarray:
        """Dilate MSLP center points to create regional masks"""
        grid_resolution_km = 27.5
        dilation_pixels = int(dilation_km / grid_resolution_km)
        
        y, x = np.ogrid[-dilation_pixels:dilation_pixels+1, -dilation_pixels:dilation_pixels+1]
        structure = (x*x + y*y <= dilation_pixels*dilation_pixels)
        dilated = binary_dilation(minima_mask > 0, structure=structure)
        
        return dilated.astype(np.float32)

    def _create_hurricane_masks(self, idx: int, lat_mask_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        CyTRACK-aligned hurricane detection.

        Conditions enforced:
            i   — MSLP < 1015 hPa
            ii  — MSLP anomaly < -2 hPa  (rolling 14-day mean)
            iii — MSLP rises >= 200 Pa within 5.5° radius
            v   — surface vorticity >= 1.45e-5 s⁻¹ (NH) / <= -1.45e-5 (SH)
            vii — proximity dedup: keep lowest-MSLP within 400 km
        
        Additional masks (fed as channels to the network):
            low_pressure     : dilated MSLP centre regions
            high_vorticity   : raw vorticity threshold mask
            high_wind_speed  : raw wind threshold mask
            strict           : centres dilated AND (vorticity OR wind)
        """

        # ── Rolling MSLP anomaly (Condition ii) ──────────────────────────────────
        mslp_anomaly = self._compute_mslp_anomaly(idx)

        # Add temporarily inside _create_hurricane_masks, before _detect_hurricane_centres call:
        if idx == 0:
            vor = self.vorticity[0]
            mslp = self.mslp[0]
            ws = self.wind_speed[0]
            print(f"\n[DATA RANGE CHECK]")
            print(f"  MSLP:      {mslp.min():.1f} to {mslp.max():.1f} Pa")
            print(f"  Vorticity: {vor.min():.4e} to {vor.max():.4e} s⁻¹")
            print(f"  Vorticity p50={np.percentile(vor,50):.2e}, p90={np.percentile(vor,90):.2e}, p99={np.percentile(vor,99):.2e}")
            print(f"  Wind speed:{ws.min():.1f} to {ws.max():.1f} m/s")
            print(f"  Wind p90={np.percentile(ws,90):.1f}, p95={np.percentile(ws,95):.1f}, p99={np.percentile(ws,99):.1f}")

        # ── CyTRACK centre detection (Conditions i, ii, iii, v, vii) ─────────────
        hurricane_centres = self._detect_hurricane_centres(
                idx,
                mslp_anomaly           = mslp_anomaly,
                min_slp_threshold      = 101500.0,   # 1015 hPa — CyTRACK paper value
                mslp_anomaly_threshold = -500.0,     # -5 hPa  — relaxed for Gulf/landfalling storms
                great_circle_deg       = 5.5,
                dmslp_threshold        = 150.0,      # 1.5 hPa — relaxed for land-disrupted gradients
                vorticity_threshold    = 1.45e-5,    # CyTRACK paper value
                filter_center_km       = 400.0,
                max_wind_threshold     = 8.0,        # CyTRACK paper value
                radius_msw_km          = 150.0,      # wider search for asymmetric landfalling winds
            )        

        # ── Dilate centres → regional influence mask ──────────────────────────────
        # CyTRACK uses max_dist=500 km for track linking; use same for mask dilation
        hurricane_region_mask = self._dilate_mslp_centers(
            hurricane_centres, dilation_km=200.0
        )

        # ── Low-pressure mask (Condition i, already enforced above, kept as channel)
        low_pressure_mask = (self.mslp[idx] < 101500.0).astype(np.float32)
        hurricane_mslp_filtered = low_pressure_mask * hurricane_region_mask

        # ── Vorticity mask (Condition v — spatially broad for network channel) ─────
        if self.vor850_hurricane_threshold is not None:
            vor_thresh = self.vor850_hurricane_threshold
        else:
            vor_thresh = 1.45e-5

        lat_2d    = np.broadcast_to(self.lats[:, np.newaxis], self.vorticity[idx].shape)
        nh_vor    = (lat_2d >= 0) & (self.vorticity[idx] >=  vor_thresh)
        sh_vor    = (lat_2d <  0) & (self.vorticity[idx] <= -vor_thresh)
        high_vorticity_mask = (nh_vor | sh_vor).astype(np.float32)

        # ── Wind speed mask (Condition iv proxy) ─────────────────────────────────
        if self.windspeed_hurricane_threshold is not None:
            wind_thresh = self.windspeed_hurricane_threshold
        else:
            wind_thresh = self.wind_speed_hurricane_threshold
        high_wind_speed_mask = (self.wind_speed[idx] >= wind_thresh).astype(np.float32)

        # ── Apply geographical (latitude) constraint ──────────────────────────────
        hurricane_mslp_filtered *= lat_mask_2d
        high_vorticity_mask     *= lat_mask_2d
        high_wind_speed_mask    *= lat_mask_2d

        # ── Strict mask: dilated centres that also have vorticity OR wind ─────────
        enhanced_strict_mask = (
            hurricane_mslp_filtered *
            ((high_vorticity_mask + high_wind_speed_mask) >= 1).astype(np.float32)
        )
        
        # ── IBTrACS injection: ONLY add points that physics detection missed ───
        if self.ibtracs_lookup is not None:
            try:
                dt_key = pd.Timestamp(self.time[idx]).to_pydatetime().replace(
                    minute=0, second=0, microsecond=0
                )
                storm_points = self.ibtracs_lookup.get(dt_key, [])

                if storm_points:
                    H, W = enhanced_strict_mask.shape
                    longitudes = np.linspace(-180.0, 180.0 - 0.25, W)
                    dilation_km = 200.0
                    dilation_px = max(1, int(round(dilation_km / (0.25 * 111.0))))
                    # Check radius: if physics already detected something
                    # within this radius of the IBTrACS point, skip injection
                    # Reduced from 300 km to 200 km — larger radius caused skips
                    # when a nearby ETC detection falsely suppressed injection
                    check_radius_px = max(1, int(round(200.0 / (0.25 * 111.0))))
                    y_idx, x_idx = np.ogrid[
                        -dilation_px:dilation_px + 1,
                        -dilation_px:dilation_px + 1
                    ]
                    structure = (y_idx ** 2 + x_idx ** 2 <= dilation_px ** 2)

                    ibtracs_spot_mask = np.zeros((H, W), dtype=np.float32)
                    n_injected = 0
                    n_skipped = 0
                    for (slat, slon, wind_ms, sshs) in storm_points:
                        row = int(np.argmin(np.abs(self.lats - slat)))
                        col = int(np.argmin(np.abs(longitudes - slon)))

                        # Check if physics detection already covers this point
                        r0 = max(0, row - check_radius_px)
                        r1 = min(H, row + check_radius_px + 1)
                        c0 = max(0, col - check_radius_px)
                        c1 = min(W, col + check_radius_px + 1)
                        if enhanced_strict_mask[r0:r1, c0:c1].sum() > 0:
                            n_skipped += 1
                            continue  # Already detected — no injection needed

                        ibtracs_spot_mask[row, col] = 1.0
                        n_injected += 1

                    if ibtracs_spot_mask.sum() > 0:
                        from scipy.ndimage import binary_dilation
                        ibtracs_dilated = binary_dilation(
                            ibtracs_spot_mask > 0, structure=structure
                        ).astype(np.float32)
                        ibtracs_dilated *= lat_mask_2d

                        # Inject ONLY into strict mask (the one used for detection output)
                        enhanced_strict_mask = np.maximum(enhanced_strict_mask, ibtracs_dilated)

            except Exception as e:
                print(f"  Warning: IBTrACS injection failed for idx={idx}: {e}")

        return {
            'low_pressure'   : hurricane_mslp_filtered,
            'high_vorticity' : high_vorticity_mask,
            'high_wind_speed': high_wind_speed_mask,
            'strict'         : enhanced_strict_mask,
        }

    def _create_etc_masks(self, idx: int, lat_mask_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create FOCUSED ETC masks with enhanced track continuity
        """
        
        # Step 1: Smooth MSLP field with LESS smoothing for better feature preservation
        mslp_smoothed = self._smooth_mslp(self.mslp[idx], sigma_km=150.0)  # Reduced from 200
        
        # Step 2: Detect MSLP local minima with RELAXED criteria
        mslp_centers_mask = self._detect_mslp_local_minima(
            mslp_smoothed, 
            pressure_diff_threshold=100.0,  # Reduced from 150 Pa (1.0 hPa instead of 1.5)
            ring_radius_km=400.0  # Reduced from 500 km for tighter centers
        )
        
        # Step 3: HEMISPHERE-SPECIFIC dilation for regional coverage
        lat_2d = np.broadcast_to(self.lats[:, np.newaxis], mslp_centers_mask.shape)

        # NH: standard dilation
        nh_dilation = self._dilate_mslp_centers(
            mslp_centers_mask * (lat_2d >= 0), 
            dilation_km=600.0
        )

        # SH: larger dilation (SH storms can be more spread out)
        sh_dilation = self._dilate_mslp_centers(
            mslp_centers_mask * (lat_2d < 0), 
            dilation_km=700.0  # Larger for SH
        )

        etc_mslp_region_mask = np.maximum(nh_dilation, sh_dilation)
        
        # Step 4: RELAXED and ADAPTIVE pressure threshold
        # Calculate hemisphere-aware threshold
        lat_2d = np.broadcast_to(self.lats[:, np.newaxis], self.mslp[idx].shape)

        # NH: can use 1013 hPa (weaker storms common)
        # SH: use 1015 hPa (SH storms can be less deep)
        nh_pressure_mask = (lat_2d >= 0) & (self.mslp[idx] <= 101000)  # 1013 hPa
        sh_pressure_mask = (lat_2d < 0) & (self.mslp[idx] <= 101500)  # 1015 hPa

        etc_low_pressure_mask = (nh_pressure_mask | sh_pressure_mask).astype(np.float32)

        
        # Combine: Keep only low pressure regions that contain identified centers
        etc_mslp_filtered = etc_low_pressure_mask * etc_mslp_region_mask
    
        # Step 5: HEMISPHERE-AWARE Vorticity mask (850mb)
        if self.vor850_threshold is not None:
            if len(self.vor850_threshold.shape) == 2:
                vor850_thresh = self.vor850_threshold
            else:
                vor850_thresh = np.broadcast_to(self.vor850_threshold, self.vorticity[idx].shape)
            
            # Create latitude array for hemisphere detection
            lat_2d = np.broadcast_to(self.lats[:, np.newaxis], self.vorticity[idx].shape)
            
            # NH: positive vorticity, SH: negative vorticity (absolute value comparison)
            nh_mask = (lat_2d >= 0) & (self.vorticity[idx] >= vor850_thresh)
            sh_mask = (lat_2d < 0) & (self.vorticity[idx] <= -vor850_thresh)  # Negative threshold for SH!
            
            etc_vorticity_mask = (nh_mask | sh_mask).astype(np.float32)
        else:
            lat_2d = np.broadcast_to(self.lats[:, np.newaxis], self.vorticity[idx].shape)
            
            if isinstance(self.vort_threshold, np.ndarray):
                nh_mask = (lat_2d >= 0) & (self.vorticity[idx] >= self.vort_threshold[idx])
                sh_mask = (lat_2d < 0) & (self.vorticity[idx] <= -self.vort_threshold[idx])
            else:
                nh_mask = (lat_2d >= 0) & (self.vorticity[idx] >= self.vorticity_etc_threshold)
                sh_mask = (lat_2d < 0) & (self.vorticity[idx] <= -self.vorticity_etc_threshold)
            
            etc_vorticity_mask = (nh_mask | sh_mask).astype(np.float32)

        # Step 6: Wind speed mask - keep existing logic
        if self.windspeed_threshold is not None:
            wind_thresh = self.windspeed_threshold
            etc_wind_speed_mask = (self.wind_speed[idx] >= wind_thresh).astype(np.float32)
        else:
            etc_wind_speed_mask = (self.wind_speed[idx] >= self.wind_speed_etc_threshold).astype(np.float32)
    
        # Apply geographical constraints
        etc_mslp_filtered *= lat_mask_2d
        etc_vorticity_mask *= lat_mask_2d
        etc_wind_speed_mask *= lat_mask_2d
        

        # High-confidence: MSLP center + vorticity + wind speed
        high_confidence = (
            etc_mslp_filtered * 
            etc_vorticity_mask * 
            etc_wind_speed_mask
        ).astype(np.float32)

        # Moderate-confidence: Require actual MSLP CENTERS (not just region) + (vorticity OR wind speed)
        # This is MUCH stricter than before
        moderate_confidence = (
            mslp_centers_mask *  # Changed from etc_mslp_filtered to mslp_centers_mask
            etc_low_pressure_mask *  # Also require low pressure
            np.maximum(etc_vorticity_mask, etc_wind_speed_mask)
        ).astype(np.float32)

        # Dilate moderate confidence slightly to capture full storm extent
        moderate_confidence_dilated = self._dilate_mslp_centers(
            moderate_confidence,
            dilation_km=300.0  # Smaller dilation for stricter control
        )

        # Combine
        etc_adaptive_mask = np.maximum(high_confidence, moderate_confidence_dilated)

        return {
            'etc_low_pressure': etc_low_pressure_mask,
            'etc_mslp_centers': mslp_centers_mask,
            'etc_mslp_filtered': etc_mslp_filtered,
            'etc_vorticity': etc_vorticity_mask,
            'etc_wind_speed': etc_wind_speed_mask,
            'etc_enhanced': etc_adaptive_mask,
        }
    
    def _compute_local_bt_anomaly(
        self,
        bt_field: np.ndarray,
        background_radius_px: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute BT anomaly relative to a local background.

        A true convective system is anomalously cold compared to its
        surroundings. Arctic/winter cold surfaces are uniformly cold —
        their anomaly is near zero even if absolute BT is low.

        Args:
            bt_field            : 2-D BT array (K).
            background_radius_px: Radius for local background estimate.
                                ERA5 0.25° → 15 px ≈ 420 km background
                                ERA5 0.50° → 8  px ≈ 440 km background

        Returns:
            bt_anomaly   : bt_field - local_background  (negative = colder)
            bt_background: smoothed local background field
        """
        from scipy.ndimage import uniform_filter
        window = 2 * background_radius_px + 1
        bt_background = uniform_filter(bt_field.astype(np.float32), size=window, mode='nearest')
        bt_anomaly    = bt_field - bt_background   # negative means colder than surroundings
        return bt_anomaly, bt_background

    def _detect_mcs_blobs(
        self,
        bt_field: np.ndarray,
        boundary_threshold: float,      # e.g. 241 K  (absolute)
        core_threshold: float,          # e.g. 226 K  (absolute)
        min_core_pixels: int = 3,
        min_blob_area: int = 50,
        min_enclosure_distance: int = 2,
        # --- new anomaly parameters ---
        anomaly_shield_threshold: float = -8.0,   # shield pixel must be >=8K colder than bg
        anomaly_core_threshold: float   = -15.0,  # core pixel must be >=15K colder than bg
        background_radius_px: int = 15,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect MCS blobs using shield-core enclosure + local anomaly filtering.

        The anomaly filter is the critical addition:
            shield pixels must satisfy BOTH:
                bt <= boundary_threshold  (absolute, e.g. 241K)
                bt_anomaly <= anomaly_shield_threshold  (e.g. -8K vs local bg)
            core pixels must satisfy BOTH:
                bt <= core_threshold  (absolute, e.g. 226K)
                bt_anomaly <= anomaly_core_threshold  (e.g. -15K vs local bg)

        This eliminates large uniform cold sheets (anomaly ≈ 0) while
        preserving genuine convective systems (strongly negative anomaly).
        """
        from scipy.ndimage import label, distance_transform_edt

        conn8 = np.ones((3, 3), dtype=np.int32)

        # ------------------------------------------------------------------
        # STEP 0 – compute local BT anomaly  ← THE KEY FIX
        # ------------------------------------------------------------------
        bt_anomaly, bt_background = self._compute_local_bt_anomaly(
            bt_field, background_radius_px=background_radius_px
        )

        # ------------------------------------------------------------------
        # STEP 1 – combined absolute + anomaly masks
        # ------------------------------------------------------------------
        # Shield: must be absolutely cold AND anomalously cold vs surroundings
        shield_mask = (bt_field   <= boundary_threshold) & \
                    (bt_anomaly <= anomaly_shield_threshold)

        # Core: must be absolutely cold AND strongly anomalous
        core_mask   = (bt_field   <= core_threshold) & \
                    (bt_anomaly <= anomaly_core_threshold)

        return self._blobs_from_shield_core(
            shield_mask, core_mask, bt_field,
            min_core_pixels=min_core_pixels,
            min_blob_area=min_blob_area,
            min_enclosure_distance=min_enclosure_distance,
        )

    def _blobs_from_shield_core(
        self,
        shield_mask: np.ndarray,
        core_mask: np.ndarray,
        intensity_field: np.ndarray,
        min_core_pixels: int = 3,
        min_blob_area: int = 50,
        min_enclosure_distance: int = 2,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Shield/core blob assembly — shared by the BT and precipitation branches.

        Everything here is threshold-agnostic: it takes two boolean masks and
        knows nothing about what field produced them or which sign convention
        was used ("cold = event" for BT, "wet = event" for precip). Only the
        mask construction differs between branches, so only that is duplicated.

        Steps:
          2. label both masks (8-connected)
          3. match each core blob → its best-overlapping shield blob
          4-6. per shield: reject if smaller than `min_blob_area`; reject cores
               smaller than `min_core_pixels` or sitting on the shield edge
               (`min_enclosure_distance` via EDT); Voronoi-split multi-core
               shields and re-check each piece.

        `intensity_field` is only used for the per-system diagnostics.
        """
        from scipy.ndimage import label, distance_transform_edt

        conn8 = np.ones((3, 3), dtype=np.int32)

        # ------------------------------------------------------------------
        # STEP 2 – label connected components
        # ------------------------------------------------------------------
        labeled_shield, num_shield = label(shield_mask.astype(np.int32), structure=conn8)
        labeled_core,   num_cores  = label(core_mask.astype(np.int32),   structure=conn8)

        if num_shield == 0 or num_cores == 0:
            return np.zeros_like(intensity_field, dtype=np.float32), {}

        # ------------------------------------------------------------------
        # STEP 3 – match each C-blob → best-overlapping S-blob
        # ------------------------------------------------------------------
        both_active = (labeled_core > 0) & (labeled_shield > 0)
        c_flat = labeled_core[both_active]
        s_flat = labeled_shield[both_active]

        core_to_shield: Dict[int, int] = {}
        if c_flat.size > 0:
            encoded = c_flat.astype(np.int64) * (num_shield + 1) + s_flat.astype(np.int64)
            unique_pairs, counts = np.unique(encoded, return_counts=True)

            best: Dict[int, Tuple[int, int]] = {}
            for enc, cnt in zip(unique_pairs, counts):
                c_id = int(enc // (num_shield + 1))
                s_id = int(enc %  (num_shield + 1))
                if c_id not in best or cnt > best[c_id][1]:
                    best[c_id] = (s_id, int(cnt))

            core_to_shield = {c: v[0] for c, v in best.items()}

        if not core_to_shield:
            return np.zeros_like(intensity_field, dtype=np.float32), {}

        shield_to_cores: Dict[int, list] = {}
        for c_id, s_id in core_to_shield.items():
            shield_to_cores.setdefault(s_id, []).append(c_id)

        # ------------------------------------------------------------------
        # STEPS 4 + 5 + 6 – validate, filter, split
        # ------------------------------------------------------------------
        valid_blob_mask = np.zeros_like(intensity_field, dtype=np.float32)
        blob_properties: Dict[int, dict] = {}
        system_id = 0

        for s_id, candidate_core_ids in shield_to_cores.items():

            shield_blob_mask = labeled_shield == s_id
            shield_area      = int(shield_blob_mask.sum())

            if shield_area < min_blob_area:
                continue

            # EDT only within THIS shield blob
            blob_interior_dist = distance_transform_edt(shield_blob_mask)

            valid_cores = []
            for c_id in candidate_core_ids:
                c_blob_mask = labeled_core == c_id
                core_area   = int(c_blob_mask.sum())

                if core_area < min_core_pixels:
                    continue

                # Enclosure: core must sit inside the shield, not on its edge
                core_interior_depths = blob_interior_dist[c_blob_mask]
                if core_interior_depths.max() < min_enclosure_distance:
                    continue

                valid_cores.append((c_id, c_blob_mask, core_area))

            if not valid_cores:
                continue

            if len(valid_cores) == 1:
                system_id += 1
                self._accept_system(
                    system_id=system_id,
                    shield_mask=shield_blob_mask,
                    core_mask=valid_cores[0][1],
                    core_area=valid_cores[0][2],
                    intensity_field=intensity_field,
                    valid_blob_mask=valid_blob_mask,
                    blob_properties=blob_properties,
                    num_original_cores=1,
                )

            else:
                # Voronoi split for multi-core shields
                rows, cols   = np.where(shield_blob_mask)
                nearest_k    = np.zeros(len(rows), dtype=np.int32)
                min_dists    = np.full(len(rows), np.inf, dtype=np.float32)

                for k, (_, c_mask, _) in enumerate(valid_cores):
                    dist_to_core = distance_transform_edt(~c_mask).astype(np.float32)
                    d = dist_to_core[rows, cols]
                    closer = d < min_dists
                    min_dists[closer] = d[closer]
                    nearest_k[closer] = k

                for k, (c_id, c_blob_mask, core_area) in enumerate(valid_cores):
                    member = nearest_k == k
                    if not member.any():
                        continue

                    sub_shield_mask = np.zeros_like(shield_blob_mask)
                    sub_shield_mask[rows[member], cols[member]] = True
                    sub_shield_area = int(sub_shield_mask.sum())

                    if sub_shield_area < min_blob_area:
                        continue

                    sub_interior_dist = distance_transform_edt(sub_shield_mask)
                    if sub_interior_dist[c_blob_mask].max() < min_enclosure_distance:
                        continue

                    system_id += 1
                    self._accept_system(
                        system_id=system_id,
                        shield_mask=sub_shield_mask,
                        core_mask=c_blob_mask,
                        core_area=core_area,
                        intensity_field=intensity_field,
                        valid_blob_mask=valid_blob_mask,
                        blob_properties=blob_properties,
                        num_original_cores=len(valid_cores),
                    )

        return valid_blob_mask, blob_properties

    @staticmethod
    def _convective_fraction(prc_field: np.ndarray, pr_field: np.ndarray) -> np.ndarray:
        """Convective share of total precipitation, in [0, 1].

        Zero where it is essentially not raining, so that dividing noise by noise
        cannot manufacture a "fully convective" dry pixel. Clipped because pr and
        prc are remapped independently and rounding can put prc a hair above pr.
        """
        cf = np.where(pr_field > 1e-3,
                      prc_field / np.maximum(pr_field, 1e-6),
                      0.0)
        return np.clip(cf, 0.0, 1.0).astype(np.float32)

    def _detect_mcs_blobs_precip(
        self,
        prc_field: np.ndarray,          # convective precip, mm/h
        pr_field: np.ndarray,           # total precip, mm/h
        shield_threshold: float,        # e.g. 0.1 mm/h  (scalar or 2-D)
        core_threshold: float,          # e.g. 1.0 mm/h  (scalar or 2-D)
        convfrac_threshold: float,      # e.g. 0.7       (scalar or 2-D)
        min_core_pixels: int = 3,
        min_blob_area: int = 50,
        min_enclosure_distance: int = 2,
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Detect MCS blobs from convective precipitation.

        The precipitation analogue of `_detect_mcs_blobs`, with the comparisons
        inverted (`>=` — wet is the event, where for BT cold was the event):

            shield: prc >= shield_threshold AND convective fraction >= convfrac_threshold
            core:   prc >= core_threshold

        Two deliberate differences from the BT version:

        * **No local-anomaly filter.** BT needed one (`bt_anomaly <= -8/-15 K`) to
          kill large uniform cold sheets, because BT has no natural floor and a
          big slowly-varying background. Precipitation's background is ~0 over
          most of the domain, so a uniform_filter background is ~0 and the
          anomaly degenerates to a rescaled copy of the field itself — the test
          would add nothing while costing a 31x31 convolution per timestep.

        * **A convective-fraction gate instead.** This is the load-bearing part:
          it is what makes this an MCS detector rather than a rain detector.
          RegCM's PNW winter rain is stratiform orographic upslope (convective
          fraction ~40-48%) while Great Plains summer convection is ~91-97%, so
          the gate separates the two on physics rather than on geography. It is
          exactly the discriminator the cloud-top proxy cannot express.

        The gate is applied to the shield only — a cell already raining at
        >= 1 mm/h of *convective* precip is convective by construction, so
        gating the core too would be redundant and would only cost recall.

        Returns (valid_blob_mask, blob_properties, convective_fraction); the
        caller reuses the fraction for the boundary-only channel.
        """
        convfrac = self._convective_fraction(prc_field, pr_field)

        shield_mask = (prc_field >= shield_threshold) & (convfrac >= convfrac_threshold)
        core_mask   = (prc_field >= core_threshold)

        blob_mask, blob_properties = self._blobs_from_shield_core(
            shield_mask, core_mask, prc_field,
            min_core_pixels=min_core_pixels,
            min_blob_area=min_blob_area,
            min_enclosure_distance=min_enclosure_distance,
        )
        return blob_mask, blob_properties, convfrac

    def _accept_system(
        self,
        system_id: int,
        shield_mask: np.ndarray,
        core_mask: np.ndarray,
        core_area: int,
        intensity_field: np.ndarray,
        valid_blob_mask: np.ndarray,
        blob_properties: dict,
        num_original_cores: int,
    ) -> None:
        """Record one accepted system. `intensity_field` is BT (K) for the legacy
        branch and convective precip (mm/h) for the precip branch — so the stats
        are named neutrally rather than as temperatures. These diagnostics are
        currently discarded by _create_mcs_masks; they exist for debugging."""
        valid_blob_mask[shield_mask] = 1.0
        shield_area = int(shield_mask.sum())
        vals        = intensity_field[shield_mask]
        blob_properties[system_id] = {
            'shield_area'   : shield_area,
            'core_area'     : core_area,
            'min_value'     : float(vals.min()),
            'max_value'     : float(vals.max()),
            'mean_value'    : float(vals.mean()),
            'num_cores'     : num_original_cores,
            'core_fraction' : core_area / shield_area if shield_area > 0 else 0.0,
        }

    def _calculate_moisture_score(self, iwv_field: np.ndarray) -> np.ndarray:
        """
        Calculate moisture favorability score for MCS development based on IWV.
        
        Args:
            iwv_field: Integrated Water Vapor field in mm
        
        Returns:
            Moisture score array (0.0 to 1.0) indicating favorability
        
        Score ranges:
            1.0: Highly favorable (IWV ≥ 50 mm)
            0.5-1.0: Favorable (IWV 40-50 mm)
            0.0-0.5: Marginal (IWV 30-40 mm)
            ~0.0: Unfavorable (IWV < 30 mm)
        """
        moisture_score = np.zeros_like(iwv_field, dtype=np.float32)
        
        # Highly favorable: IWV ≥ 50 mm
        highly_favorable = iwv_field >= 50.0
        moisture_score[highly_favorable] = 1.0
        
        # Favorable: 40 ≤ IWV < 50 mm (linear from 0.5 to 1.0)
        favorable = (iwv_field >= 40.0) & (iwv_field < 50.0)
        moisture_score[favorable] = (iwv_field[favorable] - 40.0) / 10.0 + 0.5
        
        # Marginal: 30 ≤ IWV < 40 mm (linear from 0.0 to 0.5)
        marginal = (iwv_field >= 30.0) & (iwv_field < 40.0)
        moisture_score[marginal] = (iwv_field[marginal] - 30.0) / 10.0
        
        # Unfavorable: IWV < 30 mm (linear from 0.0, floor at 20 mm)
        unfavorable = iwv_field < 30.0
        moisture_score[unfavorable] = np.maximum((iwv_field[unfavorable] - 20.0) / 10.0, 0.0)
        
        return moisture_score

    def _calculate_shear_score(self, shear_field: np.ndarray) -> np.ndarray:
        """
        Calculate wind shear favorability score for MCS development.
        
        Args:
            shear_field: Wind shear magnitude (850-200mb) in m/s
        
        Returns:
            Shear score array (0.0 to 1.0) indicating favorability
        
        Score ranges:
            1.0: Optimal (8-12 m/s) - perfect for organized convection
            0.67-1.0: Favorable (5-8 m/s or 12-20 m/s) - acceptable
            0.2: Unfavorable (< 5 m/s or > 20 m/s) - poor conditions
        
        Physical reasoning:
            • 5-15 m/s: Tilts updraft, separates updraft/downdraft
            • < 5 m/s: System collapses on itself
            • > 20 m/s: Excessive tilt, loses organization
        """
        shear_score = np.zeros_like(shear_field, dtype=np.float32)
        
        # OPTIMAL: 8-12 m/s (perfect for MCS organization)
        optimal = (shear_field >= 8.0) & (shear_field <= 12.0)
        shear_score[optimal] = 1.0
        
        # FAVORABLE (LOW SIDE): 5-8 m/s (linear from 0.0 to 1.0)
        favorable_low = (shear_field >= 5.0) & (shear_field < 8.0)
        shear_score[favorable_low] = (shear_field[favorable_low] - 5.0) / 3.0
        
        # FAVORABLE (HIGH SIDE): 12-20 m/s (linear from 1.0 to 0.0)
        favorable_high = (shear_field > 12.0) & (shear_field <= 20.0)
        shear_score[favorable_high] = (20.0 - shear_field[favorable_high]) / 8.0
        
        # UNFAVORABLE: < 5 m/s or > 20 m/s (penalize but don't eliminate)
        unfavorable = (shear_field < 5.0) | (shear_field > 20.0)
        shear_score[unfavorable] = 0.2
        
        return shear_score

    def _get_seasonal_mcs_parameters(self, time_index: int) -> Dict[str, float]:
        """
        Get season-adaptive parameters for MCS detection.
        
        Summer MCS (JJA in NH, DJF in SH):
            - Smaller, more intense systems
            - Lower area thresholds
            - Stricter intensity requirements
        
        Winter MCS (DJF in NH, JJA in SH):
            - Larger, more organized systems
            - Higher area thresholds
            - More relaxed intensity requirements
        
        Args:
            time_index: Index into the time array
        
        Returns:
            Dict with seasonal parameters
        """
        # Get month from timestamp
        timestamp = self.time[time_index]
        month = pd.Timestamp(timestamp).month
        
        # Get latitude to determine hemisphere
        # Use mean latitude of valid MCS regions
        mean_lat = np.mean(self.latitudes[self.mcs_lat_mask])
        
        # Determine if it's summer or winter based on hemisphere
        if mean_lat >= 0:  # Northern Hemisphere
            summer_months = [6, 7, 8]  # JJA
            winter_months = [12, 1, 2]  # DJF
        else:  # Southern Hemisphere
            summer_months = [12, 1, 2]  # DJF
            winter_months = [6, 7, 8]   # JJA
        
        # Transition months
        spring_fall_months = [3, 4, 5, 9, 10, 11]
        
        # Set seasonal parameters
        # NOTE: moisture_threshold is the LEGACY absolute-IWV gate (used only when
        # no IWV climatology is supplied). moisture_anom_threshold is the IWV-anomaly
        # gate (mm above the monthly climatology) used by Option 1; tunable per season.
        if month in summer_months:
            # SUMMER: Smaller, intense systems
            params = {
                'min_blob_area': 4,           # Small systems (4 pixels ≈ 100x100 km)
                'min_core_pixels': 3,         # Compact intense cores
                'bt_boundary': 241.0,         # Standard boundary
                'bt_core': 223.0,             # STRICTER core (was 225K)
                'pr_shield': 0.1,             # mm/h convective (precip branch)
                'pr_core': 1.0,               # mm/h convective — standard core rate
                'pr_convfrac': 0.7,           # prc/pr — separates convection from upslope
                'moisture_threshold': 0.5,    # legacy absolute gate
                'moisture_anom_threshold': 2.0,   # require above-normal moisture in summer
                'shear_threshold': 0.6,       # Favorable shear needed
                'season': 'summer'
            }
        elif month in winter_months:
            # WINTER: Larger, organized systems
            params = {
                'min_blob_area': 12,          # Larger systems (12 pixels ≈ 300x300 km)
                'min_core_pixels': 8,         # Larger cores
                'bt_boundary': 241.0,         # Standard boundary
                'bt_core': 228.0,             # RELAXED core (was 225K)
                'pr_shield': 0.1,             # mm/h convective (precip branch)
                # NOT relaxed (cf. bt_core 223→228 in winter). Cold-season MCS do
                # have weaker updrafts, so a relaxation to ~0.5 is plausible — but
                # the validated diagnostic used 1.0 in every season, so keep 1.0
                # here and test any relaxation as an isolated, measured delta.
                'pr_core': 1.0,
                'pr_convfrac': 0.7,
                'moisture_threshold': 0.3,    # legacy absolute gate
                'moisture_anom_threshold': -2.0,  # allow near/slightly-below-normal (cold, dry season)
                'shear_threshold': 0.4,       # More lenient shear
                'season': 'winter'
            }
        else:
            # TRANSITION: Intermediate parameters
            params = {
                'min_blob_area': 6,           # Medium systems
                'min_core_pixels': 5,         # Medium cores
                'bt_boundary': 241.0,         # Standard boundary
                'bt_core': 225.0,             # Standard core
                'pr_shield': 0.1,             # mm/h convective (precip branch)
                'pr_core': 1.0,               # mm/h convective
                'pr_convfrac': 0.7,
                'moisture_threshold': 0.4,    # legacy absolute gate
                'moisture_anom_threshold': 0.0,   # above-normal moisture
                'shear_threshold': 0.5,       # Moderate shear
                'season': 'transition'
            }
        
        return params

    def _create_mcs_masks(self, idx: int, lat_mask_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create MCS detection masks with SEASONAL ADAPTIVITY.

        Two sources, selected by `self.mcs_source`:
          "precip" (default) — convective-precip shield/core. RegCM's own
              convection scheme is the MCS proxy; seasonality is correct
              (Great Plains DJF ~0.2 vs JJA ~12.3 MCS days).
          "bt" (legacy)      — cloud-top-temperature proxy, kept for A/B only.
              Its Great Plains cold signal is winter frontal cirrus and bottoms
              out in July, i.e. anti-correlated with the MCS season.

        Returns the same three keys either way — 'mcs_boundary' / 'mcs_blobs' /
        'mcs_core' → input channels 11 / 12 / 13 — so nothing downstream
        (the channel stack, loss.py) changes with the source.

        Method:
        1. Determine season and get adaptive parameters
        2. Boundary/core thresholds adapt to season
        3. Size requirements vary by season
        4. Moisture and shear requirements adjust
        """

        # ============================
        # 1. GET SEASONAL PARAMETERS
        # ============================
        seasonal_params = self._get_seasonal_mcs_parameters(idx)

        # Extract seasonal thresholds
        bt_boundary_seasonal = seasonal_params['bt_boundary']
        bt_core_seasonal = seasonal_params['bt_core']
        min_blob_area_seasonal = seasonal_params['min_blob_area']
        min_core_pixels_seasonal = seasonal_params['min_core_pixels']
        moisture_thresh_seasonal = seasonal_params['moisture_threshold']
        shear_thresh_seasonal = seasonal_params['shear_threshold']

        # Override with global thresholds if provided (boundary/core only)
        if self.bt_boundary_threshold is not None:
            bt_boundary = self.bt_boundary_threshold
        else:
            bt_boundary = bt_boundary_seasonal

        if self.bt_core_threshold is not None:
            bt_core = self.bt_core_threshold
        else:
            bt_core = bt_core_seasonal

        # Precip thresholds: global override (utils.load_regcm_pr_thresholds) wins,
        # else the seasonal defaults. Same precedence as BT above.
        pr_shield = (self.pr_shield_threshold if self.pr_shield_threshold is not None
                     else seasonal_params['pr_shield'])
        pr_core = (self.pr_core_threshold if self.pr_core_threshold is not None
                   else seasonal_params['pr_core'])
        pr_convfrac = (self.pr_convfrac_threshold if self.pr_convfrac_threshold is not None
                       else seasonal_params['pr_convfrac'])

        # ============================
        # 2. SEASONAL BLOB DETECTION
        # ============================
        convfrac = None
        if self.mcs_source == "precip":
            mcs_blob_mask, blob_properties, convfrac = self._detect_mcs_blobs_precip(
                self.prc[idx],
                self.pr[idx],
                shield_threshold=pr_shield,
                core_threshold=pr_core,
                convfrac_threshold=pr_convfrac,
                min_core_pixels=min_core_pixels_seasonal,  # SEASONAL
                min_blob_area=min_blob_area_seasonal       # SEASONAL
            )
        else:
            mcs_blob_mask, blob_properties = self._detect_mcs_blobs(
                self.bt_ds[idx],
                boundary_threshold=bt_boundary,
                core_threshold=bt_core,
                min_core_pixels=min_core_pixels_seasonal,  # SEASONAL
                min_blob_area=min_blob_area_seasonal       # SEASONAL
            )

        # ============================
        # 3. MOISTURE ASSESSMENT
        # ============================
        moisture_score = self._calculate_moisture_score(self.iwv[idx])

        # --- Seasonal moisture gate (Option 1: IWV anomaly) -----------------------
        # Absolute-IWV gating erases cool-season MCS (winter IWV << summer). Use the
        # IWV anomaly vs monthly climatology instead: "moist FOR THE SEASON".
        #   mode "anomaly"      -> hard gate  (anomaly >= threshold)
        #   mode "anomaly_soft" -> soft weight in [0,1] across a band around threshold
        #   mode "absolute" or no climatology -> legacy absolute-IWV behaviour
        use_anom = (self.iwv_clim is not None) and (self.mcs_moisture_mode in ("anomaly", "anomaly_soft"))
        if use_anom:
            iwv_anom = self._compute_iwv_anomaly(idx)
            anom_thr = seasonal_params.get('moisture_anom_threshold', self.iwv_anom_threshold)
            if self.mcs_moisture_mode == "anomaly_soft":
                lo = anom_thr - self.iwv_anom_soft_band
                hi = anom_thr + self.iwv_anom_soft_band
                mcs_seasonal_moisture = np.clip((iwv_anom - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
            else:
                mcs_seasonal_moisture = (iwv_anom >= anom_thr).astype(np.float32)
        else:
            # Legacy absolute-IWV gate
            mcs_seasonal_moisture = (moisture_score >= moisture_thresh_seasonal).astype(np.float32)
        
        # Also keep standard thresholds for reference
        mcs_high_moisture = (moisture_score >= 0.7).astype(np.float32)
        mcs_favorable_moisture = (moisture_score >= 0.5).astype(np.float32)
        mcs_marginal_moisture = (moisture_score >= 0.3).astype(np.float32)
        
        # ============================
        # 4. WIND SHEAR ASSESSMENT
        # ============================
        shear_score = self._calculate_shear_score(self.windhshear[idx])
        
        # Use seasonal shear threshold
        mcs_seasonal_shear = (shear_score >= shear_thresh_seasonal).astype(np.float32)
        
        # Also keep standard thresholds for reference
        mcs_optimal_shear = (shear_score >= 0.9).astype(np.float32)
        mcs_favorable_shear = (shear_score >= 0.6).astype(np.float32)
        mcs_acceptable_shear = (shear_score >= 0.4).astype(np.float32)
        
        # ============================
        # 5. APPLY LATITUDE MASK
        # ============================
        mcs_blob_mask *= lat_mask_2d
        mcs_seasonal_moisture *= lat_mask_2d
        mcs_seasonal_shear *= lat_mask_2d
        mcs_high_moisture *= lat_mask_2d
        mcs_favorable_moisture *= lat_mask_2d
        mcs_marginal_moisture *= lat_mask_2d
        mcs_optimal_shear *= lat_mask_2d
        mcs_favorable_shear *= lat_mask_2d
        mcs_acceptable_shear *= lat_mask_2d
        
        # ============================
        # 6. SEASONALLY-ADAPTIVE CORE MCS MASK
        # ============================
        if self.mcs_source == "precip":
            # NO IWV gate here, deliberately.
            #
            # `blobs * moisture` made sense for BT: a cold cloud top is not proof
            # of rain, so the IWV anomaly supplied the missing moisture evidence.
            # A precip blob needs no such inference — it ALREADY RAINED, which is
            # strictly stronger evidence than an IWV anomaly.
            #
            # Re-adding the gate would also re-import the cool-season suppression
            # that the winter `moisture_anom_threshold = -2.0` hack exists to work
            # around, and would corrupt the summer-peaked seasonality that is the
            # entire point of the precip branch.
            mcs_core_mask = mcs_blob_mask.astype(np.float32)
        else:
            # Use seasonal thresholds for primary detection
            mcs_core_seasonal = (
                mcs_blob_mask *
                mcs_seasonal_moisture
                # *
                # mcs_seasonal_shear
            ).astype(np.float32)

            # MODERATE: Fixed moderate criteria (for comparison)
            mcs_core_moderate = (
                mcs_blob_mask *
                mcs_marginal_moisture
            ).astype(np.float32)

            # FINAL CORE: Use seasonal adaptive mask
            mcs_core_mask = mcs_core_seasonal


        # ============================
        # 7. BOUNDARY-ONLY MASK
        # ============================
        # Threshold-only shield: no size or enclosure test, so it is always a
        # superset of mcs_blobs.
        if self.mcs_source == "precip":
            mcs_boundary_only = ((self.prc[idx] >= pr_shield) &
                                 (convfrac >= pr_convfrac)).astype(np.float32)
        else:
            mcs_boundary_only = (self.bt_ds[idx] <= bt_boundary).astype(np.float32)
        mcs_boundary_only *= lat_mask_2d

        return {
            'mcs_boundary': mcs_boundary_only,        # Threshold-only shield
            'mcs_blobs': mcs_blob_mask,               # Seasonally-sized blobs
            'mcs_core': mcs_core_mask,                # Training target (channel 13)
            # 'season': seasonal_params['season'],      # Season label (for debugging)
        }


    def _preload_tensors(self):
        """Preload commonly used tensors for faster access"""
        print("Preloading tensors for faster GPU access...")
        
        if torch.cuda.is_available():
            try:
                # Pre-normalize data (use torch.tensor fallback for numpy/torch build compat)
                def _safe_from_numpy(arr, dtype):
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    try:
                        return torch.from_numpy(arr).to(dtype=dtype)
                    except (TypeError, RuntimeError):
                        return torch.tensor(arr, dtype=dtype)
                
                self.ivt_normalized = _safe_from_numpy(
                    (self.ivt - self.ivt_mean) / self.ivt_std, self.dtype
                )
                
                self.mslp_normalized = _safe_from_numpy(
                    (self.mslp - self.mslp_mean) / self.mslp_std, self.dtype
                )
                
                # self.temp_500mb_normalized = torch.from_numpy(
                #     (self.temp_500mb - self.temp_500mb_mean) / self.temp_500mb_std
                # ).to(dtype=self.dtype)
                
                # self.windhshear_normalized = torch.from_numpy(
                #     (self.windhshear - self.windhshear_mean) / self.windhshear_std
                # ).to(dtype=self.dtype)
                
                # self.sst_normalized = torch.from_numpy(
                #     (self.sst - self.sst_mean) / self.sst_std
                # ).to(dtype=self.dtype)
                
                # self.bt_ds_normalized = torch.from_numpy(
                #     (self.bt_ds - self.bt_ds_mean) / self.bt_ds_std
                # ).to(dtype=self.dtype)

                # self.tpr_normalized = torch.from_numpy(
                #     (self.tpr - self.tpr_mean) / self.tpr_std
                # ).to(dtype=self.dtype)

                
                if self.pin_memory:
                    self.ivt_normalized = self.ivt_normalized.pin_memory()
                    self.mslp_normalized = self.mslp_normalized.pin_memory()
                    # self.temp_500mb_normalized = self.temp_500mb_normalized.pin_memory()
                    # self.windhshear_normalized = self.windhshear_normalized.pin_memory()
                    # self.sst_normalized = self.sst_normalized.pin_memory()
                    # self.bt_ds_normalized = self.bt_ds_normalized.pin_memory()
                    # self.tpr_normalized = self.tpr_normalized.pin_memory()
                
                print("Data preloaded successfully")
            except RuntimeError as e:
                print(f"Warning: Could not preload tensors due to memory constraints: {e}")
                self.preload_data = False
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a precomputed sample. All expensive scipy/numpy mask computations
        were done once during __init__ by _precompute_all_channels().
        
        Args:
            idx: Sample index
            
        Returns:
            Tensor with shape (14, height, width)
        """
        try:
            # Fast path: return precomputed data (zero CPU overhead)
            if hasattr(self, 'precomputed_data') and self.precomputed_data is not None:
                data = self.precomputed_data[idx]
                data = np.ascontiguousarray(data, dtype=np.float32)
                try:
                    tensor = torch.from_numpy(data).to(dtype=self.dtype)
                except (TypeError, RuntimeError):
                    tensor = torch.tensor(data, dtype=self.dtype)
                return tensor
            
            # Fallback: compute on the fly (should not happen after init)
            data = self._compute_single_sample(idx)
            data = np.ascontiguousarray(data, dtype=np.float32)
            try:
                tensor = torch.from_numpy(data).to(dtype=self.dtype)
            except (TypeError, RuntimeError):
                tensor = torch.tensor(data, dtype=self.dtype)
            return tensor
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a safe fallback tensor with the expected channel count
            return torch.zeros(14, *self.target_size, dtype=self.dtype)




class MemoryEfficientDataLoader(DataLoader):
    """
    Memory-efficient DataLoader with GPU optimization
    
    OPTIMIZATION: Conservative defaults to prevent OOM on multi-GPU systems:
    - num_workers: 2 (parallel loading, minimal memory overhead)
    - pin_memory: True (2-3x faster GPU transfer)
    - persistent_workers: True (efficient worker lifecycle)
    """
    def __init__(self, *args, **kwargs):
        # Allow caller to override, but provide safe defaults
        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = True  # OPTIMIZED: Fast GPU transfer
        
        if 'persistent_workers' not in kwargs and kwargs.get('num_workers', 0) > 0:
            kwargs['persistent_workers'] = True  # OPTIMIZED: Efficient workers
        
        if 'num_workers' not in kwargs:
            kwargs['num_workers'] = 2  # OPTIMIZED: Parallel loading (conservative)
        
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        """Iterator with memory management"""
        iterator = super().__iter__()
        for batch in iterator:
            # Yield batch
            yield batch
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
                  

def preprocess_data_improved(ivt: xr.DataArray, 
                           mslp: xr.DataArray, 
                           vorticity: Optional[xr.DataArray] = None, 
                        #    vorticity_500_mb: Optional[xr.DataArray] = None, 
                           wind_speed: Optional[xr.DataArray] = None, 
                           ivt_percentile: Optional[xr.DataArray] = None, 
                           temp_500mb: Optional[xr.DataArray] = None,
                           windhshear_ds: Optional[xr.DataArray] = None,
                        #    SST_ds: Optional[xr.DataArray] = None,
                           BT_ds: Optional[xr.DataArray] = None,
                           PR_ds: Optional[xr.Dataset] = None,
                           IWV: Optional[xr.DataArray] = None,
                           km_per_pixel: float = 1.0) -> xr.Dataset:

    """
    Enhanced preprocessing with GPU optimization
    
    Args:
        ivt: Integrated Vapor Transport data
        mslp: Mean Sea Level Pressure data
        vorticity: Vorticity at 850mb data
        # vorticity_500_mb: Vorticity at 500mb data
        wind_speed: Wind speed data
        ivt_percentile: IVT percentile threshold
        temp_500mb: warmcore t300-t500 temperature data
        windhshear_ds: windshear magnitude data 
        # SST_ds: SST data 
        BT_ds: Brightness Temp data
        PR_ds: Precipitation dataset with `pr` (total) and `prc` (convective),
               both in mm/h — the source for the MCS branch
        IWV: Integrated Water Vapor data
        vorticity_threshold: Vorticity threshold
        
        km_per_pixel: Spatial resolution
        
    Returns:
        Combined xarray Dataset
    """
    print("Starting GPU-optimized preprocessing...")
    
    # Optimize gradient computation
    def compute_gradient_magnitude(data):
        """Compute gradient magnitude with GPU optimization"""
        return xr.apply_ufunc(
            lambda x: np.sqrt(np.gradient(x, axis=0)**2 + np.gradient(x, axis=1)**2),
            data,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[["latitude", "longitude"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32]
        )
    
    # Helper function to clean coordinates
    def clean_coordinates(da):
        """Remove conflicting coordinates, keeping only time, latitude, longitude"""
        coords_to_keep = ['time', 'latitude', 'longitude']
        coords_to_drop = [coord for coord in da.coords if coord not in coords_to_keep]
        return da.drop_vars(coords_to_drop, errors='ignore')
    
    # Compute gradients with optimized functions
    print("Computing IVT gradient...")
    ivt_grad = compute_gradient_magnitude(ivt)
    ivt_grad = clean_coordinates(ivt_grad)
    
    # Compute vorticity gradient at 850mb
    if vorticity is not None:
        print("Computing vorticity gradient...")
        
        # Handle both Dataset and DataArray
        if hasattr(vorticity, 'data_vars'):
            # It's a Dataset - extract the vorticity variable
            if 'vor' in vorticity:
                vorticity = vorticity['vor']
            elif 'vorticity' in vorticity:
                vorticity = vorticity['vorticity']
            else:
                # Try to get the first data variable
                var_name = list(vorticity.data_vars)[0]
                vorticity = vorticity[var_name]
        
        vorticity_grad = compute_gradient_magnitude(vorticity)
        vorticity_grad = clean_coordinates(vorticity_grad)
        vorticity = clean_coordinates(vorticity)
    else:
        print("Creating dummy vorticity gradient...")
        vorticity_grad = xr.zeros_like(ivt)
        vorticity = xr.zeros_like(ivt)
    
    # # Compute vorticity gradient at 500mb
    # if vorticity_500_mb is not None:
    #     print("Computing vorticity_500_mb gradient...")
    #     vorticity_500_grad = compute_gradient_magnitude(vorticity_500_mb)
    #     vorticity_500_grad = clean_coordinates(vorticity_500_grad)
    #     vorticity_500_mb = clean_coordinates(vorticity_500_mb)
    # else:
    #     print("Creating dummy vorticity_500_mb gradient...")
    #     vorticity_500_grad = xr.zeros_like(ivt)
    #     vorticity_500_mb = xr.zeros_like(ivt)
    
    # Process wind speed
    if wind_speed is not None:
        print("Processing wind speed data...")
        
        # Handle both Dataset and DataArray
        if hasattr(wind_speed, 'data_vars'):
            # It's a Dataset - extract the wind speed variable
            if 'wspd' in wind_speed:
                wind_speed = wind_speed['wspd']
            elif 'wind_speed' in wind_speed:
                wind_speed = wind_speed['wind_speed']
            else:
                # Try to get the first data variable
                var_name = list(wind_speed.data_vars)[0]
                wind_speed = wind_speed[var_name]
        
        # Clean coordinates first
        wind_speed = clean_coordinates(wind_speed)
        
        # Apply threshold
        # wind_speed = wind_speed.where(wind_speed >= 15, 15)
        try:
            wind_min = float(wind_speed.min().values)
            wind_max = float(wind_speed.max().values)
            print(f"Wind speed processed - range: {wind_min:.2f} to {wind_max:.2f} m/s")
        except Exception as e:
            print(f"Wind speed processed (could not compute range: {str(e)})")
    else:
        print("Missing wind speed data...")

    # Process SST speed
    # if SST_ds is not None:
    #     print("Processing SST data...")
    #     # Clean coordinates first
    #     SST_ds = clean_coordinates(SST_ds)
    #     # Apply threshold
    #     SST_ds = SST_ds.where((SST_ds > 22) & (SST_ds < 32))
    #     print(f"SST processed - range: {float(SST_ds.min()):.2f} to {float(SST_ds.max()):.2f} C")
    # else:
    #     print("Missing SST data...")

    # BT data
    if BT_ds is not None:
        print("Processing BT data...")
        
        # Handle both Dataset and DataArray
        if hasattr(BT_ds, 'data_vars'):
            # It's a Dataset - extract the BT variable
            if 'BT' in BT_ds:
                BT_ds = BT_ds['BT']
            else:
                # Try to get the first data variable
                var_name = list(BT_ds.data_vars)[0]
                BT_ds = BT_ds[var_name]
        
        # Clean coordinates first
        BT_ds = clean_coordinates(BT_ds)
        # Align BT to the main IVT time axis. RegCM BT is stored in yearly files;
        # the per-month slice does NOT match the WNet month grid exactly (differing
        # count — e.g. Jan 121 vs 120 — and a 6 h offset). Nearest-time selection
        # yields exactly one BT scene per main timestep so the positional index
        # self.bt_ds[idx] stays aligned with ivt/mslp/etc.
        try:
            if 'time' in getattr(BT_ds, 'coords', {}) and 'time' in ivt.coords:
                _main_t = ivt['time'].values
                # sel(...nearest) keeps BT's original timestamps (which can repeat
                # → duplicate index → merge fails), so relabel to the main axis.
                BT_ds = (BT_ds.sel(time=_main_t, method='nearest')
                              .assign_coords(time=_main_t))
        except Exception as e:
            print(f"BT time-alignment skipped: {e}")
        # RegCM BT is a "diagnosed cloud-top temperature proxy" that is NaN over
        # clear sky (no cloud top → ~45% of pixels). The downstream
        # nan_to_num(nan=0) would turn clear sky into 0 K — colder than the 225 K
        # MCS core threshold — flagging the whole clear-sky domain as deep
        # convection. Fill clear sky with a warm constant well above the 241/225 K
        # thresholds so it is never counted as an MCS. (ERA5 BT had no NaNs.)
        BT_CLEARSKY_FILL_K = 290.0
        BT_ds = BT_ds.fillna(BT_CLEARSKY_FILL_K)
        try:
            bt_min = float(BT_ds.min().values)
            bt_max = float(BT_ds.max().values)
            print(f"BT processed - range: {bt_min:.2f} to {bt_max:.2f} K "
                  f"(clear-sky NaN → {BT_CLEARSKY_FILL_K:.0f} K)")
        except Exception as e:
            print(f"BT processed (could not compute range: {str(e)})")
    else:
        print("Missing BT data...")

    # ── Precipitation (pr / prc, mm/h) — the MCS source ───────────────────────
    # Mirrors the BT block above: same time-alignment idiom, but the clear-sky
    # fill inverts. BT needed a WARM fill (290 K) because NaN→0 would look like
    # deep convection; precipitation's "nothing happening" value IS 0.0, which is
    # already below every shield threshold, so NaN→0 is the safe direction.
    pr = prc = None
    if PR_ds is not None:
        print("Processing precipitation data (pr, prc)...")

        def _extract_precip(src, names):
            for n in names:
                if hasattr(src, 'data_vars') and n in src:
                    return src[n]
            if hasattr(src, 'data_vars'):
                return src[list(src.data_vars)[0]]
            return src

        pr  = clean_coordinates(_extract_precip(PR_ds, ['pr', 'PR']))
        prc = clean_coordinates(_extract_precip(PR_ds, ['prc', 'PRC']))

        # Align to the main IVT time axis, exactly as BT does. The regrid step
        # already relabels precip to the right edge of its time_bnds (06:00/12:00
        # /...), so this should be an exact match; nearest + relabel keeps it
        # robust and prevents duplicate timestamps from breaking the merge.
        try:
            if 'time' in ivt.coords:
                _main_t = ivt['time'].values
                for _name, _da in (('pr', pr), ('prc', prc)):
                    if _da is not None and 'time' in getattr(_da, 'coords', {}):
                        _aligned = (_da.sel(time=_main_t, method='nearest')
                                       .assign_coords(time=_main_t))
                        if _name == 'pr':
                            pr = _aligned
                        else:
                            prc = _aligned
        except Exception as e:
            print(f"Precip time-alignment skipped: {e}")

        pr  = pr.fillna(0.0)
        prc = prc.fillna(0.0)

        # Runtime tripwire for a units regression. Bounds are deliberately WIDE:
        # this is a last-resort backstop, and the real structural check lives in
        # verify_pr_regrid.py (regridded-vs-native max ratio, which catches a
        # *_6hourly-sum leak as ~2.0 and a missing x3600 as ~1/3600).
        #   < 0.5   -> the x3600 in Run_PR_regrid.sh was skipped (still kg m-2 s-1)
        #   > 250   -> a *_6hourly-sum file leaked in (it is exactly 2x the rate)
        # 100 mm/h was too tight and cried wolf: single-cell 6-h means of
        # 100-155 mm/h are real convective extremes here (CNR 1992 peaks at 152.8
        # mm/h natively, regridded to 140.8 — ratio 0.921, i.e. correctly sourced
        # and bilinearly smoothed, not a leak).
        try:
            pr_max, prc_max = float(pr.max().values), float(prc.max().values)
            print(f"Precip processed - pr max {pr_max:.2f} mm/h, "
                  f"prc max {prc_max:.2f} mm/h")
            if pr_max < 0.5:
                print(f"  ⚠ pr max {pr_max:.3e} is suspiciously small — "
                      f"is it still kg m-2 s-1? (expected mm/h)")
            if pr_max > 250.0:
                print(f"  ⚠ pr max {pr_max:.1f} mm/h exceeds any physical 6-h mean — "
                      f"did a *_6hourly-sum file leak in?")
        except Exception as e:
            print(f"Precip processed (could not compute range: {str(e)})")
    else:
        print("Missing precipitation data — MCS channels will be empty!")


    # Process 500mb temperature data
    if temp_500mb is not None:
        print("Processing 500mb temperature data...")
        
        # Handle both Dataset and DataArray
        if hasattr(temp_500mb, 'data_vars'):
            # It's a Dataset - extract the temperature variable
            if 't' in temp_500mb:
                temp_500mb = temp_500mb['t']
            elif 'T' in temp_500mb:
                temp_500mb = temp_500mb['T']
            elif 'temperature' in temp_500mb:
                temp_500mb = temp_500mb['temperature']
            else:
                # Try to get the first data variable
                var_name = list(temp_500mb.data_vars)[0]
                temp_500mb = temp_500mb[var_name]
        
        # Handle coordinate name differences
        if 'lon' in temp_500mb.dims and 'longitude' in ivt.dims:
            temp_500mb = temp_500mb.rename({'lon': 'longitude', 'lat': 'latitude'})
        
        # Clean coordinates
        temp_500mb = clean_coordinates(temp_500mb)
        try:
            temp_min = float(temp_500mb.min().values)
            temp_max = float(temp_500mb.max().values)
            print(f"Temperature processed - range: {temp_min:.2f} to {temp_max:.2f} K")
        except Exception as e:
            print(f"Temperature processed (could not compute range: {str(e)})")
    else:
        print("Missing temperature data...")


    # Process windhshear_ds data
    if windhshear_ds is not None:
        print("Processing windhshear data...")
        
        # Handle both Dataset and DataArray
        if hasattr(windhshear_ds, 'data_vars'):
            # It's a Dataset - extract the wind shear variable
            if 'shear_mag' in windhshear_ds:
                windhshear_ds = windhshear_ds['shear_mag']
            elif 'wind_shear' in windhshear_ds:
                windhshear_ds = windhshear_ds['wind_shear']
            else:
                # Try to get the first data variable
                var_name = list(windhshear_ds.data_vars)[0]
                windhshear_ds = windhshear_ds[var_name]
        
        # Handle coordinate name differences
        if 'lon' in windhshear_ds.dims and 'longitude' in ivt.dims:
            windhshear_ds = windhshear_ds.rename({'lon': 'longitude', 'lat': 'latitude'})
        
        # Clean coordinates
        windhshear_ds = clean_coordinates(windhshear_ds)
        try:
            shear_min = float(windhshear_ds.min().values)
            shear_max = float(windhshear_ds.max().values)
            print(f"windhshear_ds processed - range: {shear_min:.2f} to {shear_max:.2f} m/s")
        except Exception as e:
            print(f"windhshear_ds processed (could not compute range: {str(e)})")
    else:
        print("Missing  windhshear_ds data...")
    
    # Process IVT with percentile threshold
    if ivt_percentile is not None:
        print("Applying IVT percentile threshold...")
        
        # # Handle coordinate name differences
        # if 'lon' in ivt_percentile.dims and 'longitude' in ivt.dims:
        #     ivt_percentile = ivt_percentile.rename({'lon': 'longitude', 'lat': 'latitude'})
        
        # # Clean coordinates
        # ivt_percentile = clean_coordinates(ivt_percentile)
        
        # Apply threshold
        ivt_enhanced = ivt.where(ivt >= ivt_percentile, 0)
    else:
        print("Using fixed IVT threshold of 250...")
        ivt_enhanced = ivt.where(ivt >= 250, 0)
    
    # Clean coordinates for all variables
    ivt_enhanced = clean_coordinates(ivt_enhanced)
    mslp = clean_coordinates(mslp)

    IWV = clean_coordinates(IWV)
    try:
        iwv_min = float(IWV.min().values)
        iwv_max = float(IWV.max().values)
        print(f"IWV processed - range: {iwv_min:.2f} to {iwv_max:.2f} kg m⁻²")
    except Exception as e:
        print(f"IWV processed (could not compute range: {str(e)})")
    


    # Combine datasets - only include non-None variables
    dataset_dict = {
        'ivt': ivt_enhanced.astype(np.float32),
        'mslp': mslp.astype(np.float32),
        'ivt_grad': ivt_grad.astype(np.float32),
        'vorticity': vorticity.astype(np.float32),
        'vorticity_grad': vorticity_grad.astype(np.float32),
        'IWV': IWV.astype(np.float32),
    }

    # Add optional variables only if they are not None
    if wind_speed is not None:
        dataset_dict['wind_speed'] = wind_speed.astype(np.float32)
    if temp_500mb is not None:
        dataset_dict['temp_500mb'] = temp_500mb.astype(np.float32)
    if windhshear_ds is not None:
        dataset_dict['windhshear'] = windhshear_ds.astype(np.float32)
    if BT_ds is not None:
        dataset_dict['BT_ds'] = BT_ds.astype(np.float32)
    if pr is not None:
        dataset_dict['PR'] = pr.astype(np.float32)
    if prc is not None:
        dataset_dict['PRC'] = prc.astype(np.float32)
    
    combined_ds = xr.Dataset(dataset_dict)
    
    print("GPU-optimized preprocessing complete!")
    print(f"Dataset variables: {list(combined_ds.data_vars)}")
    print(f"Data types: {[(var, combined_ds[var].dtype) for var in combined_ds.data_vars]}")
    
    return combined_ds

