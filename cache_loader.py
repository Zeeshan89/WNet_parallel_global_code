"""
HDF5-Aware Data Loading
=======================
Loads from HDF5 cache when available (10-20x faster),
falls back to NetCDF with preprocessing if cache doesn't exist.

To enable HDF5 caching (recommended for performance):
    pip install h5py
Or in conda:
    conda install -c conda-forge h5py
"""

try:
    import h5py
    H5PY_AVAILABLE = True
    print("✓ h5py is available - HDF5 caching enabled")
except Exception as e:
    H5PY_AVAILABLE = False
    print(f"⚠ h5py not available ({e}). Install with: pip install h5py")
    print("  Or with conda: conda install -c conda-forge h5py")
    print("  Cache operations will be slower (NetCDF fallback)")

import numpy as np
try:
    import xarray as xr
    XR_AVAILABLE = True
except Exception:
    XR_AVAILABLE = False
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import os


class CacheAwareDataLoader:
    """Load preprocessed data from HDF5 cache or NetCDF with fallback."""
    
    def __init__(self, cache_dir: str = './cache/preprocessed'):
        self.cache_dir = Path(cache_dir)
        # Ensure cache directory exists so callers don't fail on missing path
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; creation may fail on read-only filesystems
            pass
        
    def get_cache_path(self, year: int, month: int) -> Path:
        """Get the cache file path for a specific month."""
        return self.cache_dir / f"era5_preprocessed_{year:04d}{month:02d}.h5"
    
    def load_from_cache(self, year: int, month: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load preprocessed data from HDF5 cache.
        
        Args:
            year: Year
            month: Month
            
        Returns:
            Dictionary with preprocessed data arrays, or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(year, month)
        
        if not H5PY_AVAILABLE:
            print("Warning: 'h5py' not available in Python environment — HDF5 cache disabled.")
            return None

        if not cache_path.exists():
            return None
        
        try:
            data = {}
            with h5py.File(cache_path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][()]  # Load into memory
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {str(e)}")
            return None
    
    def load_data(
        self,
        year: int,
        month: int,
        data_dir: str,
        use_cache: bool = True,
        fallback_to_netcdf: bool = True,
        **preprocess_kwargs
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load data from cache or NetCDF with preprocessing.
        
        Args:
            year: Year
            month: Month
            data_dir: Directory with NetCDF files (for fallback)
            use_cache: Try to load from cache first
            fallback_to_netcdf: If cache fails, fall back to NetCDF preprocessing
            **preprocess_kwargs: Arguments for preprocessing (if using NetCDF)
                Should include: vorticity, wind_speed, temp_500mb, BT_ds, windhshear_ds
            
        Returns:
            Dictionary with preprocessed data, or None if loading fails
        """
        
        # Try cache first (only if h5py is available)
        if use_cache and not H5PY_AVAILABLE:
            print("Note: 'h5py' not found — will skip HDF5 cache and preprocess from NetCDF.")
            use_cache = False

        if use_cache:
            data = self.load_from_cache(year, month)
            if data is not None:
                print(f"✓ Loaded from HDF5 cache: {self.get_cache_path(year, month)}")
                return data
            else:
                print(f"  Cache not found: {self.get_cache_path(year, month).name}")
        
        # Fallback to NetCDF preprocessing
        if fallback_to_netcdf:
            print(f"  Preprocessing from NetCDF (slower)...")
            from main import calculate_ivt, preprocess_data_improved, find_matching_vorticity
            
            date_str = f"{year:04d}{month:02d}"
            nc_filename = f"era5dara_{date_str}.nc"
            file_path = os.path.join(data_dir, nc_filename)
            
            if not os.path.exists(file_path):
                print(f"  Error: NetCDF file not found: {file_path}")
                return None
            
            try:
                # Load and preprocess (this is what was slow before)
                if not XR_AVAILABLE:
                    print("Error: 'xarray' not available in Python environment — cannot preprocess NetCDF fallback.")
                    return None
                    
                ERA5_6hrly_ARs = xr.open_dataset(file_path)
                ERA5_6hrly_ARs = calculate_ivt(ERA5_6hrly_ARs)
                ERA5_6hrly_ARs = ERA5_6hrly_ARs.assign_coords(
                    longitude=(((ERA5_6hrly_ARs.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                
                # Extract main variables from ERA5 file
                ivt_da = ERA5_6hrly_ARs['IVT']
                mslp_da = ERA5_6hrly_ARs['MSLP']
                
                # Extract TPR and IWV with fallback
                # tpr_da = None
                iwv_da = None
                # if 'TPR' in ERA5_6hrly_ARs:
                #     tpr_da = ERA5_6hrly_ARs['TPR']
                if 'IWV' in ERA5_6hrly_ARs:
                    iwv_da = ERA5_6hrly_ARs['IWV']
                    
                era5_times = ERA5_6hrly_ARs['time'].values
                
                # Ensure TPR and IWV are not None (create dummy data if needed)
                # if tpr_da is None:
                #     print(f"    Warning: TPR not found in ERA5 file, using zeros")
                #     tpr_da = xr.zeros_like(ERA5_6hrly_ARs['IVT'])
                if iwv_da is None:
                    print(f"    Warning: IWV not found in ERA5 file, using zeros")
                    iwv_da = xr.zeros_like(ERA5_6hrly_ARs['IVT'])
                
                # Extract auxiliary data from kwargs or use None
                vorticity_data = None
                if 'vorticity' in preprocess_kwargs and preprocess_kwargs['vorticity'] is not None:
                    vort_full = preprocess_kwargs['vorticity']
                    try:
                        vorticity_data = vort_full.sel(time=era5_times, method='nearest')
                    except Exception:
                        print(f"    Warning: Could not select vorticity for dates in {date_str}")
                        vorticity_data = None
                
                wind_speed_data = None
                if 'wind_speed' in preprocess_kwargs and preprocess_kwargs['wind_speed'] is not None:
                    wind_full = preprocess_kwargs['wind_speed']
                    try:
                        # Extract the 'wspd' DataArray if it's a Dataset
                        if hasattr(wind_full, 'data_vars'):
                            # It's a Dataset - extract the 'wspd' variable
                            if 'wspd' in wind_full:
                                wind_full = wind_full['wspd']
                            elif 'wind_speed' in wind_full:
                                wind_full = wind_full['wind_speed']
                            else:
                                # Try to get the first data variable
                                var_name = list(wind_full.data_vars)[0]
                                wind_full = wind_full[var_name]
                        
                        wind_speed_data = wind_full.sel(time=era5_times, method='nearest')
                    except Exception as e:
                        print(f"    Warning: Could not select wind speed for dates in {date_str}: {str(e)}")
                        wind_speed_data = None
                
                # TPR and IWV from kwargs or ERA5
                # tpr_data = preprocess_kwargs.get('TPR', tpr_da) if 'TPR' in preprocess_kwargs else tpr_da
                iwv_data = preprocess_kwargs.get('IWV', iwv_da) if 'IWV' in preprocess_kwargs else iwv_da
                
                # Ensure they're not None
                # tpr_data = tpr_data if tpr_data is not None else tpr_da
                iwv_data = iwv_data if iwv_data is not None else iwv_da
                
                # Other optional data
                temp_500mb_data = preprocess_kwargs.get('temp_500mb')
                bt_data = preprocess_kwargs.get('BT_ds')
                windhshear_data = preprocess_kwargs.get('windhshear_ds')
                ivt_percentile_data = preprocess_kwargs.get('ivt_percentile', 400)
                
                # Call preprocessing with available data
                preprocessed = preprocess_data_improved(
                    ivt=ivt_da,
                    mslp=mslp_da,
                    vorticity=vorticity_data,
                    wind_speed=wind_speed_data,
                    ivt_percentile=ivt_percentile_data,
                    temp_500mb=temp_500mb_data,
                    BT_ds=bt_data,
                    # TPR=tpr_data,
                    IWV=iwv_data,
                    windhshear_ds=windhshear_data
                )
                
                # Convert to numpy arrays
                data = {}
                for key, value in preprocessed.items():
                    if value is not None:
                        try:
                            if isinstance(value, xr.DataArray):
                                data[key] = value.values.astype(np.float32)
                            elif isinstance(value, np.ndarray):
                                data[key] = value.astype(np.float32)
                        except Exception as e:
                            print(f"    Warning: Could not convert {key} to numpy: {str(e)}")
                
                print(f"  ✓ Loaded from NetCDF with preprocessing ({len(data)} variables)")
                return data
                
            except Exception as e:
                import traceback
                print(f"  Error preprocessing NetCDF: {str(e)}")
                print(f"  Traceback: {traceback.format_exc()}")
                return None
        
        return None


def get_cache_info(cache_dir: str = './cache/preprocessed') -> Dict[str, Any]:
    """Get information about cached files."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
    
    files = sorted(cache_path.glob('era5_preprocessed_*.h5'))
    total_size = sum(f.stat().st_size for f in files)
    
    info = {
        'total_files': len(files),
        'total_size_mb': total_size / (1024**2),
        'files': [
            {
                'name': f.name,
                'size_mb': f.stat().st_size / (1024**2),
                'path': str(f)
            }
            for f in files
        ]
    }
    
    return info
