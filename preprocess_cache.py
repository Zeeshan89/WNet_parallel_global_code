"""
HDF5 Cache Generator for Data Preprocessing
============================================
Pre-processes data once and saves to HDF5 format for 10-20x faster loading during training.
Run this once per month to cache all preprocessing:
    python preprocess_cache.py --year 1980 --month 1
Or for a range:
    python preprocess_cache.py --year 1980 --start-month 1 --end-month 12
"""

import os
import sys
import h5py
import numpy as np
import xarray as xr
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import time
import glob
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WNet_parallel_global_code.main_old import calculate_ivt, preprocess_data_improved, find_matching_vorticity


class HDF5CacheGenerator:
    """Generate and manage HDF5 cache files for preprocessed data."""
    
    def __init__(self, cache_dir: str = './cache/preprocessed'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, year: int, month: int) -> Path:
        """Get the cache file path for a specific month."""
        return self.cache_dir / f"era5_preprocessed_{year:04d}{month:02d}.h5"
    
    def cache_exists(self, year: int, month: int) -> bool:
        """Check if cache file already exists."""
        return self.get_cache_path(year, month).exists()
    
    def preprocess_and_cache_month(
        self,
        data_dir: str,
        year: int,
        month: int,
        vorticity_files: list,
        wind_speed_full: xr.Dataset,
        vor850_results_per: np.ndarray,
        windspeed_results_per: np.ndarray,
        verbose: bool = True
    ) -> bool:
        """
        Load a monthly file, preprocess it, and save to HDF5 cache.
        
        Args:
            data_dir: Directory containing era5dara_*.nc files
            year: Year to process
            month: Month to process
            vorticity_files: List of vorticity file paths
            wind_speed_full: Loaded wind speed dataset
            vor850_results_per: Vorticity percentile thresholds
            windspeed_results_per: Wind speed percentile thresholds
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        
        cache_path = self.get_cache_path(year, month)
        
        # Check if already cached
        if cache_path.exists():
            if verbose:
                print(f"✓ Cache already exists: {cache_path}")
            return True
        
        # Construct filename: era5dara_YYYYMM.nc
        date_str = f"{year:04d}{month:02d}"
        nc_filename = f"era5dara_{date_str}.nc"
        file_path = os.path.join(data_dir, nc_filename)
        
        if not os.path.exists(file_path):
            if verbose:
                print(f"✗ File not found: {file_path}")
            return False
        
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing: {nc_filename}")
                print(f"Cache: {cache_path}")
                print(f"{'='*60}")
            
            start_time = time.time()
            
            # Load main ERA5 data
            if verbose:
                print(f"[1/8] Loading ERA5 main data...")
            ERA5_6hrly_ARs = xr.open_dataset(file_path)
            ERA5_6hrly_ARs = calculate_ivt(ERA5_6hrly_ARs)
            ERA5_6hrly_ARs = ERA5_6hrly_ARs.assign_coords(
                longitude=(((ERA5_6hrly_ARs.longitude + 180) % 360) - 180)
            ).sortby('longitude')
            
            # keep DataArray objects so preprocessing can use coordinates
            ivt_da = ERA5_6hrly_ARs['IVT']
            mslp_da = ERA5_6hrly_ARs['MSLP']
            # TPR_da = ERA5_6hrly_ARs['TPR']
            IWV_da = ERA5_6hrly_ARs['IWV']
            time_values = ERA5_6hrly_ARs['time'].values
            era5_dates = ERA5_6hrly_ARs['time'].values
            
            # Load temperature data
            if verbose:
                print(f"[2/8] Loading 500mb temperature...")

            temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/monthly_output/6_hourly'
            temp_500mb_file = os.path.join(temp_500mb_dir, f'ERA5_t_500mb_{date_str}.nc')

            if os.path.exists(temp_500mb_file):
                temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level='500')
                temp_500mb_ds = temp_500mb_ds.assign_coords(
                    longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                temp_500mb_interp = temp_500mb_ds["T"]
                common_times = np.intersect1d(temp_500mb_interp.time.values, ERA5_6hrly_ARs.time.values)
                temp_500mb = temp_500mb_interp.sel(time=common_times).astype(np.float32)

            # temp_500mb_dir = '/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/GLOBAL/from-ncar-server/t/Hurricane_warmcore/monthly_output_6hourly'
            # temp_500mb_file = os.path.join(temp_500mb_dir, f'ERA5_warmcore_{date_str}_6hourly.nc')
            
            # if os.path.exists(temp_500mb_file):
            #     temp_500mb_ds = xr.open_dataset(temp_500mb_file).sel(level='300')
            #     temp_500mb_ds = temp_500mb_ds.assign_coords(
            #         longitude=(((temp_500mb_ds.longitude + 180) % 360) - 180)
            #     ).sortby('longitude')
            #     temp_500mb_interp = temp_500mb_ds["T"]
            #     common_times = np.intersect1d(temp_500mb_interp.time.values, ERA5_6hrly_ARs.time.values)
            #     temp_500mb = temp_500mb_interp.sel(time=common_times).astype(np.float32)
            # else:
            #     temp_500mb = None
            #     if verbose:
            #         print(f"  Warning: Temperature file not found, using None")
            
            # Load wind shear data
            if verbose:
                print(f"[3/8] Loading wind shear...")
            windshear_dir = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_shear_mag_sqrtu_v/'
            windshear_file = os.path.join(windshear_dir, f'6hourly_ERA5_windshear_mag_{date_str}.nc')
            
            if os.path.exists(windshear_file):
                windhshear_ds = xr.open_dataset(windshear_file).sel(level='200')
                windhshear_ds = windhshear_ds.assign_coords(
                    longitude=(((windhshear_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                windshear_ds_interp = windhshear_ds["windshear"]
                common_times = np.intersect1d(windshear_ds_interp.time.values, ERA5_6hrly_ARs.time.values)
                windhshear_ds = windshear_ds_interp.sel(time=common_times).astype(np.float32)
            else:
                windhshear_ds = None
                if verbose:
                    print(f"  Warning: Wind shear file not found, using None")
            
            # Load wind speed
            if verbose:
                print(f"[4/8] Loading wind speed...")
            wind_speed_interp = wind_speed_full['wspd']
            wind_speed_selected = wind_speed_interp.sel(time=era5_dates).astype(np.float32)
            
            # Load vorticity
            if verbose:
                print(f"[5/8] Loading vorticity...")
            vorticity_file = find_matching_vorticity(era5_dates[0], vorticity_files)
            if vorticity_file is not None:
                vorticity_ds = xr.open_dataset(vorticity_file).squeeze('plev')
                vorticity_ds = vorticity_ds.assign_coords(
                    longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                vorticity_interp = vorticity_ds['var138']
                common_times = np.intersect1d(vorticity_interp.time.values, ERA5_6hrly_ARs.time.values)
                vorticity = vorticity_interp.sel(time=common_times).astype(np.float32)
            else:
                vorticity = None
                if verbose:
                    print(f"  Warning: Vorticity file not found, using None")
            
            # Load brightness temperature
            if verbose:
                print(f"[6/8] Loading brightness temperature...")
            BT_dir = '/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new/'
            BT_file = os.path.join(BT_dir, f'era5_bt_6hourly_{year:04d}.nc')
            
            if os.path.exists(BT_file):
                BT_ds = xr.open_dataset(BT_file).rename({'valid_time': 'time'})
                BT_ds = BT_ds.assign_coords(
                    longitude=(((BT_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                BT_ds = BT_ds['bt'].sel(time=ERA5_6hrly_ARs['time']).astype(np.float32)
            else:
                BT_ds = None
                if verbose:
                    print(f"  Warning: Brightness temperature file not found, using None")
            
            # Preprocess
            if verbose:
                print(f"[7/8] Preprocessing (computing gradients, etc)...")
            preprocessed_data = preprocess_data_improved(
                ivt=ivt_da,
                mslp=mslp_da,
                vorticity=vorticity,
                wind_speed=wind_speed_selected,
                ivt_percentile=400,
                temp_500mb=temp_500mb,
                BT_ds=BT_ds,
                # TPR=TPR_da,
                IWV=IWV_da,
                windhshear_ds=windhshear_ds
            )
            
            # Save to HDF5
            if verbose:
                print(f"[8/8] Writing to HDF5 cache...")
            with h5py.File(cache_path, 'w') as f:
                # Save all preprocessed data
                for key, value in preprocessed_data.items():
                    if value is None:
                        continue
                    # DataArray objects can be converted via .values, numpy arrays remain unchanged
                    if isinstance(value, xr.DataArray):
                        data = value.values.astype(np.float32)
                    elif isinstance(value, np.ndarray):
                        data = value.astype(np.float32)
                    else:
                        # handle scalars or other types
                        try:
                            data = np.array(value)
                        except Exception:
                            continue
                    f.create_dataset(key, data=data, compression='gzip', compression_opts=4)
                
                # Save time coordinate for reference
                # convert numpy datetime64 to integer seconds since epoch for HDF5
                try:
                    time_numeric = time_values.astype('datetime64[s]').astype('int64')
                    f.create_dataset('time', data=time_numeric)
                    f.attrs['time_units'] = 'seconds since 1970-01-01 UTC'
                except Exception:
                    # fallback: save ISO strings
                    time_str = np.array([str(t) for t in time_values], dtype='S')
                    f.create_dataset('time', data=time_str)
                    f.attrs['time_units'] = 'ISO8601 string'
                
                # Save latitude/longitude coordinates for proper reconstruction
                # (ERA5 0.25° grid, longitude already shifted to -180..180)
                lat_vals = ERA5_6hrly_ARs.latitude.values.astype(np.float32)
                lon_vals = ERA5_6hrly_ARs.longitude.values.astype(np.float32)
                f.create_dataset('latitude', data=lat_vals)
                f.create_dataset('longitude', data=lon_vals)
                
                # Save metadata
                f.attrs['year'] = year
                f.attrs['month'] = month
                f.attrs['num_timesteps'] = len(time_values)
                f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            elapsed = time.time() - start_time
            cache_size = cache_path.stat().st_size / (1024**2)  # MB
            
            if verbose:
                print(f"✓ Cache created successfully!")
                print(f"  Size: {cache_size:.1f} MB")
                print(f"  Time: {elapsed:.1f}s")
                print(f"  Variables: {list(preprocessed_data.keys())}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing {nc_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate HDF5 cache files for preprocessed data'
    )
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, default=None, help='Specific month to process')
    parser.add_argument('--start-month', type=int, default=1, help='Start month (default: 1)')
    parser.add_argument('--end-month', type=int, default=12, help='End month (default: 12)')
    parser.add_argument('--data-dir', type=str, 
                       default='/lustre/orion/cli138/proj-shared/z7a/Storms_data/data',
                       help='Directory with ERA5 data')
    parser.add_argument('--cache-dir', type=str, 
                       default='./cache/preprocessed',
                       help='Output cache directory')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip months that already have cache')
    
    args = parser.parse_args()
    
    # Setup cache generator
    generator = HDF5CacheGenerator(args.cache_dir)
    
    # Load static data once (wind speed)
    print("Loading wind speed data (shared across all months)...")
    # Use same wind speed file as main.py to avoid path mismatch
    wind_speed_path = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc'
    try:
        wind_speed_full = xr.open_dataset(wind_speed_path).squeeze('level')
        wind_speed_full = wind_speed_full.assign_coords(
            longitude=(((wind_speed_full.longitude + 180) % 360) - 180)
        ).sortby('longitude')
        print(f"✓ Wind speed loaded: {wind_speed_full.dims}")
    except Exception as e:
        print(f"Warning: could not open wind speed file {wind_speed_path}: {e}")
        wind_speed_full = None

    
    # Find vorticity files (use same directory as main.py to avoid mismatch)
    vorticity_base = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/'
    if os.path.isdir(vorticity_base):
        vorticity_files = sorted(glob.glob(os.path.join(vorticity_base, 'ERA5_vor_6hourly_*.850mb_latlon.nc')))
        print(f"✓ Found {len(vorticity_files)} vorticity files")
    else:
        vorticity_files = []
        print(f"Warning: vorticity directory not found: {vorticity_base}")

    
    # Load percentile thresholds (dummy values for now)
    vor850_results_per = np.zeros((721, 1440), dtype=np.float32)
    windspeed_results_per = np.zeros((721, 1440), dtype=np.float32)
    
    # Determine months to process
    if args.month is not None:
        months = [args.month]
    else:
        months = list(range(args.start_month, args.end_month + 1))
    
    print(f"\nProcessing {args.year}, months {months}")
    print(f"Cache directory: {args.cache_dir}\n")
    
    # Process each month
    success_count = 0
    for month in months:
        if args.skip_existing and generator.cache_exists(args.year, month):
            print(f"Skipping {args.year}-{month:02d} (already cached)")
            success_count += 1
            continue
        
        success = generator.preprocess_and_cache_month(
            data_dir=args.data_dir,
            year=args.year,
            month=month,
            vorticity_files=vorticity_files,
            wind_speed_full=wind_speed_full,
            vor850_results_per=vor850_results_per,
            windspeed_results_per=windspeed_results_per,
            verbose=True
        )
        
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(months)} months cached")
    print(f"Cache directory: {args.cache_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
