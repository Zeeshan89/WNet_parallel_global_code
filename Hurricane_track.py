import os
import glob
import numpy as np
import os as _os_mod


def _os_env(k, d):
    """Env lookup for tracker thresholds; defaults reproduce original behaviour."""
    return _os_mod.environ.get(k, d)
import xarray as xr
import pandas as pd
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, center_of_mass
from skimage.measure import regionprops
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
class HurricaneTracker:
    def __init__(self, hurricane_data_dir, era5_data_dir, vorticity_dir, wind_speed_path, output_dir):
        """
        Initialize Hurricane Tracker for CONUS region
        
        Args:
            hurricane_data_dir: Directory containing hurricane detection NetCDF files
            era5_data_dir: Directory containing original ERA5 data files
            vorticity_dir: Directory containing vorticity data files
            wind_speed_path: Path to wind speed data file
            output_dir: Directory to save tracking results
        """
        self.hurricane_data_dir = hurricane_data_dir
        self.era5_data_dir = era5_data_dir
        self.vorticity_dir = vorticity_dir
        self.wind_speed_path = wind_speed_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load land-sea mask
        lsm_mask_path = '/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/era5_land_sea_mask.nc'
        print(f"Loading land-sea mask from {lsm_mask_path}")
        lsm_ds = xr.open_dataset(lsm_mask_path)
        self.lsm_mask = lsm_ds['lsm'].squeeze()
        self.lsm_mask = self.lsm_mask.assign_coords(
            longitude=(((self.lsm_mask.longitude + 180) % 360) - 180)
        ).sortby('longitude')
        self.lsm_lats = self.lsm_mask.latitude.values
        self.lsm_lons = self.lsm_mask.longitude.values
        print("Land-sea mask loaded successfully")
        
        # CONUS region boundaries (hurricanes affecting Continental US)
        self.region_bounds = {
                'lon_min': -180.0,
                'lon_max': 180.0,
                'lat_min': -45.0,
                'lat_max': 45.0
            }
        # ADDITIONAL: Define STRICT formation zone (where hurricanes actually form)
        self.formation_bounds = {
            'lat_min': -35.0,
            'lat_max': 35.0
        }
        # Tracking parameters
        self.min_hurricane_area = 3.5  # Keep low to capture all phases (validation filters noise)
        self.max_distance_km = 430   # Larger linking radius for continuity
        self.min_duration_hours = 48 # Minimum 48h (2 days)
        
        # Earth radius for distance calculations
        self.earth_radius_km = 6371.0
        
        # NEW: Additional constraints
        self.max_lat_change_per_6h = 7.0     # Moderate: allow some recurvature
        self.min_ocean_percentage = 0.50      # Allow more land time (hurricanes persist after landfall)
        # Meteorological parameters for hurricanes
        self.max_speed_kmh = 350  # 35 m/s converted to km per 6h (hurricanes can move faster than ETCs)
        self.min_speed_kmh = 40    # 2 m/s converted to km per 6h
        # Linking speed window. The FLOOR is the dangerous one: a landfalling storm
        # slows sharply, and centres are snapped to the 0.25 deg output grid, so two
        # consecutive centres can land on the same cell -> displacement exactly 0 ->
        # speed 0 -> the link is refused and the track breaks precisely where the
        # storm stalls over land. Env-overridable so it can be measured; defaults
        # reproduce the original 0.5-32 m/s window exactly.
        self.min_speed_ms = float(_os_env("TC_MIN_SPEED_MS", "0.5"))
        self.max_speed_ms = float(_os_env("TC_MAX_SPEED_MS", "32.0"))
        # Land-aware gap bridging. Over ocean a long gap usually means two DIFFERENT
        # storms, and widening the gap globally was measured to LOSE hits (318 ->
        # 317 -> 315 for gap 4 -> 8 -> 12) by letting one track claim two storms.
        # Over land the same gap means one storm weakening and possibly re-emerging
        # (GULAB->SHAHEEN crossing India, FRED crossing Cuba), so the two cases want
        # different limits. Default equals the ocean value, i.e. no behaviour change
        # until asked for.
        self.land_gap_steps = int(_os_env("TC_LAND_GAP_STEPS", "0"))
        
        self.min_path_km = 250.0        # Real hurricanes travel substantial distance
        self.min_hours = 48.0           # Minimum 48h duration (IBTrACS standard)
        self.time_gap_hours = 6
        # Hurricane-specific movement preferences
        self.preferred_direction_weight = 1.5  # Less strict than ETCs since hurricanes can have varied paths
        self.direction_consistency_weight = 1.0
        print(f"Hurricane Tracker initialized for CONUS region")
        print(f"Region bounds: Longitude [{self.region_bounds['lon_min']}, {self.region_bounds['lon_max']}], "
              f"Latitude [{self.region_bounds['lat_min']}, {self.region_bounds['lat_max']}]")
        print(f"Hurricane data directory: {hurricane_data_dir}")
        print(f"ERA5 data directory: {era5_data_dir}")
        print(f"Output directory: {output_dir}")    
    def requires_ocean_genesis(self, track_detections, min_ocean_points=3):
        """
        Require that track STARTS over ocean (genesis requirement)
        Hurricanes MUST form over warm ocean water
        
        Args:
            track_detections: List of detections in chronological order
            min_ocean_points: Number of initial points that must be over ocean
        
        Returns:
            Boolean indicating if track has valid ocean genesis
        """
        if len(track_detections) < min_ocean_points:
            return False
        
        # Check first N detections are ALL over ocean
        for i in range(min(min_ocean_points, len(track_detections))):
            if not self.is_over_ocean(
                track_detections[i]['center_lat'], 
                track_detections[i]['center_lon']
            ):
                return False
        
        return True
    def is_realistic_hurricane_movement(self, lat1, lon1, lat2, lon2, time_diff_hours):
        """
        Check if movement is meteorologically realistic (matching comparison code)
        """
        # BUG FIX (2026-09-23): this call used to pass (lon1, lat1, lon2, lat2) into
        # haversine_distance(lat1, lon1, lat2, lon2). The swap weights north-south
        # motion by cos(lon1)*cos(lon2), which is ~0 near 90W and 90E, so a storm
        # heading north at a US Gulf or Bay of Bengal landfall read as ~0 m/s, failed
        # the speed floor, and its track was cut. TC_HAVERSINE_LEGACY=1 restores the
        # old call so the pre-fix products can be reproduced byte-for-byte.
        if _os_env("TC_HAVERSINE_LEGACY", "0") == "1":
            distance = self.haversine_distance(lon1, lat1, lon2, lat2)
        else:
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)

        if time_diff_hours <= 0:
            return False
        
        speed_ms = (distance * 1000.0) / (time_diff_hours * 3600.0)
        
        # Moderate speed constraints (allow stalling hurricanes like Harvey).
        # See self.min_speed_ms in __init__ for why the floor is the risky one.
        if speed_ms < self.min_speed_ms or speed_ms > self.max_speed_ms:
            return False
        
        # Latitude change constraint
        lat_change = abs(lat2 - lat1)
        max_lat_change = self.max_lat_change_per_6h * (time_diff_hours / 6.0)
        
        if lat_change > max_lat_change:
            return False
        
        # Reject equator crossings
        if (lat1 > 0 and lat2 < 0) or (lat1 < 0 and lat2 > 0):
            return False
        
        # Reject large longitudinal jumps (>10 degrees in 6h is suspicious)
        lon_change = abs(lon2 - lon1)
        # Handle dateline wrapping
        if lon_change > 180.0:
            lon_change = 360.0 - lon_change
        max_lon_change = 10.0 * (time_diff_hours / 6.0)
        
        if lon_change > max_lon_change:
            return False
        
        return True
    
    def is_over_ocean(self, lat, lon, threshold=0.5):
        """
        Check if a location is over ocean using land-sea mask
        
        Args:
            lat: Latitude
            lon: Longitude
            threshold: LSM threshold (values < threshold are considered ocean)
                    LSM: 0 = ocean, 1 = land
        
        Returns:
            Boolean indicating if location is over ocean
        """
        try:
            # Find nearest grid point in LSM
            lat_idx = np.argmin(np.abs(self.lsm_lats - lat))
            lon_idx = np.argmin(np.abs(self.lsm_lons - lon))
            
            # Get LSM value
            lsm_value = float(self.lsm_mask.values[lat_idx, lon_idx])
            
            # LSM < threshold means ocean
            return lsm_value < threshold
        except Exception as e:
            print(f"Warning: Could not check LSM for lat={lat}, lon={lon}: {e}")
            return True  # Default to True if check fails
        
    def calculate_bearing(self, lon1, lat1, lon2, lat2):
        """Calculate bearing from point 1 to point 2 in degrees"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = (np.cos(lat1) * np.sin(lat2) - 
            np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        
        bearing = np.arctan2(y, x)
        return (np.degrees(bearing) + 360) % 360
    def is_in_conus_region(self, lat, lon):
        """
        Check if a point is within the CONUS hurricane tracking region
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Boolean indicating if point is in region
        """
        return (self.region_bounds['lon_min'] <= lon <= self.region_bounds['lon_max'] and
                self.region_bounds['lat_min'] <= lat <= self.region_bounds['lat_max'])
    
    def filter_region_data(self, data_array, lats, lons):
        """
        Filter data to CONUS region only
        
        Args:
            data_array: 2D numpy array of data
            lats: Latitude coordinates
            lons: Longitude coordinates
            
        Returns:
            Filtered data, lats, lons and index mappings
        """
        # Find indices within the region
        lat_mask = (lats >= self.region_bounds['lat_min']) & (lats <= self.region_bounds['lat_max'])
        lon_mask = (lons >= self.region_bounds['lon_min']) & (lons <= self.region_bounds['lon_max'])
        
        # Get indices
        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]
        
        if len(lat_indices) == 0 or len(lon_indices) == 0:
            return None, None, None, None, None
        
        # Extract regional data
        regional_data = data_array[np.ix_(lat_indices, lon_indices)]
        regional_lats = lats[lat_indices]
        regional_lons = lons[lon_indices]
        
        return regional_data, regional_lats, regional_lons, lat_indices, lon_indices
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Great-circle distance between two points in km.
        Arguments: lat1, lon1, lat2, lon2  (decimal degrees).
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return self.earth_radius_km * 2 * np.arcsin(np.sqrt(a))
    def extract_hurricane_centers(self, hurricane_mask, lats, lons, threshold=0.5, mslp_field=None):
        """
        Extract hurricane centers from detection mask (CONUS region only)
        
        Args:
            hurricane_mask: 2D array of hurricane detection probabilities
            lats: Latitude coordinates
            lons: Longitude coordinates
            threshold: Threshold for hurricane detection
            
        Returns:
            List of hurricane centers with their properties (CONUS region only)
        """
        # First filter to CONUS region
        regional_data, regional_lats, regional_lons, lat_indices, lon_indices = \
            self.filter_region_data(hurricane_mask, lats, lons)
        
        if regional_data is None:
            return []
        
        # Threshold the hurricane mask
        binary_mask = regional_data > threshold
        
        if not np.any(binary_mask):
            return []
        
        # Label connected components
        labeled_mask, num_features = label(binary_mask)
        
        hurricane_centers = []
        
        for i in range(1, num_features + 1):
            # Extract single hurricane region
            single_hurricane = (labeled_mask == i)
            
            # Check minimum size
            if np.sum(single_hurricane) < self.min_hurricane_area:
                continue
            
            # Centre. Default: mask centroid (original behaviour). TC_CENTER=mslp puts the
            # centre on the MSLP minimum INSIDE the blob: the full-lifecycle masks are
            # 200-500 km discs clipped by coastlines, so the centroid can sit well off
            # the low, which jitters speeds, breaks links and flips ocean/land tests.
            if _os_env("TC_CENTER", "mass") == "mslp" and mslp_field is not None:
                _reg = mslp_field[np.ix_(lat_indices, lon_indices)]
                _v = np.where(single_hurricane, _reg, np.inf)
                if np.isfinite(_v).any():
                    center_y, center_x = np.unravel_index(int(np.argmin(_v)), _v.shape)
                else:
                    center_y, center_x = center_of_mass(single_hurricane)
            else:
                center_y, center_x = center_of_mass(single_hurricane)
            
            # Convert pixel coordinates to lat/lon using regional coordinates
            if center_y < len(regional_lats) and center_x < len(regional_lons):
                center_lat = regional_lats[int(round(center_y))]
                center_lon = regional_lons[int(round(center_x))]
                
                # Double-check that center is in CONUS region
                if self.is_in_conus_region(center_lat, center_lon):
                    # Calculate hurricane area and intensity
                    area_pixels = np.sum(single_hurricane)
                    max_intensity = np.max(regional_data[single_hurricane])
                    mean_intensity = np.mean(regional_data[single_hurricane])
                    
                    # Check 1: Latitude constraint
                    if abs(center_lat) > 45.0:  # Allow recurving to mid-latitudes
                        continue
                    
                    # Check 2: Minimum intensity threshold
                    if max_intensity < float(_os_env("TC_FRAME_INTENSITY", "0.40")):
                        continue

                    # Check 3: Reasonable area.
                    # The 500 px ceiling was calibrated for the old heuristic, whose
                    # detections were 200 km dilated centres (~163 px at 0.25 deg).
                    # The IBTrACS target uses ROCI-sized discs — median 324 km
                    # (~429 px), clamped to 500 km (~1018 px) — so measured on the
                    # exported detections the old cap discards 34.1% of blobs, and
                    # preferentially the LARGEST storms, breaking track continuity
                    # for exactly the major hurricanes that matter most. A cap of
                    # 2000 px rejects 0% while still excluding anything absurd.
                    if area_pixels < int(_os_env("TC_MIN_AREA_PX", "3")) or \
                            area_pixels > int(_os_env("TC_MAX_AREA_PX", "500")):
                        continue
                    
                    hurricane_centers.append({
                        'center_lat': float(center_lat),
                        'center_lon': float(center_lon),
                        'area_pixels': int(area_pixels),
                        'max_intensity': float(max_intensity),
                        'mean_intensity': float(mean_intensity),
                        'mask': single_hurricane,
                        'regional_indices': {
                            'lat_indices': lat_indices,
                            'lon_indices': lon_indices
                        }
                    })
        
        return hurricane_centers
    
    def get_meteorological_data(self, center_lat, center_lon, timestamp, vorticity_data=None, 
                              mslp_data=None, wind_speed_data=None, lats=None, lons=None):
        """
        Extract meteorological data at hurricane center
        """
        if lats is None or lons is None:
            return {'vorticity': None, 'mslp': None, 'wind_speed': None}
        
        # Find nearest grid point
        lat_idx = np.argmin(np.abs(lats - center_lat))
        lon_idx = np.argmin(np.abs(lons - center_lon))
        
        result = {}
        
        # Extract vorticity
        if vorticity_data is not None:
            try:
                result['vorticity'] = float(vorticity_data[lat_idx, lon_idx])
            except:
                result['vorticity'] = None
        else:
            result['vorticity'] = None
        
        # Extract MSLP
        if mslp_data is not None:
            try:
                result['mslp'] = float(mslp_data[lat_idx, lon_idx])
            except:
                result['mslp'] = None
        else:
            result['mslp'] = None
        
        # Extract wind speed
        if wind_speed_data is not None:
            try:
                result['wind_speed'] = float(wind_speed_data[lat_idx, lon_idx])
            except:
                result['wind_speed'] = None
        else:
            result['wind_speed'] = None
        
        return result
        
    def link_hurricanes_temporal(self, prev_hurricanes, curr_hurricanes, time_diff_hours):
        """
        Link hurricanes between two consecutive detection lists.
        Returns a list of dicts with keys:
            prev_idx, curr_idx, distance_km
        Uses is_realistic_hurricane_movement for hard filtering and a
        distance-priority greedy assignment for soft ranking.
        Note: create_hurricane_tracks handles gap-bridging internally; this
        method is retained for external / diagnostic use.
        """
        if not prev_hurricanes or not curr_hurricanes:
            return []
        n_prev = len(prev_hurricanes)
        n_curr = len(curr_hurricanes)
        cost_matrix = np.full((n_prev, n_curr), np.inf)
        for i, prev_h in enumerate(prev_hurricanes):
            for j, curr_h in enumerate(curr_hurricanes):
                if not self.is_realistic_hurricane_movement(
                    prev_h['center_lat'], prev_h['center_lon'],
                    curr_h['center_lat'], curr_h['center_lon'],
                    time_diff_hours
                ):
                    continue
                distance = self.haversine_distance(
                    prev_h['center_lat'], prev_h['center_lon'],
                    curr_h['center_lat'], curr_h['center_lon']
                )
                score = distance
                # Penalise large latitude jumps
                lat_change = abs(curr_h['center_lat'] - prev_h['center_lat'])
                if lat_change > 8.0:
                    score *= 2.0
                # Penalise implausible intensity changes
                prev_i = prev_h.get('max_intensity', 0)
                curr_i = curr_h.get('max_intensity', 0)
                if prev_i > 0 and abs(curr_i - prev_i) / prev_i > 5.0:
                    score *= 1.5
                cost_matrix[i, j] = score
        links = []
        used_curr = set()
        row_order = sorted(range(n_prev), key=lambda i: cost_matrix[i].min())
        for i in row_order:
            if np.all(np.isinf(cost_matrix[i])):
                continue
            costs = cost_matrix[i].copy()
            for j in used_curr:
                costs[j] = np.inf
            if np.all(np.isinf(costs)):
                continue
            best_j    = int(np.argmin(costs))
            best_cost = costs[best_j]
            if best_cost < self.max_distance_km:
                links.append({
                    'prev_idx'   : i,
                    'curr_idx'   : best_j,
                    'distance_km': float(best_cost),
                })
                used_curr.add(best_j)
        return links
    
    def process_year(self, year):
        """
        Process hurricane tracking for a specific year (CONUS region only)
        """
        print(f"\nProcessing year {year} for CONUS region...")
        
        # Get all hurricane detection files for this year
        hurricane_pattern = os.path.join(self.hurricane_data_dir, f"era5dara_{year:04d}*Hurricane_detection.nc")
        hurricane_files = sorted(glob.glob(hurricane_pattern))
        
        if not hurricane_files:
            print(f"No hurricane detection files found for year {year}")
            return
        
        print(f"Found {len(hurricane_files)} hurricane detection files for {year}")
        
        # Load vorticity data for this year
        vorticity_file = self.find_vorticity_file(year)
        vorticity_ds = None
        if vorticity_file:
            print(f"Loading vorticity data: {os.path.basename(vorticity_file)}")
            vorticity_ds = xr.open_dataset(vorticity_file).squeeze('plev')
            vorticity_ds = vorticity_ds.assign_coords(
                longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
            ).sortby('longitude')
        
        # Load wind speed data
        wind_speed_ds = None
        if os.path.exists(self.wind_speed_path):
            print("Loading wind speed data...")
            wind_speed_ds = xr.open_dataset(self.wind_speed_path).squeeze('level')
            wind_speed_ds = wind_speed_ds.assign_coords(
                longitude=(((wind_speed_ds.longitude + 180) % 360) - 180)
            ).sortby('longitude')
        
        # Initialize tracking data
        all_detections = []
        conus_detections_count = 0
        total_detections_count = 0
        
        # Process each file
        for file_idx, hurricane_file in enumerate(hurricane_files):
            print(f"Processing file {file_idx + 1}/{len(hurricane_files)}: {os.path.basename(hurricane_file)}")
            
            # Load hurricane detection data
            hurricane_ds = xr.open_dataset(hurricane_file)
            
            # Load corresponding ERA5 data for MSLP
            era5_file = hurricane_file.replace('_Hurricane_detection.nc', '.nc')
            era5_file = era5_file.replace(self.hurricane_data_dir, self.era5_data_dir)
            
            mslp_data = None
            if os.path.exists(era5_file):
                era5_ds = xr.open_dataset(era5_file)
                era5_ds = era5_ds.assign_coords(
                    longitude=(((era5_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                if 'MSLP' in era5_ds:
                    mslp_data = era5_ds['MSLP']
                era5_ds.close()
            
            # Process each timestep
            for time_idx, timestamp in enumerate(hurricane_ds.time.values):
                hurricane_mask = hurricane_ds['Hurricane_detection'].values[time_idx]
                lats = hurricane_ds.latitude.values
                lons = hurricane_ds.longitude.values
                
                # Extract hurricane centers (CONUS region only)
                _mslp_c = None
                if mslp_data is not None and _os_env("TC_CENTER", "mass") == "mslp":
                    try:
                        _mslp_c = mslp_data.sel(time=timestamp, method='nearest').values
                    except Exception:
                        _mslp_c = None
                hurricanes = self.extract_hurricane_centers(hurricane_mask, lats, lons,
                                                            mslp_field=_mslp_c)
                conus_detections_count += len(hurricanes)
                
                # Count total detections for comparison (optional)
                binary_mask = hurricane_mask > 0.5
                if np.any(binary_mask):
                    labeled_mask, num_features = label(binary_mask)
                    total_detections_count += num_features
                
                if hurricanes:
                    # Get meteorological data for each hurricane
                    for hurricane in hurricanes:
                        # Get vorticity data for this timestamp
                        vor_data = None
                        if vorticity_ds is not None:
                            try:
                                vor_time_data = vorticity_ds.sel(time=timestamp, method='nearest')
                                vor_data = vor_time_data['var138'].values
                            except:
                                vor_data = None
                        
                        # Get MSLP data for this timestamp
                        mslp_time_data = None
                        if mslp_data is not None:
                            try:
                                mslp_time_data = mslp_data.sel(time=timestamp, method='nearest').values
                            except:
                                mslp_time_data = None
                        
                        # Get wind speed data for this timestamp
                        wind_time_data = None
                        if wind_speed_ds is not None:
                            try:
                                wind_time_data = wind_speed_ds.sel(time=timestamp, method='nearest')['wspd'].values
                            except:
                                wind_time_data = None
                        
                        # Extract meteorological parameters
                        met_data = self.get_meteorological_data(
                            hurricane['center_lat'], hurricane['center_lon'], timestamp,
                            vor_data, mslp_time_data, wind_time_data, lats, lons
                        )
                        
                        # Add to detection data
                        detection = {
                            'timestamp': pd.Timestamp(timestamp),
                            'center_lat': hurricane['center_lat'],
                            'center_lon': hurricane['center_lon'],
                            'area_pixels': hurricane['area_pixels'],
                            'max_intensity': hurricane['max_intensity'],
                            'mean_intensity': hurricane['mean_intensity'],
                            'vorticity': met_data['vorticity'],
                            'mslp': met_data['mslp'],
                            'wind_speed': met_data['wind_speed']
                        }
                        
                        all_detections.append(detection)
            
            hurricane_ds.close()
        
        # Clean up
        if vorticity_ds:
            vorticity_ds.close()
        if wind_speed_ds:
            wind_speed_ds.close()
        
        # Print regional filtering statistics
        print(f"Regional filtering results for {year}:")
        print(f"  - Total detections globally: {total_detections_count}")
        print(f"  - CONUS region detections: {conus_detections_count}")
        if total_detections_count > 0:
            print(f"  - CONUS percentage: {100.0 * conus_detections_count / total_detections_count:.1f}%")
        
        # Convert to DataFrame and sort by time
        if all_detections:
            df = pd.DataFrame(all_detections)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Perform tracking
            tracks = self.create_hurricane_tracks(df)
            
            # Save results
            self.save_tracking_results(tracks, year)
            
            print(f"Found {len(all_detections)} CONUS hurricane detections and {len(tracks)} tracks for year {year}")
        else:
            print(f"No CONUS hurricane detections found for year {year}")
    # Rejection tally, so a run reports WHICH criterion discards tracks rather than
    # leaving it to be guessed. These filters were tuned when the model emitted
    # ~45,000 detections and aggressive filtering was essential; the IBTrACS-target
    # model emits ~1,900, so several are now expected to cost real storms.
    REJECT = {}

    # The REJECT tally only sees tracks that already cleared close_track's own
    # >=5 points AND >=48 h gates, so it under-reports: a candidate killed by
    # those two never reaches validate_hurricane_track and is invisible here.
    # With TC_CAPTURE_CANDIDATES=1 every closed candidate is recorded together
    # with the criterion that ended it, which is what lets a miss be attributed
    # to segmentation / linking / validation rather than guessed at. Off by
    # default, so a production run behaves exactly as before.
    # Did the land-aware bridge actually engage? A null result from a lever that
    # never fired is not a null result, so this is counted rather than assumed.
    LAND_BRIDGE = {"checked": 0, "over_land": 0, "extended": 0}
    CANDIDATES = []
    CAPTURE = bool(int(_os_env("TC_CAPTURE_CANDIDATES", "0")))
    LAST_REJECT = None

    @classmethod
    def _capture(cls, dets, reason):
        if not cls.CAPTURE:
            return
        cls.CANDIDATES.append({
            "reason": reason,
            "points": [(d['timestamp'], float(d['center_lat']),
                        float(d['center_lon']), int(d.get('area_pixels', 0)))
                       for d in dets],
        })

    def _rej(self, why):
        HurricaneTracker.REJECT[why] = HurricaneTracker.REJECT.get(why, 0) + 1
        HurricaneTracker.LAST_REJECT = why
        return False

    def validate_hurricane_track(self, track_detections):
        """
        STRICT validation matching IBTrACS comparison criteria.

        Thresholds are env-overridable so they can be re-tuned for a cleaner
        detection field without editing this file. Defaults reproduce the original
        behaviour exactly.
          TC_MIN_POINTS TC_MIN_DURATION_H TC_MIN_PATH_KM TC_MIN_INTENSITY
          TC_GENESIS_LAT TC_MIN_OCEAN_FRAC TC_MIN_STRAIGHTNESS
        """
        import os as _os
        _minpts = int(_os.environ.get("TC_MIN_POINTS", "5"))
        if len(track_detections) < _minpts:
            return self._rej("n_points")
        
        # ============================
        # 1. OCEAN GENESIS REQUIREMENT
        # ============================
        # TC_OCEAN_GENESIS=0 disables this gate (default 1 = original behaviour). With the
        # full-lifecycle TC model a storm stays detected over land, so a track can begin
        # over land after a linking gap, or on a coastal/island centroid: Wilma 2005
        # (160 kt) and Elena 1985 (110 kt) were detected at 92-100% of positions and
        # rejected here.
        if _os.environ.get("TC_OCEAN_GENESIS", "1") == "1" and \
                not self.requires_ocean_genesis(track_detections, min_ocean_points=1):
            return self._rej("ocean_genesis")
        
        # ============================
        # 2. FORMATION ZONE - Must start in tropics/subtropics
        # ============================
        first_lat = track_detections[0]['center_lat']
        first_lon = track_detections[0]['center_lon']
        
        # Moderate: allow subtropical formation up to 38 degrees
        _glat = float(_os.environ.get("TC_GENESIS_LAT", "38.0"))
        if not (-_glat <= first_lat <= _glat):
            return self._rej("genesis_lat")
        
        # NEW: Exclude tracks starting north of 36N AND west of 80W (land-based systems)
        if first_lat >= 36.0 and first_lon <= -85.0:
            return self._rej("nw_landstart")
        
        # ============================
        # 3. LATITUDE CONSTRAINTS
        # ============================
        # Allow more time at higher latitudes for recurving storms
        high_lat_count = sum(1 for d in track_detections if abs(d['center_lat']) > 45.0)
        if high_lat_count / len(track_detections) > 0.50:
            return self._rej("high_lat_frac")
        
        # NEW: Reject tracks north of 40N AND west of 110W (continental systems)
        for detection in track_detections:
            if detection['center_lat'] >= 40.0 and detection['center_lon'] <= -110.0:
                return self._rej("continental")
        
        # ============================
        # 4. OCEAN PERCENTAGE - Match comparison code
        # ============================
        ocean_count = sum(1 for d in track_detections 
                        if self.is_over_ocean(d['center_lat'], d['center_lon']))
        ocean_percentage = ocean_count / len(track_detections)
        
        # Allow hurricanes that persist over land after landfall
        if ocean_percentage < float(_os.environ.get("TC_MIN_OCEAN_FRAC", "0.35")):
            return self._rej("ocean_frac")
        
        # ============================
        # 5. DURATION CHECK - Match IBTrACS minimum
        # ============================
        start_time = track_detections[0]['timestamp']
        end_time = track_detections[-1]['timestamp']
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        
        # IBTrACS standard minimum
        if duration_hours < float(_os.environ.get("TC_MIN_DURATION_H", "48.0")):
            return self._rej("duration")
        
        # ============================
        # 6. PATH LENGTH
        # ============================
        total_distance = 0
        for i in range(1, len(track_detections)):
            dist = self.haversine_distance(
                track_detections[i-1]['center_lat'], track_detections[i-1]['center_lon'],
                track_detections[i]['center_lat'], track_detections[i]['center_lon']
            )
            total_distance += dist
        
        if total_distance < float(_os.environ.get("TC_MIN_PATH_KM", "250.0")):
            return self._rej("path_km")
        
        # ============================
        # 7. MOVEMENT VALIDATION
        # ============================
        # Check for west-of-108W then northeast movement (continental pattern)
        # seen_west108 = False
        # for i, detection in enumerate(track_detections):
        #     lon = detection['center_lon']
        #     lat = detection['center_lat']
            
        #     if lon <= -108.0:
        #         seen_west108 = True
            
        #     # If seen west of 108W, then moved northeast of 100W and above 35N
        #     if seen_west108 and lat >= 35.0 and lon >= -100.0:
        #         return False  # Continental system pattern

        # AFTER — tighten to 105W so Gulf storms near 95W are never flagged
        seen_west105 = False
        for i, detection in enumerate(track_detections):
            lon = detection['center_lon']
            lat = detection['center_lat']
            
            if lon <= -105.0:
                seen_west105 = True
            
            if seen_west105 and lat >= 35.0 and lon >= -100.0:
                return self._rej("west_recurve")
            
        # ============================
        # 8. SPEED VALIDATION
        # ============================
        speeds = []
        unrealistic_movements = 0
        
        for i in range(1, len(track_detections)):
            time_diff = (track_detections[i]['timestamp'] - 
                        track_detections[i-1]['timestamp']).total_seconds() / 3600.0
            
            if time_diff > 0:
                distance = self.haversine_distance(
                    track_detections[i-1]['center_lat'], track_detections[i-1]['center_lon'],
                    track_detections[i]['center_lat'], track_detections[i]['center_lon']
                )
                speed_ms = (distance * 1000.0) / (time_diff * 3600.0)
                speeds.append(speed_ms)
                
                # Moderate hurricane speed bounds: 0.5-32 m/s
                if speed_ms < self.min_speed_ms or speed_ms > self.max_speed_ms:
                    unrealistic_movements += 1
        
        if speeds and unrealistic_movements / len(speeds) > 0.3:
            return self._rej("speed")
        
        # ============================
        # 9. BAND LOGIC - Match comparison code
        # ============================
        # Check 100W-75W band behavior
        entered_band = False
        touched_ocean_in_band = False
        east100_count = 0
        
        for detection in track_detections:
            lon = detection['center_lon']
            
            if lon >= -100.0:
                east100_count += 1
            
            if -100.0 <= lon <= -75.0:
                entered_band = True
                if self.is_over_ocean(detection['center_lat'], detection['center_lon']):
                    touched_ocean_in_band = True
        
        # If >50% of track is east of 100W AND entered band but no ocean in band
        east_fraction = east100_count / len(track_detections)
        
        # Exception: started over ocean east of 82W
        start_on_ocean = self.is_over_ocean(first_lat, first_lon)
        start_ocean_east82 = start_on_ocean and first_lon > -82.0
        
        start_in_gulf = (first_lon <= -82.0 and self.is_over_ocean(first_lat, first_lon))
        if (east_fraction > 0.5 and entered_band and 
            not touched_ocean_in_band and not start_ocean_east82 and not start_in_gulf):
            return self._rej("band_ocean")
        
        # ============================
        # 10. INTENSITY CHECK
        # ============================
        intensities = [d.get('max_intensity', 0) for d in track_detections]
        max_intensity = max(intensities) if intensities else 0
        
        if max_intensity < float(_os.environ.get("TC_MIN_INTENSITY", "0.50")):
            return self._rej("intensity")
        
        # ============================
        # 11. NET DISPLACEMENT - Progressive movement
        # ============================
        start_lat, start_lon = track_detections[0]['center_lat'], track_detections[0]['center_lon']
        end_lat, end_lon = track_detections[-1]['center_lat'], track_detections[-1]['center_lon']
        
        net_displacement = self.haversine_distance(start_lat, start_lon, end_lat, end_lon)
        
        # Net displacement should be at least 20% of total path
        if total_distance > 0:
            displacement_ratio = net_displacement / total_distance
            if displacement_ratio < float(_os.environ.get("TC_MIN_STRAIGHTNESS", "0.12")):
                return self._rej("straightness")
        
        # ============================
        # 12. BEARING CONSISTENCY - Reject zigzag/noise tracks
        # ============================
        # Real hurricanes have smooth curved paths. If too many consecutive
        # segments reverse direction sharply (>120° bearing change), it's noise.
        if len(track_detections) >= 4:
            bearings = []
            for i in range(1, len(track_detections)):
                b = self.calculate_bearing(
                    track_detections[i-1]['center_lon'], track_detections[i-1]['center_lat'],
                    track_detections[i]['center_lon'], track_detections[i]['center_lat']
                )
                bearings.append(b)
            
            sharp_turns = 0
            for i in range(1, len(bearings)):
                bearing_change = abs(bearings[i] - bearings[i-1])
                if bearing_change > 180:
                    bearing_change = 360 - bearing_change

                if bearing_change > 140:  # Sharp reversal
                    sharp_turns += 1
            
            # If >30% of segments have sharp reversals, it's noise
            if len(bearings) > 1 and sharp_turns / (len(bearings) - 1) > 0.40:
                return False
        
        return True
    def create_hurricane_tracks(self, df, max_gap_steps=None):
        """
        Create hurricane tracks from individual detections.
        Tracks persist across gaps of up to max_gap_steps consecutive missing
        timesteps (default 4 × 6 h = 24 h).  A 'ghost' position is carried
        forward using the last known velocity so that the linking window stays
        physically centred.
        Parameters
        ----------
        df : pd.DataFrame
            Sorted detection table (columns: timestamp, center_lat, center_lon, …)
        max_gap_steps : int
            Maximum number of consecutive 6-h timesteps a track may survive
            without a matched detection before it is closed. Default 4 (= 24 h).
        """
        if max_gap_steps is None:
            max_gap_steps = int(_os_env("TC_MAX_GAP_STEPS", "4"))
        if df.empty:
            return []
        df = df.sort_values('timestamp').reset_index(drop=True)
        # ------------------------------------------------------------------ #
        # Build an ordered list of unique timesteps so we can step through    #
        # them one by one and apply gap logic.                                #
        # ------------------------------------------------------------------ #
        unique_times = sorted(df['timestamp'].unique())
        # Group detections by timestep for O(1) lookup
        time_to_detections = {}
        for t in unique_times:
            subset = df[df['timestamp'] == t]
            time_to_detections[t] = subset.to_dict('records')
        # ------------------------------------------------------------------ #
        # Active track state                                                   #
        # Each entry:                                                          #
        #   'detections'  : list of detection dicts (real hits only)          #
        #   'gap_count'   : consecutive missed timesteps so far               #
        #   'ghost_lat'   : extrapolated lat at last missed step              #
        #   'ghost_lon'   : extrapolated lon at last missed step              #
        #   'last_dlat'   : last real velocity component (degrees / step)     #
        #   'last_dlon'   : last real velocity component (degrees / step)     #
        #   'last_time'   : timestamp of last real detection                  #
        # ------------------------------------------------------------------ #
        active_tracks = []
        finished_tracks = []
        track_id_counter = [0]   # list so inner helpers can mutate it
        # These two gates were hardcoded, which silently disabled the
        # TC_MIN_POINTS / TC_MIN_DURATION_H knobs in validate_hurricane_track:
        # a candidate under 5 points or 48 h died here and never reached the
        # function those variables configure. They now read the SAME variables,
        # with the same defaults, so the knobs actually do something.
        _minpts_ct = int(_os_env("TC_MIN_POINTS", "5"))
        _mindur_ct = float(_os_env("TC_MIN_DURATION_H", "48.0"))

        def close_track(tr):
            """Move a track from active → finished after validation."""
            dets = tr['detections']
            if len(dets) < _minpts_ct:
                HurricaneTracker._capture(dets, "n_points_lt5")
                return
            duration = (dets[-1]['timestamp'] - dets[0]['timestamp']).total_seconds() / 3600.0
            if duration < _mindur_ct:
                HurricaneTracker._capture(dets, "duration_lt48")
                return
            HurricaneTracker.LAST_REJECT = None
            if self.validate_hurricane_track(dets):
                HurricaneTracker._capture(dets, "accepted")
                finished_tracks.append({
                    'track_id'      : track_id_counter[0],
                    'detections'    : dets,
                    'duration_hours': duration,
                    'start_time'    : dets[0]['timestamp'],
                    'end_time'      : dets[-1]['timestamp'],
                })
                track_id_counter[0] += 1
            else:
                HurricaneTracker._capture(
                    dets, f"rej:{HurricaneTracker.LAST_REJECT}")
        # ------------------------------------------------------------------ #
        # Main loop — one iteration per timestep                              #
        # ------------------------------------------------------------------ #
        for t_idx, current_time in enumerate(unique_times):
            current_detections = list(time_to_detections.get(current_time, []))
            # Track which current detections have already been claimed
            claimed = [False] * len(current_detections)
            # -------------------------------------------------------------- #
            # 1. Try to extend every active track                             #
            # -------------------------------------------------------------- #
            still_active = []
            for tr in active_tracks:
                # Reference position: real last detection OR ghost
                if tr['gap_count'] == 0:
                    ref_lat = tr['detections'][-1]['center_lat']
                    ref_lon = tr['detections'][-1]['center_lon']
                    ref_time = tr['detections'][-1]['timestamp']
                else:
                    ref_lat  = tr['ghost_lat']
                    ref_lon  = tr['ghost_lon']
                    ref_time = tr['last_time']   # time of the most recent ghost step
                # Time gap to current step
                time_diff_h = (current_time - ref_time).total_seconds() / 3600.0
                if time_diff_h <= 0:
                    still_active.append(tr)
                    continue
                # Find best matching detection at current timestep
                best_idx  = None
                best_dist = np.inf
                for d_idx, det in enumerate(current_detections):
                    if claimed[d_idx]:
                        continue
                    # Realistic movement check (uses the ghost position when bridging)
                    if not self.is_realistic_hurricane_movement(
                        ref_lat, ref_lon,
                        det['center_lat'], det['center_lon'],
                        time_diff_h
                    ):
                        continue
                    dist = self.haversine_distance(
                        ref_lat, ref_lon,
                        det['center_lat'], det['center_lon']
                    )
                    # Scale max distance by time gap, but CAP at 2x to prevent
                    # merging unrelated systems during long gaps
                    gap_scale = min(time_diff_h / 6.0, 2.0)
                    max_dist = self.max_distance_km * gap_scale
                    if dist < max_dist and dist < best_dist:
                        best_dist = dist
                        best_idx  = d_idx
                if best_idx is not None:
                    # ---- Hit: attach detection, reset gap counter --------
                    det = current_detections[best_idx]
                    claimed[best_idx] = True
                    # Recompute velocity for future extrapolation
                    prev_lat = tr['detections'][-1]['center_lat'] if tr['detections'] else ref_lat
                    prev_lon = tr['detections'][-1]['center_lon'] if tr['detections'] else ref_lon
                    tr['last_dlat'] = det['center_lat'] - prev_lat
                    tr['last_dlon'] = det['center_lon'] - prev_lon
                    tr['detections'].append(det)
                    tr['gap_count'] = 0
                    tr['last_time'] = current_time
                    tr['ghost_lat'] = det['center_lat']
                    tr['ghost_lon'] = det['center_lon']
                    still_active.append(tr)
                else:
                    # ---- Miss: increment gap counter ---------------------
                    new_gap = tr['gap_count'] + 1
                    # Over land, allow a longer bridge: the storm is weakening and
                    # may re-emerge, whereas the same gap over ocean usually means
                    # two different storms.
                    _limit = max_gap_steps
                    if self.land_gap_steps > max_gap_steps:
                        HurricaneTracker.LAND_BRIDGE["checked"] += 1
                        if not self.is_over_ocean(tr['ghost_lat'],
                                                  tr['ghost_lon']):
                            _limit = self.land_gap_steps
                            HurricaneTracker.LAND_BRIDGE["over_land"] += 1
                            if new_gap > max_gap_steps:
                                # would have died at the ocean limit
                                HurricaneTracker.LAND_BRIDGE["extended"] += 1
                    if new_gap > _limit:
                        # Too many consecutive misses → close track
                        close_track(tr)
                    else:
                        # Keep alive; advance ghost position with last velocity
                        tr['gap_count'] = new_gap
                        tr['ghost_lat'] = tr['ghost_lat'] + tr['last_dlat']
                        tr['ghost_lon'] = tr['ghost_lon'] + tr['last_dlon']
                        tr['last_time'] = current_time
                        still_active.append(tr)
            active_tracks = still_active
            # -------------------------------------------------------------- #
            # 2. Start new tracks from unclaimed detections                   #
            # -------------------------------------------------------------- #
            for d_idx, det in enumerate(current_detections):
                if claimed[d_idx]:
                    continue
                new_track = {
                    'detections' : [det],
                    'gap_count'  : 0,
                    'ghost_lat'  : det['center_lat'],
                    'ghost_lon'  : det['center_lon'],
                    'last_dlat'  : 0.0,
                    'last_dlon'  : 0.0,
                    'last_time'  : current_time,
                }
                active_tracks.append(new_track)
        # ------------------------------------------------------------------ #
        # 3. Close all tracks still open at end of dataset                    #
        # ------------------------------------------------------------------ #
        for tr in active_tracks:
            close_track(tr)
        return finished_tracks
    def find_vorticity_file(self, year):
        """
        Find vorticity file for given year
        """
        vorticity_pattern = os.path.join(self.vorticity_dir, f"ERA5_vor_6hourly_{year}.850mb_latlon.nc")
        vorticity_files = glob.glob(vorticity_pattern)
        
        if vorticity_files:
            return vorticity_files[0]
        else:
            print(f"Warning: No vorticity file found for year {year}")
            return None
    
    def save_tracking_results(self, tracks, year):
        """
        Save hurricane tracking results to TWO text files only (CONUS region)
        1. Summary file with overview of all tracks
        2. Detailed file with all track information combined
        """
        if not tracks:
            print(f"No CONUS tracks to save for year {year}")
            return
        
        # Create year-specific output directory
        year_output_dir = os.path.join(self.output_dir, f"CONUS_year_{year}")
        os.makedirs(year_output_dir, exist_ok=True)
        
        # File 1: Save summary file
        summary_file = os.path.join(year_output_dir, f"CONUS_hurricane_tracks_summary_{year}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"CONUS Hurricane Tracking Results Summary for Year {year}\n")
            f.write("="*70 + "\n")
            f.write(f"Region: Longitude [{self.region_bounds['lon_min']}, {self.region_bounds['lon_max']}], "
                f"Latitude [{self.region_bounds['lat_min']}, {self.region_bounds['lat_max']}]\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total number of CONUS hurricane tracks: {len(tracks)}\n\n")
            
            # Summary statistics
            total_detections = sum(len(track['detections']) for track in tracks)
            total_duration = sum(track['duration_hours'] for track in tracks)
            avg_duration = total_duration / len(tracks) if tracks else 0
            
            f.write(f"Summary Statistics:\n")
            f.write(f"  - Total detections across all tracks: {total_detections}\n")
            f.write(f"  - Total tracking duration: {total_duration:.1f} hours\n")
            f.write(f"  - Average track duration: {avg_duration:.1f} hours\n")
            f.write(f"  - Longest track duration: {max(track['duration_hours'] for track in tracks):.1f} hours\n")
            f.write(f"  - Shortest track duration: {min(track['duration_hours'] for track in tracks):.1f} hours\n\n")
            
            # Track overview table
            f.write("Track Overview:\n")
            f.write("-" * 120 + "\n")
            f.write("Track ID".ljust(10))
            f.write("Duration(hrs)".ljust(15))
            f.write("Start Time".ljust(20))
            f.write("End Time".ljust(20))
            f.write("Detections".ljust(12))
            f.write("Lat Range".ljust(20))
            f.write("Lon Range".ljust(20))
            f.write("US Impact".ljust(10))
            f.write("\n")
            f.write("-" * 120 + "\n")
            
            for track in tracks:
                # Calculate track statistics
                lats = [d['center_lat'] for d in track['detections']]
                lons = [d['center_lon'] for d in track['detections']]
                
                # Check if track enters continental US (rough approximation)
                continental_us_lats = [lat for lat in lats if 24.0 <= lat <= 49.0]
                continental_us_lons = [lon for lon in lons if -125.0 <= lon <= -66.0]
                us_impact = "YES" if continental_us_lats and continental_us_lons else "NO"
                
                f.write(f"{track['track_id']}".ljust(10))
                f.write(f"{track['duration_hours']:.1f}".ljust(15))
                f.write(f"{track['start_time'].strftime('%Y-%m-%d %H:%M')}".ljust(20))
                f.write(f"{track['end_time'].strftime('%Y-%m-%d %H:%M')}".ljust(20))
                f.write(f"{len(track['detections'])}".ljust(12))
                f.write(f"{min(lats):.2f} to {max(lats):.2f}".ljust(20))
                f.write(f"{min(lons):.2f} to {max(lons):.2f}".ljust(20))
                f.write(f"{us_impact}".ljust(10))
                f.write("\n")
        
        # File 2: Save detailed tracks file (ALL TRACKS IN ONE FILE)
        detailed_file = os.path.join(year_output_dir, f"CONUS_hurricane_tracks_detailed_{year}.txt")
        
        with open(detailed_file, 'w') as f:
            f.write(f"CONUS Hurricane Tracks - Detailed Information for Year {year}\n")
            f.write("="*80 + "\n")
            f.write(f"Region: Longitude [{self.region_bounds['lon_min']}, {self.region_bounds['lon_max']}], "
                f"Latitude [{self.region_bounds['lat_min']}, {self.region_bounds['lat_max']}]\n")
            f.write(f"Total number of tracks: {len(tracks)}\n")
            f.write("="*80 + "\n\n")
            
            # Process each track
            for track_idx, track in enumerate(tracks):
                f.write(f"TRACK {track['track_id']} (Track {track_idx + 1} of {len(tracks)})\n")
                f.write("="*60 + "\n")
                f.write(f"Duration: {track['duration_hours']:.1f} hours\n")
                f.write(f"Start Time: {track['start_time']}\n")
                f.write(f"End Time: {track['end_time']}\n")
                f.write(f"Number of detections: {len(track['detections'])}\n")
                
                # Calculate track statistics
                lats = [d['center_lat'] for d in track['detections']]
                lons = [d['center_lon'] for d in track['detections']]
                
                f.write(f"Latitude range: {min(lats):.2f} to {max(lats):.2f}\n")
                f.write(f"Longitude range: {min(lons):.2f} to {max(lons):.2f}\n")
                
                # Calculate total distance traveled
                total_distance = 0
                for i in range(1, len(track['detections'])):
                    dist = self.haversine_distance(
                        track['detections'][i-1]['center_lat'], 
                        track['detections'][i-1]['center_lon'],
                        track['detections'][i]['center_lat'], 
                        track['detections'][i]['center_lon']
                    )
                    total_distance += dist
                
                f.write(f"Total distance traveled: {total_distance:.1f} km\n")
                
                # Check if track enters continental US
                continental_us_lats = [lat for lat in lats if 24.0 <= lat <= 49.0]
                continental_us_lons = [lon for lon in lons if -125.0 <= lon <= -66.0]
                
                if continental_us_lats and continental_us_lons:
                    f.write("** POTENTIALLY IMPACTS CONTINENTAL US **\n")
                
                f.write("\nDetailed Track Points:\n")
                f.write("-" * 125 + "\n")
                
                # Header for track points
                f.write("Timestamp".ljust(20))
                f.write("Latitude".ljust(12))
                f.write("Longitude".ljust(12))
                f.write("Vorticity".ljust(12))
                f.write("MSLP(Pa)".ljust(12))
                f.write("WindSpeed".ljust(12))
                f.write("Area(px)".ljust(10))
                f.write("MaxIntensity".ljust(12))
                f.write("MeanIntensity".ljust(12))
                f.write("\n")
                f.write("-" * 125 + "\n")
                
                # Track points
                for detection in track['detections']:
                    timestamp_str = detection['timestamp'].strftime('%Y-%m-%d %H:%M')
                    lat_str = f"{detection['center_lat']:.3f}" if detection['center_lat'] is not None else "N/A"
                    lon_str = f"{detection['center_lon']:.3f}" if detection['center_lon'] is not None else "N/A"
                    vor_str = f"{detection['vorticity']:.2e}" if detection['vorticity'] is not None else "N/A"
                    mslp_str = f"{detection['mslp']:.0f}" if detection['mslp'] is not None else "N/A"
                    wind_str = f"{detection['wind_speed']:.1f}" if detection['wind_speed'] is not None else "N/A"
                    area_str = f"{detection['area_pixels']}"
                    max_intensity_str = f"{detection['max_intensity']:.3f}"
                    mean_intensity_str = f"{detection['mean_intensity']:.3f}"
                    
                    f.write(timestamp_str.ljust(20))
                    f.write(lat_str.ljust(12))
                    f.write(lon_str.ljust(12))
                    f.write(vor_str.ljust(12))
                    f.write(mslp_str.ljust(12))
                    f.write(wind_str.ljust(12))
                    f.write(area_str.ljust(10))
                    f.write(max_intensity_str.ljust(12))
                    f.write(mean_intensity_str.ljust(12))
                    f.write("\n")
                
                # Add spacing between tracks (except for the last track)
                if track_idx < len(tracks) - 1:
                    f.write("\n" + "="*80 + "\n\n")
        
        print(f"Saved CONUS hurricane tracking results for year {year}:")
        print(f"  - Summary file: {os.path.basename(summary_file)}")
        print(f"  - Detailed file: {os.path.basename(detailed_file)}")
        print(f"  - Total tracks: {len(tracks)}")
        print(f"  - Location: {year_output_dir}")
    
    def run_tracking(self, start_year=1980, end_year=2024):
        """
        Run hurricane tracking for specified year range (CONUS region only)
        """
        print(f"Starting CONUS hurricane tracking from {start_year} to {end_year}")
        print(f"Tracking region: Longitude [{self.region_bounds['lon_min']}, {self.region_bounds['lon_max']}], "
              f"Latitude [{self.region_bounds['lat_min']}, {self.region_bounds['lat_max']}]")
        
        for year in range(start_year, end_year + 1):
            try:
                self.process_year(year)
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"CONUS hurricane tracking completed for years {start_year}-{end_year}")
def main():
    """
    Main function to run CONUS hurricane tracking
    """
    # Configuration - adjust these paths according to your setup
    hurricane_data_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_21"
    era5_data_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
    vorticity_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/"
    wind_speed_path = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc"
    output_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_21/Hurricanes_track_files_final_3"
    # Create hurricane tracker for CONUS region
    tracker = HurricaneTracker(
        hurricane_data_dir=hurricane_data_dir,
        era5_data_dir=era5_data_dir,
        vorticity_dir=vorticity_dir,
        wind_speed_path=wind_speed_path,
        output_dir=output_dir
    )
    
    # You can specify year range here
    start_year = 2024  # Adjust as needed
    end_year = 2024  # Adjust as needed - start with a small range for testing
    
    # Run tracking for CONUS region
    tracker.run_tracking(start_year=start_year, end_year=end_year)
if __name__ == "__main__":
    main()