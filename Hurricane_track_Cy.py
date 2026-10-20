import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, center_of_mass
from scipy.optimize import linear_sum_assignment
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
        self.min_hurricane_area = 20  # Keep low to capture all phases (validation filters noise)
        self.max_distance_km = 650   # Larger linking radius for continuity
        self.min_duration_hours = 48 # Minimum 48h (2 days)
        
        # Earth radius for distance calculations
        self.earth_radius_km = 6371.0
        
        # NEW: Additional constraints
        self.max_lat_change_per_6h = 7.0     # Moderate: allow some recurvature
        self.min_ocean_percentage = 0.50      # Allow more land time (hurricanes persist after landfall)
        # Meteorological parameters for hurricanes
        self.max_speed_kmh = 350  # 35 m/s converted to km per 6h (hurricanes can move faster than ETCs)
        self.min_speed_kmh = 40    # 2 m/s converted to km per 6h
        
        # self.min_path_km = 350.0        # Real hurricanes travel substantial distance
        self.min_hours = 48.0           # Minimum 48h duration (IBTrACS standard)
        self.time_gap_hours = 6

        # CyTRACK gap fill: only 1 missing step to prevent straight-line artifacts
        self.max_gap_steps = 1           # 1 missed step (6h) — prevents multi-ghost straight lines
        self.max_gap_angle_deg = 45.0    # Relaxed from 10° — hurricanes recurve significantly

        # Hurricane-specific movement preferences
        self.preferred_direction_weight = 1.5  # Less strict than ETCs since hurricanes can have varied paths
        self.direction_consistency_weight = 1.0
        print(f"Hurricane Tracker initialized for CONUS region")
        print(f"Region bounds: Longitude [{self.region_bounds['lon_min']}, {self.region_bounds['lon_max']}], "
              f"Latitude [{self.region_bounds['lat_min']}, {self.region_bounds['lat_max']}]")
        print(f"Hurricane data directory: {hurricane_data_dir}")
        print(f"ERA5 data directory: {era5_data_dir}")
        print(f"Output directory: {output_dir}")    
   

    def _check_trajectory_angle(self, lat0, lon0, lat1, lon1, lat2, lon2) -> bool:
        """
        CyTRACK gap-fill trajectory smoothness check.
        
        The angle between segment (t0→t1) and segment (t1→t2) must be
        less than max_gap_angle_deg (10°). If it exceeds this, the track
        ends — the movement is physically unrealistic.
        
        Returns True if angle is acceptable (track continues).
        """
        b1 = self.calculate_bearing(lon0, lat0, lon1, lat1)
        b2 = self.calculate_bearing(lon1, lat1, lon2, lat2)
        
        angle = abs(b2 - b1)
        if angle > 180.0:
            angle = 360.0 - angle
        
        return angle < self.max_gap_angle_deg


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
        # BUG FIX (2026-09-23): arguments were (lon1, lat1, lon2, lat2); see Hurricane_track.py.
        distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        if time_diff_hours <= 0:
            return False
        
        speed_ms = (distance * 1000.0) / (time_diff_hours * 3600.0)
        
        # Moderate speed constraints: 0.5-32 m/s (allow stalling hurricanes like Harvey)
        if speed_ms < 0.5 or speed_ms > 32.0:
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
    
    def extract_hurricane_centers(self, hurricane_mask, lats, lons, threshold=0.5):
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
            
            # Calculate center of mass
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
                    if max_intensity < 0.55:  # Low to capture weak/forming phases
                        continue
                    
                    # Check 3: Reasonable area (not too small, not suspiciously large)
                    if area_pixels < 20 or area_pixels > 500:  # Keep low for detection continuity
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
        
    def _predict_position(self, track_detections):
        """
        Predict next position based on recent track velocity.
        Uses last 2-3 points to estimate velocity vector.
        Returns (pred_lat, pred_lon, bearing) or None if insufficient history.
        """
        dets = track_detections
        if len(dets) < 2:
            return None

        # Use last 2 points (or weighted average of last 3)
        n = min(3, len(dets))
        recent = dets[-n:]

        # Weighted velocity: more recent segments get higher weight
        dlat_sum = 0.0
        dlon_sum = 0.0
        weight_sum = 0.0
        for i in range(1, len(recent)):
            dt_h = (recent[i]['timestamp'] - recent[i-1]['timestamp']).total_seconds() / 3600.0
            if dt_h <= 0:
                continue
            w = float(i)  # higher weight for more recent
            dlat = recent[i]['center_lat'] - recent[i-1]['center_lat']
            dlon = recent[i]['center_lon'] - recent[i-1]['center_lon']
            # Handle dateline
            if dlon > 180: dlon -= 360
            if dlon < -180: dlon += 360
            dlat_sum += w * (dlat / dt_h)
            dlon_sum += w * (dlon / dt_h)
            weight_sum += w

        if weight_sum == 0:
            return None

        vlat = dlat_sum / weight_sum  # degrees per hour
        vlon = dlon_sum / weight_sum

        last = dets[-1]
        # Predict 6h ahead
        pred_lat = last['center_lat'] + vlat * 6.0
        pred_lon = last['center_lon'] + vlon * 6.0

        # Compute expected bearing
        bearing = self.calculate_bearing(
            last['center_lon'], last['center_lat'],
            pred_lon, pred_lat
        )

        return (pred_lat, pred_lon, bearing)

    def _compute_linking_cost(self, track, det, time_diff_h):
        """
        Compute a weighted cost for linking a detection to a track.
        For young tracks (<3 real points): pure distance + MSLP bonus.
        For mature tracks (>=3 real points): adds prediction + bearing.
        Returns np.inf if link is physically impossible.
        """
        last = track['detections'][-1]
        ref_lat = last['center_lat']
        ref_lon = last['center_lon']

        # Hard filter
        if not self.is_realistic_hurricane_movement(
            ref_lat, ref_lon,
            det['center_lat'], det['center_lon'],
            time_diff_h
        ):
            return np.inf

        dist = self.haversine_distance(
            ref_lat, ref_lon,
            det['center_lat'], det['center_lon']
        )
        if dist > self.max_distance_km:
            return np.inf

        # Base cost: distance from last position
        cost = dist

        # Count real (non-interpolated) detections
        n_real = sum(1 for d in track['detections'] if not d.get('_interpolated', False))

        # Only apply prediction + bearing for mature tracks (>=3 real points)
        if n_real >= 3:
            pred = self._predict_position(track['detections'])
            if pred is not None:
                pred_lat, pred_lon, expected_bearing = pred
                pred_dist = self.haversine_distance(
                    pred_lat, pred_lon,
                    det['center_lat'], det['center_lon']
                )
                # Blend: distance + prediction
                cost = 0.5 * dist + 0.5 * pred_dist

                # Bearing deviation penalty (moderate)
                actual_bearing = self.calculate_bearing(
                    ref_lon, ref_lat,
                    det['center_lon'], det['center_lat']
                )
                bearing_diff = abs(actual_bearing - expected_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff

                # Only penalise strong reversals (>120°)
                if bearing_diff > 120:
                    cost *= 1.5
                elif bearing_diff > 90:
                    cost *= 1.2

        # MSLP bonus: prefer lower pressure (secondary)
        det_mslp = det.get('mslp') or None
        if det_mslp is not None and det_mslp < 101500:
            mslp_factor = max(0.85, 1.0 - (101500 - det_mslp) / 100000.0)
            cost *= mslp_factor

        return cost

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
                hurricanes = self.extract_hurricane_centers(hurricane_mask, lats, lons)
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
    
    def validate_hurricane_track(self, track_detections):
        """
        STRICT validation matching IBTrACS comparison criteria
        """
        if len(track_detections) < 8:
            return False

        # ============================
        # 0a. GHOST FRACTION CHECK — reject tracks dominated by interpolated points
        # ============================
        n_ghost = sum(1 for d in track_detections if d.get('_interpolated', False))
        ghost_frac = n_ghost / len(track_detections)
        if ghost_frac > 0.40:  # more than 40% interpolated = unreliable
            return False

        # ============================
        # 0b. MAX SINGLE-STEP DISTANCE — reject any step > 1200 km (catches dateline jumps)
        # ============================
        for i in range(1, len(track_detections)):
            step_dist = self.haversine_distance(
                track_detections[i-1]['center_lat'], track_detections[i-1]['center_lon'],
                track_detections[i]['center_lat'], track_detections[i]['center_lon']
            )
            if step_dist > 1200.0:  # no real hurricane moves >1200 km in 6h
                return False

        # ============================
        # 0c. CONSECUTIVE GHOST CHECK — reject >2 consecutive interpolated points
        # ============================
        consec_ghost = 0
        max_consec_ghost = 0
        for d in track_detections:
            if d.get('_interpolated', False):
                consec_ghost += 1
                max_consec_ghost = max(max_consec_ghost, consec_ghost)
            else:
                consec_ghost = 0
        if max_consec_ghost > 2:
            return False

        # ============================
        # 1. OCEAN GENESIS REQUIREMENT
        # ============================
        if not self.requires_ocean_genesis(track_detections, min_ocean_points=2):
            return False
        
        # ============================
        # 2. FORMATION ZONE - Must start in tropics/subtropics
        # ============================
        first_lat = track_detections[0]['center_lat']
        first_lon = track_detections[0]['center_lon']
        
        # Moderate: allow subtropical formation up to 38 degrees
        if not (-38.0 <= first_lat <= 38.0):
            return False
        
        # NEW: Exclude tracks starting north of 35N AND west of 80W (land-based systems)
        if first_lat >= 35.0 and first_lon <= -80.0:
            return False
        
        # ============================
        # 3. LATITUDE CONSTRAINTS
        # ============================
        # Allow more time at higher latitudes for recurving storms
        high_lat_count = sum(1 for d in track_detections if abs(d['center_lat']) > 45.0)
        if high_lat_count / len(track_detections) > 0.50:
            return False
        
        # NEW: Reject tracks north of 40N AND west of 110W (continental systems)
        for detection in track_detections:
            if detection['center_lat'] >= 40.0 and detection['center_lon'] <= -110.0:
                return False
        
        # ============================
        # 4. OCEAN PERCENTAGE - Match comparison code
        # ============================
        ocean_count = sum(1 for d in track_detections 
                        if self.is_over_ocean(d['center_lat'], d['center_lon']))
        ocean_percentage = ocean_count / len(track_detections)
        
        # Allow hurricanes that persist over land after landfall
        if ocean_percentage < 0.50:
            return False
        
        # ============================
        # 5. DURATION CHECK - Match IBTrACS minimum
        # ============================
        start_time = track_detections[0]['timestamp']
        end_time = track_detections[-1]['timestamp']
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        
        # IBTrACS standard minimum
        if duration_hours < 48.0:
            return False
        
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
        
        # if total_distance < 350.0:  # Slightly lower for short-lived but real TCs
        #     return False
        
        # ============================
        # 7. MOVEMENT VALIDATION
        # ============================
        # Check for west-of-108W then northeast movement (continental pattern)
        seen_west108 = False
        for i, detection in enumerate(track_detections):
            lon = detection['center_lon']
            lat = detection['center_lat']
            
            if lon <= -108.0:
                seen_west108 = True
            
            # If seen west of 108W, then moved northeast of 100W and above 35N
            if seen_west108 and lat >= 35.0 and lon >= -100.0:
                return False  # Continental system pattern
        
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
                if speed_ms < 0.5 or speed_ms > 32.0:
                    unrealistic_movements += 1
        
        if speeds and unrealistic_movements / len(speeds) > 0.3:
            return False
        
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
        
        if (east_fraction > 0.5 and entered_band and 
            not touched_ocean_in_band and not start_ocean_east82):
            return False
        
        # ============================
        # 10. INTENSITY CHECK
        # ============================
        intensities = [d.get('max_intensity', 0) for d in track_detections]
        max_intensity = max(intensities) if intensities else 0
        
        if max_intensity < 0.50:
            return False
        
        # ============================
        # 11. NET DISPLACEMENT - Progressive movement
        # ============================
        start_lat, start_lon = track_detections[0]['center_lat'], track_detections[0]['center_lon']
        end_lat, end_lon = track_detections[-1]['center_lat'], track_detections[-1]['center_lon']
        
        net_displacement = self.haversine_distance(start_lat, start_lon, end_lat, end_lon)
        
        # Net displacement should be at least 20% of total path
        if total_distance > 0:
            displacement_ratio = net_displacement / total_distance
            if displacement_ratio < 0.20:
                return False
        
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
                if bearing_change > 120:  # Sharp reversal
                    sharp_turns += 1
            
            # If >30% of segments have sharp reversals, it's noise
            if len(bearings) > 1 and sharp_turns / (len(bearings) - 1) > 0.30:
                return False
        
        return True
    
    def create_hurricane_tracks(self, df):
        """
        Improved hurricane track building with optimal assignment.

        Key improvements over greedy linking:
        1. Global optimal assignment (Hungarian algorithm) prevents
           tracks from stealing each other's detections.
        2. Velocity-based prediction penalises candidates that deviate
           from the track's recent direction of motion.
        3. Bearing consistency penalty prevents zigzag links.
        4. MSLP is a secondary bonus, not the primary criterion.
        5. Relaxed gap-fill angle (45°) and 2-step gap window.
        """
        if df.empty:
            return []

        df = df.sort_values('timestamp').reset_index(drop=True)
        unique_times = sorted(df['timestamp'].unique())

        time_to_detections = {}
        for t in unique_times:
            time_to_detections[t] = df[df['timestamp'] == t].to_dict('records')

        active_tracks  = []
        finished_tracks = []
        track_id_counter = [0]

        def close_track(tr):
            dets = tr['detections']
            if len(dets) < 2:
                return

            duration = (dets[-1]['timestamp'] - dets[0]['timestamp']).total_seconds() / 3600.0

            if duration < self.min_duration_hours:
                return

            wind_speeds = [d.get('wind_speed') for d in dets if d.get('wind_speed') is not None]
            if wind_speeds:
                if max(wind_speeds) < 10.0:
                    return
            else:
                intensities = [d.get('max_intensity', 0) for d in dets]
                if max(intensities) < 0.50:
                    return

            if not self.validate_hurricane_track(dets):
                return

            finished_tracks.append({
                'track_id'      : track_id_counter[0],
                'detections'    : dets,
                'duration_hours': duration,
                'start_time'    : dets[0]['timestamp'],
                'end_time'      : dets[-1]['timestamp'],
            })
            track_id_counter[0] += 1

        # ------------------------------------------------------------------ #
        # Main loop — global optimal assignment per timestep                   #
        # ------------------------------------------------------------------ #
        BIG_COST = 1e9  # sentinel for impossible links

        for t_idx, current_time in enumerate(unique_times):
            current_detections = list(time_to_detections.get(current_time, []))
            n_det = len(current_detections)

            # ============================================================== #
            # STAGE 1: Build cost matrix (tracks × detections) and solve      #
            #          globally optimal assignment with Hungarian algorithm    #
            # ============================================================== #
            tracks_to_assign = []  # indices into active_tracks that need linking
            skip_tracks = []       # tracks whose last timestamp == current (no linking needed)

            for tr_i, tr in enumerate(active_tracks):
                last_time = tr['detections'][-1]['timestamp']
                time_diff_h = (current_time - last_time).total_seconds() / 3600.0
                if time_diff_h <= 0:
                    skip_tracks.append(tr_i)
                else:
                    tracks_to_assign.append(tr_i)

            n_tracks = len(tracks_to_assign)
            claimed = [False] * n_det
            matched_track_indices = set()

            if n_tracks > 0 and n_det > 0:
                # Build cost matrix
                cost_matrix = np.full((n_tracks, n_det), BIG_COST)

                for row, tr_i in enumerate(tracks_to_assign):
                    tr = active_tracks[tr_i]
                    last_time = tr['detections'][-1]['timestamp']
                    time_diff_h = (current_time - last_time).total_seconds() / 3600.0

                    for col, det in enumerate(current_detections):
                        c = self._compute_linking_cost(tr, det, time_diff_h)
                        cost_matrix[row, col] = c

                # Replace any np.inf with BIG_COST (linear_sum_assignment
                # cannot handle actual infinity values).
                cost_matrix = np.where(np.isinf(cost_matrix), BIG_COST, cost_matrix)

                # Pad to square so Hungarian algorithm is always feasible.
                nR, nC = cost_matrix.shape
                size = max(nR, nC)
                padded = np.full((size, size), BIG_COST)
                padded[:nR, :nC] = cost_matrix
                row_ind, col_ind = linear_sum_assignment(padded)

                for r, c in zip(row_ind, col_ind):
                    if r >= n_tracks or c >= n_det:
                        continue  # skip dummy assignments
                    if cost_matrix[r, c] < BIG_COST:
                        tr_i = tracks_to_assign[r]
                        det = current_detections[c]
                        active_tracks[tr_i]['detections'].append(det)
                        active_tracks[tr_i]['gap_count'] = 0
                        claimed[c] = True
                        matched_track_indices.add(tr_i)

            # ============================================================== #
            # STAGE 2: Handle unmatched tracks — gap fill or close            #
            # ============================================================== #
            still_active = []

            for tr_i, tr in enumerate(active_tracks):
                if tr_i in matched_track_indices or tr_i in skip_tracks:
                    still_active.append(tr)
                    continue

                # This track had no match at current timestep
                new_gap = tr['gap_count'] + 1

                if new_gap > self.max_gap_steps:
                    close_track(tr)
                    continue

                # ------------------------------------------------------ #
                # Gap-fill: look ahead up to max_gap_steps future steps   #
                # ------------------------------------------------------ #
                last_det = tr['detections'][-1]
                ref_lat  = last_det['center_lat']
                ref_lon  = last_det['center_lon']
                ref_time = last_det['timestamp']
                gap_filled = False

                for look_ahead in range(1, self.max_gap_steps + 1):
                    future_idx = t_idx + look_ahead
                    if future_idx >= len(unique_times):
                        break
                    future_time = unique_times[future_idx]
                    future_dets = list(time_to_detections.get(future_time, []))
                    gap_time_diff_h = (future_time - ref_time).total_seconds() / 3600.0

                    # Use prediction-aware cost for gap fill too
                    gap_best_idx  = None
                    gap_best_cost = np.inf

                    for d_idx, det in enumerate(future_dets):
                        dist = self.haversine_distance(
                            ref_lat, ref_lon,
                            det['center_lat'], det['center_lon']
                        )
                        if dist > (1 + look_ahead) * self.max_distance_km:
                            continue
                        if not self.is_realistic_hurricane_movement(
                            ref_lat, ref_lon,
                            det['center_lat'], det['center_lon'],
                            gap_time_diff_h
                        ):
                            continue

                        # Use prediction cost if track has history
                        pred = self._predict_position(tr['detections'])
                        if pred is not None:
                            pred_lat, pred_lon, _ = pred
                            # Scale prediction to the gap duration
                            scale = gap_time_diff_h / 6.0
                            adj_pred_lat = last_det['center_lat'] + (pred_lat - last_det['center_lat']) * scale
                            adj_pred_lon = last_det['center_lon'] + (pred_lon - last_det['center_lon']) * scale
                            pred_dist = self.haversine_distance(
                                adj_pred_lat, adj_pred_lon,
                                det['center_lat'], det['center_lon']
                            )
                            cost = 0.4 * dist + 0.6 * pred_dist
                        else:
                            cost = dist

                        if cost < gap_best_cost:
                            gap_best_cost = cost
                            gap_best_idx  = d_idx

                    if gap_best_idx is not None:
                        far_det = future_dets[gap_best_idx]

                        # Trajectory angle check
                        # Dateline-aware midpoint for longitude
                        mid_lat = (ref_lat + far_det['center_lat']) / 2.0
                        dlon_gap = far_det['center_lon'] - ref_lon
                        if dlon_gap > 180:  dlon_gap -= 360
                        if dlon_gap < -180: dlon_gap += 360
                        mid_lon = ref_lon + dlon_gap / 2.0
                        if mid_lon > 180:  mid_lon -= 360
                        if mid_lon < -180: mid_lon += 360

                        angle_ok = self._check_trajectory_angle(
                            ref_lat, ref_lon,
                            mid_lat, mid_lon,
                            far_det['center_lat'], far_det['center_lon']
                        )

                        if angle_ok:
                            # Insert interpolated ghost points for missing steps
                            # Use dateline-aware longitude interpolation
                            n_missing = look_ahead
                            dlon_total = far_det['center_lon'] - ref_lon
                            if dlon_total > 180:  dlon_total -= 360
                            if dlon_total < -180: dlon_total += 360

                            for step in range(1, n_missing + 1):
                                frac = step / (n_missing + 1)
                                ghost_lat = ref_lat + frac * (far_det['center_lat'] - ref_lat)
                                ghost_lon = ref_lon + frac * dlon_total
                                if ghost_lon > 180:  ghost_lon -= 360
                                if ghost_lon < -180: ghost_lon += 360
                                ghost_time = ref_time + timedelta(hours=6.0 * step)
                                ghost_det = {
                                    'timestamp'     : ghost_time,
                                    'center_lat'    : ghost_lat,
                                    'center_lon'    : ghost_lon,
                                    'area_pixels'   : last_det.get('area_pixels', 0),
                                    'max_intensity' : last_det.get('max_intensity', 0),
                                    'mean_intensity': last_det.get('mean_intensity', 0),
                                    'vorticity'     : None,
                                    'mslp'          : None,
                                    'wind_speed'    : None,
                                    '_interpolated' : True,
                                }
                                tr['detections'].append(ghost_det)

                            tr['gap_count'] = new_gap
                            still_active.append(tr)
                            gap_filled = True
                            break  # stop looking further ahead

                if not gap_filled:
                    # Keep track alive if within gap window, else close
                    if new_gap <= self.max_gap_steps:
                        tr['gap_count'] = new_gap
                        still_active.append(tr)
                    else:
                        close_track(tr)

            active_tracks = still_active

            # Start new tracks from unclaimed detections
            for d_idx, det in enumerate(current_detections):
                if claimed[d_idx]:
                    continue
                active_tracks.append({
                    'detections': [det],
                    'gap_count' : 0,
                })

        # Close all remaining tracks
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
    hurricane_data_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_24"
    era5_data_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
    vorticity_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/vor850_latlongrid/vor850_latlongrid/"
    wind_speed_path = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc"
    output_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_24/Hurricanes_track_files_final_v2"
    # Create hurricane tracker for CONUS region
    tracker = HurricaneTracker(
        hurricane_data_dir=hurricane_data_dir,
        era5_data_dir=era5_data_dir,
        vorticity_dir=vorticity_dir,
        wind_speed_path=wind_speed_path,
        output_dir=output_dir
    )
    
    # You can specify year range here
    start_year = 1980  # Adjust as needed
    end_year = 2024  # Adjust as needed - start with a small range for testing
    
    # Run tracking for CONUS region
    tracker.run_tracking(start_year=start_year, end_year=end_year)
if __name__ == "__main__":
    main()