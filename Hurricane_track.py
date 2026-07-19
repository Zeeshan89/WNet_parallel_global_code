
## python Hurricane_track.py --model CNR --start-year 1980 --end-year 2014

import os
import glob
import numpy as np
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
    def __init__(self, hurricane_data_dir, era5_data_dir, vorticity_dir, wind_speed_path, output_dir,
                 file_prefix="era5dara_", inline_met_fields=False,
                 mask_to_domain=False, domain_erode_cells=2):
        """
        Initialize Hurricane Tracker for CONUS region

        Args:
            hurricane_data_dir: Directory containing hurricane detection NetCDF files
            era5_data_dir: Directory containing the paired input files (for MSLP, and
                for RegCM also vorticity / wind_speed — see inline_met_fields)
            vorticity_dir: Directory of separate yearly vorticity files (ERA5 layout).
                Pass None for RegCM, where vorticity is inline in the paired file.
            wind_speed_path: Path to a single wind-speed file (ERA5 layout). Pass None
                for RegCM, where wind_speed is inline in the paired file.
            output_dir: Directory to save tracking results
            file_prefix: detection/paired filename prefix. "era5dara_" for ERA5,
                "<MODEL>dara_" for a RegCM model (e.g. "ACCdara_").
            inline_met_fields: if True, read vorticity ('vorticity') and wind speed
                ('wind_speed') from the paired file itself instead of the separate
                ERA5 dirs. This is the RegCM layout (WNet_6hourly_latlon files carry
                MSLP/vorticity/wind_speed together). Detection masks are unchanged by
                the precip-MCS work, so hurricane tracking is identical across runs.
            mask_to_domain: if True, zero the detection field wherever the driving
                MSLP is NaN, i.e. outside RegCM's curved Lambert domain, before any
                blob is formed. REQUIRED for RegCM — without it ~45% of all track
                points are boundary artifacts pinned to lat 18.75. See _domain_mask.
                Leave False for ERA5, which is global and has no such boundary.
            domain_erode_cells: additionally drop cells within this many grid cells
                of the domain boundary, where the CNN's zero-padding response reaches
                inside the valid region. 0 disables the erosion.
        """
        self.hurricane_data_dir = hurricane_data_dir
        self.era5_data_dir = era5_data_dir
        self.vorticity_dir = vorticity_dir
        self.wind_speed_path = wind_speed_path
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.inline_met_fields = inline_met_fields
        self.mask_to_domain = mask_to_domain
        self.domain_erode_cells = domain_erode_cells
        self._domain_mask_cache = None
        
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
        
        # CONUS region boundaries (hurricanes affecting Continental US).
        # For RegCM these are narrowed to the model's own lat/lon box: the global
        # +/-180 / +/-45 defaults are far wider than the data, which let the tracker
        # accept detections right on the domain edge. The domain mask (mask_to_domain)
        # is the load-bearing fix; this is the coarse outer guard.
        if mask_to_domain:
            self.region_bounds = {
                'lon_min': -138.75, 'lon_max': -57.0,
                'lat_min': 18.0,    'lat_max': 56.75,
            }
        else:
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
        distance = self.haversine_distance(lon1, lat1, lon2, lat2)
        
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
    def _domain_mask(self, mslp_data):
        """True where the simulation actually exists, eroded by `domain_erode_cells`.

        WHY THIS EXISTS — it removes ~half of all raw TC track points as artifacts.

        RegCM runs on a curved Lambert grid inscribed in a lat/lon box, so ~28% of
        the box has no simulated atmosphere and is NaN in every driving field. The
        WNet nonetheless emits detections there, because the tensors were built with
        fillna(0.0) and zero-padding convolutions produce a boundary response.

        Measured on ACC (before this fix):
          - detection fraction 9.1% in the southern rows vs 2.1% in the interior,
            a 4.4x edge excess spanning ~7 rows (lat 18.0-19.5)
          - 20,378 of 44,924 track points (45%) pinned to the single row lat=18.75
          - 24 of 64 tracks in 2005 lay ENTIRELY outside the domain
          - raw counts 59-71 tracks/yr against ~20/yr in IBTrACS

        test.py::clean_edge_artifacts cannot fix this: it is written for a GLOBAL
        periodic field, where the first/last rows are poles. On this regional domain
        those rows are the box edge, and its "overwrite the edge with its interior
        neighbour" rule PROPAGATES the artifact outward when the neighbour is itself
        contaminated — which is why rows 0-2 (lat 18.00/18.25/18.50) are bit-identical
        in the detection files. Its pole_rows=2 is also far shallower than the ~7-row
        excess. (The same global-field assumption appears in
        post_processing._label_lonwrap, which unions column 0 with column -1 and so
        merges Oregon with the Carolinas.)

        Masking on the driving field's own NaN pattern is preferable to trimming a
        fixed number of edge rows: it follows the true curved boundary exactly and
        needs no arbitrary constant. The erosion additionally drops cells within
        `domain_erode_cells` of the boundary, where the CNN's padding response
        reaches inside the valid region.
        """
        if self._domain_mask_cache is not None:
            return self._domain_mask_cache
        valid = ~np.isnan(np.asarray(mslp_data.isel(time=0).values))
        if self.domain_erode_cells > 0:
            valid = ndimage.binary_erosion(
                valid, structure=np.ones((3, 3), dtype=bool),
                iterations=self.domain_erode_cells, border_value=0)
        self._domain_mask_cache = valid
        print(f"  domain mask: {100.0 * valid.mean():.1f}% of the lat/lon box is "
              f"inside the simulation (eroded {self.domain_erode_cells} cells)")
        return valid

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
                    if max_intensity < 0.40:  # Low to capture weak/forming phases
                        continue
                    
                    # Check 3: Reasonable area (not too small, not suspiciously large)
                    if area_pixels < 3 or area_pixels > 500:  # Keep low for detection continuity
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
        hurricane_pattern = os.path.join(self.hurricane_data_dir, f"{self.file_prefix}{year:04d}*Hurricane_detection.nc")
        hurricane_files = sorted(glob.glob(hurricane_pattern))
        
        if not hurricane_files:
            print(f"No hurricane detection files found for year {year}")
            return
        
        print(f"Found {len(hurricane_files)} hurricane detection files for {year}")
        
        # Load vorticity data for this year (ERA5 layout only). For RegCM
        # (inline_met_fields) vorticity comes from the paired file per-timestep.
        vorticity_ds = None
        if not self.inline_met_fields and self.vorticity_dir:
            vorticity_file = self.find_vorticity_file(year)
            if vorticity_file:
                print(f"Loading vorticity data: {os.path.basename(vorticity_file)}")
                vorticity_ds = xr.open_dataset(vorticity_file).squeeze('plev')
                vorticity_ds = vorticity_ds.assign_coords(
                    longitude=(((vorticity_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')

        # Load wind speed data (ERA5 layout only). RegCM reads it inline below.
        wind_speed_ds = None
        if not self.inline_met_fields and self.wind_speed_path and os.path.exists(self.wind_speed_path):
            print("Loading wind speed data...")
            wind_speed_ds = xr.open_dataset(self.wind_speed_path).squeeze('level')
            wind_speed_ds = wind_speed_ds.assign_coords(
                longitude=(((wind_speed_ds.longitude + 180) % 360) - 180)
            ).sortby('longitude')
        
        # Initialize tracking data
        all_detections = []
        conus_detections_count = 0
        total_detections_count = 0
        masked_out_points = 0        # detections dropped as outside the domain
        
        # Process each file
        for file_idx, hurricane_file in enumerate(hurricane_files):
            print(f"Processing file {file_idx + 1}/{len(hurricane_files)}: {os.path.basename(hurricane_file)}")
            
            # Load hurricane detection data
            hurricane_ds = xr.open_dataset(hurricane_file)
            
            # Load corresponding ERA5 data for MSLP
            era5_file = hurricane_file.replace('_Hurricane_detection.nc', '.nc')
            era5_file = era5_file.replace(self.hurricane_data_dir, self.era5_data_dir)
            
            mslp_data = None
            vort_month = None       # RegCM inline vorticity for this month
            wind_month = None       # RegCM inline wind speed for this month
            domain_valid = None     # True where the simulation actually exists
            if os.path.exists(era5_file):
                era5_ds = xr.open_dataset(era5_file)
                era5_ds = era5_ds.assign_coords(
                    longitude=(((era5_ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')
                if 'MSLP' in era5_ds:
                    mslp_data = era5_ds['MSLP']
                    if self.mask_to_domain:
                        domain_valid = self._domain_mask(mslp_data)
                # RegCM: vorticity + wind_speed live inline in the paired file
                # (renamed vo850->vorticity, wspd850->wind_speed by the CDO step),
                # mirroring how the WNet dataset loaders read them.
                if self.inline_met_fields:
                    if 'vorticity' in era5_ds:
                        vort_month = era5_ds['vorticity'].load()
                    if 'wind_speed' in era5_ds:
                        wind_month = era5_ds['wind_speed'].load()
                era5_ds.close()
            
            # Process each timestep
            for time_idx, timestamp in enumerate(hurricane_ds.time.values):
                hurricane_mask = hurricane_ds['Hurricane_detection'].values[time_idx]
                lats = hurricane_ds.latitude.values
                lons = hurricane_ds.longitude.values

                # Drop detections outside the simulated domain BEFORE any blob is
                # formed, so spurious edge cells can neither seed a track nor drag a
                # centroid. See _domain_mask for the measurements motivating this.
                if domain_valid is not None:
                    hurricane_mask = np.where(domain_valid, hurricane_mask, 0.0)
                    masked_out_points += int(np.sum(
                        (hurricane_ds['Hurricane_detection'].values[time_idx] > 0.5)
                        & ~domain_valid))

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
                        # Get vorticity data for this timestamp.
                        # ERA5: separate yearly file (var138). RegCM: inline 'vorticity'
                        # from the paired monthly file loaded above.
                        vor_data = None
                        if vorticity_ds is not None:
                            try:
                                vor_time_data = vorticity_ds.sel(time=timestamp, method='nearest')
                                vor_data = vor_time_data['var138'].values
                            except:
                                vor_data = None
                        elif vort_month is not None:
                            try:
                                vor_data = vort_month.sel(time=timestamp, method='nearest').values
                            except:
                                vor_data = None
                        
                        # Get MSLP data for this timestamp
                        mslp_time_data = None
                        if mslp_data is not None:
                            try:
                                mslp_time_data = mslp_data.sel(time=timestamp, method='nearest').values
                            except:
                                mslp_time_data = None
                        
                        # Get wind speed data for this timestamp.
                        # ERA5: single file (wspd). RegCM: inline 'wind_speed'.
                        wind_time_data = None
                        if wind_speed_ds is not None:
                            try:
                                wind_time_data = wind_speed_ds.sel(time=timestamp, method='nearest')['wspd'].values
                            except:
                                wind_time_data = None
                        elif wind_month is not None:
                            try:
                                wind_time_data = wind_month.sel(time=timestamp, method='nearest').values
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
        if self.mask_to_domain:
            print(f"  - Dropped as outside the simulated domain: {masked_out_points}")
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
        if len(track_detections) < 5:
            return False
        
        # ============================
        # 1. OCEAN GENESIS REQUIREMENT
        # ============================
        if not self.requires_ocean_genesis(track_detections, min_ocean_points=1):
            return False
        
        # ============================
        # 2. FORMATION ZONE - Must start in tropics/subtropics
        # ============================
        first_lat = track_detections[0]['center_lat']
        first_lon = track_detections[0]['center_lon']
        
        # Moderate: allow subtropical formation up to 38 degrees
        if not (-38.0 <= first_lat <= 38.0):
            return False
        
        # NEW: Exclude tracks starting north of 36N AND west of 80W (land-based systems)
        if first_lat >= 36.0 and first_lon <= -85.0:
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
        if ocean_percentage < 0.35:
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
        
        if total_distance < 250.0:  # Slightly lower for short-lived but real TCs
            return False
        
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
                return False
            
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
        
        start_in_gulf = (first_lon <= -82.0 and self.is_over_ocean(first_lat, first_lon))
        if (east_fraction > 0.5 and entered_band and 
            not touched_ocean_in_band and not start_ocean_east82 and not start_in_gulf):
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
            if displacement_ratio < 0.12:
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

                if bearing_change > 140:  # Sharp reversal
                    sharp_turns += 1
            
            # If >30% of segments have sharp reversals, it's noise
            if len(bearings) > 1 and sharp_turns / (len(bearings) - 1) > 0.40:
                return False
        
        return True
    def create_hurricane_tracks(self, df, max_gap_steps=4):
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
        def close_track(tr):
            """Move a track from active → finished after validation."""
            dets = tr['detections']
            if len(dets) < 5:
                return
            duration = (dets[-1]['timestamp'] - dets[0]['timestamp']).total_seconds() / 3600.0
            if duration < 48.0:
                return
            if self.validate_hurricane_track(dets):
                finished_tracks.append({
                    'track_id'      : track_id_counter[0],
                    'detections'    : dets,
                    'duration_hours': duration,
                    'start_time'    : dets[0]['timestamp'],
                    'end_time'      : dets[-1]['timestamp'],
                })
                track_id_counter[0] += 1
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
                    if new_gap > max_gap_steps:
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
    Run CONUS hurricane tracking for a RegCM model and write per-year .txt tracks.

    Hurricane detection masks are UNCHANGED by the precipitation-MCS work
    (channels 0-10 are bit-identical), so this reads the same detection files the
    precip test run emits — <MODEL>dara_YYYYMM_Hurricane_detection.nc — and needs
    no re-inference for hurricanes. vorticity + wind_speed are read INLINE from the
    paired WNet dara file (RegCM layout), so no separate ERA5 aux dirs are used.

    Usage:
        REGCM_MODEL=ACC python Hurricane_track.py
        python Hurricane_track.py --model BCC --start-year 2005 --end-year 2014
        python Hurricane_track.py --model ACC \
            --hurricane-dir .../Outputs_RegCM_PR_ACC   # override detection source
    """
    KNOWN = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]
    p = argparse.ArgumentParser(description="RegCM hurricane tracking -> per-year .txt")
    p.add_argument("--model", default=os.environ.get("REGCM_MODEL", "ACC"),
                   help=f"RegCM model ({'/'.join(KNOWN)}). Default $REGCM_MODEL or ACC.")
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2014)
    p.add_argument("--hurricane-dir", default=None,
                   help="Detection dir (default: ARs_Hurricanes_ETCs_MCS/Outputs_RegCM_PR_<MODEL>)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write tracks (default: <hurricane-dir>/Hurricane_tracks)")
    args = p.parse_args()
    model = args.model

    HERE       = os.path.dirname(os.path.abspath(__file__))
    REGCM_BASE = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"

    # Detection masks from the precip test run; paired inputs (MSLP/vorticity/
    # wind_speed inline) from the regridded WNet files.
    hurricane_data_dir = args.hurricane_dir or os.path.join(
        HERE, "ARs_Hurricanes_ETCs_MCS", f"Outputs_RegCM_PR_{model}")
    era5_data_dir = os.path.join(REGCM_BASE, model, "WNet_6hourly_latlon")
    output_dir = args.output_dir or os.path.join(hurricane_data_dir, "Hurricane_tracks")

    if not os.path.isdir(hurricane_data_dir):
        raise SystemExit(f"Detection dir not found: {hurricane_data_dir}\n"
                         f"  (has {model} been tested? see Run_test_distributed.sh)")

    tracker = HurricaneTracker(
        hurricane_data_dir=hurricane_data_dir,
        era5_data_dir=era5_data_dir,
        vorticity_dir=None,          # RegCM: vorticity is inline in the paired file
        wind_speed_path=None,        # RegCM: wind_speed is inline in the paired file
        output_dir=output_dir,
        file_prefix=f"{model}dara_",
        inline_met_fields=True,
        # RegCM is a regional domain: mask detections to where the simulation
        # actually exists, otherwise ~45% of track points are boundary artifacts.
        mask_to_domain=True,
        domain_erode_cells=int(os.environ.get("TC_DOMAIN_ERODE", 2)),
    )
    print(f"\n=== RegCM {model} hurricane tracking {args.start_year}-{args.end_year} ===")
    tracker.run_tracking(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()