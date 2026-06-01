import numpy as np
import xarray as xr
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import glob


# =============================================================================
# MOUNTAIN / OROGRAPHIC MASK
# =============================================================================
# ETCs rarely form over or persist over high terrain.
# These boxes are defined as (lon_min, lon_max, lat_min, lat_max, elev_label).
# Any detected center whose (lon, lat) falls inside is discarded.
#
# Key regions:
#   - Rocky Mountains / SW USA (the main offender you described)
#   - Tibetan Plateau (massive orographic low factory)
#   - Andes (strong orographic lows, not synoptic ETCs)
#   - Greenland Ice Sheet
#   - Antarctic continent (not relevant here but safe to include)
#
# TUNING TIP: if a real ETC occasionally passes over the edge of a box,
# narrow the box.  If orographic noise still leaks, widen it.
# =============================================================================
MOUNTAIN_MASK_BOXES = [
    # ---------- Rocky Mountains / SW USA (your primary complaint) ----------
    # Core Rockies – Colorado, Utah, Wyoming, Nevada high terrain
    (-115.0, -100.0, 35.0, 48.0),
    # Sierra Nevada + Great Basin
    (-122.0, -112.0, 36.0, 42.0),
    # Mexican Plateau / Sierra Madre
    (-108.0,  -95.0, 18.0, 32.0),

    # ---------- Tibetan Plateau ----------
    (70.0, 105.0, 28.0, 40.0),

    # ---------- Andes ----------
    (-80.0, -62.0, -55.0, 10.0),

    # ---------- Greenland ----------
    (-55.0, -15.0, 60.0, 85.0),

    # ---------- Alps / Pyrenees (small – true ETCs pass through, orographic
    #             lows stall; keep box tight so real tracks aren't blocked) ----------
    (  5.0,  16.0, 44.0, 48.0),   # Alps core
    ( -2.0,   3.0, 42.0, 44.0),   # Pyrenees core

    # ---------- East African Highlands ----------
    ( 33.0,  42.0, -5.0, 15.0),
]


def point_in_mountain_mask(lon, lat):
    """Return True if (lon, lat) is inside any mountain mask box."""
    for (lo_min, lo_max, la_min, la_max) in MOUNTAIN_MASK_BOXES:
        if lo_min <= lon <= lo_max and la_min <= lat <= la_max:
            return True
    return False


# =============================================================================
# MOUNTAIN FRACTION FILTER FOR TRACKS
# =============================================================================
# A track is rejected if more than MOUNTAIN_TRACK_FRAC of its points lie
# inside mountain boxes.  This catches tracks that spend most of their life
# over terrain but briefly wander into clear sky at start/end.
MOUNTAIN_TRACK_FRAC = 0.30   # >30 % of points over terrain → reject


class ETCTracker:
    def __init__(self, max_distance_km=800, min_track_length=8, region_bounds=None):
        """
        Initialize meteorologically-aware ETC tracker

        Parameters:
        max_distance_km: Maximum distance (km) between consecutive ETC positions
        min_track_length: Minimum number of points required for a valid track
        region_bounds: Dictionary with regional constraints
        """
        self.max_distance_km = max_distance_km
        self.min_track_length = min_track_length
        self.earth_radius = 6371.0

        # ── Speed limits ────────────────────────────────────────────────────
        self.max_speed_kmh = 28.0 * 6    # 28 m/s → km per 6 h = 604 km/6h
        self.min_speed_kmh = 2.0 * 6     #  2 m/s →  12 km/6h

        # ── Track quality thresholds ────────────────────────────────────────
        # Raised from 1500 → 1800 km  (wobbling lows rarely travel this far)
        self.min_path_km = 1800.0

        # Minimum 48-hour lifetime
        self.min_hours = 48.0

        # ── Path efficiency (NEW tighter value) ─────────────────────────────
        # total_path / duration must exceed this.  Wobbling lows that rack up
        # 1800 km over 12 days average only ~1.7 m/s; real ETCs ~6-15 m/s.
        # Raised from 2.5 → 3.5 m/s.
        self.min_avg_propagation_ms = 3.5

        # ── Median speed floor (raised from 4.0 → 5.0 m/s) ─────────────────
        self.min_median_speed_ms = 5.0
        self.max_median_speed_ms = 28.0

        # ── Retrograde motion ────────────────────────────────────────────────
        self.west_sector = (210.0, 330.0)
        self.max_retro_run_hours = 18.0      # tighter: was 24 h
        self.max_retro_total_frac = 0.18     # tighter: was 0.20
        self.require_net_east = True
        # Raised from 300 → 500 km
        self.min_net_east_km = 500.0

        # ── Linking weights ──────────────────────────────────────────────────
        self.preferred_direction_weight = 2.8
        self.direction_consistency_weight = 1.8
        self.time_gap_hours = 6

        # ── Gap bridging ─────────────────────────────────────────────────────
        self.max_gap_steps = 1   # allow 1 missed 6-h step

        # ── Spatial filter ───────────────────────────────────────────────────
        # ETCs only in mid/high latitudes (kept at 28°, consistent with v1)
        self.min_abs_lat = 28.0

        # ── Minimum detection area ───────────────────────────────────────────
        # Raised from 12 → 15 pixels
        self.min_area = 15

        # ── Regional bounds ──────────────────────────────────────────────────
        if region_bounds is None:
            self.region_bounds = {
                'lon_min': -180.0, 'lon_max': 0.0,
                'lat_min': 20.0,   'lat_max': 70.0
            }
        else:
            self.region_bounds = region_bounds

        self.track_both_hemispheres = (self.region_bounds['lat_min'] < 0 and
                                        self.region_bounds['lat_max'] > 0)

        print(f"Meteorological ETC Tracker v2 initialized:")
        print(f"  Regional bounds: Lon [{self.region_bounds['lon_min']}°, "
              f"{self.region_bounds['lon_max']}°], "
              f"Lat [{self.region_bounds['lat_min']}°, {self.region_bounds['lat_max']}°]")
        print(f"  Tracking both hemispheres: {self.track_both_hemispheres}")
        print(f"  Max speed: {self.max_speed_kmh:.0f} km per 6h")
        print(f"  Max link distance: {self.max_distance_km} km")
        print(f"  Min path: {self.min_path_km} km")
        print(f"  Min avg propagation: {self.min_avg_propagation_ms} m/s")
        print(f"  Min median speed: {self.min_median_speed_ms} m/s")
        print(f"  Min net eastward: {self.min_net_east_km} km")
        print(f"  Mountain mask: {len(MOUNTAIN_MASK_BOXES)} boxes active")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_hemisphere(self, lat):
        return 'NH' if lat >= 0 else 'SH'

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """Great-circle distance in km, handles dateline."""
        if np.any(np.isnan([lon1, lat1, lon2, lat2])) or \
           np.any(np.isinf([lon1, lat1, lon2, lat2])):
            return np.inf
        lon1 = ((lon1 + 180) % 360) - 180
        lon2 = ((lon2 + 180) % 360) - 180
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        if abs(dlon) > np.pi:
            dlon -= np.sign(dlon) * 2 * np.pi
        a = np.clip(
            np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2,
            0.0, 1.0
        )
        return 2 * self.earth_radius * np.arcsin(np.sqrt(a))

    def calculate_bearing(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = (np.cos(lat1) * np.sin(lat2) -
             np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        return (np.degrees(np.arctan2(y, x)) + 360) % 360

    def is_within_region(self, lon, lat):
        return (self.region_bounds['lon_min'] <= lon <= self.region_bounds['lon_max'] and
                self.region_bounds['lat_min'] <= lat <= self.region_bounds['lat_max'])

    def is_realistic_movement(self, lon1, lat1, lon2, lat2, time_diff_hours):
        distance = self.haversine_distance(lon1, lat1, lon2, lat2)
        if time_diff_hours <= 0:
            return False
        speed_kmh = distance / time_diff_hours
        return self.min_speed_kmh <= speed_kmh <= self.max_speed_kmh

    # =========================================================================
    # RETROGRADE / DISPLACEMENT STATS
    # =========================================================================

    def calculate_retrograde_stats(self, track):
        if len(track) < 2:
            return 0.0, 0.0
        seg_hours, bearings = [], []
        for i in range(len(track) - 1):
            prev_time, prev_lon, prev_lat, _ = track[i]
            curr_time, curr_lon, curr_lat, _ = track[i+1]
            dt = (pd.to_datetime(curr_time) - pd.to_datetime(prev_time)).total_seconds() / 3600.0
            if dt <= 0:
                continue
            seg_hours.append(dt)
            bearings.append(self.calculate_bearing(prev_lon, prev_lat, curr_lon, curr_lat))
        if not seg_hours:
            return 0.0, 0.0
        seg_hours = np.array(seg_hours)
        bearings  = np.array(bearings)
        wlo, whi  = self.west_sector
        west_mask = (bearings >= wlo) & (bearings <= whi)
        retro_total_h = float(seg_hours[west_mask].sum()) if west_mask.any() else 0.0
        total_h       = float(seg_hours.sum())
        retro_frac    = retro_total_h / total_h if total_h > 0 else 0.0
        retro_longest = run_h = 0.0
        for is_west, dh in zip(west_mask, seg_hours):
            if is_west:
                run_h += dh
            else:
                retro_longest = max(retro_longest, run_h)
                run_h = 0.0
        retro_longest = max(retro_longest, run_h)
        return retro_frac, retro_longest

    def calculate_net_zonal_displacement(self, track):
        if len(track) < 2:
            return 0.0
        start_lon, start_lat = track[0][1], track[0][2]
        end_lon,   end_lat   = track[-1][1], track[-1][2]
        start_lon = ((start_lon + 180) % 360) - 180
        end_lon   = ((end_lon   + 180) % 360) - 180
        dlon = end_lon - start_lon
        if abs(dlon) > 180:
            dlon -= np.sign(dlon) * 360
        avg_lat  = np.radians((start_lat + end_lat) / 2.0)
        dlon_rad = np.radians(dlon)
        return 6371.0 * np.cos(avg_lat) * dlon_rad

    # =========================================================================
    # DETECTION: find centers in a single time-step field
    # =========================================================================

    def find_etc_centers(self, detection_field, lons, lats):
        labeled, num_features = ndimage.label(detection_field)
        centers = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) < self.min_area:
                continue
            y_idx, x_idx = np.where(mask)
            cx = int(round(np.clip(np.mean(x_idx), 0, len(lons) - 1)))
            cy = int(round(np.clip(np.mean(y_idx), 0, len(lats) - 1)))
            clon, clat = lons[cx], lats[cy]
            if np.isnan(clon) or np.isnan(clat):
                continue
            if not self.is_within_region(clon, clat):
                continue
            if abs(clat) < self.min_abs_lat:
                continue
            # ── MOUNTAIN MASK (per-detection) ────────────────────────────────
            if point_in_mountain_mask(clon, clat):
                continue
            centers.append((clon, clat, int(np.sum(mask))))
        return centers

    # =========================================================================
    # LINKING
    # =========================================================================

    def calculate_movement_score(self, prev_center, curr_center, track_history=None):
        prev_lon, prev_lat, prev_intensity = prev_center
        curr_lon, curr_lat, curr_intensity = curr_center
        distance = self.haversine_distance(prev_lon, prev_lat, curr_lon, curr_lat)
        if distance > self.max_distance_km:
            return np.inf
        bearing = self.calculate_bearing(prev_lon, prev_lat, curr_lon, curr_lat)
        if not self.is_realistic_movement(prev_lon, prev_lat, curr_lon, curr_lat,
                                          self.time_gap_hours):
            return np.inf
        score = distance
        eastward_component = np.cos(np.radians(bearing - 90))
        if eastward_component > 0:
            score *= (1.0 / self.preferred_direction_weight)
        else:
            score *= self.preferred_direction_weight
        lat_change = abs(curr_lat - prev_lat)
        if lat_change > 5.0:
            score *= 1.5
        if track_history and len(track_history) >= 2:
            prev_bearing = self.calculate_bearing(
                track_history[-2][1], track_history[-2][2],
                track_history[-1][1], track_history[-1][2]
            )
            bearing_change = abs(bearing - prev_bearing)
            if bearing_change > 180:
                bearing_change = 360 - bearing_change
            if bearing_change < 45:
                score *= (1.0 / self.direction_consistency_weight)
            elif bearing_change > 90:
                score *= self.direction_consistency_weight
        intensity_change = abs(curr_intensity - prev_intensity) / max(prev_intensity, 1.0)
        if intensity_change > 3.0:
            score *= 1.3
        return score

    def link_tracks_meteorological(self, prev_centers, curr_centers, track_histories):
        if not prev_centers or not curr_centers:
            return []
        n_prev, n_curr = len(prev_centers), len(curr_centers)
        cost_matrix = np.full((n_prev, n_curr), np.inf)
        for i, pc in enumerate(prev_centers):
            hist = track_histories.get(i, [])
            for j, cc in enumerate(curr_centers):
                cost_matrix[i, j] = self.calculate_movement_score(pc, cc, hist)
        links, used_curr = [], set()
        for i in sorted(range(n_prev), key=lambda k: np.min(cost_matrix[k, :])):
            avail = cost_matrix[i, :].copy()
            avail[list(used_curr)] = np.inf
            if np.all(np.isinf(avail)):
                continue
            j = int(np.argmin(avail))
            if avail[j] < np.inf and avail[j] < self.max_distance_km:
                links.append((i, j))
                used_curr.add(j)
        return links

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_track_meteorology(self, track):
        """Comprehensive meteorological + terrain validation."""
        if len(track) < self.min_track_length:
            return False

        # 1. Duration
        t0 = pd.to_datetime(track[0][0])
        t1 = pd.to_datetime(track[-1][0])
        duration_h = (t1 - t0).total_seconds() / 3600.0
        if duration_h < self.min_hours:
            return False

        # 2. Total path length
        total_dist = sum(
            self.haversine_distance(track[i-1][1], track[i-1][2],
                                    track[i][1],   track[i][2])
            for i in range(1, len(track))
        )
        if total_dist < self.min_path_km:
            return False

        # 3. Median speed
        speeds = []
        for i in range(1, len(track)):
            dist = self.haversine_distance(track[i-1][1], track[i-1][2],
                                           track[i][1],   track[i][2])
            dt_h = (pd.to_datetime(track[i][0]) -
                    pd.to_datetime(track[i-1][0])).total_seconds() / 3600.0
            if dt_h > 0:
                speeds.append((dist * 1000.0) / (dt_h * 3600.0))   # m/s
        if speeds:
            med = np.median(speeds)
            if med < self.min_median_speed_ms or med > self.max_median_speed_ms:
                return False

        # 4. Average propagation speed (path efficiency)
        if duration_h > 0:
            avg_ms = (total_dist * 1000.0) / (duration_h * 3600.0)
            if avg_ms < self.min_avg_propagation_ms:
                return False

        # 5. Retrograde constraints
        retro_frac, retro_longest = self.calculate_retrograde_stats(track)
        if retro_longest > self.max_retro_run_hours:
            return False
        if retro_frac > self.max_retro_total_frac:
            return False

        # 6. Net eastward displacement
        if self.require_net_east:
            net_east = self.calculate_net_zonal_displacement(track)
            if net_east < self.min_net_east_km:
                return False

        # 7. Mountain-mask fraction (track-level)
        mountain_pts = sum(1 for _, lon, lat, _ in track
                           if point_in_mountain_mask(lon, lat))
        if mountain_pts / len(track) > MOUNTAIN_TRACK_FRAC:
            return False

        return True

    # =========================================================================
    # SINGLE-MONTH PROCESSOR (kept for backwards compatibility)
    # =========================================================================

    def process_month(self, nc_file_path, output_file_path):
        print(f"Processing {os.path.basename(nc_file_path)}...")
        ds = xr.open_dataset(nc_file_path)
        times   = ds.time.values
        lons    = ds.longitude.values
        lats    = ds.latitude.values
        etc_data = ds.ETCs_detection.values

        active_tracks       = {}
        next_track_id       = 1
        prev_centers        = []
        prev_center_to_track = {}

        for t_idx, time in enumerate(times):
            curr_centers = self.find_etc_centers(etc_data[t_idx], lons, lats)
            if prev_centers:
                hists  = {ci: active_tracks[tid]
                          for ci, tid in prev_center_to_track.items()
                          if tid in active_tracks}
                links  = self.link_tracks_meteorological(prev_centers, curr_centers, hists)
                curr_map = {}
                for pi, ci in links:
                    if pi in prev_center_to_track:
                        tid = prev_center_to_track[pi]
                        lon, lat, intensity = curr_centers[ci]
                        active_tracks[tid].append((time, lon, lat, intensity))
                        curr_map[ci] = tid
                for ci, (lon, lat, intensity) in enumerate(curr_centers):
                    if ci not in curr_map:
                        active_tracks[next_track_id] = [(time, lon, lat, intensity)]
                        curr_map[ci] = next_track_id
                        next_track_id += 1
                prev_center_to_track = curr_map
            else:
                prev_center_to_track = {}
                for ci, (lon, lat, intensity) in enumerate(curr_centers):
                    active_tracks[next_track_id] = [(time, lon, lat, intensity)]
                    prev_center_to_track[ci] = next_track_id
                    next_track_id += 1
            prev_centers = curr_centers

        valid = {tid: t for tid, t in active_tracks.items()
                 if len(t) >= self.min_track_length and
                 self.validate_track_meteorology(t)}

        print(f"  Initial: {len(active_tracks)}  →  Valid: {len(valid)}")
        self.write_tracks_to_file(valid, output_file_path,
                                  os.path.basename(nc_file_path))
        ds.close()
        return len(valid)

    # =========================================================================
    # MULTI-MONTH PROCESSOR (main workhorse, with gap bridging)
    # =========================================================================

    def process_multiple_months(self, nc_file_paths, output_file_path):
        print(f"Processing {len(nc_file_paths)} files for continuous tracking...")

        active_tracks = {}
        next_track_id = 1
        total_detections = 0
        files_processed  = 0

        for file_idx, nc_file_path in enumerate(nc_file_paths):
            print(f"  File {file_idx+1}/{len(nc_file_paths)}: "
                  f"{os.path.basename(nc_file_path)}")
            ds = xr.open_dataset(nc_file_path)
            times    = ds.time.values
            lons     = ds.longitude.values
            lats     = ds.latitude.values
            etc_data = ds.ETCs_detection.values

            for t_idx, time in enumerate(times):
                curr_centers = self.find_etc_centers(etc_data[t_idx], lons, lats)
                total_detections += len(curr_centers)

                claimed_curr = set()

                for track_id, track_info in list(active_tracks.items()):
                    if track_info.get('closed', False):
                        continue
                    points    = track_info['points']
                    gap_count = track_info.get('gap_count', 0)
                    ref_lon   = track_info.get('ghost_lon', points[-1][1])
                    ref_lat   = track_info.get('ghost_lat', points[-1][2])

                    best_idx, best_dist = None, np.inf
                    for c_idx, (c_lon, c_lat, c_int) in enumerate(curr_centers):
                        if c_idx in claimed_curr:
                            continue
                        dist = self.haversine_distance(ref_lon, ref_lat, c_lon, c_lat)
                        eff_max = self.max_distance_km * (1 + gap_count * 0.5)
                        if dist < eff_max and dist < best_dist:
                            dt_h = self.time_gap_hours * (1 + gap_count)
                            if self.is_realistic_movement(ref_lon, ref_lat,
                                                          c_lon, c_lat, dt_h):
                                best_dist = dist
                                best_idx  = c_idx

                    if best_idx is not None:
                        c_lon, c_lat, c_int = curr_centers[best_idx]
                        claimed_curr.add(best_idx)
                        points.append((time, c_lon, c_lat, c_int))
                        if len(points) >= 2:
                            track_info['last_dlon'] = c_lon - points[-2][1]
                            track_info['last_dlat'] = c_lat - points[-2][2]
                        track_info['gap_count']  = 0
                        track_info['ghost_lon']  = c_lon
                        track_info['ghost_lat']  = c_lat
                    else:
                        new_gap = gap_count + 1
                        if new_gap > self.max_gap_steps:
                            track_info['closed'] = True
                        else:
                            track_info['gap_count'] = new_gap
                            dlon = track_info.get('last_dlon', 0.0)
                            dlat = track_info.get('last_dlat', 0.0)
                            track_info['ghost_lon'] = ref_lon + dlon
                            track_info['ghost_lat'] = ref_lat + dlat

                for c_idx, (c_lon, c_lat, c_int) in enumerate(curr_centers):
                    if c_idx not in claimed_curr:
                        active_tracks[next_track_id] = {
                            'points':    [(time, c_lon, c_lat, c_int)],
                            'gap_count': 0,
                            'ghost_lon': c_lon,
                            'ghost_lat': c_lat,
                            'last_dlon': 0.0,
                            'last_dlat': 0.0,
                            'closed':    False,
                        }
                        next_track_id += 1

            ds.close()
            files_processed += 1
            open_n = sum(1 for t in active_tracks.values()
                         if not t.get('closed', False))
            print(f"    → {open_n} open tracks, {len(active_tracks)} total")

        # Final validation
        valid_tracks = {}
        for tid, track_info in active_tracks.items():
            track = track_info['points'] if isinstance(track_info, dict) else track_info
            if (len(track) >= self.min_track_length and
                    self.validate_track_meteorology(track)):
                valid_tracks[tid] = track

        print(f"\nFinal results:")
        print(f"  Total detections : {total_detections}")
        print(f"  Initial tracks   : {len(active_tracks)}")
        print(f"  Valid tracks     : {len(valid_tracks)}")

        durations = []
        for track in valid_tracks.values():
            dh = (pd.to_datetime(track[-1][0]) - pd.to_datetime(track[0][0])
                  ).total_seconds() / 3600.0
            durations.append(dh)
        if durations:
            print(f"  Duration (h): min={np.min(durations):.0f}  "
                  f"med={np.median(durations):.0f}  "
                  f"max={np.max(durations):.0f}  "
                  f"mean={np.mean(durations):.0f}")

        self.write_tracks_to_file(valid_tracks, output_file_path,
                                  f"{files_processed} monthly files")
        return len(valid_tracks)

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def write_tracks_to_file(self, tracks, output_file, source_file):
        with open(output_file, 'w') as f:
            f.write("# Meteorological ETC Track File v2\n")
            f.write(f"# Source: {source_file}\n")
            f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Lon [{self.region_bounds['lon_min']}°, "
                    f"{self.region_bounds['lon_max']}°]  "
                    f"Lat [{self.region_bounds['lat_min']}°, "
                    f"{self.region_bounds['lat_max']}°]\n")
            f.write(f"# min_path_km            : {self.min_path_km}\n")
            f.write(f"# min_avg_propagation_ms : {self.min_avg_propagation_ms}\n")
            f.write(f"# min_median_speed_ms    : {self.min_median_speed_ms}\n")
            f.write(f"# min_net_east_km        : {self.min_net_east_km}\n")
            f.write(f"# max_retro_run_hours    : {self.max_retro_run_hours}\n")
            f.write(f"# max_retro_total_frac   : {self.max_retro_total_frac}\n")
            f.write(f"# mountain_mask_boxes    : {len(MOUNTAIN_MASK_BOXES)}\n")
            f.write(f"# Total valid tracks     : {len(tracks)}\n")
            f.write("#\n# Format: TRACK_ID TIME LONGITUDE LATITUDE INTENSITY\n#\n")

            for counter, orig_id in enumerate(sorted(tracks.keys()), start=1):
                track = tracks[orig_id]
                total_dist = sum(
                    self.haversine_distance(track[i-1][1], track[i-1][2],
                                            track[i][1],   track[i][2])
                    for i in range(1, len(track))
                )
                f.write(f"# Track {counter} "
                        f"({len(track)} pts, {total_dist:.0f} km)\n")
                for time, lon, lat, intensity in track:
                    ts = pd.to_datetime(time).strftime('%Y-%m-%d_%H:%M:%S')
                    f.write(f"{counter:4d} {ts} {lon:8.3f} {lat:7.3f} "
                            f"{intensity:6.1f}\n")
                f.write("\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    etc_data_dir = (
        '/lustre/orion/cli138/proj-shared/z7a/Storms_data/'
        'Storms_precipitation_notebooks/Percentile_ARs_detection/'
        'ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/'
        'ETCs_enchanced_test_etc'
    )
    output_dir = (
        '/lustre/orion/cli138/proj-shared/z7a/Storms_data/'
        'Storms_precipitation_notebooks/Percentile_ARs_detection/'
        'ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/'
        'ETCs_enchanced_test_etc/ETC_track_files_yearly_5'
    )

    min_year = 1980
    max_year = 2024

    os.makedirs(output_dir, exist_ok=True)

    region_bounds = {
        'lon_min': -180.0, 'lon_max': 180.0,
        'lat_min':  -70.0, 'lat_max':  70.0
    }

    print("ETC Tracking System v2 (mountain-masked, tighter filters)")
    print("=" * 60)
    print(f"Processing years: {min_year} to {max_year}")
    print("=" * 60)

    tracker = ETCTracker(
        max_distance_km=800,
        min_track_length=8,
        region_bounds=region_bounds
    )

    file_pattern = os.path.join(etc_data_dir, "era5dara_*_ETCs_detection.nc")
    nc_files = sorted(glob.glob(file_pattern))

    if not nc_files:
        print(f"No files found: {file_pattern}")
        return

    files_by_year = {}
    for nc_file in nc_files:
        basename  = os.path.basename(nc_file)
        date_part = basename.split('_')[1]
        try:
            yr = int(date_part[:4])
            if min_year <= yr <= max_year:
                files_by_year.setdefault(yr, []).append(nc_file)
        except (ValueError, IndexError):
            print(f"Warning: cannot parse year from {basename}")

    print(f"Found {len(nc_files)} files; "
          f"{sum(len(v) for v in files_by_year.values())} in range "
          f"{min_year}-{max_year}")
    print("=" * 60)

    total_all, ok_years = 0, 0

    for year in sorted(files_by_year.keys()):
        year_files = sorted(files_by_year[year])
        print(f"\n{'='*60}\nPROCESSING YEAR {year}  ({len(year_files)} files)\n{'='*60}")
        try:
            out_path = os.path.join(output_dir,
                                    f"ETC_tracks_{year}_meteorological_v2.txt")
            n = tracker.process_multiple_months(year_files, out_path)
            total_all += n
            ok_years  += 1
            print(f"✓ Year {year}: {n} tracks")
        except Exception as e:
            import traceback
            print(f"✗ Year {year}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Years {min_year}-{max_year}: {ok_years}/{len(files_by_year)} OK")
    print(f"Total valid tracks: {total_all}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()