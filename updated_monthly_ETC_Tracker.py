import numpy as np
import xarray as xr
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import glob


# =============================================================================
# RegCM-CMIP6 DOMAIN
# =============================================================================
# The regridded RegCM grid: lon -138.75..-57.0, lat 18.0..56.75 (0.25 deg).
# Everything outside this box is absent from the data, so global-scale criteria
# written for the ERA5 version are inapplicable here and have been removed.
REGCM_DOMAIN = {
    'lon_min': -138.75, 'lon_max': -57.0,
    'lat_min':   18.0,  'lat_max':  56.75,
}

# Paired WNet inputs, used only for their MSLP NaN pattern -> the true (curved
# Lambert) domain mask. See ETCTracker._domain_mask.
REGCM_INPUT_BASE = ("/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/"
                    "RegCM-CMIP6/Data-processed")


# =============================================================================
# MOUNTAIN / OROGRAPHIC MASK — REMOVED for the RegCM North American domain
# =============================================================================
# The global ERA5 version of this tracker masked out high terrain (Rockies,
# Tibetan Plateau, Andes, Greenland, Alps/Pyrenees, East African Highlands) to
# suppress quasi-stationary orographic lows that are not synoptic ETCs.
#
# That filter is dropped here, for two independent reasons:
#
#   1. Most of those boxes lie entirely OUTSIDE the RegCM domain
#      (lon -138.75..-57, lat 18..56.75), so they could never match — Tibet,
#      the Andes, Greenland, the Alps/Pyrenees and East Africa were dead code.
#
#   2. The remaining boxes (Rockies, Sierra Nevada/Great Basin, Mexican
#      Plateau) cover the single most important ETC genesis region in this
#      domain. Rocky Mountain LEE CYCLOGENESIS — Colorado lows and Alberta
#      clippers — is genuine, well-documented synoptic ETC activity, not
#      orographic noise. Masking it would delete real storms and bias the
#      North American ETC climatology precisely where it matters most.
#
# The remaining track-quality filters (path length, propagation speed, median
# speed, net eastward displacement, retrograde limits) already discriminate
# stationary orographic lows from propagating ETCs on kinematic grounds, which
# is the physically appropriate test in a regional domain.


class ETCTracker:
    def __init__(self, max_distance_km=800, min_track_length=8, region_bounds=None,
                 domain_erode_cells=2):
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

        # ── Domain mask ──────────────────────────────────────────────────────
        self.domain_erode_cells = domain_erode_cells
        self._dmask_cache = None

        # ── Gap bridging ─────────────────────────────────────────────────────
        self.max_gap_steps = 1   # allow 1 missed 6-h step

        # ── Spatial filter ───────────────────────────────────────────────────
        # ETCs only in mid/high latitudes (kept at 28°, consistent with v1)
        self.min_abs_lat = 28.0

        # ── Minimum detection area ───────────────────────────────────────────
        # Raised from 12 → 15 pixels
        self.min_area = 15

        # ── Regional bounds ──────────────────────────────────────────────────
        # Default = the RegCM North American domain. The global default
        # (lon -180..0, lat 20..70) spanned ocean basins and both hemispheres
        # that this regional dataset does not cover.
        if region_bounds is None:
            self.region_bounds = REGCM_DOMAIN.copy()
        else:
            self.region_bounds = region_bounds

        # NOTE: the Southern-Hemisphere branch (track_both_hemispheres /
        # get_hemisphere) has been dropped — this domain is entirely Northern
        # Hemisphere (18..56.75 N), so that code could never execute.

        print(f"Meteorological ETC Tracker (RegCM North America) initialized:")
        print(f"  Regional bounds: Lon [{self.region_bounds['lon_min']}°, "
              f"{self.region_bounds['lon_max']}°], "
              f"Lat [{self.region_bounds['lat_min']}°, {self.region_bounds['lat_max']}°]")
        print(f"  Max speed: {self.max_speed_kmh:.0f} km per 6h")
        print(f"  Max link distance: {self.max_distance_km} km")
        print(f"  Min path: {self.min_path_km} km")
        print(f"  Min avg propagation: {self.min_avg_propagation_ms} m/s")
        print(f"  Min median speed: {self.min_median_speed_ms} m/s")
        print(f"  Min net eastward: {self.min_net_east_km} km")
        print(f"  Orographic mask: none (lee cyclogenesis retained)")

    # =========================================================================
    # HELPERS
    # =========================================================================

    # (get_hemisphere removed — the RegCM domain is entirely Northern
    #  Hemisphere, and nothing called it.)

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
            # (no orographic mask — see the note at the top of this file:
            #  Rocky Mountain lee cyclogenesis is real ETC activity here)
            centers.append((clon, clat, int(np.sum(mask))))
        return centers

    def _domain_mask(self, nc_file_path, lons, lats):
        """True where the simulation exists, from the paired WNet input's MSLP NaNs.

        Same rationale as Hurricane_track._domain_mask: RegCM's curved Lambert domain
        is inscribed in the lat/lon box, leaving ~28% of the box with no simulated
        atmosphere, where the WNet still emits a boundary response (the tensors were
        built with fillna(0.0) and the convolutions zero-pad).

        Returns None if the paired file cannot be found, in which case tracking
        proceeds unmasked exactly as before — this must never be a hard failure on a
        layout it does not recognise.
        """
        if self._dmask_cache is not None:
            return self._dmask_cache
        base = os.path.basename(nc_file_path).replace('_ETCs_detection.nc', '.nc')
        model = base.split('dara_')[0] if 'dara_' in base else None
        if model is None:
            return None
        paired = os.path.join(REGCM_INPUT_BASE, model, "WNet_6hourly_latlon", base)
        if not os.path.exists(paired):
            print(f"    (no paired input {paired} — tracking UNMASKED)")
            return None
        with xr.open_dataset(paired) as pds:
            if 'MSLP' not in pds:
                return None
            valid = ~np.isnan(pds['MSLP'].isel(time=0).values)
        if valid.shape != (len(lats), len(lons)):
            print(f"    (paired grid {valid.shape} != detection grid "
                  f"{(len(lats), len(lons))} — tracking UNMASKED)")
            return None
        if self.domain_erode_cells > 0:
            valid = ndimage.binary_erosion(
                valid, structure=np.ones((3, 3), dtype=bool),
                iterations=self.domain_erode_cells, border_value=0)
        self._dmask_cache = valid
        print(f"    domain mask: {100.0 * valid.mean():.1f}% of the box is inside "
              f"the simulation (eroded {self.domain_erode_cells} cells)")
        return valid

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

        # (7. track-level mountain-fraction test removed — see header note.
        #  Stationary orographic lows are rejected kinematically by the path,
        #  propagation-speed and net-eastward criteria above.)

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

            # Zero detections outside the simulated domain before any blob forms.
            # ETC is far less affected than TC (98.9% of ETC track points are already
            # in-domain, because extratropical cyclones live at mid-latitudes where the
            # curved Lambert domain fills the lat/lon box), but the same boundary
            # response produces a specific artifact: ~8% of tracks (102 of 1283 for
            # ACC) end with a spurious jump of ~1300 km in one 6 h step onto the single
            # out-of-domain cell (-61.50, 28.00). Masking removes it at the source.
            dmask = self._domain_mask(nc_file_path, lons, lats)
            if dmask is not None:
                etc_data = np.where(dmask[None, :, :], etc_data, 0.0)

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
            f.write(f"# orographic_mask        : none (lee cyclogenesis retained)\n")
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
    """Track ETCs for a RegCM model and write per-year .txt tracks.

    ETC detection is UNCHANGED by the precipitation-MCS work (channels 0-10 are
    bit-identical), so this reads the same detection files the precip test run
    emits: <MODEL>dara_YYYYMM_ETCs_detection.nc.

    Usage:
        REGCM_MODEL=ACC python updated_monthly_ETC_Tracker.py
        python updated_monthly_ETC_Tracker.py --model BCC --start-year 2005 --end-year 2014
    """
    import argparse
    KNOWN = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]
    p = argparse.ArgumentParser(description="RegCM ETC tracking -> per-year .txt")
    p.add_argument("--model", default=os.environ.get("REGCM_MODEL", "ACC"),
                   help=f"RegCM model ({'/'.join(KNOWN)}). Default $REGCM_MODEL or ACC.")
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2014)
    p.add_argument("--etc-dir", default=None,
                   help="Detection dir (default: ARs_Hurricanes_ETCs_MCS/Outputs_RegCM_PR_<MODEL>)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write tracks (default: <etc-dir>/ETC_tracks)")
    args = p.parse_args()
    model = args.model

    HERE = os.path.dirname(os.path.abspath(__file__))
    etc_data_dir = args.etc_dir or os.path.join(
        HERE, "ARs_Hurricanes_ETCs_MCS", f"Outputs_RegCM_PR_{model}")
    output_dir = args.output_dir or os.path.join(etc_data_dir, "ETC_tracks")

    min_year = args.start_year
    max_year = args.end_year

    if not os.path.isdir(etc_data_dir):
        raise SystemExit(f"Detection dir not found: {etc_data_dir}\n"
                         f"  (has {model} been tested? see Run_test_distributed.sh)")

    os.makedirs(output_dir, exist_ok=True)

    # RegCM North American domain (see REGCM_DOMAIN at the top of this file).
    region_bounds = REGCM_DOMAIN.copy()

    print(f"ETC Tracking System — RegCM {model} (North American domain)")
    print("=" * 60)
    print(f"Processing years: {min_year} to {max_year}")
    print(f"Input : {etc_data_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    tracker = ETCTracker(
        max_distance_km=800,
        min_track_length=8,
        region_bounds=region_bounds
    )

    file_pattern = os.path.join(etc_data_dir, f"{model}dara_*_ETCs_detection.nc")
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
                                    f"ETC_tracks_{model}_{year}.txt")
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