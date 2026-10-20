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
#
# Both mask stages are env-configurable so the mask can be MEASURED rather than
# assumed. Defaults reproduce the original behaviour exactly.
#   ETC_MOUNTAIN_TRACK_FRAC   track-level threshold; >= 1.0 disables the rule
#   ETC_MOUNTAIN_FRAME        "0" disables the per-detection centre rejection
#
# Measured against TempestExtremes ETCs 2000-2010 (1465 tracks): 29.1% touch a
# mountain box, 18.6% have their GENESIS inside one, and 19.1% exceed the 0.30
# rule. The Rockies box alone is touched by 392 of them — that is lee cyclogenesis
# (Colorado lows, Alberta clippers), real storms rather than orographic artefacts.
# The artefact the mask targets is genuine (sea-level-reduced MSLP over high
# terrain, and an 850 mb level that is below ground there), but a STATIONARY
# artefact is better caught by the propagation criteria this tracker already has
# (min_avg_propagation_ms, min_net_east_km, retrograde limits) than by a
# geographic box that cannot distinguish a lee cyclone from an artefact.
MOUNTAIN_TRACK_FRAC = float(os.environ.get("ETC_MOUNTAIN_TRACK_FRAC", "0.30"))
MOUNTAIN_FRAME_MASK = os.environ.get("ETC_MOUNTAIN_FRAME", "1") not in ("0", "false", "False")

# =============================================================================
# TRACKER DESIGN SWITCHES (all default to the shipped behaviour)
# =============================================================================
# phase4_etc_miss_attribution.py (4123 NAECC cyclones, 2000-2010) showed where the misses
# are: 35 % of all cyclones are lost at the TRACK stage (the mask covers them and a centre
# is nearby, but no valid track results) against 11 % lost in the mask. Applied to the NAECC
# tracks themselves, the rules below let only 53 % of real cyclones through: 52 % have a
# 6-hourly step under the link floor, 19 % fail the path length, 18 % the retrograde limits,
# 15 % the net-eastward rule. Slow, stalling and retrograde mature lows (Gulf of Alaska,
# Labrador Sea) are exactly what these rules delete.
#   ETC_MAX_GAP_STEPS     missed 6-h steps a track may bridge (shipped: 1; the TC tracker uses 4)
#   ETC_FILTER_PROFILE    "legacy" = the shipped validation rules. "common" = only the QC every
#                         reference catalogue gets (>= 48 h, >= 8 points, >= 1000 km net
#                         displacement, median speed <= 28 m/s), and no lower link speed.
#                         The link floor min_speed_kmh = 12 was meant as 12 km per 6 h (2 m/s)
#                         but is compared with km/h, so a step under 72 km cannot be linked.
#   ETC_LONWRAP           "1" joins blobs across the +-180 seam before centres are taken
#   ETC_LINK_MODE         "legacy" = each track, oldest first, takes its nearest free centre.
#                         "global" = all admissible (track, centre) pairs, continuing tracks
#                         before gapped ones, nearest pair first.
#   ETC_GAP_RADIUS_GROWTH the link radius grows to max_distance_km * (1 + gap * growth) while a
#                         track is bridging a gap (shipped: 0.5, i.e. 2400 km at gap 4). The
#                         first sweep showed gap bridging LOSING hits (NAEC POD 0.56 -> 0.22 at
#                         gap 4): the grown radius lets a track jump to another low. 0 keeps
#                         the radius at 800 km.
#   ETC_CENTER_MODE       "centroid" = the blob centroid (shipped). "mslp" = one centre per
#                         closed low of the ERA5 MSLP inside the blob (etc_closed_contour.py,
#                         the TempestExtremes criterion: 200 Pa closed within 6 degrees), with
#                         the intensity column = minimum MSLP in hPa. Blobs with no closed low
#                         give no centre: the physical form of topography awareness.
#   ETC_MSLP_DIR          where era5dara_YYYYMM.nc (variable MSLP) is read from in mslp mode
#   ETC_CC_DEPTH_PA / ETC_CC_RADIUS_DEG / ETC_CC_SIGMA_KM   closed-contour depth, radius (also
#                         the merge distance) and smoothing (defaults 200, 6, 75)
MAX_GAP_STEPS = int(os.environ.get("ETC_MAX_GAP_STEPS", "1"))
FILTER_PROFILE = os.environ.get("ETC_FILTER_PROFILE", "legacy")
LONWRAP = os.environ.get("ETC_LONWRAP", "0") not in ("0", "false", "False")
LINK_MODE = os.environ.get("ETC_LINK_MODE", "legacy")
GAP_GROWTH = float(os.environ.get("ETC_GAP_RADIUS_GROWTH", "0.5"))
CENTER_MODE = os.environ.get("ETC_CENTER_MODE", "centroid")
MSLP_DIR = os.environ.get("ETC_MSLP_DIR", "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data")
CC_DEPTH_PA = float(os.environ.get("ETC_CC_DEPTH_PA", "200"))
CC_RADIUS_DEG = float(os.environ.get("ETC_CC_RADIUS_DEG", "6"))
CC_SIGMA_KM = float(os.environ.get("ETC_CC_SIGMA_KM", "75"))
#   ETC_CENTER_FALLBACK_MIN  mslp mode only: a blob of at least this many cells that holds NO closed
#                         low still yields its centroid (intensity = smoothed MSLP there, hPa).
#                         0 (default) = such blobs yield nothing. Sweep 2: dropping them cut
#                         NAEC FAR 0.34 -> 0.12 and duplicates 1027 -> 14 but POD 0.77 -> 0.61;
#                         the fallback keeps cyclone-scale blobs (a 600 km label disc is ~1460
#                         cells at 0.25 deg) while one-pixel fragments (~14 cells) stay dropped.
CENTER_FALLBACK_MIN = int(os.environ.get("ETC_CENTER_FALLBACK_MIN", "0"))
if FILTER_PROFILE not in ("legacy", "common") or LINK_MODE not in ("legacy", "global") \
        or CENTER_MODE not in ("centroid", "mslp"):
    raise ValueError(f"ETC_FILTER_PROFILE={FILTER_PROFILE!r} ETC_LINK_MODE={LINK_MODE!r} "
                     f"ETC_CENTER_MODE={CENTER_MODE!r}")
NON_DEFAULT = (MAX_GAP_STEPS != 1 or FILTER_PROFILE != "legacy" or LONWRAP or LINK_MODE != "legacy"
               or GAP_GROWTH != 0.5 or CENTER_MODE != "centroid" or CENTER_FALLBACK_MIN > 0)
if CENTER_MODE == "mslp":
    import etc_closed_contour as _cc          # same directory as this file (not in the release copy)


def label_blobs_lonwrap(field):
    """ndimage.label (4-connected, as the shipped tracker) with blobs that touch across the
    first and last longitude joined into one."""
    labeled, n = ndimage.label(field)
    if n < 2:
        return labeled, n
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    left, right = labeled[:, 0], labeled[:, -1]
    k = (left > 0) & (right > 0) & (left != right)
    for a, b in set(zip(left[k].tolist(), right[k].tolist())):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(n + 1)])
    uniq, lut = np.unique(roots, return_inverse=True)
    return lut[labeled], uniq.size - 1


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
        self.max_gap_steps = MAX_GAP_STEPS   # missed 6-h steps a track may bridge (shipped: 1)

        # Net (first to last point) displacement floor; 0 disables. Only the
        # "common" profile uses it, in place of the rules it switches off.
        self.min_net_km = 0.0
        if FILTER_PROFILE == "common":
            self.min_speed_kmh = 0.0
            self.min_path_km = 0.0
            self.min_avg_propagation_ms = 0.0
            self.min_median_speed_ms = 0.0
            self.max_retro_run_hours = np.inf
            self.max_retro_total_frac = 1.0
            self.require_net_east = False
            self.min_net_km = 1000.0
        self.rejected = {}

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

    def find_etc_centers(self, detection_field, lons, lats, mslp=None):
        if CENTER_MODE == "mslp":
            if mslp is None:
                raise ValueError("ETC_CENTER_MODE=mslp needs the MSLP field of the time step")
            return self._mslp_centers(detection_field, lons, lats, mslp)
        if LONWRAP:
            labeled, num_features = label_blobs_lonwrap(detection_field)
        else:
            labeled, num_features = ndimage.label(detection_field)
        centers = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) < self.min_area:
                continue
            y_idx, x_idx = np.where(mask)
            if LONWRAP and x_idx.min() == 0 and x_idx.max() == len(lons) - 1:
                # a blob across the seam: the plain mean of x would land near longitude 0
                ang = 2.0 * np.pi * x_idx / len(lons)
                mean_x = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2.0 * np.pi)
                          ) * len(lons) / (2.0 * np.pi)
                cx = int(round(mean_x)) % len(lons)
            else:
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
            if MOUNTAIN_FRAME_MASK and point_in_mountain_mask(clon, clat):
                continue
            centers.append((clon, clat, int(np.sum(mask))))
        return centers

    def _mslp_centers(self, detection_field, lons, lats, mslp):
        """One centre per closed MSLP low inside the mask blobs (ETC_CENTER_MODE=mslp). Blobs
        under min_area are ignored as in centroid mode; a blob without a closed low yields
        nothing. The candidate region is the filled blobs grown by 2 degrees, because the
        label's wind gate leaves a hole at the calm centre of a cyclone."""
        labeled, n = (label_blobs_lonwrap if LONWRAP else ndimage.label)(detection_field)
        if n == 0:
            return []
        sizes = np.bincount(labeled.ravel(), minlength=n + 1)
        ids = np.flatnonzero(sizes >= self.min_area)
        ids = ids[ids > 0]
        if ids.size == 0:
            return []
        cand = _cc.blob_candidate(np.isin(labeled, ids), lats, grow_deg=2.0)
        p = _cc.smooth(mslp, lats, CC_SIGMA_KM) if CC_SIGMA_KM else np.asarray(mslp, np.float64)
        centers = []
        claimed = set()
        dlat, dlon = abs(float(lats[1] - lats[0])), abs(float(lons[1] - lons[0]))
        for la, lo, p0 in _cc.closed_lows(p, lats, lons, candidate=cand, depth_pa=CC_DEPTH_PA,
                                          radius_deg=CC_RADIUS_DEG, merge_deg=CC_RADIUS_DEG,
                                          sigma_km=0):
            # the blob this low belongs to: the commonest blob label within ~2 degrees (the low
            # may sit in a hole of the blob or in its grown margin, where `labeled` is 0)
            iy = int(round((lats[0] - la) / -dlat)) if lats[0] < lats[-1] else int(round((lats[0] - la) / dlat))
            ix = int(round((lo - lons[0]) / dlon)) % len(lons)
            k = max(int(round(2.0 / dlat)), 1)
            win = labeled[max(iy - k, 0):iy + k + 1][:, np.arange(ix - k, ix + k + 1) % len(lons)]
            near = win[win > 0]
            if near.size:
                claimed.add(int(np.bincount(near).argmax()))
            if not self.is_within_region(lo, la) or abs(la) < self.min_abs_lat:
                continue
            if MOUNTAIN_FRAME_MASK and point_in_mountain_mask(lo, la):
                continue
            centers.append((float(lo), float(la), float(p0) / 100.0))      # intensity = hPa
        if CENTER_FALLBACK_MIN > 0:
            for i in ids:
                if int(i) in claimed or sizes[i] < CENTER_FALLBACK_MIN:
                    continue
                y_idx, x_idx = np.where(labeled == i)
                cy = int(round(np.clip(np.mean(y_idx), 0, len(lats) - 1)))
                cx = int(round(np.clip(np.mean(x_idx), 0, len(lons) - 1)))
                clon, clat = float(lons[cx]), float(lats[cy])
                if not self.is_within_region(clon, clat) or abs(clat) < self.min_abs_lat:
                    continue
                if MOUNTAIN_FRAME_MASK and point_in_mountain_mask(clon, clat):
                    continue
                centers.append((clon, clat, float(p[cy, cx]) / 100.0))
        return centers

    def _load_mslp(self, det_path, times, lons, lats):
        """ERA5 MSLP (Pa) of the detection file's month, on the detection grid."""
        from netCDF4 import Dataset, num2date
        tag = os.path.basename(det_path).split("_")[1]          # era5dara_YYYYMM_ETCs_detection.nc
        path = os.path.join(MSLP_DIR, f"era5dara_{tag}.nc")
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            m = ds.variables["MSLP"][:].astype(np.float32)      # whole variable: time-deep chunks
            lon = ds.variables["longitude"][:].astype(float)
            lat = ds.variables["latitude"][:].astype(float)
            tv = ds.variables["time"]
            t = num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                         only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        t = np.array([np.datetime64(x, "h") for x in t])
        det_t = np.asarray(times).astype("datetime64[h]")
        if t.size != det_t.size or not (t == det_t).all():
            raise ValueError(f"{path}: time axis differs from {det_path}")
        if lat[0] < lat[-1]:
            m, lat = m[:, ::-1], lat[::-1]
        if not np.allclose(lat, lats, atol=1e-3):
            raise ValueError(f"{path}: latitude grid differs from the detections")
        shift = int(np.argmin(np.abs(((lon - lons[0]) + 180.0) % 360.0 - 180.0)))
        m = np.roll(m, -shift, axis=2)                          # source 0..360 -> detection -180..180
        if np.nanmedian(m) < 2000.0:                            # hPa -> Pa
            m *= 100.0
        return m

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
        reason = self._first_failure(track)
        if reason is not None:
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
        return reason is None

    def _first_failure(self, track):
        """Name of the first validation rule the track fails, or None."""
        if len(track) < self.min_track_length:
            return "npts"

        # 1. Duration
        t0 = pd.to_datetime(track[0][0])
        t1 = pd.to_datetime(track[-1][0])
        duration_h = (t1 - t0).total_seconds() / 3600.0
        if duration_h < self.min_hours:
            return "hours"

        # 2. Total path length
        total_dist = sum(
            self.haversine_distance(track[i-1][1], track[i-1][2],
                                    track[i][1],   track[i][2])
            for i in range(1, len(track))
        )
        if total_dist < self.min_path_km:
            return "path"

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
                return "median_speed"

        # 4. Average propagation speed (path efficiency)
        if duration_h > 0:
            avg_ms = (total_dist * 1000.0) / (duration_h * 3600.0)
            if avg_ms < self.min_avg_propagation_ms:
                return "propagation"

        # 5. Retrograde constraints
        retro_frac, retro_longest = self.calculate_retrograde_stats(track)
        if retro_longest > self.max_retro_run_hours:
            return "retro_run"
        if retro_frac > self.max_retro_total_frac:
            return "retro_frac"

        # 6. Net eastward displacement
        if self.require_net_east:
            net_east = self.calculate_net_zonal_displacement(track)
            if net_east < self.min_net_east_km:
                return "net_east"

        # 6b. Net displacement, first point to last (the "common" profile only)
        if self.min_net_km > 0 and self.haversine_distance(
                track[0][1], track[0][2], track[-1][1], track[-1][2]) < self.min_net_km:
            return "net_km"

        # 7. Mountain-mask fraction (track-level). MOUNTAIN_TRACK_FRAC >= 1.0
        #    disables the rule, since a fraction can never exceed 1.
        if MOUNTAIN_TRACK_FRAC < 1.0:
            mountain_pts = sum(1 for _, lon, lat, _ in track
                               if point_in_mountain_mask(lon, lat))
            if mountain_pts / len(track) > MOUNTAIN_TRACK_FRAC:
                return "mountain"

        return None

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
    # GLOBAL LINKING (ETC_LINK_MODE=global)
    # =========================================================================

    def _link_global(self, active_tracks, curr_centers, time, claimed_curr):
        """Link one time step over all admissible (track, centre) pairs at once: continuing
        tracks before gapped ones, nearest pair first. Admissibility, the track update and
        the gap rules are those of the legacy loop in process_multiple_months; only the
        order in which tracks get to choose differs (legacy: oldest track first)."""
        open_tracks = [(tid, ti) for tid, ti in active_tracks.items()
                       if not ti.get('closed', False)]
        pairs = []
        for tid, ti in open_tracks:
            gap_count = ti.get('gap_count', 0)
            ref_lon = ti.get('ghost_lon', ti['points'][-1][1])
            ref_lat = ti.get('ghost_lat', ti['points'][-1][2])
            eff_max = self.max_distance_km * (1 + gap_count * GAP_GROWTH)
            dt_h = self.time_gap_hours * (1 + gap_count)
            for c_idx, (c_lon, c_lat, _) in enumerate(curr_centers):
                dist = self.haversine_distance(ref_lon, ref_lat, c_lon, c_lat)
                if dist < eff_max and self.is_realistic_movement(ref_lon, ref_lat,
                                                                 c_lon, c_lat, dt_h):
                    pairs.append((gap_count, dist, tid, c_idx))
        pairs.sort()
        linked = {}
        for _, _, tid, c_idx in pairs:
            if tid not in linked and c_idx not in claimed_curr:
                linked[tid] = c_idx
                claimed_curr.add(c_idx)

        for tid, ti in open_tracks:
            points = ti['points']
            if tid in linked:
                c_lon, c_lat, c_int = curr_centers[linked[tid]]
                points.append((time, c_lon, c_lat, c_int))
                if len(points) >= 2:
                    ti['last_dlon'] = c_lon - points[-2][1]
                    ti['last_dlat'] = c_lat - points[-2][2]
                ti['gap_count'] = 0
                ti['ghost_lon'] = c_lon
                ti['ghost_lat'] = c_lat
            else:
                new_gap = ti.get('gap_count', 0) + 1
                if new_gap > self.max_gap_steps:
                    ti['closed'] = True
                else:
                    ref_lon = ti.get('ghost_lon', points[-1][1])
                    ref_lat = ti.get('ghost_lat', points[-1][2])
                    ti['gap_count'] = new_gap
                    ti['ghost_lon'] = ref_lon + ti.get('last_dlon', 0.0)
                    ti['ghost_lat'] = ref_lat + ti.get('last_dlat', 0.0)

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
            mslp_month = None
            if CENTER_MODE == "mslp":
                mslp_month = self._load_mslp(nc_file_path, times, lons, lats)
                inside = etc_data[0] > 0.5                  # a roll error would invert this
                print(f"    MSLP check, first step: {mslp_month[0][inside].mean() / 100:.1f} hPa inside "
                      f"the mask vs {mslp_month[0][~inside].mean() / 100:.1f} hPa outside")

            for t_idx, time in enumerate(times):
                curr_centers = self.find_etc_centers(
                    etc_data[t_idx], lons, lats,
                    mslp_month[t_idx] if mslp_month is not None else None)
                total_detections += len(curr_centers)

                claimed_curr = set()

                if LINK_MODE == "global":
                    self._link_global(active_tracks, curr_centers, time, claimed_curr)

                for track_id, track_info in (list(active_tracks.items())
                                             if LINK_MODE == "legacy" else []):
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
                        eff_max = self.max_distance_km * (1 + gap_count * GAP_GROWTH)
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
        if self.rejected:
            print("  Rejected by rule : " + ", ".join(
                f"{k} {v}" for k, v in sorted(self.rejected.items(), key=lambda kv: -kv[1])))

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
            if NON_DEFAULT:          # written only for non-shipped settings: the default file stays identical
                f.write(f"# design switches        : ETC_MAX_GAP_STEPS={MAX_GAP_STEPS} "
                        f"ETC_FILTER_PROFILE={FILTER_PROFILE} ETC_LONWRAP={int(LONWRAP)} "
                        f"ETC_LINK_MODE={LINK_MODE} ETC_GAP_RADIUS_GROWTH={GAP_GROWTH} "
                        f"ETC_CENTER_MODE={CENTER_MODE} min_net_km={self.min_net_km}\n")
                if CENTER_MODE == "mslp":
                    f.write(f"# closed contour         : depth {CC_DEPTH_PA:.0f} Pa within {CC_RADIUS_DEG:g} deg, "
                            f"smoothing {CC_SIGMA_KM:.0f} km, centroid fallback for blobs >= "
                            f"{CENTER_FALLBACK_MIN} cells; INTENSITY column = MSLP (hPa)\n")
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