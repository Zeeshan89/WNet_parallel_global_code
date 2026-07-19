import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.morphology import binary_opening, binary_dilation, binary_erosion, binary_closing
from scipy.ndimage import binary_fill_holes, binary_dilation as scipy_binary_dilation
from skimage.filters import gaussian
from skimage.measure import regionprops
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
from skimage.transform import resize


# =============================================================================
# MCS-ONLY POST-PROCESSING
#   Raw model blobs (no physics/model filtering) → minimum cloud-shield area
#   (60,000 km²) → temporal-overlap persistence (50–90% with prev/next frame).
#   Applied to the full monthly (time, lat, lon) stack in test.py.
# =============================================================================
_EARTH_R_KM = 6371.0


def _pixel_area_km2(latitudes, longitudes):
    """Per-grid-cell area (km²) on a regular lat/lon grid, latitude-weighted.

    cell_area(lat) = R² · Δlon_rad · Δlat_rad · cos(lat)
    Returned as a 2-D (nlat, nlon) array (longitude-independent, broadcast).
    cos() makes the result invariant to ascending/descending latitude order.
    """
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    dlat_deg = np.abs(np.mean(np.diff(lats))) if lats.size > 1 else 0.25
    dlon_deg = np.abs(np.mean(np.diff(lons))) if lons.size > 1 else 0.25
    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)
    cell_per_lat = (_EARTH_R_KM ** 2) * dlon_rad * dlat_rad * np.cos(np.deg2rad(lats))
    cell_per_lat = np.clip(cell_per_lat, 0.0, None)        # guard tiny negatives at |lat|→90
    return np.repeat(cell_per_lat[:, None], lons.size, axis=1).astype(np.float32)


def _label_lonwrap(binary):
    """8-connected labeling that merges blobs across the ±180° longitude seam.

    The output grid's first and last columns are physically adjacent, so a blob
    straddling the antimeridian is otherwise split in two. Mirrors the union-find
    merge in evaluations/ar_mars_eval/build_ar_object_metrics.py:label_wrap.
    Returns (labels, n_unique).
    """
    structure = np.ones((3, 3), dtype=int)
    labels, n = ndimage.label(binary, structure=structure)
    if n <= 1:
        return labels, n

    left, right = labels[:, 0], labels[:, -1]
    pairs = set()
    for i in range(labels.shape[0]):
        l, r = left[i], right[i]
        if l > 0 and r > 0 and l != r:
            pairs.add((min(l, r), max(l, r)))
    if not pairs:
        return labels, n

    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(x) for x in range(n + 1)])
    uniq = sorted(set(roots[1:].tolist()))
    remap = {old: new for new, old in enumerate(uniq, start=1)}
    out = np.zeros_like(labels)
    for x in range(1, n + 1):
        out[labels == x] = remap[roots[x]]
    return out, len(uniq)


def _filter_min_area(mask2d, area2d, min_area_km2):
    """Drop connected blobs whose total (latitude-weighted) area < min_area_km2."""
    binary = mask2d > 0
    labels, n = _label_lonwrap(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=np.float32)
    ids = np.arange(1, n + 1)
    areas = ndimage.sum(area2d, labels, ids)
    keep_ids = ids[np.asarray(areas) >= min_area_km2]
    if keep_ids.size == 0:
        return np.zeros_like(binary, dtype=np.float32)
    return np.isin(labels, keep_ids).astype(np.float32)


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two (lon, lat) points."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def track_mcs_timeseries(mcs_stack, latitudes, longitudes,
                         min_area_km2=60000.0, link_overlap=0.5,
                         min_lifetime_steps=2, min_net_displacement_km=400.0,
                         verbose=True):
    """Full MCS criteria + overlap tracking → a storm-ID field (like Feng's
    `cloudtracknumber`), so the SHARED benchmark metric code can compute true
    ID-based storm-hours for WNet exactly as for the other methods.

    Applied to a monthly (time, lat, lon) stack of RAW model MCS blobs:
      1. Minimum cloud-shield area: drop blobs < `min_area_km2` per frame
         (latitude-weighted area; ±180° seam merged via `_label_lonwrap`).
      2. Overlap tracking: a blob at t inherits the ID of the previous-frame blob
         it overlaps most, when that overlap (intersection / area of the SMALLER
         of the two blobs) ≥ `link_overlap`; otherwise it starts a new track ID.
         Merges/splits resolve to the maximum-overlap parent.
      3. Minimum lifetime: drop whole tracks present in fewer than
         `min_lifetime_steps` frames (2 ≈ 6 h at 6-hourly cadence).
      4. Minimum net propagation: drop tracks whose start→end displacement is
         below `min_net_displacement_km`. MCSs propagate; orographic convection
         is terrain-anchored, so this removes quasi-stationary terrain-locked
         systems that the area/lifetime tests admit. Set to None/0 to disable.

    Tracking is per monthly file; IDs restart each month. That is sufficient for
    the monthly storm-hours metric (it groups by (month, id)); a system crossing
    a month boundary is counted in each month regardless.

    Args:
        mcs_stack : (T, H, W) array of raw MCS detections (0/1).
        latitudes, longitudes : 1-D coordinate arrays of the output grid.
    Returns:
        (binary, ids) where
          binary : (T, H, W) float32, 1 where a surviving track is present.
          ids    : (T, H, W) int32, storm track ID per pixel (0 = none).
    """
    stack = np.asarray(mcs_stack)
    if stack.ndim != 3:
        raise ValueError(f"mcs_stack must be (T, H, W); got shape {stack.shape}")
    T, H, W = stack.shape
    area2d = _pixel_area_km2(latitudes, longitudes)   # used for area-weighted overlap below

    # ── Step 1: optional per-frame minimum-area filter ───────────────────────
    # min_area_km2 None/0 → keep RAW model blobs (no area criterion).
    if min_area_km2 and min_area_km2 > 0:
        binar = np.zeros((T, H, W), dtype=np.uint8)
        for t in range(T):
            binar[t] = (_filter_min_area(stack[t], area2d, min_area_km2) > 0).astype(np.uint8)
    else:
        binar = (stack > 0).astype(np.uint8)

    # ── Step 2: overlap-based temporal linking → global track IDs ─────────────
    ids = np.zeros((T, H, W), dtype=np.int32)
    next_id = 1
    prev_ids = None                                  # previous frame's global-ID field
    for t in range(T):
        lab, n = _label_lonwrap(binar[t] > 0)
        out_t = np.zeros((H, W), dtype=np.int32)
        if n == 0:
            ids[t] = out_t
            prev_ids = out_t
            continue

        cur_idx = np.arange(1, n + 1)
        cur_area = np.asarray(ndimage.sum(area2d, lab, cur_idx), dtype=np.float64)

        assigned = {}                                # current label -> global id
        if prev_ids is not None and prev_ids.max() > 0:
            pu = np.unique(prev_ids)
            pu = pu[pu > 0]
            prev_area = dict(zip(pu.tolist(),
                                 np.asarray(ndimage.sum(area2d, prev_ids, pu), dtype=np.float64)))

            overlap = lab > 0
            overlap &= prev_ids > 0
            if overlap.any():
                cf = lab[overlap].astype(np.int64)
                pf = prev_ids[overlap].astype(np.int64)
                af = area2d[overlap]
                base = int(prev_ids.max()) + 1
                key = cf * base + pf
                uk, inv = np.unique(key, return_inverse=True)
                inter = np.zeros(len(uk), dtype=np.float64)
                np.add.at(inter, inv, af)
                best = {}                            # cur label -> (inter_area, prev_id)
                for k, ia in zip(uk.tolist(), inter):
                    c = k // base
                    p = k % base
                    if c not in best or ia > best[c][0]:
                        best[c] = (ia, p)
                for c, (ia, p) in best.items():
                    denom = min(cur_area[c - 1], prev_area[p])
                    if denom > 0 and (ia / denom) >= link_overlap:
                        assigned[c] = int(p)

        for c in cur_idx:
            gid = assigned.get(int(c))
            if gid is None:
                gid = next_id
                next_id += 1
            out_t[lab == c] = gid

        ids[t] = out_t
        prev_ids = out_t

    # ── Step 3: minimum-lifetime filter (drop short-lived tracks) ─────────────
    if min_lifetime_steps > 1 and next_id > 1:
        life = np.zeros(next_id, dtype=np.int32)     # frames each ID appears in
        for t in range(T):
            u = np.unique(ids[t])
            u = u[u > 0]
            life[u] += 1
        short = np.nonzero(life < min_lifetime_steps)[0]
        if short.size:
            drop = np.isin(ids, short)
            ids[drop] = 0

    # ── Step 4: minimum net propagation (drop quasi-stationary systems) ───────
    # An MCS PROPAGATES; orographic convection is terrain-anchored. Measured on
    # this pipeline's own tracks (ACC 2005-2010), winter Cascades systems move a
    # median 275 km at 3.7 m/s while summer Midwest MCSs move 608 km at 6.3 m/s
    # — same duration, 2.2x the displacement. That separates the two regimes on a
    # different axis from rain rate / convective fraction, which is why it works
    # where an 8-combo threshold sweep (core 1.0-3.0, cf 0.7-0.9) could not.
    #
    # This is the MCS analogue of the ETC tracker's kinematic criteria
    # (min_path_km / min_avg_propagation_ms / min_net_east_km), and it is applied
    # AFTER linking for the same reason: it is a property of a whole track.
    #
    # Net (start->end) displacement, not path length, so a system that wanders
    # and returns is correctly rejected.
    if min_net_displacement_km and min_net_displacement_km > 0 and next_id > 1:
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)
        first = {}   # id -> (lon, lat) at first appearance
        last = {}    # id -> (lon, lat) at last appearance
        for t in range(T):
            present = np.unique(ids[t])
            present = present[present > 0]
            for gid in present:
                rows, cols = np.nonzero(ids[t] == gid)
                # centroid in index space -> geographic coords
                clat = lat[int(round(rows.mean()))]
                clon = lon[int(round(cols.mean()))]
                if gid not in first:
                    first[gid] = (clon, clat)
                last[gid] = (clon, clat)

        stationary = [gid for gid in first
                      if _haversine_km(*first[gid], *last[gid]) < min_net_displacement_km]
        if stationary:
            ids[np.isin(ids, stationary)] = 0

    binary = (ids > 0).astype(np.float32)

    if verbose:
        n_tracks = int(np.unique(ids[ids > 0]).size)
        area_txt = (f"{min_area_km2:.0f} km² area + " if (min_area_km2 and min_area_km2 > 0)
                    else "no area filter, ")
        life_txt = (f"{min_lifetime_steps}-step lifetime"
                    if min_lifetime_steps and min_lifetime_steps > 1 else "no lifetime filter")
        prop_txt = (f" + {min_net_displacement_km:.0f} km propagation"
                    if (min_net_displacement_km and min_net_displacement_km > 0) else "")
        print(f"[MCS] raw pixels: {int((stack > 0).sum())} -> "
              f"{area_txt}overlap-track + {life_txt}{prop_txt}: {int((binary > 0).sum())} pixels, "
              f"{n_tracks} tracks")
    return binary, ids


def _to_numpy_float32(arr):
    """
    Safely convert any array-like (torch.Tensor, xarray, object array, etc.)
    to a contiguous numpy float32 ndarray.
    Returns None if input is None.
    """
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if hasattr(arr, 'values'):
        arr = arr.values
    arr = np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(arr)


def validate_eye_structure_enhanced(component, km_per_pixel, min_radius_km=5, max_radius_km=100):
    props = regionprops(component.astype(int))
    if not props:
        return False, 0, 0, 0
    prop = props[0]
    area = prop.area
    perimeter = prop.perimeter
    equivalent_radius_pixels = np.sqrt(area / np.pi)
    radius_km = equivalent_radius_pixels * km_per_pixel
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
    compactness = area / bbox_area if bbox_area > 0 else 0
    size_valid = min_radius_km <= radius_km <= max_radius_km
    circular_enough = circularity > 0.05
    if prop.minor_axis_length > 0:
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length
        not_too_elongated = aspect_ratio < 8.0
    else:
        not_too_elongated = True
    compact_enough = compactness > 0.1
    is_valid_eye = size_valid and (circular_enough or compact_enough) and not_too_elongated
    return is_valid_eye, radius_km, circularity, compactness


def validate_mcs_structure(component, km_per_pixel, min_radius_km=5, max_radius_km=1000):
    props = regionprops(component.astype(int))
    if not props:
        return False, 0, 0, 0, 0, 0
    prop = props[0]
    area = prop.area
    perimeter = prop.perimeter
    equivalent_radius_pixels = np.sqrt(area / np.pi)
    radius_km = equivalent_radius_pixels * km_per_pixel
    area_km2 = area * (km_per_pixel ** 2)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
    compactness = area / bbox_area if bbox_area > 0 else 0
    if prop.minor_axis_length > 0:
        elongation = prop.major_axis_length / prop.minor_axis_length
    else:
        elongation = float('inf')
    size_valid = min_radius_km <= radius_km <= max_radius_km
    shape_reasonable = circularity > 0.05
    reasonably_compact = compactness > 0.08
    area_reasonable = 500 <= area_km2 <= 100000
    is_valid_mcs = (size_valid and area_reasonable and
                    (shape_reasonable or reasonably_compact or elongation < 5.0))
    return is_valid_mcs, radius_km, circularity, compactness, elongation, area_km2


def validate_etc_structure(component, km_per_pixel, min_radius_km=100, max_radius_km=2000):
    props = regionprops(component.astype(int))
    if not props:
        return False, 0, 0, 0, 0
    prop = props[0]
    area = prop.area
    perimeter = prop.perimeter
    equivalent_radius_pixels = np.sqrt(area / np.pi)
    radius_km = equivalent_radius_pixels * km_per_pixel
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
    compactness = area / bbox_area if bbox_area > 0 else 0
    if prop.minor_axis_length > 0:
        elongation = prop.major_axis_length / prop.minor_axis_length
    else:
        elongation = float('inf')
    size_valid = min_radius_km <= radius_km <= max_radius_km
    shape_reasonable = circularity > 0.02
    not_extremely_elongated = elongation < 15.0
    reasonably_compact = compactness > 0.05
    is_valid_etc = size_valid and (shape_reasonable or reasonably_compact) and not_extremely_elongated
    return is_valid_etc, radius_km, circularity, compactness, elongation


def enhanced_post_process_ar_segmentation(
    seg_map, storm_class, ivt_data, vor_data=None, mslp_data=None, percentile_threshold=None,
    min_size=50, min_aspect_ratio=2.0,
    slope_threshold=0.3, equatorial_band_size=0.3,
    km_per_pixel=50, min_length_km=2000, min_span_km=1000,
    vor_threshold=0.00005, mslp_eye_threshold=1005, lat_bounds=None,
    gradient_threshold=0.3, temporal_weight=0.1,
    preprocessed_masks=None, sample_data=None):

    processed_map = seg_map.copy()
    target_shape = processed_map.shape

    print(f"Enhanced processing storm class {storm_class} with shape {target_shape}")
    print(f"Initial segmentation map sum: {np.sum(processed_map)}")

    # --- SAFELY CONVERT sample_data TO NUMPY ---
    if sample_data is not None:
        if isinstance(sample_data, torch.Tensor):
            sample_data = sample_data.detach().cpu().numpy()
        elif hasattr(sample_data, 'values'):
            sample_data = sample_data.values
        sample_data = np.asarray(sample_data, dtype=np.float32)

    # -------------------------------------------------------------------------
    # FIX 1: resize_to_target ALWAYS calls _to_numpy_float32 first.
    # This prevents silent failures when cached data arrives as xarray/object
    # arrays, which caused zoom() to produce garbage interpolated values and
    # created the "connected circular blob" artefacts in MCS output.
    # -------------------------------------------------------------------------
    def resize_to_target(arr, target_shape, order=1):
        if arr is None:
            return None
        arr = _to_numpy_float32(arr)          # <-- CRITICAL: safe conversion first
        if arr.shape == target_shape:
            return arr
        zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        return zoom(arr, zoom_factors, order=order)

    ivt_data_resized   = resize_to_target(ivt_data,  target_shape, order=1) if ivt_data  is not None else None
    vor_data_resized   = resize_to_target(vor_data,  target_shape, order=1) if vor_data  is not None else None
    mslp_data_resized  = resize_to_target(mslp_data, target_shape, order=1) if mslp_data is not None else None

    if percentile_threshold is not None:
        if isinstance(percentile_threshold, (int, float)):
            percentile_threshold_resized = percentile_threshold
        else:
            percentile_threshold_resized = resize_to_target(percentile_threshold, target_shape, order=1)
    else:
        percentile_threshold_resized = None

    # --- CHANNEL EXTRACTORS ---

    def extract_hurricane_channels(sample_data, target_shape):
        if sample_data is None or sample_data.shape[0] < 14:
            return None
        channels = {}
        channels['low_pressure']    = resize_to_target(_to_numpy_float32(sample_data[2]),  target_shape, order=0)
        channels['high_vorticity']  = resize_to_target(_to_numpy_float32(sample_data[3]),  target_shape, order=0)
        channels['high_wind_speed'] = resize_to_target(_to_numpy_float32(sample_data[4]),  target_shape, order=0)
        channels['strict']          = resize_to_target(_to_numpy_float32(sample_data[5]),  target_shape, order=0)
        channels['wind_binary']     = resize_to_target(_to_numpy_float32(sample_data[6]),  target_shape, order=0)
        return channels

    def extract_etc_channels(sample_data, target_shape):
        if sample_data is None or sample_data.shape[0] < 14:
            print(f"Warning: Expected at least 14 channels for ETC detection, got "
                  f"{sample_data.shape[0] if sample_data is not None else 'None'}")
            return None
        channels = {}
        channels['etc_low_pressure'] = resize_to_target(_to_numpy_float32(sample_data[7]),  target_shape, order=0)
        channels['etc_vorticity']    = resize_to_target(_to_numpy_float32(sample_data[8]),  target_shape, order=0)
        channels['etc_wind_speed']   = resize_to_target(_to_numpy_float32(sample_data[9]),  target_shape, order=0)
        channels['etc_adaptive']     = resize_to_target(_to_numpy_float32(sample_data[10]), target_shape, order=1)
        if channels['etc_adaptive'] is not None:
            channels['etc_high_confidence'] = (channels['etc_adaptive'] >= 0.90).astype(np.float32)
            high_conf_pixels     = np.sum(channels['etc_high_confidence'])
            total_adaptive_pixels = np.sum(channels['etc_adaptive'] > 0.1)
            print(f"Extracted ETC high-confidence regions: {high_conf_pixels} pixels (>=0.90 intensity)")
            print(f"  Filtered from {total_adaptive_pixels} total adaptive pixels")
            print(f"  High-confidence ratio: {high_conf_pixels/max(total_adaptive_pixels,1)*100:.1f}%")
        else:
            channels['etc_high_confidence'] = None
        return channels

    def extract_mcs_channels(sample_data, target_shape):
        """
        Extract MCS channels.

        FIX 2: Binary masks (mcs_boundary, mcs_core_pixels, mcs_blobs) MUST be
        resized with order=0 (nearest-neighbour).  Using order=1 (bilinear) on
        a 0/1 mask produces fractional edge values (e.g. 0.3, 0.7) that are
        then thresholded with > 0, artificially fattening every blob boundary
        and merging nearby blobs into large connected circular shapes.

        FIX 3: After resizing, binarise each mask explicitly so that any
        floating-point residue from the zoom is removed before downstream use.
        """
        if sample_data is None or sample_data.shape[0] < 14:
            return None

        channels = {}

        # ----- FIX 2: order=0 for ALL binary MCS masks -----
        raw_boundary    = _to_numpy_float32(sample_data[11])
        raw_core_pixels = _to_numpy_float32(sample_data[13])
        raw_blobs       = _to_numpy_float32(sample_data[12])

        channels['mcs_boundary']    = resize_to_target(raw_boundary,    target_shape, order=0)
        channels['mcs_core_pixels'] = resize_to_target(raw_core_pixels, target_shape, order=0)
        channels['mcs_blobs']       = resize_to_target(raw_blobs,       target_shape, order=0)

        # ----- FIX 3: explicit binarisation after resize -----
        # Cached data may store masks as floats (0.0 / 1.0) or uint8; round-trip
        # through zoom can introduce tiny non-zero values.  Threshold at 0.5 to
        # recover a clean binary mask regardless of the source dtype.
        for key in ('mcs_boundary', 'mcs_core_pixels', 'mcs_blobs'):
            if channels[key] is not None:
                channels[key] = (channels[key] >= 0.5).astype(np.float32)

        print(f"MCS channel pixel counts after binarisation:")
        for key in ('mcs_boundary', 'mcs_core_pixels', 'mcs_blobs'):
            if channels[key] is not None:
                print(f"  {key}: {int(np.sum(channels[key]))} pixels")

        return channels

    # =========================================================================
    # STORM-CLASS DISPATCH
    # =========================================================================

    if storm_class == 1:  # AR
        print("Processing AR detection...")
        if percentile_threshold_resized is not None:
            threshold_mask = (ivt_data_resized > percentile_threshold_resized).astype(np.uint8)
        else:
            threshold_mask = (ivt_data_resized > 250).astype(np.uint8)

    elif storm_class == 2:  # Hurricane
        print("Processing hurricane detection with enhanced channel-based approach...")
        hurricane_channels = extract_hurricane_channels(sample_data, target_shape)

        if hurricane_channels is None:
            print("Warning: No hurricane channels available, falling back to basic detection")
            threshold_mask = (processed_map > 0).astype(np.uint8)
        else:
            # USE STRICT MASK ONLY — do not OR with raw segmentation
            # strict = channel 5 = dilated centres AND (vorticity OR wind)
            # This is already the tightest hurricane signal available
            if hurricane_channels['strict'] is not None:
                threshold_mask = (hurricane_channels['strict'] > 0).astype(np.uint8)
                print(f"Hurricane mask from strict channel: {np.sum(threshold_mask)} pixels")
            else:
                threshold_mask = np.zeros(target_shape, dtype=np.uint8)
                print("Warning: strict channel empty — no hurricane detections")

            # REMOVED: seg_positive OR — this was inflating 238 → 2114
            print(f"Total hurricane threshold mask: {np.sum(threshold_mask)} pixels")

    elif storm_class == 3:  # ETC
        print("Processing ETC detection with HIGH-CONFIDENCE focus (>=0.90 intensity)...")
        etc_channels = extract_etc_channels(sample_data, target_shape)

        if etc_channels is None or etc_channels['etc_high_confidence'] is None:
            print("Warning: No ETC high-confidence channels available")
            threshold_mask = np.zeros(target_shape, dtype=np.uint8)
        else:
            high_conf_mask = etc_channels['etc_high_confidence']
            threshold_mask = high_conf_mask.astype(np.uint8)
            print(f"Base high-confidence mask: {np.sum(threshold_mask)} pixels (>=0.50 intensity)")

            seg_positive = (processed_map > 0).astype(np.uint8)
            valid_predictions = np.logical_and(seg_positive, high_conf_mask).astype(np.uint8)
            threshold_mask = np.logical_or(threshold_mask, valid_predictions).astype(np.uint8)

            # --- ETC model predictions analysis and validation DISABLED for speed ---
            # All high-confidence ETC predictions kept without per-component validation

            print(f"HIGH-CONFIDENCE ETC processing complete: {np.sum(threshold_mask)} pixels")

    elif storm_class == 4:
        print("Processing MCS detection with two-threshold blob approach...")
        mcs_channels = extract_mcs_channels(sample_data, target_shape)

        if mcs_channels is None:
            print("Warning: No MCS channels available, falling back to basic detection")
            threshold_mask = (processed_map > 0).astype(np.uint8)
        else:
            threshold_mask = np.zeros(target_shape, dtype=np.uint8)

            if mcs_channels['mcs_core_pixels'] is not None:
                core_mask = mcs_channels['mcs_core_pixels'].astype(np.uint8)
                
                # Dilate core mask slightly to bridge spatial offset between
                # physics channels and model predictions (~2-3 pixels ≈ 50-75 km)
                from scipy.ndimage import binary_dilation as _bdil
                core_dilated = _bdil(
                    core_mask.astype(bool),
                    structure=np.ones((5, 5))   # 5×5 ≈ 125 km proximity window
                ).astype(np.uint8)
                
                seg_positive = (processed_map > 0).astype(np.uint8)
                
                # Keep model pixels that are near a physics core blob
                threshold_mask = np.logical_and(core_dilated, seg_positive).astype(np.uint8)
                
                # Also always keep the core blob itself regardless of model agreement
                # (physics signal is high-confidence even if model missed it)
                threshold_mask = np.logical_or(threshold_mask, core_mask).astype(np.uint8)
                
                print(f"Core mask pixels: {np.sum(core_mask)}")
                print(f"Dilated proximity gate pixels: {np.sum(core_dilated)}")
                print(f"After proximity-gated union: {np.sum(threshold_mask)}")

    else:
        raise ValueError(f"Unsupported storm class: {storm_class}. "
                         f"Supported classes are 1 (AR), 2 (Hurricane), 3 (ETC), 4 (MCS)")

    processed_map = threshold_mask.copy()

    # --- MINIMAL MORPHOLOGICAL CLEAN-UP ---
    if storm_class == 2:
        # No fill_holes — hurricane centres are point-like, filling creates blobs
        # Just remove isolated single pixels (noise)
        if np.any(processed_map):
            processed_map = binary_opening(
                processed_map, footprint=np.ones((3, 3))
            ).astype(np.uint8)

    elif storm_class == 3:
        if np.any(processed_map):
            small_holes_filled = binary_fill_holes(
                processed_map, structure=np.ones((3, 3))).astype(np.uint8)
            holes_added = np.sum(small_holes_filled) - np.sum(processed_map)
            if holes_added < np.sum(processed_map) * 0.1:
                processed_map = small_holes_filled
        print(f"After minimal ETC morphological operations: {np.sum(processed_map)}")


    elif storm_class == 4:
        if np.any(processed_map):
            # Only remove truly isolated single pixels, not small clusters
            processed_map = binary_opening(
                processed_map, footprint=np.ones((2, 2))  # reduced from (3,3)
            ).astype(np.uint8)

    elif storm_class == 1:
        if np.any(processed_map):
            processed_map = binary_opening(
                processed_map, footprint=np.ones((3, 3))).astype(np.uint8)

    # --- FINAL COMPONENT VALIDATION ---
    labeled_array, num_features = ndimage.label(processed_map, structure=np.ones((3, 3)))
    if num_features == 0:
        print("No features found after morphological operations")
        return np.zeros_like(processed_map)

    components_kept = 0
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        component_size = np.sum(component)

        if storm_class == 1:
            image_height = processed_map.shape[0]
            equatorial_band_height = int(image_height * equatorial_band_size)
            equatorial_center = image_height // 2
            equatorial_top    = equatorial_center - equatorial_band_height // 2
            equatorial_bottom = equatorial_center + equatorial_band_height // 2

            if component_size < min_size:
                processed_map[component] = 0
                continue

            props = regionprops(component.astype(int))
            if not props:
                continue
            prop = props[0]

            y_start, x_start, y_stop, x_stop = prop.bbox
            height = y_stop - y_start

            equatorial_coverage = 0
            if y_start <= equatorial_bottom and y_stop >= equatorial_top:
                overlap_top    = max(y_start, equatorial_top)
                overlap_bottom = min(y_stop, equatorial_bottom)
                equatorial_coverage = (overlap_bottom - overlap_top) / height

            aspect_ratio = (prop.major_axis_length / prop.minor_axis_length
                            if prop.minor_axis_length > 0 else float('inf'))
            slope = abs(math.tan(prop.orientation))
            length_pixels = prop.major_axis_length

            y0, x0 = prop.centroid
            dx = 0.5 * length_pixels * math.cos(prop.orientation)
            dy = 0.5 * length_pixels * math.sin(prop.orientation)
            span_pixels = math.sqrt((2*dx)**2 + (2*dy)**2)

            length_km = length_pixels * km_per_pixel
            span_km   = span_pixels   * km_per_pixel

            failed_physical_constraints = (
                length_km < min_length_km or
                aspect_ratio < min_aspect_ratio or
                span_km < min_span_km
            )

            if equatorial_coverage > 0.5:
                if (failed_physical_constraints or
                        slope < slope_threshold * 1.5 or
                        length_km < min_length_km * 1.2 or
                        span_km < min_span_km * 1.2):
                    if ivt_data_resized is not None:
                        component_ivt = ivt_data_resized[component]
                        if percentile_threshold_resized is not None:
                            tv = (percentile_threshold_resized if isinstance(percentile_threshold_resized, (int, float))
                                  else np.mean(percentile_threshold_resized[component]))
                            if np.mean(component_ivt) < 1.5 * tv:
                                processed_map[component] = 0
                                continue
                        else:
                            if np.mean(component_ivt) < 500:
                                processed_map[component] = 0
                                continue
            else:
                if failed_physical_constraints or slope < slope_threshold:
                    if ivt_data_resized is not None:
                        component_ivt = ivt_data_resized[component]
                        if percentile_threshold_resized is not None:
                            tv = (percentile_threshold_resized if isinstance(percentile_threshold_resized, (int, float))
                                  else np.mean(percentile_threshold_resized[component]))
                            if np.mean(component_ivt) < 1.2 * tv:
                                processed_map[component] = 0
                                continue
                        else:
                            if np.mean(component_ivt) < 350:
                                processed_map[component] = 0
                                continue

        elif storm_class == 2:
            # --- Hurricane final validation DISABLED for speed ---
            pass

        elif storm_class == 3:
            # --- ETC final validation DISABLED for speed ---
            pass

        elif storm_class == 4:
            # --- MCS final validation DISABLED for speed ---
            pass

        components_kept += 1

    print(f"Final components kept: {components_kept}")
    return processed_map