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