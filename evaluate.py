#!/usr/bin/env python
"""
evaluate.py — Offline benchmark evaluator for the WNet storm-detection model.

Consumes prediction NetCDFs written by test.py (one per month per class:
<stem>_AR_detection.nc, _Hurricane_detection.nc, _ETCs_detection.nc,
_MCS_detection.nc) and produces paper-aligned skill metrics against
independent references:

  Class      | Track skill                              | Pixel skill
  -----------|------------------------------------------|---------------------
  Hurricane  | POD / FAR / HSS / Bias vs IBTrACS        | IoU/Dice/F1 vs heuristic
  ETC        | (TODO: vs TempestExtremes track files)   | IoU/Dice/F1 vs heuristic
  AR         | — (no track concept)                     | IoU/Dice/F1 vs IVT mask
  MCS        | — (object-level optional, deferred)      | IoU/Dice/F1 vs BT mask

Outputs (under --out-dir, tagged with --tag):
  * summary_<tag>.txt   human-readable report (paper-quotable)
  * summary_<tag>.csv   one row per (category, metric) for downstream plotting
  * tracks_<tag>.json   raw predicted-vs-observed track matches (for audit)

Usage:
  python evaluate.py \\
    --predictions-dir inference_runs/ema_checkpoint_epoch_54_primary_2006_2024 \\
    --start-year 2006 --end-year 2024 \\
    --out-dir evaluations/trail11_e54_primary \\
    --tag primary

Pixel skill uses the SAME heuristic masks as training (channel-wise targets
from the input batch's aux fields). For a truly independent comparison, swap
the heuristic mask loader for a published catalog (Guan-Waliser AR, Feng MCS)
— interface kept small for that purpose.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label as cc_label


# ─────────────────────────────────────────────────────────────────────────────
# Constants — tweak if the tracker / matcher tolerances need tuning
# ─────────────────────────────────────────────────────────────────────────────
EARTH_R_KM            = 6371.0
DEFAULT_PROB_THRESH   = 0.5      # binarisation threshold for detection prob
MIN_BLOB_PIXELS       = 5        # discard < this in tracker
MAX_LINK_JUMP_KM      = 500.0    # max centroid displacement between 6h frames
MIN_TRACK_HOURS       = 24       # discard tracks shorter than this
TRACK_DIST_MATCH_KM   = 300.0    # mean centroid distance for "match"
TRACK_OVERLAP_HOURS   = 12       # min time overlap for "match"
HOURS_PER_FRAME       = 6        # ERA5 6-hourly cadence

# IBTrACS basin bounding boxes (rough — for stratification only)
BASINS = {
    "NA": dict(lat=(  0,  60), lon=(-100,  -10)),
    "EP": dict(lat=(  0,  50), lon=(-180, -100)),
    "WP": dict(lat=(  0,  50), lon=( 100,  180)),
    "NI": dict(lat=(  0,  40), lon=(  40,  100)),
    "SI": dict(lat=(-50,   0), lon=(  20,  130)),
    "SP": dict(lat=(-50,   0), lon=( 130,  230)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(a))


def basin_of(lat, lon):
    """Return basin code for a point. 'Other' if outside known basins."""
    for code, box in BASINS.items():
        if box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]:
            return code
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def find_prediction_file(predictions_dir, year, month, class_name):
    """Locate the per-month prediction NetCDF. Returns path or None."""
    # test.py's save_results_to_netcdf names files relative to the input
    # filename's stem + "_<class>_detection.nc". The exact prefix depends on
    # the input, so we glob.
    for pattern in (
        f"*{year:04d}{month:02d}*_{class_name}_detection.nc",
        f"*_{year:04d}_{month:02d}_{class_name}_detection.nc",
    ):
        matches = sorted(glob.glob(os.path.join(predictions_dir, "**", pattern),
                                   recursive=True))
        if matches:
            return matches[0]
    return None


def load_prediction(predictions_dir, year, month, class_name):
    """Return (probs ndarray (T, H, W), times, lats, lons) or (None, ...)."""
    path = find_prediction_file(predictions_dir, year, month, class_name)
    if path is None:
        return None, None, None, None
    ds = xr.open_dataset(path)
    var_name = f"{class_name}_detection"
    if var_name not in ds.data_vars:
        # Try a few common variants the save function may have written
        for cand in (f"{class_name}_detection", f"{class_name}",
                     class_name.lower() + "_detection"):
            if cand in ds.data_vars:
                var_name = cand
                break
        else:
            ds.close()
            return None, None, None, None
    probs = ds[var_name].values
    times = pd.to_datetime(ds.time.values)
    lats  = ds.latitude.values
    lons  = ds.longitude.values
    ds.close()
    return probs, times, lats, lons


def load_ibtracs_tracks(ibtracs_path, start_year, end_year,
                        wind_threshold_kt=0.0, lat_band=(-50, 50)):
    """
    Read IBTrACS NetCDF and return one dict per track:
      {sid, name, basin, times[N], lats[N], lons[N], wind_kt[N], sshs[N]}
    Restricted to start_year..end_year + lat_band.
    """
    ds = xr.open_dataset(ibtracs_path)

    def _arr(name, default=None):
        return ds[name].values if name in ds else default

    sid_all   = _arr("sid")
    name_all  = _arr("name")
    times_all = _arr("time")
    lat_all   = _arr("usa_lat",  default=_arr("lat"))
    lon_all   = _arr("usa_lon",  default=_arr("lon"))
    wind_all  = _arr("usa_wind", default=None)
    sshs_all  = _arr("usa_sshs", default=None)
    basin_all = _arr("basin",    default=None)

    n_storms = sid_all.shape[0]
    tracks = []
    for i in range(n_storms):
        sid  = bytes(sid_all[i]).decode("utf-8", errors="ignore").strip() \
               if isinstance(sid_all[i], (bytes, np.bytes_)) else str(sid_all[i])
        name = bytes(name_all[i]).decode("utf-8", errors="ignore").strip() \
               if isinstance(name_all[i], (bytes, np.bytes_)) else str(name_all[i])
        basin = "?"
        if basin_all is not None:
            b0 = basin_all[i, 0] if basin_all.ndim > 1 else basin_all[i]
            basin = bytes(b0).decode("utf-8", errors="ignore").strip() \
                    if isinstance(b0, (bytes, np.bytes_)) else str(b0)

        ts_row   = times_all[i]
        lat_row  = lat_all[i]
        lon_row  = lon_all[i]
        wind_row = wind_all[i] if wind_all is not None else np.full_like(lat_row, np.nan, dtype=float)
        sshs_row = sshs_all[i] if sshs_all is not None else np.full_like(lat_row, np.nan, dtype=float)

        times, lats, lons, winds, sshss = [], [], [], [], []
        for t, la, lo, w, s in zip(ts_row, lat_row, lon_row, wind_row, sshs_row):
            if pd.isna(t) or pd.isna(la) or pd.isna(lo):
                continue
            ts = pd.Timestamp(t).to_pydatetime()
            if ts.year < start_year or ts.year > end_year:
                continue
            la_f = float(la); lo_f = float(lo)
            if not (lat_band[0] <= la_f <= lat_band[1]):
                continue
            w_f = float(w) if not pd.isna(w) else np.nan
            if not np.isnan(w_f) and w_f < wind_threshold_kt:
                continue
            times.append(ts); lats.append(la_f); lons.append(lo_f)
            winds.append(w_f)
            sshss.append(float(s) if not pd.isna(s) else np.nan)

        if len(times) < 2:
            continue
        tracks.append(dict(
            sid=sid, name=name, basin=basin,
            times=times, lats=lats, lons=lons,
            wind_kt=winds, sshs=sshss,
        ))
    ds.close()
    return tracks


# ─────────────────────────────────────────────────────────────────────────────
# Loaders for already-extracted track .txt files (skip the internal tracker
# when these exist — usually faster + more battle-tested than running our own
# centroid+linker on freshly inferred probabilities)
# ─────────────────────────────────────────────────────────────────────────────

def load_hurricane_tracks_from_dir(hurr_dir, start_year, end_year):
    """
    Parse per-year CONUS_hurricane_tracks_detailed_YYYY.txt files from
    Hurricanes_WNet_final/CONUS_year_YYYY/ folders.

    Detailed file layout (per track):
        TRACK 0 (Track 1 of 326)
        ============================
        Duration: 48.0 hours
        Start Time: 2000-01-01 00:00:00
        ...
        Detailed Track Points:
        ----------
        Timestamp           Latitude    Longitude   Vorticity   MSLP(Pa)    WindSpeed   Area(px)  ...
        ----------
        2000-01-01 00:00    -11.750     50.500      -4.39e-04   100232      10.9        168   ...

    Returns: list of dicts with the same shape extract_hurricane_tracks() does:
      {times, lats, lons, max_prob (=NaN here), mean_area, n_points, duration_h, basin}
    """
    out = []
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(hurr_dir, f"CONUS_year_{year:04d}")
        candidates = [
            os.path.join(year_dir, f"CONUS_hurricane_tracks_detailed_{year:04d}.txt"),
            os.path.join(year_dir, f"hurricane_tracks_detailed_{year:04d}.txt"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print(f"  WARN: no detailed hurricane track file for {year} under {year_dir}")
            continue

        with open(path) as fh:
            lines = fh.readlines()

        i = 0
        n_parsed = 0
        while i < len(lines):
            line = lines[i]
            if not line.lstrip().startswith("TRACK "):
                i += 1
                continue

            # Found a TRACK block — scan forward for the "Detailed Track Points:" marker,
            # then read lines that start with a YYYY-MM-DD timestamp.
            track_points = []
            j = i + 1
            in_points = False
            while j < len(lines):
                ln = lines[j]
                if ln.lstrip().startswith("TRACK "):
                    break
                if "Detailed Track Points" in ln:
                    in_points = True
                    j += 1
                    continue
                if in_points:
                    s = ln.strip()
                    # Skip header/separator lines (Timestamp header, dashes)
                    if not s or s.startswith("-") or s.startswith("Timestamp"):
                        j += 1
                        continue
                    parts = s.split()
                    if len(parts) >= 4 and parts[0][:4].isdigit() and "-" in parts[0]:
                        # Format: <date> <time> <lat> <lon> <other...>
                        try:
                            ts = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M")
                            la = float(parts[2])
                            lo = float(parts[3])
                            area = float(parts[7]) if len(parts) > 7 else float("nan")
                            track_points.append((ts, la, lo, area))
                        except (ValueError, IndexError):
                            pass
                j += 1

            if len(track_points) >= 2:
                ts_all   = [p[0] for p in track_points]
                la_all   = [p[1] for p in track_points]
                lo_all   = [p[2] for p in track_points]
                area_all = [p[3] for p in track_points if not np.isnan(p[3])]
                mid      = len(track_points) // 2
                out.append(dict(
                    times      = ts_all,
                    lats       = la_all,
                    lons       = lo_all,
                    max_prob   = float("nan"),  # not in the text format
                    mean_area  = float(np.mean(area_all)) if area_all else float("nan"),
                    n_points   = len(track_points),
                    duration_h = (ts_all[-1] - ts_all[0]).total_seconds() / 3600.0,
                    basin      = basin_of(la_all[mid], lo_all[mid]),
                ))
                n_parsed += 1
            i = j

        print(f"  Hurricane tracks {year}: parsed {n_parsed} tracks from {os.path.basename(path)}")
    print(f"  TOTAL: {len(out)} hurricane tracks loaded from {hurr_dir}")
    return out


def load_etc_tracks_from_txt(etc_dir, start_year, end_year):
    """
    Parse per-year ETC_tracks_YYYY_meteorological.txt files.

    Per-line format (after stripping '#' comments and blank lines):
        TRACK_ID  YYYY-MM-DD_HH:MM:SS  LONGITUDE  LATITUDE  INTENSITY

    Note: this format puts LON before LAT (opposite of the hurricane format).

    Returns: list of dicts with the same shape used by match_tracks().
    """
    out = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(etc_dir, f"ETC_tracks_{year:04d}_meteorological.txt")
        if not os.path.exists(path):
            print(f"  WARN: no ETC track file for {year}: {path}")
            continue

        # Group by track id within this file
        by_tid = defaultdict(list)
        n_lines = 0
        with open(path) as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 5:
                    continue
                try:
                    tid = int(parts[0])
                    ts  = datetime.strptime(parts[1], "%Y-%m-%d_%H:%M:%S")
                    lon = float(parts[2])
                    lat = float(parts[3])
                    intensity = float(parts[4])
                except ValueError:
                    continue
                by_tid[tid].append((ts, lat, lon, intensity))
                n_lines += 1

        n_tracks = 0
        for tid, pts in by_tid.items():
            pts.sort(key=lambda p: p[0])
            if len(pts) < 2:
                continue
            ts_all   = [p[0] for p in pts]
            la_all   = [p[1] for p in pts]
            lo_all   = [p[2] for p in pts]
            int_all  = [p[3] for p in pts]
            mid      = len(pts) // 2
            out.append(dict(
                times       = ts_all,
                lats        = la_all,
                lons        = lo_all,
                max_prob    = float("nan"),
                mean_area   = float("nan"),
                mean_intensity = float(np.mean(int_all)),
                n_points    = len(pts),
                duration_h  = (ts_all[-1] - ts_all[0]).total_seconds() / 3600.0,
                basin       = basin_of(la_all[mid], lo_all[mid]),
                track_id    = f"{year:04d}_{tid}",
            ))
            n_tracks += 1
        print(f"  ETC tracks {year}: parsed {n_tracks} tracks "
              f"from {n_lines} rows of {os.path.basename(path)}")
    print(f"  TOTAL: {len(out)} ETC tracks loaded from {etc_dir}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hurricane tracker — pixel probability maps → connected tracks
# ─────────────────────────────────────────────────────────────────────────────

def _extract_centroids_from_frame(prob_frame, lats, lons,
                                  prob_thresh, min_pixels):
    """For one (H,W) probability frame: return list of (lat, lon, max_prob, area)."""
    mask = prob_frame > prob_thresh
    if not mask.any():
        return []
    labeled, n = cc_label(mask)
    out = []
    for blob_id in range(1, n + 1):
        blob = (labeled == blob_id)
        npix = int(blob.sum())
        if npix < min_pixels:
            continue
        ys, xs = np.where(blob)
        blob_probs = prob_frame[ys, xs]
        wsum = blob_probs.sum()
        if wsum <= 0:
            continue
        cy = float((ys * blob_probs).sum() / wsum)
        cx = float((xs * blob_probs).sum() / wsum)
        lat = float(np.interp(cy, np.arange(len(lats)), lats))
        lon = float(np.interp(cx, np.arange(len(lons)), lons))
        out.append((lat, lon, float(blob_probs.max()), npix))
    return out


def extract_hurricane_tracks(predictions_dir, start_year, end_year,
                             prob_thresh=DEFAULT_PROB_THRESH,
                             min_blob_pixels=MIN_BLOB_PIXELS,
                             max_jump_km=MAX_LINK_JUMP_KM,
                             min_track_hours=MIN_TRACK_HOURS):
    """
    Extract hurricane tracks from per-month prediction NetCDFs.

    Returns: list of track dicts {times, lats, lons, max_prob, mean_area}
    """
    # ── Phase 1: collect centroids per timestamp ─────────────────────────────
    by_time = defaultdict(list)  # time → [(lat, lon, max_prob, area), ...]
    months_seen = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            probs, times, lats, lons = load_prediction(
                predictions_dir, year, month, "Hurricane",
            )
            if probs is None:
                continue
            months_seen += 1
            for t_idx in range(probs.shape[0]):
                cents = _extract_centroids_from_frame(
                    probs[t_idx], lats, lons, prob_thresh, min_blob_pixels,
                )
                if cents:
                    t_py = times[t_idx].to_pydatetime() if hasattr(times[t_idx], "to_pydatetime") \
                           else pd.Timestamp(times[t_idx]).to_pydatetime()
                    by_time[t_py].extend(cents)

    print(f"  Hurricane tracker: {months_seen} months scanned, "
          f"{sum(len(v) for v in by_time.values())} raw centroids across "
          f"{len(by_time)} timestamps")

    # ── Phase 2: greedy NN linking across consecutive frames ────────────────
    times_sorted = sorted(by_time.keys())
    active   = []          # list of in-progress tracks
    finished = []

    for t in times_sorted:
        cents = by_time[t]
        if not active:
            for c in cents:
                active.append([(t, *c)])
            continue

        # Distance matrix: active-tail vs centroids
        active_pts = np.array([[trk[-1][1], trk[-1][2]] for trk in active])  # lat, lon
        cent_pts   = np.array([[c[0], c[1]] for c in cents])
        if cent_pts.size == 0:
            # No centroids this frame — terminate stale tracks below
            cent_pts = np.zeros((0, 2))

        dist = np.full((len(active), len(cents)), np.inf)
        for i, (al, alo) in enumerate(active_pts):
            dt_h = (t - active[i][-1][0]).total_seconds() / 3600.0
            if dt_h > HOURS_PER_FRAME * 2.5:  # >15h gap → can't link
                continue
            for j, (cl, clo) in enumerate(cent_pts):
                dist[i, j] = haversine_km(al, alo, cl, clo)

        # Greedy assignment
        used_t, used_c = set(), set()
        while True:
            idx = int(np.argmin(dist))
            i, j = np.unravel_index(idx, dist.shape) if dist.size else (0, 0)
            if dist.size == 0 or dist[i, j] > max_jump_km:
                break
            if i in used_t or j in used_c:
                dist[i, j] = np.inf
                continue
            active[i].append((t, *cents[j]))
            used_t.add(i); used_c.add(j)
            dist[i, :] = np.inf
            dist[:, j] = np.inf

        # New tracks for unmatched centroids
        for j, c in enumerate(cents):
            if j not in used_c:
                active.append([(t, *c)])

        # Move stale tracks to finished
        still_active = []
        for trk in active:
            dt_h = (t - trk[-1][0]).total_seconds() / 3600.0
            if dt_h <= HOURS_PER_FRAME * 2.5:
                still_active.append(trk)
            else:
                if _track_duration_hours(trk) >= min_track_hours:
                    finished.append(trk)
        active = still_active

    for trk in active:
        if _track_duration_hours(trk) >= min_track_hours:
            finished.append(trk)

    return [_track_to_dict(trk) for trk in finished]


def _track_duration_hours(trk):
    if len(trk) < 2:
        return 0.0
    return (trk[-1][0] - trk[0][0]).total_seconds() / 3600.0


def _track_to_dict(trk):
    return dict(
        times      = [p[0] for p in trk],
        lats       = [p[1] for p in trk],
        lons       = [p[2] for p in trk],
        max_prob   = float(max(p[3] for p in trk)),
        mean_area  = float(np.mean([p[4] for p in trk])),
        n_points   = len(trk),
        duration_h = _track_duration_hours(trk),
        basin      = basin_of(trk[len(trk) // 2][1], trk[len(trk) // 2][2]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Track matching: predicted ↔ observed (IBTrACS)
# ─────────────────────────────────────────────────────────────────────────────

def match_tracks(predicted, observed,
                 dist_thresh_km=TRACK_DIST_MATCH_KM,
                 overlap_thresh_hours=TRACK_OVERLAP_HOURS):
    """
    Greedy track-to-track matching by mean centroid distance over overlap.

    Returns: (matches, unmatched_pred_idx, unmatched_obs_idx).
    matches is a list of dicts: {pred_idx, obs_idx, mean_dist_km, overlap_h, sid, basin}.
    """
    n_pred, n_obs = len(predicted), len(observed)
    score = np.full((n_pred, n_obs), np.inf)
    overlap_h_mat = np.zeros((n_pred, n_obs))

    # Pre-build hourly lookups per predicted track for speed
    pred_lookups = []
    for p in predicted:
        d = {}
        for t, la, lo in zip(p["times"], p["lats"], p["lons"]):
            key = t.replace(minute=0, second=0, microsecond=0)
            d[key] = (la, lo)
        pred_lookups.append(d)

    for j, o in enumerate(observed):
        o_keys = [t.replace(minute=0, second=0, microsecond=0) for t in o["times"]]
        for i, lookup in enumerate(pred_lookups):
            dists = []
            for t, la, lo in zip(o_keys, o["lats"], o["lons"]):
                if t in lookup:
                    pla, plo = lookup[t]
                    dists.append(haversine_km(la, lo, pla, plo))
            if not dists:
                continue
            overlap_h = len(dists) * HOURS_PER_FRAME
            overlap_h_mat[i, j] = overlap_h
            if overlap_h < overlap_thresh_hours:
                continue
            md = float(np.mean(dists))
            if md > dist_thresh_km:
                continue
            score[i, j] = md

    matches = []
    used_p, used_o = set(), set()
    while True:
        if score.size == 0 or np.isinf(score).all():
            break
        idx = int(np.argmin(score))
        i, j = np.unravel_index(idx, score.shape)
        if np.isinf(score[i, j]):
            break
        if i in used_p or j in used_o:
            score[i, j] = np.inf
            continue
        matches.append(dict(
            pred_idx     = int(i),
            obs_idx      = int(j),
            mean_dist_km = float(score[i, j]),
            overlap_h    = float(overlap_h_mat[i, j]),
            sid          = observed[j]["sid"],
            name         = observed[j]["name"],
            basin        = observed[j]["basin"],
        ))
        used_p.add(i); used_o.add(j)
        score[i, :] = np.inf
        score[:, j] = np.inf

    unmatched_pred = set(range(n_pred)) - used_p
    unmatched_obs  = set(range(n_obs))  - used_o
    return matches, unmatched_pred, unmatched_obs


def compute_track_skill(matches, unmatched_pred, unmatched_obs):
    """POD / FAR / HSS / Bias from track-match results."""
    hits         = len(matches)
    misses       = len(unmatched_obs)
    false_alarms = len(unmatched_pred)
    total_obs    = hits + misses
    total_pred   = hits + false_alarms

    pod  = hits / total_obs                       if total_obs  > 0 else 0.0
    far  = false_alarms / total_pred              if total_pred > 0 else 0.0
    bias = total_pred / total_obs                 if total_obs  > 0 else 0.0

    # Heidke skill score (no clear TN at object level — use observed-tracks-relative form)
    expected = total_obs * total_pred / (hits + misses + false_alarms) \
               if (hits + misses + false_alarms) > 0 else 0.0
    denom = total_obs + total_pred - hits - expected
    hss = (hits - expected) / denom if denom > 0 else 0.0

    return dict(
        hits         = hits,
        misses       = misses,
        false_alarms = false_alarms,
        POD          = pod,
        FAR          = far,
        HSS          = hss,
        BIAS         = bias,
        mean_dist_km = float(np.mean([m["mean_dist_km"] for m in matches])) if matches else float("nan"),
        mean_overlap_h = float(np.mean([m["overlap_h"]    for m in matches])) if matches else float("nan"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level metrics (per class, period-aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixel_metrics(predictions_dir, class_name, target_loader,
                          start_year, end_year, prob_thresh=DEFAULT_PROB_THRESH):
    """
    Aggregate pixel-level TP/FP/FN across all timesteps in the period.

    target_loader(year, month) → bool array of shape matching predictions,
                                  or None if no target available.
    """
    tp = fp = fn = 0
    n_months_used = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            probs, times, lats, lons = load_prediction(
                predictions_dir, year, month, class_name,
            )
            if probs is None:
                continue
            target = target_loader(year, month)
            if target is None:
                continue
            if target.shape != probs.shape:
                print(f"  WARN: shape mismatch for {class_name} {year}-{month:02d}: "
                      f"pred {probs.shape} vs target {target.shape}; skipping")
                continue
            pred = probs > prob_thresh
            tp += int(np.logical_and(pred,  target).sum())
            fp += int(np.logical_and(pred, ~target).sum())
            fn += int(np.logical_and(~pred, target).sum())
            n_months_used += 1

    eps = 1e-8
    iou  = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    return dict(
        IoU=iou, Dice=dice, F1=dice, Precision=prec, Recall=rec,
        TP=tp, FP=fp, FN=fn, n_months=n_months_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default target loaders — heuristic masks computed directly from prediction NC's
# coordinate. Replace these with catalog loaders for stronger independence.
# ─────────────────────────────────────────────────────────────────────────────

def make_self_target_loader(predictions_dir, class_name, prob_thresh=DEFAULT_PROB_THRESH):
    """
    Placeholder target loader: uses the prediction itself as target. ONLY
    useful as a sanity check (will yield IoU=1.0 by construction). Replace
    with a real catalog loader for the paper.
    """
    def _loader(year, month):
        probs, _, _, _ = load_prediction(predictions_dir, year, month, class_name)
        return None if probs is None else (probs > prob_thresh)
    return _loader


def make_held_out_target_loader(holdout_nc_dir, class_name, var_name=None):
    """
    Load heuristic target masks from pre-saved NetCDFs (one per month).
    Expects files like {holdout_nc_dir}/{class}_target_{YYYY}{MM}.nc with a
    variable named class_name (or var_name override) holding a bool/0-1 mask
    of shape (T, H, W).

    Use this when you've pre-computed the heuristic targets once and want to
    avoid recomputing them per evaluation.
    """
    def _loader(year, month):
        path = os.path.join(holdout_nc_dir, f"{class_name}_target_{year:04d}{month:02d}.nc")
        if not os.path.exists(path):
            return None
        ds = xr.open_dataset(path)
        var = var_name or class_name
        if var not in ds.data_vars:
            ds.close()
            return None
        target = ds[var].values.astype(bool)
        ds.close()
        return target
    return _loader


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Offline evaluator: Hurricane track POD/FAR vs IBTrACS + per-class pixel IoU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--predictions-dir", required=True,
                   help="Dir with test.py prediction NetCDFs")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year",   type=int, required=True)
    p.add_argument("--out-dir",    required=True,
                   help="Where to write summary + CSV + tracks JSON")
    p.add_argument("--tag",        default="primary",
                   help="Tag for output filenames (e.g. 'primary', 'oop_1980_2005')")
    p.add_argument("--ibtracs",
                   default=("/lustre/orion/proj-shared/cli138/dr6/eventtype/"
                            "run-tracking/StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc"),
                   help="IBTrACS NetCDF path")
    p.add_argument("--prob-threshold",     type=float, default=DEFAULT_PROB_THRESH)
    p.add_argument("--dist-threshold-km",  type=float, default=TRACK_DIST_MATCH_KM)
    p.add_argument("--min-track-hours",    type=int,   default=MIN_TRACK_HOURS)
    p.add_argument("--wind-threshold-kt",  type=float, default=0.0,
                   help="Min IBTrACS wind (kt) to include obs track (0 = all)")
    p.add_argument("--skip-tracks", action="store_true",
                   help="Skip Hurricane track-match (only do pixel metrics)")
    p.add_argument("--skip-pixel",  action="store_true",
                   help="Skip pixel metrics (only do track-match)")
    p.add_argument("--pixel-targets-dir", default=None,
                   help="If set, load pre-saved heuristic-target NetCDFs from here. "
                        "Otherwise pixel metrics are skipped (no targets available).")
    p.add_argument("--hurricane-tracks-dir", default=None,
                   help="If set, load Hurricane tracks from already-extracted "
                        "CONUS_year_YYYY/CONUS_hurricane_tracks_detailed_YYYY.txt "
                        "files instead of running the internal centroid tracker on "
                        "the prediction NetCDFs.")
    p.add_argument("--etc-tracks-dir", default=None,
                   help="If set, load ETC tracks from already-extracted "
                        "ETC_tracks_YYYY_meteorological.txt files (for downstream "
                        "use; ETC track-vs-reference matching is not yet wired in).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*78}\n EVALUATE — tag={args.tag}, period={args.start_year}-{args.end_year}\n{'='*78}")
    print(f"  Predictions    : {args.predictions_dir}")
    print(f"  Out dir        : {args.out_dir}")
    print(f"  Prob threshold : {args.prob_threshold}")
    print(f"  Dist threshold : {args.dist_threshold_km} km")
    print(f"  Min track hrs  : {args.min_track_hours}")
    print(f"  IBTrACS        : {args.ibtracs}")

    results = {}

    # ─── (1) Hurricane track POD/FAR vs IBTrACS ──────────────────────────────
    if not args.skip_tracks:
        if args.hurricane_tracks_dir:
            print(f"\n[1/2] Hurricane: loading PRE-EXTRACTED tracks from "
                  f"{args.hurricane_tracks_dir}")
            pred_tracks = load_hurricane_tracks_from_dir(
                args.hurricane_tracks_dir, args.start_year, args.end_year,
            )
        else:
            print(f"\n[1/2] Hurricane: extracting predicted tracks from prediction NetCDFs…")
            pred_tracks = extract_hurricane_tracks(
                args.predictions_dir, args.start_year, args.end_year,
                prob_thresh=args.prob_threshold,
                min_track_hours=args.min_track_hours,
            )
        print(f"  → {len(pred_tracks)} predicted tracks (≥{args.min_track_hours} h)")

        print(f"      loading IBTrACS for {args.start_year}-{args.end_year}…")
        obs_tracks = load_ibtracs_tracks(
            args.ibtracs, args.start_year, args.end_year,
            wind_threshold_kt=args.wind_threshold_kt,
        )
        print(f"  → {len(obs_tracks)} IBTrACS tracks")

        print("      matching…")
        matches, up, uo = match_tracks(
            pred_tracks, obs_tracks,
            dist_thresh_km=args.dist_threshold_km,
        )
        skill = compute_track_skill(matches, up, uo)
        results["hurricane_track_overall"] = skill
        print(f"  → hits={skill['hits']}  misses={skill['misses']}  "
              f"false_alarms={skill['false_alarms']}")
        print(f"  → POD={skill['POD']:.3f}  FAR={skill['FAR']:.3f}  "
              f"HSS={skill['HSS']:.3f}  BIAS={skill['BIAS']:.3f}  "
              f"mean_dist={skill['mean_dist_km']:.1f} km")

        # Stratify by basin
        for code in list(BASINS.keys()) + ["Other"]:
            pred_basin = [t for t in pred_tracks
                          if basin_of(t["lats"][len(t["lats"]) // 2],
                                      t["lons"][len(t["lons"]) // 2]) == code]
            obs_basin = [t for t in obs_tracks if t["basin"] == code or
                         basin_of(t["lats"][len(t["lats"]) // 2],
                                  t["lons"][len(t["lons"]) // 2]) == code]
            if not obs_basin and not pred_basin:
                continue
            mb, upb, uob = match_tracks(
                pred_basin, obs_basin, dist_thresh_km=args.dist_threshold_km,
            )
            results[f"hurricane_track_basin_{code}"] = compute_track_skill(mb, upb, uob)

        # Save raw matches for audit
        tracks_path = os.path.join(args.out_dir, f"tracks_{args.tag}.json")
        with open(tracks_path, "w") as fh:
            json.dump(
                dict(
                    predicted_count=len(pred_tracks),
                    observed_count =len(obs_tracks),
                    matches=[{**m, "mean_dist_km": round(m["mean_dist_km"], 1)} for m in matches],
                    unmatched_pred=sorted(list(up)),
                    unmatched_obs =sorted(list(uo)),
                ),
                fh, indent=2, default=str,
            )
        print(f"  → tracks audit JSON: {tracks_path}")

    # ─── (1b) ETC tracks: parse + count if requested ────────────────────────
    # (Object-level ETC POD/FAR needs an independent reference catalog — not
    # yet wired in. For now we just count + dump the predictions for record.)
    if args.etc_tracks_dir and not args.skip_tracks:
        print(f"\n[1b/2] ETC: loading pre-extracted tracks from {args.etc_tracks_dir}")
        etc_tracks = load_etc_tracks_from_txt(
            args.etc_tracks_dir, args.start_year, args.end_year,
        )
        results["etc_track_counts"] = dict(
            n_tracks   = len(etc_tracks),
            n_basin_NA = sum(1 for t in etc_tracks if t["basin"] == "NA"),
            n_basin_EP = sum(1 for t in etc_tracks if t["basin"] == "EP"),
            n_basin_WP = sum(1 for t in etc_tracks if t["basin"] == "WP"),
            n_basin_SI = sum(1 for t in etc_tracks if t["basin"] == "SI"),
            n_basin_SP = sum(1 for t in etc_tracks if t["basin"] == "SP"),
            n_basin_NI = sum(1 for t in etc_tracks if t["basin"] == "NI"),
            n_basin_Other = sum(1 for t in etc_tracks if t["basin"] == "Other"),
            mean_duration_h = float(np.mean([t["duration_h"] for t in etc_tracks])) if etc_tracks else 0.0,
        )

    # ─── (2) Pixel metrics per class ─────────────────────────────────────────
    if not args.skip_pixel:
        if args.pixel_targets_dir is None:
            print("\n[2/2] Pixel metrics: skipped (no --pixel-targets-dir provided)")
        else:
            for cls in ("AR", "Hurricane", "ETCs", "MCS"):
                print(f"\n[2/2] Pixel metrics: {cls} vs heuristic mask from "
                      f"{args.pixel_targets_dir}…")
                loader = make_held_out_target_loader(args.pixel_targets_dir, cls)
                pm = compute_pixel_metrics(
                    args.predictions_dir, cls, loader,
                    args.start_year, args.end_year,
                    prob_thresh=args.prob_threshold,
                )
                results[f"{cls.lower()}_pixel"] = pm
                print(f"  → IoU={pm['IoU']:.4f}  Dice={pm['Dice']:.4f}  "
                      f"P={pm['Precision']:.4f}  R={pm['Recall']:.4f}  "
                      f"(over {pm['n_months']} months)")

    # ─── Write reports ───────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir, f"summary_{args.tag}.txt")
    csv_path = os.path.join(args.out_dir, f"summary_{args.tag}.csv")

    with open(txt_path, "w") as fh:
        fh.write("=" * 78 + "\n")
        fh.write("WNet Storm Detection — Offline Evaluation Summary\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Predictions    : {args.predictions_dir}\n")
        fh.write(f"Period         : {args.start_year}-{args.end_year}\n")
        fh.write(f"Tag            : {args.tag}\n")
        fh.write(f"Prob threshold : {args.prob_threshold}\n")
        fh.write(f"Dist threshold : {args.dist_threshold_km} km\n")
        fh.write(f"Min track hrs  : {args.min_track_hours}\n\n")

        for key, val in results.items():
            fh.write(f"{key}:\n")
            for k, v in val.items():
                if isinstance(v, float):
                    fh.write(f"  {k:<22} = {v:.6f}\n")
                else:
                    fh.write(f"  {k:<22} = {v}\n")
            fh.write("\n")
    print(f"\nSummary text  : {txt_path}")

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "metric", "value"])
        for key, val in results.items():
            for k, v in val.items():
                w.writerow([key, k, v])
    print(f"Summary CSV   : {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
