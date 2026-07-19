"""
Track loaders for the RegCM ETC / TC comparison figures.

Turns the trackers' fixed-width .txt output into the track-dict schema the ERA5
analysis already uses, so the ERA5 figure functions can be reused directly.

TRACK SCHEMA
    dict(track_id, source, times[list[datetime]], lats[np], lons[np],
         n_points, duration_h, net_disp_km, in_domain[np bool],
         mslp[np], wind[np], vort[np], area[np])
    Fields not available for a given storm type are all-NaN arrays.

TWO THINGS TO KNOW BEFORE USING THIS

1. THE DOMAIN MASK IS NOT THE BOUNDING BOX.
   RegCM runs on a curved Lambert grid inscribed in the lat/lon box, so ~28% of
   the box lies outside the simulation and is NaN in the driving fields — 93% of
   the row at lat 18.75. `domain_mask()` derives the true domain from the NaN
   pattern of MSLP in the paired WNet input, which is a free and principled mask
   (no hand-drawn terrain or latitude cutoff).

   This matters most for TC: measured on ACC 2005, 24 of 64 tracks (38%) lie
   ENTIRELY outside the domain — artifacts of the fillna(0.0) region. Restricting
   to fully-in-domain tracks gives 14/yr against a reality check of ~10-15 TCs/yr
   reaching this domain, which is very likely the explanation for the long-standing
   "ACC hurricane counts look ~4x too high" observation. ETC is nearly unaffected
   (98.9% of points in-domain) because ETCs live at mid-latitudes where the domain
   fills the box.

   Per the agreed decision, tracks are NOT filtered on this. `in_domain` is carried
   through and surfaced on the figures instead, so the diagnostic is visible rather
   than silently applied.

2. NEITHER STORM TYPE HAS A USABLE INTENSITY FROM THE TRACKER ALONE.
   ETC "INTENSITY" is the connected-component PIXEL COUNT
   (updated_monthly_ETC_Tracker.py:249), not vorticity or MSLP — it is stored as
   `area`, and `mslp`/`wind`/`vort` stay NaN. Do not plot ETC area as intensity.
   TC does carry real MSLP / wind / vorticity, sampled from the paired WNet file —
   but only where the point is inside the domain (elsewhere it is correctly NaN).
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import xarray as xr

from storm_plot_common import REGCM_EXTENT

# --------------------------------------------------------------------------- paths
BASE = os.path.dirname(os.path.abspath(__file__))
REGCM_ROOT = os.path.join(BASE, "ARs_Hurricanes_ETCs_MCS")
REGCM_INPUT_BASE = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"

_NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
ETC_REF_DIR = os.path.join(_NB, "Global_ARs_Hurr_ETCs_MCS_finalized", "ETCs_WNet_final")
IBTRACS_PATH = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/IBTrACS/IBTrACS.ALL.v04r01.nc"

EARTH_R_KM = 6371.0


# --------------------------------------------------------------------------- helpers
def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def _empty(n):
    return np.full(n, np.nan, dtype=float)


def _finalize(track_id, source, times, lats, lons,
              mslp=None, wind=None, vort=None, area=None, valid=None):
    """Assemble a track dict and derive duration / displacement / in-domain."""
    n = len(times)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    dur = (times[-1] - times[0]).total_seconds() / 3600.0 if n > 1 else 0.0
    disp = float(haversine_km(lons[0], lats[0], lons[-1], lats[-1])) if n > 1 else 0.0
    if valid is None:
        in_dom = np.ones(n, dtype=bool)
    else:
        in_dom = _points_in_domain(valid, lats, lons)
    return dict(
        track_id=track_id, source=source, times=times, lats=lats, lons=lons,
        n_points=n, duration_h=dur, net_disp_km=disp, in_domain=in_dom,
        mslp=_empty(n) if mslp is None else np.asarray(mslp, dtype=float),
        wind=_empty(n) if wind is None else np.asarray(wind, dtype=float),
        vort=_empty(n) if vort is None else np.asarray(vort, dtype=float),
        area=_empty(n) if area is None else np.asarray(area, dtype=float),
    )


# ----------------------------------------------------------------------- domain mask
_MASK_CACHE = {}


def domain_mask(model):
    """(valid2d, lat, lon) for a model — True where the simulation actually exists.

    Derived from the NaN pattern of MSLP in the paired WNet input. The pattern is a
    property of the fixed Lambert grid, so any one month serves; the first available
    file is used and the result cached per model.
    """
    if model in _MASK_CACHE:
        return _MASK_CACHE[model]
    pat = os.path.join(REGCM_INPUT_BASE, model, "WNet_6hourly_latlon", f"{model}dara_*.nc")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no paired WNet input for {model} under {pat}")
    with xr.open_dataset(files[0]) as ds:
        var = "MSLP" if "MSLP" in ds else list(ds.data_vars)[0]
        valid = ~np.isnan(ds[var].isel(time=0).values)
        lat = ds.latitude.values.copy()
        lon = ds.longitude.values.copy()
    _MASK_CACHE[model] = (valid, lat, lon)
    return _MASK_CACHE[model]


def _points_in_domain(valid_tuple, lats, lons):
    valid, lat, lon = valid_tuple
    j = np.abs(lat[None, :] - np.asarray(lats)[:, None]).argmin(axis=1)
    i = np.abs(lon[None, :] - np.asarray(lons)[:, None]).argmin(axis=1)
    return valid[j, i]


def in_domain_fraction(tracks):
    """Fraction of all track POINTS falling inside the true model domain."""
    pts = np.concatenate([t["in_domain"] for t in tracks]) if tracks else np.array([])
    return float(pts.mean()) if pts.size else float("nan")


def fully_outside_count(tracks):
    return sum(1 for t in tracks if not t["in_domain"].any())


# --------------------------------------------------------------------------- ETC
_ETC_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.eE+]+)\s*$")


def _parse_etc_txt(path, source, valid=None):
    """Parse `TRACK_ID TIME LONGITUDE LATITUDE INTENSITY` rows (note lon BEFORE lat).

    One parser serves both the RegCM files (ETC_tracks_<MODEL>_<YYYY>.txt) and the
    ERA5 reference (ETC_tracks_<YYYY>_meteorological.txt) — the row format is
    byte-compatible and only the filename differs. Deliberately implemented here
    rather than by adding a pattern argument to
    WNet_parallel_global_code/evaluate.py:312, so the ERA5 pipeline is not perturbed
    AND model and reference are guaranteed to be parsed identically (asymmetric
    parsing is an easy way to manufacture a fake model bias).
    """
    by_id = {}
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            m = _ETC_ROW.match(ln)
            if not m:
                continue
            tid, tstr, lon, lat, inten = m.groups()
            try:
                t = datetime.strptime(tstr, "%Y-%m-%d_%H:%M:%S")
            except ValueError:
                continue
            by_id.setdefault(tid, []).append((t, float(lat), float(lon), float(inten)))

    tracks = []
    for tid, rows in by_id.items():
        rows.sort(key=lambda r: r[0])
        times = [r[0] for r in rows]
        tracks.append(_finalize(
            f"{source}:{os.path.basename(path)}:{tid}", source, times,
            [r[1] for r in rows], [r[2] for r in rows],
            # INTENSITY is pixel count -> `area`, never `mslp`/`wind`.
            area=[r[3] for r in rows], valid=valid,
        ))
    return tracks


def load_regcm_etc_tracks(model, years=None):
    d = os.path.join(REGCM_ROOT, f"Outputs_RegCM_PR_{model}", "ETC_tracks")
    valid = domain_mask(model)
    out = []
    for fp in sorted(glob.glob(os.path.join(d, f"ETC_tracks_{model}_*.txt"))):
        m = re.search(r"_(\d{4})\.txt$", fp)
        if not m:
            continue
        yr = int(m.group(1))
        if years and yr not in years:
            continue
        out.extend(_parse_etc_txt(fp, model, valid=valid))
    return out


def load_reference_etc_tracks(years=None, qc=None, domain_criteria=None):
    """ERA5-WNet ETC tracks, QC'd globally and THEN clipped to the RegCM box.

    The clip is essential: the reference is global (~1150 tracks/yr) while RegCM
    covers CONUS only (~20-40/yr). Comparing them unclipped would attribute a ~30x
    domain difference to model skill.

    ORDER MATTERS — QC BEFORE CLIP. Clipping first truncates a genuine ETC that
    merely passes through the domain down to a short segment, which then fails the
    48 h / 8 pt / 1000 km gate. Measured: clip-then-QC kept only 49% of reference
    tracks against 96-98% for the models, i.e. it manufactured a large fake bias
    against the reference. QC-then-clip asks the right question — "which real ETCs
    touch this domain, and where do they go while inside it".

    A residual asymmetry remains and is NOT papered over: RegCM tracks are truncated
    at the domain edge by construction (a cyclone entering from the Pacific appears
    mid-life), while the reference is tracked globally. That is a genuine limitation
    of a regional model, not something QC should hide.
    """
    out = []
    for fp in sorted(glob.glob(os.path.join(ETC_REF_DIR, "ETC_tracks_*_meteorological.txt"))):
        m = re.search(r"ETC_tracks_(\d{4})_", fp)
        if not m:
            continue
        yr = int(m.group(1))
        if years and yr not in years:
            continue
        out.extend(_parse_etc_txt(fp, "REF", valid=None))
    if qc is not None:
        out = qc_filter(out, label="REF(global)", **qc)
    out = clip_to_box(out)
    if domain_criteria is not None:
        out = domain_criteria_filter(out, label="REF", **domain_criteria)
    return out


# --------------------------------------------------------------------------- TC
_TC_ROW = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
)


def _f(tok):
    try:
        v = float(tok)
        return v if np.isfinite(v) else np.nan
    except ValueError:
        return np.nan


def load_regcm_tc_tracks(model, years=None):
    """Parse CONUS_hurricane_tracks_detailed_<YYYY>.txt.

    Columns: Timestamp Latitude Longitude Vorticity MSLP(Pa) WindSpeed Area(px) ...
    The timestamp contains a space, so the date and time are two whitespace tokens —
    a naive split() puts latitude at index 2, not 1. (Getting this wrong silently
    reads longitude as vorticity and MSLP as wind speed, with plausible-looking
    magnitudes.)
    """
    d = os.path.join(REGCM_ROOT, f"Outputs_RegCM_PR_{model}", "Hurricane_tracks")
    valid = domain_mask(model)
    out = []
    for ydir in sorted(glob.glob(os.path.join(d, "CONUS_year_*"))):
        m = re.search(r"CONUS_year_(\d{4})$", ydir)
        if not m:
            continue
        yr = int(m.group(1))
        if years and yr not in years:
            continue
        fps = glob.glob(os.path.join(ydir, "CONUS_hurricane_tracks_detailed_*.txt"))
        if not fps:
            continue
        cur, rows = None, []

        def flush():
            if cur is None or len(rows) < 2:
                return
            rows.sort(key=lambda r: r[0])
            out.append(_finalize(
                f"{model}:{yr}:{cur}", model, [r[0] for r in rows],
                [r[1] for r in rows], [r[2] for r in rows],
                vort=[r[3] for r in rows], mslp=[r[4] for r in rows],
                wind=[r[5] for r in rows], area=[r[6] for r in rows],
                valid=valid,
            ))

        with open(fps[0]) as fh:
            for ln in fh:
                if ln.startswith("TRACK "):
                    flush()
                    cur, rows = ln.split()[1], []
                    continue
                mm = _TC_ROW.match(ln)
                if mm and cur is not None:
                    date, hhmm, lat, lon, vort, mslp, wind, area = mm.groups()
                    rows.append((
                        datetime.strptime(f"{date} {hhmm}", "%Y-%m-%d %H:%M"),
                        float(lat), float(lon),
                        _f(vort), _f(mslp), _f(wind), _f(area),
                    ))
            flush()
    return out


def load_reference_tc_tracks(years=None, qc=None, domain_criteria=None):
    """IBTrACS, QC'd globally and THEN clipped to the RegCM box.

    Same clip-order reasoning as load_reference_etc_tracks — a TC that crosses the
    domain must be judged on its whole life, not on the fragment inside the box.
    """
    import sys
    p = os.path.join(_NB, "WNet_parallel_global_code")
    if p not in sys.path:
        sys.path.insert(0, p)
    import evaluate  # reused rather than reimplemented: the IBTrACS NetCDF layout
                     # (bytes sids, usa_* fallbacks, ragged rows) is fiddly.

    yrs = sorted(years) if years else [1980, 2014]
    raw = evaluate.load_ibtracs_tracks(IBTRACS_PATH, yrs[0], yrs[-1],
                                       lat_band=(REGCM_EXTENT[2], REGCM_EXTENT[3]))
    out = []
    for i, t in enumerate(raw):
        if len(t["times"]) < 2:
            continue
        out.append(_finalize(
            t.get("sid", f"ib{i}"), "REF", list(t["times"]),
            t["lats"], t["lons"], wind=t.get("wind_kt"), valid=None,
        ))
    if qc is not None:
        out = qc_filter(out, label="REF(global)", **qc)
    out = clip_to_box(out)
    if domain_criteria is not None:
        out = domain_criteria_filter(out, label="REF", **domain_criteria)
    return out


# ------------------------------------------------- matched in-domain criteria
# The RegCM ETC tracker's kinematic criteria (updated_monthly_ETC_Tracker.py:69-98),
# re-applied to a CLIPPED reference track so the reference is scored on what it does
# INSIDE the domain — which is all a regional tracker could ever see.
ETC_DOMAIN_CRITERIA = dict(min_points=8, min_duration_h=48.0, min_path_km=1800.0,
                           min_avg_propagation_ms=3.5, min_median_speed_ms=5.0,
                           min_net_east_km=500.0)

# None means "criterion not applied". It must NOT be written as 0.0: net_east_km is
# SIGNED, so `>= 0.0` silently means "must move eastward" and would discard every
# westward-tracking Atlantic TC — which is most of them. (Observed: it cut the
# IBTrACS reference from 17.8 to 6.5 tracks/yr.) The TC tracker has no zonal or
# speed criterion of its own, so only count and duration are matched here.
TC_DOMAIN_CRITERIA = dict(min_points=4, min_duration_h=24.0, min_path_km=None,
                          min_avg_propagation_ms=None, min_median_speed_ms=None,
                          min_net_east_km=None)


def _segments_km(t):
    return haversine_km(t["lons"][:-1], t["lats"][:-1], t["lons"][1:], t["lats"][1:])


def path_km(t):
    return float(np.sum(_segments_km(t))) if t["n_points"] > 1 else 0.0


def avg_propagation_ms(t):
    h = (t["times"][-1] - t["times"][0]).total_seconds()
    return t["net_disp_km"] * 1000.0 / h if h > 0 else 0.0


def median_speed_ms(t):
    if t["n_points"] < 2:
        return 0.0
    s = _segments_km(t)
    dt = np.array([(t["times"][i + 1] - t["times"][i]).total_seconds()
                   for i in range(len(s))], dtype=float)
    return float(np.median(s * 1000.0 / np.maximum(dt, 1e-9)))


def net_east_km(t):
    d = haversine_km(t["lons"][0], t["lats"][0], t["lons"][-1], t["lats"][0])
    return float(d * np.sign(t["lons"][-1] - t["lons"][0]))


def domain_criteria_filter(tracks, min_points=None, min_duration_h=None,
                           min_path_km=None, min_avg_propagation_ms=None,
                           min_median_speed_ms=None, min_net_east_km=None,
                           label="", verbose=True):
    """Score a clipped reference track on what it does INSIDE the domain.

    WHY THIS IS NEEDED — the annual-count figures are meaningless without it.
    clip_to_box keeps any track with >=2 points in the box, so a cyclone that merely
    clips a corner for 12 h counts the same as one that crosses the whole domain.
    Measured over ERA5 1995-1999: 14.3% of reference ETC tracks have only 2-3
    in-domain points, and the 10th percentile of in-domain path length is 609 km —
    far below the 1800 km the RegCM tracker demands. That inflates the reference
    count ~2x for a purely methodological reason.

    Applying these criteria takes the ERA5 ETC reference from 141/yr to 68/yr.
    Self-check: applying the same criteria to ACC changes it only 37.4 -> 36.2/yr
    (-3%), because the RegCM tracker already enforced them — which is what confirms
    this reimplementation matches the tracker rather than inventing a second gate.

    The residual REF-vs-model difference after this (68 vs 36/yr) is NOT
    methodological and should be reported as a genuine result.
    """
    def ok(t):
        for lim, val in ((min_points, t["n_points"]),
                         (min_duration_h, t["duration_h"]),
                         (min_path_km, None), (min_avg_propagation_ms, None),
                         (min_median_speed_ms, None), (min_net_east_km, None)):
            if lim is None:
                continue
            if val is None:
                continue
            if val < lim:
                return False
        # the derived quantities are computed lazily, only when actually gated on
        if min_path_km is not None and path_km(t) < min_path_km:
            return False
        if min_avg_propagation_ms is not None and avg_propagation_ms(t) < min_avg_propagation_ms:
            return False
        if min_median_speed_ms is not None and median_speed_ms(t) < min_median_speed_ms:
            return False
        if min_net_east_km is not None and net_east_km(t) < min_net_east_km:
            return False
        return True

    keep = [t for t in tracks if ok(t)]
    if verbose:
        n = len(tracks)
        pct = 100.0 * len(keep) / n if n else 0.0
        print(f"    in-domain criteria {label:>6s}: {n:5d} -> {len(keep):5d} ({pct:5.1f}% kept)")
    return keep


# --------------------------------------------------------------------------- QC
def clip_to_box(tracks, extent=None):
    """Keep only the track segment inside the RegCM lon/lat box; drop empties.

    Applied to the REFERENCE datasets so model and reference cover the same region.
    """
    lo0, lo1, la0, la1 = extent or REGCM_EXTENT
    out = []
    for t in tracks:
        m = ((t["lons"] >= lo0) & (t["lons"] <= lo1) &
             (t["lats"] >= la0) & (t["lats"] <= la1))
        if m.sum() < 2:
            continue
        idx = np.where(m)[0]
        times = [t["times"][k] for k in idx]
        out.append(_finalize(
            t["track_id"], t["source"], times, t["lats"][idx], t["lons"][idx],
            mslp=t["mslp"][idx], wind=t["wind"][idx], vort=t["vort"][idx],
            area=t["area"][idx], valid=None,
        ))
    return out


def qc_filter(tracks, min_duration_h=48.0, min_points=8, min_disp_km=1000.0,
              label="", verbose=True):
    """The ERA5 ETC QC gate (global_etc_analysis.py:245-254), applied SYMMETRICALLY
    to models and reference. Asymmetric QC is the easiest way to invent a model bias,
    so the in/out counts are logged for every dataset."""
    keep = [t for t in tracks
            if t["duration_h"] >= min_duration_h
            and t["n_points"] >= min_points
            and t["net_disp_km"] >= min_disp_km]
    if verbose:
        n = len(tracks)
        pct = 100.0 * len(keep) / n if n else 0.0
        print(f"    QC {label:>10s}: {n:5d} -> {len(keep):5d} tracks ({pct:5.1f}% kept)")
    return keep


# --------------------------------------------------------------------------- derived
def all_points(tracks, months=None):
    """Pooled (lons, lats) over tracks, optionally restricted to given months."""
    lo, la = [], []
    for t in tracks:
        if months:
            m = np.array([tt.month in months for tt in t["times"]], dtype=bool)
            if not m.any():
                continue
            lo.append(t["lons"][m]); la.append(t["lats"][m])
        else:
            lo.append(t["lons"]); la.append(t["lats"])
    if not lo:
        return np.array([]), np.array([])
    return np.concatenate(lo), np.concatenate(la)


def genesis_points(tracks):
    if not tracks:
        return np.array([]), np.array([])
    return (np.array([t["lons"][0] for t in tracks]),
            np.array([t["lats"][0] for t in tracks]))


def year_of(track):
    return track["times"][0].year


def tracks_by_year(tracks):
    out = {}
    for t in tracks:
        out.setdefault(year_of(t), []).append(t)
    return out


def storm_days(tracks):
    """Total storm-days: summed track duration, in days."""
    return sum(t["duration_h"] for t in tracks) / 24.0
