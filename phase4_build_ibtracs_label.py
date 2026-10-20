#!/usr/bin/env python3
"""
phase4_build_ibtracs_label.py — build the TC target from IBTrACS observations and
patch it into the split cache, replacing the CyTRACK heuristic.

Why replace the heuristic
-------------------------
The other three classes are targeted on something meaningful; TC was targeted on a
heuristic the model could only imitate. Tier B measured the consequence: the model
converges onto its teacher (label CSI 0.086 -> model 0.077; label 0.119 -> model
0.103, tracking at ~87%), and the teacher marks ~36-72 objects per timestep against
~2.3 real storms. No threshold inside CyTRACK closes that — the strictest candidate
still sat at FAR 0.909 against TempestExtremes' 0.52.

Learnability was checked first, not assumed (phase4_ibtracs_learnability.py). Within
250 km of each IBTrACS position >= 34 kt, ERA5 has a local MSLP minimum 99.6% of the
time, with depth >= 300 Pa and correct-sign vorticity at the same rate, and vorticity
AND wind >= 10 m/s at 99.2%. The fields the model reads carry an unambiguous
signature at essentially every real storm, so an observational target is learnable
here — the POD ceiling is ~0.99, not something limiting.

Construction
------------
  * positions with usa_wind >= --wind-kt (34 kt, tropical-storm strength), matching
    how every TC evaluation in this project defines a storm
  * nature != ET. A storm that has gone extratropical IS an extratropical cyclone,
    and this project has a separate ETC class; including those points would teach
    the TC head to fire on extratropical systems. Measured at ~6% of points.
  * radius from usa_r34, the observed 34-kt wind radius averaged over the four
    quadrants, where available (40% of points, median 232 km, p10 147, p90 333).
    Elsewhere --default-radius-km. The old label used a flat 200 km for everything,
    which is slightly under the observed median and ignores real size variation.
  * radius clamped to [--min-radius-km, --max-radius-km] so a bad record cannot
    paint a continent.

IMPORTANT — this makes IBTrACS a training signal, so it is no longer an independent
evaluator. Evaluation must move to strictly held-out years, with TempestExtremes as
the independent algorithmic check. Ignoring that is precisely what made the previous
system's POD of 0.99 meaningless.

Usage
  phase4_build_ibtracs_label.py --year 2003 --month 9 --outdir <cache_v5> [--stats-only]
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_tc_label_screen import IBTRACS   # noqa: E402

SPLIT_CACHE = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
               "Storms_precipitation_notebooks/tensor_cache_v3_split")
HURR_CH = 7
TARGET = (192, 384)
NM_TO_KM = 1.852
EARTH_R = 6371.0


def _nature_code(nat, i, j):
    try:
        return b"".join(
            [bytes(c) if isinstance(c, (bytes, np.bytes_)) else
             str(c).encode() for c in np.ma.filled(nat[i, j], b" ")]
        ).decode(errors="ignore").strip()
    except Exception:
        return ""


def lifecycle_mask(wd, tt, codes, ok_pos, wind_kt, td_lead_hours=0.0, through_et=False):
    """Positions to label under the full-lifecycle rule (--lifecycle).

    From the first position that is >= wind_kt and not ET, through the storm's
    last position, stopping at the first ET position. Wind is ignored after
    genesis. The >= 34 kt rule switched the label off exactly where a storm
    weakens over land (Bonnie'86: 3 of 17 land positions >= 34 kt, median 20 kt),
    so the model learned to go silent at landfall, while TRACK, TempestExtremes
    and the IBTrACS best track all follow the decaying low inland.
    td_lead_hours > 0 also labels the pre-genesis stage, up to that many hours
    before genesis (non-ET positions only).
    """
    is_et = np.array([c == "ET" for c in codes], bool)
    ts = ok_pos & (np.nan_to_num(wd) >= wind_kt) & ~is_et
    sel = np.zeros(wd.size, bool)
    if not ts.any():
        return sel
    j0 = int(np.argmax(ts))
    et_after = np.where(is_et & ok_pos & (np.arange(wd.size) > j0))[0]
    # through_et: keep labelling after extratropical transition (IBTrACS, TRACK and the
    # advisor's figure all follow the storm through ET; the default stops at the first ET).
    j1 = wd.size if (through_et or not et_after.size) else int(et_after[0])
    sel[j0:j1] = True
    if td_lead_hours > 0:
        lead = ok_pos & ~is_et & (np.arange(wd.size) < j0) & \
            (tt >= tt[j0] - td_lead_hours / 24.0)
        sel |= lead
    return sel & ok_pos


def ibtracs_discs(year, wind_kt, exclude_nature=("ET",), lifecycle=False,
                  td_lead_hours=0.0, through_et=False):
    """dict[datetime] -> [(lat, lon, radius_km_or_nan)] for qualifying positions.

    Selects seasons year-1 .. year+1, NOT season == year. IBTrACS `season` is a
    season label, not a calendar year: season 1999 spans 1998-07 to 1999-11, so
    December 1999 belongs to season 2000 (the Southern Hemisphere 1999/2000
    season). Filtering on season == year therefore drops SH storms in Nov/Dec of
    every year — a systematic hole in exactly the hemisphere whose spurious tracks
    this project has been chasing. Positions are keyed by their own timestamps, so
    over-selecting seasons is harmless; the caller only reads the stamps it needs.
    """
    from netCDF4 import Dataset, num2date
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    idx = np.where((season >= year - 1) & (season <= year + 1))[0]
    lat = ds.variables["lat"][idx]
    lon = ds.variables["lon"][idx]
    # Intensity from the MAXIMUM over every agency that reported, not usa_wind
    # alone. usa_wind is US-agency-reported and its coverage outside the Atlantic
    # and E Pacific is patchy: May 1995 has 156 storm positions worldwide and NOT
    # ONE has a finite usa_wind >= 34 kt, which silently produced an empty target
    # for that whole month. Using the agency maximum recovers ~9% more qualifying
    # positions and removes a systematic bias against the W Pacific, N Indian and
    # Australian basins.
    _wcols = [v for v in ds.variables
              if v.endswith("_wind") and ds.variables[v].ndim == 2]
    _stack = [np.ma.filled(ds.variables[v][idx].astype(float), np.nan)
              for v in _wcols]
    with np.errstate(all="ignore"):
        wind = np.nanmax(np.dstack(_stack), axis=2) if _stack else \
            np.ma.filled(ds.variables["usa_wind"][idx].astype(float), np.nan)
    traw = ds.variables["time"][idx]
    units = ds.variables["time"].units
    cal = getattr(ds.variables["time"], "calendar", "standard")
    r34 = ds.variables["usa_r34"][idx] if "usa_r34" in ds.variables else None
    roci = ds.variables["usa_roci"][idx] if "usa_roci" in ds.variables else None
    nat = ds.variables["nature"][idx] if "nature" in ds.variables else None
    ds.close()

    out = defaultdict(list)
    n_kept = n_drop_nat = 0
    for i in range(len(idx)):
        la = np.ma.filled(lat[i].astype(float), np.nan)
        lo = np.ma.filled(lon[i].astype(float), np.nan)
        wd = np.asarray(wind[i], dtype=float)
        tt = np.ma.filled(traw[i].astype(float), np.nan)
        ok_pos = np.isfinite(la) & np.isfinite(lo) & np.isfinite(tt)
        if lifecycle:
            # tt is in days (IBTrACS "days since 1858-11-17"), hence lead/24 above
            codes = [_nature_code(nat, i, j) if nat is not None else ""
                     for j in range(wd.size)]
            ok = lifecycle_mask(wd, tt, codes, ok_pos, wind_kt, td_lead_hours, through_et)
        else:
            ok = ok_pos & (np.nan_to_num(wd) >= wind_kt)
        if not ok.any():
            continue
        ts = num2date(tt[ok], units, cal, only_use_cftime_datetimes=False,
                      only_use_python_datetimes=True)
        lo_ok = np.where(lo[ok] > 180, lo[ok] - 360, lo[ok])
        la_ok = la[ok]
        widx = np.where(ok)[0]
        for k, t in enumerate(ts):
            j = widx[k]
            if nat is not None and not lifecycle:
                code = _nature_code(nat, i, j)
                if code in exclude_nature:
                    n_drop_nat += 1
                    continue
            # Size preference: ROCI first. The model reads MSLP, vorticity and
            # wind, so the outermost closed isobar is the extent of the feature it
            # can actually resolve; r34 is the 34-kt WIND extent, tighter and partly
            # sub-grid. Measured over 2000-2012 (>=34 kt): ROCI median 324 km vs r34
            # 162 km. At 192x384 (~104 km/cell) that is ~30 px/storm vs ~7 -- and a
            # 7 px target at a ~0.0001 positive rate risks the head collapsing to
            # all-zero, which would be the worst possible outcome for recall.
            # r34 is kept as a secondary source, scaled by the observed ROCI/r34
            # ratio, so storms with only r34 reported still contribute real size.
            rad = np.nan
            if roci is not None:
                v = float(np.ma.filled(roci[i, j].astype(float), np.nan))
                if np.isfinite(v) and v > 0:
                    rad = v * NM_TO_KM
            if not np.isfinite(rad) and r34 is not None:
                q = np.ma.filled(r34[i, j].astype(float), np.nan)
                if np.isfinite(q).any():
                    rad = float(np.nanmean(q)) * NM_TO_KM * 2.0
            n_kept += 1
            out[t.replace(minute=0, second=0, microsecond=0)].append(
                (float(la_ok[k]), float(lo_ok[k]), rad))
    return out, n_kept, n_drop_nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--default-radius-km", type=float, default=300.0,
                    help="used where neither ROCI nor r34 is reported; near the "
                         "observed ROCI median of 324 km")
    ap.add_argument("--min-radius-km", type=float, default=200.0)
    ap.add_argument("--max-radius-km", type=float, default=500.0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--src-cache", default=SPLIT_CACHE,
                    help="cache whose other 9 channels are copied through; use the cache "
                         "the checkpoint being fine-tuned was trained on (v7 for H_final_*)")
    ap.add_argument("--lifecycle", action="store_true",
                    help="label every position from TS genesis to dissipation or ET, "
                         "regardless of wind (see lifecycle_mask)")
    ap.add_argument("--td-lead-hours", type=float, default=0.0,
                    help="with --lifecycle: also label up to this many hours of the "
                         "pre-genesis depression stage")
    ap.add_argument("--through-et", action="store_true",
                    help="with --lifecycle: keep labelling through extratropical transition")
    args = ap.parse_args()

    tag = f"{args.year:04d}{args.month:02d}"
    src_npy = os.path.join(args.src_cache, f"split_{tag}.npy")
    src_meta = os.path.join(args.src_cache, f"split_{tag}.meta.npy")
    if not os.path.exists(src_npy):
        raise SystemExit(f"missing {src_npy}")
    if args.outdir and not args.stats_only:
        dst = os.path.join(args.outdir, f"split_{tag}.npy")
        if os.path.exists(dst):
            print(f"  {tag} already done"); return 0

    times = np.load(src_meta, allow_pickle=True)
    old = np.load(src_npy, mmap_mode="r")
    nT = old.shape[0]

    # Build directly on the MODEL grid (192x384). The label is a disc of observed
    # radius, so it is defined in kilometres rather than pixels and needs no
    # resampling from the native grid — unlike the heuristic, which had to be
    # computed at 721x1440 and bilinearly reduced.
    H, W = TARGET
    glat = np.linspace(90.0, -90.0, H)
    glon = np.linspace(-180.0, 180.0 - 360.0 / W, W)
    lat2d = np.broadcast_to(glat[:, None], (H, W))
    lon2d = np.broadcast_to(glon[None, :], (H, W))
    coslat = np.cos(np.radians(lat2d))

    discs, n_kept, n_drop = ibtracs_discs(args.year, args.wind_kt,
                                          lifecycle=args.lifecycle,
                                          td_lead_hours=args.td_lead_hours,
                                          through_et=args.through_et)

    lab = np.zeros((nT, H, W), dtype=np.float32)
    n_pos = 0
    rads = []
    for i in range(nT):
        stamp = times[i].astype("datetime64[s]").astype(object)
        stamp = stamp.replace(minute=0, second=0, microsecond=0)
        for (la, lo, rad) in discs.get(stamp, []):
            r = args.default_radius_km if not np.isfinite(rad) else rad
            r = float(np.clip(r, args.min_radius_km, args.max_radius_km))
            rads.append(r)
            # Local equirectangular distance: exact enough at these radii and far
            # cheaper than haversine over the whole grid for every storm.
            dy = (lat2d - la) * 111.0
            dlon = (lon2d - lo + 180.0) % 360.0 - 180.0
            dx = dlon * 111.0 * coslat
            lab[i][(dx * dx + dy * dy) <= r * r] = 1.0
            n_pos += 1

    rate_new = float((lab > 0.5).mean())
    rate_old = float((np.asarray(old[:, HURR_CH]) > 0.5).mean())
    print(f"  {tag}: {n_pos} storm-positions ({n_kept} kept, {n_drop} dropped as ET)"
          f"  radius median {np.median(rads) if rads else float('nan'):.0f} km")
    print(f"    hurricane positive rate {rate_old:.5f} -> {rate_new:.5f} "
          f"({100*(rate_new-rate_old)/max(rate_old,1e-9):+.1f}%)", flush=True)
    if args.stats_only:
        return 0

    arr = np.array(old, dtype=np.float32)
    arr[:, HURR_CH] = lab
    os.makedirs(args.outdir, exist_ok=True)
    dst = os.path.join(args.outdir, f"split_{tag}.npy")
    tmp = f"{dst}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)
    import shutil
    shutil.copyfile(src_meta, os.path.join(args.outdir, f"split_{tag}.meta.npy"))
    print(f"    -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
