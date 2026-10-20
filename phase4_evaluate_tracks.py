#!/usr/bin/env python3
"""
phase4_evaluate_tracks.py — TRACK-level POD / FAR / CSI, and a post-tracker filter
sweep that needs no retraining and no tracker re-runs.

Why this exists
---------------
phase4_evaluate_tc.py scores per-timestep BLOB CENTROIDS. At p=0.5 it counts 3,688
hits against 34,073 false alarms, and raising the probability threshold to 0.99 only
gets to 28,707. That metric cannot see a track filter at all: a spurious blob that
appears for one time step and vanishes is counted exactly like a real storm's blob.
But transience is precisely what separates the model's false alarms from real
storms — the model emits ~3.9 spurious detections per step against
TempestExtremes' 0.34 — so the natural filter is temporal continuity, and it is
invisible to the existing score.

Track matching already existed, but only inside the plotting script
plot_tracks_peryear.py (it is where the "old 1775 / new 3723 / real 433" counts came
from). This script imports that matcher rather than reimplementing it, so the
numbers here and the colours in the figures always mean the same thing.

Scoring convention
------------------
  hit          a REFERENCE track matched by at least one model track
  miss         a reference track no model track matched
  false alarm  a model track matching no reference track
  duplicate    a model track matching a reference already claimed by another
               model track. Reported separately and NOT counted as a false alarm,
               because a split track is a tracking artefact rather than an
               invented storm. It is printed so the choice stays visible.

  POD = hits / n_reference
  FAR = false_alarms / n_model
  CSI = hits / (hits + misses + false_alarms)

Post-tracker filters (the sweep)
--------------------------------
Every --min-*/--max-* flag is applied to the PARSED tracks before matching, so a
threshold sweep costs seconds and touches neither the trackers nor the detection
NetCDFs. This is the cheap way to explore the levers that Hurricane_track.py
(:59-79, :578-788) and updated_monthly_ETC_Tracker.py (:342-402) already hardcode,
before committing any of them to a tracker re-run.

Environment: the xarray/cartopy env (plot_tracks_peryear imports cartopy).
  /ccs/home/zaslam/.conda/envs/my_env/bin/python

Usage
  # baseline, no filtering
  phase4_evaluate_tracks.py --kind tc

  # one candidate filter set
  phase4_evaluate_tracks.py --kind tc --min-duration-h 48 --min-path-km 250

  # sweep one lever
  phase4_evaluate_tracks.py --kind tc --sweep min-duration-h 0 24 48 72 96
"""
import os
import csv
import sys
import glob
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the parsers and the matcher rather than rewriting them, so this script and
# plot_tracks_peryear.py can never drift apart.
from plot_tracks_peryear import (          # noqa: E402
    parse_hurricane_txt,
    parse_etc_txt,
    parse_te_etc,
    ibtracs_tracks,
    match_tracks,
    haversine,
    clip_to_box,
    OLD, NEW, SD, TEST_YEARS, ETC_BOX,
)

TE_TC_DIR = os.path.join(SD, "TCs", "TE")
TRACK_ETC_DIR = os.path.join(SD, "ETCs", "TRACK")
NAEC_CSV = os.path.join(SD, "ETCs", "NAEC", "NAEC_1979_2020.csv")

# Same box as phase4_evaluate_tc.py --te-domain, so the two scripts agree.
TE_BOX = dict(lon=(-135.0, -55.0), lat=(5.0, 60.0))

# No ETC reference is global, so a tracker cannot be designed against one alone.
#   TE (CONUS)  lon -135..-55, lat 10..55   -- ETC_BOX in plot_tracks_peryear
#   NAEC        lon -180..0,   lat 20..80   -- NAEC_DOMAIN in etc_analysis/
# COMMON_BOX is their intersection, the only window where both references and the
# model can be scored on identical ground.
NAEC_DOMAIN = dict(lon=(-180.0, 0.0), lat=(20.0, 80.0))
COMMON_BOX = dict(lon=(-135.0, -55.0), lat=(20.0, 55.0))


# ── reference / competitor loaders ───────────────────────────────────────────
def parse_te_tc(year):
    """TempestExtremes TC trajectories -> list of tracks [(datetime, lat, lon)].

    Same CSV layout as the ETC trajectories parsed by parse_te_etc: column 0 is the
    track id, 1-4 are year/month/day/hour, 7 is longitude (0..360) and 8 latitude.
    """
    cands = sorted(glob.glob(os.path.join(
        TE_TC_DIR, f"ERA5_trajectories_hur_{year}_6hr_csv*.txt")))
    if not cands:
        return []
    byid = defaultdict(list)
    with open(cands[-1], errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(row) < 9:
                continue
            try:
                t = datetime(int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                lon = float(row[7])
                lat = float(row[8])
            except (ValueError, IndexError):
                continue
            if lon > 180:
                lon -= 360
            byid[int(row[0])].append((t, lat, lon))
    return [sorted(v) for v in byid.values() if len(v) >= 2]


_NAEC_CACHE = {}


def parse_naec(year):
    """NAEC storms for one year -> list of tracks [(datetime, lat, lon)].

    Conventions copied from etc_analysis/global_etc_analysis.py:load_naec_csv so
    the two agree: grouped by 'storm', datetime is YYYYMMDDHH, longitude is 0-360
    and is wrapped to -180..180.
    """
    if not _NAEC_CACHE:
        import pandas as pd
        df = pd.read_csv(NAEC_CSV)
        df.columns = [c.strip() for c in df.columns]
        for sid, g in df.groupby("storm"):
            g = g.sort_values("datetime")
            try:
                ts = [datetime.strptime(str(int(d)), "%Y%m%d%H") for d in g["datetime"]]
            except (ValueError, TypeError):
                continue
            if len(ts) < 2:
                continue
            la = g["latitude"].astype(float).tolist()
            lo = (((g["longitude"].astype(float) + 180) % 360) - 180).tolist()
            _NAEC_CACHE.setdefault(ts[0].year, []).append(list(zip(ts, la, lo)))
    return _NAEC_CACHE.get(year, [])


# ── track geometry, shared by every filter ───────────────────────────────────
def track_stats(track):
    """Duration (h), path length (km), net displacement (km), speeds (m/s)."""
    t = sorted(track)
    times = [p[0] for p in t]
    la = np.array([p[1] for p in t], dtype=float)
    lo = np.array([p[2] for p in t], dtype=float)

    duration_h = (times[-1] - times[0]).total_seconds() / 3600.0
    seg_km = haversine(la[:-1], lo[:-1], la[1:], lo[1:])
    dt_h = np.array([(times[i + 1] - times[i]).total_seconds() / 3600.0
                     for i in range(len(times) - 1)], dtype=float)
    # A zero-length time step would divide by zero; such duplicate stamps carry no
    # movement information, so drop them from the speed statistics rather than
    # letting them produce an infinite speed that trips the max-speed gate.
    ok = dt_h > 0
    speeds = np.where(ok, seg_km * 1000.0 / np.where(ok, dt_h * 3600.0, 1.0), np.nan)
    net_km = float(haversine(la[0], lo[0], la[-1], lo[-1]))
    path_km = float(seg_km.sum())
    return {
        "n_points": len(t),
        "duration_h": float(duration_h),
        "path_km": path_km,
        "net_km": net_km,
        "straightness": float(net_km / path_km) if path_km > 0 else 0.0,
        "median_speed_ms": float(np.nanmedian(speeds)) if ok.any() else 0.0,
        "genesis_lat": float(la[0]),
        "max_abs_lat": float(np.max(np.abs(la))),
    }


def qc_filter(tracks, min_hours=48.0, min_points=8, min_net_km=1000.0):
    """Symmetric quality control, applied to MODEL and REFERENCE alike.

    Mirrors etc_analysis/global_etc_analysis.py:qc_filter. Without it a comparison
    against NAEC is unfair in the model's favour and against it at the same time:
    NAEC is an observational database that includes short, weak systems this
    tracker structurally cannot emit (it enforces 48 h AND an 1800 km path), so
    those storms would be counted as misses the model never had a chance at.
    Applying the same floor to both sides is what makes the two benchmarks
    comparable to each other as well as to us.

    Note min_net_km is NET displacement (first point to last), not path length,
    matching the reference implementation.
    """
    keep = []
    for tr in tracks:
        s = track_stats(tr)
        if s["duration_h"] < min_hours or s["n_points"] < min_points:
            continue
        if s["net_km"] < min_net_km:
            continue
        keep.append(tr)
    return keep


def filter_tracks(tracks, args):
    """Apply the post-tracker gates. Returns (kept_tracks, n_dropped_by_reason)."""
    kept = []
    dropped = defaultdict(int)
    for tr in tracks:
        s = track_stats(tr)
        if args.min_points is not None and s["n_points"] < args.min_points:
            dropped["n_points"] += 1
            continue
        if args.min_duration_h is not None and s["duration_h"] < args.min_duration_h:
            dropped["duration"] += 1
            continue
        if args.min_path_km is not None and s["path_km"] < args.min_path_km:
            dropped["path"] += 1
            continue
        if args.min_straightness is not None and s["straightness"] < args.min_straightness:
            dropped["straightness"] += 1
            continue
        if args.min_speed_ms is not None and s["median_speed_ms"] < args.min_speed_ms:
            dropped["slow"] += 1
            continue
        if args.max_speed_ms is not None and s["median_speed_ms"] > args.max_speed_ms:
            dropped["fast"] += 1
            continue
        if args.genesis_lat_max is not None and abs(s["genesis_lat"]) > args.genesis_lat_max:
            dropped["genesis_lat"] += 1
            continue
        if args.max_abs_lat is not None and s["max_abs_lat"] > args.max_abs_lat:
            dropped["abs_lat"] += 1
            continue
        kept.append(tr)
    return kept, dict(dropped)


# ── scoring ──────────────────────────────────────────────────────────────────
def score(model, ref, radius_km, min_hits, tol_h):
    """Track-level contingency table. See the convention in the module docstring."""
    if not ref:
        return None
    assign = match_tracks(model, ref, radius_km, min_hits=min_hits, tol_h=tol_h)

    matched_refs = set()
    duplicates = 0
    false_alarms = 0
    for j in assign:
        if j is None:
            false_alarms += 1
        elif j in matched_refs:
            duplicates += 1
        else:
            matched_refs.add(j)

    hits = len(matched_refs)
    misses = len(ref) - hits
    denom = hits + misses + false_alarms
    return {
        "n_model": len(model),
        "n_ref": len(ref),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "duplicates": duplicates,
        "pod": hits / len(ref) if ref else 0.0,
        "far": false_alarms / len(model) if model else 0.0,
        "csi": hits / denom if denom else 0.0,
    }


def aggregate(per_year):
    """Pool the contingency counts across years, then recompute the rates.

    Pooling counts rather than averaging rates keeps a year with two storms from
    carrying the same weight as a year with forty.
    """
    tot = defaultdict(int)
    for r in per_year.values():
        if r is None:
            continue
        for k in ("n_model", "n_ref", "hits", "misses", "false_alarms", "duplicates"):
            tot[k] += r[k]
    denom = tot["hits"] + tot["misses"] + tot["false_alarms"]
    out = dict(tot)
    out["pod"] = tot["hits"] / tot["n_ref"] if tot["n_ref"] else 0.0
    out["far"] = tot["false_alarms"] / tot["n_model"] if tot["n_model"] else 0.0
    out["csi"] = tot["hits"] / denom if denom else 0.0
    return out


# ── loading one year ─────────────────────────────────────────────────────────
_YEAR_CACHE = {}


def load_year(kind, year, new_dir, wind_kt, ref_name="te"):
    """Return (sources dict, reference tracks, reference name, box).

    Memoised because a --sweep re-scores the same years once per value, and
    ibtracs_tracks re-opens a 23 MB NetCDF and loops every storm each call. Without
    this a three-lever sweep spends most of its time re-reading IBTrACS.
    """
    key = (kind, year, new_dir, wind_kt, ref_name)
    if key not in _YEAR_CACHE:
        _YEAR_CACHE[key] = _load_year_uncached(kind, year, new_dir, wind_kt, ref_name)
    return _YEAR_CACHE[key]


def _load_year_uncached(kind, year, new_dir, wind_kt, ref_name="te"):
    if kind == "tc":
        old = parse_hurricane_txt(os.path.join(
            OLD, "Hurricanes_WNet_final", f"CONUS_year_{year}",
            f"CONUS_hurricane_tracks_detailed_{year}.txt"))
        new = parse_hurricane_txt(os.path.join(
            new_dir, "hurricane", f"CONUS_year_{year}",
            f"CONUS_hurricane_tracks_detailed_{year}.txt"))
        te = parse_te_tc(year)
        ref = ibtracs_tracks(year, wind_kt=wind_kt)
        return ({"old": old, "new": new, "TempestExtremes": te},
                ref, f"IBTrACS (>={wind_kt:g} kt)", None)

    old = parse_etc_txt(os.path.join(
        OLD, "ETCs_WNet_final", f"ETC_tracks_{year}_meteorological.txt"))
    new = parse_etc_txt(os.path.join(
        new_dir, "etc", f"ETC_tracks_{year}_meteorological_v2.txt"))
    if ref_name == "naec":
        return ({"old": old, "new": new}, parse_naec(year),
                "NAEC (NH-western)", NAEC_DOMAIN)
    return ({"old": old, "new": new}, parse_te_etc(year),
            "TempestExtremes (CONUS)", ETC_BOX)


def run(args, years):
    """Score every source for every year. Returns (per_source_per_year, totals)."""
    results = defaultdict(dict)
    counts = defaultdict(dict)
    drops = defaultdict(lambda: defaultdict(int))

    for y in years:
        sources, ref, refname, box = load_year(args.kind, y, args.track_dir,
                                               args.wind_kt,
                                               getattr(args, "ref", "te"))
        # --te-domain overrides the per-kind default box, and applies to every
        # detector including the reference, so all three are scored in the same
        # window.
        if args.te_domain:
            box = TE_BOX
        elif getattr(args, "common_box", False):
            box = COMMON_BOX
        elif args.box == "none":
            box = None
        if box:
            ref, _ = clip_to_box(ref, box)
        if getattr(args, "qc", False):
            ref = qc_filter(ref)
        for name, tracks in sources.items():
            if not tracks:
                continue
            if box:
                tracks, _ = clip_to_box(tracks, box)
            if getattr(args, "qc", False):
                tracks = qc_filter(tracks)
            # The competitor and the previous system are shown unfiltered: the
            # filters are a proposal for OUR output, and applying them to the
            # baseline would flatter the comparison.
            if name == "new":
                tracks, d = filter_tracks(tracks, args)
                for k, v in d.items():
                    drops[name][k] += v
            results[name][y] = score(tracks, ref, args.radius_km,
                                     args.min_hits, args.tol_h)
            counts[name][y] = len(tracks)
        results["_ref"][y] = {"n_ref": len(ref), "name": refname}

    totals = {name: aggregate(per_year)
              for name, per_year in results.items() if name != "_ref"}
    return results, totals, dict(drops)


def print_table(args, years, results, totals, drops):
    ref_name = results["_ref"][years[0]]["name"] if years else "?"
    print(f"\n  kind={args.kind}  reference={ref_name}  "
          f"radius={args.radius_km:g} km  min_hits={args.min_hits}  tol={args.tol_h} h")
    if args.te_domain:
        print(f"  domain: TE box lon {TE_BOX['lon']}, lat {TE_BOX['lat']} "
              f"(all detectors and the reference)")
    elif args.box != "none" and args.kind == "etc":
        print(f"  domain: CONUS box lon {ETC_BOX['lon']}, lat {ETC_BOX['lat']}")

    active = [f"{k}={v}" for k, v in sorted(vars(args).items())
              if k in ("min_points", "min_duration_h", "min_path_km",
                       "min_straightness", "min_speed_ms", "max_speed_ms",
                       "genesis_lat_max", "max_abs_lat") and v is not None]
    print(f"  filters on 'new': {', '.join(active) if active else 'NONE (baseline)'}")

    for name in [n for n in ("new", "old", "TempestExtremes") if n in results]:
        print(f"\n  === {name} ===")
        print(f"  {'year':<7}{'tracks':>8}{'ref':>6}{'hits':>6}{'miss':>6}"
              f"{'FA':>7}{'dup':>6}{'POD':>8}{'FAR':>8}{'CSI':>8}")
        print("  " + "-" * 70)
        for y in years:
            r = results[name].get(y)
            if r is None:
                continue
            print(f"  {y:<7}{r['n_model']:>8}{r['n_ref']:>6}{r['hits']:>6}"
                  f"{r['misses']:>6}{r['false_alarms']:>7}{r['duplicates']:>6}"
                  f"{r['pod']:>8.3f}{r['far']:>8.3f}{r['csi']:>8.3f}")
        t = totals[name]
        print("  " + "-" * 70)
        print(f"  {'ALL':<7}{t['n_model']:>8}{t['n_ref']:>6}{t['hits']:>6}"
              f"{t['misses']:>6}{t['false_alarms']:>7}{t['duplicates']:>6}"
              f"{t['pod']:>8.3f}{t['far']:>8.3f}{t['csi']:>8.3f}")
        if name in drops and drops[name]:
            tot = sum(drops[name].values())
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(drops[name].items()))
            print(f"  filtered out {tot} tracks: {reasons}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tc", "etc"], required=True)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--track-dir", default=NEW,
                    help="directory holding the new model's tracks")
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--min-hits", type=int, default=2,
                    help="corresponding points needed to call a track matched")
    ap.add_argument("--tol-h", type=int, default=3,
                    help="timestamp tolerance when matching, hours")
    ap.add_argument("--wind-kt", type=float, default=34.0,
                    help="IBTrACS intensity floor for a storm to count as real")
    ap.add_argument("--box", choices=["none", "default"], default="default",
                    help="'default' applies the CONUS box for ETC, none for TC")
    ap.add_argument("--ref", choices=["te", "naec"], default="te",
                    help="ETC reference: TempestExtremes (CONUS) or NAEC "
                         "(NH-western observational database)")
    ap.add_argument("--common-box", action="store_true",
                    help="score inside the TE-NAEC intersection "
                         "(lon -135..-55, lat 20..55) — the only window where "
                         "both references and the model share ground")
    ap.add_argument("--qc", action="store_true",
                    help="apply the symmetric QC of "
                         "etc_analysis/global_etc_analysis.py (48 h, 8 points, "
                         "1000 km net displacement) to MODEL and REFERENCE alike")
    ap.add_argument("--te-domain", action="store_true",
                    help="restrict every detector to the TempestExtremes box "
                         "(lat 5..60, lon -135..-55). The TE TC trajectory files "
                         "are NOT global — they span roughly lat 10..46, "
                         "lon -135..-60 — so scoring a regional tracker against "
                         "global IBTrACS counts every storm outside its box as a "
                         "miss and makes its POD meaningless. Matches the "
                         "--te-domain flag in phase4_evaluate_tc.py.")

    g = ap.add_argument_group("post-tracker filters (applied to 'new' only)")
    g.add_argument("--min-points", type=int, default=None)
    g.add_argument("--min-duration-h", type=float, default=None)
    g.add_argument("--min-path-km", type=float, default=None)
    g.add_argument("--min-straightness", type=float, default=None,
                   help="net displacement / path length, 0-1")
    g.add_argument("--min-speed-ms", type=float, default=None,
                   help="median segment speed floor")
    g.add_argument("--max-speed-ms", type=float, default=None)
    g.add_argument("--genesis-lat-max", type=float, default=None,
                   help="reject tracks whose first point is poleward of this")
    g.add_argument("--max-abs-lat", type=float, default=None,
                   help="reject tracks reaching poleward of this")

    ap.add_argument("--sweep", nargs="+", default=None, metavar="LEVER VALUE",
                    help="sweep one filter, e.g. --sweep min-duration-h 0 24 48 72")
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args()

    years = args.years or TEST_YEARS

    if args.sweep:
        lever = args.sweep[0].replace("-", "_")
        if not hasattr(args, lever):
            raise SystemExit(f"unknown sweep lever '{args.sweep[0]}'")
        values = [float(v) for v in args.sweep[1:]]
        if not values:
            raise SystemExit("--sweep needs at least one value")
        print(f"\n  sweeping {args.sweep[0]} over {values}  (kind={args.kind})")
        print(f"\n  {'value':>10}{'tracks':>9}{'hits':>6}{'FA':>7}"
              f"{'POD':>8}{'FAR':>8}{'CSI':>8}")
        print("  " + "-" * 56)
        rows = []
        for v in values:
            setattr(args, lever, v)
            _, totals, _ = run(args, years)
            t = totals["new"]
            print(f"  {v:>10g}{t['n_model']:>9}{t['hits']:>6}{t['false_alarms']:>7}"
                  f"{t['pod']:>8.3f}{t['far']:>8.3f}{t['csi']:>8.3f}")
            rows.append({"value": v, **t})
        # The competitor does not move with our filter, so print it once as the bar.
        setattr(args, lever, None)
        _, totals, _ = run(args, years)
        for comp in ("TempestExtremes", "old"):
            if comp in totals and totals[comp]["n_ref"]:
                t = totals[comp]
                print(f"  {comp:>10}{t['n_model']:>9}{t['hits']:>6}"
                      f"{t['false_alarms']:>7}{t['pod']:>8.3f}{t['far']:>8.3f}"
                      f"{t['csi']:>8.3f}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"kind": args.kind, "lever": args.sweep[0],
                           "years": years, "sweep": rows}, f, indent=1)
            print(f"\n  wrote {args.json}")
        return 0

    results, totals, drops = run(args, years)
    years = [y for y in years if any(results[n].get(y) for n in totals)]
    if not years:
        raise SystemExit("no years with tracks — check --track-dir")
    print_table(args, years, results, totals, drops)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"kind": args.kind, "years": years,
                       "radius_km": args.radius_km,
                       "filters": {k: v for k, v in vars(args).items()
                                   if k.startswith(("min_", "max_", "genesis_"))},
                       "per_year": {n: results[n] for n in totals},
                       "totals": totals}, f, indent=1, default=str)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
