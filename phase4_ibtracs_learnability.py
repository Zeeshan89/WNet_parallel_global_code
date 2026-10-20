#!/usr/bin/env python3
"""
phase4_ibtracs_learnability.py — before building an IBTrACS-based TC target, find
out how much of IBTrACS is actually learnable from the model's inputs.

The question
------------
IBTrACS is observations, so it is the right thing to aim at. But a label is only
learnable if it is a function of what the model can see. IBTrACS records that a
storm existed at a position; it says nothing about whether ERA5 at 0.25 deg shows
anything there. Where ERA5 has no usable signature, an IBTrACS positive asks the
model to detect a storm from fields that do not contain one, and the only way it
can comply is by loosening its criteria everywhere — which is a plausible
contributor to the over-detection already measured.

So this measures, per IBTrACS position, whether the fields the model reads carry a
detectable signature:

  local MSLP minimum within --search-km
  vorticity of the correct sign at/near that minimum
  a pressure depression relative to surroundings
  850 mb wind maximum nearby

The fraction with a signature is the POD ceiling an IBTrACS-trained model could
approach. The fraction without is the part of the target that is noise to the model
no matter how long it trains.

Also reports how usable usa_r34 is, because if it is widely available the mask can
be sized from observed 34-kt wind radii instead of an arbitrary constant disc, and
whether `nature` lets us drop extratropical-transition points — those are ETCs, and
including them would teach the TC head to fire on extratropical systems.

Usage
  phase4_ibtracs_learnability.py --year 2003 --month 9
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_tc_label_screen import load_month, IBTRACS, haversine   # noqa: E402

NM_TO_KM = 1.852


def ibtracs_records(year):
    """Per 6-hourly stamp: [(lat, lon, wind_kt, nature, r34_km, rmw_km)]."""
    from netCDF4 import Dataset, num2date
    ds = Dataset(IBTRACS)
    season = np.array(ds.variables["season"][:], dtype=int)
    idx = np.where(season == year)[0]

    def g(name):
        return ds.variables[name][idx] if name in ds.variables else None

    lat, lon, wind = g("lat"), g("lon"), g("usa_wind")
    traw = ds.variables["time"][idx]
    units = ds.variables["time"].units
    cal = getattr(ds.variables["time"], "calendar", "standard")
    r34, rmw = g("usa_r34"), g("usa_rmw")
    nat = g("nature")
    ds.close()

    out = defaultdict(list)
    n_r34 = n_pts = 0
    for i in range(len(idx)):
        la = np.ma.filled(lat[i].astype(float), np.nan)
        lo = np.ma.filled(lon[i].astype(float), np.nan)
        wd = np.ma.filled(wind[i].astype(float), np.nan)
        tt = np.ma.filled(traw[i].astype(float), np.nan)
        ok = np.isfinite(la) & np.isfinite(lo) & np.isfinite(tt)
        if not ok.any():
            continue
        ts = num2date(tt[ok], units, cal, only_use_cftime_datetimes=False,
                      only_use_python_datetimes=True)
        lo_ok = np.where(lo[ok] > 180, lo[ok] - 360, lo[ok])
        la_ok, wd_ok = la[ok], wd[ok]
        widx = np.where(ok)[0]

        for k, t in enumerate(ts):
            j = widx[k]
            # nature is a 2-char code per point: TS tropical, ET extratropical,
            # SS subtropical, DS disturbance, MX mixed, NR not reported.
            nature = ""
            if nat is not None:
                try:
                    nature = b"".join(
                        [bytes(c) if isinstance(c, (bytes, np.bytes_)) else
                         str(c).encode() for c in np.ma.filled(nat[i, j], b" ")]
                    ).decode(errors="ignore").strip()
                except Exception:
                    nature = ""
            r34_km = np.nan
            if r34 is not None:
                q = np.ma.filled(r34[i, j].astype(float), np.nan)
                if np.isfinite(q).any():
                    r34_km = float(np.nanmean(q)) * NM_TO_KM
                    n_r34 += 1
            rmw_km = np.nan
            if rmw is not None:
                v = float(np.ma.filled(rmw[i, j].astype(float), np.nan))
                if np.isfinite(v):
                    rmw_km = v * NM_TO_KM
            n_pts += 1
            out[t.replace(minute=0, second=0, microsecond=0)].append(
                (float(la_ok[k]), float(lo_ok[k]), float(np.nan_to_num(wd_ok[k])),
                 nature, r34_km, rmw_km))
    return out, n_pts, n_r34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--search-km", type=float, default=250.0)
    ap.add_argument("--wind-kt", type=float, default=34.0)
    ap.add_argument("--outdir", default="tc_learnability")
    args = ap.parse_args()

    from scipy.ndimage import minimum_filter, maximum_filter

    tag = f"{args.year:04d}{args.month:02d}"
    os.makedirs(args.outdir, exist_ok=True)
    outf = os.path.join(args.outdir, f"learn_{tag}.json")
    if os.path.exists(outf):
        print(f"  {tag} already done"); return 0

    mslp, vort, wind, lats, times, clim = load_month(args.year, args.month)
    nT, H, W = mslp.shape
    lons = np.linspace(-180.0, 180.0 - 360.0 / W, W)
    recs, n_pts, n_r34 = ibtracs_records(args.year)

    grid_km = 0.25 * 111.0
    rad_px = max(1, int(round(args.search_km / grid_km)))

    c = defaultdict(int)
    natures = defaultdict(int)
    r34_vals, depth_vals = [], []

    for i in range(nT):
        stamp = times[i].to_pydatetime().replace(minute=0, second=0, microsecond=0)
        pts = recs.get(stamp, [])
        if not pts:
            continue
        locmin = (mslp[i] == minimum_filter(mslp[i], size=5, mode="wrap"))
        ringmax = maximum_filter(mslp[i], size=2 * 22 + 1, mode="wrap")
        wmax = maximum_filter(wind[i], size=2 * 4 + 1, mode="nearest")

        for (la, lo, wkt, nature, r34_km, rmw_km) in pts:
            natures[nature or "??"] += 1
            if wkt < args.wind_kt:
                continue
            c["n"] += 1
            if np.isfinite(r34_km):
                r34_vals.append(r34_km)
            r = int(np.argmin(np.abs(lats - la)))
            cc = int(np.argmin(np.abs(lons - lo)))
            r0, r1 = max(0, r - rad_px), min(H, r + rad_px + 1)
            c0, c1 = max(0, cc - rad_px), min(W, cc + rad_px + 1)

            box_lm = locmin[r0:r1, c0:c1]
            if box_lm.any():
                c["has_localmin"] += 1
                sub = np.where(box_lm, mslp[i, r0:r1, c0:c1], np.inf)
                k = np.unravel_index(np.argmin(sub), sub.shape)
                rr, ccx = r0 + k[0], c0 + k[1]
                depth = float(ringmax[rr, ccx] - mslp[i, rr, ccx])
                depth_vals.append(depth)
                if depth >= 150.0:
                    c["has_depth150"] += 1
                if depth >= 300.0:
                    c["has_depth300"] += 1
                vs = vort[i, rr, ccx]
                if (la >= 0 and vs >= 1.45e-5) or (la < 0 and vs <= -1.45e-5):
                    c["has_vort"] += 1
                if wmax[rr, ccx] >= 10.0:
                    c["has_wind10"] += 1
                if wmax[rr, ccx] >= 17.5:
                    c["has_wind175"] += 1
                if ((la >= 0 and vs >= 1.45e-5) or (la < 0 and vs <= -1.45e-5)) \
                        and wmax[rr, ccx] >= 10.0:
                    c["has_vort_and_wind"] += 1

    n = max(c["n"], 1)
    res = dict(tag=tag, n_positions=c["n"], n_all_points=n_pts,
               n_with_r34=n_r34, natures=dict(natures),
               frac={k: c[k] / n for k in c if k != "n"},
               r34_km_median=float(np.median(r34_vals)) if r34_vals else None,
               r34_km_p10=float(np.percentile(r34_vals, 10)) if r34_vals else None,
               r34_km_p90=float(np.percentile(r34_vals, 90)) if r34_vals else None,
               depth_pa_median=float(np.median(depth_vals)) if depth_vals else None)
    with open(outf, "w") as f:
        json.dump(res, f, indent=1)

    print(f"  {tag}: {c['n']} IBTrACS positions >= {args.wind_kt:g} kt "
          f"(of {n_pts} total points)")
    for k in ("has_localmin", "has_depth150", "has_depth300", "has_vort",
              "has_wind10", "has_wind175", "has_vort_and_wind"):
        print(f"    {k:<20} {c[k]:>5} / {c['n']:<5} = {c[k]/n:.3f}")
    print(f"    usa_r34 available on {n_r34}/{n_pts} points "
          f"({n_r34/max(n_pts,1):.2f})", end="")
    if r34_vals:
        print(f"  median {np.median(r34_vals):.0f} km "
              f"(p10 {np.percentile(r34_vals,10):.0f}, "
              f"p90 {np.percentile(r34_vals,90):.0f})")
    else:
        print()
    print(f"    nature codes: {dict(natures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
