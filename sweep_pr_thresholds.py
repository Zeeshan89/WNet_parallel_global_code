#!/usr/bin/env python
"""Threshold validation sweep for the precipitation-based MCS detector.

WHY THIS EXISTS
---------------
The trained WNet reproduces a winter-peaked MCS maximum over the Cascades/PNW:

    ACC grid-cell frequency   Jan      Jul
      Midwest                 0.20%   11.03%   <- correct, summer-peaked
      Cascades                6.50%    4.55%   <- WRONG, winter-peaked

That signal is present in the TRAINING TARGET (tensor cache ch13: Cascades Jan
7.53% vs Jul 1.96%), so the network is faithfully learning a flawed target. The
per-pixel convective-fraction gate (cf >= 0.7) holds on a regional average
(Cascades Jan cf ~ 0.44) but leaks on individual convective bursts embedded in
winter orographic precipitation, which are extensive enough to clear the
60,000 km^2 area test.

Fixing that means re-deriving the masks -> regenerate caches -> retrain -> retest
for all six models. This script validates the threshold change FIRST, on CPU,
with nothing retrained.

WHAT IT DOES
------------
Drives the REAL detector (dataset._detect_mcs_blobs_precip, via the same shim
verify_precip_mcs.py uses) over native-grid pr/prc, sweeping shield/core/cf
combinations. Native grid on purpose: it isolates the detector from the regrid.

Seasonal geometry (min_blob_area / min_core_pixels) is held FIXED at the values
in dataset._get_seasonal_mcs_parameters — only shield/core/cf vary, so the result
is attributable to the thresholds alone.

USAGE
-----
    python sweep_pr_thresholds.py --model ACC
    python sweep_pr_thresholds.py --model ACC --combos 0 3   # subset
    python sweep_pr_thresholds.py --model ACC --years 1995 2000 --quick

Writes sweep_pr_thresholds_<MODEL>.csv (one row per model/combo/region/month).
Aggregate across models afterwards with --summarize.

HARNESS CHECK
-------------
Combo 0 is the CURRENT production setting and must reproduce the already-measured
ACC values (Cascades Jan ~12.0, Midwest Jul ~10.8 for 1995+2000). If it does not,
the harness is wrong, not the thresholds — fix that before trusting any row.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARDatasetImproved as D

BASE = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"
KNOWN_MODELS = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]

# Candidate thresholds: (shield mm/h, core mm/h, convective fraction).
# Combo 0 is the current production setting and acts as the control.
COMBOS = [
    (0.1, 1.0, 0.7),   # 0  current / control
    (0.1, 1.0, 0.9),   # 1  tighten cf only
    (0.1, 2.0, 0.8),   # 2
    (0.1, 3.0, 0.8),   # 3
    (0.2, 2.0, 0.8),   # 4
    # --- round 2 -----------------------------------------------------------
    # The 12-month sweep showed the optimum is BRACKETED between combos 2 and 3:
    #   combo 2: ACC flips + retains 87%, but MPI/NOR do not flip
    #   combo 3: MPI/NOR flip, but ACC retention falls to 71% (below the 75% bar)
    # No combo cleared 4/6. These probe the gap between them. The pass criteria
    # are UNCHANGED — only the threshold search is finer.
    (0.1, 2.5, 0.8),   # 5  midpoint — most promising
    (0.1, 2.5, 0.85),  # 6  midpoint + slightly tighter cf
    (0.1, 3.0, 0.75),  # 7  strong core, relaxed cf (buys back retention)
]

# Seasonal geometry, mirroring dataset._get_seasonal_mcs_parameters. HELD FIXED.
SEASONAL = {**{m: (4, 3) for m in (6, 7, 8)},              # summer: area, core_px
            **{m: (12, 8) for m in (12, 1, 2)},            # winter
            **{m: (6, 5) for m in (3, 4, 5, 9, 10, 11)}}   # transition

MIN_CELLS = 96          # 60,000 km^2 at ~25 km native spacing

# Regions as (lat_min, lat_max, lon_min_360, lon_max_360)
REGIONS = {
    "Cascades":    (42, 49, 235, 239),   # the artifact
    "GreatPlains": (40, 45, 260, 268),
    "Midwest":     (38, 43, 268, 274),
    "Southeast":   (31, 36, 268, 277),
    "Gulf":        (18, 28, 262, 285),
}


class _Det:
    """Minimal shim exposing the real detection methods (same as verify_precip_mcs.py)."""
    _convective_fraction = D.__dict__['_convective_fraction']
    _blobs_from_shield_core = D._blobs_from_shield_core
    _accept_system = D._accept_system
    _detect_mcs_blobs_precip = D._detect_mcs_blobs_precip


def _filter_min_cells(binary, min_cells=MIN_CELLS):
    """60,000 km^2 minimum-area criterion, applied per frame."""
    lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(binary)
    sizes = ndimage.sum(np.ones_like(binary), lab, np.arange(1, n + 1))
    keep = np.arange(1, n + 1)[sizes >= min_cells]
    return np.isin(lab, keep)


def _load_month(model, year, month):
    """(prc, pr) in mm/h on the native Lambert grid, or (None, None)."""
    fp = glob.glob(f"{BASE}/{model}/pr/*.{year}{month:02d}0100_pr_6hourly.nc")
    fc = glob.glob(f"{BASE}/{model}/prc/*.{year}{month:02d}0100_prc_6hourly.nc")
    if not fp or not fc:
        return None, None
    # native files are kg m-2 s-1; x3600 -> mm/h (what Run_PR_regrid.sh bakes in)
    pr = xr.open_dataset(fp[0], decode_times=False).pr.values * 3600.0
    prc = xr.open_dataset(fc[0], decode_times=False).prc.values * 3600.0
    return prc.astype(np.float32), pr.astype(np.float32)


def _region_masks(model):
    f = sorted(glob.glob(f"{BASE}/{model}/pr/*_pr_6hourly.nc"))
    if not f:
        sys.exit(f"No native pr files for {model} under {BASE}/{model}/pr")
    g = xr.open_dataset(f[0], decode_times=False)
    xlat, xlon = g.xlat.values, g.xlon.values
    lon360 = np.where(xlon < 0, xlon + 360, xlon)
    return {name: (xlat >= a) & (xlat <= b) & (lon360 >= c) & (lon360 <= d)
            for name, (a, b, c, d) in REGIONS.items()}


def run(model, years, months, combo_ids):
    det = _Det()
    masks = _region_masks(model)
    rows = []

    for ci in combo_ids:
        shield, core, cf_thr = COMBOS[ci]
        tag = f"{shield}/{core}/{cf_thr}"
        print(f"\n[{model}] combo {ci}: shield prc>={shield}, core prc>={core}, cf>={cf_thr}",
              flush=True)

        for month in months:
            min_area, min_core = SEASONAL[month]
            acc = {r: 0.0 for r in REGIONS}
            nstep = 0

            for year in years:
                prc_f, pr_f = _load_month(model, year, month)
                if prc_f is None:
                    continue
                for t in range(prc_f.shape[0]):
                    blob, _, _ = det._detect_mcs_blobs_precip(
                        prc_f[t], pr_f[t],
                        shield_threshold=shield,
                        core_threshold=core,
                        convfrac_threshold=cf_thr,
                        min_core_pixels=min_core,
                        min_blob_area=min_area,
                    )
                    kept = _filter_min_cells(blob > 0)
                    for r, m in masks.items():
                        acc[r] += kept[m].mean()
                    nstep += 1

            if nstep == 0:
                continue
            for r in REGIONS:
                rows.append({
                    "model": model, "combo": ci, "thresholds": tag,
                    "shield": shield, "core": core, "cf": cf_thr,
                    "region": r, "month": month,
                    # % of 6-h steps a grid cell in this region is flagged MCS
                    "freq_pct": 100.0 * acc[r] / nstep,
                    # contribution of this month to annual MCS days
                    "mcs_days": acc[r] / 4.0 / max(len(years), 1),
                    "nsteps": nstep,
                })
            print(f"    month {month:2d}: " +
                  "  ".join(f"{r}={100*acc[r]/nstep:5.2f}%" for r in
                            ("Cascades", "Midwest", "GreatPlains")), flush=True)

    return pd.DataFrame(rows)


def summarize(df):
    """Report the four pre-registered pass criteria."""
    print("\n" + "=" * 78)
    print("PRE-REGISTERED PASS CRITERIA (a combo is adopted if >=4 of 6 models pass)")
    print("=" * 78)
    print("  1. Cascades flips to SUMMER-peaked (Jul > Jan)        <- the actual bug")
    print("  2. Great Plains stays summer-peaked (JJA/DJF >= 20x)")
    print("  3. Midwest JJA retained >= 75% of control (combo 0)")
    print("  4. Interior annual mean moves toward ERA5 30-50 d/yr")

    def _f(d, region, months):
        s = d[(d.region == region) & (d.month.isin(months))]
        return s.freq_pct.mean() if len(s) else np.nan

    for combo in sorted(df.combo.unique()):
        c = df[df.combo == combo]
        tag = c.thresholds.iloc[0]
        print(f"\n--- combo {combo}  ({tag}) " + "-" * 44)
        print(f"{'model':<7}{'CasJan':>8}{'CasJul':>8}{'flip':>7}"
              f"{'GP JJA/DJF':>12}{'MidJJA':>9}{'vs ctrl':>9}   pass")
        npass = 0
        for m in sorted(c.model.unique()):
            d = c[c.model == m]
            cas_j, cas_l = _f(d, "Cascades", [1]), _f(d, "Cascades", [7])
            gp_djf = _f(d, "GreatPlains", [12, 1, 2])
            gp_jja = _f(d, "GreatPlains", [6, 7, 8])
            mid_jja = _f(d, "Midwest", [6, 7, 8])
            ctrl = df[(df.combo == 0) & (df.model == m) & (df.region == "Midwest")
                      & (df.month.isin([6, 7, 8]))].freq_pct.mean()
            ratio = gp_jja / gp_djf if gp_djf and gp_djf > 0 else np.inf
            frac = mid_jja / ctrl if ctrl and ctrl > 0 else np.nan
            c1, c2, c3 = cas_l > cas_j, ratio >= 20, (frac >= 0.75)
            ok = c1 and c2 and c3
            npass += ok
            print(f"{m:<7}{cas_j:8.2f}{cas_l:8.2f}{'YES' if c1 else 'no':>7}"
                  f"{ratio:12.0f}{mid_jja:9.2f}{frac:8.0%}   {'PASS' if ok else 'fail'}")
        print(f"  -> {npass}/6 models pass"
              + ("   ** ADOPT **" if npass >= 4 else ""))


def main():
    p = argparse.ArgumentParser(description="Sweep precip-MCS thresholds (CPU, no retraining)")
    p.add_argument("--model", default=os.environ.get("REGCM_MODEL", "ACC"))
    p.add_argument("--years", type=int, nargs="+", default=[1995, 2000, 2005, 2010])
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--combos", type=int, nargs="+", default=list(range(len(COMBOS))))
    p.add_argument("--quick", action="store_true",
                   help="Jan+Jul only — the artifact test; for a fast harness check")
    p.add_argument("--out", default=None)
    p.add_argument("--summarize", action="store_true",
                   help="Read existing per-model CSVs and print the criteria table")
    args = p.parse_args()

    if args.summarize:
        fs = sorted(glob.glob("sweep_pr_thresholds_*.csv"))
        if not fs:
            sys.exit("No sweep_pr_thresholds_*.csv found — run the sweep first.")
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        print(f"Loaded {len(fs)} model files: {[os.path.basename(f) for f in fs]}")
        summarize(df)
        return

    months = [1, 7] if args.quick else args.months
    out = args.out or f"sweep_pr_thresholds_{args.model}.csv"

    print(f"=== threshold sweep: {args.model} ===")
    print(f"years={args.years}  months={months}  combos={args.combos}")
    print("NOTE combo 0 is the control and must reproduce ACC Cascades Jan ~12.0 / Midwest Jul ~10.8")

    df = run(args.model, args.years, months, args.combos)
    if df.empty:
        sys.exit("No data produced — check the native pr/prc paths.")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")
    if len(args.combos) > 1:
        summarize(df)


if __name__ == "__main__":
    main()
