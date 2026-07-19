#!/usr/bin/env python
"""Stage-3 exit criteria: does the pipeline's own detection code reproduce the
MCS climatology the design was validated on?

Runs `dataset._detect_mcs_blobs_precip` (the real code path, not a re-implementation)
over ACC's NATIVE-grid pr/prc for a few sample years and reports MCS days/yr by
region, plus the Great Plains seasonality that is the whole point of the change.

Native grid on purpose: it isolates the DETECTION logic from the regrid, so this
can run before Run_PR_regrid.sh finishes. Once the regridded files exist, rerun
with --regridded to confirm the signal survived the remap.

    python verify_precip_mcs.py                # native grid, ACC, 1990/1995/2000
    python verify_precip_mcs.py --regridded    # after Run_PR_regrid.sh

Reference (the numbers the thresholds were chosen against):
    GreatPlains ~21.8   Midwest ~23.4   Southeast ~34.2   Cascades ~36.0 days/yr
    GreatPlains DJF ~0.2 vs JJA ~12.3   <-- PASS/FAIL for the whole project
"""
import sys, glob, os
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARDatasetImproved as D

BASE = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"
MODEL = os.environ.get("REGCM_MODEL", "ACC")
YEARS = ["1990", "1995", "2000"]
REGRIDDED = "--regridded" in sys.argv

# Thresholds under test — read from the real loader so this tracks the pipeline.
from utils import load_regcm_pr_thresholds
THR = load_regcm_pr_thresholds(MODEL, f"{BASE}/{MODEL}/global_stats")
SHIELD, CORE, CF = THR["PR_shield"], THR["PR_core"], THR["PR_convfrac"]

# Seasonal geometry params, mirroring dataset._get_seasonal_mcs_parameters
SEASONAL = {**{m: (4, 3) for m in (6, 7, 8)},          # summer: area, core_px
            **{m: (12, 8) for m in (12, 1, 2)},        # winter
            **{m: (6, 5) for m in (3, 4, 5, 9, 10, 11)}}  # transition

MIN_CELLS = 96   # 60,000 km^2 at ~25 km


class _Det:
    """Minimal shim exposing the real detection methods."""
    _convective_fraction = D.__dict__['_convective_fraction']
    _blobs_from_shield_core = D._blobs_from_shield_core
    _accept_system = D._accept_system
    _detect_mcs_blobs_precip = D._detect_mcs_blobs_precip


def main():
    det = _Det()
    print(f"\n=== Stage-3 verification: {MODEL} "
          f"({'regridded 0.25deg' if REGRIDDED else 'native Lambert'}) ===")
    print(f"thresholds: shield prc>={SHIELD} mm/h & convfrac>={CF}, core prc>={CORE} mm/h")
    print(f"area >= {MIN_CELLS} cells (~60,000 km2)\n")

    if REGRIDDED:
        f0 = f"{BASE}/{MODEL}/PR_6hourly_latlon/{MODEL}_pr_6hourly_{YEARS[0]}.nc"
        if not os.path.exists(f0):
            sys.exit(f"No regridded file yet: {f0}\nRun Run_PR_regrid.sh first.")
        g = xr.open_dataset(f0)
        lat2d, lon2d = np.meshgrid(g.latitude.values, g.longitude.values, indexing="ij")
    else:
        f0 = sorted(glob.glob(f"{BASE}/{MODEL}/pr/*_pr_6hourly.nc"))[0]
        g = xr.open_dataset(f0, decode_times=False)
        lat2d, lon2d = g.xlat.values, g.xlon.values

    lon360 = np.where(lon2d < 0, lon2d + 360, lon2d)
    regions = {
        "GreatPlains":  (lat2d >= 40) & (lat2d <= 45) & (lon360 >= 260) & (lon360 <= 268),
        "Midwest":      (lat2d >= 38) & (lat2d <= 43) & (lon360 >= 268) & (lon360 <= 274),
        "Southeast":    (lat2d >= 31) & (lat2d <= 36) & (lon360 >= 268) & (lon360 <= 277),
        "Cascades_PNW": (lat2d >= 42) & (lat2d <= 49) & (lon360 >= 235) & (lon360 <= 239),
    }

    totals = {k: [] for k in regions}
    gp_seasonal = {"DJF": [], "JJA": []}

    for yr in YEARS:
        days = {k: 0.0 for k in regions}
        djf = jja = 0.0
        for mon in range(1, 13):
            prc_f, pr_f = _load_month(yr, mon)
            if prc_f is None:
                continue
            min_area, min_core = SEASONAL[mon]
            for t in range(prc_f.shape[0]):
                blob, _, _ = det._detect_mcs_blobs_precip(
                    prc_f[t], pr_f[t],
                    shield_threshold=SHIELD, core_threshold=CORE, convfrac_threshold=CF,
                    min_core_pixels=min_core, min_blob_area=min_area,
                )
                kept = _filter_min_cells(blob > 0, MIN_CELLS)
                for k, msk in regions.items():
                    days[k] += kept[msk].mean() / 4.0
                if mon in (12, 1, 2):
                    djf += kept[regions["GreatPlains"]].mean() / 4.0
                if mon in (6, 7, 8):
                    jja += kept[regions["GreatPlains"]].mean() / 4.0
        for k in regions:
            totals[k].append(days[k])
        gp_seasonal["DJF"].append(djf)
        gp_seasonal["JJA"].append(jja)
        print(f"  {yr}: " + "  ".join(f"{k}={days[k]:5.1f}" for k in regions))

    print("\n-- mean MCS days/yr --")
    ref = {"GreatPlains": 21.8, "Midwest": 23.4, "Southeast": 34.2, "Cascades_PNW": 36.0}
    ok = True
    for k in regions:
        got = float(np.mean(totals[k]))
        # generous band: this is a climatology reproduction check, not a unit test
        good = abs(got - ref[k]) <= max(6.0, 0.35 * ref[k])
        ok &= good
        print(f"  {'OK ' if good else 'OFF'} {k:13s} {got:5.1f}   (reference {ref[k]})")

    djf, jja = float(np.mean(gp_seasonal["DJF"])), float(np.mean(gp_seasonal["JJA"]))
    print(f"\n-- Great Plains seasonality (the pass/fail for the whole project) --")
    print(f"  DJF = {djf:.1f} d/yr   JJA = {jja:.1f} d/yr   (reference DJF 0.2 / JJA 12.3)")
    summer_peaked = jja > 5 * max(djf, 0.05)
    print(f"  {'PASS' if summer_peaked else 'FAIL'}: summer-peaked "
          f"(JJA/DJF = {jja/max(djf,0.01):.0f}x)")
    if not summer_peaked:
        print("  !! This is the failure the BT branch had. Do not proceed to training.")

    print(f"\n=== {'PASSED' if (ok and summer_peaked) else 'CHECK THE OFF/FAIL ROWS ABOVE'} ===\n")
    return 0 if (ok and summer_peaked) else 1


def _load_month(yr, mon):
    """Return (prc, pr) in mm/h for one month, or (None, None)."""
    if REGRIDDED:
        f = f"{BASE}/{MODEL}/PR_6hourly_latlon/{MODEL}_pr_6hourly_{yr}.nc"
        if not os.path.exists(f):
            return None, None
        ds = xr.open_dataset(f)
        ds = ds.sel(time=ds.time.dt.month == mon)
        if ds.sizes.get("time", 0) == 0:
            return None, None
        return ds.prc.values.astype(np.float32), ds.pr.values.astype(np.float32)

    fp = glob.glob(f"{BASE}/{MODEL}/pr/*.{yr}{mon:02d}0100_pr_6hourly.nc")
    fc = glob.glob(f"{BASE}/{MODEL}/prc/*.{yr}{mon:02d}0100_prc_6hourly.nc")
    if not fp or not fc:
        return None, None
    # native files are kg m-2 s-1; the regrid script does this x3600 for the pipeline
    pr = xr.open_dataset(fp[0], decode_times=False).pr.values * 3600.0
    prc = xr.open_dataset(fc[0], decode_times=False).prc.values * 3600.0
    return prc.astype(np.float32), pr.astype(np.float32)


def _filter_min_cells(binary, min_cells):
    from scipy import ndimage
    lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(binary)
    sizes = ndimage.sum(np.ones_like(binary), lab, np.arange(1, n + 1))
    keep = np.arange(1, n + 1)[sizes >= min_cells]
    return np.isin(lab, keep)


if __name__ == "__main__":
    sys.exit(main())
