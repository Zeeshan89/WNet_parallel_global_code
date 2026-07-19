#!/usr/bin/env python3
"""
compute_bt_thresholds.py
────────────────────────────────────────────────────────────────────────────────
Per-model MCS brightness-temperature thresholds for the RegCM WNet pipeline.

Problem: the MCS shield/core masks use fixed ABSOLUTE Tb thresholds (241 K / 225 K)
calibrated for ERA5 *satellite* Tb. The RegCM "diagnosed cloud-top temperature
proxy" runs ~10-25 K colder, so 241/225 K flags deep convection almost everywhere.

Fix: replace the literals with per-model Tb values that reproduce ERA5's cold-cloud
FREQUENCY over the RegCM domain:
  1. ERA5 reference: fraction of ERA5 Tb < 241 K (shield) and < 225 K (core),
     measured over the RegCM domain box.
  2. RegCM threshold: the Tb value at that same fraction of the RegCM BT field
     (clear-sky NaN filled with 290 K, exactly as the training pipeline does).

Writes  <MODEL>_BT_boundary.txt  and  <MODEL>_BT_core.txt  into each model's
global_stats/ dir (same convention as the mean/std stats). Loaded at runtime by
utils.load_regcm_bt_thresholds().

Run on a CPU node:  python compute_bt_thresholds.py
"""
import os
import numpy as np
import xarray as xr

BASE      = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"
ERA5_BT   = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_new"
MODELS    = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]
SAMPLE_YEARS = [1985, 1995, 2004]     # representative spread of the training period
TIME_STRIDE  = 4                       # every 4th 6-hourly step (~daily) — plenty for a percentile
CLEARSKY_FILL_K = 290.0                # must match preprocess_data_improved

# RegCM domain box (from regcm_latlon_grid.txt)
LAT_MIN, LAT_MAX = 18.0, 56.75
LON_MIN, LON_MAX = -138.75, -57.0      # in -180..180


def era5_reference_fractions():
    """Fraction of ERA5 Tb below 241 K and 225 K over the RegCM domain box."""
    lo360_min, lo360_max = LON_MIN % 360, LON_MAX % 360   # -138.75->221.25, -57->303
    n_b = d_b = n_c = d_c = 0
    for yr in SAMPLE_YEARS:
        f = os.path.join(ERA5_BT, f"era5_bt_6hourly_{yr}.nc")
        if not os.path.exists(f):
            print(f"  [ERA5] missing {f}, skipping"); continue
        ds = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        da = ds["bt"].isel({tdim: slice(None, None, TIME_STRIDE)})
        da = da.sel(latitude=slice(LAT_MAX, LAT_MIN),          # ERA5 lat is descending
                    longitude=slice(lo360_min, lo360_max))
        a = da.values.astype(np.float32)
        a = a[np.isfinite(a)]
        n_b += int((a < 241.0).sum()); n_c += int((a < 225.0).sum()); d_b += a.size; d_c += a.size
        print(f"  [ERA5 {yr}] domain px={a.size/1e6:.1f}M  <241={100*(a<241).mean():.2f}%  <225={100*(a<225).mean():.2f}%")
    f_boundary = n_b / max(d_b, 1)
    f_core     = n_c / max(d_c, 1)
    print(f"  ERA5 reference over RegCM domain: <241K={100*f_boundary:.2f}%  <225K={100*f_core:.2f}%")
    return f_boundary, f_core


def regcm_thresholds(model, f_boundary, f_core):
    """RegCM Tb values at the ERA5 cold-cloud fractions (clear-sky filled 290 K)."""
    vals = []
    for yr in SAMPLE_YEARS:
        f = os.path.join(BASE, model, "BT_6hourly_latlon", f"{model}_bt_6hourly_{yr}.nc")
        if not os.path.exists(f):
            print(f"  [{model}] missing {f}, skipping"); continue
        ds = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        a = ds["bt"].isel({tdim: slice(None, None, TIME_STRIDE)}).values.astype(np.float32)
        a = np.nan_to_num(a, nan=CLEARSKY_FILL_K)     # clear sky -> warm, exactly like the pipeline
        vals.append(a.ravel())
    a = np.concatenate(vals)
    bt_boundary = float(np.percentile(a, 100.0 * f_boundary))
    bt_core     = float(np.percentile(a, 100.0 * f_core))
    return bt_boundary, bt_core


def _era5_percell_fractions():
    """
    Per-CELL fraction of ERA5 Tb below 241 K / 225 K, on the RegCM grid.
    The ERA5 0.25° global grid subsets EXACTLY onto the RegCM 156x328 box, so we
    only flip latitude (ERA5 is descending) — no regridding needed.
    Returns (f_boundary(lat,lon), f_core(lat,lon)).
    """
    lo360_min, lo360_max = LON_MIN % 360, LON_MAX % 360
    nb = nc = nt = None
    for yr in SAMPLE_YEARS:
        f = os.path.join(ERA5_BT, f"era5_bt_6hourly_{yr}.nc")
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        da = ds["bt"].isel({tdim: slice(None, None, TIME_STRIDE)}).sel(
            latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(lo360_min, lo360_max))
        a = da.values[:, ::-1, :].astype(np.float32)      # flip lat -> ascending (RegCM order)
        b = (a < 241.0).sum(axis=0); c = (a < 225.0).sum(axis=0); t = a.shape[0]
        nb = b if nb is None else nb + b
        nc = c if nc is None else nc + c
        nt = t if nt is None else nt + t
    return nb / nt, nc / nt


def _regcm_percell_thresholds(model, f_boundary, f_core):
    """
    Per-CELL RegCM Tb value at each cell's ERA5 cold-cloud frequency.
    Vectorised: sort each cell's Tb time series once, then index at rank f*(T-1).
    """
    vals = []
    for yr in SAMPLE_YEARS:
        f = os.path.join(BASE, model, "BT_6hourly_latlon", f"{model}_bt_6hourly_{yr}.nc")
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        a = ds["bt"].isel({tdim: slice(None, None, TIME_STRIDE)}).values.astype(np.float32)
        vals.append(np.nan_to_num(a, nan=CLEARSKY_FILL_K))
    a = np.concatenate(vals, axis=0)
    a.sort(axis=0)                                   # in-place sort along time
    T = a.shape[0]
    def pick(frac):
        idx = np.clip(np.rint(frac * (T - 1)).astype(np.int32), 0, T - 1)
        return np.take_along_axis(a, idx[None, :, :], axis=0)[0]
    b, c = pick(f_boundary), pick(f_core)

    # ── Clamp strictly BELOW the clear-sky fill ──────────────────────────────
    # In dry/cloud-free cells the proxy is almost always clear (BT = NaN -> the
    # CLEARSKY_FILL), so the percentile lands inside the fill block and the raw
    # threshold comes back as exactly 290 K. Since the masks test `bt <= thr`,
    # a 290 K threshold would flag the ENTIRE clear sky as cold cloud (~28% of
    # cells). Clamping just below the fill makes clear sky untouchable
    # (290 <= 289.99 is False) while leaving every real threshold intact.
    #
    # NOTE: do NOT cap at the ERA5 literals (241/225 K). The proxy's cold bias is
    # latitude-dependent: at 40-45 N its cold-cloud population sits ~230-240 K, so
    # those cells legitimately need thresholds WARMER than 225 K to reproduce
    # ERA5's cold-cloud frequency. Capping at 225 K zeroed out the Great Plains.
    cap = CLEARSKY_FILL_K - 0.01
    b = np.minimum(b, cap)
    c = np.minimum(c, cap)
    return b, c


def _write_field(path, arr, ref_ds, name):
    da = xr.DataArray(arr.astype(np.float32),
                      dims=("latitude", "longitude"),
                      coords={"latitude": ref_ds["latitude"].values,
                              "longitude": ref_ds["longitude"].values},
                      name=name,
                      attrs={"units": "K",
                             "long_name": f"per-cell MCS {name} Tb threshold "
                                          f"(matched to ERA5 cold-cloud frequency)"})
    da.to_dataset(name=name).to_netcdf(path)


def main():
    print("== ERA5 reference cold-cloud frequency over the RegCM domain ==")
    f_boundary_s, f_core_s = era5_reference_fractions()

    print("\n== Per-CELL ERA5 frequencies (for the 2-D threshold fields) ==")
    fb2, fc2 = _era5_percell_fractions()
    print(f"  ERA5 per-cell frac<241K: min={100*fb2.min():.2f}% mean={100*fb2.mean():.2f}% max={100*fb2.max():.2f}%")
    print(f"  ERA5 per-cell frac<225K: min={100*fc2.min():.2f}% mean={100*fc2.mean():.2f}% max={100*fc2.max():.2f}%")

    print(f"\n{'MODEL':6s} {'scalar bnd/core':>18s}   {'per-cell bnd (min-max)':>26s} {'per-cell core (min-max)':>26s}")
    for m in MODELS:
        gs = os.path.join(BASE, m, "global_stats")
        if not os.path.isdir(gs):
            print(f"{m:6s} global_stats dir missing — skipping"); continue

        # legacy domain-wide scalars (kept as fallback / reference)
        bnd, core = regcm_thresholds(m, f_boundary_s, f_core_s)
        with open(os.path.join(gs, f"{m}_BT_boundary.txt"), "w") as fh: fh.write(f"{bnd:.4f}\n")
        with open(os.path.join(gs, f"{m}_BT_core.txt"), "w") as fh: fh.write(f"{core:.4f}\n")

        # NEW: per-cell 2-D threshold fields (removes the latitude skew)
        b2, c2 = _regcm_percell_thresholds(m, fb2, fc2)
        ref = xr.open_dataset(os.path.join(BASE, m, "BT_6hourly_latlon",
                                           f"{m}_bt_6hourly_{SAMPLE_YEARS[0]}.nc"))
        _write_field(os.path.join(gs, f"{m}_BT_boundary_2d.nc"), b2, ref, "BT_boundary")
        _write_field(os.path.join(gs, f"{m}_BT_core_2d.nc"),     c2, ref, "BT_core")

        print(f"{m:6s} {bnd:7.1f}/{core:.1f} K   "
              f"{b2.min():9.1f}-{b2.max():.1f} K {c2.min():14.1f}-{c2.max():.1f} K"
              f"   -> {m}_BT_{{boundary,core}}_2d.nc")


if __name__ == "__main__":
    main()
