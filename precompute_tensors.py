"""
precompute_tensors.py
─────────────────────────────────────────────────────────────────────────────
ONE-TIME script to precompute all 14-channel tensors for every monthly file
and save them to disk as .npy files.

After this runs ONCE, dataset.py loads tensors in ~5 seconds instead of
~23 minutes per file.

Usage:
srun -N4 -n32 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 precompute_tensors.py

Each rank processes its own subset of files (same round-robin sharding
as training). No GPU needed — this is pure CPU work.
"""

import os
import sys
import time
import numpy as np
import glob
import argparse
import calendar

# ── Distributed setup (needed to avoid all ranks processing all files) ────────
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    MPI_AVAILABLE = True
except Exception:
    MPI_AVAILABLE = False
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    comm = None

# ── RegCM model selection ──────────────────────────────────────────────────────
# One code copy serves all six RegCM-CMIP6 runs; pick via the REGCM_MODEL env var
# (export REGCM_MODEL=ACC before launching). Also drives cache_loader's filename.
MODEL      = os.environ.get("REGCM_MODEL", "ACC")
REGCM_BASE = "/lustre/orion/cli138/proj-shared/dr6/eventtype/Data/RegCM-CMIP6/Data-processed"
GS_DIR     = f"{REGCM_BASE}/{MODEL}/global_stats"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = f"{REGCM_BASE}/{MODEL}/WNet_6hourly_latlon/"
# Fresh, per-model HDF5 cache dir (kept EMPTY on purpose → forces the RegCM-aware
# NetCDF fallback in cache_loader; must NOT point at the shared ERA5 cache).
CACHE_DIR       = f"../Percentile_ARs_detection/cache/preprocessed_RegCM_{MODEL}"

# Precip-MCS tensors live in their OWN dir, separate from the BT-branch cache
# (tensor_cache_RegCM_<MODEL>), which stays intact as the A/B baseline.
# MUST stay in sync with dataset.ARDatasetImproved.TENSOR_CACHE_DIR.
TENSOR_DIR      = (f"/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_RegCM_PR_{MODEL}")

VOR850_PATH     = f"{GS_DIR}/{MODEL}_vor850_75th.nc"
WINDSPEED_PATH  = f"{GS_DIR}/{MODEL}_windspeed850_75th.nc"
MSLP_CLIM_PATH  = f"{GS_DIR}/{MODEL}_mslp_monthly_climatology.nc"
IBTRACS_PATH    = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
                   "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc")

# RegCM provides vorticity & wind_speed INLINE in the regridded file (read by
# cache_loader); no separate aux dirs, and no temp500/shear. BT IS available.
VORTICITY_DIR   = None
WIND_SPEED_FILE = None
TEMP_500MB_DIR  = None
WINDSHEAR_DIR   = None
BRIGHT_TEMP_DIR = f"{REGCM_BASE}/{MODEL}/BT_6hourly_latlon"
# Precipitation for the MCS branch: yearly files carrying `pr` and `prc` in mm/h,
# produced by Run_PR_regrid.sh (mirrors the BT dir layout).
PRECIP_DIR      = f"{REGCM_BASE}/{MODEL}/PR_6hourly_latlon"
IWV_CLIM_PATH   = f"{GS_DIR}/{MODEL}_iwv_monthly_climatology.nc"

TARGET_SIZE  = (192, 384)
START_YEAR   = 1980
END_YEAR     = 2014

# Per-model normalization stats + MCS BT thresholds (read once from .txt files).
from utils import (load_regcm_global_stats, load_regcm_bt_thresholds,
                   load_regcm_pr_thresholds)
_REGCM_STATS = load_regcm_global_stats(MODEL, GS_DIR)
_REGCM_BT    = load_regcm_bt_thresholds(MODEL, GS_DIR)   # legacy BT A/B path only
_REGCM_PR    = load_regcm_pr_thresholds(MODEL, GS_DIR)   # MCS shield/core precip rates


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rank_files(all_files, rank, world_size):
    """Same round-robin sharding as distributed_dataset.py"""
    return all_files[rank::world_size]


def _parse_date_str(nc_file):
    """
    Extract YYYYMM string robustly from filenames like 'era5dara_198001.nc'.
    Returns (date_str, year, month) or raises ValueError.
    """
    basename = os.path.basename(nc_file)
    date_str = basename.split("_")[1].split(".")[0]   # "198001"
    if len(date_str) != 6 or not date_str.isdigit():
        raise ValueError(f"Cannot parse date from filename: {nc_file}")
    return date_str, int(date_str[:4]), int(date_str[4:6])


def file_already_cached(nc_file, tensor_dir):
    """Check if .npy tensor cache already exists for this file."""
    try:
        date_str, _, _ = _parse_date_str(nc_file)
    except ValueError:
        return False
    npy_path  = os.path.join(tensor_dir, f"tensors_{date_str}.npy")
    meta_path = os.path.join(tensor_dir, f"tensors_{date_str}.meta.npy")
    return os.path.exists(npy_path) and os.path.exists(meta_path)


def _barrier():
    """Call MPI barrier if available, else no-op."""
    if MPI_AVAILABLE and comm is not None:
        comm.Barrier()


# ─────────────────────────────────────────────────────────────────────────────
# Load auxiliary data for one month  (mirrors _process_one_file in main.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_aux_data(year, month, vorticity_files, wind_speed_full):
    """
    RegCM: `vorticity` and `wind_speed` are read INLINE from the regridded main
    file inside cache_loader. `temp_500mb` / `windhshear_ds` do not exist → None.
    `BT_ds` IS available — yearly file `<MODEL>_bt_6hourly_{year}.nc`, var `bt`,
    sliced to this month (clear-sky NaN filled later in preprocess_data_improved).
    `PR_ds` (vars `pr`, `prc`, mm/h) drives the precipitation-based MCS branch —
    yearly file `<MODEL>_pr_6hourly_{year}.nc`, sliced the same way.
    """
    import xarray as xr
    from utils import open_wnet_dataset

    aux = {
        "temp_500mb":    None,
        "windhshear_ds": None,
        "vorticity":     None,   # cache_loader pulls it from the main file
        "wind_speed":    None,   # cache_loader pulls it from the main file
        "BT_ds":         None,
        "PR_ds":         None,
    }

    bt_file = os.path.join(BRIGHT_TEMP_DIR, f"{MODEL}_bt_6hourly_{year:04d}.nc")
    if os.path.exists(bt_file):
        try:
            bt_ds = xr.open_dataset(bt_file)
            if "valid_time" in bt_ds.dims:
                bt_ds = bt_ds.rename({"valid_time": "time"})
            last_day = calendar.monthrange(year, month)[1]
            aux["BT_ds"] = bt_ds.sel(
                time=slice(f"{year:04d}-{month:02d}-01",
                           f"{year:04d}-{month:02d}-{last_day:02d}")
            )
        except Exception as e:
            print(f"  [WARN] BT load failed: {e}")
            aux["BT_ds"] = None
    else:
        print(f"  [WARN] BT file missing: {bt_file}")

    # ── Precipitation (pr / prc, mm/h) — MCS source ───────────────────────────
    # open_wnet_dataset (not plain xr.open_dataset): BCC and NOR are noleap in
    # the precip files too, and cftime vs datetime64 would break the time
    # alignment against the main file in preprocess_data_improved.
    pr_file = os.path.join(PRECIP_DIR, f"{MODEL}_pr_6hourly_{year:04d}.nc")
    if os.path.exists(pr_file):
        try:
            pr_ds = open_wnet_dataset(pr_file)
            last_day = calendar.monthrange(year, month)[1]
            aux["PR_ds"] = pr_ds.sel(
                time=slice(f"{year:04d}-{month:02d}-01",
                           f"{year:04d}-{month:02d}-{last_day:02d}")
            )
            if aux["PR_ds"].sizes.get("time", 0) == 0:
                print(f"  [WARN] PR slice empty for {year:04d}-{month:02d}")
                aux["PR_ds"] = None
        except Exception as e:
            print(f"  [WARN] PR load failed: {e}")
            aux["PR_ds"] = None
    else:
        # Expected for 198001 only (RegCM SRF output starts 1980-02).
        print(f"  [WARN] PR file missing: {pr_file}")

    return aux


def _load_aux_data_ERA5_UNUSED(year, month, vorticity_files, wind_speed_full):
    """
    Load per-month auxiliary data: temp_500mb, BT, wind shear, vorticity.
    Returns a dict with keys matching cache_loader.load_data() kwarg names.
    Missing data is returned as None (dataset handles graceful fallback).
    """
    import xarray as xr
    from utils import find_matching_vorticity

    date_str_ym = f"{year:04d}{month:02d}"
    aux = {}

    # ── 500 mb temperature ────────────────────────────────────────────────────
    temp_file = os.path.join(TEMP_500MB_DIR, f"ERA5_t_500mb_{date_str_ym}.nc")
    if os.path.exists(temp_file):
        try:
            ds = xr.open_dataset(temp_file).sel(level="500")
            ds = ds.assign_coords(
                longitude=(((ds.longitude + 180) % 360) - 180)
            ).sortby("longitude")
            aux["temp_500mb"] = ds["T"]
        except Exception as e:
            print(f"  [WARN] temp_500mb load failed: {e}")
            aux["temp_500mb"] = None
    else:
        aux["temp_500mb"] = None

    # ── Brightness temperature ─────────────────────────────────────────────────
    # ERA5 BT files use either 'time' or 'valid_time' as the time dimension
    # depending on how they were downloaded. Normalise to 'time' before slicing.
    bt_file = os.path.join(BRIGHT_TEMP_DIR, f"era5_bt_6hourly_{year:04d}.nc")
    if os.path.exists(bt_file):
        try:
            bt_ds = xr.open_dataset(bt_file)

            # ── Normalise time dimension name ─────────────────────────────────
            time_dim = None
            for candidate in ("time", "valid_time"):
                if candidate in bt_ds.dims or candidate in bt_ds.coords:
                    time_dim = candidate
                    break

            if time_dim is None:
                raise ValueError(
                    f"No recognisable time dimension in BT file. "
                    f"Available dims: {list(bt_ds.dims)}"
                )

            if time_dim != "time":
                # Rename so downstream code always sees 'time'
                bt_ds = bt_ds.rename({time_dim: "time"})

            last_day = calendar.monthrange(year, month)[1]
            month_start = f"{year:04d}-{month:02d}-01"
            month_end   = f"{year:04d}-{month:02d}-{last_day:02d}"
            aux["BT_ds"] = bt_ds.sel(time=slice(month_start, month_end))

        except Exception as e:
            print(f"  [WARN] BT load failed: {e}")
            aux["BT_ds"] = None
    else:
        aux["BT_ds"] = None

    # ── Wind shear ────────────────────────────────────────────────────────────
    # Variable name varies across file versions: 'shear_mag', 'windshear', etc.
    # Try known names in order of preference.
    SHEAR_VAR_CANDIDATES = ("shear_mag", "windshear", "wind_shear",
                            "shear", "ws")
    shear_file = os.path.join(WINDSHEAR_DIR,
                              f"6hourly_ERA5_windshear_mag_{date_str_ym}.nc")
    if os.path.exists(shear_file):
        try:
            shear_ds = xr.open_dataset(shear_file)

            shear_var = None
            for candidate in SHEAR_VAR_CANDIDATES:
                if candidate in shear_ds.data_vars:
                    shear_var = candidate
                    break

            if shear_var is None:
                raise ValueError(
                    f"No recognised wind-shear variable in {shear_file}. "
                    f"Available vars: {list(shear_ds.data_vars)}"
                )

            aux["windhshear_ds"] = shear_ds[shear_var].squeeze()

        except Exception as e:
            print(f"  [WARN] wind shear load failed: {e}")
            aux["windhshear_ds"] = None
    else:
        aux["windhshear_ds"] = None

    # ── Vorticity 850 mb ───────────────────────────────────────────────────────
    import pandas as pd
    vorticity_file = find_matching_vorticity(
        pd.Timestamp(year=year, month=month, day=1), vorticity_files
    )
    if vorticity_file is not None:
        try:
            vds = xr.open_dataset(vorticity_file).squeeze("plev")
            vds = vds.assign_coords(
                longitude=(((vds.longitude + 180) % 360) - 180)
            ).sortby("longitude")
            aux["vorticity"] = vds["var138"]
        except Exception as e:
            print(f"  [WARN] vorticity load failed: {e}")
            aux["vorticity"] = None
    else:
        aux["vorticity"] = None

    aux["wind_speed"] = wind_speed_full   # full dataset — cache_loader slices by time

    return aux


# ─────────────────────────────────────────────────────────────────────────────
# Per-file precompute
# ─────────────────────────────────────────────────────────────────────────────

def precompute_one_file(
    nc_file, tensor_dir,
    vor850_per, windspeed_per, mslp_clim, ibtracs_lookup,
    vorticity_files, wind_speed_full,
    rank,
    iwv_clim=None,
):
    """
    Load one monthly file, run ARDatasetImproved precompute,
    save tensors to disk.

    Returns (date_str, n_samples, elapsed_seconds) or None on error.
    """
    import xarray as xr
    import pandas as pd
    from dataset import ARDatasetImproved
    from cache_loader import CacheAwareDataLoader

    try:
        date_str, year, month = _parse_date_str(nc_file)
    except ValueError as e:
        print(f"  [ERROR] Rank {rank}: {e}")
        return None

    npy_path  = os.path.join(tensor_dir, f"tensors_{date_str}.npy")
    meta_path = os.path.join(tensor_dir, f"tensors_{date_str}.meta.npy")

    # ── Skip if already cached ────────────────────────────────────────────────
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        print(f"  [SKIP] {nc_file} already cached → {npy_path}")
        return date_str, -1, 0.0

    t0 = time.time()

    try:
        # ── Load auxiliary data for this month ────────────────────────────────
        aux = _load_aux_data(year, month, vorticity_files, wind_speed_full=None)

        # ── CRITICAL-AUX GATE: precipitation drives the MCS branch ────────────
        # Without this, a missing PR file zero-fills → `prc >= threshold` is
        # false everywhere → all-zero MCS targets → the model silently learns
        # "MCS never happens". Skip the month loudly instead.
        # Expected to fire for 198001 only (RegCM SRF output starts 1980-02).
        if aux["PR_ds"] is None:
            print(f"  [SKIP] Rank {rank}: {nc_file} — no precipitation data "
                  f"({year:04d}-{month:02d}); MCS channels would be all-zero.")
            return None

        # ── Load from HDF5 cache / NetCDF fallback (same as main.py) ──────────
        cache_loader = CacheAwareDataLoader(cache_dir=CACHE_DIR)

        preprocessed_data = cache_loader.load_data(
            year=year,
            month=month,
            data_dir=DATA_DIR,
            use_cache=True,
            fallback_to_netcdf=True,
            # Pass auxiliary data exactly as main.py does
            vorticity=aux["vorticity"],
            wind_speed=aux["wind_speed"],
            temp_500mb=aux["temp_500mb"],
            BT_ds=aux["BT_ds"],
            PR_ds=aux["PR_ds"],
            windhshear_ds=aux["windhshear_ds"],
            ivt_percentile=400,
        )

        if preprocessed_data is None:
            print(f"  [ERROR] Rank {rank}: Could not load {nc_file}")
            return None

        # ── Reconstruct time values ───────────────────────────────────────────
        if "time" in preprocessed_data:
            time_values = preprocessed_data["time"]
            try:
                if np.issubdtype(time_values.dtype, np.integer):
                    time_values = time_values.astype("datetime64[s]")
                elif (time_values.dtype.type is np.bytes_
                      or time_values.dtype.char == "S"):
                    str_times = [
                        t.decode("utf-8") if isinstance(t, (bytes, np.bytes_))
                        else str(t)
                        for t in time_values
                    ]
                    time_values = np.array(str_times, dtype="datetime64[s]")
            except Exception:
                file_path = os.path.join(DATA_DIR, nc_file)
                from utils import open_wnet_dataset
                time_values = open_wnet_dataset(file_path)["time"].values
        else:
            file_path = os.path.join(DATA_DIR, nc_file)
            from utils import open_wnet_dataset
            time_values = open_wnet_dataset(file_path)["time"].values

        # ── Reconstruct lat / lon ─────────────────────────────────────────────
        if ("latitude" in preprocessed_data
                and "longitude" in preprocessed_data):
            lat_values = preprocessed_data["latitude"]
            lon_values = preprocessed_data["longitude"]
        else:
            _sample = next(
                (v for k, v in preprocessed_data.items()
                 if k != "time" and isinstance(v, np.ndarray) and v.ndim == 3),
                None,
            )
            if _sample is not None:
                _n_lat, _n_lon = _sample.shape[1], _sample.shape[2]
                lat_values = np.linspace(90, -90, _n_lat)
                lon_values = np.linspace(-180, 180 - 360 / _n_lon, _n_lon)
            else:
                lat_values = lon_values = None

        # ── Rebuild xarray Dataset ────────────────────────────────────────────
        ERA5_6hrly_ARs = xr.Dataset()
        _skip_keys = {"time", "latitude", "longitude"}
        for key, value in preprocessed_data.items():
            if (key not in _skip_keys
                    and isinstance(value, np.ndarray)
                    and value.ndim == 3):
                coords_dict = {"time": time_values}
                if lat_values is not None:
                    coords_dict["latitude"] = lat_values
                if lon_values is not None:
                    coords_dict["longitude"] = lon_values
                ERA5_6hrly_ARs[key] = xr.DataArray(
                    value,
                    dims=["time", "latitude", "longitude"],
                    coords=coords_dict,
                )

        # ── IBTrACS slice for this month ──────────────────────────────────────
        era5_6h_keys = [
            pd.Timestamp(t).to_pydatetime().replace(
                minute=0, second=0, microsecond=0
            )
            for t in time_values
        ]
        ibtracs_this_month = {dt: ibtracs_lookup.get(dt, [])
                              for dt in era5_6h_keys}

        active = [(dt, pts) for dt, pts in ibtracs_this_month.items() if pts]
        print(f"  IBTrACS {year}-{month:02d}: "
              f"{len(active)}/{len(era5_6h_keys)} timesteps have active storms")

        # ── Global percentiles / stats (identical to main.py) ─────────────────
        global_percentiles = {
            "vor850":            vor850_per,
            "windspeed":         windspeed_per,
            "BT_boundary":       _REGCM_BT["BT_boundary"],   # per-model (was ERA5 241.0)
            "BT_core":           _REGCM_BT["BT_core"],       # per-model (was ERA5 225.0)
            # MCS is precipitation-based; BT_* above only feed the legacy
            # mcs_source="bt" A/B path.
            "PR_shield":         _REGCM_PR["PR_shield"],     # mm/h, convective
            "PR_core":           _REGCM_PR["PR_core"],       # mm/h, convective
            "PR_convfrac":       _REGCM_PR["PR_convfrac"],   # prc/pr, dimensionless
            "Temp500":           270.0,
            "vor850_per_hurr":   1.45e-5,
            "windspeed_per_hurr": 8.0,
        }
        # RegCM per-model stats loaded from the CDO .txt files (Steps 1–3).
        global_stats = _REGCM_STATS

        # ── Build dataset — precompute=True runs the expensive part ───────────
        # Pass year/month so the dataset's own caching logic is consistent,
        # but we save manually below to tensor_dir (which may differ from
        # dataset.TENSOR_CACHE_DIR). Set precompute=True and let it run.
        dataset = ARDatasetImproved(
            ERA5_6hrly_ARs,
            ivt_percentile=400,
            vorticity_threshold=1.45e-5,
            target_size=TARGET_SIZE,
            global_percentiles=global_percentiles,
            global_stats=global_stats,
            precompute=True,
            mslp_climatology=mslp_clim,
            iwv_climatology=iwv_clim,            # MCS moisture anomaly gate
            mcs_moisture_mode="anomaly",
            ibtracs_lookup=ibtracs_this_month,
            year=year,
            month=month,
        )

        # ── Save tensors to the requested tensor_dir ──────────────────────────
        os.makedirs(tensor_dir, exist_ok=True)
        np.save(npy_path,  dataset.precomputed_data)   # (N, 14, H, W) float32 (MCS on)
        np.save(meta_path, time_values)                # time axis metadata

        elapsed  = time.time() - t0
        n_samples = len(dataset)
        size_mb  = dataset.precomputed_data.nbytes / 1024 ** 2

        print(f"  [Rank {rank}] ✓ Saved {nc_file} → {npy_path} "
              f"({n_samples} samples, {elapsed:.0f}s, {size_mb:.0f} MB)")

        return date_str, n_samples, elapsed

    except Exception as e:
        print(f"  [Rank {rank}] ✗ ERROR on {nc_file}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


    finally:
        import gc
        for _var in ['ERA5_6hrly_ARs', 'dataset', 'preprocessed_data', 'aux', 'cache_loader']:
            try:
                exec(f"{_var} = None")
            except Exception:
                pass
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute 14-channel tensors for all monthly ERA5 files."
    )
    parser.add_argument("--start_year", type=int, default=START_YEAR)
    parser.add_argument("--end_year",   type=int, default=END_YEAR)
    # Month bounds apply to the FIRST and LAST year respectively, so
    # --start_year 2000 --end_year 2000 --start_month 7 --end_month 7 is one file.
    # Used by the single-month smoke test; the full run leaves them at 1/12.
    parser.add_argument("--start_month", type=int, default=1, choices=range(1, 13),
                        metavar="1-12", help="First month of start_year (default 1)")
    parser.add_argument("--end_month",   type=int, default=12, choices=range(1, 13),
                        metavar="1-12", help="Last month of end_year (default 12)")
    parser.add_argument("--tensor_dir", type=str, default=TENSOR_DIR)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if .npy cache already exists",
    )
    args = parser.parse_args()

    tensor_dir = args.tensor_dir

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TENSOR PRECOMPUTE JOB")
        print(f"{'='*60}")
        print(f"  World size : {world_size} ranks")
        print(f"  Year range : {args.start_year}–{args.end_year}")
        print(f"  Output dir : {tensor_dir}")
        print(f"  Overwrite  : {args.overwrite}")
        print(f"{'='*60}\n")
        os.makedirs(tensor_dir, exist_ok=True)

    _barrier()   # wait for rank 0 to create output dir

    # ── Load shared data (all ranks load — these are small files) ─────────────
    if rank == 0:
        print("Loading shared data (percentiles, wind speed, IBTrACS)...")

    from utils import load_percentile_dataset, find_matching_vorticity
    import xarray as xr
    from ibtracs_loader import load_ibtracs_for_training

    vor850_per    = load_percentile_dataset(VOR850_PATH,    rank)
    windspeed_per = load_percentile_dataset(WINDSPEED_PATH, rank)
    mslp_clim     = load_percentile_dataset(MSLP_CLIM_PATH, rank)
    iwv_clim      = (load_percentile_dataset(IWV_CLIM_PATH, rank)
                     if os.path.exists(IWV_CLIM_PATH) else None)
    if rank == 0 and iwv_clim is None:
        print(f"⚠ IWV climatology missing at {IWV_CLIM_PATH} — MCS uses absolute-IWV gate")

    # RegCM: wind_speed and vorticity are read INLINE from the main regridded file
    # inside cache_loader, so there is no separate WIND_SPEED_FILE / VORTICITY_DIR.
    # _load_aux_data ignores both of these, so leave them empty (the ERA5 code that
    # opened them is dead here and crashed because the paths are None).
    wind_speed_full = None
    vorticity_files = []

    ibtracs_lookup = load_ibtracs_for_training(
        IBTRACS_PATH,
        start_year=1980,
        end_year=2024,
        wind_threshold_kt=0.0,
        lat_min=-50.0,
        lat_max=50.0,
    )

    if rank == 0:
        print(f"Shared data loaded. "
              f"IBTrACS: {len(ibtracs_lookup)} timesteps, "
              f"Vorticity files: {len(vorticity_files)}\n")

    # ── Collect all files in year range ───────────────────────────────────────
    def _in_range(fname):
        try:
            _, y, m = _parse_date_str(fname)
            return ((args.start_year, args.start_month) <= (y, m)
                    <= (args.end_year, args.end_month))
        except ValueError:
            return False

    all_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(f"{MODEL}dara_") and f.endswith(".nc") and _in_range(f)
    ])

    if rank == 0:
        print(f"Found {len(all_files)} files in range "
              f"{args.start_year}–{args.end_year}\n")
        if not all_files:
            print("No files found — exiting.")
            return

    # ── Shard files across ranks (round-robin) ────────────────────────────────
    my_files = get_rank_files(all_files, rank, world_size)

    # Optionally filter out already-cached files
    if not args.overwrite:
        pending = [f for f in my_files
                   if not file_already_cached(f, tensor_dir)]
        skipped_upfront = len(my_files) - len(pending)
        if skipped_upfront > 0:
            print(f"[Rank {rank}] Skipping {skipped_upfront} already-cached "
                  f"files, processing {len(pending)} remaining")
        my_files = pending

    print(f"[Rank {rank}] Processing {len(my_files)} files")

    if not my_files:
        print(f"[Rank {rank}] Nothing to do.")
        _barrier()
        return

    # ── Main loop ─────────────────────────────────────────────────────────────
    t_job_start = time.time()
    results = []

    for i, nc_file in enumerate(my_files):
        print(f"\n[Rank {rank}] File {i+1}/{len(my_files)}: {nc_file}")

        result = precompute_one_file(
            nc_file=nc_file,
            tensor_dir=tensor_dir,
            vor850_per=vor850_per,
            windspeed_per=windspeed_per,
            mslp_clim=mslp_clim,
            ibtracs_lookup=ibtracs_lookup,
            vorticity_files=vorticity_files,
            wind_speed_full=wind_speed_full,
            rank=rank,
            iwv_clim=iwv_clim,
        )

        if result is not None:
            results.append(result)

        # Progress estimate
        elapsed_total = time.time() - t_job_start
        files_done    = i + 1
        rate          = files_done / elapsed_total        # files/sec
        remaining     = ((len(my_files) - files_done) / rate
                         if rate > 0 else 0)
        print(f"  [Rank {rank}] Progress: {files_done}/{len(my_files)} files "
              f"| Elapsed: {elapsed_total/60:.1f}m "
              f"| Remaining: {remaining/60:.1f}m")

    # ── Per-rank summary ──────────────────────────────────────────────────────
    total_elapsed  = time.time() - t_job_start
    successful     = [r for r in results if r is not None and r[1] != -1]
    skipped_count  = len([r for r in results if r is not None and r[1] == -1])
    error_count    = len([r for r in results if r is None])

    print(f"\n[Rank {rank}] DONE: "
          f"{len(successful)} computed, "
          f"{skipped_count} skipped, "
          f"{error_count} errors | "
          f"Total: {total_elapsed/60:.1f} min")

    # ── Global barrier + rank-0 summary ──────────────────────────────────────
    _barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ALL RANKS COMPLETE")
        print(f"Tensors saved to: {tensor_dir}")
        print(f"{'='*60}")
        print(f"\nNext step: submit your training job!")
        print(f"Training will now load tensors in ~5 sec instead of ~23 min")


if __name__ == "__main__":
    main()