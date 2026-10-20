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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/"
CACHE_DIR       = "../Percentile_ARs_detection/cache/preprocessed_v3"

TENSOR_DIR      = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/tensor_cache_v2")

VOR850_PATH     = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                   "vor850_latlongrid/vor850_latlongrid/output/"
                   "vor850_75th_percentile_global.nc")
WINDSPEED_PATH  = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                   "wind_speed_850/output/windspeed850_75th_percentile_global.nc")
MSLP_CLIM_PATH  = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                   "data/global_stats/mslp_monthly_climatology.nc")
IBTRACS_PATH    = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/"
                   "StormType-Data/TCs/ibtracs/IBTrACS.ALL.v04r01.nc")

VORTICITY_DIR   = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                   "vor850_latlongrid/vor850_latlongrid/")
WIND_SPEED_FILE = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                   "wind_speed_850/ERA5_windspeed_850mb_6hourly_final.nc")
TEMP_500MB_DIR  = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/"
                   "GLOBAL/from-ncar-server/t/monthly_output/6_hourly")
WINDSHEAR_DIR   = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                   "wind_shear_mag_sqrtu_v/")
BRIGHT_TEMP_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                   "OLR/brightness_temp_new/")

TARGET_SIZE  = (192, 384)
START_YEAR   = 1980
END_YEAR     = 2024


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


def file_already_cached(nc_file, tensor_dir, split=False):
    """Check if the .npy tensor cache already exists for this file."""
    try:
        date_str, _, _ = _parse_date_str(nc_file)
    except ValueError:
        return False
    prefix = "split" if split else "tensors"
    npy_path  = os.path.join(tensor_dir, f"{prefix}_{date_str}.npy")
    meta_path = os.path.join(tensor_dir, f"{prefix}_{date_str}.meta.npy")
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
    rank, split=False,
):
    """
    Load one monthly file, run ARDatasetImproved precompute,
    save tensors to disk.

    split=False : legacy (N, 14, H, W) tensors — input and target entangled.
    split=True  : Phase 1 (N, 10, H, W) tensors — 6 RAW physical predictors
                  followed by the 4 heuristic label masks. Reuses an existing
                  legacy month for the masks when one is present, which makes
                  this seconds per month instead of ~23 minutes.

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

    prefix = "split" if split else "tensors"
    npy_path  = os.path.join(tensor_dir, f"{prefix}_{date_str}.npy")
    meta_path = os.path.join(tensor_dir, f"{prefix}_{date_str}.meta.npy")

    # ── Skip if already cached ────────────────────────────────────────────────
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        print(f"  [SKIP] {nc_file} already cached → {npy_path}")
        return date_str, -1, 0.0

    t0 = time.time()

    try:
        # ── Load auxiliary data for this month ────────────────────────────────
        aux = _load_aux_data(year, month, vorticity_files, wind_speed_full=None)
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
                time_values = xr.open_dataset(file_path)["time"].values
        else:
            file_path = os.path.join(DATA_DIR, nc_file)
            time_values = xr.open_dataset(file_path)["time"].values

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

        # ── Raw IVT for the split predictor stack ─────────────────────────────
        # The preprocessed cache's `ivt` was built with ivt_percentile=400, so
        # it is zeroed below that and has IoU 0.917 with the AR label — using it
        # as a model input would re-create the Phase 0 copy shortcut. Recompute
        # the true field from the source Qu/Qv, matching the cache's longitude
        # roll (-180..180, sorted) so it lands on the same grid.
        if split:
            src = xr.open_dataset(os.path.join(DATA_DIR, nc_file))
            src = src.assign_coords(
                longitude=(((src.longitude + 180) % 360) - 180)).sortby('longitude')
            ivt_raw = np.sqrt(src['Qu'].values ** 2 + src['Qv'].values ** 2
                              ).astype(np.float32)
            src.close()
            if ivt_raw.shape != ERA5_6hrly_ARs['ivt'].shape:
                raise ValueError(
                    f"raw IVT shape {ivt_raw.shape} != cached ivt shape "
                    f"{ERA5_6hrly_ARs['ivt'].shape} for {nc_file}")
            ERA5_6hrly_ARs['ivt_raw'] = xr.DataArray(
                ivt_raw, dims=["time", "latitude", "longitude"],
                coords=ERA5_6hrly_ARs['ivt'].coords)
            print(f"  [split] raw IVT from Qu/Qv: "
                  f"{100*(ivt_raw == 0).mean():.2f}% zeros "
                  f"(cached ivt is {100*(ERA5_6hrly_ARs['ivt'].values == 0).mean():.2f}%)")

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
            "BT_boundary":       241.0,
            "BT_core":           225.0,
            "Temp500":           270.0,
            "vor850_per_hurr":   1.45e-5,
            "windspeed_per_hurr": 8.0,
        }
        global_stats = {
            "IVT":        {"mean": 170.566,      "std": 11.6209},
            "IWV":        {"mean": 24.1535,      "std": 1.12097},
            "BT":         {"mean": 254.675,      "std": 0.648636},
            "MSLP":       {"mean": 101138.0,     "std": 15.6262},
            "vorticity":  {"mean": -4.09177e-10, "std": 4.02875e-05},
            "wind_speed": {"mean": 8.35723,      "std": 0.306227},
            "temp_500mb": {"mean": 258.428,      "std": 0.832617},
        }

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
            ibtracs_lookup=ibtracs_this_month,
            year=year,
            month=month,
            output_mode=("split" if split else "legacy"),
        )

        # ── Save tensors to the requested tensor_dir ──────────────────────────
        # legacy: (N, 14, H, W)   split: (N, 10, H, W) = 6 raw predictors + 4 labels
        # In split mode the dataset already saved to its own SPLIT_CACHE_DIR, so
        # only write again when tensor_dir points somewhere else — a redundant
        # 366 MB rewrite per month is real I/O on Lustre.
        os.makedirs(tensor_dir, exist_ok=True)
        already_saved = split and os.path.exists(npy_path)
        if not already_saved:
            np.save(npy_path,  dataset.precomputed_data)
            np.save(meta_path, time_values)            # time axis metadata

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
    parser.add_argument("--tensor_dir", type=str, default=None,
                        help="Output dir (defaults to the legacy or split cache "
                             "depending on --split)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if .npy cache already exists",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Build the Phase 1 split cache: (N, 10, H, W) = 6 RAW physical "
             "predictors + 4 label masks, instead of the legacy 14-channel "
             "tensors where inputs and targets were entangled. Reuses existing "
             "legacy months for the masks, so it runs in seconds per month.",
    )
    args = parser.parse_args()

    from dataset import ARDatasetImproved
    tensor_dir = args.tensor_dir or (
        ARDatasetImproved.SPLIT_CACHE_DIR if args.split else TENSOR_DIR)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TENSOR PRECOMPUTE JOB")
        print(f"{'='*60}")
        print(f"  World size : {world_size} ranks")
        print(f"  Year range : {args.start_year}–{args.end_year}")
        print(f"  Mode       : {'SPLIT (6 predictors + 4 labels)' if args.split else 'legacy (14 ch)'}")
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

    # Full wind speed dataset — sliced per-month inside _load_aux_data
    wind_speed_full = xr.open_dataset(WIND_SPEED_FILE).squeeze("level")
    wind_speed_full = wind_speed_full.assign_coords(
        longitude=(((wind_speed_full.longitude + 180) % 360) - 180)
    ).sortby("longitude")

    vorticity_files = sorted(glob.glob(
        os.path.join(VORTICITY_DIR, "ERA5_vor_6hourly_*.850mb_latlon.nc")
    ))

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
            return (args.start_year, 1) <= (y, m) <= (args.end_year, 12)
        except ValueError:
            return False

    all_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith("era5dara_") and f.endswith(".nc") and _in_range(f)
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
                   if not file_already_cached(f, tensor_dir, split=args.split)]
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
            split=args.split,
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