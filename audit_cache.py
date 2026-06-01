#!/usr/bin/env python
"""
audit_cache.py — Single-rank audit of preprocessed HDF5 caches and (optionally)
auxiliary NetCDFs (BT, temp_500mb, wind_shear). Surfaces files that fail to
read so they can be deleted and rebuilt before the next training run.

Why single-rank: the training failures may be either
  (a) persistent file corruption — this script will find them, or
  (b) live HDF5/Lustre concurrency races at training time — this script
      will NOT find those (everything will look OK), and the real culprit
      is multi-rank concurrent access patterns, not the files themselves.

So: if this script flags BAD files → delete + rebuild those months.
    If this script says all OK but training still fails → the issue is
    concurrent access; fix that path instead.

Examples:
    # Quick cache audit (default paths, sampled reads, 1980-2024)
    python audit_cache.py

    # Also probe BT / temp_500mb / wind_shear yearly/monthly NetCDFs
    python audit_cache.py --aux

    # Full read of every chunk (slow; use as a SLURM single-rank job)
    python audit_cache.py --deep

    # Restrict to a specific year range
    python audit_cache.py --year-range 1980-1995 --aux
"""

import argparse
import glob
import os
import time
from collections import Counter


# ── Defaults match main.py's hard-coded paths ────────────────────────────────
CACHE_DIR_DEFAULT     = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                         "Storms_precipitation_notebooks/Percentile_ARs_detection/"
                         "cache/preprocessed_v3")
BT_DIR_DEFAULT        = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
                         "OLR/brightness_temp_new")
TEMP_DIR_DEFAULT      = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ERA5/"
                         "GLOBAL/from-ncar-server/t/monthly_output/6_hourly")
WINDSHEAR_DIR_DEFAULT = ("/lustre/orion/cli138/proj-shared/z7a/Storms_data/"
                         "wind_shear_mag_sqrtu_v")
OUT_DEFAULT           = "cache_audit_results.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def probe_h5(path: str, deep: bool = False) -> tuple:
    """
    Open an HDF5 cache file and force chunk decompression on each dataset.
    Returns (ok, info_dict). info_dict contains size_gb, dataset_count, and
    a list of bad_datasets when chunk reads fail.
    """
    info = {"path": path, "datasets": [], "bad_datasets": []}
    if not os.path.exists(path):
        info["size_gb"]    = 0.0
        info["error_type"] = "FileNotFoundError"
        info["error"]      = "Cache file does not exist"
        return False, info
    info["size_gb"] = os.path.getsize(path) / 1024**3
    try:
        import h5py
    except ImportError:
        info["error_type"] = "ImportError"
        info["error"] = "h5py not installed in this Python env"
        return False, info

    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            info["dataset_count"] = len(keys)
            for k in keys:
                try:
                    ds = f[k]
                    info["datasets"].append(
                        {"name": k, "shape": tuple(ds.shape), "dtype": str(ds.dtype)}
                    )
                    # Force chunk decompression (this is what fails at training time).
                    if deep:
                        _ = ds[...]
                    else:
                        if ds.ndim <= 1 or ds.size < 100_000:
                            _ = ds[...]                       # tiny/aux array — read all
                        else:
                            n = ds.shape[0]
                            for i in (0, n // 2, n - 1):       # 3 time slices
                                _ = ds[i]
                except Exception as e:
                    info["bad_datasets"].append({
                        "name": k,
                        "error_type": type(e).__name__,
                        "error": str(e)[:300],
                    })
        return (len(info["bad_datasets"]) == 0), info
    except Exception as e:
        info["error_type"] = type(e).__name__
        info["error"] = str(e)[:300]
        return False, info


def probe_netcdf(path: str, varname: str = None) -> tuple:
    """Open a NetCDF file and force a read on the named variable."""
    info = {"path": path, "size_gb": os.path.getsize(path) / 1024**3}
    try:
        import xarray as xr
    except ImportError:
        info["error_type"] = "ImportError"
        info["error"] = "xarray not installed in this Python env"
        return False, info

    try:
        with xr.open_dataset(path) as ds:
            info["vars"] = list(ds.data_vars)
            info["dims"] = {k: int(v) for k, v in ds.dims.items()}
            if varname and varname in ds.data_vars:
                arr = ds[varname]
                if "time" in arr.dims and arr.sizes["time"] > 0:
                    _ = arr.isel(time=0).values     # force chunk read
                else:
                    _ = arr.values
        return True, info
    except Exception as e:
        info["error_type"] = type(e).__name__
        info["error"] = str(e)[:300]
        return False, info


# ─────────────────────────────────────────────────────────────────────────────
# Audit drivers
# ─────────────────────────────────────────────────────────────────────────────

def audit_cache(cache_dir: str, y0: int, y1: int, deep: bool,
                targeted_months: list = None) -> tuple:
    """
    Audit every era5_preprocessed_YYYYMM.h5 in the given dir.

    If `targeted_months` is given (list of YYYYMM stems), only those files
    are probed — useful for verifying that a re-processed subset is now OK
    without re-auditing all 540 files.
    """
    print(f"\n{'='*78}")
    print(f"AUDIT: HDF5 preprocessed caches in {cache_dir}")
    print(f"Mode : {'DEEP (full read)' if deep else 'sampled (3 time slices/dataset)'}")
    if targeted_months:
        print(f"Scope: {len(targeted_months)} targeted month(s)")
    else:
        print(f"Years: {y0}-{y1}")
    print('='*78)

    if targeted_months:
        targeted = set(targeted_months)
        cache_files = [os.path.join(cache_dir, f"era5_preprocessed_{m}.h5")
                       for m in sorted(targeted)]
        # Warn (but don't drop) for any targeted months whose file is missing —
        # those will be reported as BAD by probe_h5's missing-file path.
    else:
        cache_files = sorted(glob.glob(os.path.join(cache_dir, "era5_preprocessed_*.h5")))

        def _in_range(p):
            stem = os.path.basename(p).replace("era5_preprocessed_", "").replace(".h5", "")
            try:
                return y0 <= int(stem[:4]) <= y1
            except Exception:
                return False

        cache_files = [p for p in cache_files if _in_range(p)]

    n = len(cache_files)
    print(f"Found {n} cache files to probe.\n")

    bad, ok_list = [], []
    err_counter  = Counter()
    t0 = time.time()

    for i, fp in enumerate(cache_files, 1):
        ok, info = probe_h5(fp, deep=deep)
        stem = os.path.basename(fp).replace("era5_preprocessed_", "").replace(".h5", "")
        if ok:
            ok_list.append(stem)
            # Print only every 25th OK to keep output readable (540 files)
            if i % 25 == 0 or i == n:
                print(f"[{i:>3}/{n}] OK  (last) {stem}  "
                      f"({info['size_gb']:.2f} GB, {info['dataset_count']} datasets)")
        else:
            bad.append((stem, info))
            if "error_type" in info:
                err_counter[info["error_type"]] += 1
                err_summary = info["error"]
                err_type    = info["error_type"]
            elif info.get("bad_datasets"):
                bd = info["bad_datasets"][0]
                err_counter[bd["error_type"]] += 1
                err_summary = bd["error"]
                err_type    = bd["error_type"]
            else:
                err_summary, err_type = "(no detail)", "?"
            print(f"[{i:>3}/{n}] BAD {stem}  ({info.get('size_gb', 0):.2f} GB)  "
                  f"{err_type}: {err_summary[:140]}")

    elapsed = time.time() - t0
    print(f"\n{'─'*78}")
    print(f"CACHE SUMMARY  ({elapsed:.0f}s)")
    print('─'*78)
    print(f"  OK  : {len(ok_list):>4} / {n}")
    print(f"  BAD : {len(bad):>4} / {n}")
    if err_counter:
        print(f"  Error type counts:")
        for et, c in err_counter.most_common():
            print(f"    {et:30}  {c}")
    if bad:
        print(f"\n  BAD months (delete + rebuild from NetCDF):")
        for stem, _ in bad:
            print(f"    {stem}")
    return ok_list, bad


def audit_aux(label: str, ddir: str, var: str, pattern: str,
              y0: int, y1: int) -> tuple:
    """
    Probe yearly/monthly aux NetCDFs. Pattern uses {year} (and optionally {month}).
    Returns (n_total, missing, bad).
    """
    print(f"\n{'='*78}")
    print(f"AUDIT: {label} aux NetCDFs in {ddir}")
    print('='*78)

    if "{month" in pattern:
        files_to_probe = [os.path.join(ddir, pattern.format(year=y, month=m))
                          for y in range(y0, y1 + 1) for m in range(1, 13)]
    else:
        files_to_probe = [os.path.join(ddir, pattern.format(year=y))
                          for y in range(y0, y1 + 1)]

    n = len(files_to_probe)
    missing, bad, ok_count = [], [], 0

    for i, fp in enumerate(files_to_probe, 1):
        stem = os.path.basename(fp)
        if not os.path.exists(fp):
            missing.append(stem)
            print(f"[{i:>4}/{n}] MISSING  {stem}")
            continue
        ok, info = probe_netcdf(fp, varname=var)
        if ok:
            ok_count += 1
            if i % 50 == 0 or i == n:
                print(f"[{i:>4}/{n}] OK (last sampled) {stem}")
        else:
            bad.append((stem, info))
            print(f"[{i:>4}/{n}] BAD      {stem}  "
                  f"{info.get('error_type', '?')}: {str(info.get('error', ''))[:140]}")

    print(f"\n  {label} summary: OK={ok_count}  MISSING={len(missing)}  BAD={len(bad)}  TOTAL={n}")
    if missing:
        print(f"  These months silently use ZEROS during training "
              f"(corrupts {label}-dependent target masks):")
        for s in missing:
            print(f"    {s}")
    return n, missing, bad


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Single-rank audit of preprocessed HDF5 caches "
                    "and (optionally) aux NetCDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cache-dir",     default=CACHE_DIR_DEFAULT)
    p.add_argument("--bt-dir",        default=BT_DIR_DEFAULT)
    p.add_argument("--temp-dir",      default=TEMP_DIR_DEFAULT)
    p.add_argument("--windshear-dir", default=WINDSHEAR_DIR_DEFAULT)
    p.add_argument("--aux", action="store_true",
                   help="Also probe BT/temp_500mb/wind_shear NetCDFs")
    p.add_argument("--deep", action="store_true",
                   help="Full read of every dataset (slow; use as SLURM job)")
    p.add_argument("--year-range", default="1980-2024",
                   help="YYYY-YYYY range to probe (default: 1980-2024). "
                        "Ignored when --months or --months-from-file is used.")
    p.add_argument("--months", default=None,
                   help="Comma-separated YYYYMM stems to probe (e.g. "
                        "'198110,199203,201311'). Overrides --year-range.")
    p.add_argument("--months-from-file", default=None,
                   help="Path to a file with one YYYYMM stem per line "
                        "(e.g. cache_audit_results.txt.bad_months.txt). "
                        "Overrides --year-range.")
    p.add_argument("--out", default=OUT_DEFAULT,
                   help="Output report path (default: cache_audit_results.txt)")
    args = p.parse_args()

    # Resolve which months to probe: explicit list overrides year-range
    targeted_months = None
    if args.months_from_file:
        with open(args.months_from_file) as f:
            targeted_months = [ln.strip().split("\t")[0]
                               for ln in f if ln.strip() and not ln.startswith("#")]
    elif args.months:
        targeted_months = [s.strip() for s in args.months.split(",") if s.strip()]

    if targeted_months:
        y0, y1 = 1900, 2100   # placeholder; filter_targets applies instead
    else:
        try:
            y0, y1 = (int(s) for s in args.year_range.split("-"))
        except Exception:
            raise SystemExit(f"--year-range must be YYYY-YYYY, got {args.year_range!r}")

    ok_list, bad = audit_cache(args.cache_dir, y0, y1, args.deep,
                               targeted_months=targeted_months)

    aux_report = []
    if args.aux:
        aux_report.append(("BT",
                           *audit_aux("BT", args.bt_dir, "BT",
                                      "era5_bt_6hourly_{year}.nc", y0, y1)))
        aux_report.append(("temp_500mb",
                           *audit_aux("temp_500mb", args.temp_dir, "T",
                                      "ERA5_t_500mb_{year}{month:02d}.nc", y0, y1)))
        aux_report.append(("windshear",
                           *audit_aux("windshear", args.windshear_dir, "shear_mag",
                                      "6hourly_ERA5_windshear_mag_{year}{month:02d}.nc",
                                      y0, y1)))

    # ── Write report + bad-months skip list ────────────────────────────────
    with open(args.out, "w") as f:
        f.write("Cache audit report\n")
        f.write(f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cache dir  : {args.cache_dir}\n")
        f.write(f"Year range : {y0}-{y1}\n")
        f.write(f"Mode       : {'deep' if args.deep else 'sampled'}\n\n")
        f.write(f"Cache: OK={len(ok_list)}  BAD={len(bad)}\n\n")
        if bad:
            f.write("BAD cache months (one per line, tab-separated with error type):\n")
            for stem, info in bad:
                et = info.get("error_type") or (
                    info["bad_datasets"][0]["error_type"] if info.get("bad_datasets") else ""
                )
                f.write(f"{stem}\t{et}\n")
        for entry in aux_report:
            label, n, missing, bad_aux = entry
            f.write(f"\n{label}: OK={n - len(missing) - len(bad_aux)}  "
                    f"MISSING={len(missing)}  BAD={len(bad_aux)}  TOTAL={n}\n")
            for s in missing:
                f.write(f"  MISSING\t{s}\n")
            for s, info in bad_aux:
                f.write(f"  BAD    \t{s}\t{info.get('error_type','')}\n")

    # Plain-text bad-months list (one stem per line) — feed to a skip list later
    if bad:
        skip_path = args.out + ".bad_months.txt"
        with open(skip_path, "w") as g:
            for stem, _ in bad:
                g.write(stem + "\n")
        print(f"\nBad cache month stems written to: {skip_path}")

    print(f"Full report written to: {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
