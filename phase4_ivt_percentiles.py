#!/usr/bin/env python3
"""
phase4_ivt_percentiles.py — per-grid-cell, per-calendar-month IVT percentiles for ERA5
1980-2024, the locally relative threshold of the literature-based AR label.

The current AR label is one global scalar (IVT > 400). Guan and Waliser (2015) and the
tARget family use the 85th percentile of IVT at EACH grid cell for EACH season (the five
months centred on the month in question), and retry objects that fail the geometry tests at
the 87.5th, 90th, 92.5th and 95th percentiles. This script builds those maps.

No CDO (not available on Frontier) and no sorting of 27,000 samples per cell: one pass over
the 540 monthly files accumulates a histogram per cell (2.5 kg/m/s bins, 0 to 1500, plus an
overflow bin), and the percentiles are read off the cumulative histograms with linear
interpolation inside the bin. The error is well below one bin width.

Three stages, run one after the other by Run_ivt_percentiles.sh:

  --stage hist      N ranks, N a multiple of 12. Rank r takes calendar month r % 12 + 1 and
                    year chunk r // 12, reads Qu and Qv, IVT = sqrt(Qu^2 + Qv^2), and writes
                    hist_mMM_cK.npy  (ny, nx, NB + 1) uint16.
  --stage merge     13 tasks (12 months + all months), any number of ranks. For month m the
                    histograms of months m-2 .. m+2 (wrapping December to January) and all
                    year chunks are summed; writes pctl_mMM.npy (5, ny, nx) float32 and
                    pctl_all.npy.
  --stage assemble  one rank: writes ivt_pctl_monthly_5mo.nc and checks the all-months 85th
                    percentile against the existing ivt_only/spatial_85th.nc (CDO timpctl,85).

Grid: ERA5 native, latitude 90 to -90, longitude 0 to 359.75 (no roll; consumers roll).
"""
import os
import sys
import glob
import argparse

import numpy as np

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
NB_DIR = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
OUT_DIR = os.path.join(NB_DIR, "ivt_only", "ivt_pctl_monthly")
CDO_P85 = os.path.join(NB_DIR, "ivt_only", "spatial_85th.nc")
Y0, Y1 = 1980, 2024
BIN_W, IVT_TOP = 2.5, 1500.0
NB = int(IVT_TOP / BIN_W)                  # 600 regular bins; index NB is the overflow bin
PCTLS = (85.0, 87.5, 90.0, 92.5, 95.0)
WINDOW = 2                                 # months either side: a 5-month season
ROWS = 30                                  # latitude rows per bincount block


def read_ivt(path):
    """IVT (nt, ny, nx) float32 from one monthly file. The whole variable is read in one call:
    the 2024-08..12 files are chunked deep in time and a per-step read is very slow."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        qu = ds.variables["Qu"][:].astype(np.float32)
        qv = ds.variables["Qv"][:].astype(np.float32)
        lat = ds.variables["latitude"][:].astype(np.float64)
        lon = ds.variables["longitude"][:].astype(np.float64)
    ivt = np.sqrt(qu * qu + qv * qv)
    del qu, qv
    return ivt, lat, lon


def add_hist(hist, ivt):
    """Accumulate ivt (nt, ny, nx) into hist (ny, nx, NB + 1). Non-finite values are skipped."""
    nt, ny, nx = ivt.shape
    nbin = NB + 2                                          # NB + 1 real bins and one for non-finite
    for r0 in range(0, ny, ROWS):
        blk = ivt[:, r0:r0 + ROWS, :]
        b = np.floor(blk / BIN_W)
        b = np.where(np.isfinite(b), np.clip(b, 0, NB), NB + 1).astype(np.int64)
        ncell = b.shape[1] * b.shape[2]
        b = b.reshape(nt, ncell) + np.arange(ncell, dtype=np.int64)[None, :] * nbin
        c = np.bincount(b.ravel(), minlength=ncell * nbin).reshape(ncell, nbin)[:, :NB + 1]
        hist[r0:r0 + ROWS] += c.reshape(blk.shape[1], nx, NB + 1).astype(hist.dtype)


def stage_hist(args, rank, world):
    if world % 12:
        sys.exit(f"--stage hist needs a multiple of 12 ranks, got {world}")
    nchunk = world // 12
    month, chunk = rank % 12 + 1, rank // 12
    years = np.array_split(np.arange(args.years[0], args.years[1] + 1), nchunk)[chunk]
    out = os.path.join(args.outdir, f"hist_m{month:02d}_c{chunk}.npy")
    if os.path.exists(out) and not args.overwrite:
        print(f"  rank {rank}: {os.path.basename(out)} exists", flush=True)
        return
    hist, n_t = None, 0
    for y in years:
        p = os.path.join(DATA_DIR, f"era5dara_{y:04d}{month:02d}.nc")
        if not os.path.exists(p):
            print(f"  rank {rank}: MISSING {p}", flush=True)
            continue
        ivt, lat, lon = read_ivt(p)
        if hist is None:
            hist = np.zeros(ivt.shape[1:] + (NB + 1,), np.uint16)
        add_hist(hist, ivt)
        n_t += ivt.shape[0]
        print(f"  rank {rank}: {y}-{month:02d} steps {ivt.shape[0]} max {np.nanmax(ivt):.0f}", flush=True)
        del ivt
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, hist)
    os.replace(tmp, out)
    np.savez(os.path.join(args.outdir, f"grid_m{month:02d}_c{chunk}.npz"), lat=lat, lon=lon, n_t=n_t,
             years=years)
    print(f"  rank {rank}: wrote {out} ({n_t} steps, years {years[0]}-{years[-1]})", flush=True)


def percentiles(hist):
    """(len(PCTLS), ny, nx) float32 from a summed histogram (ny, nx, NB + 1)."""
    ny, nx, _ = hist.shape
    out = np.zeros((len(PCTLS), ny, nx), np.float32)
    for r0 in range(0, ny, ROWS):
        h = hist[r0:r0 + ROWS].astype(np.float64)
        cum = np.cumsum(h, axis=-1)
        tot = cum[..., -1]
        for k, q in enumerate(PCTLS):
            target = tot * (q / 100.0)
            i = np.argmax(cum >= target[..., None], axis=-1)            # first bin reaching the target
            below = np.take_along_axis(cum, np.maximum(i - 1, 0)[..., None], -1)[..., 0]
            below = np.where(i > 0, below, 0.0)
            inbin = np.take_along_axis(h, i[..., None], -1)[..., 0]
            frac = np.where(inbin > 0, (target - below) / np.maximum(inbin, 1.0), 0.0)
            out[k, r0:r0 + ROWS] = ((i + np.clip(frac, 0.0, 1.0)) * BIN_W).astype(np.float32)
    return out


def stage_merge(args, rank, world):
    nchunk = len(glob.glob(os.path.join(args.outdir, "hist_m01_c*.npy")))
    if nchunk == 0:
        sys.exit("no histograms found; run --stage hist first")
    for task in list(range(1, 13)) + ["all"]:
        if (12 if task == "all" else task - 1) % world != rank:
            continue
        out = os.path.join(args.outdir, "pctl_all.npy" if task == "all" else f"pctl_m{task:02d}.npy")
        if os.path.exists(out) and not args.overwrite:
            print(f"  rank {rank}: {os.path.basename(out)} exists", flush=True)
            continue
        months = (range(1, 13) if task == "all"
                  else [(task - 1 + d) % 12 + 1 for d in range(-WINDOW, WINDOW + 1)])
        tot = None
        for m in months:
            for c in range(nchunk):
                h = np.load(os.path.join(args.outdir, f"hist_m{m:02d}_c{c}.npy"))
                tot = h.astype(np.uint32) if tot is None else tot + h
        p = percentiles(tot)
        np.save(out, p)
        print(f"  rank {rank}: {task} months {list(months)} n={int(tot[0, 0].sum())} "
              f"p85 global mean {p[0].mean():.1f} max {p[0].max():.1f}", flush=True)


def stage_assemble(args):
    from netCDF4 import Dataset
    g = np.load(sorted(glob.glob(os.path.join(args.outdir, "grid_m01_c*.npz")))[0])
    lat, lon = g["lat"], g["lon"]
    monthly = np.stack([np.load(os.path.join(args.outdir, f"pctl_m{m:02d}.npy")) for m in range(1, 13)], 1)
    allm = np.load(os.path.join(args.outdir, "pctl_all.npy"))
    out = os.path.join(os.path.dirname(args.outdir.rstrip("/")), "ivt_pctl_monthly_5mo.nc")
    tmp = out + f".tmp{os.getpid()}"
    with Dataset(tmp, "w") as ds:
        ds.createDimension("pctl", len(PCTLS))
        ds.createDimension("month", 12)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        ds.createVariable("pctl", "f4", ("pctl",))[:] = PCTLS
        ds.createVariable("month", "i4", ("month",))[:] = np.arange(1, 13)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        v = ds.createVariable("ivt_pctl", "f4", ("pctl", "month", "latitude", "longitude"), zlib=True,
                              complevel=2)
        v[:] = monthly
        v.units = "kg m-1 s-1"
        v.long_name = ("IVT percentile at each grid cell for the 5-month season centred on each "
                       "calendar month, ERA5 6-hourly")
        a = ds.createVariable("ivt_pctl_allmonths", "f4", ("pctl", "latitude", "longitude"), zlib=True,
                              complevel=2)
        a[:] = allm
        a.units = "kg m-1 s-1"
        ds.source = (f"phase4_ivt_percentiles.py; IVT = sqrt(Qu^2 + Qv^2) from era5dara_YYYYMM.nc, "
                     f"{args.years[0]}-{args.years[1]}; histogram bins {BIN_W} kg/m/s")
    os.replace(tmp, out)
    print(f"wrote {out}")
    w = np.cos(np.radians(lat))[:, None]
    for k, q in enumerate(PCTLS):
        zm = (monthly[k] * w).sum((1, 2)) / (w.sum() * lon.size)
        print(f"  p{q:<5} area-mean by month: " + " ".join(f"{x:6.1f}" for x in zm))

    if os.path.exists(CDO_P85):                        # the CDO per-cell 85th percentile, all months
        with Dataset(CDO_P85) as ds:
            ds.set_auto_mask(False)
            ref = np.squeeze(ds.variables["IVT"][:]).astype(np.float64)
            names = [n for n in ds.variables if n.lower().startswith("lat")]
            rlat = ds.variables[names[0]][:] if names else lat
        if rlat[0] < rlat[-1]:
            ref = ref[::-1]
        mine = allm[0].astype(np.float64)
        for name, r in (("as stored", ref), ("rolled 720", np.roll(ref, 720, axis=1))):
            d = mine - r
            print(f"  vs spatial_85th.nc ({name}): r = {np.corrcoef(mine.ravel(), r.ravel())[0, 1]:.5f}, "
                  f"mean diff {d.mean():+.2f}, mean |diff| {np.abs(d).mean():.2f}, "
                  f"p99 |diff| {np.percentile(np.abs(d), 99):.2f} kg/m/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("hist", "merge", "assemble"))
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--years", nargs=2, type=int, default=[Y0, Y1])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.outdir, exist_ok=True)
    if args.stage == "hist":
        stage_hist(args, rank, world)
    elif args.stage == "merge":
        stage_merge(args, rank, world)
    elif rank == 0:
        stage_assemble(args)


if __name__ == "__main__":
    sys.exit(main())
