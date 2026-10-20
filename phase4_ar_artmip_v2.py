#!/usr/bin/env python3
"""
phase4_ar_artmip_v2.py — put the WNet AR product in the context of established AR
detection methods, not of one benchmark alone.

There is no observed truth for atmospheric rivers: ARTMIP showed that established
detection methods differ widely, and MARS is itself a detection method (the tARget v3
algorithm applied to ERA5), not a benchmark. Scoring WNet against MARS alone
(phase4_ar_vs_mars_v2.py) therefore cannot say whether an IoU of 0.3 is poor or ordinary.
Every method is treated the same way here. This pass accumulates, on one grid and at the
same 6-hourly times, six global AR masks for 1980-2019:

  MARS       tARget v3 on ERA5 (variable MARS in era5dara_YYYYMM.nc)
  WNet       Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final
  GW         Guan and Waliser v2      ARTMIP Tier 2, ERA5, hourly
  Mundhenk   Mundhenk et al. v3       "
  TempestLR  TempestExtremes (Laplacian of IVT)   "
  Reid500    Reid et al., 500 kg/m/s  "

so that every pair of methods can be compared afterwards, anywhere, from per-pixel maps.

Common grid: ERA5 native, latitude 90 to -90, longitude 0 to 359.75 (the MARS grid).
ARTMIP files come in three orientations; each is detected from its coordinates. TempestLR
carries NaN coordinates, so it is taken to be on the native grid, which was checked
against Guan-Waliser (r = 0.68 for the native orientation, 0.10 to 0.35 for the others).
ARTMIP times are matched to the MARS times by their actual timestamps.

One rank per half-year (80 tasks for 1980-2019), so the pass fits a 15-minute job. Each
writes evaluations/ar_artmip_eval_v2/part_YYYY_H.npz (H = 0 for Jan-Jun, 1 for Jul-Dec) with
  freq     (4 seasons, 6 methods, ny, nx)  number of AR time steps
  both     (15 pairs, ny, nx)              steps on which both methods of a pair have an AR
  ardays   (6, ny, nx)                     days with an AR at any of the four daily times
  n_t      (4,)                            time steps per season
  series   (12, 6, regions)                monthly area-weighted AR frequency by region
  voteA    (6, 6, ny) and (6, 6, regions)  cells where method i has an AR and exactly k of
                                           the OTHER five methods have one too
  voteB    same shapes                     cells where exactly k of the other five have an
                                           AR, whatever method i says
voteA and voteB give, with no method called the truth, the share of a method's AR area that
other methods confirm and the share of the area others agree on that the method captures.

Usage
  srun -n 80 python3 phase4_ar_artmip_v2.py
"""
import os
import sys
import glob
import argparse
from itertools import combinations

import numpy as np

DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PRED_DIR = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
            "Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final")
ARTMIP = "/lustre/orion/proj-shared/cli138/dr6/eventtype/Data/ARTMIP"
ARTMIP_PAT = {"GW": "ERA5.ar_tag.GuanWaliser_v2.1hr.{y}0101-{y}1231.nc",
              "Mundhenk": "ERA5.ar_tag.Mundhenk_v3.1hr.{y}0101-{y}1231.nc",
              "TempestLR": "ERA5.ar_tag.TempestLR.1hr.{y}.nc",
              "Reid500": "ERA5.ar_tag.Reid500.1hr.{y}0101-{y}1231.nc"}
METHODS = ("MARS", "WNet", "GW", "Mundhenk", "TempestLR", "Reid500")
PAIRS = tuple(combinations(range(len(METHODS)), 2))
Y0, Y1 = 1980, 2019
SEASON_OF = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
# name, lat0, lat1, lon0, lon1 (degrees east, -180..180); the six boxes of plot_ar_regional.py
REGIONS = (("G60", -60, 60, -180, 180), ("NHMID", 20, 60, -180, 180), ("SHMID", -60, -20, -180, 180),
           ("TROP", -20, 20, -180, 180),
           ("NEPAC", 25, 60, -180, -120), ("NATL", 30, 65, -65, -5), ("SEPAC", -60, -25, -130, -70),
           ("WNA", 25, 60, -125, -105), ("WEUR", 30, 65, -10, 10), ("SSA", -60, -25, -75, -55))


class Artmip:
    """One yearly ARTMIP file, read onto the common grid at requested times."""

    def __init__(self, name, year):
        from netCDF4 import Dataset, num2date
        self.name = name
        p = os.path.join(ARTMIP, ARTMIP_PAT[name].format(y=year))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        self.ds = Dataset(p)
        self.var = self.ds.variables["ar_binary_tag"]
        tv = self.ds.variables["time"]
        t = num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        self.t = np.array([np.datetime64(x, "h") for x in t])
        lat = np.ma.filled(self.ds.variables["lat"][:].astype(float), np.nan)
        lon = np.ma.filled(self.ds.variables["lon"][:].astype(float), np.nan)
        # NaN coordinates (TempestLR): native grid, verified against Guan-Waliser
        self.flip = bool(np.isfinite(lat).all() and lat[0] < lat[-1])
        self.roll = bool(np.isfinite(lon).all() and lon.min() < -1.0)

    def at(self, times):
        """Boolean (n, ny, nx) on the common grid; rows of a missing time are all False,
        and `ok` says which times were found."""
        idx = np.searchsorted(self.t, times)
        idx = np.clip(idx, 0, self.t.size - 1)
        ok = self.t[idx] == times
        out = np.zeros((times.size,) + self.var.shape[1:], bool)
        if ok.any():
            lo, hi = int(idx[ok].min()), int(idx[ok].max()) + 1
            step = 6 if np.all(np.diff(idx[ok]) == 6) else 1
            blk = np.ma.filled(self.var[lo:hi:step], 0) > 0          # strided: only the needed steps
            out[ok] = blk if step == 6 else blk[idx[ok] - lo]
        if self.flip:
            out = out[:, ::-1, :]
        if self.roll:
            out = np.roll(out, out.shape[2] // 2, axis=2)
        return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="evaluations/ar_artmip_eval_v2")
    ap.add_argument("--years", nargs=2, type=int, default=[Y0, Y1])
    ap.add_argument("--year-list", nargs="+", type=int, help="explicit years instead of a range")
    ap.add_argument("--pred-dir", default=PRED_DIR,
                    help="directory of era5dara_YYYYMM_AR_detection.nc taken as 'WNet': the exported "
                         "product, or a label candidate of phase4_build_ar_label.py")
    args = ap.parse_args()
    import xarray as xr

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(args.outdir, exist_ok=True)
    years = args.year_list or range(args.years[0], args.years[1] + 1)
    tasks = [(y, h) for y in years for h in (0, 1)][rank::world]

    for y, half in tasks:
        out = os.path.join(args.outdir, f"part_{y}_{half}.npz")
        if os.path.exists(out):
            print(f"  rank {rank}: {y}/{half} done", flush=True)
            continue
        art = {k: Artmip(k, y) for k in ARTMIP_PAT}
        print(f"  rank {rank}: {y} " + ", ".join(f"{k} n={a.t.size} flip={a.flip} roll={a.roll}"
                                                  for k, a in art.items()), flush=True)
        freq = both = ardays = series = None
        n_t, dropped = np.zeros(4, np.int64), 0
        for m in range(1 + 6 * half, 7 + 6 * half):
            td = xr.open_dataset(os.path.join(DATA_DIR, f"era5dara_{y:04d}{m:02d}.nc"))
            times = td.time.values.astype("datetime64[h]")
            lat, lon = td.latitude.values.astype(float), td.longitude.values.astype(float)
            masks = [td["MARS"].values > 0]
            td.close()
            pd_ = xr.open_dataset(os.path.join(args.pred_dir, f"era5dara_{y:04d}{m:02d}_AR_detection.nc"))
            da = pd_["AR_detection"].assign_coords(
                longitude=(pd_["AR_detection"].longitude % 360)).sortby("longitude")
            masks.append(da.values > 0.5)
            pd_.close()
            n = min(masks[0].shape[0], masks[1].shape[0])
            times, masks = times[:n], [a[:n] for a in masks]
            keep = np.ones(n, bool)
            for k in ARTMIP_PAT:
                a, ok = art[k].at(times)
                masks.append(a)
                keep &= ok
            if not keep.all():                     # a time missing in any catalogue is dropped for all
                dropped += int((~keep).sum())
                times, masks = times[keep], [a[keep] for a in masks]
            if freq is None:
                shp = masks[0].shape[1:]
                freq = np.zeros((4, len(METHODS)) + shp, np.uint16)
                both = np.zeros((len(PAIRS),) + shp, np.uint16)
                ardays = np.zeros((len(METHODS),) + shp, np.uint16)
                w = np.cos(np.radians(lat))[:, None] * np.ones((1, lon.size))
                lon180 = ((lon + 180) % 360) - 180
                reg = []
                for _, la0, la1, lo0, lo1 in REGIONS:
                    k2 = ((lat[:, None] >= la0) & (lat[:, None] <= la1)
                          & (lon180[None, :] >= lo0) & (lon180[None, :] <= lo1))
                    reg.append(w * k2 / (w * k2).sum())
                series = np.zeros((12, len(METHODS), len(REGIONS)))
                nm = len(METHODS)
                regb = [r > 0 for r in reg]
                voteA_lat = np.zeros((nm, nm, shp[0]), np.int64)
                voteB_lat = np.zeros((nm, nm, shp[0]), np.int64)
                voteA_reg = np.zeros((nm, nm, len(REGIONS)), np.int64)
                voteB_reg = np.zeros((nm, nm, len(REGIONS)), np.int64)
            s = SEASON_OF[m]
            n_t[s] += times.size
            day = times.astype("datetime64[D]")
            _, first = np.unique(day, return_index=True)
            for i, a in enumerate(masks):
                f = a.sum(0, dtype=np.uint16)
                freq[s, i] += f
                ardays[i] += np.logical_or.reduceat(a, first, axis=0).sum(0, dtype=np.uint16)
                fr = f.astype(np.float64) / max(times.size, 1)
                series[m - 1, i] = [(fr * r).sum() for r in reg]
            for p, (i, j) in enumerate(PAIRS):
                both[p] += np.logical_and(masks[i], masks[j]).sum(0, dtype=np.uint16)
            votes = np.zeros(masks[0].shape, np.uint8)
            for a in masks:
                votes += a
            for i, a in enumerate(masks):
                others = votes - a                                   # how many of the other five
                for k in range(nm):
                    eq = others == k
                    mb = eq.sum(0, dtype=np.int64)
                    ma = np.logical_and(eq, a).sum(0, dtype=np.int64)
                    voteB_lat[i, k] += mb.sum(1)
                    voteA_lat[i, k] += ma.sum(1)
                    voteB_reg[i, k] += [int(mb[r].sum()) for r in regb]
                    voteA_reg[i, k] += [int(ma[r].sum()) for r in regb]
            del masks, votes
        np.savez_compressed(out, freq=freq, both=both, ardays=ardays, n_t=n_t, series=series,
                            voteA_lat=voteA_lat, voteB_lat=voteB_lat, voteA_reg=voteA_reg,
                            voteB_reg=voteB_reg,
                            lat=lat, lon=lon, year=y, half=half, dropped=dropped)
        print(f"  rank {rank}: {y}/{half} wrote {out} (steps {int(n_t.sum())}, dropped {dropped})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
