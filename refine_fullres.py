#!/usr/bin/env python3
"""
refine_fullres.py — give the WNet AR and MCS masks their 0.25 deg shape back.

Why: the model runs on the 192x384 (~0.94 deg) split cache and phase4_export_detections.py
nearest-neighbour upsamples its output to 721x1440, so in the v2 products every value change
sits on a 0.94 deg cell boundary (measured 100%, against 26-27% for the old full-resolution
products). Climatologies built from them show 1-degree squares. The old pipeline looked fine
because post_processing.py OR'ed full-resolution physics blobs into the output, which made the
product a copy of its input channel.

Rule (the model still decides WHERE a feature is; the 0.25 deg field only gives its SHAPE):
  1. grow each detected object by one model cell (+-GROW cells at 0.25 deg, default 4)
  2. inside the grown area keep the 0.25 deg cells that pass the class's physical criterion
       MCS: BT <= 241 K (cold-cloud shield, Feng/PyFLEXTRKR; the MCS label's shield threshold)
       AR : IVT > max(85th percentile of IVT at that cell for the 5-month season,
                       hybrid175 latitude floor) -- the intensity test of the shipped AR
                       label L6 (ar_label_lib.py), on its own native-grid percentiles
  3. keep only components of step 2 that overlap the ORIGINAL detection (8-connected, periodic
     in longitude), so growing can never add a feature the model did not detect.
A detection with no qualifying 0.25 deg cell at that time is dropped.

MCS_id (6-hourly MCS files) is NOT carried over; regenerate it on the refined files with
phase4_add_mcs_ids.py. Output layout and compression match the input. Resumable per month.

Usage
  python refine_fullres.py --kind mcs --src <.../MCS_WNet_final> --dst <dir> --years 2003 --months 7
  python refine_fullres.py --kind ar  --src <.../ARs_WNet_final> --dst <dir> --years 2003 --months 7
  (under SLURM the (year, month) list is strided over ranks)
"""
import os
import argparse

import numpy as np
import netCDF4 as nc
from scipy.ndimage import binary_dilation

import ar_label_lib as L

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
BTDIR = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/OLR/brightness_temp_hourly"
DATA_DIR = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data"
PCTL_NC = os.path.join(NB, "ivt_only", "ivt_pctl_monthly_5mo.nc")
KIND = {"mcs": dict(var="MCS_detection", suffix="MCS_detection"),
        "ar": dict(var="AR_detection", suffix="AR_detection")}


def refine_frame(m, field, keep_fn, grow):
    if not m.any():
        return m
    g = binary_dilation(m, structure=np.ones((2 * grow + 1, 2 * grow + 1), bool))
    ok = g & keep_fn(field)
    if not ok.any():
        return np.zeros_like(m)
    lab, _ = L.label_wrap(ok)
    keep = np.unique(lab[m & ok])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def to_dt(var):
    return nc.num2date(var[:], var.units, getattr(var, "calendar", "standard"),
                       only_use_cftime_datetimes=False, only_use_python_datetimes=True)


def roll_to(lon_src, lon_dst):
    """Shift s.t. np.roll(field, -s, axis=-1) maps a 0..360 grid onto lon_dst (-180..180)."""
    s = int(np.argmin(np.abs(((lon_src + 180) % 360) - 180 - lon_dst[0])))
    assert np.allclose(((np.roll(lon_src, -s) + 180) % 360) - 180, lon_dst), "longitude mismatch"
    return s


class MCSField:
    def __init__(self, year, lat, lon, thr):
        self.b = nc.Dataset(os.path.join(BTDIR, f"era5_bt_hourly_{year:04d}.nc"))
        assert np.allclose(self.b["latitude"][:], lat), "BT latitude mismatch"
        self.idx = {t: i for i, t in enumerate(to_dt(self.b["valid_time"]))}
        self.s = roll_to(self.b["longitude"][:], lon)
        self.keep = lambda bt: bt <= thr
        self.note = f"BT <= {thr:.0f} K (hourly 0.25 deg BT)"

    def get(self, t):
        j = self.idx.get(t)
        return None if j is None else np.roll(np.asarray(self.b["bt"][j]), -self.s, axis=1)

    def close(self):
        self.b.close()


class ARField:
    def __init__(self, year, month, lat, lon, pctl=85.0, floor="hybrid175"):
        d = nc.Dataset(os.path.join(DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc"))
        d.set_auto_mask(False)
        assert np.allclose(d["latitude"][:], lat), "IVT latitude mismatch"
        s = roll_to(d["longitude"][:], lon)
        self.idx = {t: i for i, t in enumerate(to_dt(d["time"]))}
        qu = np.nan_to_num(d["Qu"][:].astype(np.float32))
        qv = np.nan_to_num(d["Qv"][:].astype(np.float32))
        d.close()
        self.ivt = np.roll(np.hypot(qu, qv), -s, axis=2)
        p = nc.Dataset(PCTL_NC)
        k = int(np.argmin(np.abs(p["pctl"][:] - pctl)))
        thr = np.maximum(np.asarray(p["ivt_pctl"][k, month - 1], np.float32), L.floor_profile(lat, floor))
        s2 = roll_to(p["longitude"][:], lon)
        p.close()
        thr = np.roll(thr, -s2, axis=1)
        self.keep = lambda ivt: ivt > thr
        self.note = f"IVT > max(p{pctl:g} seasonal percentile, {floor} floor) (0.25 deg, AR label L6)"

    def get(self, t):
        j = self.idx.get(t)
        return None if j is None else self.ivt[j]

    def close(self):
        pass


def refine_month(kind, y, mo, src, dst, grow, thr):
    k = KIND[kind]
    fin = os.path.join(src, f"era5dara_{y:04d}{mo:02d}_{k['suffix']}.nc")
    fout = os.path.join(dst, os.path.basename(fin))
    if os.path.exists(fout):
        return "skip"
    a = nc.Dataset(fin)
    v = a[k["var"]]
    times = to_dt(a[v.dimensions[0]])
    lat, lon = a["latitude"][:], a["longitude"][:]
    F = MCSField(y, lat, lon, thr) if kind == "mcs" else ARField(y, mo, lat, lon)

    tmp = f"{fout}.tmp{os.getpid()}"
    o = nc.Dataset(tmp, "w", format="NETCDF4")
    for dn, dim in a.dimensions.items():
        o.createDimension(dn, None if dim.isunlimited() else len(dim))
    for name, var in a.variables.items():
        if name == k["var"] or name == "MCS_id":
            continue
        w = o.createVariable(name, var.dtype, var.dimensions)
        w.setncatts({kk: var.getncattr(kk) for kk in var.ncattrs()})
        w[:] = var[:]
    ch = v.chunking()
    w = o.createVariable(k["var"], "f4", v.dimensions, zlib=True, complevel=4, shuffle=True,
                         chunksizes=None if ch == "contiguous" else ch)
    w.setncatts({kk: v.getncattr(kk) for kk in v.ncattrs() if kk != "_FillValue"})
    w.setncatts(dict(refine=f"grow {grow} cells (0.25 deg), keep {F.note}, only components "
                            f"overlapping the model detection (refine_fullres.py)",
                     refine_src=fin))
    o.setncatts({kk: a.getncattr(kk) for kk in a.ncattrs()})

    T = len(times)
    step = ch[0] if ch != "contiguous" else T          # write whole time-chunks at once
    n_in = n_out = n_missing = 0
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        blk = np.asarray(v[t0:t1]) > 0.5
        out = np.zeros(blk.shape, np.float32)
        for i in range(t1 - t0):
            fld = F.get(times[t0 + i])
            if fld is None:
                n_missing += 1
                r = blk[i]
            else:
                r = refine_frame(blk[i], fld, F.keep, grow)
            out[i] = r
            n_in += int(blk[i].sum()); n_out += int(r.sum())
        w[t0:t1] = out
    o.close(); a.close(); F.close()
    os.replace(tmp, fout)
    return (f"px {n_in} -> {n_out} ({n_out / max(n_in, 1):.2f})"
            + (f", {n_missing} steps without field (left unrefined)" if n_missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(KIND), required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--grow", type=int, default=4)
    ap.add_argument("--thr", type=float, default=241.0, help="MCS BT threshold (K)")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    rank, nr = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    for y, m in [(y, m) for y in a.years for m in a.months][rank::nr]:
        print(f"[rank {rank}] {a.kind} {y}-{m:02d}: {refine_month(a.kind, y, m, a.src, a.dst, a.grow, a.thr)}",
              flush=True)


if __name__ == "__main__":
    main()
