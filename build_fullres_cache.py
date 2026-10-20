#!/usr/bin/env python3
"""
build_fullres_cache.py — the WNet training cache at native ERA5 0.25 deg (721 x 1440) instead of
192 x 384, with the label definitions of the final shipped models.

One month -> <outdir>/split_YYYYMM.npy (T, 10, 721, 1440) float32 + split_YYYYMM.meta.npy, the same
channel contract as every split cache (SplitCacheDataset reads it unchanged), lon -180..179.75, lat
90..-90:

  0-5  predictors  IVT (from Qu/Qv), MSLP, vort850, wind850, BT, IWV  ARDatasetImproved._build_predictor_stack
                   on the precompute_tensors.py inputs (the path that built every cache's ch 0-5)
  6    AR  = L6    phase4_build_ar_label.build(.., ["L6"])           (v6/v7 ch 6)
  7    TC  = IBTrACS full lifecycle + pre-genesis depression          (v9 ch 7, the cache of the lead member
                   phase4_build_ibtracs_label.ibtracs_discs(lifecycle, td_lead 10000 h)   of the final TC union)
                   discs drawn on the 0.25 deg grid (the label is defined in km)
  8    ETC = cc    phase4_build_etc_label.build(.., ["cc"])          (v7 ch 8)
  9    MCS = legacy mcs_core   ARDatasetImproved._create_mcs_masks, absolute-IWV gate (no IWV
                   climatology, as precompute_tensors.py ran it)  (v2 -> v7 ch 9)

Every earlier cache made channels 0-6, 8, 9 at this resolution and then reduced them with
F.interpolate(bilinear, align_corners=False) to 192 x 384; here that last step is simply not done.
Each month is CHECKED against that: the new channels, reduced the same way, must reproduce v7
(ch 0-6, 8, 9) and v9 (ch 7; drawn directly on 192 x 384 there, so IoU not identity). The per-month
check goes to <outdir>/check/check_YYYYMM.json and the month is written only if it passes, unless
--no-check-gate.

One month per rank task, round-robin; finished months are skipped (write .tmp, then rename).
  srun -n 8 python build_fullres_cache.py --years 2003 2003 --months 7 --outdir <dir>
"""
import os
import sys
import json
import time
import argparse

import numpy as np

# precompute_tensors imports mpi4py at module level; under srun without an MPI launch that aborts the
# process (not an exception), so make the import fail cleanly and let it use its SLURM fallback.
sys.modules["mpi4py"] = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
REF = {"v7": os.path.join(NB, "tensor_cache_v7_arlit_etccc"), "v9": os.path.join(NB, "tensor_cache_v9_tclife_td")}
OUTDIR = os.path.join(NB, "tensor_cache_v11_fullres")
NATIVE = (721, 1440)
SMALL = (192, 384)
TC_TD_LEAD_H = 10000.0          # v9: Run_build_tclife_label.sh EXTRA="--lifecycle --td-lead-hours 10000"
# pass thresholds on the check, per channel: predictors must be identical; labels IoU at 0.5
PASS = {"pred_max_rel": 1e-5, "label_iou": 0.999, "tc_iou": 0.80}


def reduce(a):
    """(T, H, W) -> (T, 192, 384): the bilinear step every earlier cache applied."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(a[:, None], dtype=np.float32))
    return F.interpolate(t, size=SMALL, mode="bilinear", align_corners=False)[:, 0].numpy()


def dataset_month(year, month):
    """ARDatasetImproved over one native month, built from the precompute_tensors.py inputs, with no
    precompute: gives the predictor stack and the legacy MCS mask per step at 721 x 1440."""
    import xarray as xr
    from dataset import ARDatasetImproved
    from cache_loader import CacheAwareDataLoader
    from utils import load_percentile_dataset
    import precompute_tensors as PT
    import glob
    vfiles = sorted(glob.glob(os.path.join(PT.VORTICITY_DIR, "ERA5_vor_6hourly_*.850mb_latlon.nc")))
    aux = PT._load_aux_data(year, month, vfiles, wind_speed_full=None)
    d = CacheAwareDataLoader(cache_dir=os.path.join(HERE, PT.CACHE_DIR)).load_data(
        year=year, month=month, data_dir=PT.DATA_DIR, use_cache=True, fallback_to_netcdf=True,
        vorticity=aux["vorticity"], wind_speed=aux["wind_speed"], temp_500mb=aux["temp_500mb"],
        BT_ds=aux["BT_ds"], windhshear_ds=aux["windhshear_ds"], ivt_percentile=400)
    src = xr.open_dataset(os.path.join(PT.DATA_DIR, f"era5dara_{year:04d}{month:02d}.nc"))
    src = src.assign_coords(longitude=(((src.longitude + 180) % 360) - 180)).sortby("longitude")
    ivt_raw = np.sqrt(src["Qu"].values ** 2 + src["Qv"].values ** 2).astype(np.float32)
    times = src["time"].values
    src.close()
    lat = np.linspace(90, -90, NATIVE[0])
    lon = np.linspace(-180, 180 - 360 / NATIVE[1], NATIVE[1])
    ds = xr.Dataset()
    for k, v in d.items():
        if k not in ("time", "latitude", "longitude") and isinstance(v, np.ndarray) and v.ndim == 3:
            ds[k] = xr.DataArray(v, dims=["time", "latitude", "longitude"],
                                 coords={"time": times, "latitude": lat, "longitude": lon})
    ds["ivt_raw"] = xr.DataArray(ivt_raw, dims=["time", "latitude", "longitude"], coords=ds["ivt"].coords)
    gp = {"vor850": load_percentile_dataset(PT.VOR850_PATH), "windspeed": load_percentile_dataset(PT.WINDSPEED_PATH),
          "BT_boundary": 241.0, "BT_core": 225.0, "Temp500": 270.0,
          "vor850_per_hurr": 1.45e-5, "windspeed_per_hurr": 8.0}
    gs = {"IVT": {"mean": 170.566, "std": 11.6209}, "IWV": {"mean": 24.1535, "std": 1.12097},
          "BT": {"mean": 254.675, "std": 0.648636}, "MSLP": {"mean": 101138.0, "std": 15.6262},
          "vorticity": {"mean": -4.09177e-10, "std": 4.02875e-05},
          "wind_speed": {"mean": 8.35723, "std": 0.306227}, "temp_500mb": {"mean": 258.428, "std": 0.832617}}
    dset = ARDatasetImproved(ds, ivt_percentile=400, vorticity_threshold=1.45e-5, target_size=NATIVE,
                             global_percentiles=gp, global_stats=gs, precompute=False,
                             mslp_climatology=load_percentile_dataset(PT.MSLP_CLIM_PATH),
                             ibtracs_lookup={}, year=year, month=month, output_mode="split")
    return dset, np.asarray(times).astype("datetime64[h]")


_G = {}      # per-month state shared with forked pool workers (copy-on-write, nothing pickled)
NPROC = int(os.environ.get("FULLRES_NPROC", "0")) or None


def _pool_map(fn, n):
    """fn(i) for i in range(n) over a fork pool; results in step order."""
    from multiprocessing import get_context
    with get_context("fork").Pool(NPROC) as pool:
        return pool.map(fn, range(n), chunksize=1)


def _mcs_step(i):
    return _G["dset"]._create_mcs_masks(i, _G["m2"])["mcs_core"].astype(np.float32)


def mcs_label(dset):
    """ch 9: _compute_single_sample's mcs_core, per step, before its resize."""
    h, w = dset.ivt[0].shape
    lm = dset.mcs_lat_mask
    m2 = (np.broadcast_to(np.asarray(lm, bool)[:h, None], (h, w)) if np.ndim(lm) == 1 else lm).astype(np.float32)
    _G.update(dset=dset, m2=m2)
    out = np.stack(_pool_map(_mcs_step, dset.total_samples))
    _G.clear()
    return out


def _ar_step(i):
    import ar_label_lib as L
    g = _G
    _, r = L.detect(g["qu"][i], g["qv"][i], g["thr"], g["lat"], g["lon"], g["area"], params=dict(max_width_km=None))
    return r


def ar_label(year, month):
    """ch 6: phase4_build_ar_label.build(.., ['L6']) -- floor hybrid175, object must reach |lat| 23.5
    (the `reach` output of detect) -- step-parallel. Returns (T, 721, 1440) on lon -180..180, times."""
    import ar_label_lib as L
    import phase4_build_ar_label as AB
    floor, reach, width = AB.CAND_SPEC["L6"]
    assert reach and width is None
    qu, qv, lat, lon, times = AB.read_month(year, month)
    _G.update(qu=qu, qv=qv, lat=lat, lon=lon, thr=AB.thresholds(month, lat, {floor})[floor],
              area=L.cell_area(lat, lon))
    out = np.stack(_pool_map(_ar_step, qu.shape[0]))
    _G.clear()
    return np.roll(out, lon.size // 2, axis=2), times


def _etc_step(i):
    import phase4_build_etc_label as EB
    g = _G
    shim = EB._Shim(g["mslp"][i:i + 1], g["vort"][i:i + 1], g["wind"][i:i + 1], g["lats"], *g["thr"])
    return EB.cc_label(shim, g["lats"], g["lons"])[0]


def etc_label(year, month):
    """ch 8: phase4_build_etc_label.build(.., ['cc']) (cc_label works step by step), step-parallel."""
    import phase4_build_etc_label as EB
    mslp, vort, wind, lats, times, _ = EB.load_month(year, month)
    lons = np.linspace(-180.0, 180.0 - 360.0 / mslp.shape[2], mslp.shape[2])
    _G.update(mslp=mslp, vort=vort, wind=wind, lats=lats, lons=lons, thr=EB.thresholds())
    out = np.stack(_pool_map(_etc_step, mslp.shape[0]))
    _G.clear()
    return out, times


def tc_label(year, times):
    """ch 7: the v9 discs (radius ROCI > 2 x r34 > 300 km, clamped 200-500 km) on the 0.25 deg grid."""
    import phase4_build_ibtracs_label as TB
    discs, _, _ = TB.ibtracs_discs(year, 34.0, lifecycle=True, td_lead_hours=TC_TD_LEAD_H)
    glat = np.linspace(90.0, -90.0, NATIVE[0])
    glon = np.linspace(-180.0, 180.0 - 360.0 / NATIVE[1], NATIVE[1])
    lat2d = np.broadcast_to(glat[:, None], NATIVE)
    lon2d = np.broadcast_to(glon[None, :], NATIVE)
    coslat = np.cos(np.radians(lat2d))
    lab = np.zeros((times.size,) + NATIVE, np.float32)
    for i, t in enumerate(times):
        stamp = t.astype("datetime64[s]").astype(object).replace(minute=0, second=0, microsecond=0)
        for la, lo, rad in discs.get(stamp, []):
            r = float(np.clip(300.0 if not np.isfinite(rad) else rad, 200.0, 500.0))
            dy = (lat2d - la) * 111.0                               # same local metric as the builder
            dx = ((lon2d - lo + 180.0) % 360.0 - 180.0) * 111.0 * coslat
            lab[i][(dx * dx + dy * dy) <= r * r] = 1.0
    return lab


_T = [time.time()]


def tick(what):
    """stage timing to stdout, so a slow stage is visible while the month runs"""
    now = time.time()
    print(f"  [{os.environ.get('SLURM_PROCID', 0)}] {what}: {now - _T[0]:.0f} s", flush=True)
    _T[0] = now


def build(year, month):
    t0 = time.time()
    dset, times = dataset_month(year, month)
    T = dset.total_samples
    arr = np.zeros((T, 10) + NATIVE, np.float32)
    for i in range(T):
        arr[i, :6] = dset._build_predictor_stack(i)
    tick(f"{year}{month:02d} dataset + predictors")
    arr[:, 9] = mcs_label(dset)
    tick(f"{year}{month:02d} MCS label")
    del dset
    t1 = time.time()
    ar, t_ar = ar_label(year, month)
    if not (t_ar.astype("datetime64[h]") == times).all():
        raise SystemExit(f"{year}{month:02d}: AR source time axis differs")
    arr[:, 6] = ar
    del ar
    tick(f"{year}{month:02d} AR label")
    etc, t_etc = etc_label(year, month)
    got = np.array([np.datetime64(t.to_pydatetime(), "h") for t in t_etc])
    if not (got == times).all():
        raise SystemExit(f"{year}{month:02d}: ETC source time axis differs")
    arr[:, 8] = etc
    del etc
    tick(f"{year}{month:02d} ETC label")
    arr[:, 7] = tc_label(year, times)
    tick(f"{year}{month:02d} TC label")
    return arr, times, dict(t_dataset_s=round(t1 - t0), t_labels_s=round(time.time() - t1))


def check(tag, arr, times):
    out = {}
    for name, d in REF.items():
        meta = np.load(os.path.join(d, f"split_{tag}.meta.npy"), allow_pickle=True).astype("datetime64[h]")
        if meta.shape != times.shape or not (meta == times).all():
            raise SystemExit(f"{tag}: time axis differs from {name}")
    v7 = np.load(os.path.join(REF["v7"], f"split_{tag}.npy"), mmap_mode="r")
    v9 = np.load(os.path.join(REF["v9"], f"split_{tag}.npy"), mmap_mode="r")
    ok = True
    for c in range(10):
        small = reduce(arr[:, c])
        ref = np.asarray((v9 if c == 7 else v7)[:, c], np.float32)
        if c < 6:
            rel = float(np.abs(small - ref).max() / max(np.abs(ref).max(), 1e-30))
            out[f"ch{c}"] = dict(max_rel_diff=rel)
            ok &= rel <= PASS["pred_max_rel"]
        else:
            a, b = small > 0.5, ref > 0.5
            iou = float((a & b).sum() / max((a | b).sum(), 1))
            out[f"ch{c}"] = dict(iou=iou, rate_ref=float(b.mean()), rate_new_small=float(a.mean()),
                                 rate_native=float((arr[:, c] > 0.5).mean()))
            ok &= iou >= (PASS["tc_iou"] if c == 7 else PASS["label_iou"])
    out["pass"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[1980, 2024])
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--no-check-gate", action="store_true", help="write the month even if the check fails")
    a = ap.parse_args()
    rank, nr = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    os.makedirs(os.path.join(a.outdir, "check"), exist_ok=True)
    tags = [(y, m) for y in range(a.years[0], a.years[1] + 1) for m in a.months]
    for y, m in tags[rank::nr]:
        tag = f"{y:04d}{m:02d}"
        dst = os.path.join(a.outdir, f"split_{tag}.npy")
        if os.path.exists(dst):
            print(f"[{rank}] {tag}: done", flush=True)
            continue
        t0 = time.time()
        try:
            arr, times, tm = build(y, m)
        except Exception as e:          # one bad month (e.g. a corrupt preprocessed .h5) must not kill the srun step
            import traceback
            with open(os.path.join(a.outdir, "check", f"error_{tag}.txt"), "w") as f:
                f.write(traceback.format_exc())
            print(f"[{rank}] {tag}: ERROR {type(e).__name__}: {e}", flush=True)
            _G.clear()
            continue
        res = check(tag, arr, times)
        res.update(tm, t_total_s=round(time.time() - t0), shape=list(arr.shape))
        with open(os.path.join(a.outdir, "check", f"check_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
        line = " ".join(f"{k}:{v.get('iou', v.get('max_rel_diff')):.4g}" for k, v in res.items() if k.startswith("ch"))
        print(f"[{rank}] {tag}: {'PASS' if res['pass'] else 'FAIL'} {line}  ({res['t_total_s']} s)", flush=True)
        if not res["pass"] and not a.no_check_gate:
            continue
        import shutil                                   # the reference meta, byte for byte (same time axis, checked)
        shutil.copyfile(os.path.join(REF["v7"], f"split_{tag}.meta.npy"), os.path.join(a.outdir, f"split_{tag}.meta.npy"))
        tmp = f"{dst}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, dst)


if __name__ == "__main__":
    main()
