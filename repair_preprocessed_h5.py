#!/usr/bin/env python3
"""
repair_preprocessed_h5.py — rebuild a preprocessed_v3/era5_preprocessed_YYYYMM.h5 whose mslp / ivt / ivt_grad
datasets are unreadable ("filter returned failure during read"; 2013-09, file rewritten 2026-05-29).

Readable datasets are copied as they are. The broken ones are recomputed from the source
Storms_data/data/era5dara_YYYYMM.nc with the recipes that reproduce a healthy month (checked on 2013-08):
  mslp     = MSLP, longitude rolled to -180..179.75                           (identical)
  ivt      = sqrt(Qu^2 + Qv^2), set to 0 where <= 400                         (identical)
  ivt_grad = |grad| of the UNthresholded IVT, np.gradient over (lat, lon)     (max diff 3e-5)
Same chunks, gzip level, attributes. Writes <out>, then the caller swaps it in.
  python repair_preprocessed_h5.py --tag 201309 --out <staging.h5>
"""
import argparse
import numpy as np
import h5py
import xarray as xr

H5 = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/cache/preprocessed_v3/era5_preprocessed_{}.h5"
SRC = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/data/era5dara_{}.nc"


def readable(d):
    try:
        for t in range(d.shape[0]):
            d[t]
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = xr.open_dataset(SRC.format(a.tag))
    s = s.assign_coords(longitude=(((s.longitude + 180) % 360) - 180)).sortby("longitude")
    with h5py.File(H5.format(a.tag), "r") as f, h5py.File(a.out, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        for k in f:
            d = f[k]
            if d.ndim < 3 or readable(d):
                data = d[...]
                how = "copied"
            elif k == "mslp":
                data = s["MSLP"].values.astype(np.float32); how = "recomputed from MSLP"
            elif k in ("ivt", "ivt_grad"):
                ivt = np.sqrt(s["Qu"].values ** 2 + s["Qv"].values ** 2).astype(np.float32)
                if k == "ivt":
                    data = np.where(ivt > 400.0, ivt, 0).astype(np.float32); how = "recomputed sqrt(Qu2+Qv2)>400"
                else:
                    gy, gx = np.gradient(ivt, axis=(1, 2))
                    data = np.hypot(gx, gy).astype(np.float32); how = "recomputed |grad IVT|"
            else:
                raise SystemExit(f"{k}: unreadable and no recipe")
            assert data.shape == d.shape, (k, data.shape, d.shape)
            kw = dict(compression=d.compression, compression_opts=d.compression_opts) if d.compression else {}
            o = g.create_dataset(k, data=data.astype(d.dtype), chunks=d.chunks, **kw)
            for ak, av in d.attrs.items():
                o.attrs[ak] = av
            print(f"  {k:<15} {d.shape} {how}", flush=True)
    with h5py.File(a.out, "r") as g:
        for k in g:
            assert readable(g[k]), k
    print("all datasets readable end to end")


if __name__ == "__main__":
    main()
