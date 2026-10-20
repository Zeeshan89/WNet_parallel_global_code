#!/usr/bin/env python3
"""
verify_ar_v2.py — acceptance tests for the rebuilt v2 AR cache.

The rebuild added accumulators to phase4_ar_vs_mars_v2.py. The point of these
checks is that nothing about the original comparison moved: the contingency
counts must reproduce the run of 2026-09-05 exactly, digit for digit. Anything
else means the refactor changed the answer.

The rest of the checks are internal consistency (the season and period splits
must both partition the record) and one genuinely independent cross-check: the
python day-binning is compared against the CDO-built MARS climatology that was
produced years earlier by a completely different toolchain. If
pix_targ_days/45 matches MARS_AvgARDaysPerYear.nc, then the 6-hourly to daily
reduction, the longitude remap and the target threshold are all correct
simultaneously.

Usage
  python verify_ar_v2.py [--cache-dir .]
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import PERIODS, SEASONS, cos_w, load_parts, map_metrics, metrics  # noqa: E402

MARS_AVG = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/"
            "Analysis_MARS/MARS_AvgARDaysPerYear.nc")

# measured from the 2026-09-05 run; the rebuild must reproduce them exactly
EXPECT_TOTALS = dict(tp=1951915874, fp=1994705213, fn=2542419290)
EXPECT_SCALAR = {"oop": [1094348816, 1101354958, 1422902534], "primary":
                 [857567058, 893350255, 1119516756]}
EXPECT_NT = {"oop": 37988, "primary": 27760}
EXPECT_NT_TOTAL, EXPECT_NDAYS = 65748, 16437
EXPECT_WEIGHTED = dict(IoU=0.3072, F1=0.4700, POD=0.4992, FAR=0.5559, BIAS=1.1242)
EXPECT_MAX_PRED_DAYS = 236.29
EXPECT_MAX_TARG_DAYS = 116.73

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    C = load_parts(args.cache_dir)
    A, lat, lon = C["all"], C["lat"], C["lon"]
    w = cos_w(lat)

    print("\n--- reproduces the original run exactly ---")
    check("32 parts", C["n_parts"] == 32, f"got {C['n_parts']}")
    for k, want in EXPECT_TOTALS.items():
        got = int(A[k].sum())
        check(f"total {k}", got == want, f"{got} vs {want}")
    for q in PERIODS:
        s = C["scal"][q]
        check(f"{q} scalar tp/fp/fn", list(map(int, s[:3])) == EXPECT_SCALAR[q],
              f"{list(map(int, s[:3]))}")
        check(f"{q} n_t", int(s[4]) == EXPECT_NT[q], f"{int(s[4])}")
    check("n_t total", C["n_t"] == EXPECT_NT_TOTAL, str(C["n_t"]))
    check("n_days", C["n_days"] == EXPECT_NDAYS, str(C["n_days"]))

    print("\n--- partitions are exhaustive ---")
    ssum = sum(int(C["sea_scal"][s][4]) for s in SEASONS)
    check("seasonal n_t sums to record", ssum == EXPECT_NT_TOTAL, str(ssum))
    for k in ("tp", "fp", "fn", "pred", "targ"):
        a = int(sum(C["sea"][s][k].sum() for s in SEASONS))
        check(f"seasonal {k} sums to total", a == int(A[k].sum()), f"{a} vs {int(A[k].sum())}")
    mr = C["month_rows"]
    check("540 months", len(mr) == 540, str(len(mr)))
    check("monthly n_t sums to record", int(mr[:, 6].sum()) == EXPECT_NT_TOTAL,
          str(int(mr[:, 6].sum())))
    check("monthly tp sums to total", int(mr[:, 2].sum()) == int(A["tp"].sum()))

    print("\n--- area weighting ---")
    mw = map_metrics(A["tp"], A["fp"], A["fn"], w=w)
    for k, want in EXPECT_WEIGHTED.items():
        check(f"cos-lat {k}", abs(mw[k] - want) < 5e-4, f"{mw[k]:.4f} vs {want}")
    unw = metrics(A["tp"].sum(), A["fp"].sum(), A["fn"].sum())
    print(f"        (unweighted BIAS {unw['BIAS']:.4f} vs weighted {mw['BIAS']:.4f} — "
          f"the sign of the bias depends on the weighting)")

    print("\n--- day binning, cross-checked against the CDO-built MARS climatology ---")
    dp = A["ardays"] / 45.0
    dt = A["ardays_t"] / 45.0
    check("max pred days/yr", abs(dp.max() - EXPECT_MAX_PRED_DAYS) < 0.02, f"{dp.max():.2f}")
    check("max targ days/yr", abs(dt.max() - EXPECT_MAX_TARG_DAYS) < 0.02, f"{dt.max():.2f}")
    if os.path.exists(MARS_AVG):
        import xarray as xr
        ds = xr.open_dataset(MARS_AVG)
        ref = np.asarray(ds["MARS"].squeeze().values, float)
        ds.close()
        d = np.abs(dt - ref)
        check("python daymax == CDO MARS_AvgARDaysPerYear", float(d.max()) < 1e-3,
              f"max abs diff {float(d.max()):.2e}")
    else:
        print(f"  [SKIP] MARS climatology not found at {MARS_AVG}")

    print("\n--- per-year totals ---")
    yd = C["yr_days"].astype(np.int64)
    check("yr_days sums to ardays", int(yd.sum()) == int(A["ardays"].sum()),
          f"{int(yd.sum())} vs {int(A['ardays'].sum())}")
    check("45 years", yd.shape[0] == 45, str(yd.shape[0]))

    print("\n--- duration inputs present ---")
    check("rise nonzero", int(C["rise"].sum()) > 0, f"{int(C['rise'].sum())}")
    check("540 edge months", len(C["edge"]["keys"]) == 540, str(len(C["edge"]["keys"])))

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
