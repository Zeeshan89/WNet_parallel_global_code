#!/usr/bin/env python3
"""
merge_ar_v2.py — merge the part_*.npz cache and write every AR summary product.

Writes into the cache directory:
  summary_ar_v2_primary_oop.txt   published text layout, plus a latitude-band block
  summary_ar_v2_primary_oop.csv   v1 schema, with unweighted AND area-weighted rows
  ar_v2_latband.csv               skill by latitude band
  ar_v2_seasonal.csv              skill by season
  ar_monthly_v2.csv               monthly time series
  ar_v2_merged.npz                consolidated maps, so the plotters need no re-merge
  ar_duration_v2.npz              mean AR event duration (pred and target)

Why both weightings are reported: on a 721x1440 regular grid an unweighted pixel
count over-weights polar rows about 15x. For the v2 AR product that is enough to
move the frequency bias from 0.878 (under-detection) to 1.124 (over-detection),
so a single unweighted number is not a safe summary. The v1 analysis already
carried both rows in its CSV; this restores that.

Usage
  python merge_ar_v2.py [--cache-dir .] [--label v2]
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import (  # noqa: E402
    BANDS, PERIODS, PERIOD_YEARS, SEASONS, YEAR0,
    cos_w, duration, load_parts, map_metrics, metrics)

CSV_COLS = ("period,weighting,IoU,F1,Precision,POD,FAR,BIAS,HSS,"
            "hits,false_alarms,misses,n_timesteps")


def block(name, tp, fp, fn, tn, n_t):
    """The published text block, unchanged in layout."""
    m = metrics(tp, fp, fn, tn)
    return (f"{name}:\n"
            f"  IoU          = {m['IoU']:.6f}\n  Dice         = {m['Dice']:.6f}\n"
            f"  F1           = {m['F1']:.6f}\n  Precision    = {m['Precision']:.6f}\n"
            f"  POD          = {m['POD']:.6f}\n  FAR          = {m['FAR']:.6f}\n"
            f"  BIAS         = {m['BIAS']:.6f}\n  HSS          = {m['HSS']:.6f}\n"
            f"  tp           = {int(tp)}\n  fp           = {int(fp)}\n"
            f"  fn           = {int(fn)}\n  n_t          = {int(n_t)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--label", default="v2")
    ap.add_argument("--pred-name", default="Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final")
    args = ap.parse_args()
    D = args.cache_dir

    C = load_parts(D)
    lat, lon = C["lat"], C["lon"]
    w = cos_w(lat)
    alat = np.abs(np.asarray(lat, float))
    n_t = C["n_t"]
    print(f"merged {C['n_parts']} parts: n_t={n_t}, n_days={C['n_days']}, "
          f"months={len(C['month_rows'])}")

    # ---------------------------------------------------------------- text
    lines = ["=" * 70,
             f"WNet-{args.label} AR Detection vs MARS (modified tARget-v3) - pixel skill",
             "=" * 70,
             f"Predictions : {args.pred_name}",
             "Target var  : MARS", ""]
    for q in PERIODS:
        tp, fp, fn, tn, nt = (int(x) for x in C["scal"][q])
        lines.append(block(f"ar_pixel_{q}", tp, fp, fn, tn, nt))
    tot = sum(C["scal"][q] for q in PERIODS)
    lines.append(block("ar_pixel_entire", *(int(x) for x in tot)))

    # area-weighted whole record, from the per-pixel maps
    A = C["all"]
    mw = map_metrics(A["tp"], A["fp"], A["fn"], w=w)
    lines.append("ar_pixel_entire_area_weighted:\n" + "".join(
        f"  {k:<12} = {mw[k]:.6f}\n" for k in
        ("IoU", "Dice", "F1", "Precision", "POD", "FAR", "BIAS")))

    # latitude-band block -- the diagnostic the single global number hides
    lines.append("latitude bands (whole record, cos-lat weighted):")
    lines.append(f"  {'band':<24}{'IoU':>9}{'POD':>9}{'FAR':>9}{'BIAS':>9}"
                 f"{'freq_MARS':>11}{'freq_pred':>11}")
    band_rows = []
    for bname, sel in BANDS:
        msk = sel(alat)
        m = map_metrics(A["tp"], A["fp"], A["fn"], w=w, mask=msk)
        fq_t = float((A["targ"][msk] * w[msk]).sum() / (w[msk].sum() * lon.size * n_t))
        fq_p = float((A["pred"][msk] * w[msk]).sum() / (w[msk].sum() * lon.size * n_t))
        lines.append(f"  {bname:<24}{m['IoU']:>9.4f}{m['POD']:>9.4f}{m['FAR']:>9.4f}"
                     f"{m['BIAS']:>9.4f}{fq_t:>11.4f}{fq_p:>11.4f}")
        band_rows.append((bname, m, fq_t, fq_p))
    lines.append("")

    txt = os.path.join(D, f"summary_ar_{args.label}_primary_oop.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {txt}")

    # ---------------------------------------------------------------- csv
    rows = [CSV_COLS]

    def csv_row(period, weighting, m, tp=None, fp=None, fn=None, nt=None):
        hss = f"{m['HSS']:.4f}" if "HSS" in m else ""
        cnt = ("" if tp is None else f"{float(tp)},{float(fp)},{float(fn)}")
        if tp is None:
            cnt = ",,"
        return (f"{period},{weighting},{m['IoU']:.4f},{m['F1']:.4f},"
                f"{m['Precision']:.4f},{m['POD']:.4f},{m['FAR']:.4f},"
                f"{m['BIAS']:.4f},{hss},{cnt},{nt}")

    for q in PERIODS:
        tp, fp, fn, tn, nt = (int(x) for x in C["scal"][q])
        rows.append(csv_row(q, "unweighted", metrics(tp, fp, fn, tn), tp, fp, fn, nt))
        P = C["per"][q]
        rows.append(csv_row(q, "area-weighted",
                            map_metrics(P["tp"], P["fp"], P["fn"], w=w), nt=nt))
    tp, fp, fn, tn, nt = (int(x) for x in tot)
    rows.append(csv_row("entire", "unweighted", metrics(tp, fp, fn, tn), tp, fp, fn, nt))
    rows.append(csv_row("entire", "area-weighted", mw, nt=nt))
    csvp = os.path.join(D, f"summary_ar_{args.label}_primary_oop.csv")
    with open(csvp, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  wrote {csvp}")

    # ---------------------------------------------------------------- lat bands
    out = ["band,weighting,IoU,F1,POD,FAR,BIAS,freq_MARS,freq_pred"]
    for bname, sel in BANDS:
        msk = sel(alat)
        for wt, ww in (("unweighted", None), ("area-weighted", w)):
            m = map_metrics(A["tp"], A["fp"], A["fn"], w=ww, mask=msk)
            den = (msk.sum() * lon.size * n_t) if ww is None else (
                ww[msk].sum() * lon.size * n_t)
            num_t = A["targ"][msk].sum() if ww is None else (A["targ"][msk] * ww[msk]).sum()
            num_p = A["pred"][msk].sum() if ww is None else (A["pred"][msk] * ww[msk]).sum()
            out.append(f"{bname},{wt},{m['IoU']:.4f},{m['F1']:.4f},{m['POD']:.4f},"
                       f"{m['FAR']:.4f},{m['BIAS']:.4f},"
                       f"{float(num_t/den):.5f},{float(num_p/den):.5f}")
    p = os.path.join(D, f"ar_{args.label}_latband.csv")
    with open(p, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {p}")

    # ---------------------------------------------------------------- seasons
    out = ["season,weighting,IoU,F1,POD,FAR,BIAS,HSS,n_timesteps"]
    for s in SEASONS:
        tp, fp, fn, tn, nt = (int(x) for x in C["sea_scal"][s])
        m = metrics(tp, fp, fn, tn)
        out.append(f"{s},unweighted,{m['IoU']:.4f},{m['F1']:.4f},{m['POD']:.4f},"
                   f"{m['FAR']:.4f},{m['BIAS']:.4f},{m['HSS']:.4f},{nt}")
        S = C["sea"][s]
        mw_s = map_metrics(S["tp"], S["fp"], S["fn"], w=w)
        out.append(f"{s},area-weighted,{mw_s['IoU']:.4f},{mw_s['F1']:.4f},"
                   f"{mw_s['POD']:.4f},{mw_s['FAR']:.4f},{mw_s['BIAS']:.4f},,{nt}")
    p = os.path.join(D, f"ar_{args.label}_seasonal.csv")
    with open(p, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {p}")

    # ---------------------------------------------------------------- monthly
    out = ["year,month,IoU,F1,POD,FAR,BIAS,HSS,tp,fp,fn,n_timesteps"]
    for y, mo, tp, fp, fn, tn, nt in C["month_rows"]:
        m = metrics(tp, fp, fn, tn)
        out.append(f"{y},{mo},{m['IoU']:.4f},{m['F1']:.4f},{m['POD']:.4f},"
                   f"{m['FAR']:.4f},{m['BIAS']:.4f},{m['HSS']:.4f},{tp},{fp},{fn},{nt}")
    p = os.path.join(D, f"ar_monthly_{args.label}.csv")
    with open(p, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {p} ({len(C['month_rows'])} months)")

    # ---------------------------------------------------------------- duration
    dur_p, zon_p = duration(C, "pred")
    dur_t, zon_t = duration(C, "targ")
    p = os.path.join(D, f"ar_duration_{args.label}.npz")
    # zon_wnet / dur_wnet are the key names merge_and_plot_ar_3row.py looks for
    np.savez_compressed(p, lat=lat, dur_wnet=dur_p, zon_wnet=zon_p,
                        dur_mars=dur_t, zon_mars=zon_t)
    print(f"  wrote {p}  (pred zonal-mean duration "
          f"{np.nanmean(zon_p):.2f} d, MARS {np.nanmean(zon_t):.2f} d)")

    # ---------------------------------------------------------------- maps
    p = os.path.join(D, f"ar_{args.label}_merged.npz")
    payload = dict(lat=lat, lon=lon, n_t=n_t, n_days=C["n_days"],
                   yr_days=C["yr_days"], yr_days_t=C["yr_days_t"],
                   year0=YEAR0, month_rows=C["month_rows"])
    payload.update({f"all_{k}": v for k, v in A.items()})
    for q in PERIODS:
        payload.update({f"per_{q}_{k}": v for k, v in C["per"][q].items()})
        payload[f"scal_{q}"] = C["scal"][q]
    for s in SEASONS:
        payload.update({f"sea_{s}_{k}": v for k, v in C["sea"][s].items()})
        payload[f"scal_{s}"] = C["sea_scal"][s]
    np.savez_compressed(p, **payload)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
