#!/usr/bin/env python3
"""
merge_ar_objects_v2.py — combine the objpart_*.json shards and plot POD/FAR.

Writes ar_object_metrics_<label>.json and ar_object_monthly_<label>.csv in the
same shape as the v1 evaluations/ar_mars_eval/ar_object_metrics.json, and draws
POD and FAR against overlap fraction with the v1 numbers alongside when they are
available.

Usage
  python merge_ar_objects_v2.py [--cache-dir .] [--label v2]
"""
import os
import sys
import csv
import glob
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V1_JSON = os.path.join(HERE, "..", "ar_mars_eval", "ar_object_metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=HERE)
    ap.add_argument("--label", default="v2")
    args = ap.parse_args()
    D = args.cache_dir

    parts = sorted(glob.glob(os.path.join(D, "objpart_*.json")))
    if not parts:
        raise SystemExit(f"no objpart_*.json in {D}; run phase4_ar_objects_v2.py first")

    fracs = minpx = None
    acc, monthly = {}, []
    for p in parts:
        with open(p) as fh:
            d = json.load(fh)
        fracs, minpx = d["fracs"], d["minpx"]
        monthly.extend(d["monthly"])
        for per, a in d["acc"].items():
            t = acc.setdefault(per, dict(n_obs=0, n_pred=0,
                                         obs_hit={str(f): 0 for f in fracs},
                                         pred_match={str(f): 0 for f in fracs}))
            t["n_obs"] += a["n_obs"]
            t["n_pred"] += a["n_pred"]
            for f in fracs:
                t["obs_hit"][str(f)] += a["obs_hit"][str(f)]
                t["pred_match"][str(f)] += a["pred_match"][str(f)]

    summ = {}
    for per, a in acc.items():
        d = dict(n_obs=a["n_obs"], n_pred=a["n_pred"],
                 obj_count_bias=a["n_pred"] / max(a["n_obs"], 1))
        for f in fracs:
            d[f"POD_{f}"] = a["obs_hit"][str(f)] / max(a["n_obs"], 1)
            d[f"FAR_{f}"] = 1 - a["pred_match"][str(f)] / max(a["n_pred"], 1)
        summ[per] = d

    jp = os.path.join(D, f"ar_object_metrics_{args.label}.json")
    with open(jp, "w") as fh:
        json.dump(dict(fracs=fracs, minpx=minpx, periods=summ), fh, indent=2)
    print(f"  wrote {jp}")

    monthly.sort(key=lambda r: (r["year"], r["month"]))
    cp = os.path.join(D, f"ar_object_monthly_{args.label}.csv")
    if monthly:
        with open(cp, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(monthly[0].keys()))
            wtr.writeheader()
            wtr.writerows(monthly)
        print(f"  wrote {cp} ({len(monthly)} months)")

    v1 = None
    if os.path.exists(V1_JSON):
        with open(V1_JSON) as fh:
            v1 = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.25, left=0.07, right=0.98, top=0.88, bottom=0.13)
    styles = {"oop": "-o", "primary": "--s"}
    for ax, met, title in ((axes[0], "POD", "Object detection rate (POD)"),
                           (axes[1], "FAR", "Object false-alarm rate (FAR)")):
        for per in sorted(summ):
            ax.plot(fracs, [summ[per][f"{met}_{f}"] for f in fracs],
                    styles.get(per, "-o"), color="#1baf7a", label=f"{args.label} {per}")
            if v1 and per in v1["periods"]:
                ax.plot(fracs, [v1["periods"][per][f"{met}_{f}"] for f in fracs],
                        styles.get(per, "-o"), color="#eb6834", label=f"v1 {per}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("required overlap fraction", fontsize=9)
        ax.set_ylabel(met, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    out = os.path.join(D, f"ar_objects_{args.label}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  wrote {out}")

    print("\n=== OBJECT-OVERLAP SUMMARY ===")
    for per in sorted(summ):
        d = summ[per]
        print(f"{per:8s} n_obs={d['n_obs']} n_pred={d['n_pred']} "
              f"count_bias={d['obj_count_bias']:.3f}")
        for f in fracs:
            extra = ""
            if v1 and per in v1["periods"]:
                extra = (f"   (v1 POD={v1['periods'][per][f'POD_{f}']:.4f} "
                         f"FAR={v1['periods'][per][f'FAR_{f}']:.4f})")
            print(f"         @frac{f}: POD={d[f'POD_{f}']:.4f}  "
                  f"FAR={d[f'FAR_{f}']:.4f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
