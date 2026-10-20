#!/usr/bin/env python3
"""
plot_ar_label_sweep.py — choose the AR training label on evidence: every candidate label of
phase4_build_ar_label.py scored against the five established AR methods, before any training.

Each candidate directory <HERE>/<Lx>/ holds the part files that phase4_ar_artmip_v2.py wrote
with --pred-dir pointing at that label, so the label sits in the "WNet" slot and every number
is computed exactly as for the paper figures (area weighted; evaluations/ar_artmip_eval_v2/
plot_ar_paper_figs.py). MARS and Guan-Waliser are one algorithm family (tARget), so the "mean
IoU with the independent families" counts them once: mean of [mean(MARS, GW), Mundhenk,
TempestExtremes, Reid].

Writes ar_label_sweep_table.csv / .txt and ar_label_sweep_days_per_year.png / .pdf (AR days
per year: the five established methods, then the candidates, on the scale of the paper figure).
"""
import os
import sys
import csv
import glob
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ar_mars_eval_v2"))
from ar_v2_common import LV_DAYS_120, disc  # noqa: E402

METHODS = ("MARS", "WNet", "GW", "Mundhenk", "TempestLR", "Reid500")
NAME = {"MARS": "MARS", "GW": "Guan-Waliser", "Mundhenk": "Mundhenk", "TempestLR": "TempestExtremes",
        "Reid500": "Reid"}
PAIRS = tuple(combinations(range(len(METHODS)), 2))
REGIONS = ("G60", "NHMID", "SHMID", "TROP", "NEPAC", "NATL", "SEPAC", "WNA", "WEUR", "SSA")
CAND = {"L0": "L0 shipped: IVT > 400", "L1": "L1 percentile, floor 100",
        "L2": "L2 percentile, floor 250/100", "L3": "L3 = L2 + reaches 23.5°",
        "L4": "L4 = L3 + width ≤ 1500 km", "L5": "L5 floor 100 + reaches 23.5°",
        "L6": "L6 floor 175/100 + reaches 23.5°"}
PC = ccrs.PlateCarree()
INK = "#000000"
W_ = 1                                                               # the label's slot in METHODS


def load(cand):
    parts = sorted(glob.glob(os.path.join(HERE, cand, "part_*_*.npz")))
    if not parts:
        return None
    acc = {}
    years = sorted({int(os.path.basename(p).split("_")[1]) for p in parts})
    series = np.full((len(years), 12, len(METHODS), len(REGIONS)), np.nan)
    for p in parts:
        z = np.load(p)
        for k in ("freq", "both", "ardays", "n_t"):
            a = z[k].astype(np.int64 if k == "n_t" else np.uint32)
            acc[k] = a if k not in acc else acc[k] + a
        half, yi = int(z["half"]), years.index(int(z["year"]))
        series[yi, 6 * half:6 * half + 6] = z["series"][6 * half:6 * half + 6]
        lat, lon = z["lat"].astype(float), z["lon"].astype(float)
    sh = lon.size // 2
    return dict(lat=lat, lon=np.roll(((lon + 180) % 360) - 180, sh), f=np.roll(acc["freq"].sum(0), sh, -1).astype(float),
                both=np.roll(acc["both"], sh, -1).astype(float), days=np.roll(acc["ardays"], sh, -1) / len(years),
                N=float(acc["n_t"].sum()), series=np.nanmean(series, 0), nparts=len(parts), years=years)


def pattern_r(a, b, w, m):
    a, b, w = a[m], b[m], w[m]
    am, bm = (a * w).sum() / w.sum(), (b * w).sum() / w.sum()
    return float((w * (a - am) * (b - bm)).sum() / np.sqrt((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum()))


def metrics(D, i):
    """Row of the table for method slot i of a loaded candidate set."""
    lat = D["lat"]
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, D["lon"].size))
    band = lambda a0, a1: ((np.abs(lat) >= a0) & (np.abs(lat) <= a1))[:, None] * np.ones((1, D["lon"].size), bool)  # noqa: E731
    mid, g60, trop, polar = band(20, 60), band(0, 60), band(0, 15), band(60, 90)

    def iou(j, m):
        p = PAIRS.index((min(i, j), max(i, j)))
        return float((D["both"][p] * w)[m].sum() / ((D["f"][i] + D["f"][j] - D["both"][p]) * w)[m].sum())

    # every row is compared with the ESTABLISHED methods only, so the rows of the established
    # methods do not depend on which label happens to sit in the WNet slot
    others = [j for j in range(len(METHODS)) if j != i and j != W_]
    ious = {METHODS[j]: iou(j, mid) for j in others}
    me = METHODS[i]
    fam = [] if me in ("MARS", "GW") else [np.mean([ious["MARS"], ious["GW"]])]   # tARget counted once,
    fam += [ious[k] for k in ("Mundhenk", "TempestLR", "Reid500") if k != me]     # and never against itself
    mean_others = D["f"][others].mean(0)
    fr = lambda m: 100.0 * (D["f"][i] * w)[m].sum() / (D["N"] * w[m].sum())   # noqa: E731
    nh = 100.0 * D["series"][:, i, REGIONS.index("NHMID")]
    d = D["days"][i]
    return dict(freq_mid=fr(mid), freq_trop=fr(trop), freq_polar=fr(polar),
                land_wna=100.0 * np.nanmean(D["series"][:, i, REGIONS.index("WNA")]),
                land_weur=100.0 * np.nanmean(D["series"][:, i, REGIONS.index("WEUR")]),
                iou_fam=float(np.mean(fam)), **{f"iou_{k}": ious.get(k, float("nan")) for k in NAME},
                r_mid=pattern_r(D["f"][i], mean_others, w, mid), r_g60=pattern_r(D["f"][i], mean_others, w, g60),
                nh_min=float(np.nanmin(nh)), nh_max=float(np.nanmax(nh)),
                days_max=float(d.max()), area_gt120=100.0 * (w * (d > 120)).sum() / w.sum())


def main():
    data = {c: load(c) for c in CAND}
    data = {c: d for c, d in data.items() if d is not None}
    if not data:
        raise SystemExit("no part files; run Run_ar_label_sweep.sh first")
    first = next(iter(data.values()))
    print(f"years {first['years']}, candidates {list(data)}, parts " + ", ".join(f"{c}:{d['nparts']}" for c, d in data.items()))

    rows = [(NAME[m], metrics(first, METHODS.index(m))) for m in NAME]          # the five established methods
    rows += [(CAND[c], metrics(d, W_)) for c, d in data.items()]
    cols = [("freq_mid", "AR freq 20-60 %"), ("freq_trop", "freq |lat|<15 %"), ("freq_polar", "freq |lat|>60 %"),
            ("land_wna", "W N America %"), ("land_weur", "W Europe %"), ("iou_fam", "mean IoU indep."),
            ("r_mid", "pattern r 20-60"), ("r_g60", "pattern r 60S-60N"), ("nh_min", "NH month min %"),
            ("nh_max", "NH month max %"), ("days_max", "max days/yr"), ("area_gt120", "area >120 d %")]
    lines = [f"{'':<32}" + "".join(f"{h:>19}" for _, h in cols)]
    for name, m in rows:
        lines.append(f"{name:<32}" + "".join(f"{m[k]:>19.3f}" for k, _ in cols))
    lines += ["", "Pairwise IoU of each candidate with the established methods, 20-60 deg, area weighted:",
              f"{'':<32}" + "".join(f"{NAME[k]:>17}" for k in NAME)]
    for c, d in data.items():
        m = metrics(d, W_)
        lines.append(f"{CAND[c]:<32}" + "".join(f"{m['iou_' + k]:>17.3f}" for k in NAME))
    lines += ["", "'mean IoU indep.' = mean IoU (20-60 deg) with the independent established families: tARget (MARS and",
              "Guan-Waliser averaged, and left out of each other's row), Mundhenk, TempestExtremes, Reid. Pattern r is",
              "against the mean frequency map of the other established methods. 'W N America' and 'W Europe' are the",
              "boxes 25-60N 125-105W and 30-65N 10W-10E of the paper."]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "ar_label_sweep_table.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(HERE, "ar_label_sweep_table.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        keys = list(rows[-1][1].keys())
        wr.writerow(["method"] + keys)
        for name, m in rows:
            wr.writerow([name] + [f"{m.get(k, float('nan')):.4f}" for k in keys])

    panels = [(NAME[m], first["days"][METHODS.index(m)]) for m in NAME] + [(CAND[c], d["days"][W_]) for c, d in data.items()]
    ncol = 5
    nrow = (len(panels) + ncol - 1) // ncol
    Wd, left, right, gap, top, midg, bot = 22.0, 0.55, 0.15, 0.25, 0.34, 0.55, 1.0
    w = (Wd - left - right - (ncol - 1) * gap) / ncol
    h = w / 2.0
    H = top + nrow * h + (nrow - 1) * midg + bot
    fig = plt.figure(figsize=(Wd, H))
    cmap, norm = disc(LV_DAYS_120)
    for k, (title, f) in enumerate(panels):
        r, c = divmod(k, ncol)
        ax = fig.add_axes([(left + c * (w + gap)) / Wd, (bot + (nrow - 1 - r) * (h + midg)) / H, w / Wd, h / H],
                          projection=PC)
        ax.set_global()
        mesh = ax.pcolormesh(first["lon"], first["lat"], f, cmap=cmap, norm=norm, transform=PC, shading="auto",
                             rasterized=True)
        ax.coastlines(linewidth=0.4, color=INK)
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.xlocator = FixedLocator(np.arange(-120, 180, 60))
        gl.ylocator = MultipleLocator(30)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (c == 0)
        gl.xlabel_style = gl.ylabel_style = {"size": 9, "color": INK}
        ax.set_title(f"({'abcdefghijkl'[k]}) {title}", fontsize=12, color=INK, loc="left", pad=5)
    cb = fig.colorbar(mesh, cax=fig.add_axes([0.30, 0.40 / H, 0.40, 0.16 / H]), orientation="horizontal",
                      extend="max", ticks=LV_DAYS_120)
    cb.set_label(f"Mean AR days per year ({', '.join(map(str, first['years']))})", fontsize=11.5, color=INK)
    cb.ax.tick_params(labelsize=10, colors=INK)
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"ar_label_sweep_days_per_year.{ext}")
        fig.savefig(p, dpi=250, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
