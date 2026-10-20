#!/usr/bin/env python3
"""
plot_ar_paper_figs.py — the paper's atmospheric river (AR) figures: WNet among established
AR detection methods on ERA5, 1980-2019, 6-hourly.

There is no observed truth for ARs, and MARS is itself a detection method (tARget v3), so no
method is treated as a benchmark. Every comparison is symmetric: pairwise overlap, each
method against the mean of the OTHER methods, and confirmation by other methods.

  MARS (tARget v3) | Guan-Waliser v2 | Mundhenk v3 | TempestExtremes (TempestLR) | Reid500 | WNet

All six are global, so they share one domain. Maps show the four methods of --show; the
statistics use all six.

  Fig 1  ar_paper_frequency        AR frequency maps, r = pattern correlation with the mean of
                                   the other five methods
  Fig 2  ar_paper_agreement        (a) pairwise IoU in the midlatitudes, (b) mean IoU with the
                                   other methods against latitude, (c) confirmation by others
  Fig 3  ar_paper_zonal_seasonal   zonal-mean frequency; seasonal cycle, both midlatitude belts
  Fig 4  ar_paper_regional_cycle   seasonal cycle in six regions (ocean approach, coast to land)
  Fig 5  ar_paper_regional_annual  annual AR frequency in the same regions, 1980-2019
  Fig 6  ar_paper_days_per_year    mean AR days per year, all six methods (2 x 3), in the style and on
                                   the 0 to 120 scale of ar_mars_eval_v2/ar_days_per_year_mars_wnet
  Tables ar_paper_iou.csv, ar_paper_confirmation.csv, ar_paper_regional.csv

Reads the part_YYYY_H.npz files of phase4_ar_artmip_v2.py; the merge is cached.
"""
import os
import sys
import csv
import glob
import argparse
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ar_mars_eval_v2"))
from ar_v2_common import LV_DAYS_120, disc  # noqa: E402  (scale of ar_days_per_year_mars_wnet)
MERGED = os.path.join(HERE, "ar_artmip_merged.npz")
METHODS = ("MARS", "WNet", "GW", "Mundhenk", "TempestLR", "Reid500")
PAIRS = tuple(combinations(range(len(METHODS)), 2))
# label, colour, marker
STYLE = {"MARS": ("MARS", "#1b9e77", "o"), "GW": ("Guan-Waliser", "#66a61e", "v"),
         "Mundhenk": ("Mundhenk", "#d95f02", "^"), "TempestLR": ("TempestExtremes", "#e7298a", "D"),
         "Reid500": ("Reid", "#a6761d", "P"), "WNet": ("WNet", "#7570b3", "s")}
ORDER = ("MARS", "GW", "Mundhenk", "TempestLR", "Reid500", "WNet")          # display order
REGIONS = ("G60", "NHMID", "SHMID", "TROP", "NEPAC", "NATL", "SEPAC", "WNA", "WEUR", "SSA")
REGION_NAME = {"NEPAC": "Northeast Pacific", "NATL": "North Atlantic", "SEPAC": "Southeast Pacific",
               "WNA": "Western North America", "WEUR": "Western Europe", "SSA": "Southern South America"}
SIX = ("NEPAC", "NATL", "SEPAC", "WNA", "WEUR", "SSA")
INK = "#000000"                       # black throughout; grey prints blurry
FS_TITLE, FS_LAB, FS_TICK, FS_LEG = 13, 11.5, 10, 11
MONTHS = list("JFMAMJJASOND")
PC = ccrs.PlateCarree()
MID = (20.0, 60.0)                    # the midlatitude belt of both hemispheres


# ---------------------------------------------------------------- data
def merge():
    parts = sorted(glob.glob(os.path.join(HERE, "part_*_*.npz")))
    if not parts:
        raise SystemExit("no part files; run Run_ar_artmip_v2.sh first")
    years = sorted({int(os.path.basename(p).split("_")[1]) for p in parts})
    acc, series, dropped = {}, np.full((len(years), 12, len(METHODS), len(REGIONS)), np.nan), 0
    for p in parts:
        z = np.load(p)
        for k in ("freq", "both", "ardays", "n_t", "voteA_lat", "voteB_lat", "voteA_reg", "voteB_reg"):
            a = z[k].astype(np.int64 if k.startswith("vote") or k == "n_t" else np.uint32)
            acc[k] = a if k not in acc else acc[k] + a
        half, yi = int(z["half"]), years.index(int(z["year"]))
        series[yi, 6 * half:6 * half + 6] = z["series"][6 * half:6 * half + 6]
        dropped += int(z["dropped"])
        lat, lon = z["lat"], z["lon"]
    np.savez_compressed(MERGED, lat=lat, lon=lon, years=np.array(years), series=series,
                        n_parts=len(parts), dropped=dropped, **acc)
    print(f"  merged {len(parts)} parts, {years[0]}-{years[-1]}, {int(acc['n_t'].sum())} time steps, "
          f"{dropped} dropped -> {MERGED}", flush=True)


class Data:
    def __init__(self):
        z = np.load(MERGED)
        self.lat, lon = z["lat"].astype(float), z["lon"].astype(float)
        self.shift = lon.size // 2                                   # 0..360 -> -180..180
        self.lon = np.roll(((lon + 180) % 360) - 180, self.shift)
        self.freq = np.roll(z["freq"], self.shift, axis=-1)          # (season, method, ny, nx)
        self.both = np.roll(z["both"], self.shift, axis=-1)
        # a day counts when an AR is present at any of its four 6-hourly times
        self.ardays = np.roll(z["ardays"], self.shift, axis=-1).astype(np.float64)
        self.n_t, self.years, self.series = z["n_t"], z["years"], z["series"]
        self.vA, self.vB = z["voteA_lat"], z["voteB_lat"]
        self.w = np.cos(np.radians(self.lat))[:, None] * np.ones((1, self.lon.size))
        self.f = self.freq.sum(0).astype(np.float64)                 # all-year counts per method
        self.N = float(self.n_t.sum())

    def mask(self, la0, la1, both_hemispheres=True):
        a = np.abs(self.lat) if both_hemispheres else self.lat
        return ((a >= la0) & (a <= la1))[:, None] * np.ones((1, self.lon.size), bool)

    def iou(self, i, j, m):
        p = PAIRS.index((min(i, j), max(i, j)))
        inter = (self.both[p] * self.w)[m].sum()
        return float(inter / ((self.f[i] + self.f[j] - self.both[p]) * self.w)[m].sum())


def idx(name):
    return METHODS.index(name)


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  wrote {p}", flush=True)
    plt.close(fig)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=FS_TITLE, color=INK, loc="left", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LAB, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LAB, color=INK)
    ax.tick_params(labelsize=FS_TICK, colors=INK)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def line_legend(fig, names, y=1.0):
    fig.legend(handles=[Line2D([], [], color=STYLE[n][1], lw=2.2, marker=STYLE[n][2], ms=6,
                               label=STYLE[n][0]) for n in names],
               loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(names), frameon=False,
               fontsize=FS_LEG, labelcolor=INK, columnspacing=2.0)


def pattern_r(a, b, w, m):
    a, b, w = a[m], b[m], w[m]
    am, bm = (a * w).sum() / w.sum(), (b * w).sum() / w.sum()
    return float((w * (a - am) * (b - bm)).sum()
                 / np.sqrt((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum()))


# ---------------------------------------------------------------- figures
def fig_frequency(D, show):
    vmax = 16.0
    fr = 100.0 * D.f / D.N                                           # % of time steps
    belt = D.mask(*MID)              # the 20-60 degree belt every other statistic here uses
    W, left, right, gap, top, mid, bot = 15.5, 0.7, 1.35, 0.55, 0.34, 0.62, 0.42
    w = (W - left - right - gap) / 2
    h = w / (360.0 / 160.0)                                          # 80S-80N at true shape
    H = top + 2 * h + mid + bot
    fig = plt.figure(figsize=(W, H))
    for k, name in enumerate(show):
        r, c = divmod(k, 2)
        ax = fig.add_axes([(left + c * (w + gap)) / W, (bot + (1 - r) * (h + mid)) / H, w / W, h / H],
                          projection=PC)
        ax.set_extent([-180, 180, -80, 80], crs=PC)
        ax.add_feature(cfeature.LAND, facecolor="#ededed", zorder=0)
        mm = ax.pcolormesh(D.lon, D.lat, np.ma.masked_less(fr[idx(name)], 0.04 * vmax), cmap="YlGnBu",
                           vmin=0, vmax=vmax, shading="auto", transform=PC, rasterized=True, zorder=2)
        ax.coastlines(linewidth=0.6, color=INK, zorder=4)
        for la in (-MID[1], -MID[0], MID[0], MID[1]):                 # the belt r refers to
            ax.plot([-180, 180], [la, la], ls=(0, (4, 3)), color=INK, lw=0.7, transform=PC, zorder=5)
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.xlocator, gl.ylocator = MultipleLocator(60), MultipleLocator(30)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (c == 0)
        gl.xlabel_style = gl.ylabel_style = {"size": FS_TICK, "color": INK}
        ax.set_title(f"({'abcd'[k]}) {STYLE[name][0]}", fontsize=FS_TITLE, color=INK, loc="left", pad=5)
        others = np.mean([fr[idx(o)] for o in METHODS if o != name], axis=0)
        ax.text(0.992, 0.05, f"r = {pattern_r(fr[idx(name)], others, D.w, belt):.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_TICK, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK, lw=0.5))
    cb = fig.colorbar(mm, cax=fig.add_axes([1 - 0.95 / W, (bot + 0.25 * h) / H, 0.011, (1.5 * h + mid) / H]),
                      extend="max", ticks=np.arange(0, vmax + 0.1, 2))
    cb.set_label("AR frequency (% of time steps)", fontsize=FS_LAB, color=INK)
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK)
    cb.outline.set_edgecolor(INK)
    save(fig, "ar_paper_frequency")


def fig_days(D, order):
    """Mean AR days per year for every method. All six are global, so every panel is global.
    TempestExtremes detects no ARs within 15 degrees of the equator by design."""
    days = D.ardays / float(D.years.size)
    cmap, norm = disc(LV_DAYS_120)
    ncol = 3                                                         # 2 rows x 3 columns
    W, left, right, gap, top, mid, bot = 15.5, 0.62, 0.15, 0.30, 0.34, 0.60, 1.05
    w = (W - left - right - (ncol - 1) * gap) / ncol
    h = w / 2.0                                                      # the whole globe at true shape
    nrow = (len(order) + ncol - 1) // ncol
    H = top + nrow * h + (nrow - 1) * mid + bot
    fig = plt.figure(figsize=(W, H))
    print(f"\n{'AR days per year':<18}{'max':>7}{'area > 120':>12}{'mean 20-60':>12}")
    belt = D.mask(*MID)
    for k, name in enumerate(order):
        r, c = divmod(k, ncol)
        ax = fig.add_axes([(left + c * (w + gap)) / W, (bot + (nrow - 1 - r) * (h + mid)) / H,
                           w / W, h / H], projection=PC)
        ax.set_global()
        f = days[idx(name)]
        m = ax.pcolormesh(D.lon, D.lat, f, cmap=cmap, norm=norm, transform=PC, shading="auto",
                          rasterized=True)
        ax.coastlines(linewidth=0.4, color=INK)
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        # neighbouring panels meet at the dateline, where both would print 180 and collide:
        # only the outer edges of a row keep their 180 label
        gl.xlocator = FixedLocator(np.arange(-180 if c == 0 else -120,
                                             181 if c == ncol - 1 else 180, 60))
        gl.ylocator = MultipleLocator(30)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (c == 0)
        gl.xlabel_style = gl.ylabel_style = {"size": FS_TICK, "color": INK}
        ax.set_title(f"({'abcdef'[k]}) {STYLE[name][0]}", fontsize=FS_TITLE + 1, color=INK,
                     loc="left", pad=6)
        print(f"{STYLE[name][0]:<18}{f.max():>7.0f}{100 * (D.w * (f > 120)).sum() / D.w.sum():>11.1f}%"
              f"{(f * D.w)[belt].sum() / D.w[belt].sum():>12.1f}")
    cb = fig.colorbar(m, cax=fig.add_axes([0.27, 0.42 / H, 0.46, 0.17 / H]), orientation="horizontal",
                      extend="max", ticks=LV_DAYS_120)
    cb.set_label("Mean AR days per year", fontsize=FS_LAB, color=INK)
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK)
    cb.outline.set_edgecolor(INK)
    save(fig, "ar_paper_days_per_year")


def confirmation(D, name, rows):
    """(share of the method's AR area that >= 2 others also flag,
        share of the area that >= 3 others flag which the method also flags)."""
    i = idx(name)
    wl = np.cos(np.radians(D.lat))[rows]
    A = (D.vA[i][:, rows] * wl).sum(1)                                # by number of agreeing others
    B = (D.vB[i][:, rows] * wl).sum(1)
    return float(A[2:].sum() / A.sum()), float(A[3:].sum() / B[3:].sum())


def fig_agreement(D):
    mid = D.mask(*MID)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw=dict(width_ratios=[1.05, 1.15, 1.0]))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.15, wspace=0.30)

    ax = axes[0]                                                     # (a) pairwise IoU
    n = len(ORDER)
    M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if a != b:
                M[a, b] = D.iou(idx(ORDER[a]), idx(ORDER[b]), mid)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8)
    for a in range(n):
        for b in range(n):
            if a != b:
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=FS_TICK,
                        color="white" if M[a, b] > 0.5 else INK)
    labs = [STYLE[o][0].replace("TempestExtremes", "TempestExt.") for o in ORDER]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, fontsize=FS_TICK, color=INK, rotation=35, ha="right")
    ax.set_yticklabels(labs, fontsize=FS_TICK, color=INK)
    ax.tick_params(length=0)
    ax.set_title("(a) Pairwise IoU, 20° to 60° latitude", fontsize=FS_TITLE, color=INK, loc="left", pad=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=FS_TICK - 1, colors=INK)

    ax = axes[1]                                                     # (b) mean IoU with the others, by latitude
    edges = np.arange(-80, 81, 10)
    for name in ORDER:
        y = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = ((D.lat >= a) & (D.lat < b))[:, None] * np.ones((1, D.lon.size), bool)
            y.append(np.mean([D.iou(idx(name), idx(o), m) for o in METHODS if o != name]))
        ax.plot(0.5 * (edges[:-1] + edges[1:]), y, "-", color=STYLE[name][1], marker=STYLE[name][2],
                lw=2.4 if name == "WNet" else 1.8, ms=5.5, zorder=4 if name == "WNet" else 2)
    style(ax, "(b) Mean IoU with the other five methods", "Latitude", "IoU")
    ax.set_xlim(-80, 80)
    ax.set_ylim(0, 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "0°" if v == 0 else f"{abs(v):.0f}°{'N' if v > 0 else 'S'}"))

    ax = axes[2]                                                     # (c) confirmation by other methods
    rows = (np.abs(D.lat) >= MID[0]) & (np.abs(D.lat) <= MID[1])
    conf = {o: confirmation(D, o, rows) for o in ORDER}
    x, wd = np.arange(2), 0.8 / n
    for k, o in enumerate(ORDER):
        ax.bar(x + (k - (n - 1) / 2) * wd, conf[o], wd * 0.92, color=STYLE[o][1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Own AR area that\ntwo or more other\nmethods confirm",
                        "Area of three or more\nother methods that\nthe method captures"],
                       fontsize=FS_TICK, color=INK)
    style(ax, "(c) Agreement with the other methods", None, "Fraction")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", alpha=0)
    fig.legend(handles=[Patch(color=STYLE[o][1], label=STYLE[o][0]) for o in ORDER], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=n, frameon=False, fontsize=FS_LEG, labelcolor=INK,
               columnspacing=1.8)
    save(fig, "ar_paper_agreement")
    return M, conf


def fig_zonal_seasonal(D):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.15, wspace=0.24)
    fr = 100.0 * D.f / D.N
    clim = 100.0 * np.nanmean(D.series, axis=0)                      # (month, method, region)
    for name in ORDER:
        lab, col, mk = STYLE[name]
        kw = dict(color=col, lw=2.4 if name == "WNet" else 1.8, zorder=4 if name == "WNet" else 2)
        axes[0].plot(fr[idx(name)].mean(1), D.lat, "-", **kw)
        for ax, reg in ((axes[1], "NHMID"), (axes[2], "SHMID")):
            ax.plot(range(12), clim[:, idx(name), REGIONS.index(reg)], "-", marker=mk, ms=5.5, **kw)
    style(axes[0], "(a) Zonal-mean AR frequency", "AR frequency (%)", "Latitude")
    axes[0].set_ylim(-80, 80)
    axes[0].yaxis.set_major_locator(MultipleLocator(20))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "0°" if v == 0 else f"{abs(v):.0f}°{'N' if v > 0 else 'S'}"))
    axes[0].set_xlim(left=0)
    for ax, t in ((axes[1], "(b) Northern midlatitudes, 20°N to 60°N"),
                  (axes[2], "(c) Southern midlatitudes, 20°S to 60°S")):
        style(ax, t, "Month", "AR frequency (%)")
        ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
        ax.set_ylim(bottom=0)
    line_legend(fig, ORDER)
    save(fig, "ar_paper_zonal_seasonal")


def fig_regional(D, annual):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.08, wspace=0.18, hspace=0.32)
    rows = []
    for k, (ax, reg) in enumerate(zip(axes.ravel(), SIX)):
        s = 100.0 * D.series[:, :, :, REGIONS.index(reg)]            # (year, month, method)
        y = np.nanmean(s, axis=1) if annual else np.nanmean(s, axis=0)
        x = D.years if annual else np.arange(12)
        for name in ORDER:
            ax.plot(x, y[:, idx(name)], "-", color=STYLE[name][1], marker=STYLE[name][2],
                    lw=2.4 if name == "WNet" else 1.6, ms=4 if annual else 5.5,
                    zorder=4 if name == "WNet" else 2)
        style(ax, f"({'abcdef'[k]}) {REGION_NAME[reg]}", "Year" if annual else "Month", "AR frequency (%)")
        ax.set_ylim(0, 1.22 * np.nanmax(y))
        if annual:
            ax.xaxis.set_major_locator(MultipleLocator(10))
        else:
            ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
        # each method against the mean of the other five: no method is the reference
        for name in ORDER:
            oth = np.mean([y[:, idx(o)] for o in METHODS if o != name], axis=0)
            rows.append([reg, "annual" if annual else "seasonal", STYLE[name][0],
                         f"{np.mean(y[:, idx(name)]):.2f}", f"{np.corrcoef(y[:, idx(name)], oth)[0, 1]:.3f}"])
            if name == "WNet":
                ax.text(0.02, 0.97, f"r = {np.corrcoef(y[:, idx(name)], oth)[0, 1]:.2f}", transform=ax.transAxes,
                        va="top", fontsize=FS_TICK, color=INK)
    line_legend(fig, ORDER)
    save(fig, "ar_paper_regional_annual" if annual else "ar_paper_regional_cycle")
    return rows


def write_tables(D, M, conf, reg_rows):
    p = os.path.join(HERE, "ar_paper_iou.csv")
    bands = [("20-60 both hemispheres", D.mask(*MID)), ("20-60N", D.mask(20, 60, False)),
             ("20-60S", D.mask(-60, -20, False)), ("60S-60N", D.mask(0, 60)),
             ("20S-20N", D.mask(0, 20)), ("poleward of 60", D.mask(60, 90))]
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "method_a", "method_b", "IoU"])
        for bname, m in bands:
            for a, b in combinations(ORDER, 2):
                w.writerow([bname, STYLE[a][0], STYLE[b][0], f"{D.iou(idx(a), idx(b), m):.3f}"])
    print(f"  wrote {p}")
    p = os.path.join(HERE, "ar_paper_confirmation.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "AR_frequency_pct_20_60", "mean_IoU_with_others_20_60",
                    "area_confirmed_by_2plus_others", "captures_area_of_3plus_others",
                    "pattern_r_with_mean_of_others_20_60", "pattern_r_with_mean_of_others_60S_60N"])
        mid = D.mask(*MID)
        print(f"\n{'method':<16}{'freq %':>8}{'mean IoU':>10}{'confirmed':>11}{'captures':>10}")
        for k, o in enumerate(ORDER):
            fr = 100 * (D.f[idx(o)] * D.w)[mid].sum() / D.N / D.w[mid].sum()
            mi = np.nanmean(M[k])
            fq = 100.0 * D.f / D.N
            oth = np.mean([fq[idx(q)] for q in METHODS if q != o], axis=0)
            r1, r2 = pattern_r(fq[idx(o)], oth, D.w, mid), pattern_r(fq[idx(o)], oth, D.w, D.mask(0, 60))
            w.writerow([STYLE[o][0], f"{fr:.2f}", f"{mi:.3f}", f"{conf[o][0]:.3f}", f"{conf[o][1]:.3f}",
                        f"{r1:.3f}", f"{r2:.3f}"])
            print(f"{'':<16}pattern r with the mean of the others: 20-60 deg {r1:.2f}, 60S-60N {r2:.2f}")
            print(f"{STYLE[o][0]:<16}{fr:>8.2f}{mi:>10.3f}{conf[o][0]:>11.3f}{conf[o][1]:>10.3f}")
    print(f"  wrote {p}")
    p = os.path.join(HERE, "ar_paper_regional.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "series", "method", "mean_AR_frequency_pct", "r_with_mean_of_other_methods"])
        w.writerows(reg_rows)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs=4, default=["MARS", "Mundhenk", "TempestLR", "WNet"],
                    help="the four methods of the map figure")
    ap.add_argument("--days-order", nargs="+", default=["MARS", "WNet", "GW", "Mundhenk", "TempestLR",
                                                        "Reid500"], help="panel order of Fig 6")
    ap.add_argument("--only-days", action="store_true", help="draw Fig 6 only")
    ap.add_argument("--remerge", action="store_true")
    args = ap.parse_args()
    if args.remerge or not os.path.exists(MERGED):
        merge()
    D = Data()
    fig_days(D, args.days_order)
    if args.only_days:
        return 0
    fig_frequency(D, args.show)
    M, conf = fig_agreement(D)
    fig_zonal_seasonal(D)
    rows = fig_regional(D, annual=False) + fig_regional(D, annual=True)
    write_tables(D, M, conf, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
