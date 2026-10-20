#!/usr/bin/env python3
"""
plot_ar_regional.py — WNet AR detections against MARS in three AR landfall regions.

A global score mixes regions where an AR is a meaningful object with regions
where it is not: 60 % of the global false positives come from |lat| < 15, where
the two AR definitions differ most, and a quarter of the misses come from
poleward of 60. The regional evaluation therefore uses the three west-coast
landfall regions on which the AR literature concentrates. All three boxes are
the same size (35 deg x 60 deg) so that the panels compare like with like.

Each box runs from the open-ocean storm track to a few degrees inland of the
coast (120W, 5W, 70W), so it keeps the coastal strip but little land: 4 to 8 %
of the box area, against 18 % when the boxes reached 110W / 10E. The eastern
edges were set from the coastline and the land fraction, not from the scores.
WNet detects few ARs over land (its label is an absolute IVT threshold and IVT
decays inland), so the table also reports ocean and land separately; the box
score is close to the ocean score and must not be read as landfall skill.

Everything is read from ar_v2_merged.npz; no new pass over the data is needed.

  Fig 1  ar_regional_f1_maps      1 x 3  per-pixel F1 against MARS
  Fig 2  ar_regional_days_maps    2 x 3  mean AR days per year, MARS above WNet
  Fig 3  ar_regional_skill        1 x 3  IoU, F1, POD, FAR, annual and by season
  Fig 4  ar_regional_timeseries   1 x 3  annual AR days per year, MARS and WNet
  Fig 5  ar_regional_timeseries_anomaly  same, each series minus its own mean
  Table  ar_regional_skill.csv

A second region set, --set land, is the complement: from each coast 20 degrees
inland at the same latitudes (35 deg x 20 deg, 53 to 78 % land). It shows how
the agreement changes once the ARs cross the coast, and writes the same figures
and table to regional_coast_to_land/.

Usage
  python plot_ar_regional.py [--set ocean|land] [--only 1 3]
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_v2_common import (  # noqa: E402
    LV_DAYS_120, LV_F1, NYEAR, YEAR0, cos_w, disc, map_metrics)

PC = ccrs.PlateCarree()
INK = "#000000"            # black throughout; grey renders blurry in print
C_MARS, C_WNET = "#2a78d6", "#1baf7a"
FS_TITLE, FS_LAB, FS_TICK, FS_LEG = 13, 11.5, 10.5, 11

# name, lat_min, lat_max, lon_min, lon_max (degrees east, -180..180).
# "ocean": storm track up to a few degrees inland of the coast, all 35 x 60,
#          land is 8.1 / 8.2 / 4.2 % of the box.
# "land":  the complement, from the coast 20 degrees inland at the same latitudes,
#          all 35 x 20, land is 78 / 54 / 53 % (the rest is the North Sea, Biscay and
#          the western Mediterranean, and the Atlantic side of a narrow continent).
REGION_SETS = {
    "ocean": [("Northeast Pacific and US West Coast", 25, 60, -180, -120),
              ("North Atlantic and Western Europe", 30, 65, -65, -5),
              ("Southeast Pacific and Chile", -60, -25, -130, -70)],
    "land": [("Western North America", 25, 60, -125, -105),
             ("Western Europe", 30, 65, -10, 10),
             ("Southern South America", -60, -25, -75, -55)],
}
SET_DIR = {"ocean": "", "land": "regional_coast_to_land"}   # output folder under HERE
REGIONS = REGION_SETS["ocean"]      # main() rebinds this to the requested set
SUBSETS = [("Annual", "all_"), ("DJF", "sea_DJF_"), ("MAM", "sea_MAM_"),
           ("JJA", "sea_JJA_"), ("SON", "sea_SON_")]
PERIODS = [("1980-2005", "per_oop_"), ("2006-2024", "per_primary_")]
METRICS = [("IoU", "o", "#2a78d6"), ("F1", "D", "#1baf7a"),
           ("POD", "s", "#000000"), ("FAR", "^", "#eb6834")]
TAGS = "abcdef"


class Cache:
    """ar_v2_merged.npz with every map reordered onto -180..180 longitudes."""

    def __init__(self, path):
        self.z = np.load(path)
        self.lat = np.asarray(self.z["lat"], float)
        lon = ((np.asarray(self.z["lon"], float) + 180.0) % 360.0) - 180.0
        self.order = np.argsort(lon)
        self.lon = lon[self.order]
        self.w = cos_w(self.lat) * np.ones((1, len(self.lon)))

    def get(self, key):
        return np.asarray(self.z[key])[..., self.order]

    def box(self, la0, la1, lo0, lo1):
        return ((self.lat[:, None] >= la0) & (self.lat[:, None] <= la1) &
                (self.lon[None, :] >= lo0) & (self.lon[None, :] <= lo1))

    def skill(self, prefix, mask):
        return map_metrics(self.get(prefix + "tp"), self.get(prefix + "fp"),
                           self.get(prefix + "fn"), w=self.w, mask=mask)

    def land(self, mask):
        """Natural Earth 50m land, evaluated only inside `mask` (it is slow globally)."""
        import shapely
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        if not hasattr(self, "_land"):
            self._land = unary_union(list(shpreader.Reader(shpreader.natural_earth(
                resolution="50m", category="physical", name="land")).geometries()))
        lon2, lat2 = np.meshgrid(self.lon, self.lat)
        out = np.zeros_like(mask)
        out[mask] = shapely.contains_xy(self._land, lon2[mask], lat2[mask])
        return out

    def annual_days(self, mask):
        """Region-mean (cos-lat weighted) AR days in each year, WNet and MARS."""
        ww = self.w * mask
        f = lambda k: (self.get(k) * ww).sum((1, 2)) / ww.sum()   # noqa: E731
        return f("yr_days"), f("yr_days_t")


def save(fig, out):
    for ext in ("png", "pdf"):
        p = f"{out}.{ext}"
        fig.savefig(p, dpi=250 if ext == "png" else 300, bbox_inches="tight")
        print(f"  wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    plt.close(fig)


def tall():
    """True when the boxes of the active set are taller than wide (the land set)."""
    _, la0, la1, lo0, lo1 = REGIONS[0]
    return (lo1 - lo0) < (la1 - la0)


def region_map(ax, C, field, reg, levels, left_labels=True):
    _, la0, la1, lo0, lo1 = reg
    cmap, norm = disc(levels)
    ax.set_extent([lo0, lo1, la0, la1], crs=PC)
    m = ax.pcolormesh(C.lon, C.lat, field, cmap=cmap, norm=norm, transform=PC,
                      shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor=INK)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=INK)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=INK, alpha=0.35,
                      linestyle=(0, (4, 4)))
    # a 20 deg wide box needs 5 deg longitude ticks, or only one label survives
    gl.xlocator = MultipleLocator(5 if (lo1 - lo0) <= 30 else 10)
    gl.ylocator = MultipleLocator(5)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.xlabel_style = gl.ylabel_style = {"size": FS_TICK, "color": INK}
    return m


def shared_cbar(fig, mappable, levels, label, y=0.06, width=0.46, height=0.035,
                extend="both"):
    cax = fig.add_axes([0.5 - width / 2, y, width, height])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                      extend=extend, ticks=levels)
    cb.set_label(label, fontsize=FS_LAB, color=INK)
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK)
    cb.outline.set_edgecolor(INK)


# ---------------------------------------------------------------- figures
def fig_f1_maps(C, out):
    f1 = 2 * C.get("all_tp") / (2 * C.get("all_tp") + C.get("all_fp")
                                + C.get("all_fn") + 1e-12)
    if tall():      # 20 x 35 boxes: narrower figure, colour bar lower
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 6.6), subplot_kw={"projection": PC})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.19, wspace=0.22)
        cb = dict(y=0.075, height=0.026, width=0.62)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.05), subplot_kw={"projection": PC})
        fig.subplots_adjust(left=0.04, right=0.99, top=0.91, bottom=0.27, wspace=0.13)
        cb = dict(y=0.13, height=0.045)
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        m = region_map(ax, C, f1, reg, LV_F1)
        ax.set_title(f"({TAGS[i]}) {reg[0]}", fontsize=FS_TITLE, color=INK,
                     loc="left", pad=7)
    shared_cbar(fig, m, LV_F1, "Per-pixel F1 of WNet against MARS", **cb)
    save(fig, os.path.join(out, "ar_regional_f1_maps"))


def fig_days_maps_pairs(C, out, rows):
    """Tall boxes: one row, a MARS | WNet pair per region, region name above the pair."""
    fig = plt.figure(figsize=(15.5, 6.0))
    outer = GridSpec(1, 3, figure=fig, left=0.05, right=0.99, top=0.84, bottom=0.19,
                     wspace=0.16)
    for c, reg in enumerate(REGIONS):
        inner = outer[c].subgridspec(1, 2, wspace=0.06)
        for k, (label, field) in enumerate(rows):
            ax = fig.add_subplot(inner[k], projection=PC)
            m = region_map(ax, C, field, reg, LV_DAYS_120, left_labels=(k == 0))
            ax.set_title(label, fontsize=FS_LAB + 0.5, color=INK, pad=5)
            if k == 0:
                ax.annotate(f"({TAGS[c]}) {reg[0]}", xy=(0, 1), xycoords="axes fraction",
                            xytext=(0, 26), textcoords="offset points", ha="left",
                            va="bottom", fontsize=FS_TITLE, color=INK)
    shared_cbar(fig, m, LV_DAYS_120, "Mean AR days per year", y=0.075, height=0.03,
                extend="max")
    save(fig, os.path.join(out, "ar_regional_days_maps"))


def fig_days_maps(C, out):
    rows = [("MARS", C.get("all_ardays_t") / float(NYEAR)),
            ("WNet", C.get("all_ardays") / float(NYEAR))]
    if tall():
        return fig_days_maps_pairs(C, out, rows)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.0), subplot_kw={"projection": PC})
    fig.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.17,
                        wspace=0.13, hspace=0.10)
    for r, (label, field) in enumerate(rows):
        for c, reg in enumerate(REGIONS):
            m = region_map(axes[r, c], C, field, reg, LV_DAYS_120)
            if r == 0:
                axes[r, c].set_title(f"({TAGS[c]}) {reg[0]}", fontsize=FS_TITLE,
                                     color=INK, loc="left", pad=7)
        axes[r, 0].text(-0.17, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=FS_TITLE + 1, color=INK)
    # same 0 to 120 scale as the global days-per-year figure
    shared_cbar(fig, m, LV_DAYS_120, "Mean AR days per year", y=0.075, height=0.028,
                extend="max")
    save(fig, os.path.join(out, "ar_regional_days_maps"))


def fig_skill(C, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.14, wspace=0.07)
    ypos = np.arange(len(SUBSETS))[::-1]
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        mask = C.box(*reg[1:])
        for y, (_, pre) in zip(ypos, SUBSETS):
            s = C.skill(pre, mask)
            ax.plot([min(s[k] for k, _, _ in METRICS), max(s[k] for k, _, _ in METRICS)],
                    [y, y], color=INK, lw=0.8, zorder=1)
            for key, mk, col in METRICS:
                ax.plot(s[key], y, mk, color=col, ms=10 if key == "POD" else 8.5,
                        markeredgecolor="white", markeredgewidth=0.9,
                        zorder=3 if key == "POD" else 4)
        ax.axhline(ypos[0] - 0.5, color=INK, lw=0.6, ls=(0, (2, 3)))   # annual | seasons
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(SUBSETS) - 0.4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([s for s, _ in SUBSETS], fontsize=FS_LAB, color=INK)
        ax.set_xlabel("Score", fontsize=FS_LAB, color=INK)
        ax.set_title(f"({TAGS[i]}) {reg[0]}", fontsize=FS_TITLE, color=INK,
                     loc="left", pad=7)
        ax.tick_params(labelsize=FS_TICK, colors=INK)
        ax.grid(axis="x", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker=mk, ls="", color=col, ms=9, label=key)
                        for key, mk, col in METRICS],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               fontsize=FS_LEG, labelcolor=INK, handletextpad=0.3, columnspacing=2.2)
    save(fig, os.path.join(out, "ar_regional_skill"))


def fig_timeseries(C, out, anomaly=False):
    """Annual AR days per year. With anomaly=True each series is shown relative to
    its own 1980-2024 mean, which removes the constant offset between the two
    datasets and leaves the year-to-year variability they are compared on."""
    years = np.arange(YEAR0, YEAR0 + NYEAR)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.14, wspace=0.07)
    for i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        wn, ma = C.annual_days(C.box(*reg[1:]))
        if anomaly:
            wn, ma = wn - wn.mean(), ma - ma.mean()
            ax.axhline(0, color=INK, lw=0.7)
        ax.plot(years, ma, "-o", color=C_MARS, lw=1.8, ms=3.5)
        ax.plot(years, wn, "-o", color=C_WNET, lw=1.8, ms=3.5)
        ax.set_title(f"({TAGS[i]}) {reg[0]}", fontsize=FS_TITLE, color=INK,
                     loc="left", pad=7)
        ax.set_xlabel("Year", fontsize=FS_LAB, color=INK)
        ax.set_xlim(years[0] - 1, years[-1] + 1)
        ax.tick_params(labelsize=FS_TICK, colors=INK)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("AR days per year anomaly" if anomaly else "AR days per year",
                       fontsize=FS_LAB, color=INK)
    fig.legend(handles=[Line2D([], [], marker="o", color=C_MARS, lw=1.8, ms=4, label="MARS"),
                        Line2D([], [], marker="o", color=C_WNET, lw=1.8, ms=4, label="WNet")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=FS_LEG, labelcolor=INK, columnspacing=2.5)
    save(fig, os.path.join(out, "ar_regional_timeseries" + ("_anomaly" if anomaly else "")))


def write_table(C, out):
    years = np.arange(YEAR0, YEAR0 + NYEAR)
    p = os.path.join(out, "ar_regional_skill.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "lat_min", "lat_max", "lon_min", "lon_max", "subset",
                    "area_pct", "IoU", "F1", "POD", "FAR", "BIAS", "ARdays_MARS",
                    "ARdays_WNet", "interannual_r", "trend_MARS_d_per_dec",
                    "trend_WNet_d_per_dec"])
        print(f"\n{'region':<38}{'subset':<11}{'area':>6}{'IoU':>7}{'F1':>7}{'POD':>7}"
              f"{'FAR':>7}{'BIAS':>7}{'MARS d':>8}{'WNet d':>8}{'r':>7}")
        for reg in REGIONS:
            box = C.box(*reg[1:])
            isl = C.land(box)
            parts = [(n, p, box) for n, p in SUBSETS + PERIODS]
            parts += [("Ocean only", "all_", box & ~isl), ("Land only", "all_", box & isl)]
            for name, pre, mask in parts:
                s = C.skill(pre, mask)
                area = (C.w * mask).sum() / (C.w * box).sum() * 100
                extra, tail = [""] * 5, ""
                if pre == "all_":        # day counts and trends exist for the full record only
                    wn, ma = C.annual_days(mask)
                    r = np.corrcoef(wn, ma)[0, 1]
                    tm, tw = (np.polyfit(years, v, 1)[0] * 10 for v in (ma, wn))
                    extra = [f"{ma.mean():.1f}", f"{wn.mean():.1f}", f"{r:.3f}",
                             f"{tm:+.2f}", f"{tw:+.2f}"]
                    tail = f"{ma.mean():>8.1f}{wn.mean():>8.1f}{r:>7.3f}"
                w.writerow([reg[0], *reg[1:], name, f"{area:.1f}"] +
                           [f"{s[k]:.4f}" for k in ("IoU", "F1", "POD", "FAR", "BIAS")] + extra)
                print(f"{reg[0]:<38}{name:<11}{area:>5.0f}%{s['IoU']:>7.3f}{s['F1']:>7.3f}"
                      f"{s['POD']:>7.3f}{s['FAR']:>7.3f}{s['BIAS']:>7.3f}{tail}")
            print()
    print(f"  wrote {p}")


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=HERE)
    ap.add_argument("--set", choices=sorted(REGION_SETS), default="ocean",
                    help="ocean: storm track up to the coast; land: coast to 20 deg inland")
    ap.add_argument("--out", default=None, help="default: HERE, or HERE/<set folder>")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    args = ap.parse_args()

    global REGIONS
    REGIONS = REGION_SETS[args.set]
    if args.out is None:
        args.out = os.path.join(HERE, SET_DIR[args.set])
    os.makedirs(args.out, exist_ok=True)

    C = Cache(os.path.join(args.v2_dir, "ar_v2_merged.npz"))
    figs = {1: fig_f1_maps, 2: fig_days_maps, 3: fig_skill, 4: fig_timeseries,
            5: lambda c, o: fig_timeseries(c, o, anomaly=True)}
    for k in (args.only or sorted(figs)):
        figs[k](C, args.out)
    write_table(C, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
