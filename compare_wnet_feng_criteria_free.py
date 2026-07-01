#!/usr/bin/env python3
"""
Criteria-free WNet (id_only) vs Feng comparison over CONUS.
Metrics that need NO area/overlap/lifetime/ID thresholds:
  - Mean annual MCS-days (resolution-robust)
  - % of precipitation from MCS (ratio -> resolution/dt robust)
  - Seasonal cycle of MCS coverage (mean cell-hours), NORMALIZED to compare
    timing fairly across the different native grids / dt.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
WNET = sys.argv[1] if len(sys.argv) > 1 else "outputs_wnet_standard_id_only"  # relative to script dir
FENG = "../outputs_feng_standard"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "comparison_wnet_vs_feng"
EXTENT = [-110, -70, 25, 51]
MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def years_in(base):
    return set(d for d in os.listdir(base) if d.isdigit())

def latlon(da):
    la = "latitude" if "latitude" in da.coords else "lat"
    lo = "longitude" if "longitude" in da.coords else "lon"
    return da[lo].values, da[la].values

def mean_map(base, fname, years):
    arrs = [xr.open_dataarray(os.path.join(base, y, fname)) for y in years
            if os.path.exists(os.path.join(base, y, fname))]
    stk = xr.concat(arrs, dim="yr").mean("yr")
    return stk

def monthly_norm(base, years):
    rows = []
    for y in years:
        p = os.path.join(base, y, "monthly_mean_cell_hours.csv")
        if os.path.exists(p):
            s = pd.read_csv(p, index_col=0).iloc[:, 0]
            s.index = pd.PeriodIndex(s.index, freq="M").month
            rows.append(s.groupby(s.index).sum())
    m = pd.concat(rows, axis=1).mean(axis=1)
    m = m.reindex(range(1, 13))
    return m / m.mean()                       # normalize -> shape/timing only

def domain_mean(base, fname, years):
    vals = [float(xr.open_dataarray(os.path.join(base, y, fname)).mean())
            for y in years if os.path.exists(os.path.join(base, y, fname))]
    return float(np.mean(vals))

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs(OUTDIR, exist_ok=True)
    years = sorted(years_in(WNET) & years_in(FENG))
    print(f"Common years: {years[0]}..{years[-1]} (n={len(years)})")

    wd = mean_map(WNET, "yearly_MCS_days.nc", years)
    fd = mean_map(FENG, "yearly_MCS_days.nc", years)
    wp = mean_map(WNET, "percent_precip.nc", years)
    fp = mean_map(FENG, "percent_precip.nc", years)
    wcyc = monthly_norm(WNET, years)
    fcyc = monthly_norm(FENG, years)
    days = (domain_mean(WNET, "yearly_MCS_days.nc", years), domain_mean(FENG, "yearly_MCS_days.nc", years))
    pct  = (domain_mean(WNET, "percent_precip.nc", years), domain_mean(FENG, "percent_precip.nc", years))

    fig = plt.figure(figsize=(17, 9))
    def addmap(pos, da, title, vmax, cmap, label):
        ax = fig.add_subplot(2, 3, pos, projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        lon, lat = latlon(da)
        im = ax.pcolormesh(lon, lat, da.values, cmap=cmap, vmin=0, vmax=vmax,
                           shading="auto", transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.5); ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="gray")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, label=label)

    addmap(1, wd, "WNet — Mean annual MCS-days", 50, "YlOrRd", "days/yr")
    addmap(2, fd, "Feng — Mean annual MCS-days", 50, "YlOrRd", "days/yr")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(range(1, 13), wcyc.values, "-o", label="WNet", color="tab:blue")
    ax3.plot(range(1, 13), fcyc.values, "-s", label="Feng", color="tab:red")
    ax3.set_xticks(range(1, 13)); ax3.set_xticklabels(MON, rotation=45, fontsize=8)
    ax3.set_title("Seasonal cycle of MCS coverage (normalized)", fontsize=11)
    ax3.set_ylabel("coverage / annual mean"); ax3.grid(alpha=0.3); ax3.legend()

    addmap(4, wp, "WNet — % precip from MCS", 60, "Blues", "%")
    addmap(5, fp, "Feng — % precip from MCS", 60, "Blues", "%")

    ax6 = fig.add_subplot(2, 3, 6)
    x = np.arange(2)
    ax6.bar(x - 0.2, [days[0], pct[0]], 0.4, label="WNet", color="tab:blue")
    ax6.bar(x + 0.2, [days[1], pct[1]], 0.4, label="Feng", color="tab:red")
    ax6.set_xticks(x); ax6.set_xticklabels(["MCS-days/yr", "% precip"])
    ax6.set_title("Domain-mean (CONUS)", fontsize=11)
    for i, (a, b) in enumerate([days, pct]):
        ax6.text(i - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax6.text(i + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax6.grid(alpha=0.3, axis="y"); ax6.legend()

    fig.suptitle(f"WNet (unsupervised, no MCS criteria) vs Feng — criteria-free metrics, CONUS {years[0]}–{years[-1]}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, "criteria_free_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", os.path.join(here, out))
    print(f"Domain-mean MCS-days: WNet={days[0]:.1f}  Feng={days[1]:.1f}")
    print(f"Domain-mean %precip : WNet={pct[0]:.1f}  Feng={pct[1]:.1f}")

if __name__ == "__main__":
    main()
