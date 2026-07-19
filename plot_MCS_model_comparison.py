#!/usr/bin/env python3
"""Professional multi-model MCS (mesoscale convective system) comparison figures.

Reads the precip-based WNet MCS outputs produced by mcs-wnet-newfinalcode.py, i.e.
    outputs_wnet_PR_<MODEL>/<YEAR>/{yearly_MCS_days,freq_hours,hours_per_active_day,
                                    avg_precip,percent_precip}.nc
    outputs_wnet_PR_<MODEL>/<YEAR>/{monthly_domain_cell_hours.csv, summary.csv, ...}
and makes side-by-side (1 row x N models) comparison figures over the CONUS domain.

Models are auto-detected: any outputs_wnet_PR_<MODEL>/ that has at least one
processed year is included, so this runs now with whatever models are ready
(ACC/BCC) and again later once the rest finish — just re-run it.

Set WNET_MCS_PREFIX=outputs_wnet_full to plot the BT-era analysis instead.

Each per-model map is the multi-year mean over whatever years that model has; the
panel subtitle records the year span and count so partial models are clearly labelled.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ----------------------------------------------------------------------------- config
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "Figures_MCS_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# Preferred display order; only those with output are actually plotted.
MODEL_ORDER = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]
COLORS = {"ACC": "#1f77b4", "BCC": "#d62728", "CNR": "#2ca02c",
          "MPI": "#9467bd", "MRI": "#ff7f0e", "NOR": "#8c564b"}
MARKERS = {"ACC": "o", "BCC": "s", "CNR": "^", "MPI": "D", "MRI": "v", "NOR": "P"}
PANEL_LETTERS = list("abcdefgh")

# Map extent is derived from the actual data bounds at plot time (see data_extent),
# so the full available lat/lon field is shown edge-to-edge with nothing cut off.
PROJ = ccrs.PlateCarree()

MON_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ------------------------------------------------------------------------ global style
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.3",
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# --------------------------------------------------------------------------- discovery
# Precip-based MCS analysis from mcs-wnet-newfinalcode.py (outputs_wnet_PR_<MODEL>).
# The BT-era analysis is outputs_wnet_full_<MODEL>; set WNET_MCS_PREFIX to compare.
_PREFIX = os.environ.get("WNET_MCS_PREFIX", "outputs_wnet_PR")


def model_dir(model):
    return os.path.join(BASE, f"{_PREFIX}_{model}")


def year_dirs(model):
    """Sorted list of (year:int, path) for a model that actually contains a map file."""
    out = []
    for d in sorted(glob.glob(os.path.join(model_dir(model), "*"))):
        base = os.path.basename(d)
        if os.path.isdir(d) and base.isdigit() and os.path.exists(os.path.join(d, "yearly_MCS_days.nc")):
            out.append((int(base), d))
    return out


def available_models():
    return [m for m in MODEL_ORDER if year_dirs(m)]


# --------------------------------------------------------------------------- loaders
def load_map_mean(model, filename):
    """Multi-year mean of a single-variable map file. Returns (field2d, lon, lat, years)."""
    stack, lon, lat, years = [], None, None, []
    for yr, d in year_dirs(model):
        fp = os.path.join(d, filename)
        if not os.path.exists(fp):
            continue
        ds = xr.open_dataset(fp)
        var = list(ds.data_vars)[0]
        arr = np.asarray(ds[var].values, float)
        if lon is None:
            lon = ds["longitude"].values.astype(float)
            lat = ds["latitude"].values.astype(float)
        ds.close()
        stack.append(arr)
        years.append(yr)
    if not stack:
        return None, None, None, []
    field = np.nanmean(np.stack(stack, axis=0), axis=0)
    return field, lon, lat, sorted(years)


def load_annual_domain_mean(model, filename):
    """Per-year spatial mean of a map field. Returns (years, values)."""
    years, vals = [], []
    for yr, d in year_dirs(model):
        fp = os.path.join(d, filename)
        if not os.path.exists(fp):
            continue
        ds = xr.open_dataset(fp)
        var = list(ds.data_vars)[0]
        vals.append(float(np.nanmean(ds[var].values)))
        ds.close()
        years.append(yr)
    order = np.argsort(years)
    return np.array(years)[order], np.array(vals)[order]


def load_monthly_climatology(model, csv_name, value_col):
    """Mean seasonal cycle (12 months) from a monthly CSV, dropping cross-year spillover.

    Each year's CSV may carry a small spillover month (e.g. next-Jan with ~6 hours);
    we keep only rows whose calendar year matches the folder year before averaging.
    Returns a length-12 array (Jan..Dec), NaN where a month is absent.
    """
    per_month = {m: [] for m in range(1, 13)}
    for yr, d in year_dirs(model):
        fp = os.path.join(d, csv_name)
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        tcol = df.columns[0]
        t = pd.to_datetime(df[tcol].astype(str), format="%Y-%m", errors="coerce")
        df = df.assign(_yr=t.dt.year, _mo=t.dt.month)
        df = df[df["_yr"] == yr]                 # drop spillover into next year
        for _, row in df.iterrows():
            if np.isfinite(row["_mo"]):
                per_month[int(row["_mo"])].append(float(row[value_col]))
    clim = np.array([np.mean(per_month[m]) if per_month[m] else np.nan for m in range(1, 13)])
    return clim


def load_summary(model):
    """Concatenate per-year summary.csv track-property tables for a model."""
    frames = []
    for yr, d in year_dirs(model):
        fp = os.path.join(d, "summary.csv")
        if os.path.exists(fp):
            try:
                frames.append(pd.read_csv(fp))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- plotting
def data_extent(lons, lats):
    """Exact [lon0, lon1, lat0, lat1] bounds spanning all available model grids."""
    lo = [l for l in lons if l is not None]
    la = [a for a in lats if a is not None]
    return [min(l.min() for l in lo), max(l.max() for l in lo),
            min(a.min() for a in la), max(a.max() for a in la)]


def add_geo_features(ax, extent, panel_letter=None, title=None):
    ax.set_extent(extent, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="0.96", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(lw=0.5, color="0.2")
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.4")
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.6")
    # Draw lat/lon tick labels only — no graticule lines (they tile the map into
    # boxes that read as a projection artifact over this smaller CONUS domain).
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    if title:
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=6)
    if panel_letter:
        ax.text(0.015, 0.97, f"({panel_letter})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def panel_subtitle(model, years):
    if not years:
        return model
    if years[0] == years[-1]:
        return f"{model}  ({years[0]})"
    partial = "" if len(years) == (years[-1] - years[0] + 1) else "*"
    return f"{model}  ({years[0]}–{years[-1]}, {len(years)} yr{partial})"


def map_row_figure(models, filename, cmap, cbar_label, suptitle, name,
                   vmin=0, vmax_pct=99.5):
    fields, lons, lats, subs = [], [], [], []
    for m in models:
        f, lon, lat, yrs = load_map_mean(m, filename)
        fields.append(f)
        lons.append(lon)
        lats.append(lat)
        subs.append(panel_subtitle(m, yrs))
    valid = [f for f in fields if f is not None]
    if not valid:
        print(f"  (skip {name}: no data)")
        return
    vmax = float(np.nanpercentile(np.concatenate([f.ravel() for f in valid]), vmax_pct))
    extent = data_extent(lons, lats)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.4),
                             subplot_kw=dict(projection=PROJ))
    if n == 1:
        axes = [axes]
    mesh = None
    for i, (m, ax) in enumerate(zip(models, axes)):
        if fields[i] is not None:
            # gouraud interpolates between cell centres -> smooth field, no blocky
            # per-cell "grid box" tiling from the coarse 0.25-deg grid.
            mesh = ax.pcolormesh(lons[i], lats[i], fields[i], transform=PROJ,
                                 cmap=cmap, vmin=vmin, vmax=vmax,
                                 shading="gouraud", rasterized=True)
        add_geo_features(ax, extent, panel_letter=PANEL_LETTERS[i], title=subs[i])
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    cbar_ax = fig.add_axes([0.30, 0.05, 0.42, 0.028])
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", extend="max")
    cb.set_label(cbar_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.13, wspace=0.08)
    save_fig(fig, name)


# ------------------------------------------------------------------ map figures (1..5)
def figures_maps(models):
    map_row_figure(models, "yearly_MCS_days.nc", "YlOrRd",
                   "Mean MCS days / year", "Mean Annual MCS Days",
                   "fig1_mean_MCS_days")
    map_row_figure(models, "freq_hours.nc", "YlOrRd",
                   "MCS hours / year", "Mean Annual MCS Frequency (hours)",
                   "fig2_freq_hours")
    map_row_figure(models, "hours_per_active_day.nc", "YlOrRd",
                   "Hours per active day", "MCS Hours per Active Day",
                   "fig3_hours_per_active_day")
    map_row_figure(models, "percent_precip.nc", "YlOrRd",
                   "% of precipitation from MCS", "MCS Contribution to Total Precipitation",
                   "fig4_percent_precip")
    map_row_figure(models, "avg_precip.nc", "YlOrRd",
                   "Mean precip during MCS (mm/step)", "Mean Precipitation during MCS",
                   "fig5_avg_precip")


# --------------------------------------------------------- Fig 6: annual timeseries
def figure_annual_timeseries(models):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    plotted = False
    for m in models:
        years, vals = load_annual_domain_mean(m, "yearly_MCS_days.nc")
        if years.size == 0:
            continue
        plotted = True
        ax.plot(years, vals, color=COLORS[m], marker=MARKERS[m], ms=5,
                lw=1.8, label=m, zorder=3)
        if years.size >= 3:
            coef = np.polyfit(years, vals, 1)
            ax.plot(years, np.polyval(coef, years), color=COLORS[m],
                    lw=1.1, ls="--", alpha=0.55, zorder=2)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Year")
    ax.set_ylabel("Domain-mean MCS days / year")
    ax.set_title("Annual Domain-Mean MCS Days by Model", fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.margins(x=0.02)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig6_annual_timeseries")


# ---------------------------------------------------- Fig 7: seasonal cycle
def figure_seasonal_cycle(models):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(1, 13)
    plotted = False
    for m in models:
        clim = load_monthly_climatology(m, "monthly_domain_cell_hours.csv", "cell_hours")
        if np.all(np.isnan(clim)):
            continue
        plotted = True
        ax.plot(x, clim, color=COLORS[m], marker=MARKERS[m], ms=5, lw=1.9, label=m, zorder=3)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xticks(x)
    ax.set_xticklabels(MON_NAMES)
    ax.set_xlabel("Month")
    ax.set_ylabel("Domain MCS cell-hours (per month)")
    ax.set_title("Mean Seasonal Cycle of MCS Activity", fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig7_seasonal_cycle")


# --------------------------------- Figs 9-10: monthly bar climatologies (1 x N)
def bar_row_figure(models, csv_name, value_col, ylabel, suptitle, name):
    """One monthly-climatology bar panel per model (12 bars, mean across years)."""
    clims = {m: load_monthly_climatology(m, csv_name, value_col) for m in models}
    models_with = [m for m in models if not np.all(np.isnan(clims[m]))]
    if not models_with:
        print(f"  (skip {name}: no data)")
        return
    ymax = np.nanmax([np.nanmax(clims[m]) for m in models_with]) * 1.08
    x = np.arange(1, 13)

    n = len(models_with)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for i, (m, ax) in enumerate(zip(models_with, axes)):
        ax.bar(x, clims[m], color=COLORS[m], edgecolor="white", linewidth=0.4, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MON_NAMES, rotation=45, fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_title(f"({PANEL_LETTERS[i]}) {m}", fontsize=12.5, fontweight="bold")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, name)


def figure_monthly_mean_cell_hours(models):
    bar_row_figure(models, "monthly_mean_cell_hours.csv", "hours_per_cell",
                   "Hours / month per grid cell (mean)",
                   "Monthly Mean Cell-Hours by Model (averaged across years)",
                   "fig9_monthly_mean_cell_hours")


def figure_monthly_storm_hours_id(models):
    bar_row_figure(models, "monthly_mcs_hours_by_id.csv", "hours",
                   "Storm-hours / month",
                   "Monthly MCS Storm-Hours (ID-based) by Model (averaged across years)",
                   "fig10_monthly_storm_hours_id")


# ------------------------------------------- Fig 8: track-property distributions
def figure_track_properties(models):
    # (col, ylabel, title, log?, moving_only?)
    # Most WNet "tracks" are single 6-h timesteps (Duration=6h, path=0, speed=0), so the
    # distribution lives in the tail: use log axes + 5-95th-pct whiskers, and restrict the
    # motion metrics (path, speed) to multi-timestep "moving" tracks (Duration > 6h).
    metrics = [("Duration_hrs", "Track duration (hours)", "Duration", True, False),
               ("Max_Area_cells", "Max area (grid cells)", "Max area", True, False),
               ("Path_Length_km", "Path length (km)", "Path (moving)", True, True),
               ("Avg_Speed_kmph", "Mean speed (km/h)", "Speed (moving)", False, True)]
    data = {m: load_summary(m) for m in models}
    models_with = [m for m in models if not data[m].empty]
    if not models_with:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for j, (col, label, title, logy, moving) in enumerate(metrics):
        ax = axes[j]
        series = []
        for m in models_with:
            df = data[m]
            if moving and "Duration_hrs" in df:
                df = df[df["Duration_hrs"] > 6.0]
            v = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
            series.append(v)
        # whiskers to 5th/95th pct capture the tail without plotting 100k fliers
        bp = ax.boxplot(series, showfliers=False, patch_artist=True, widths=0.6,
                        whis=(5, 95), medianprops=dict(color="black", lw=1.4))
        for patch, m in zip(bp["boxes"], models_with):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.65)
        ax.set_xticklabels(models_with, rotation=0)
        ax.set_ylabel(label)
        ax.set_title(f"({PANEL_LETTERS[j]}) {title}", fontsize=12, fontweight="bold")
        allv = np.concatenate([s for s in series if s.size]) if any(s.size for s in series) else np.array([])
        if logy and allv.size and np.nanmax(allv) > 0:
            ax.set_yscale("log")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("MCS Track-Property Distributions by Model  "
                 "(box = IQR, whiskers = 5–95th pct; path/speed for tracks > 6 h)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig8_track_properties")


def main():
    models = available_models()
    if not models:
        print("No processed MCS model outputs found under outputs_wnet_standard_*/")
        return
    print(f"Models found: {models}")
    print(f"Output directory: {OUT_DIR}")
    figures_maps(models)
    figure_annual_timeseries(models)
    figure_seasonal_cycle(models)
    figure_monthly_mean_cell_hours(models)
    figure_monthly_storm_hours_id(models)
    figure_track_properties(models)
    print("Done.")


if __name__ == "__main__":
    main()
