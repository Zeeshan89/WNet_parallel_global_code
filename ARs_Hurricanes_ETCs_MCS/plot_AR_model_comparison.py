#!/usr/bin/env python3
"""Professional multi-model atmospheric-river (AR) comparison figures.

Compares 4 RegCM-downscaled models (ACC, BCC, CNR, MPI) side-by-side (1 row x 4 cols):
  Fig 1 - Mean annual AR days (spatial maps)
  Fig 2 - Annual timeseries of domain-mean AR days (single axis, 4 lines + trends)
  Fig 3 - Weighted spatial KDE density of AR occurrence
  Fig 4 - Interannual variability (std-dev of annual AR days) spatial maps

Reads the per-model Analysis/ netCDF outputs and writes PNG + PDF (300 dpi) into
Figures_AR_comparison/.  Regional North America domain.

Data note: the single variable is always `AR_detection`; its units attribute reads
"probability" but in the yearly / average files the values are AR-day *counts*.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------------- config
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "Figures_AR_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]
COLORS = {"ACC": "#1f77b4", "BCC": "#d62728", "CNR": "#2ca02c", "MPI": "#9467bd", "MRI": "#8c564b", "NOR": "#e377c2"}
MARKERS = {"ACC": "o", "BCC": "s", "CNR": "^", "MPI": "D", "MRI": "v", "NOR": "<"}
PANEL_LETTERS = ["a", "b", "c", "d", "e", "f"]

# Regional North America domain (lon already in -180..180, no 0-360 fix needed)
EXTENT = [-138.75, -57.0, 18.0, 56.75]
PROJ = ccrs.PlateCarree()

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


# --------------------------------------------------------------------------- helpers
def analysis_path(model, kind):
    """Path to a per-model Analysis netCDF. kind in {AvgARDaysPerYear, ARDaysPerYear, daily, merged}."""
    return os.path.join(BASE, f"Outputs_RegCM_PR_{model}", "Analysis",
                        f"{model}_AR_detection_{kind}.nc")


def load_da(model, kind):
    """Open a per-model file and return (DataArray of AR_detection, lon, lat)."""
    ds = xr.open_dataset(analysis_path(model, kind))
    da = ds["AR_detection"]
    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)
    return da, lon, lat, ds


def add_geo_features(ax, panel_letter=None, title=None):
    """Standard modern GeoAxes styling: extent, coastlines, borders, states, gridlines."""
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="0.96", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(lw=0.5, color="0.2")
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.4")
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.6")
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    if panel_letter:
        ax.text(0.015, 0.97, f"({panel_letter})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))


def load_annual(model):
    """Load ARDaysPerYear, dropping incomplete years.

    `cdo yearsum` emits a trailing (and sometimes leading) partial-year step whose
    spatial total is a small fraction of a full year. Those are dropped so they do not
    create a spurious plunge in the timeseries or inflate the interannual std.
    Returns (arr[time,lat,lon], years, lon, lat).
    """
    da, lon, lat, ds = load_da(model, "ARDaysPerYear")
    arr = np.asarray(da.values, float)
    years = xr.decode_cf(ds)["time"].dt.year.values
    ds.close()
    totals = np.nansum(arr, axis=(1, 2))
    med = np.nanmedian(totals)
    keep = totals >= 0.5 * med          # drop incomplete years (< 50% of a typical year)
    return arr[keep], years[keep], lon, lat


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def map_row_figure(fields, lons, lats, cmap, vmin, vmax, cbar_label, suptitle, name):
    """Generic 1x4 GeoAxes map row with a single shared horizontal colorbar."""
    fig, axes = plt.subplots(1, 6, figsize=(20, 5.6),
                             subplot_kw=dict(projection=PROJ))
    mesh = None
    for i, (model, ax) in enumerate(zip(MODELS, axes)):
        mesh = ax.pcolormesh(lons[i], lats[i], fields[i], transform=PROJ,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             shading="auto", rasterized=True)
        add_geo_features(ax, panel_letter=PANEL_LETTERS[i], title=model)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    # shared horizontal colorbar under the row
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.028])
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", extend="max")
    cb.set_label(cbar_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.14, wspace=0.08)
    save_fig(fig, name)


# ------------------------------------------------------------------ Fig 1: mean AR days
def figure_mean_ar_days():
    fields, lons, lats = [], [], []
    for m in MODELS:
        da, lon, lat, ds = load_da(m, "AvgARDaysPerYear")
        fields.append(np.asarray(da.squeeze().values, float))
        lons.append(lon)
        lats.append(lat)
        ds.close()
    vmax = float(np.nanpercentile(np.concatenate([f.ravel() for f in fields]), 99.5))
    map_row_figure(fields, lons, lats, cmap="YlGnBu", vmin=0, vmax=100,
                   cbar_label="Mean AR days / year",
                   suptitle="Mean Annual Atmospheric-River Days (1980–2014)",
                   name="fig1_mean_AR_days")


# ------------------------------------------------------------ Fig 2: annual timeseries
def figure_annual_timeseries():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for m in MODELS:
        arr, years, lon, lat = load_annual(m)  # (time, lat, lon), incomplete years dropped
        # area weights by cos(latitude)
        w = np.cos(np.deg2rad(lat))
        w2d = np.broadcast_to(w[:, None], arr.shape[1:])
        series = np.array([
            np.nansum(arr[t] * w2d) / np.nansum(np.where(np.isfinite(arr[t]), w2d, 0.0))
            for t in range(arr.shape[0])
        ])

        ax.plot(years, series, color=COLORS[m], marker=MARKERS[m], ms=4.5,
                lw=1.8, label=m, zorder=3)
        # dashed linear trend
        if np.isfinite(series).sum() >= 2:
            coef = np.polyfit(years, series, 1)
            ax.plot(years, np.polyval(coef, years), color=COLORS[m],
                    lw=1.1, ls="--", alpha=0.6, zorder=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Domain-mean AR days / year")
    ax.set_title("Annual Domain-Mean Atmospheric-River Days by Model",
                 fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.margins(x=0.01)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig2_annual_timeseries")


# ------------------------------------------------------------- Fig 3: weighted KDE
def figure_spatial_kde():
    # common evaluation mesh spanning the domain
    gx = np.linspace(EXTENT[0], EXTENT[1], 200)
    gy = np.linspace(EXTENT[2], EXTENT[3], 200)
    mesh_lon, mesh_lat = np.meshgrid(gx, gy)
    eval_pts = np.vstack([mesh_lon.ravel(), mesh_lat.ravel()])

    dens_fields = []
    for m in MODELS:
        da, lon, lat, ds = load_da(m, "AvgARDaysPerYear")
        field = np.asarray(da.squeeze().values, float)
        ds.close()
        lon2d, lat2d = np.meshgrid(lon, lat)
        mask = np.isfinite(field) & (field > 0)
        coords = np.vstack([lon2d[mask], lat2d[mask]])
        weights = field[mask]
        kde = gaussian_kde(coords, weights=weights, bw_method=0.2)
        dens = kde(eval_pts).reshape(mesh_lat.shape)
        dmin, dmax = dens.min(), dens.max()
        dens = (dens - dmin) / (dmax - dmin) if dmax > dmin else dens * 0.0
        dens_fields.append(dens)

    fig, axes = plt.subplots(1, 6, figsize=(20, 5.6),
                             subplot_kw=dict(projection=PROJ))
    mesh = None
    for i, (m, ax) in enumerate(zip(MODELS, axes)):
        mesh = ax.pcolormesh(mesh_lon, mesh_lat, dens_fields[i], transform=PROJ,
                             cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
        add_geo_features(ax, panel_letter=PANEL_LETTERS[i], title=m)
    fig.suptitle("Weighted Spatial KDE of Atmospheric-River Occurrence",
                 fontsize=15, fontweight="bold", y=1.02)
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.028])
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Normalized AR density", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.14, wspace=0.08)
    save_fig(fig, "fig3_spatial_KDE")


# --------------------------------------------------------- Fig 4: interannual std
def figure_interannual_std():
    fields, lons, lats = [], [], []
    for m in MODELS:
        arr, years, lon, lat = load_annual(m)  # incomplete years dropped
        fields.append(np.nanstd(arr, axis=0))
        lons.append(lon)
        lats.append(lat)
    vmax = float(np.nanpercentile(np.concatenate([f.ravel() for f in fields]), 99.5))
    map_row_figure(fields, lons, lats, cmap="YlOrRd", vmin=0, vmax=vmax,
                   cbar_label="Std. dev. of annual AR days",
                   suptitle="Interannual Variability of Atmospheric-River Days",
                   name="fig4_interannual_std")


def main():
    print(f"Output directory: {OUT_DIR}")
    figure_mean_ar_days()
    figure_annual_timeseries()
    figure_spatial_kde()
    figure_interannual_std()
    print("Done.")


if __name__ == "__main__":
    main()
