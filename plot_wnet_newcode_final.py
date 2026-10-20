#!/usr/bin/env python3
"""
Expanded plotting script (backward compatible).

Existing plots preserved:
- Mean annual MCS days map
- Mean annual MCS hours map
- Mean precip during MCS map
- % precip from MCS map
- Monthly bars: (1) storm-hours by ID, (2) domain presence hours, (3) domain cell-hours
- Annual series: mean days per grid cell (domain mean)

NEW (optional, if files exist):
- Map: mean hours per active day (hours/day when MCS occurs) from hours_per_active_day.nc
       or computed as mean_hours/mean_days (masked where days=0).
- Monthly bar: monthly_mean_cell_hours.csv (mean over grid cells; more comparable scale)
- Annual series:
    * domain-mean hours/year (from freq_hours)
    * domain-mean hours per active day (from hours_per_active_day)
    * domain-wide days/year from domain_MCS_days_by_year.csv (if present)

Jan->Dec ordering enforced.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

# White->blue->green->yellow->red sequential colormap: 0 is WHITE so empty/low
# background stays clean (unlike turbo/viridis, whose low end is dark).
WBGYR = LinearSegmentedColormap.from_list("wbgyr", [
    (0.00, "#ffffff"), (0.08, "#d8ecf9"), (0.18, "#a6d5ee"), (0.30, "#5fb0db"),
    (0.38, "#3f93c6"), (0.46, "#2fa08c"), (0.56, "#5cbf5a"), (0.64, "#a5d63f"),
    (0.72, "#f2e534"), (0.82, "#f6a623"), (0.91, "#e8551f"), (1.00, "#b00000"),
], N=256)
OVER_COLOR = "#ff00d4"   # magenta "over" arrow (values above the top level)


def _disc(levels, base=WBGYR):
    """Discrete banded (cmap, norm) from `levels`: white below levels[0], magenta
    above levels[-1] — the NCL-style stepped colorbar (use extend='both', uniform
    spacing, ticks=levels on the colorbar)."""
    cols = base(np.linspace(0.06, 1.0, len(levels) - 1))
    cmap = ListedColormap(cols)
    cmap.set_under("white")
    cmap.set_over(OVER_COLOR)
    return cmap, BoundaryNorm(levels, cmap.N)

# ------------------------------------------
# Helpers
# ------------------------------------------
CALENDAR_ORDER = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def regrid_mask_to_target_grid(fmask_dataset, var_name, target_lat2d, target_lon2d):
    lat_mask = fmask_dataset['latitude'].values
    lon_mask = fmask_dataset['longitude'].values
    msk_data = fmask_dataset[var_name].isel(time=0).values

    if lat_mask[0] > lat_mask[-1]:
        lat_mask = lat_mask[::-1]
        msk_data = msk_data[::-1, :]

    interp = RegularGridInterpolator(
        (lat_mask, lon_mask), msk_data,
        bounds_error=False, fill_value=np.nan
    )
    points = np.stack([target_lat2d.ravel(), target_lon2d.ravel()], axis=-1)
    return interp(points).reshape(target_lat2d.shape)

def plot_spatial_map(data, lat2d, lon2d, title, vmin, vmax, cmap, label, mask=None,
                     save_path=None, levels=None):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if mask is not None:
        data = np.ma.masked_where(np.isnan(mask), data)

    cbar_kw = dict(orientation='vertical', pad=0.02, extend='neither')
    if levels is not None:
        # discrete banded fill + NCL-style stepped colorbar (white under, magenta over)
        dcmap, norm = _disc(levels)
        im = ax.pcolormesh(lon2d, lat2d, data, shading='auto', cmap=dcmap,
                           norm=norm, transform=ccrs.PlateCarree())
        cbar_kw.update(extend='both', ticks=levels, spacing='uniform')
    else:
        im = ax.pcolormesh(lon2d, lat2d, data, shading='auto', cmap=cmap,
                           transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_extent([np.nanmin(lon2d), np.nanmax(lon2d), np.nanmin(lat2d), np.nanmax(lat2d)])

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, **cbar_kw)
    cbar.set_label(label)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_bar_months(series, title, ylabel, save_path=None, ylim=None):
    """series indexed by Jan..Dec in order"""
    plt.figure(figsize=(10, 4))
    series.plot(kind='bar', color='steelblue')
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def safe_open_dataarray(path):
    """Open a DataArray robustly; fallback to opening Dataset and extracting first var."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        da = xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)
        da = next(iter(ds.data_vars.values()))
    if np.issubdtype(da.dtype, np.timedelta64):
        da = da / np.timedelta64(1, 'h')
    return da

def _read_monthly_series(csv_path):
    """
    Reads a 1-col CSV like:
      index: YYYY-MM (or datetime)
      value: hours
    Returns pandas Series with datetime index.
    """
    s = pd.read_csv(csv_path, index_col=0).iloc[:, 0]
    idx = pd.to_datetime(s.index.astype(str), errors="coerce")
    s = s[~idx.isna()]
    s.index = idx[~idx.isna()]
    return s

def monthly_avg_by_calendar_month(series_list):
    """
    series_list: list of Series indexed by datetime (monthly timestamps)
    Returns Series indexed by Jan..Dec, averaged across all years.
    """
    if not series_list:
        return None

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    by_month = df.groupby(df.index.month).mean().mean(axis=1)
    by_month = by_month.reindex(range(1, 13), fill_value=0)

    by_month.index = [CALENDAR_ORDER[m-1] for m in by_month.index]
    by_month.index = pd.CategoricalIndex(by_month.index, categories=CALENDAR_ORDER, ordered=True)
    by_month = by_month.sort_index()
    return by_month

def _domain_mean(da):
    """mean over spatial dims only."""
    spatial_dims = [d for d in da.dims if d != "time"]
    return da.mean(dim=spatial_dims, skipna=True)

# ------------------------------------------
# Aggregation & plotting (main)
# ------------------------------------------
def load_outputs_and_plot(
    base_output_dir,
    dataset_name=None,
    out_plot_dir=None,
    landmask_path="ERA5_landmask.nc",
):
    """
    Expects each year folder under base_output_dir to contain:
      core:
        - yearly_MCS_days.nc
        - freq_hours.nc
        - avg_precip.nc
        - percent_precip.nc
      monthly (optional):
        - monthly_mcs_hours_by_id.csv
        - monthly_domain_presence_hours.csv OR monthly_counts.csv
        - monthly_domain_cell_hours.csv
        - monthly_mean_cell_hours.csv (NEW optional)
      other (optional):
        - hours_per_active_day.nc (NEW optional)
        - domain_MCS_days_by_year.csv (NEW optional; per-year folder or root)
    """

    yearly_days_maps = []
    freq_hours_maps = []
    avg_precips = []
    percent_precips = []
    hours_per_active_day_maps = []  # NEW optional

    monthly_id_hours_all = []
    monthly_presence_hours_all = []
    monthly_cell_hours_all = []
    monthly_mean_cell_hours_all = []  # NEW optional

    annual_domain_days_rows = []  # NEW optional

    sample_lat = sample_lon = None
    lat2d = lon2d = None

    year_dirs = [d for d in sorted(os.listdir(base_output_dir)) if os.path.isdir(os.path.join(base_output_dir, d))]

    for year_dir in year_dirs:
        ypath = os.path.join(base_output_dir, year_dir)
        print(f"Loading {year_dir}...")

        try:
            yearly_days = safe_open_dataarray(os.path.join(ypath, "yearly_MCS_days.nc"))
            freq_hours  = safe_open_dataarray(os.path.join(ypath, "freq_hours.nc"))
            avg_precip  = safe_open_dataarray(os.path.join(ypath, "avg_precip.nc"))
            percent     = safe_open_dataarray(os.path.join(ypath, "percent_precip.nc"))
        except Exception as e:
            print(f"  Skipping {year_dir} (missing core maps): {e}")
            continue

        yearly_days_maps.append(yearly_days)
        freq_hours_maps.append(freq_hours)
        avg_precips.append(avg_precip)
        percent_precips.append(percent)

        # optional: hours_per_active_day.nc
        p_hpad = os.path.join(ypath, "hours_per_active_day.nc")
        if os.path.exists(p_hpad):
            try:
                hours_per_active_day_maps.append(safe_open_dataarray(p_hpad))
            except Exception as e:
                print(f"  Warning reading {p_hpad}: {e}")

        # monthly (1) ID-based storm-hours
        p = os.path.join(ypath, "monthly_mcs_hours_by_id.csv")
        if os.path.exists(p):
            try:
                monthly_id_hours_all.append(_read_monthly_series(p))
            except Exception as e:
                print(f"  Warning reading {p}: {e}")

        # monthly (2) domain presence hours
        p = os.path.join(ypath, "monthly_domain_presence_hours.csv")
        if os.path.exists(p):
            try:
                monthly_presence_hours_all.append(_read_monthly_series(p))
            except Exception as e:
                print(f"  Warning reading {p}: {e}")
        else:
            p2 = os.path.join(ypath, "monthly_counts.csv")
            if os.path.exists(p2):
                try:
                    monthly_presence_hours_all.append(_read_monthly_series(p2))
                except Exception as e:
                    print(f"  Warning reading {p2}: {e}")

        # monthly (3) domain cell-hours
        p = os.path.join(ypath, "monthly_domain_cell_hours.csv")
        if os.path.exists(p):
            try:
                monthly_cell_hours_all.append(_read_monthly_series(p))
            except Exception as e:
                print(f"  Warning reading {p}: {e}")

        # monthly (4) mean cell-hours (NEW)
        p = os.path.join(ypath, "monthly_mean_cell_hours.csv")
        if os.path.exists(p):
            try:
                monthly_mean_cell_hours_all.append(_read_monthly_series(p))
            except Exception as e:
                print(f"  Warning reading {p}: {e}")

        # per-year domain days (NEW; optional)
        p = os.path.join(ypath, "domain_MCS_days_by_year.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if {"year", "domain_MCS_days"}.issubset(df.columns) and len(df) >= 1:
                    annual_domain_days_rows.append(df.iloc[0].to_dict())
            except Exception as e:
                print(f"  Warning reading {p}: {e}")

        # lat/lon setup once
        if sample_lat is None:
            lat_name = "lat" if "lat" in yearly_days.coords else "latitude"
            lon_name = "lon" if "lon" in yearly_days.coords else "longitude"
            sample_lat = yearly_days[lat_name].values
            sample_lon = yearly_days[lon_name].values
            lon2d, lat2d = np.meshgrid(sample_lon, sample_lat)

    if not yearly_days_maps:
        print("No yearly maps loaded. Exiting.")
        return

    # output folder naming
    if out_plot_dir is None:
        tag = dataset_name if dataset_name else os.path.basename(os.path.abspath(base_output_dir))
        out_plot_dir = f"summary_plots_{tag}"
    os.makedirs(out_plot_dir, exist_ok=True)

    # Multi-year means
    mean_days = xr.concat(yearly_days_maps, dim="year").mean(dim="year", skipna=True)
    mean_hours = xr.concat(freq_hours_maps, dim="year").mean(dim="year", skipna=True)
    mean_avg_precip = xr.concat(avg_precips, dim="year").mean(dim="year", skipna=True)
    mean_percent_precip = xr.concat(percent_precips, dim="year").mean(dim="year", skipna=True)
    mean_days_from_hours = mean_hours / 24.0
    mean_days_from_hours = mean_days_from_hours.where(mean_days_from_hours > 0)

    # mean hours per active day (NEW): prefer file if available; else compute from mean maps
    if len(hours_per_active_day_maps) > 0:
        mean_hpad = xr.concat(hours_per_active_day_maps, dim="year").mean(dim="year", skipna=True)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_hpad = xr.where(mean_days > 0, mean_hours / mean_days, np.nan)

    # landmask (optional)
    try:
        if landmask_path and os.path.exists(landmask_path):
            fmask = xr.open_dataset(landmask_path)
            if "longitude" in fmask.coords and float(fmask.longitude.max()) > 180:
                fmask = fmask.assign_coords(longitude=(((fmask.longitude + 180) % 360) - 180)).sortby("longitude")
            # NOTE: many landmasks won’t have 'ppt'; keep your existing var_name
            msk = regrid_mask_to_target_grid(fmask, var_name="ppt", target_lat2d=lat2d, target_lon2d=lon2d)
        else:
            msk = None
    except Exception as e:
        print("Warning: could not load/regrid landmask:", e)
        msk = None

    # ---- Spatial maps ----
    plot_spatial_map(
        mean_days.values, lat2d, lon2d,
        title="Mean Annual MCS Days (gated)",
        vmin=0,
        vmax=np.nanpercentile(mean_days.values, 99),
        cmap=WBGYR,
        label="Days/year",
        mask=msk,
        levels=[2, 5, 10, 15, 20, 30, 40, 50, 70, 100],
        save_path=os.path.join(out_plot_dir, "mean_annual_mcs_days.png")
    )

    plot_spatial_map(
        mean_hours.values, lat2d, lon2d,
        title="Mean Annual MCS Hours (gated)",
        vmin=0,
        vmax=np.nanpercentile(mean_hours.values, 99),
        cmap=WBGYR,
        label="Hours/year",
        mask=msk,
        levels=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000],
        save_path=os.path.join(out_plot_dir, "mean_annual_mcs_hours.png")
    )

    plot_spatial_map(
        mean_hpad.values, lat2d, lon2d,
        title="Mean Hours per Active MCS Day (hours/day) (gated)",
        vmin=0,
        vmax=np.nanpercentile(mean_hpad.values, 99),
        cmap=WBGYR,
        label="Hours per MCS day",
        mask=msk,
        levels=[6, 7, 8, 9, 10, 12, 14, 16, 20],
        save_path=os.path.join(out_plot_dir, "mean_hours_per_active_day.png")
    )

    plot_spatial_map(
        mean_avg_precip.values, lat2d, lon2d,
        title="Mean Precipitation During MCS (gated)",
        vmin=0,
        vmax=np.nanpercentile(mean_avg_precip.values, 99),
        cmap=WBGYR,
        label="(native precip units per step/rate)",
        mask=msk,
        levels=[0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12],
        save_path=os.path.join(out_plot_dir, "mean_mcs_precip.png")
    )

    plot_spatial_map(
        mean_percent_precip.values, lat2d, lon2d,
        title="% of Precipitation from MCS (gated)",
        vmin=0,
        vmax=50,
        cmap=WBGYR,
        label="%",
        mask=msk,
        levels=[1, 2, 5, 10, 15, 20, 25, 30, 40],
        save_path=os.path.join(out_plot_dir, "percent_precip_from_mcs.png")
    )
    plot_spatial_map(
    mean_days_from_hours.values,
    lat2d,
    lon2d,
    title="Mean Annual MCS Days (from Hours / 24)",
    vmin=0,
    vmax=50,
    cmap=WBGYR,
    label="Equivalent days/year",
    mask=msk,
    levels=[2, 5, 10, 15, 20, 25, 30, 40, 50],
    save_path=os.path.join(out_plot_dir, "mean_annual_mcs_days_from_hours.png")
)

    # ---- Monthly bars ----
    # (A) ID-based storm-hours
    by_month_id = monthly_avg_by_calendar_month(monthly_id_hours_all)
    if by_month_id is not None:
        plot_bar_months(
            by_month_id,
            title="Monthly MCS Storm-Hours (ID-based) — averaged across years",
            ylabel="Storm-hours / month",
            save_path=os.path.join(out_plot_dir, "monthly_storm_hours_id_based.png")
        )
    else:
        print("No monthly_mcs_hours_by_id.csv found; skipping ID-based monthly plot.")

    # (B) Domain presence hours (0..744)
    by_month_presence = monthly_avg_by_calendar_month(monthly_presence_hours_all)
    if by_month_presence is not None:
        plot_bar_months(
            by_month_presence,
            title="Monthly MCS Domain Presence Hours — averaged across years",
            ylabel="Hours / month (anywhere in domain)",
            save_path=os.path.join(out_plot_dir, "monthly_domain_presence_hours.png"),
            ylim=(0, 31*24)
        )
    else:
        print("No monthly_domain_presence_hours.csv or monthly_counts.csv found; skipping presence-hours plot.")

    # (C) Domain cell-hours
    by_month_cell = monthly_avg_by_calendar_month(monthly_cell_hours_all)
    if by_month_cell is not None:
        plot_bar_months(
            by_month_cell,
            title="Monthly MCS Domain Cell-Hours — averaged across years",
            ylabel="Cell-hours / month (sum over grid cells)",
            save_path=os.path.join(out_plot_dir, "monthly_domain_cell_hours.png")
        )
    else:
        print("No monthly_domain_cell_hours.csv found; skipping cell-hours plot.")

    # (D) Monthly mean cell-hours (NEW; better comparable scale)
    by_month_mean_cell = monthly_avg_by_calendar_month(monthly_mean_cell_hours_all)
    if by_month_mean_cell is not None:
        plot_bar_months(
            by_month_mean_cell,
            title="Monthly Mean Cell-Hours — averaged across years",
            ylabel="Hours/month per grid cell (mean)",
            save_path=os.path.join(out_plot_dir, "monthly_mean_cell_hours.png")
        )
    else:
        print("No monthly_mean_cell_hours.csv found; skipping monthly mean cell-hours plot.")

    # ---- Annual series (domain means) ----
    # Use year_dirs integers where possible (but keep only those that were actually loaded)
    loaded_years = []
    for yd in year_dirs:
        try:
            loaded_years.append(int(yd))
        except Exception:
            pass

    # Safest: build annual series from loaded arrays rather than directory list
    annual_days_mean = pd.Series(
        [float(_domain_mean(da).values) for da in yearly_days_maps],
        index=loaded_years[:len(yearly_days_maps)] if len(loaded_years) >= len(yearly_days_maps) else range(len(yearly_days_maps)),
        name="days_per_cell"
    )
    annual_hours_mean = pd.Series(
        [float(_domain_mean(da).values) for da in freq_hours_maps],
        index=annual_days_mean.index,
        name="hours_per_cell"
    )

    # hours per active day annual (prefer file if exists)
    if len(hours_per_active_day_maps) > 0 and len(hours_per_active_day_maps) == len(yearly_days_maps):
        annual_hpad = pd.Series(
            [float(_domain_mean(da).values) for da in hours_per_active_day_maps],
            index=annual_days_mean.index,
            name="hours_per_active_day"
        )
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            annual_hpad = (annual_hours_mean / annual_days_mean).replace([np.inf, -np.inf], np.nan)

    # Plot annual mean days
    if not annual_days_mean.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(annual_days_mean.index, annual_days_mean.values, marker="o")
        plt.title("Annual Mean MCS Days per Grid Cell (domain mean)")
        plt.xlabel("Year")
        plt.ylabel("Days/year (mean per cell)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plot_dir, "annual_mean_mcs_days_series.png"), dpi=150)
        plt.show()

    # Plot annual mean hours
    if not annual_hours_mean.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(annual_hours_mean.index, annual_hours_mean.values, marker="o")
        plt.title("Annual Mean MCS Hours per Grid Cell (domain mean)")
        plt.xlabel("Year")
        plt.ylabel("Hours/year (mean per cell)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plot_dir, "annual_mean_mcs_hours_series.png"), dpi=150)
        plt.show()

    # Plot annual hours per active day
    if annual_hpad is not None and not annual_hpad.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(annual_hpad.index, annual_hpad.values, marker="o")
        plt.title("Annual Mean Hours per Active MCS Day (domain mean)")
        plt.xlabel("Year")
        plt.ylabel("Hours per MCS day")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plot_dir, "annual_mean_hours_per_active_day_series.png"), dpi=150)
        plt.show()

    # Plot domain-wide days/year if present
    if annual_domain_days_rows:
        ddf = pd.DataFrame(annual_domain_days_rows)
        if "year" in ddf and "domain_MCS_days" in ddf:
            ddf = ddf.sort_values("year")
            plt.figure(figsize=(10, 5))
            plt.plot(ddf["year"], ddf["domain_MCS_days"], marker="o")
            plt.title("Domain-wide MCS Days per Year (anywhere in domain)")
            plt.xlabel("Year")
            plt.ylabel("Days/year (domain-anywhere)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_plot_dir, "annual_domain_mcs_days_series.png"), dpi=150)
            plt.show()

    print(f"Saved summary plots in: {out_plot_dir}")


# ------------------------------------------
# Run
# ------------------------------------------
if __name__ == "__main__":
    # directory containing per-year subfolders created by the analysis
    base_output_dir = "outputs_wnet_hourly_IWV"  # change per dataset
    load_outputs_and_plot(
        base_output_dir,
        dataset_name="TE",
        out_plot_dir="summary_plots_hourly_IWV"
    )

