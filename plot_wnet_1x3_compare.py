#!/usr/bin/env python
"""
1x3 side-by-side comparison of the mcs-wnet-newfinalcode outputs for 3 datasets:
  col1 = old 6-hourly (outputs_wnet_standard_IWV, full-channel)
  col2 = new 6-hourly (outputs_wnet_new6h_IWV, MCS_ONLY)
  col3 = new hourly   (outputs_wnet_hourly_IWV, MCS_ONLY)
All restricted to YEARS (2014-2018), CONUS. Reproduces the 14 summary plots as 1x3 panels.
Spatial maps share one colour scale across the 3 columns for a fair comparison.
"""
import os, glob, numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CODE = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code"
DATASETS = [
    ("Old 6-hourly (full)",  f"{CODE}/outputs_wnet_standard_IWV"),
    ("New 6-hourly (MCS_ONLY)", f"{CODE}/outputs_wnet_new6h_IWV"),
    ("New hourly (MCS_ONLY)",   f"{CODE}/outputs_wnet_hourly_IWV"),
]
YEARS = list(range(2004, 2019))
OUT = f"{CODE}/summary_plots_1x3_compare"
os.makedirs(OUT, exist_ok=True)

def years_present(d):
    return [y for y in YEARS if os.path.isdir(os.path.join(d, str(y)))]

def mean_map(d, fname, var):
    das = []
    for y in years_present(d):
        f = os.path.join(d, str(y), fname)
        if os.path.exists(f):
            das.append(xr.open_dataset(f)[var])
    if not das: return None
    return xr.concat(das, dim="year").mean("year", skipna=True)

def monthly_by_calmonth(d, fname, valcol):
    vals = {}
    for y in years_present(d):
        f = os.path.join(d, str(y), fname)
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        tcol = df.columns[0]
        mon = pd.to_datetime(df[tcol].astype(str), format="mixed", errors="coerce").dt.month
        for m, v in zip(mon, df[valcol]):
            if pd.notna(m): vals.setdefault(int(m), []).append(float(v))
    return pd.Series({m: np.mean(vs) for m, vs in sorted(vals.items())})

def annual_domain_mean(d, fname, var):
    out = {}
    for y in years_present(d):
        f = os.path.join(d, str(y), fname)
        if os.path.exists(f): out[y] = float(xr.open_dataset(f)[var].mean(skipna=True))
    return pd.Series(out)

def annual_domain_days(d):
    out = {}
    for y in years_present(d):
        f = os.path.join(d, str(y), "domain_MCS_days_by_year.csv")
        if os.path.exists(f): out[y] = float(pd.read_csv(f)["domain_MCS_days"].iloc[0])
    return pd.Series(out)

def fig_maps(fname, var, title, cbar, out_png, fixed_vmax=None, divide=None):
    maps = [mean_map(d, fname, var) for _, d in DATASETS]
    vals = [m for m in maps if m is not None]
    if not vals: print(f"  skip {out_png}: no data"); return
    if divide: maps = [m/divide if m is not None else None for m in maps]; vals=[m for m in maps if m is not None]
    vmax = fixed_vmax if fixed_vmax else np.nanpercentile(np.concatenate([m.values.ravel() for m in vals]), 99)
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.4))
    for k,(lab,_) in enumerate(DATASETS):
        m = maps[k]
        if m is None: ax[k].set_title(f"{lab}\n(no data)"); ax[k].axis("off"); continue
        im = ax[k].pcolormesh(m["longitude"], m["latitude"], m.values, cmap="viridis",
                              vmin=0, vmax=vmax, shading="auto")
        ax[k].set_title(lab); ax[k].set_xlabel("lon"); ax[k].set_ylabel("lat")
        plt.colorbar(im, ax=ax[k], shrink=.85, label=cbar)
    fig.suptitle(f"{title} — CONUS, {YEARS[0]}-{YEARS[-1]}", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/{out_png}", dpi=110); plt.close(fig)
    print(f"  wrote {out_png}")

def fig_bars(fname, valcol, title, ylab, out_png, ylim=None):
    series = [monthly_by_calmonth(d, fname, valcol) for _, d in DATASETS]
    ymax = max([s.max() for s in series if len(s)] + [1])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for k,(lab,_) in enumerate(DATASETS):
        s = series[k]
        if len(s): ax[k].bar(s.index, s.values, color="steelblue")
        ax[k].set_title(lab); ax[k].set_xlabel("month"); ax[k].set_xticks(range(1,13))
        ax[k].set_ylim(ylim if ylim else (0, ymax*1.1))
    ax[0].set_ylabel(ylab)
    fig.suptitle(f"{title} — mean by calendar month, {YEARS[0]}-{YEARS[-1]}", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/{out_png}", dpi=110); plt.close(fig)
    print(f"  wrote {out_png}")

def fig_series(getter, title, ylab, out_png):
    series = [getter(d) for _, d in DATASETS]
    ymax = max([s.max() for s in series if len(s)] + [1])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for k,(lab,_) in enumerate(DATASETS):
        s = series[k]
        if len(s): ax[k].plot(s.index, s.values, "-o", color="darkgreen")
        ax[k].set_title(lab); ax[k].set_xlabel("year"); ax[k].set_ylim(0, ymax*1.1); ax[k].grid(alpha=.3)
    ax[0].set_ylabel(ylab)
    fig.suptitle(f"{title}, {YEARS[0]}-{YEARS[-1]}", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/{out_png}", dpi=110); plt.close(fig)
    print(f"  wrote {out_png}")

if __name__ == "__main__":
    for lab, d in DATASETS: print(f"{lab}: years {years_present(d)}")
    # 6 spatial maps
    fig_maps("yearly_MCS_days.nc","mcs_days","Mean annual MCS days","days/yr","mean_annual_mcs_days.png")
    fig_maps("freq_hours.nc","freq","Mean annual MCS hours","hours/yr","mean_annual_mcs_hours.png")
    fig_maps("hours_per_active_day.nc","hours_per_active_day","Mean hours per active MCS day","hours","mean_hours_per_active_day.png")
    fig_maps("avg_precip.nc","tp","Mean precip during MCS","mm/step","mean_mcs_precip.png")
    fig_maps("percent_precip.nc","tp","% of precip from MCS","%","percent_precip_from_mcs.png", fixed_vmax=50)
    fig_maps("freq_hours.nc","freq","Mean annual MCS days (from hours/24)","days/yr","mean_annual_mcs_days_from_hours.png", divide=24.0)
    # 4 monthly bars
    fig_bars("monthly_mcs_hours_by_id.csv","hours","Monthly MCS storm-hours (ID-based)","hours","monthly_storm_hours_id_based.png")
    fig_bars("monthly_domain_presence_hours.csv","hours","Monthly domain presence hours","hours","monthly_domain_presence_hours.png", ylim=(0,744))
    fig_bars("monthly_domain_cell_hours.csv","cell_hours","Monthly domain cell-hours","cell-hours","monthly_domain_cell_hours.png")
    fig_bars("monthly_mean_cell_hours.csv","hours_per_cell","Monthly mean cell-hours","hours/cell","monthly_mean_cell_hours.png")
    # 4 annual series
    fig_series(lambda d: annual_domain_mean(d,"yearly_MCS_days.nc","mcs_days"), "Annual mean MCS days/cell","days","annual_mean_mcs_days_series.png")
    fig_series(lambda d: annual_domain_mean(d,"freq_hours.nc","freq"), "Annual mean MCS hours/cell","hours","annual_mean_mcs_hours_series.png")
    fig_series(lambda d: annual_domain_mean(d,"hours_per_active_day.nc","hours_per_active_day"), "Annual mean hours per active day","hours","annual_mean_hours_per_active_day_series.png")
    fig_series(annual_domain_days, "Annual domain MCS days","days","annual_domain_mcs_days_series.png")
    print("DONE ->", OUT)
