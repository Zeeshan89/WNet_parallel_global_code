#!/usr/bin/env python
"""
3-column comparison [ OLD 6-hourly | NEW 6-hourly | HOURLY ] for EVERY standard plot in
summary_plots_mcs_wnet_IWV: 6 spatial maps, 4 monthly bar charts, 4 annual series.
Reductions/colormaps/units identical to plot_wnet_newcode_final.py (helpers imported).
HOURLY column auto-fills if its stats dir has year folders; else labeled blank.
Output -> summary_plots_compare_old_new_hourly/
"""
import os, sys, numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

CODE = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/WNet_parallel_global_code"
sys.path.insert(0, CODE)
from plot_wnet_newcode_final import (safe_open_dataarray, _read_monthly_series,
                                     monthly_avg_by_calendar_month, _domain_mean)

DATASETS = [("OLD 6-hourly (full)",    f"{CODE}/outputs_wnet_standard_IWV"),
            ("NEW 6-hourly (MCS_ONLY)",f"{CODE}/outputs_wnet_new6h_IWV"),
            ("HOURLY (MCS_ONLY)",      f"{CODE}/outputs_wnet_hourly_IWV")]
OUTDIR = f"{CODE}/summary_plots_compare_old_new_hourly"
os.makedirs(OUTDIR, exist_ok=True)
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def load_dataset(stats_dir):
    """Mirror plot_wnet_newcode_final loading; return maps/monthly/annual reductions."""
    if not os.path.isdir(stats_dir): return None
    ydirs = [d for d in sorted(os.listdir(stats_dir)) if os.path.isdir(os.path.join(stats_dir, d))]
    days, hours, hpad, precip, pct = [], [], [], [], []
    m_id, m_pres, m_cell, m_mcell = [], [], [], []
    dom_rows, years = [], []
    lat = lon = None
    for yd in ydirs:
        yp = os.path.join(stats_dir, yd)
        try:
            dy = safe_open_dataarray(os.path.join(yp, "yearly_MCS_days.nc"))
            fh = safe_open_dataarray(os.path.join(yp, "freq_hours.nc"))
            ap = safe_open_dataarray(os.path.join(yp, "avg_precip.nc"))
            pc = safe_open_dataarray(os.path.join(yp, "percent_precip.nc"))
        except Exception:
            continue
        days.append(dy); hours.append(fh); precip.append(ap); pct.append(pc)
        try: years.append(int(yd))
        except Exception: years.append(len(years))
        p = os.path.join(yp, "hours_per_active_day.nc")
        if os.path.exists(p): hpad.append(safe_open_dataarray(p))
        for lst, fn in [(m_id,"monthly_mcs_hours_by_id.csv"),
                        (m_cell,"monthly_domain_cell_hours.csv"),
                        (m_mcell,"monthly_mean_cell_hours.csv")]:
            fp = os.path.join(yp, fn)
            if os.path.exists(fp):
                try: lst.append(_read_monthly_series(fp))
                except Exception: pass
        fp = os.path.join(yp, "monthly_domain_presence_hours.csv")
        if not os.path.exists(fp): fp = os.path.join(yp, "monthly_counts.csv")
        if os.path.exists(fp):
            try: m_pres.append(_read_monthly_series(fp))
            except Exception: pass
        fp = os.path.join(yp, "domain_MCS_days_by_year.csv")
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                if {"year","domain_MCS_days"}.issubset(df.columns): dom_rows.append(df.iloc[0].to_dict())
            except Exception: pass
        if lat is None:
            ln = "lat" if "lat" in dy.coords else "latitude"; on = "lon" if "lon" in dy.coords else "longitude"
            lat, lon = dy[ln].values, dy[on].values
    if not days: return None
    def mmean(l): return xr.concat(l, dim="year").mean("year", skipna=True).values if l else None
    mean_hours = xr.concat(hours, dim="year").mean("year", skipna=True)
    R = {"lat": lat, "lon": lon, "years": years[:len(days)]}
    R["map_days"]   = mmean(days)
    R["map_hours"]  = mean_hours.values
    R["map_hpad"]   = mmean(hpad)
    R["map_precip"] = mmean(precip)
    R["map_pct"]    = mmean(pct)
    R["map_days_from_hours"] = (mean_hours/24.0).where(mean_hours/24.0 > 0).values
    R["mon_id"]    = monthly_avg_by_calendar_month(m_id)
    R["mon_pres"]  = monthly_avg_by_calendar_month(m_pres)
    R["mon_cell"]  = monthly_avg_by_calendar_month(m_cell)
    R["mon_mcell"] = monthly_avg_by_calendar_month(m_mcell)
    R["ann_days"]  = pd.Series([float(_domain_mean(d).values) for d in days], index=R["years"])
    R["ann_hours"] = pd.Series([float(_domain_mean(d).values) for d in hours], index=R["years"])
    if hpad and len(hpad)==len(days):
        R["ann_hpad"] = pd.Series([float(_domain_mean(d).values) for d in hpad], index=R["years"])
    else:
        R["ann_hpad"] = (R["ann_hours"]/R["ann_days"]).replace([np.inf,-np.inf], np.nan)
    if dom_rows:
        ddf = pd.DataFrame(dom_rows).sort_values("year"); R["ann_domain"] = ddf.set_index("year")["domain_MCS_days"]
    else: R["ann_domain"] = None
    return R

DS = [(t, load_dataset(d)) for t, d in DATASETS]

def blank(ax, msg="hourly stats\n(pending)"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=13, color="gray")

# ---------- 6 spatial maps ----------
MAPS = [("mean_annual_mcs_days","map_days","YlOrRd","Days/year","Mean Annual MCS Days",None),
        ("mean_annual_mcs_hours","map_hours","YlOrRd","Hours/year","Mean Annual MCS Hours",None),
        ("mean_hours_per_active_day","map_hpad","viridis","Hours per MCS day","Mean Hours per Active MCS Day",None),
        ("mean_mcs_precip","map_precip","Blues","precip/step","Mean Precip During MCS",None),
        ("percent_precip_from_mcs","map_pct","YlGnBu","%","% of Precip from MCS",50.0),
        ("mean_annual_mcs_days_from_hours","map_days_from_hours","YlGnBu","Equiv days/year","Mean Annual MCS Days (Hours/24)",50.0)]
for out, key, cmap, label, title, fvmax in MAPS:
    present = [d[key] for _, d in DS if d and d.get(key) is not None]
    if not present: print(f"  map {out}: no data, skip"); continue
    vmax = fvmax if fvmax is not None else float(np.nanpercentile(np.concatenate([p.ravel() for p in present]),99))
    fig, ax = plt.subplots(1,3,figsize=(21,4.8),subplot_kw={"projection":ccrs.PlateCarree()},constrained_layout=True)
    im=None
    for k,(t,d) in enumerate(DS):
        a=ax[k]; dat = d[key] if d else None
        if dat is None: blank(a); a.set_title(t,fontsize=11); continue
        lon2d,lat2d=np.meshgrid(d["lon"],d["lat"])
        im=a.pcolormesh(lon2d,lat2d,dat,shading="auto",cmap=cmap,vmin=0,vmax=vmax,transform=ccrs.PlateCarree())
        a.coastlines(resolution="50m",linewidth=0.4); a.add_feature(cfeature.BORDERS,linewidth=0.3)
        a.set_title(f"{t}\n(mean {np.nanmean(dat):.2f})",fontsize=11)
    if im is not None: fig.colorbar(im,ax=ax,orientation="vertical",shrink=0.8,pad=0.01,label=label)
    fig.suptitle(f"{title} — 2004-2018",fontsize=15)
    fig.savefig(os.path.join(OUTDIR,f"compare_{out}.png"),dpi=140); plt.close(fig); print(f"[map] {out}")

# ---------- 4 monthly bar charts ----------
BARS=[("monthly_storm_hours_id_based","mon_id","Storm-hours/month","Monthly MCS Storm-Hours (ID-based)"),
      ("monthly_domain_presence_hours","mon_pres","Hours/month","Monthly MCS Domain Presence Hours"),
      ("monthly_domain_cell_hours","mon_cell","Cell-hours/month","Monthly MCS Domain Cell-Hours"),
      ("monthly_mean_cell_hours","mon_mcell","Hours/month per cell","Monthly Mean Cell-Hours")]
for out,key,ylab,title in BARS:
    present=[d[key] for _,d in DS if d and d.get(key) is not None]
    if not present: print(f"  bar {out}: no data, skip"); continue
    ymax=max(float(np.nanmax(s.values)) for s in present)*1.1
    fig,ax=plt.subplots(1,3,figsize=(21,4.2),constrained_layout=True,sharey=True)
    for k,(t,d) in enumerate(DS):
        a=ax[k]; s=d[key] if d else None
        if s is None: blank(a)
        else: a.bar(range(12), s.values, color="steelblue"); a.set_xticks(range(12)); a.set_xticklabels(MONTHS,rotation=45,fontsize=8); a.set_ylim(0,ymax)
        a.set_title(t,fontsize=11); a.set_ylabel(ylab if k==0 else "")
    fig.suptitle(f"{title} — avg across years 2004-2018",fontsize=14)
    fig.savefig(os.path.join(OUTDIR,f"compare_{out}.png"),dpi=140); plt.close(fig); print(f"[bar] {out}")

# ---------- 4 annual series ----------
SER=[("annual_mean_mcs_days_series","ann_days","Days/year (mean per cell)","Annual Mean MCS Days per Grid Cell"),
     ("annual_mean_mcs_hours_series","ann_hours","Hours/year (mean per cell)","Annual Mean MCS Hours per Grid Cell"),
     ("annual_mean_hours_per_active_day_series","ann_hpad","Hours per MCS day","Annual Mean Hours per Active MCS Day"),
     ("annual_domain_mcs_days_series","ann_domain","Days/year (domain-anywhere)","Domain-wide MCS Days per Year")]
for out,key,ylab,title in SER:
    present=[d[key] for _,d in DS if d and d.get(key) is not None]
    if not present: print(f"  series {out}: no data, skip"); continue
    ymax=max(float(np.nanmax(s.values)) for s in present)*1.1
    fig,ax=plt.subplots(1,3,figsize=(21,4.2),constrained_layout=True,sharey=True)
    for k,(t,d) in enumerate(DS):
        a=ax[k]; s=d[key] if d else None
        if s is None: blank(a)
        else: a.plot(s.index, s.values, marker="o"); a.grid(True,alpha=0.3); a.set_ylim(0,ymax); a.set_xlabel("Year")
        a.set_title(t,fontsize=11); a.set_ylabel(ylab if k==0 else "")
    fig.suptitle(f"{title} — 2004-2018",fontsize=14)
    fig.savefig(os.path.join(OUTDIR,f"compare_{out}.png"),dpi=140); plt.close(fig); print(f"[series] {out}")

print(f"\nAll comparison figures -> {OUTDIR}  ({len(os.listdir(OUTDIR))} files)", flush=True)
