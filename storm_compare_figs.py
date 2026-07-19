"""
The 6-model + reference comparison figure set, shared by ETC and TC.

Both storm types need the same eight figures over the same track schema, so the
figure logic lives here once and the two drivers
(plot_ETC_model_comparison.py / plot_TC_model_comparison.py) only supply data and
labels. Writing them separately would have duplicated ~400 lines and let the two
figure sets drift, which is exactly the problem storm_plot_common.py exists to stop.

Panel data is passed as an ordered list of `Panel(key, tracks, years)` — key is a
model code or "REF". Every map figure shares one colour scale across panels so the
models are directly comparable.

The hemispheric ERA5 figures (NH-vs-SH latitude bands, global winter contrast, TC
basin counts) are deliberately absent: the RegCM domain is a single CONUS box, so
they are either degenerate or misleading here.
"""

from collections import namedtuple

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

import storm_plot_common as spc
from storm_plot_common import (COLORS, MARKERS, MON_NAMES, PANEL_LETTERS, PROJ,
                               REGCM_EXTENT, add_geo_features, add_model_legend,
                               joint_vmax, panel_grid, panel_subtitle, save_fig,
                               shared_colorbar, style_line_axes)
import storm_tracks_io as sio

Panel = namedtuple("Panel", "key tracks years")


# --------------------------------------------------------------------------- grids
def binned_density(tracks, gridres, extent, n_years, months=None, genesis=False):
    """Track-point (or genesis-point) density in points per year per grid cell."""
    lo0, lo1, la0, la1 = extent
    xe = np.arange(lo0, lo1 + gridres, gridres)
    ye = np.arange(la0, la1 + gridres, gridres)
    if genesis:
        lons, lats = sio.genesis_points(tracks)
    else:
        lons, lats = sio.all_points(tracks, months=months)
    if lons.size == 0:
        return None, xe, ye
    H, _, _ = np.histogram2d(lons, lats, bins=[xe, ye])
    return (H.T / max(n_years, 1)), xe, ye


def kde_field(tracks, extent, bw=0.2, nx=200, ny=200, maxpts=15000):
    """Normalised (0-1) gaussian-KDE surface over the domain.

    Thinning is deterministic (stride, not random) so a figure is reproducible.
    At RegCM sample sizes (~700-1400 tracks per model pooled over 35 yr) the cap is
    normally inert; it matters only for the pooled reference.
    """
    lons, lats = sio.all_points(tracks)
    if lons.size < 5:
        return None, None, None
    if lons.size > maxpts:
        step = int(np.ceil(lons.size / maxpts))
        lons, lats = lons[::step], lats[::step]
    lo0, lo1, la0, la1 = extent
    xs = np.linspace(lo0, lo1, nx)
    ys = np.linspace(la0, la1, ny)
    X, Y = np.meshgrid(xs, ys)
    try:
        k = gaussian_kde(np.vstack([lons, lats]), bw_method=bw)
    except np.linalg.LinAlgError:
        return None, None, None       # degenerate (collinear / too few points)
    Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    if Z.max() > 0:
        Z = Z / Z.max()
    return Z, xs, ys


# ----------------------------------------------------------------------- map figures
def _map_panels(panels, fielder, cmap, cbar_label, suptitle, name, out_dir,
                notes=None, mask_below=None):
    fields, coords = [], []
    for p in panels:
        f, x, y = fielder(p)
        fields.append(f)
        coords.append((x, y))
    vmax = joint_vmax(fields)
    if vmax is None or vmax <= 0:
        print(f"  (skip {name}: no data)")
        return

    n = len(panels)
    fig, axes = panel_grid(n)
    mesh = None
    for i, (p, ax) in enumerate(zip(panels, axes)):
        f = fields[i]
        if f is not None:
            plot_f = np.ma.masked_less(f, mask_below * vmax) if mask_below else f
            x, y = coords[i]
            mesh = ax.pcolormesh(x, y, plot_f, transform=PROJ, cmap=cmap,
                                 vmin=0, vmax=vmax, shading="auto", rasterized=True)
        add_geo_features(ax, REGCM_EXTENT, panel_letter=PANEL_LETTERS[i],
                         title=panel_subtitle(p.key, p.years),
                         note=(notes or {}).get(p.key))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.99)
    nrow = 1 if n <= 4 else 2
    shared_colorbar(fig, mesh, cbar_label, nrow_hint=nrow)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.92,
                        bottom=0.17 if nrow == 1 else 0.12,
                        wspace=0.08, hspace=0.02)
    save_fig(fig, name, out_dir)


def fig_track_density(panels, out_dir, storm, gridres=2.0, notes=None):
    _map_panels(
        panels,
        lambda p: binned_density(p.tracks, gridres, REGCM_EXTENT, len(p.years)),
        "hot_r", f"{storm} track points yr$^{{-1}}$ ({gridres}° cell)$^{{-1}}$",
        f"{storm} track density — RegCM-CMIP6 ensemble", "fig1_track_density",
        out_dir, notes=notes)


def fig_kde_density(panels, out_dir, storm, bw=0.2, notes=None):
    _map_panels(
        panels,
        lambda p: kde_field(p.tracks, REGCM_EXTENT, bw=bw),
        "Blues", "normalised track-point density",
        f"{storm} track-point KDE — RegCM-CMIP6 ensemble", "fig2_kde_density",
        out_dir, notes=notes, mask_below=0.04)


def fig_genesis_density(panels, out_dir, storm, gridres=2.0, notes=None):
    _map_panels(
        panels,
        lambda p: binned_density(p.tracks, gridres, REGCM_EXTENT, len(p.years),
                                 genesis=True),
        "viridis", f"genesis events yr$^{{-1}}$ ({gridres}° cell)$^{{-1}}$",
        f"{storm} genesis density — RegCM-CMIP6 ensemble", "fig3_genesis_density",
        out_dir, notes=notes)


# ---------------------------------------------------------------------- line figures
def _series_figure(panels, valfunc, ylabel, title, name, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in panels:
        by = sio.tracks_by_year(p.tracks)
        yrs = sorted(by)
        if not yrs:
            continue
        vals = [valfunc(by[y]) for y in yrs]
        ax.plot(yrs, vals, color=COLORS.get(p.key, "0.3"),
                marker=MARKERS.get(p.key, "o"), ms=4, lw=1.6,
                label=p.key, zorder=3 if p.key == "REF" else 2,
                alpha=1.0 if p.key == "REF" else 0.9)
    style_line_axes(ax, "year", ylabel, title)
    ax.grid(alpha=0.25, lw=0.5)
    add_model_legend(fig, [p.key for p in panels])
    save_fig(fig, name, out_dir)


def fig_annual_counts(panels, out_dir, storm):
    _series_figure(panels, len, f"{storm} tracks per year",
                   f"{storm} annual track count", "fig4_annual_counts", out_dir)


def fig_storm_days(panels, out_dir, storm):
    _series_figure(panels, sio.storm_days, f"{storm} storm-days per year",
                   f"{storm} annual storm-days", "fig5_storm_days", out_dir)


def fig_seasonal_cycle(panels, out_dir, storm):
    """Mean tracks per month, counting a track in the month it forms."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in panels:
        if not p.years:
            continue
        counts = np.zeros(12)
        for t in p.tracks:
            counts[t["times"][0].month - 1] += 1
        counts /= len(p.years)
        ax.plot(range(1, 13), counts, color=COLORS.get(p.key, "0.3"),
                marker=MARKERS.get(p.key, "o"), ms=5, lw=1.8, label=p.key)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MON_NAMES)
    style_line_axes(ax, None, f"{storm} genesis events per month (yr$^{{-1}}$)",
                    f"{storm} seasonal cycle")
    ax.grid(alpha=0.25, lw=0.5)
    add_model_legend(fig, [p.key for p in panels])
    save_fig(fig, "fig6_seasonal_cycle", out_dir)


def fig_duration(panels, out_dir, storm):
    """Duration distributions, one panel per dataset, shared axes."""
    n = len(panels)
    fig, axes = panel_grid(n, projection=False, panel_w=3.4, panel_h=3.0)
    allv = [t["duration_h"] / 24.0 for p in panels for t in p.tracks]
    if not allv:
        print("  (skip fig7_duration: no data)")
        plt.close(fig)
        return
    hi = float(np.nanpercentile(allv, 99))
    bins = np.linspace(0, max(hi, 1.0), 25)
    for i, (p, ax) in enumerate(zip(panels, axes)):
        v = [t["duration_h"] / 24.0 for t in p.tracks]
        if v:
            ax.hist(v, bins=bins, color=COLORS.get(p.key, "0.3"), alpha=0.85)
            ax.axvline(np.median(v), color="0.15", ls="--", lw=1.2)
        style_line_axes(ax, "duration (days)", "tracks" if i % 4 == 0 else None,
                        panel_subtitle(p.key, p.years))
        ax.text(0.97, 0.95, f"({PANEL_LETTERS[i]})", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold")
    fig.suptitle(f"{storm} track duration (dashed = median)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig7_duration", out_dir)


def fig_example_year_tracks(panels, out_dir, storm, year):
    """Track maps for one year, one panel per dataset."""
    n = len(panels)
    fig, axes = panel_grid(n)
    for i, (p, ax) in enumerate(zip(panels, axes)):
        yr_tracks = [t for t in p.tracks if sio.year_of(t) == year]
        cmap = plt.get_cmap("turbo")
        for j, t in enumerate(yr_tracks):
            c = cmap(j / max(len(yr_tracks) - 1, 1))
            ax.plot(t["lons"], t["lats"], color=c, lw=1.1, transform=PROJ, alpha=0.9)
            ax.plot(t["lons"][0], t["lats"][0], marker="o", ms=3.2, color=c,
                    mec="0.2", mew=0.4, transform=PROJ)
        add_geo_features(ax, REGCM_EXTENT, panel_letter=PANEL_LETTERS[i],
                         title=f"{p.key}  ({year}, {len(yr_tracks)} tracks)")
    fig.suptitle(f"{storm} tracks, {year} (dot = genesis)",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.04,
                        wspace=0.08, hspace=0.02)
    save_fig(fig, "fig8_example_year_tracks", out_dir)


def fig_intensity(panels, out_dir, storm, field="mslp", label="minimum MSLP (hPa)",
                  scale=0.01, take="min"):
    """Per-track intensity distribution. TC only — ETC has no physical intensity
    (its tracker records connected-component pixel count, not vorticity or MSLP)."""
    n = len(panels)
    fig, axes = panel_grid(n, projection=False, panel_w=3.4, panel_h=3.0)
    vals_by = {}
    for p in panels:
        v = []
        for t in p.tracks:
            arr = t[field][np.isfinite(t[field])]
            if arr.size:
                v.append((arr.min() if take == "min" else arr.max()) * scale)
        vals_by[p.key] = v
    allv = [x for v in vals_by.values() for x in v]
    if not allv:
        print(f"  (skip fig9_intensity: no finite {field})")
        plt.close(fig)
        return
    bins = np.linspace(np.nanpercentile(allv, 1), np.nanpercentile(allv, 99), 25)
    for i, (p, ax) in enumerate(zip(panels, axes)):
        v = vals_by[p.key]
        if v:
            ax.hist(v, bins=bins, color=COLORS.get(p.key, "0.3"), alpha=0.85)
            ax.axvline(np.median(v), color="0.15", ls="--", lw=1.2)
        style_line_axes(ax, label, "tracks" if i % 4 == 0 else None,
                        panel_subtitle(p.key, p.years))
        ax.text(0.97, 0.95, f"({PANEL_LETTERS[i]})", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold")
    fig.suptitle(f"{storm} {label} (dashed = median)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig9_intensity", out_dir)
