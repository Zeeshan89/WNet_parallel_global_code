"""
Shared plotting conventions for the RegCM 6-model storm comparison figures.

WHY THIS MODULE EXISTS
    plot_MCS_model_comparison.py and plot_AR_model_comparison.py each carry their
    own copy of add_geo_features / save_fig / map_row_figure / the rcParams block /
    COLORS / MARKERS. The copies have already drifted: the AR copy is missing MRI
    and NOR entirely, and the two add_geo_features bodies differ (gridlines drawn
    vs labels-only, "auto" vs "gouraud" shading). Adding ETC and TC scripts without
    factoring this out would make it four divergent copies and would not guarantee
    that ACC is the same blue in every figure of one paper.

    Definitions here are lifted from the MCS versions, which are the newer and more
    considered of the two.

SCOPE
    The AR and MCS scripts are deliberately NOT migrated onto this module in the
    same change — they currently produce accepted figures, and a refactor would put
    regression risk on them. Migrating them is a separate, verifiable follow-up.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------------------------------- models
# Canonical display order, consistent with plot_MCS_model_comparison.py:36,
# mcs-wnet-newfinalcode.py:56 and updated_monthly_ETC_Tracker.py:603.
MODEL_ORDER = ["ACC", "BCC", "CNR", "MPI", "MRI", "NOR"]

# tab10, deliberately non-sequential so adjacent models are easy to tell apart.
# REF is the observational/reanalysis reference panel (IBTrACS for TC, ERA5-WNet
# for ETC) — neutral dark grey so it never reads as "just another model".
COLORS = {"ACC": "#1f77b4", "BCC": "#d62728", "CNR": "#2ca02c",
          "MPI": "#9467bd", "MRI": "#ff7f0e", "NOR": "#8c564b",
          "REF": "#333333"}
MARKERS = {"ACC": "o", "BCC": "s", "CNR": "^", "MPI": "D",
           "MRI": "v", "NOR": "P", "REF": "*"}

PANEL_LETTERS = list("abcdefghij")

# Full CMIP6 driving-model names are NOT recorded anywhere in this repo, and each
# code has CM/ESM and low/high-resolution variants. Panels therefore carry the
# 3-letter code until the driving-model global attributes are confirmed.
MODEL_LABELS = {m: m for m in MODEL_ORDER}
MODEL_LABELS["REF"] = "Reference"

# --------------------------------------------------------------------------- domain
# The RegCM lat/lon bounding box. Matches plot_AR_model_comparison.py:37 and
# mcs-wnet-newfinalcode.py:70-71.
#
# CAUTION: this is the bounding box, not the model domain. RegCM runs on a curved
# Lambert grid inscribed in this box, so ~28% of the box is outside the simulation
# and is NaN in the driving fields (93% of the row at lat 18.75). Use
# storm_tracks_io.domain_mask() when validity matters — do not treat this extent as
# "where there is data".
REGCM_EXTENT = [-138.75, -57.0, 18.0, 56.75]

PROJ = ccrs.PlateCarree()

MON_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------- global style
RCPARAMS = {
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
}


def apply_style():
    plt.rcParams.update(RCPARAMS)


# --------------------------------------------------------------------------- maps
def add_geo_features(ax, extent, panel_letter=None, title=None, note=None):
    """Standard basemap. `note` prints a small annotation in the lower-left, used to
    carry the in-domain fraction onto TC panels so a single figure is not misleading."""
    ax.set_extent(extent, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor="0.96", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(lw=0.5, color="0.2")
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.4")
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.6")
    # Labels only, no graticule — lines tile this smaller CONUS domain into boxes
    # that read as a projection artifact.
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
    if note:
        ax.text(0.015, 0.03, note, transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left", color="0.25",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))


def panel_grid(n, projection=True, panel_w=4.6, panel_h=None):
    """Lay out n panels, wrapping to two rows past four.

    The MCS template only has a 1xN row, which at the 7 panels this comparison needs
    (6 models + reference) would be a ~32-inch-wide strip. Returns (fig, axes) where
    axes has exactly n entries; surplus grid cells are removed.

    For map panels the height defaults to the DOMAIN's aspect ratio rather than a
    fixed 5.4in. The RegCM box is 81.75 deg wide by 38.75 deg tall (~2.1:1), so a
    fixed tall panel leaves most of each cell empty and the two rows drift far apart.
    """
    if panel_h is None:
        if projection:
            aspect = (REGCM_EXTENT[3] - REGCM_EXTENT[2]) / (REGCM_EXTENT[1] - REGCM_EXTENT[0])
            panel_h = panel_w * aspect + 1.0     # + room for title and tick labels
        else:
            panel_h = 3.0
    if n <= 4:
        nrow, ncol = 1, n
    else:
        nrow, ncol = 2, int(np.ceil(n / 2))
    kw = dict(projection=PROJ) if projection else {}
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                             subplot_kw=kw, squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:          # drop unused cells so they leave no ghost frame
        fig.delaxes(ax)
    return fig, flat[:n]


def shared_colorbar(fig, mesh, label, nrow_hint=1):
    """One horizontal colorbar under the panel block."""
    if mesh is None:
        return
    y = 0.05 if nrow_hint == 1 else 0.045
    cbar_ax = fig.add_axes([0.30, y, 0.42, 0.022 if nrow_hint > 1 else 0.028])
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", extend="max")
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=10)


def joint_vmax(fields, pct=99.5):
    """Shared upper limit across panels so colours are comparable between models."""
    valid = [np.asarray(f).ravel() for f in fields if f is not None]
    valid = [v[np.isfinite(v)] for v in valid]
    valid = [v for v in valid if v.size]
    if not valid:
        return None
    return float(np.nanpercentile(np.concatenate(valid), pct))


# --------------------------------------------------------------------------- output
def save_fig(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def panel_subtitle(model, years):
    """`ACC (1980-2014, 35 yr)`, with a trailing * when the record has gaps."""
    label = MODEL_LABELS.get(model, model)
    years = sorted(set(years or []))
    if not years:
        return label
    if years[0] == years[-1]:
        return f"{label}  ({years[0]})"
    partial = "" if len(years) == (years[-1] - years[0] + 1) else "*"
    return f"{label}  ({years[0]}–{years[-1]}, {len(years)} yr{partial})"


def style_line_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def add_model_legend(fig, entries, ncol=None):
    """Single figure-level legend, matching the MCS convention."""
    handles = [plt.Line2D([], [], color=COLORS.get(k, "0.3"),
                          marker=MARKERS.get(k, "o"), ms=5, lw=1.8,
                          label=MODEL_LABELS.get(k, k))
               for k in entries]
    fig.legend(handles=handles, loc="lower center", frameon=False,
               ncol=ncol or len(entries), bbox_to_anchor=(0.5, -0.04))
