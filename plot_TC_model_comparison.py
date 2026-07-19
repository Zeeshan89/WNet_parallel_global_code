#!/usr/bin/env python
"""
TC (hurricane) 6-model comparison figures for RegCM-CMIP6, plus an IBTrACS reference.

    python plot_TC_model_comparison.py [--start-year N] [--end-year N]
                                       [--example-year N] [--no-reference]

Writes Figures_TC_comparison/{fig1..fig9}.{png,pdf}.

READ THIS BEFORE INTERPRETING THE COUNTS
    ~28% of the lat/lon bounding box lies outside RegCM's curved Lambert domain and
    is NaN in the driving fields. TC detections landing there are artifacts of the
    fillna(0.0) region — there is no simulated atmosphere at those points. Measured
    on ACC 2005, 24 of 64 raw tracks (38%) lie ENTIRELY outside the domain, and
    restricting to fully-in-domain tracks gives 14/yr against a reality check of
    ~10-15 TCs/yr reaching this domain.

    By explicit decision these tracks are RETAINED, not filtered. The in-domain
    fraction is therefore printed to stdout AND annotated on every map panel so no
    single figure can be read as if the raw counts were clean. fig10 reports the
    diagnostic directly. If the counts are later judged to need correcting, the fix
    is to mask the detection field before tracking (not to post-filter, which cannot
    undo fake points that bridged or dragged a track) — see the plan.

    ETC is nearly unaffected by this (98.9% of points in-domain), because ETCs live
    at mid-latitudes where the domain fills the box.

QC is gentler than the ETC gate: TCs are shorter-lived and less zonally displaced
than extratropical cyclones, so the ERA5 ETC thresholds (48 h / 8 points / 1000 km)
would cut ~58% of tracks. Applied symmetrically to models and IBTrACS regardless.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import storm_plot_common as spc
import storm_compare_figs as scf
import storm_tracks_io as sio
from storm_compare_figs import Panel

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Figures_TC_comparison")
STORM = "TC"

# TC-appropriate QC (see module docstring).
QC = dict(min_duration_h=24.0, min_points=4, min_disp_km=300.0)


def available_models():
    out = []
    for m in spc.MODEL_ORDER:
        d = os.path.join(sio.REGCM_ROOT, f"Outputs_RegCM_PR_{m}", "Hurricane_tracks")
        if os.path.isdir(d) and any(x.startswith("CONUS_year_") for x in os.listdir(d)):
            out.append(m)
    return out


def fig_domain_diagnostic(panels, stats, out_dir):
    """The in-domain diagnostic as a figure in its own right, since the tracks
    themselves are deliberately unfiltered."""
    keys = [p.key for p in panels if p.key != "REF"]
    if not keys:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(keys))
    ax1.bar(x, [100 * stats[k]["frac"] for k in keys],
            color=[spc.COLORS.get(k, "0.3") for k in keys], alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(keys)
    ax1.set_ylim(0, 100)
    ax1.axhline(100, color="0.6", lw=0.8, ls=":")
    spc.style_line_axes(ax1, None, "% of track points in domain",
                        "Track points inside the true model domain")

    ax2.bar(x - 0.2, [stats[k]["n"] for k in keys], width=0.4,
            color="0.75", label="all tracks")
    ax2.bar(x + 0.2, [stats[k]["n_in"] for k in keys], width=0.4,
            color=[spc.COLORS.get(k, "0.3") for k in keys], label="fully in-domain")
    ax2.set_xticks(x); ax2.set_xticklabels(keys)
    spc.style_line_axes(ax2, None, "tracks per year",
                        "Effect of the domain on TC counts")
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle("TC domain diagnostic — tracks are NOT filtered on this",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    spc.save_fig(fig, "fig10_domain_diagnostic", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=2014)
    ap.add_argument("--example-year", type=int, default=2005)
    ap.add_argument("--no-reference", action="store_true")
    a = ap.parse_args()

    spc.apply_style()
    years = list(range(a.start_year, a.end_year + 1))
    models = available_models()
    print(f"models with TC tracks: {models}")
    print(f"output: {OUT_DIR}\n")

    panels, notes, stats = [], {}, {}
    for m in models:
        tr = sio.load_regcm_tc_tracks(m, years)
        tr = sio.qc_filter(tr, label=m, **QC)
        yrs = sorted({sio.year_of(t) for t in tr})
        panels.append(Panel(m, tr, yrs))
        frac = sio.in_domain_fraction(tr)
        n_in = sum(1 for t in tr if t["in_domain"].all())
        ny = max(len(yrs), 1)
        stats[m] = dict(frac=frac, n=len(tr) / ny, n_in=n_in / ny,
                        outside=sio.fully_outside_count(tr))
        notes[m] = f"in-domain {100*frac:.0f}%"

    if not a.no_reference:
        # QC inside the loader, on the un-clipped global tracks, then clip — see
        # storm_tracks_io.load_reference_etc_tracks for why the order matters.
        # Matched in-domain criteria, as on the ETC side. The effect is small here
        # (19.0 -> 17.8/yr): only 1.1% of IBTrACS tracks reaching this domain clip a
        # corner, median in-domain length is 32 points. So unlike ETC, the TC
        # reference is sound as-is and the discrepancy is on the MODEL side — see
        # the domain-artifact note above.
        ref = sio.load_reference_tc_tracks(
            years, qc=QC, domain_criteria=sio.TC_DOMAIN_CRITERIA)
        yrs = sorted({sio.year_of(t) for t in ref})
        panels.append(Panel("REF", ref, yrs))
        notes["REF"] = "IBTrACS, clipped"

    print()
    for p in panels:
        ny = max(len(p.years), 1)
        extra = ""
        if p.key in stats:
            s = stats[p.key]
            extra = (f"   in-domain pts {100*s['frac']:4.1f}%   "
                     f"{s['n']:.1f}/yr all vs {s['n_in']:.1f}/yr fully in-domain")
        print(f"  {p.key:>4s}: {len(p.tracks):5d} tracks over {len(p.years)} yr{extra}")
    print()

    scf.fig_track_density(panels, OUT_DIR, STORM, notes=notes)
    scf.fig_kde_density(panels, OUT_DIR, STORM, bw=0.18, notes=notes)
    scf.fig_genesis_density(panels, OUT_DIR, STORM, notes=notes)
    scf.fig_annual_counts(panels, OUT_DIR, STORM)
    scf.fig_storm_days(panels, OUT_DIR, STORM)
    scf.fig_seasonal_cycle(panels, OUT_DIR, STORM)
    scf.fig_duration(panels, OUT_DIR, STORM)
    scf.fig_example_year_tracks(panels, OUT_DIR, STORM, a.example_year)
    # MSLP is in Pa in the track files; scale=0.01 -> hPa.
    scf.fig_intensity(panels, OUT_DIR, STORM, field="mslp",
                      label="minimum MSLP (hPa)", scale=0.01, take="min")
    fig_domain_diagnostic(panels, stats, OUT_DIR)


if __name__ == "__main__":
    main()
