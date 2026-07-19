#!/usr/bin/env python
"""
ETC 6-model comparison figures for RegCM-CMIP6, plus an ERA5-WNet reference panel.

    python plot_ETC_model_comparison.py [--start-year N] [--end-year N]
                                        [--example-year N] [--no-reference]

Writes Figures_ETC_comparison/{fig1..fig8}.{png,pdf}, mirroring the AR and MCS
comparison figure sets.

NO INTENSITY FIGURE. The ETC tracker records connected-component PIXEL COUNT in its
INTENSITY column (updated_monthly_ETC_Tracker.py:249), not vorticity or MSLP, so
there is nothing physical to plot. Giving ETC a real intensity would need a tracker
change plus a re-run — see the plan's out-of-scope list.
"""

import argparse
import os

import storm_plot_common as spc
import storm_compare_figs as scf
import storm_tracks_io as sio
from storm_compare_figs import Panel

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Figures_ETC_comparison")
STORM = "ETC"


def available_models():
    """Models that actually have ETC track files, in canonical display order."""
    out = []
    for m in spc.MODEL_ORDER:
        d = os.path.join(sio.REGCM_ROOT, f"Outputs_RegCM_PR_{m}", "ETC_tracks")
        if os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d)):
            out.append(m)
    return out


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
    print(f"models with ETC tracks: {models}")
    print(f"output: {OUT_DIR}\n")

    panels, notes = [], {}
    for m in models:
        tr = sio.load_regcm_etc_tracks(m, years)
        tr = sio.qc_filter(tr, label=m)
        yrs = sorted({sio.year_of(t) for t in tr})
        panels.append(Panel(m, tr, yrs))
        frac = sio.in_domain_fraction(tr)
        notes[m] = f"in-domain {100*frac:.0f}%"

    if not a.no_reference:
        # QC is applied INSIDE the loader, on the un-clipped global tracks, then the
        # geometry is clipped to the domain. Clipping first would truncate genuine
        # ETCs that merely cross the domain and fail them on duration/displacement —
        # measured at 49% kept vs 96-98% for the models, a large fake bias.
        # The reference is scored on what it does INSIDE the domain, using the same
        # kinematic criteria the RegCM tracker applies (1800 km path, 3.5 m/s avg
        # propagation, 5 m/s median speed, 500 km net east; verified identical in
        # both file headers). Without this the counts are not comparable: clipping
        # alone keeps any track with >=2 points in the box, and 14.3% of reference
        # tracks have only 2-3 in-domain points, inflating REF ~2x (141 -> 68/yr).
        # Self-check: the same criteria change ACC by only -3%, because its tracker
        # already enforced them.
        ref = sio.load_reference_etc_tracks(
            years, qc={}, domain_criteria=sio.ETC_DOMAIN_CRITERIA)
        yrs = sorted({sio.year_of(t) for t in ref})
        panels.append(Panel("REF", ref, yrs))
        notes["REF"] = "ERA5-WNet, matched in-domain criteria"

        print("\n  NOTE: REF is scored on its IN-DOMAIN track only, under the same")
        print("        kinematic criteria as the RegCM tracker, so counts are")
        print("        comparable. A residual REF-vs-model difference is a genuine")
        print("        result, not a methodological artifact.")

    print()
    for p in panels:
        print(f"  {p.key:>4s}: {len(p.tracks):5d} tracks over {len(p.years)} yr")
    print()

    scf.fig_track_density(panels, OUT_DIR, STORM, notes=notes)
    scf.fig_kde_density(panels, OUT_DIR, STORM, notes=notes)
    scf.fig_genesis_density(panels, OUT_DIR, STORM, notes=notes)
    scf.fig_annual_counts(panels, OUT_DIR, STORM)
    scf.fig_storm_days(panels, OUT_DIR, STORM)
    scf.fig_seasonal_cycle(panels, OUT_DIR, STORM)
    scf.fig_duration(panels, OUT_DIR, STORM)
    scf.fig_example_year_tracks(panels, OUT_DIR, STORM, a.example_year)


if __name__ == "__main__":
    main()
