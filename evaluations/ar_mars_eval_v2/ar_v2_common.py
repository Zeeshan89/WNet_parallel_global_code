"""
ar_v2_common.py — shared loading, metrics and map style for the v2 AR analysis.

The house conventions here are copied from the existing AR figures so the new
panels sit next to the published ones:
  * WBGYR discrete colormap and the LV_* level breaks, from
    ../ar_mars_eval_entire/regen_ar_spatial_entire.py
  * to_pm180 longitude rewrap, because the caches, the MARS climatology and the
    WNet climatology each carry a different longitude convention
  * the IoU / Dice / POD / FAR / BIAS / HSS block from merge_and_plot_ar_3row.py

Two things differ deliberately from the first v2 pass:
  * LV_FREQ and LV_DAYS are extended. The published breaks topped out at 0.12
    and 120, which put ~20 % of every panel into the single over-range colour --
    including the whole tropical over-detection that the figure is meant to show.
  * every metric is available cos-lat weighted as well as unweighted. On a
    721x1440 regular grid an unweighted pixel count over-weights polar rows by
    about 15x, which is enough to flip the sign of the frequency bias.
"""
import os
import glob

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

EPS = 1e-12
YEAR0, YEAR1 = 1980, 2024
NYEAR = YEAR1 - YEAR0 + 1
PERIODS = ("oop", "primary")
PERIOD_YEARS = {"oop": (1980, 2005), "primary": (2006, 2024)}
SEASONS = ("DJF", "MAM", "JJA", "SON")
PIX_KEYS = ("tp", "fp", "fn", "pred", "targ", "ardays", "ardays_t")
SEA_KEYS = ("tp", "fp", "fn", "pred", "targ")

# latitude bands used for the diagnostic tables
BANDS = [("deep tropics |lat|<15", lambda a: a < 15),
         ("subtropics 15-30", lambda a: (a >= 15) & (a < 30)),
         ("midlatitudes 30-60", lambda a: (a >= 30) & (a < 60)),
         ("polar |lat|>=60", lambda a: a >= 60)]

# ---------------------------------------------------------------- map style
WBGYR = LinearSegmentedColormap.from_list("wbgyr", [
    (0.00, "#ffffff"), (0.08, "#d8ecf9"), (0.18, "#a6d5ee"), (0.30, "#5fb0db"),
    (0.38, "#3f93c6"), (0.46, "#2fa08c"), (0.56, "#5cbf5a"), (0.64, "#a5d63f"),
    (0.72, "#f2e534"), (0.82, "#f6a623"), (0.91, "#e8551f"), (1.00, "#b00000"),
], N=256)
OVER = "#ff00d4"

# published breaks, kept for reference / reproducing the old figure
LV_FREQ_PUB = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
LV_DAYS_PUB = [5, 20, 40, 60, 80, 100, 120]
# extended breaks: measured maxima are MARS freq 0.142, v2 freq 0.533,
# v2 days/yr 236, so the old top break hid roughly a fifth of every map
LV_FREQ = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30]
LV_DAYS = [5, 20, 40, 60, 80, 100, 120, 150, 200, 250]
# paper scale for the days-per-year maps: 0 to 120 in steps of 20, one colour for
# everything above. MARS peaks at 117, so its full range is resolved; WNet storm-
# track cores (up to 236) fall in the over-range colour.
LV_DAYS_120 = [0, 20, 40, 60, 80, 100, 120]
# Seasonal frequency needs more headroom than the annual mean: measured v2 peaks
# are 0.562 (DJF), 0.472 (MAM), 0.965 (JJA), against 0.533 for the annual map.
# The JJA maximum is the Asian monsoon, where IVT sits above the AR threshold
# almost permanently, so the top bin has to reach 1.0 or the worst season is the
# one the figure cannot show.
LV_FREQ_SEA = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20,
               0.30, 0.45, 0.65, 1.00]
LV_F1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LV_DUR = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
# diverging scale for model-minus-benchmark frequency differences
LV_DIFF = [-0.12, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.12]


def disc(levels, cmap=None):
    """Discrete colormap + norm, white under the first break, magenta over."""
    base = cmap if cmap is not None else WBGYR
    cm = ListedColormap(base(np.linspace(0.06, 1.0, len(levels) - 1)))
    cm.set_under("white")
    cm.set_over(OVER)
    return cm, BoundaryNorm(levels, cm.N)


def disc_diff(levels):
    """Symmetric blue-white-red for differences; over/under are the end colours."""
    import matplotlib.pyplot as plt
    cm = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.03, 0.97, len(levels) - 1)))
    cm.set_under(plt.get_cmap("RdBu_r")(0.0))
    cm.set_over(plt.get_cmap("RdBu_r")(1.0))
    return cm, BoundaryNorm(levels, cm.N)


def to_pm180(field, src_lon):
    """Rewrap 0..360 longitudes to -180..180 and reorder the field to match."""
    lp = np.asarray(src_lon, float)
    if lp.max() > 180:
        lp = ((lp + 180.0) % 360.0) - 180.0
    o = np.argsort(lp)
    return lp[o], np.asarray(field)[..., o]


def cos_w(lat):
    """cos-lat area weights, shaped to broadcast over (lat, lon) maps."""
    return np.cos(np.deg2rad(np.asarray(lat, float)))[:, None]


def audit(name, field, levels, limit=0.5):
    """Report how much of a panel falls past the top break.

    The first version of this figure put 19.2 % of the v2 frequency map and
    7.0 % of its days/yr map into the single over-range colour, which is where
    the tropical over-detection lives. Anything above `limit` percent means the
    levels need extending, not that the panel is fine.
    """
    f = np.asarray(field, float)
    over = 100.0 * float(np.nanmean(f > levels[-1]))
    flag = "  <-- EXTEND LEVELS" if over > limit else ""
    print(f"  audit {name:34s} max={np.nanmax(f):9.4f}  over-top={over:5.2f}%{flag}")
    return over


# ---------------------------------------------------------------- metrics
def metrics(tp, fp, fn, tn=None):
    """Contingency skill. Same algebra as the published summary block."""
    tp, fp, fn = float(tp), float(fp), float(fn)
    out = {
        "IoU": tp / (tp + fp + fn + EPS),
        "F1": 2 * tp / (2 * tp + fp + fn + EPS),
        "Precision": tp / (tp + fp + EPS),
        "POD": tp / (tp + fn + EPS),
        "FAR": fp / (tp + fp + EPS),
        "BIAS": (tp + fp) / (tp + fn + EPS),
    }
    out["Dice"] = out["F1"]
    if tn is not None:
        tn = float(tn)
        n = tp + fp + fn + tn
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n + EPS)
        acc = (tp + tn) / (n + EPS)
        out["HSS"] = (acc - pe) / (1 - pe + EPS)
    return out


def map_metrics(tp, fp, fn, w=None, mask=None):
    """Aggregate per-pixel maps into one skill dict, optionally weighted/masked."""
    ww = 1.0 if w is None else w
    sel = (slice(None),) if mask is None else (mask,)
    a = (tp * ww)[sel] if mask is not None else tp * ww
    b = (fp * ww)[sel] if mask is not None else fp * ww
    c = (fn * ww)[sel] if mask is not None else fn * ww
    return metrics(a.sum(), b.sum(), c.sum())


# ---------------------------------------------------------------- cache loading
def load_parts(cache_dir):
    """Merge every part_*.npz in cache_dir into one dict of summed arrays."""
    parts = sorted(glob.glob(os.path.join(cache_dir, "part_*.npz")))
    if not parts:
        raise SystemExit(f"no part_*.npz in {cache_dir}; run phase4_ar_vs_mars_v2.py")

    out = {"n_days": 0, "per": {}, "sea": {}, "scal": {}, "sea_scal": {}}
    month_rows, ek, ef, el, eft, elt = [], [], [], [], [], []
    for i, p in enumerate(parts):
        z = np.load(p)
        if i == 0:
            out["lat"], out["lon"] = z["lat"], z["lon"]
            out["per"] = {q: {k: np.zeros_like(z[f"pix_{q}_{k}"]) for k in PIX_KEYS}
                          for q in PERIODS}
            out["sea"] = {s: {k: np.zeros_like(z[f"sea_{s}_{k}"]) for k in SEA_KEYS}
                          for s in SEASONS}
            out["scal"] = {q: np.zeros(5, np.int64) for q in PERIODS}
            out["sea_scal"] = {s: np.zeros(5, np.int64) for s in SEASONS}
            out["yr_days"] = np.zeros_like(z["yr_days"], np.int32)
            out["yr_days_t"] = np.zeros_like(z["yr_days_t"], np.int32)
            out["rise"] = np.zeros_like(z["rise"], np.int64)
            out["rise_t"] = np.zeros_like(z["rise_t"], np.int64)
        for q in PERIODS:
            for k in PIX_KEYS:
                out["per"][q][k] += z[f"pix_{q}_{k}"]
            out["scal"][q] += z[f"{q}_scalar"]
        for s in SEASONS:
            for k in SEA_KEYS:
                out["sea"][s][k] += z[f"sea_{s}_{k}"]
            out["sea_scal"][s] += z[f"{s}_scalar"]
        out["yr_days"] += z["yr_days"]
        out["yr_days_t"] += z["yr_days_t"]
        out["rise"] += z["rise"]
        out["rise_t"] += z["rise_t"]
        out["n_days"] += int(z["n_days"])
        if z["month_rows"].size:
            month_rows.append(z["month_rows"])
        if z["edge_keys"].size:
            ek.append(z["edge_keys"]); ef.append(z["edge_first"])
            el.append(z["edge_last"]); eft.append(z["edge_first_t"])
            elt.append(z["edge_last_t"])
        z.close()

    out["month_rows"] = (np.concatenate(month_rows) if month_rows
                         else np.zeros((0, 7), np.int64))
    out["month_rows"] = out["month_rows"][np.lexsort(
        (out["month_rows"][:, 1], out["month_rows"][:, 0]))]
    out["edge"] = dict(keys=np.concatenate(ek) if ek else np.zeros((0, 2), np.int64),
                       first=np.concatenate(ef) if ef else None,
                       last=np.concatenate(el) if el else None,
                       first_t=np.concatenate(eft) if eft else None,
                       last_t=np.concatenate(elt) if elt else None)
    # totals over both periods = the legacy whole-record maps
    out["all"] = {k: sum(out["per"][q][k] for q in PERIODS) for k in PIX_KEYS}
    out["n_t"] = int(sum(out["scal"][q][4] for q in PERIODS))
    out["n_parts"] = len(parts)
    return out


def duration(cache, which="pred"):
    """Mean AR event duration in days per pixel, plus its zonal mean.

    Reproduces compute_ar_duration.py: duration = total AR-days / number of
    0->1 daily transitions. The within-month edges were counted in the MPI pass;
    the edges that fall on a month boundary are stitched here from the stored
    first/last day masks. The first day of the whole record is not treated as a
    transition, which is what `diff` does in the original script.
    """
    shp = cache["rise"].shape
    npix = int(np.prod(shp))
    total = cache["all"]["ardays" if which == "pred" else "ardays_t"].astype(float)
    rising = (cache["rise"] if which == "pred" else cache["rise_t"]).astype(float).copy()

    e = cache["edge"]
    if e["keys"].size:
        order = np.lexsort((e["keys"][:, 1], e["keys"][:, 0]))
        fkey = "first" if which == "pred" else "first_t"
        lkey = "last" if which == "pred" else "last_t"
        first = e[fkey][order]
        last = e[lkey][order]

        def unpack(row):
            return np.unpackbits(row, count=npix).reshape(shp).astype(bool)

        prev_last = None
        for i in range(len(order)):
            f = unpack(first[i])
            if prev_last is not None:
                rising += (f & ~prev_last)
            prev_last = unpack(last[i])

    dur = np.where(rising > 0, total / np.maximum(rising, EPS), np.nan)
    with np.errstate(invalid="ignore"):
        zon = np.nanmean(dur, axis=1)
    return dur.astype(np.float32), zon.astype(np.float32)
