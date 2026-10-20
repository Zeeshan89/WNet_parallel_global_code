#!/usr/bin/env python3
"""
merge_and_plot_ar_3row.py — three-row AR-versus-MARS figure and skill summary.

Row 1  MARS          : AR frequency | mean AR days/yr | mean AR duration profile
Row 2  WNet (old)    : AR frequency | mean AR days/yr | per-pixel F1 vs MARS
Row 3  WNet (new)    : AR frequency | mean AR days/yr | per-pixel F1 vs MARS

Colour scales, level breaks and the WBGYR discrete colormap are copied from
regen_ar_spatial_entire.py so the panels are readable against the existing
published figure.

Sources
  new WNet : part_*.npz written by phase4_ar_vs_mars_v2.py (this directory)
  old WNet : ../ar_mars_eval/ar_eval_cache.npz  (already computed, not recomputed)
  MARS     : target counts from the new cache, plus the published days/yr file
  duration : ../ar_mars_eval/ar_duration.npz (MARS and old WNet; the new model's
             line is added when its daily-presence file has been built)

Outputs (this directory)
  ar_spatial_3row.png / .pdf
  summary_ar_v2_primary_oop.txt
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_EVAL = os.path.join(HERE, "..", "ar_mars_eval")
EPS = 1e-12

MARS_DAYS = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/data/"
             "Analysis_MARS/MARS_AvgARDaysPerYear.nc")
OLD_DAYS = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/"
            "Global_ARs_Hurr_ETCs_MCS_finalized/ARs_WNet_final/Analysis/"
            "AR_detection_AvgARDaysPerYear.nc")
N_YEARS = 45.0

WBGYR = LinearSegmentedColormap.from_list("wbgyr", [
    (0.00, "#ffffff"), (0.08, "#d8ecf9"), (0.18, "#a6d5ee"), (0.30, "#5fb0db"),
    (0.38, "#3f93c6"), (0.46, "#2fa08c"), (0.56, "#5cbf5a"), (0.64, "#a5d63f"),
    (0.72, "#f2e534"), (0.82, "#f6a623"), (0.91, "#e8551f"), (1.00, "#b00000"),
], N=256)
OVER = "#ff00d4"
LV_FREQ = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
LV_DAYS = [5, 20, 40, 60, 80, 100, 120]
LV_F1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def disc(levels):
    cmap = ListedColormap(WBGYR(np.linspace(0.06, 1.0, len(levels) - 1)))
    cmap.set_under("white")
    cmap.set_over(OVER)
    return cmap, BoundaryNorm(levels, cmap.N)


def to_pm180(field, src_lon):
    lp = np.asarray(src_lon, float)
    if lp.max() > 180:
        lp = ((lp + 180.0) % 360.0) - 180.0
    o = np.argsort(lp)
    return lp[o], np.asarray(field)[:, o]


# ── merge the new-model partials ─────────────────────────────────────────────
parts = sorted(glob.glob(os.path.join(HERE, "part_*.npz")))
if not parts:
    raise SystemExit("no part_*.npz found; run phase4_ar_vs_mars_v2.py first")
print(f"merging {len(parts)} partials")

new = None
scal = {"oop": np.zeros(5, np.int64), "primary": np.zeros(5, np.int64)}
n_days_new = 0
lat = lon = None
for p in parts:
    z = np.load(p)
    if new is None:
        lat, lon = z["lat"], z["lon"]
        new = {k: np.zeros_like(z[f"pix_{k}"]) for k in
               ("tp", "fp", "fn", "pred", "targ", "ardays")}
    for k in new:
        new[k] += z[f"pix_{k}"]
    for per in scal:
        scal[per] += z[f"{per}_scalar"]
    n_days_new += int(z["n_days"])

n_t_new = int(scal["oop"][4] + scal["primary"][4])
print(f"  new model: {n_t_new} time steps, {n_days_new} days")

# ── old model, from the existing cache ───────────────────────────────────────
zo = np.load(os.path.join(OLD_EVAL, "ar_eval_cache.npz"), allow_pickle=True)
old = {k: (zo[f"primary_pix_{k}"].astype(np.int64) + zo[f"oop_pix_{k}"].astype(np.int64))
       for k in ("tp", "fp", "fn")}
o_ps, o_op = zo["primary_scalar"], zo["oop_scalar"]
n_t_old = int(o_ps[4] + o_op[4])
print(f"  old model: {n_t_old} time steps")

# ── fields ───────────────────────────────────────────────────────────────────
freq_mars = new["targ"] / max(n_t_new, 1)
freq_new = new["pred"] / max(n_t_new, 1)
freq_old = (old["tp"] + old["fp"]) / max(n_t_old, 1)

f1_new = 2 * new["tp"] / (2 * new["tp"] + new["fp"] + new["fn"] + EPS)
f1_old = 2 * old["tp"] / (2 * old["tp"] + old["fp"] + old["fn"] + EPS)

days_new = new["ardays"] / N_YEARS

_m = xr.open_dataset(MARS_DAYS)
days_mars = np.asarray(_m["MARS"].squeeze().values, float)
mars_lon = _m.longitude.values.astype(float)
_m.close()
_o = xr.open_dataset(OLD_DAYS)
days_old = np.asarray(_o["AR_detection"].squeeze().values, float)
old_lon = _o.longitude.values.astype(float)
_o.close()

# duration profile (MARS and old model already computed; new model optional)
dur_path = os.path.join(OLD_EVAL, "ar_duration.npz")
DUR = np.load(dur_path) if os.path.exists(dur_path) else None
new_dur_path = os.path.join(HERE, "ar_duration_v2.npz")
DUR_NEW = np.load(new_dur_path) if os.path.exists(new_dur_path) else None

# ── figure ───────────────────────────────────────────────────────────────────
PC = ccrs.PlateCarree()
fig, axes = plt.subplots(3, 3, figsize=(21, 10.5),
                         subplot_kw={"projection": PC})
fig.subplots_adjust(left=0.035, right=0.975, top=0.93, bottom=0.05,
                    hspace=0.30, wspace=0.12)


def draw(ax, field, src_lon, levels, title):
    lonp, fld = to_pm180(field, src_lon)
    cmap, norm = disc(levels)
    ax.set_global()
    m = ax.pcolormesh(lonp, lat, fld, cmap=cmap, norm=norm,
                      transform=PC, shading="auto")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#555555")
    ax.set_title(title, fontsize=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    plt.colorbar(m, ax=ax, extend="both", fraction=0.026, pad=0.015)


rows = [
    ("MARS (benchmark)", freq_mars, mars_lon, days_mars, mars_lon, None, None),
    ("WNet old", freq_old, lon, days_old, old_lon, f1_old, lon),
    ("WNet new (Phase 2)", freq_new, lon, days_new, lon, f1_new, lon),
]

for r, (label, fq, fq_lon, dy, dy_lon, f1, f1_lon) in enumerate(rows):
    draw(axes[r, 0], fq, fq_lon, LV_FREQ, f"{label} — AR frequency")
    draw(axes[r, 1], dy, dy_lon, LV_DAYS, f"{label} — mean AR days/yr")
    if f1 is not None:
        draw(axes[r, 2], f1, f1_lon, LV_F1, f"{label} — per-pixel F1 vs MARS")

# duration profile replaces the empty MARS F1 slot
box = axes[0, 2].get_position()
axes[0, 2].remove()
axd = fig.add_axes([box.x0, box.y0 + 0.035, box.width * 0.80, box.height * 0.80])
if DUR is not None:
    axd.plot(DUR["zon_mars"], DUR["lat"], color="#2a78d6", lw=2, label="MARS")
    axd.plot(DUR["zon_wnet"], DUR["lat"], color="#eb6834", lw=2, label="WNet old")
if DUR_NEW is not None:
    axd.plot(DUR_NEW["zon_wnet"], DUR_NEW["lat"], color="#1baf7a", lw=2,
             label="WNet new")
else:
    axd.text(0.97, 0.02, "new-model duration pending", transform=axd.transAxes,
             ha="right", va="bottom", fontsize=8, color="#8a959b")
axd.set_title("Mean AR duration", fontsize=11)
axd.set_xlabel("Mean Duration (Days)", fontsize=9)
axd.set_ylabel("Latitude", fontsize=9)
axd.grid(alpha=0.3)
axd.legend(fontsize=9, frameon=False)
axd.tick_params(labelsize=8)

for ext in ("png", "pdf"):
    out = os.path.join(HERE, f"ar_spatial_3row.{ext}")
    fig.savefig(out, dpi=180 if ext == "png" else 300, bbox_inches="tight")
    print(f"  wrote {out}")
plt.close(fig)


# ── skill summary, same layout as summary_ar_primary_oop.txt ─────────────────
def block(name, tp, fp, fn, tn, n_t):
    iou = tp / (tp + fp + fn + EPS)
    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    pod = tp / (tp + fn + EPS)
    far = fp / (tp + fp + EPS)
    bias = (tp + fp) / (tp + fn + EPS)
    n = tp + fp + fn + tn
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n + EPS)
    acc = (tp + tn) / (n + EPS)
    hss = (acc - pe) / (1 - pe + EPS)
    return (f"{name}:\n"
            f"  IoU          = {iou:.6f}\n  Dice         = {dice:.6f}\n"
            f"  F1           = {dice:.6f}\n  Precision    = {prec:.6f}\n"
            f"  POD          = {pod:.6f}\n  FAR          = {far:.6f}\n"
            f"  BIAS         = {bias:.6f}\n  HSS          = {hss:.6f}\n"
            f"  tp           = {tp}\n  fp           = {fp}\n  fn           = {fn}\n"
            f"  n_t          = {n_t}\n")


lines = ["=" * 70,
         "WNet-v2 (Phase 2) AR Detection vs MARS (modified tARget-v3) - pixel skill",
         "=" * 70,
         "Predictions : Global_ARs_Hurr_ETCs_MCS_finalized_v2/ARs_WNet_final",
         "Target var  : MARS", ""]
for per in ("oop", "primary"):
    tp, fp, fn, tn, n_t = (int(x) for x in scal[per])
    lines.append(block(f"ar_pixel_{per}", tp, fp, fn, tn, n_t))
tot = scal["oop"] + scal["primary"]
lines.append(block("ar_pixel_entire", *(int(x) for x in tot)))

txt = os.path.join(HERE, "summary_ar_v2_primary_oop.txt")
with open(txt, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  wrote {txt}")
print("\n".join(lines[-14:]))
