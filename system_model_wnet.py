"""
Illustrated system-model diagram for the Phase 2 multi-label W-Net storm detector.

    python system_model_wnet.py              # with in-figure title
    python system_model_wnet.py --no-title   # paper version (caption carries it)

Writes system_model_wnet[_notitle].{png,pdf} next to this script. Four numbered bands read top
to bottom (inputs, preprocess, W-Net, output), the idiom of a system-architecture figure: real
ERA5 field maps and label masks from one cached time step, the two U-Nets drawn block by block
with their true widths and grid sizes, a real four-class detection map from the selected
checkpoint, and a dashed training-only strip for the loss and compute. Every data read has a
constant fallback, so the script always renders. Facts follow METHODOLOGY.md. No em-dashes in
rendered text.
"""
import os
import glob
import textwrap
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, FancyArrow, Rectangle, Circle, Ellipse, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.dirname(HERE)
OUT_DIR = HERE

# ── data locations (all optional) ─────────────────────────────────────────────────────────
# the 0.25 deg cache the shipped model was trained on (721 x 1440; the figure crops the -90 row
# to the 720 x 1440 the network sees, as --crop-rows 720 does in training)
CACHE_DIR = os.path.join(NB, "tensor_cache_v11_fullres")
CROP_ROWS = 720                                           # None keeps all 721 rows
MONTH, INDEX = "202107", 123                              # 2021-07-31 18 UTC
TS_FALLBACK = "2021-07-31 18 UTC"
PRED_NPZ = os.path.join(HERE, "system_model_wnet_pred_2021073118.npz")
EXPORTS = {                                               # fallback for the output map
    "AR": os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_fullres025_V2", "ARs_WNet_final", f"era5dara_{MONTH}_AR_detection.nc"),
    "TC": os.path.join(NB, "Global_pw50_e200", "Hurricanes_WNet_final", f"era5dara_{MONTH}_Hurricane_detection.nc"),
    "ETC": os.path.join(NB, "Global_pw50_e200", "ETCs_WNet_final", f"era5dara_{MONTH}_ETCs_detection.nc"),
    "MCS": os.path.join(NB, "Global_ARs_Hurr_ETCs_MCS_fullres025_V2", "MCS_WNet_final", f"era5dara_{MONTH}_MCS_detection.nc"),
}
EXPORT_VARS = {"AR": "AR_detection", "TC": "Hurricane_detection", "ETC": "ETCs_detection", "MCS": "MCS_detection"}

# ── palette ───────────────────────────────────────────────────────────────────────────────
INK, RULE = "#1c2733", "#cfd7de"
BODY = MUTED = INK          # all text in ink: grey body text read as faded in print
SKIP = "#6f7e8c"            # skip-connection arrows keep a lighter grey (lines, not text)
BAND = {                       # accent, tint
    "inputs":   ("#3d6d9e", "#e8eff7"),
    "harness":  ("#4a5a6a", "#eef1f4"),
    "analysis": ("#0072b2", "#e3eef8"),
    "outputs":  ("#c58a1f", "#fbf3e3"),
}
LANE = {"A": ("#0072b2", "#e9f2fa"), "B": ("#d55e00", "#fdeee4")}
CLASSES = [("AR", "#0072b2", "Atmospheric rivers"),
           ("TC", "#d55e00", "Tropical cyclones"),
           ("ETC", "#7b4fa6", "Extratropical cyclones"),
           ("MCS", "#009e73", "Mesoscale convective systems")]
CCOL = {k: c for k, c, _ in CLASSES}
SHADOW = [pe.withSimplePatchShadow(offset=(0.6, -0.8), shadow_rgbFace="#1c2733", alpha=0.10)]
W = 100.0

# type scale (pt) at a 13in canvas
F_BAND, F_TITLE, F_SUB, F_LEAD, F_BODY, F_SMALL = 14.0, 12.8, 10.6, 11.6, 10.4, 9.2

FIELDS = [  # cache channel, tile name, role, colormap, vmin, vmax
    (0, "IVT", "", "Blues", 0, 700),
    (1, "MSLP", "", "RdBu_r", 98500, 103500),
    (2, "Vorticity at 850 hPa", "", "RdBu_r", -1.2e-4, 1.2e-4),
    (3, "wind at 850 hPa", "", "YlGnBu", 0, 25),
    (4, "BT", "", "Greys_r", 205, 285),
    (5, "IWV", "", "BuGn", 0, 65),
]


def _register_fonts():
    """matplotlib's font cache may not list Liberation although the files are installed."""
    for p in glob.glob("/usr/share/fonts/truetype/Liberation*.ttf"):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass


# ── data (all optional; constants stand in when a file is absent) ─────────────────────────
def _smooth_noise(rng, shape=(720, 1440), scale=45):
    z = rng.normal(size=(shape[0] // scale + 2, shape[1] // scale + 2))
    z = np.kron(z, np.ones((scale, scale)))[:shape[0], :shape[1]]
    for _ in range(3):
        z = (z + np.roll(z, 3, 0) + np.roll(z, -3, 0) + np.roll(z, 3, 1) + np.roll(z, -3, 1)) / 5
    return (z - z.min()) / (z.max() - z.min() + 1e-9)


def _load_step(cache_dir):
    """(raw fields (6,H,W), label masks (4,H,W), timestamp, is_real)."""
    p = os.path.join(cache_dir, f"split_{MONTH}.npy")
    m = os.path.join(cache_dir, f"split_{MONTH}.meta.npy")
    if os.path.exists(p):
        try:
            a = np.load(p, mmap_mode="r")
            x = np.array(a[INDEX], dtype=np.float32)
            ts = TS_FALLBACK
            if os.path.exists(m):
                ts = str(np.load(m, allow_pickle=True)[INDEX])[:16].replace("T", " ") + " UTC"
            if CROP_ROWS:                 # what the network is given: the -90 row dropped
                x = x[:, :CROP_ROWS]
            return x[:6], x[6:10] > 0.5, ts, True
        except Exception:
            pass
    rng = np.random.default_rng(3)
    lo = np.array([0, 98500, -1.2e-4, 0, 205, 0], dtype=np.float32)
    hi = np.array([700, 103500, 1.2e-4, 25, 285, 65], dtype=np.float32)
    x = np.stack([lo[i] + (hi[i] - lo[i]) * _smooth_noise(rng) for i in range(6)])
    lab = np.stack([_smooth_noise(rng, scale=60) > q for q in (0.78, 0.9, 0.8, 0.84)])
    return x, lab, TS_FALLBACK, False


def _load_prediction():
    """(binary maps (4,h,w) or None, reconstruction (6,H,W) or None, source tag)."""
    if os.path.exists(PRED_NPZ):
        try:
            d = np.load(PRED_NPZ)
            rec = d["rec"].astype(np.float32) if "rec" in d else None
            return d["preds"].astype(bool), rec, "model"
        except Exception:
            pass
    try:
        import h5py
        maps = []
        for key, _, _ in CLASSES:
            with h5py.File(EXPORTS[key], "r") as f:
                maps.append(np.asarray(f[EXPORT_VARS[key]][INDEX][::4, ::4]) > 0.5)
        return np.stack(maps), None, "export"
    except Exception:
        return None, None, "none"


# ── drawing helpers ───────────────────────────────────────────────────────────────────────
def wrap(s, n):
    """Wrap to n characters; explicit newlines in s are kept as forced breaks."""
    if "$" in s:
        return s
    return "\n".join("\n".join(textwrap.wrap(part, n, break_on_hyphens=False)) if part else ""
                     for part in s.split("\n"))


class Canvas:
    def __init__(self, fig, ax, H):
        self.fig, self.ax, self.H = fig, ax, H

    def inset(self, x, y, w, h):
        """Axes placed in canvas units (for mini-plots and thumbnails)."""
        a = self.fig.add_axes([x / W, y / self.H, w / W, h / self.H])
        a.set_facecolor("white")
        for s in a.spines.values():
            s.set_color(RULE); s.set_linewidth(0.8)
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return a


def band(ax, x, y, w, h, num, title, key):
    acc, tint = BAND[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=tint, ec=acc, lw=1.3, zorder=1))
    pw = 0.66 * len(title) + 6.4
    ax.add_patch(FancyBboxPatch((x + 2.0, y + h - 1.45), pw, 2.9,
                                boxstyle="round,pad=0,rounding_size=1.45", fc=acc, ec="none",
                                zorder=4))
    ax.add_patch(Circle((x + 3.75, y + h), 1.05, fc="white", ec="none", zorder=5))
    ax.text(x + 3.75, y + h, num, color=acc, fontsize=F_BAND - 1, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(x + 5.6, y + h, title, color="white", fontsize=F_BAND, fontweight="bold",
            ha="left", va="center", zorder=6)


def card(ax, x, y, w, h, fc="white", ec=RULE, lw=1.1, ls="-", shadow=True, r=0.9, z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, ls=ls, zorder=z)
    if shadow and ls == "-":
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def arrow(ax, p, q, color=MUTED, rad=0.0, lw=1.8, ms=15, ls="-", z=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=0, shrinkB=0))


def chevron(ax, x, y_top, y_bot, color, label=None):
    """Wide block arrow pointing down between two bands."""
    ax.add_patch(FancyArrow(x, y_top, 0, y_bot - y_top, width=3.2, head_width=6.4,
                            head_length=1.15, length_includes_head=True, fc=color, ec="none",
                            alpha=0.92, zorder=7))
    if label:
        ax.text(x + 4.2, (y_top + y_bot) / 2, label, fontsize=F_SMALL + 0.4, color=BODY,
                ha="left", va="center", zorder=7, style="italic")


def pill(ax, x, y, title, col, letter=None, fs=F_SUB):
    pw = 0.62 * len(title) + (5.6 if letter else 2.4)
    ax.add_patch(FancyBboxPatch((x, y), pw, 2.5, boxstyle="round,pad=0,rounding_size=1.25",
                                fc=col, ec="none", zorder=5))
    tx = x + 1.2
    if letter:
        ax.add_patch(Circle((x + 1.5, y + 1.25), 0.95, fc="white", ec="none", zorder=6))
        ax.text(x + 1.5, y + 1.25, letter, color=col, fontsize=F_SUB, fontweight="bold",
                ha="center", va="center", zorder=7)
        tx = x + 3.1
    ax.text(tx, y + 1.25, title, color="white", fontsize=fs + 0.6, fontweight="bold",
            ha="left", va="center", zorder=7)
    return pw


def dot(ax, x, y, col, r=0.85, text=None):
    ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=6))
    if text:
        ax.text(x, y, text, color="white", fontsize=F_SMALL - 2.4, fontweight="bold",
                ha="center", va="center", zorder=7)


def framed_inset(cv, ax, x, y, w, h):
    """Thumbnail frame with the soft drop shadow used for the real crops."""
    ax.add_patch(Rectangle((x + 0.3, y - 0.4), w + 0.5, h + 0.5, fc="#1c2733", ec="none",
                           alpha=0.10, zorder=3))
    ax.add_patch(Rectangle((x - 0.25, y - 0.25), w + 0.5, h + 0.5, fc="white", ec=RULE,
                           lw=0.9, zorder=4))
    a = cv.inset(x, y, w, h)
    a.set_zorder(5)
    for s in a.spines.values():
        s.set_color("white"); s.set_linewidth(1.0)
    return a


def map_inset(cv, ax, x, y, w, h, field, cmap, vmin, vmax):
    a = framed_inset(cv, ax, x, y, w, h)
    a.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear",
             rasterized=True)
    return a


def mask_rgba(masks, cols, base="#f3f5f7", alpha=1.0):
    """RGBA composite of binary masks in their colours over a flat base."""
    h, w = masks[0].shape
    img = np.ones((h, w, 3)) * np.array(to_rgb(base))
    for m, c in zip(masks, cols):
        img[m] = (1 - alpha) * img[m] + alpha * np.array(to_rgb(c))
    return img


def _thicken(mask, k=3):
    """Grow a mask by k cells for DISPLAY only. At 0.25 deg a thumbnail is a few hundred pixels
    wide, so the sparse classes (TC discs above all) fall below one pixel and vanish."""
    if k <= 1:
        return mask
    m = np.asarray(mask, bool)
    out = m.copy()
    for s in range(1, k):
        out |= np.roll(m, s, 0) | np.roll(m, -s, 0) | np.roll(m, s, 1) | np.roll(m, -s, 1)
    return out


def mask_inset(cv, ax, x, y, w, h, mask, col, thicken=1):
    a = framed_inset(cv, ax, x, y, w, h)
    a.imshow(mask_rgba([_thicken(mask, thicken)], [col]), aspect="auto",
             interpolation="nearest", rasterized=True)
    return a


# ── pictograms ────────────────────────────────────────────────────────────────────────────
def glyph_cylinder(ax, x, y, w, h, col):
    ry = w * 0.18
    ax.add_patch(Rectangle((x, y + ry), w, h - 2 * ry, fc=col, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + ry), w, 2 * ry, fc=col, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + h - ry), w, 2 * ry, fc="white", ec=col, lw=1.2, zorder=6))
    ax.plot([x, x], [y + ry, y + h - ry], color=col, lw=1.2, zorder=6)
    ax.plot([x + w, x + w], [y + ry, y + h - ry], color=col, lw=1.2, zorder=6)


def glyph_doc(ax, x, y, w, h, col):
    f = 0.9
    ax.add_patch(Polygon([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h), (x, y + h)],
                         closed=True, fc="white", ec=col, lw=1.2, zorder=5))
    ax.add_patch(Polygon([(x + w - f, y), (x + w - f, y + f), (x + w, y + f)], closed=True,
                         fc=col, ec="none", alpha=0.5, zorder=6))
    for k in range(4):
        ax.plot([x + 0.7, x + w - 0.7 - (0.9 if k == 3 else 0)], [y + h - 1.0 - k * 0.85] * 2,
                color=col, lw=1.0, alpha=0.6, zorder=6)


def glyph_gpus(ax, x, y, w, h, col):
    cw, ch, g = (w - 3 * 0.3) / 4, (h - 0.3) / 2, 0.3
    for i in range(2):
        for j in range(4):
            ax.add_patch(FancyBboxPatch((x + j * (cw + g), y + i * (ch + g)), cw, ch,
                                        boxstyle="round,pad=0,rounding_size=0.25", fc=col,
                                        ec="none", alpha=0.85, zorder=5))
            ax.plot([x + j * (cw + g) + 0.25, x + j * (cw + g) + cw - 0.25],
                    [y + i * (ch + g) + ch * 0.5] * 2, color="white", lw=0.9, zorder=6)


def glyph_regrid(ax, x, y, w, h, col, fine=8, coarse=2, fine_first=True):
    """A fine grid fading into a coarse one (or the reverse), for the two regrid steps."""
    ax.add_patch(Rectangle((x, y), w, h, fc=col, ec="none", alpha=0.12, zorder=5))
    for k in range(1, fine):
        a = 0.35 if fine_first else 0.2
        ax.plot([x + k * w / fine] * 2, [y, y + h], color=col, lw=0.5, alpha=a, zorder=6)
        ax.plot([x, x + w], [y + k * h / fine] * 2, color=col, lw=0.5, alpha=a, zorder=6)
    for k in range(1, coarse):
        ax.plot([x + k * w / coarse] * 2, [y, y + h], color=col, lw=1.3, zorder=6)
        ax.plot([x, x + w], [y + k * h / coarse] * 2, color=col, lw=1.3, zorder=6)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=col, lw=1.2, zorder=6))


def glyph_bell(ax, x, y, w, h, col):
    """Wide raw distribution and the narrow standardized one."""
    t = np.linspace(-3, 3, 80)
    ax.plot(x + (t + 3) / 6 * w, y + 0.35 * h * np.exp(-(t + 0.6) ** 2 / 2.4), color=RULE,
            lw=1.4, zorder=5)
    ax.plot(x + (t + 3) / 6 * w, y + h * np.exp(-t ** 2 / 0.5), color=col, lw=1.5, zorder=6)
    ax.plot([x, x + w], [y, y], color=col, lw=0.9, zorder=6)
    ax.plot([x + w / 2] * 2, [y, y + h * 1.05], color=col, lw=0.7, ls=(0, (2, 1.5)), zorder=6)


def glyph_years(ax, x, y, w, h, cols, val, test, start=1980, n=45, per_row=15):
    """One square per year of the record, coloured by split (the actual years)."""
    rows = int(np.ceil(n / per_row))
    g = 0.14
    cw, ch = (w - g * (per_row - 1)) / per_row, (h - g * (rows - 1)) / rows
    for k in range(n):
        yr = start + k
        r, c = k // per_row, k % per_row
        col = cols["test"] if yr in test else cols["val"] if yr in val else cols["train"]
        ax.add_patch(Rectangle((x + c * (cw + g), y + h - (r + 1) * ch - r * g), cw, ch, fc=col,
                               ec="none", zorder=5))


def glyph_step(ax, x, y, s, col):
    """A sigmoid curve cut at one half: the detection threshold."""
    t = np.linspace(-5, 5, 60)
    p = 1 / (1 + np.exp(-t))
    ax.plot(x + (t + 5) / 10 * s, y + p * s, color=RULE, lw=1.4, zorder=5)
    ax.plot([x, x + s], [y + s / 2] * 2, color=col, lw=0.9, ls=(0, (2, 1.5)), zorder=6)
    ax.plot([x + s / 2, x + s / 2], [y, y + s], color=col, lw=0.9, ls=(0, (2, 1.5)), zorder=6)
    ax.plot([x + s / 2, x + s], [y + s, y + s], color=col, lw=2.2, zorder=6)
    ax.plot([x, x + s / 2], [y, y], color=col, lw=2.2, zorder=6)


def glyph_stack(ax, x, y, w, h, cols):
    """Four offset map slabs, one per class."""
    n = len(cols)
    for k, c in enumerate(cols):
        dx = (n - 1 - k) * 0.55
        ax.add_patch(FancyBboxPatch((x + dx, y + k * 0.55), w - (n - 1) * 0.55, h - (n - 1) * 0.55,
                                    boxstyle="round,pad=0,rounding_size=0.3", fc=c, ec="white",
                                    lw=0.9, alpha=0.9, zorder=5 + k))


def formula(ax, x, y, s, col, fs=11.0):
    ax.text(x, y, s, fontsize=fs, color=col, ha="center", va="center", zorder=6)


# ── the U-Net drawer ──────────────────────────────────────────────────────────────────────
def ublock(ax, x, y, w, h, title, sub, fc, ec, tc=INK, sc=MUTED, r=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec, lw=1.0, zorder=5))
    ax.text(x + w / 2, y + h * 0.66, title, fontsize=F_SMALL - 0.6, fontweight="bold", color=tc,
            ha="center", va="center", zorder=6)
    ax.text(x + w / 2, y + h * 0.28, sub, fontsize=F_SMALL - 2.2, color=sc, ha="center",
            va="center", zorder=6, family="Liberation Mono")


def draw_unet(ax, x0, y0, w, h, letter, tag, c_in, c_out, pill_title, caption):
    """One encoder-decoder U with its real widths, grids, pools, upsamples and skips."""
    col, tint = LANE[letter]
    bw, bh, stag = 10.2, 2.9, 2.2
    ys = [y0 + h - 5.6 - 4.4 * i for i in range(4)]
    widths, grids = [64, 128, 256, 512], ["720x1440", "360x720", "180x360", "90x180"]
    enc_x = [x0 + 1.6 + stag * i for i in range(3)]
    dec_x = [x0 + w - 1.6 - bw - stag * i for i in range(3)]
    bx, bwid = x0 + 10.6, w - 21.2
    # encoder
    for i in range(3):
        ublock(ax, enc_x[i], ys[i] - bh / 2, bw, bh, f"Enc {tag}.{i} · {widths[i]}",
               f"{c_in if i == 0 else widths[i - 1]} to {widths[i]} | {grids[i]}", tint, col)
    ublock(ax, bx, ys[3] - bh / 2, bwid, bh, "Bottleneck · 512", f"256 to 512 | {grids[3]}", col, col,
           tc="white", sc="white")
    # decoder
    for i in range(3):
        ublock(ax, dec_x[i], ys[i] - bh / 2, bw, bh, f"Dec {tag}.{i} · {widths[i]}",
               f"{widths[i + 1]}+{widths[i]} to {widths[i]} | {grids[i]}", "white", col)
    # pools (down) and bilinear upsamples (up)
    for i in range(3):
        cx = enc_x[i + 1] + bw / 2 if i < 2 else enc_x[2] + bw / 2
        arrow(ax, (cx, ys[i] - bh / 2), (cx, ys[i + 1] + bh / 2), color=col, lw=1.5, ms=10)
        cx = dec_x[i + 1] + bw / 2 if i < 2 else dec_x[2] + bw / 2
        arrow(ax, (cx, ys[i + 1] + bh / 2), (cx, ys[i] - bh / 2), color=col, lw=1.5, ms=10)
    ax.text(enc_x[1] + bw / 2 - 0.8, (ys[0] + ys[1]) / 2, "max-pool ½", fontsize=F_SMALL - 2.4,
            color=col, ha="right", va="center", zorder=7)
    ax.text(dec_x[1] + bw / 2 + 0.8, (ys[0] + ys[1]) / 2, "bilinear ×2", fontsize=F_SMALL - 2.4,
            color=col, ha="left", va="center", zorder=7)
    # skips
    for i in range(3):
        p, q = (enc_x[i] + bw, ys[i]), (dec_x[i], ys[i])
        arrow(ax, p, q, color=SKIP, lw=1.0, ms=9, ls=(0, (3, 2)))
    ax.text((enc_x[0] + bw + dec_x[0]) / 2, ys[0] + 0.85, "skip", fontsize=F_SMALL - 2.4,
            color=MUTED, ha="center", va="center", zorder=7, style="italic")
    # caption
    ax.text(x0 + w / 2, y0 + 1.35, caption, fontsize=F_SMALL - 1.6, color=MUTED, ha="center",
            va="center", zorder=6, linespacing=1.15)
    return dict(ys=ys, enc_x=enc_x, dec_x=dec_x, bw=bw, bh=bh, c_out=c_out)


# ── figure ────────────────────────────────────────────────────────────────────────────────
def main(with_title=True, stem="system_model_wnet", cache_dir=CACHE_DIR, out_dir=OUT_DIR):
    _register_fonts()
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],
                         "text.color": INK, "axes.edgecolor": "none",
                         "mathtext.fontset": "dejavusans"})
    H = 93.0 if with_title else 89.0     # no-title: just clears the band-1 tab (top 88.95)
    fig = plt.figure(figsize=(13.0, 13.0 * H / W), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")
    cv = Canvas(fig, ax, H)

    fields, labels, ts, real = _load_step(cache_dir)
    preds, rec, pred_src = _load_prediction()

    if with_title:
        ax.text(1.5, 91.4, "W-Net storm detection system model", fontsize=19, fontweight="bold",
                color=INK, va="center", ha="left")
        ax.text(1.5, 89.5, "Six ERA5 fields in, four independent storm maps out; the four labels "
                "are seen only by the loss.", fontsize=F_SUB + 0.4, color=MUTED, va="center",
                ha="left")

    # ══ band 1 · inputs ═══════════════════════════════════════════════════════════════════
    B1 = (1.5, 75.0, 97.0, 12.5)
    band(ax, *B1, "1", "Inputs", "inputs")
    acc1 = BAND["inputs"][0]
    card(ax, 3.0, 76.6, 21.0, 8.6, ec=acc1, lw=1.2)
    glyph_cylinder(ax, 4.4, 78.4, 3.6, 4.8, acc1)
    ax.text(9.6, 83.5, "ERA5 reanalysis", fontsize=F_TITLE, fontweight="bold", color=INK,
            ha="left", va="center", zorder=6)
    ax.text(9.6, 81.5, "1980 to 2024 at 6-hourly", fontsize=F_SUB - 0.4, color=BODY,
            ha="left", va="center", zorder=6)
    ax.text(9.6, 79.7, "0.25° grid · 721 × 1440 cells", fontsize=F_SMALL, color=MUTED,
            ha="left", va="center", zorder=6)
    # ax.text(9.6, 78.1, "65,748 time steps · 540 months", fontsize=F_SMALL, color=MUTED,
    #         ha="left", va="center", zorder=6)
    ax.text(26.5, 85.7, "6 physical fields", fontsize=F_SUB, fontweight="bold",
            color=INK, ha="left", va="center", zorder=6)
    # ax.text(96.9, 85.7, "each enters at least one storm definition", fontsize=F_SMALL,
    #         color=MUTED, ha="right", va="center", zorder=6)
    for i, (ch, name, role, cmap, vmin, vmax) in enumerate(FIELDS):
        x = 26.5 + i * 11.8
        map_inset(cv, ax, x, 79.0, 10.4, 5.2, fields[ch], cmap, vmin, vmax)
        ax.text(x + 5.2, 78.0, name, fontsize=F_SMALL, fontweight="bold", color=INK,
                ha="center", va="center", zorder=6)
        ax.text(x + 5.2, 76.85, role, fontsize=F_SMALL - 1.4, color=MUTED, ha="center",
                va="center", zorder=6)

    # ══ band 2 · preprocess ═══════════════════════════════════════════════════════════════
    B2 = (1.5, 54.0, 97.0, 19.0)
    band(ax, *B2, "2", "Preprocess", "harness")
    acc2 = BAND["harness"][0]
    # three step cards spread evenly over the same span as the labels card below (x 3 to 97)
    bw, by, bh = 29.3, 65.0, 6.8
    bx = [3.0, 3.0 + bw + 3.05, 3.0 + 2 * (bw + 3.05)]
    steps = [("Recompute IVT", ""),
             ("Standardize", "Per-field mean and std"),
             ("Cache and split", "Monthly tensors, 721 to 720 rows;\n"
                                 "Split by year, 80% / 9% / 11%")]
    for i, (t, s) in enumerate(steps):
        card(ax, bx[i], by, bw, bh, ec=acc2, lw=1.1)
        if s:
            ax.text(bx[i] + 8.2, by + bh - 1.5, t, fontsize=F_TITLE - 0.8, fontweight="bold",
                    color=INK, ha="left", va="center", zorder=6)
            ax.text(bx[i] + 8.2, by + bh - 4.15, wrap(s, 36), fontsize=F_SMALL + 0.2, color=BODY,
                    ha="left", va="center", zorder=6, linespacing=1.25)
        else:   # no subtitle: centre the title in the card
            ax.text(bx[i] + 8.2, by + bh / 2, wrap(t, 24), fontsize=F_TITLE - 0.8,
                    fontweight="bold", color=INK, ha="left", va="center", zorder=6,
                    linespacing=1.15)
        if i < len(steps) - 1:
            arrow(ax, (bx[i] + bw, by + bh / 2), (bx[i + 1], by + bh / 2), color=acc2, lw=2.0)
    formula(ax, bx[0] + 4.1, by + 3.3, r"$\sqrt{Q_u^2 + Q_v^2}$", acc2, fs=11.5)
    glyph_bell(ax, bx[1] + 1.2, by + 1.5, 5.8, 3.6, acc2)
    split_cols = {"train": acc2, "val": LANE["B"][0], "test": CCOL["MCS"]}
    glyph_years(ax, bx[2] + 1.2, by + 1.9, 6.0, 3.4, split_cols, per_row=9,
                val=(1987, 1998, 2007, 2016), test=(1983, 1994, 2003, 2012, 2021))
    for k, (lab, key) in enumerate([("train", "train"), ("val", "val"), ("test", "test")]):
        dot(ax, bx[2] + 1.6 + k * 2.1, by + 1.0, split_cols[key], r=0.32)
        ax.text(bx[2] + 2.05 + k * 2.1, by + 1.0, lab, fontsize=F_SMALL - 3.0, color=MUTED,
                ha="left", va="center", zorder=6)
    # labels: training-only card (the split contract)
    card(ax, 3.0, 55.2, 94.0, 8.6, fc="#fafbfc", ec=acc2, lw=1.0, ls=(0, (3, 2)), shadow=False)
    pill(ax, 4.2, 60.6, "TRAINING ONLY: storm labels from physical criteria, seen only by the loss",
         acc2, fs=F_SUB - 1.0)
    # the four label definitions carried by tensor_cache_v11_fullres (build_fullres_cache.py):
    # AR = L6, TC = IBTrACS discs, ETC = closed contour, MCS unchanged
    defs = {"AR": "IVT above the local seasonal 85th percentile, long, narrow and poleward",
            "TC": "Pressure minimum with spin and wind",
            "ETC": "Closed pressure contour around a surface low",
            "MCS": "Cold cloud-top core and shield, moist for the season"}
    for i, (key, col, name) in enumerate(CLASSES):
        gx = 4.2 + i * 23.4
        mask_inset(cv, ax, gx, 55.9, 8.0, 4.0, labels[i], col,
                   thicken=4 if labels[i].shape[0] > 400 else 1)
        ax.text(gx + 9.2, 59.25, wrap(name, 22), fontsize=F_SUB - 0.6, fontweight="bold",
                color=col, ha="left", va="center", zorder=6, linespacing=1.05)
        ax.text(gx + 9.2, 57.25, wrap(defs[key], 30), fontsize=F_SMALL - 1.2, color=BODY,
                ha="left", va="center", zorder=6, linespacing=1.2)
    # dashed route of the labels down the right margin to the segmentation loss
    ax.plot([96.0, 96.0], [55.2, 22.7], color=acc2, lw=1.3, ls=(0, (4, 2.5)), zorder=6)
    arrow(ax, (96.0, 22.7), (94.6, 22.7), color=acc2, lw=1.3, ms=11, ls=(0, (4, 2.5)))
    ax.text(95.2, 53.0, "4 × 720 × 1440 labels, to the loss only", fontsize=F_SMALL, color=INK,
            ha="right", va="center", zorder=7, style="italic")

    # ══ band 3 · W-Net ════════════════════════════════════════════════════════════════════
    B3 = (1.5, 18.0, 97.0, 34.0)
    band(ax, *B3, "3", "W-Net", "analysis")
    acc3 = BAND["analysis"][0]
    LA = (3.0, 26.8, 39.0, 23.0)
    LB = (56.0, 26.8, 38.6, 23.0)
    card(ax, *LA, fc=LANE["A"][1], ec=LANE["A"][0], lw=1.5, r=1.1)
    card(ax, *LB, fc=LANE["B"][1], ec=LANE["B"][0], lw=1.5, r=1.1, ls=(0, (4, 2.5)), shadow=False)
    pill(ax, LA[0] + 1.0, LA[1] + LA[3] - 3.3, "U-Net 1 · six fields to four maps", LANE["A"][0], "A")
    pill(ax, LB[0] + 1.0, LB[1] + LB[3] - 3.3, "U-Net 2 · four maps to six fields", LANE["B"][0], "B")
    ua = draw_unet(ax, *LA, "A", "1", 6, 4, None,
                   "Block = 2 × (Conv 3×3 + BN + ReLU) + spatial attention 7×7\n"
                   "dropout 0.2 after each pool · head = 1×1 conv to 4 logits")
    ub = draw_unet(ax, *LB, "B", "2", 4, 6, None,
                #    "same topology, no skip from U-Net 1 · head = 1×1 conv to 6 fields\n"
                   "Rebuilds the six standardized fields")
    # U-Net 1 head to the centre column
    y_head = ua["ys"][0]
    arrow(ax, (ua["dec_x"][0] + ua["bw"], y_head), (43.6, y_head), color=LANE["A"][0], lw=1.8)
    ax.text(42.0, y_head + 0.95, "4 logits", fontsize=F_SMALL - 2.4, color=LANE["A"][0],
            ha="center", va="bottom", zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    # centre column
    CX, CW = 43.6, 10.4
    card(ax, CX, 40.6, CW, 6.6, ec=acc3, lw=1.2)
    ax.text(CX + CW / 2, 46.1, "4 sigmoids", fontsize=F_SUB, fontweight="bold", color=INK,
            ha="center", va="center", zorder=6)
    ax.text(CX + CW / 2, 43.15, "One per class \nIndependent: a cell\nmay hold several\nstorms at once",
            fontsize=F_SMALL - 1.4, color=BODY, ha="center", va="center", zorder=6, linespacing=1.15)
    # arrow(ax, (CX + CW / 2, 40.6), (CX + CW / 2, 38.9), color=acc3, lw=1.8)
    # card(ax, CX, 32.2, CW, 6.6, ec=acc3, lw=1.2)
    # ax.text(CX + CW / 2, 37.7, "Round at 0.5", fontsize=F_SUB, fontweight="bold", color=INK,
    #         ha="center", va="center", zorder=6)
    # ax.text(CX + CW / 2, 34.75, "straight-through:\nexactly 0 or 1,\n4 bits per cell,\ngradient passes",
    #         fontsize=F_SMALL - 1.4, color=BODY, ha="center", va="center", zorder=6, linespacing=1.15)
    # ax.text(CX + CW / 2, 29.9, "at inference only\nU-Net 1 runs; the\nprobabilities go out",
    #         fontsize=F_SMALL - 1.6, color=MUTED, ha="center", va="center", zorder=6,
    #         style="italic", linespacing=1.15)
    # the four probability maps go straight into U-Net 2 (the rounding card was dropped)
    y_in = ub["ys"][0]
    arrow(ax, (CX + CW, y_in), (ub["enc_x"][0], y_in), color=LANE["B"][0], lw=1.8)
    # U-Net 2 head: a real rebuilt field
    rx, ry, rw, rh = LB[0] + LB[2] - 7.4, LB[1] + LB[3] - 3.5, 6.2, 3.1
    a = framed_inset(cv, ax, rx, ry, rw, rh)
    if rec is not None:
        a.imshow(rec[0], cmap="Blues", vmin=-1.0, vmax=3.8, aspect="auto", interpolation="bilinear",
                 rasterized=True)
    else:
        a.imshow(fields[0], cmap="Blues", vmin=0, vmax=700, aspect="auto", interpolation="bilinear",
                 rasterized=True)
    cxd = ub["dec_x"][0] + ub["bw"] / 2
    arrow(ax, (cxd, y_in + ub["bh"] / 2), (cxd, ry - 0.3), color=LANE["B"][0], lw=1.5, ms=10)
    ax.text(rx - 0.6, ry + rh / 2, "6 fields\nrebuilt", fontsize=F_SMALL - 2.4,
            color=LANE["B"][0], ha="right", va="center", zorder=7, linespacing=1.1)
    # training-only strip
    ax.text(3.0, 23.6, "TRAINING\nONLY", fontsize=F_SMALL - 1.0, color=MUTED, ha="left",
            va="center", zorder=6, linespacing=1.2)
    formula(ax, 7.1, 20.6, r"$Loss = w_{seg} L_{seg} + w_{rec} L_{rec}$", BODY, fs=F_SMALL - 1.8)
    sx = [12.8, 40.4, 68.0]; sw, sy, sh = 26.6, 19.4, 6.6
    for x in sx:
        card(ax, x, sy, sw, sh, fc="#fafbfc", ec=acc3, lw=1.0, ls=(0, (3, 2)), shadow=False)
    # 1 training
    glyph_gpus(ax, sx[0] + 1.0, sy + 1.4, 6.4, 4.0, acc3)
    ax.text(sx[0] + 8.4, sy + sh - 1.4, "Training", fontsize=F_SUB + 0.4, fontweight="bold",
            color=INK, ha="left", va="center", zorder=6)
    for k, ln in enumerate(["64 GPUs on Frontier, DDP",
                            "AdamW with cosine decay"]):
        ax.text(sx[0] + 8.4, sy + sh - 2.9 - 1.2 * k, ln, fontsize=F_SMALL - 1.0, color=BODY,
                ha="left", va="center", zorder=6)
    # 2 reconstruction loss
    formula(ax, sx[1] + 4.0, sy + 3.2, r"$\|\hat{x} - x\|^2$", acc3, fs=12.0)
    ax.text(sx[1] + 8.4, sy + sh - 1.4, "Reconstruction loss", fontsize=F_SUB + 0.4,
            fontweight="bold", color=INK, ha="left", va="center", zorder=6)
    for k, ln in enumerate(["MSE of the six rebuilt fields vs the input"]):
        ax.text(sx[1] + 8.4, sy + sh - 2.9 - 1.2 * k, ln, fontsize=F_SMALL - 1.0, color=BODY,
                ha="left", va="center", zorder=6)
    # 3 segmentation loss with the per-class table
    ax.text(sx[2] + 1.0, sy + sh - 1.4, "Segmentation loss", fontsize=F_SUB + 0.4,
            fontweight="bold", color=INK, ha="left", va="center", zorder=6)
    for k, ln in enumerate(["Per class:", "BCE with an imbalance cap",
                            "+ Tversky(α, β)", ""]):
        ax.text(sx[2] + 1.0, sy + sh - 2.75 - 1.1 * k, ln, fontsize=F_SMALL - 1.2, color=BODY,
                ha="left", va="center", zorder=6)
    tcols = [sx[2] + 19.5, sx[2] + 22.4, sx[2] + 25.3]
    for cx_, hd in zip(tcols, ["α", "β", "cap"]):
        ax.text(cx_, sy + sh - 1.3, hd, fontsize=F_SMALL - 1.0, fontweight="bold", color=INK,
                ha="center", va="center", zorder=6)
    table = {"AR": (0.5, 0.5, 8), "TC": (0.3, 0.7, 50), "ETC": (0.4, 0.6, 25), "MCS": (0.45, 0.55, 15)}
    for r_, (key, col, _) in enumerate(CLASSES):
        yy = sy + sh - 2.55 - 1.1 * r_
        dot(ax, sx[2] + 14.9, yy, col, r=0.42)
        ax.text(sx[2] + 15.7, yy, key, fontsize=F_SMALL - 1.6, fontweight="bold", color=col,
                ha="left", va="center", zorder=6)
        for cx_, v in zip(tcols, table[key]):
            ax.text(cx_, yy, f"{v:g}", fontsize=F_SMALL - 1.4, color=BODY, ha="center",
                    va="center", zorder=6)

    # ══ band 4 · output ═══════════════════════════════════════════════════════════════════
    B4 = (1.5, 1.5, 97.0, 14.5)
    band(ax, *B4, "4", "Output", "outputs")
    acc4 = BAND["outputs"][0]
    # two step cards sharing the span from the band edge to the map (x 3 to 59.5)
    ow, oy, oh = 26.7, 4.4, 7.6
    ox = [3.0, 3.0 + ow + 3.1]
    outs = [("Threshold", "A cell is marked where p exceeds one half"),
            ("Monthly NetCDF files", "")]
    for i, (t, s) in enumerate(outs):
        card(ax, ox[i], oy, ow, oh, ec=acc4, lw=1.1)
        if s:
            ax.text(ox[i] + 6.8, oy + oh - 1.5, t, fontsize=F_TITLE - 1.2, fontweight="bold",
                    color=INK, ha="left", va="center", zorder=6)
            ax.text(ox[i] + 6.8, oy + oh - 4.4, wrap(s, 32), fontsize=F_SMALL, color=BODY,
                    ha="left", va="center", zorder=6, linespacing=1.25)
        else:   # no subtitle: centre the title in the card
            ax.text(ox[i] + 6.8, oy + oh / 2, wrap(t, 24), fontsize=F_TITLE - 1.2,
                    fontweight="bold", color=INK, ha="left", va="center", zorder=6,
                    linespacing=1.15)
        if i < len(outs) - 1:
            arrow(ax, (ox[i] + ow, oy + oh / 2), (ox[i + 1], oy + oh / 2), color=acc4, lw=2.0)
    glyph_step(ax, ox[0] + 1.3, oy + 1.5, 4.4, acc4)
    glyph_doc(ax, ox[1] + 1.5, oy + 1.2, 4.2, 5.0, acc4)
    # the real detection map for the same time step
    MX, MY, MW, MH = 62.6, 3.0, 24.0, 12.0
    if preds is not None:
        a = framed_inset(cv, ax, MX, MY, MW, MH)
        bg_field = fields[1]
        bg = (bg_field - np.nanmin(bg_field)) / (np.nanmax(bg_field) - np.nanmin(bg_field) + 1e-9)
        if bg.shape != preds.shape[1:]:
            bg = np.kron(bg, np.ones((1, 1)))[:preds.shape[1], :preds.shape[2]] if bg.shape[0] >= preds.shape[1] else bg
        base = np.repeat((0.86 + 0.11 * bg)[..., None], 3, axis=-1) if bg.shape == preds.shape[1:] \
            else np.ones(preds.shape[1:] + (3,)) * 0.95
        order = ["ETC", "AR", "MCS", "TC"]
        idx = {k: i for i, (k, _, _) in enumerate(CLASSES)}
        img = base.copy()
        # same display thickening as the label strip: at 0.25 deg the TC discs are sub-pixel here
        tk = 4 if preds.shape[1] > 400 else 1
        for key in order:
            m = _thicken(preds[idx[key]], tk)
            img[m] = 0.12 * img[m] + 0.88 * np.array(to_rgb(CCOL[key]))
        a.imshow(np.clip(img, 0, 1), aspect="auto", interpolation="nearest", rasterized=True)
    else:
        glyph_stack(ax, MX + 6.0, MY + 2.0, 12.0, 8.0, [CCOL[k] for k, _, _ in CLASSES])
    ax.text(MX + MW / 2, 2.15, "Four classes results", fontsize=F_SMALL, fontweight="bold",
            color=INK, ha="center", va="center", zorder=6)
    arrow(ax, (ox[-1] + ow, oy + oh / 2), (MX - 0.3, oy + oh / 2), color=acc4, lw=2.0)
    for k, (key, col, name) in enumerate(CLASSES):
        yy = 13.4 - 2.75 * k
        dot(ax, 89.0, yy, col, r=0.75)
        ax.text(90.4, yy, wrap(name, 18), fontsize=F_SMALL - 0.8, color=INK, ha="left",
                va="center", zorder=6, linespacing=1.1)

    # ══ block arrows between bands ════════════════════════════════════════════════════════
    chevron(ax, 50.0, 75.0, 73.0, BAND["inputs"][0], "Six raw fields, one stack every six hours")
    chevron(ax, 50.0, 54.0, 52.0, BAND["harness"][0], "6 × 720 × 1440 standardized predictors")
    chevron(ax, 50.0, 18.0, 16.0, BAND["analysis"][0], "Four probability maps per time step")

    # legend: the actual glyphs, not their names
    ly, fs = 0.72, F_SMALL - 0.6
    ax.add_patch(FancyBboxPatch((1.6, ly - 0.6), 3.2, 1.2, boxstyle="round,pad=0,rounding_size=0.35",
                                fc="#fafbfc", ec=BAND["analysis"][0], lw=0.9, ls=(0, (3, 2))))
    ax.text(5.5, ly, "training only", fontsize=fs, color=MUTED, ha="left", va="center")
    for k, (key, col, _) in enumerate(CLASSES):
        dot(ax, 16.0 + 1.7 * k, ly, col, r=0.6)
    ax.text(22.6, ly, "the four storm classes", fontsize=fs, color=MUTED, ha="left", va="center")
    arrow(ax, (36.2, ly), (39.8, ly), color=SKIP, lw=1.1, ms=9, ls=(0, (3, 2)))
    ax.text(40.6, ly, "skip connection", fontsize=fs, color=MUTED, ha="left", va="center")
    arrow(ax, (51.6, ly), (55.2, ly), color=BAND["harness"][0], lw=1.3, ms=11, ls=(0, (4, 2.5)))
    ax.text(56.0, ly, "labels, loss only", fontsize=fs, color=MUTED, ha="left", va="center")
    arrow(ax, (68.2, ly), (71.8, ly), color=LANE["A"][0], lw=1.6, ms=12)
    ax.text(72.6, ly, "data flow", fontsize=fs, color=MUTED, ha="left", va="center")
    if not real:
        ax.text(98.4, ly, "field tiles are synthetic (cache not found)", fontsize=fs, color=MUTED,
                ha="right", va="center")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    apr = argparse.ArgumentParser()
    apr.add_argument("--no-title", action="store_true",
                     help="omit the in-figure title block (for the paper; caption carries it)")
    apr.add_argument("--cache-dir", default=CACHE_DIR)
    apr.add_argument("--out-dir", default=OUT_DIR)
    args = apr.parse_args()
    if args.no_title:
        main(with_title=False, stem="system_model_wnet_notitle", cache_dir=args.cache_dir,
             out_dir=args.out_dir)
    else:
        main(cache_dir=args.cache_dir, out_dir=args.out_dir)
