"""
Illustrated system-model diagram for the TinyOVD paper.

    cd src && python -m figures.system_model              # with in-figure title
    cd src && python -m figures.system_model --no-title   # paper version (caption carries it)

Writes runs/figures/modern/system_model[_notitle].{png,pdf}. Four layered bands read top to
bottom (inputs, shared harness, the three analyses, what is released), the idiom of a system
architecture figure: real image thumbnails with their tiny-object boxes, pictograms for the
detector families and harness stages, and mini-plots drawn from the actual result files so the
glyphs are data, not decoration. Every data read has a constant fallback, so the script always
renders. No em-dashes in rendered text (FIGURE_FIXES.md, fix 1).
"""
import os
import sys
import json
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402

import numpy as np                                                                   # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                      # noqa: E402
from matplotlib import patheffects as pe                                             # noqa: E402
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, FancyArrow,         # noqa: E402
                                Rectangle, Circle, Ellipse, Polygon)

OUT_DIR = os.path.join(cfg.RUNS_DIR, "figures", "modern")

# ── palette ─────────────────────────────────────────────────────────────────────────────
INK, BODY, MUTED, RULE = "#1c2733", "#3b4855", "#6f7e8c", "#cfd7de"
BAND = {                       # accent, tint
    "inputs":   ("#3d6d9e", "#e8eff7"),
    "harness":  ("#4a5a6a", "#eef1f4"),
    "analysis": ("#0072b2", "#e3eef8"),
    "outputs":  ("#c58a1f", "#fbf3e3"),
}
LANE = {"A": ("#0072b2", "#e9f2fa"), "B": ("#d55e00", "#fdeee4"), "C": ("#009e73", "#e6f5ee")}
RAMP = ["#0b3d66", "#1a5a8c", "#3f80b3", "#8db8da", "#cfe0ef"]       # VT (dark) to L (light)
FIX = "#2563eb"                                                        # "our fix" blue
SHADOW = [pe.withSimplePatchShadow(offset=(0.6, -0.8), shadow_rgbFace="#1c2733", alpha=0.10)]
W = 100.0

# type scale (pt) at a 13in canvas
F_BAND, F_TITLE, F_SUB, F_LEAD, F_BODY, F_SMALL = 14.0, 12.8, 10.6, 11.6, 10.4, 9.2


# ── data (all optional; constants stand in when a file is absent) ───────────────────────
def _metrics(run):
    p = os.path.join(cfg.RUNS_DIR, run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _coco_ap_by_bin():
    m = _metrics("owlv2_coco_val")
    if m:
        return [m[f"AP_{b}"] for b in ("VT", "T", "S", "M", "L")]
    return [0.140, 0.264, 0.363, 0.475, 0.600]


def _waterfall_vt():
    vals = []
    for run, fb in (("owlv2_aitodv2", 0.0083), ("owlv2_aitodv2_sahi_t384_o0.2", 0.0275),
                    ("owlv2_aitodv2_sahi_calib", 0.0398)):
        m = _metrics(run)
        vals.append(m["AP_VT"] if m else fb)
    return vals


def _predictor_points():
    p = os.path.join(cfg.RUNS_DIR, "tile_predictor.json")
    if os.path.exists(p):
        d = json.load(open(p))
        rows = (d.get("pooled") or {}).get("loo") or d.get("loo") or []
        return [(r["measured"], r["pred"], r.get("dataset", "aitodv2")) for r in rows]
    return []


def _controlled():
    p = os.path.join(cfg.RUNS_DIR, "collapse_threshold.json")
    if os.path.exists(p):
        c = json.load(open(p)).get("controlled") or {}
        out = []
        for fam in c.values():
            pts = [(q["eff_stride_px"], q["threshold_px"]) for q in fam["points"] if q["in_dist"]]
            out.append((fam.get("anchor", fam["family"]), pts))
        if out:
            return out
    return [("owlv2", [(41.67, 38.49), (31.25, 28.83), (20.83, 24.45), (16.67, 22.12)]),
            ("owlvit_l14", [(33.33, 32.65), (25.0, 27.46), (20.0, 24.06), (16.67, 21.97)])]


def _showcase(ds, iid, frac):
    """(PIL crop, [boxes in crop coords]) for a dense tiny-object window of one image."""
    try:
        from PIL import Image
    except ImportError:
        return None, []
    cache = os.path.join(cfg.RUNS_DIR, "figures", "qual_cache.json")
    rec = None
    if os.path.exists(cache):
        rec = json.load(open(cache)).get(f"{ds}:{iid}")
    if rec is None:
        annp = os.path.join(cfg.DATASETS_DIR, ds, "annotations", "test.json")
        if not os.path.exists(annp):
            return None, []
        d = json.load(open(annp))
        im = next((i for i in d["images"] if i["id"] == iid), None)
        if im is None:
            return None, []
        rec = {"file_name": im["file_name"],
               "gt": [a["bbox"] for a in d["annotations"] if a["image_id"] == iid]}
    path = os.path.join(cfg.DATASETS_DIR, ds, "images", rec["file_name"])
    if not os.path.exists(path):
        return None, []
    img = Image.open(path).convert("RGB")
    Wi, Hi = img.size
    win = int(min(Wi, Hi) * frac)
    boxes = rec["gt"]
    # densest 8x8 cell of box centres -> crop window (same rule as the qualitative figure)
    if boxes:
        cx = np.array([b[0] + b[2] / 2 for b in boxes]); cy = np.array([b[1] + b[3] / 2 for b in boxes])
        gx = np.clip((cx / Wi * 8).astype(int), 0, 7); gy = np.clip((cy / Hi * 8).astype(int), 0, 7)
        hist = np.zeros((8, 8))
        for a, b in zip(gy, gx):
            hist[a, b] += 1
        iy, ix = np.unravel_index(hist.argmax(), hist.shape)
        x0 = int((ix + 0.5) / 8 * Wi - win / 2); y0 = int((iy + 0.5) / 8 * Hi - win / 2)
    else:
        x0, y0 = Wi // 2 - win // 2, Hi // 2 - win // 2
    x0 = max(0, min(x0, Wi - win)); y0 = max(0, min(y0, Hi - win))
    crop = img.crop((x0, y0, x0 + win, y0 + win))
    if win > 600:
        crop = crop.resize((600, 600))
    s = crop.size[0] / win
    inside = [((b[0] - x0) * s, (b[1] - y0) * s, b[2] * s, b[3] * s) for b in boxes
              if x0 <= b[0] + b[2] / 2 <= x0 + win and y0 <= b[1] + b[3] / 2 <= y0 + win]
    return crop, inside


# ── drawing helpers ─────────────────────────────────────────────────────────────────────
def wrap(s, n):
    if "$" in s:
        return s
    return "\n".join(textwrap.wrap(s, n, break_on_hyphens=False))


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
    # header tab straddling the top border, with a circled number
    pw = 0.66 * len(title) + 6.4
    ax.add_patch(FancyBboxPatch((x + 2.0, y + h - 1.45), pw, 2.9,
                                boxstyle="round,pad=0,rounding_size=1.45", fc=acc, ec="none",
                                zorder=4))
    ax.add_patch(Circle((x + 3.75, y + h), 1.05, fc="white", ec="none", zorder=5))
    ax.text(x + 3.75, y + h, num, color=acc, fontsize=F_BAND - 1, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(x + 5.6, y + h, title, color="white", fontsize=F_BAND, fontweight="bold",
            ha="left", va="center", zorder=6)


def card(ax, x, y, w, h, fc="white", ec=RULE, lw=1.1, ls="-", shadow=True, r=0.9):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, ls=ls, zorder=3)
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


def lane_pill(ax, x, y, letter, title):
    col = LANE[letter][0]
    pw = 0.62 * len(title) + 5.6
    ax.add_patch(FancyBboxPatch((x, y), pw, 2.5, boxstyle="round,pad=0,rounding_size=1.25",
                                fc=col, ec="none", zorder=5))
    ax.add_patch(Circle((x + 1.5, y + 1.25), 0.95, fc="white", ec="none", zorder=6))
    ax.text(x + 1.5, y + 1.25, letter, color=col, fontsize=F_SUB, fontweight="bold",
            ha="center", va="center", zorder=7)
    ax.text(x + 3.1, y + 1.25, title, color="white", fontsize=F_SUB + 0.6, fontweight="bold",
            ha="left", va="center", zorder=7)


def dot(ax, x, y, letter, r=0.85):
    col = LANE[letter][0]
    ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=6))
    ax.text(x, y, letter, color="white", fontsize=F_SMALL, fontweight="bold", ha="center",
            va="center", zorder=7)


# ── pictograms ──────────────────────────────────────────────────────────────────────────
def glyph_grid(ax, x, y, s, n, col, gap=0.12):
    c = (s - gap * (n - 1)) / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * (c + gap), y + i * (c + gap)), c, c, fc=col,
                                   ec="none", alpha=0.35 + 0.65 * ((i + j) % 2), zorder=5))


def glyph_swin(ax, x, y, s, col):
    half = (s - 0.35) / 2
    for i in range(2):
        for j in range(2):
            gx, gy = x + j * (half + 0.35), y + i * (half + 0.35)
            ax.add_patch(Rectangle((gx, gy), half, half, fc="none", ec=col, lw=1.1, zorder=5))
            glyph_grid(ax, gx + 0.25, gy + 0.25, half - 0.5, 2, col, gap=0.1)


def glyph_cnn(ax, x, y, s, col):
    for k, (dx, a) in enumerate([(0.0, 0.35), (0.7, 0.6), (1.4, 0.95)]):
        ax.add_patch(Rectangle((x + dx, y + dx * 0.55), s - 1.6, s - 1.6, fc=col, ec="white",
                               lw=0.8, alpha=a, zorder=5 + k))


def glyph_gpus(ax, x, y, w, h, col):
    cw, ch, g = (w - 3 * 0.3) / 4, (h - 0.3) / 2, 0.3
    for i in range(2):
        for j in range(4):
            ax.add_patch(FancyBboxPatch((x + j * (cw + g), y + i * (ch + g)), cw, ch,
                                        boxstyle="round,pad=0,rounding_size=0.25", fc=col,
                                        ec="none", alpha=0.85, zorder=5))
            ax.plot([x + j * (cw + g) + 0.25, x + j * (cw + g) + cw - 0.25],
                    [y + i * (ch + g) + ch * 0.5] * 2, color="white", lw=0.9, zorder=6)


def glyph_doc(ax, x, y, w, h, col):
    f = 0.9
    ax.add_patch(Polygon([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h), (x, y + h)],
                         closed=True, fc="white", ec=col, lw=1.2, zorder=5))
    ax.add_patch(Polygon([(x + w - f, y), (x + w - f, y + f), (x + w, y + f)], closed=True,
                         fc=col, ec="none", alpha=0.5, zorder=6))
    for k in range(4):
        ax.plot([x + 0.7, x + w - 0.7 - (0.9 if k == 3 else 0)], [y + h - 1.0 - k * 0.85] * 2,
                color=col, lw=1.0, alpha=0.6, zorder=6)


def glyph_bubble(ax, x, y, w, h, col, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                fc="white", ec=col, lw=1.2, zorder=5))
    ax.add_patch(Polygon([(x + 1.2, y + 0.05), (x + 2.6, y + 0.05), (x + 1.0, y - 1.0)],
                         closed=True, fc="white", ec=col, lw=1.2, zorder=5))
    ax.add_patch(Rectangle((x + 1.25, y - 0.05), 1.3, 0.35, fc="white", ec="none", zorder=6))
    for k, ln in enumerate(lines):
        ax.text(x + 0.7, y + h - 0.75 - k * 1.15, ln, fontsize=F_SMALL - 1.6, color=col,
                ha="left", va="center", zorder=6, family="Liberation Mono")


def glyph_tiles(ax, x, y, s, col):
    ax.add_patch(Rectangle((x, y), s, s, fc=col, ec="none", alpha=0.18, zorder=5))
    for k in range(1, 3):
        ax.plot([x + k * s / 3] * 2, [y, y + s], color=col, lw=1.0, zorder=6)
        ax.plot([x, x + s], [y + k * s / 3] * 2, color=col, lw=1.0, zorder=6)
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=col, lw=1.2, zorder=6))


def glyph_resize(ax, x, y, s, col):
    ax.add_patch(Rectangle((x, y), s * 0.55, s * 0.55, fc=col, ec="none", alpha=0.25, zorder=5))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=col, lw=1.2, ls=(0, (2, 1.5)), zorder=6))
    arrow(ax, (x + s * 0.5, y + s * 0.5), (x + s * 0.95, y + s * 0.95), color=col, lw=1.3, ms=10)


def glyph_cylinder(ax, x, y, w, h, col):
    ry = w * 0.18
    ax.add_patch(Rectangle((x, y + ry), w, h - 2 * ry, fc=col, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + ry), w, 2 * ry, fc=col, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + h - ry), w, 2 * ry, fc="white", ec=col, lw=1.2, zorder=6))
    ax.plot([x, x], [y + ry, y + h - ry], color=col, lw=1.2, zorder=6)
    ax.plot([x + w, x + w], [y + ry, y + h - ry], color=col, lw=1.2, zorder=6)


def glyph_person(ax, x, y, s, col):
    ax.add_patch(Circle((x + s / 2, y + s * 0.74), s * 0.2, fc=col, ec="none", zorder=5))
    ax.add_patch(FancyBboxPatch((x + s * 0.15, y), s * 0.7, s * 0.46,
                                boxstyle="round,pad=0,rounding_size=0.9", fc=col, ec="none",
                                zorder=5))


def glyph_code(ax, x, y, s, col):
    ax.text(x + s / 2, y + s / 2, "</>", fontsize=F_TITLE + 6, fontweight="bold", color=col,
            ha="center", va="center", zorder=6, family="Liberation Mono")


# ── figure ──────────────────────────────────────────────────────────────────────────────
def main(with_title=True, stem="system_model"):
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],
                         "text.color": INK, "axes.edgecolor": "none",
                         "mathtext.fontset": "dejavusans"})
    H = 69.0 if with_title else 65.6     # no-title: just clears the band-1 tab (top 64.95)
    fig = plt.figure(figsize=(13.0, 13.0 * H / W), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")
    cv = Canvas(fig, ax, H)

    if with_title:
        ax.text(1.5, 67.4, "TinyOVD system model", fontsize=19, fontweight="bold", color=INK,
                va="center", ha="left")
        ax.text(1.5, 65.5, "Six datasets and eleven zero-shot detectors through one harness; "
                "three analyses; one released artifact.", fontsize=F_SUB + 0.4, color=MUTED,
                va="center", ha="left")

    # ══ band 1 · inputs ═══════════════════════════════════════════════════════════════
    B1 = (1.5, 51.5, 97.0, 12.0)
    band(ax, *B1, "1", "Inputs", "inputs")
    acc1 = BAND["inputs"][0]
    # two real crops, side by side, with their tiny-object boxes
    shots = [("aitodv2", 13417, 0.22, 3.0, "AI-TOD (aerial)"),
             ("sodad", 23181, 0.16, 11.2, "SODA-D (driving)")]
    S = 7.2
    for ds, iid, frac, px, lab in shots:
        py = 53.4
        crop, boxes = _showcase(ds, iid, frac)
        ax.add_patch(Rectangle((px + 0.3, py - 0.4), S + 0.5, S + 0.5, fc="#1c2733", ec="none",
                               alpha=0.10, zorder=3))
        ax.add_patch(Rectangle((px - 0.25, py - 0.25), S + 0.5, S + 0.5, fc="white", ec=RULE,
                               lw=0.9, zorder=4))
        a = cv.inset(px, py, S, S)
        a.set_zorder(5)
        if crop is not None:
            a.imshow(crop, rasterized=True)
            n = crop.size[0]
            a.set_xlim(0, n); a.set_ylim(n, 0)
            for (bx, by, bw, bh) in boxes:
                a.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=FIX, lw=0.75))
        else:
            a.text(0.5, 0.5, ds, ha="center", va="center", transform=a.transAxes)
        for s in a.spines.values():
            s.set_color("white"); s.set_linewidth(1.0)
        ax.text(px + S / 2, 52.5, lab, fontsize=F_SMALL - 0.8, color=MUTED, ha="center",
                va="center", zorder=6)
    # dataset label
    ax.text(20.0, 60.2, "6 datasets", fontsize=F_TITLE, fontweight="bold", color=INK, ha="left",
            va="center", zorder=6)
    ax.text(20.0, 58.1, "web · driving · aerial", fontsize=F_SUB, color=BODY, ha="left",
            va="center", zorder=6)
    ax.text(20.0, 55.7, "COCO, LVIS, SODA-D,\nAI-TOD, DIOR, DOTA", fontsize=F_SMALL, color=MUTED,
            ha="left", va="center", zorder=6, linespacing=1.25)
    # size taxonomy: five shrinking squares on the ramp
    ax.text(31.8, 60.2, "√area size bins (px)", fontsize=F_TITLE - 1, fontweight="bold",
            color=INK, ha="left", va="center", zorder=6)
    sizes = [(4.0, "L", ">96"), (2.6, "M", "32–96"), (1.6, "S", "16–32"), (0.95, "T", "8–16"),
             (0.5, "VT", "<8")]
    sx0, base = 31.8, 55.2
    xc = sx0
    for (s, lab, edge), col in zip(sizes, RAMP[::-1]):
        ax.add_patch(Rectangle((xc, base), s, s, fc=col, ec="none", zorder=5))
        ax.text(xc + s / 2, base - 1.0, lab, fontsize=F_SMALL, fontweight="bold", color=INK,
                ha="center", va="center", zorder=6)
        ax.text(xc + s / 2, base - 2.15, edge, fontsize=F_SMALL - 1.6, color=MUTED, ha="center",
                va="center", zorder=6)
        xc += s + 1.5
    # divider sits in the gap between the VT label and the first detector card
    ax.plot([49.0, 49.0], [52.6, 61.4], color=RULE, lw=1.0, zorder=3)
    # detector roster
    ax.text(50.0, 60.4, "11 zero-shot detectors", fontsize=F_TITLE, fontweight="bold", color=INK,
            ha="left", va="center", zorder=6)
    ax.text(97.3, 60.4, "+ 2 closed-set references (RT-DETR)", fontsize=F_SMALL, color=MUTED,
            ha="right", va="center", zorder=6)
    fams = [("OWL patch-ViT", "5 models", glyph_grid), ("G-DINO Swin", "2 models", glyph_swin),
            ("YOLO-World CNN", "4 models", glyph_cnn)]
    for i, (name, cnt, g) in enumerate(fams):
        x = 50.0 + i * 16.1
        card(ax, x, 52.8, 15.3, 6.3, ec=acc1, lw=1.2)
        if g is glyph_grid:
            g(ax, x + 1.0, 53.8, 4.0, 4, acc1)
        else:
            g(ax, x + 1.0, 53.8, 4.0, acc1)
        ax.text(x + 5.7, 57.8, name, fontsize=F_SUB - 0.4, fontweight="bold", color=INK,
                ha="left", va="center", zorder=6)
        ax.text(x + 5.7, 55.5, cnt, fontsize=F_SMALL + 0.2, color=BODY, ha="left", va="center",
                zorder=6)

    # ══ band 2 · shared harness ═══════════════════════════════════════════════════════
    B2 = (1.5, 33.5, 97.0, 16.0)
    band(ax, *B2, "2", "Shared harness", "harness")
    acc2 = BAND["harness"][0]
    bx = [3.0, 27.5, 52.0, 76.5]; bw, by, bh = 21.5, 40.2, 7.6
    steps = [("Zero-shot prompt", "class names as text"),
             ("Sharded inference", "images split over GPUs"),
             ("COCO predictions", "one file per run"),
             ("Score once", "AP per size bin")]
    for i, (t, s) in enumerate(steps):
        card(ax, bx[i], by, bw, bh, ec=acc2, lw=1.1)
        ax.text(bx[i] + 8.4, by + bh - 1.5, t, fontsize=F_TITLE, fontweight="bold", color=INK,
                ha="left", va="center", zorder=6)
        ax.text(bx[i] + 8.4, by + bh - 4.1, wrap(s, 22), fontsize=F_SUB, color=BODY, ha="left",
                va="center", zorder=6, linespacing=1.25)
        if i < 3:
            arrow(ax, (bx[i] + bw, by + bh / 2), (bx[i + 1], by + bh / 2), color=acc2, lw=2.0)
    glyph_bubble(ax, bx[0] + 1.0, by + 1.9, 6.6, 4.9, acc2, ["airplane", "ship", "vehicle"])
    glyph_gpus(ax, bx[1] + 1.0, by + 1.4, 6.6, 4.8, acc2)
    glyph_doc(ax, bx[2] + 1.6, by + 1.0, 5.2, 5.6, acc2)
    ap = _coco_ap_by_bin()
    a = cv.inset(bx[3] + 1.0, by + 1.9, 6.6, 4.9)
    a.bar(range(5), ap, color=RAMP, width=0.8)
    a.set_ylim(0, max(ap) * 1.15); a.set_xlim(-0.6, 4.6)
    a.set_xticks(range(5)); a.set_xticklabels(["VT", "T", "S", "M", "L"], fontsize=5.2, color=MUTED)
    a.tick_params(bottom=False, labelbottom=True, pad=1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    # run modes reused by the analysis
    ax.text(3.0, 36.9, "RUN MODES\nFOR THE ANALYSIS", fontsize=F_SMALL - 1.0, color=MUTED,
            ha="left", va="center", zorder=6, linespacing=1.2)
    # the tile sweep feeds both the fixes (B) and the predictor (C); the resolution sweep feeds A
    for x, title, sub, g, letters in [(27.5, "Sliced inference", "tile × overlap", glyph_tiles, "BC"),
                                      (76.5, "Resolution override", "weights frozen", glyph_resize, "A")]:
        card(ax, x, 34.6, bw, 4.4, fc="#fafbfc", ec=acc2, lw=1.0, ls=(0, (3, 2)), shadow=False)
        g(ax, x + 1.2, 35.2, 3.2, acc2)
        ax.text(x + 5.6, 37.7, title, fontsize=F_SUB + 0.6, fontweight="bold", color=INK,
                ha="left", va="center", zorder=6)
        ax.text(x + 5.6, 35.7, sub, fontsize=F_SMALL, color=BODY, ha="left", va="center", zorder=6)
        for k, letter in enumerate(reversed(letters)):
            dot(ax, x + bw - 1.7 - 2.2 * k, 36.8, letter)

    # ══ band 3 · analysis ═════════════════════════════════════════════════════════════
    B3 = (1.5, 10.5, 97.0, 21.0)
    band(ax, *B3, "3", "Analysis", "analysis")
    CX, CY, CW, CH = [3.0, 35.0, 67.0], 13.4, 30.0, 15.8
    lanes = {
        "A": ("Collapse and its threshold",
              [("4.3× drop, L to VT", "every detector and dataset"),
               ("Collapse threshold θ", "size at 50% of peak AP"),
               ("One law, R² = 0.97", "θ vs stride, 2 families")]),
        "B": ("Training-free fixes",
              [("Sliced inference", "tile × overlap sweep"),
               ("Per-size calibration", "one temperature per bin"),
               ("+41% AP, +376% VT-AP", "helps only in-distribution")]),
        "C": ("Tile predictor",
              [(r"$\theta_{\mathrm{sahi}} = \eta(t)\,\theta_{\mathrm{base}}\,t/S$",
                "η is detector-independent"),
               ("Leave-one-detector-out", "AI-TOD and DIOR, 10 runs"),
               ("1.2 px error", "61% better than naive")]),
    }
    for (letter, (title, items)), x in zip(lanes.items(), CX):
        col, tint = LANE[letter]
        card(ax, x, CY, CW, CH, fc=tint, ec=col, lw=1.5, r=1.1)
        lane_pill(ax, x + 1.0, CY + CH - 3.3, letter, title)
        ty = CY + CH - 5.4
        for lead, body in items:
            ax.text(x + 14.6, ty, lead, fontsize=F_LEAD, fontweight="bold", color=INK,
                    ha="left", va="center", zorder=6)
            ax.text(x + 14.6, ty - 1.75, body, fontsize=F_BODY, color=BODY, ha="left",
                    va="center", zorder=6)
            ty -= 3.75
        # mini-plot from real results
        a = cv.inset(x + 1.2, CY + 1.2, 12.4, 9.4)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if letter == "A":
            fams_pts = _controlled()
            allx, ally = [], []
            for (fam, pts), c in zip(fams_pts, [LANE["A"][0], LANE["B"][0]]):
                xs, ys = zip(*pts); allx += xs; ally += ys
                a.scatter(xs, ys, s=22, color=c, edgecolor="white", linewidth=0.6, zorder=4)
            k = np.polyfit(allx, ally, 1)
            gx = np.linspace(min(allx) - 2, max(allx) + 2, 20)
            a.plot(gx, np.polyval(k, gx), color=INK, lw=1.0, ls=(0, (3, 2)), zorder=3)
            a.text(0.05, 0.92, "R² = 0.97", transform=a.transAxes, fontsize=6.4, color=INK,
                   va="top", fontweight="bold")
            a.set_xlabel("effective stride (px)", fontsize=5.4, color=MUTED, labelpad=1)
            a.set_ylabel("θ (px)", fontsize=5.4, color=MUTED, labelpad=1)
        elif letter == "B":
            vals = _waterfall_vt()
            a.bar(range(3), vals, color=[RULE, LANE["A"][0], FIX], width=0.66)
            a.set_xticks(range(3)); a.set_xticklabels(["base", "+tile", "+calib"], fontsize=5.4,
                                                       color=MUTED)
            a.tick_params(bottom=False, labelbottom=True, pad=1)
            a.set_ylim(0, max(vals) * 1.3)
            a.text(2, vals[2] * 1.06, f"+{(vals[2] / vals[0] - 1) * 100:.0f}%", ha="center",
                   va="bottom", fontsize=6.2, fontweight="bold", color=FIX)
            a.set_ylabel("very-tiny AP", fontsize=5.4, color=MUTED, labelpad=1)
        else:
            pts = _predictor_points()
            if pts:
                m = [p[0] for p in pts]; q = [p[1] for p in pts]
                lo, hi = min(m + q) * 0.85, max(m + q) * 1.08
                a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
                for mk, dsn in (("o", "aitodv2"), ("^", "dior")):
                    sel = [(u, v) for u, v, d in pts if d == dsn]
                    if sel:
                        a.scatter(*zip(*sel), s=18, marker=mk, color=LANE["C"][0],
                                  edgecolor="white", linewidth=0.5, zorder=4)
                a.set_xlim(lo, hi); a.set_ylim(lo, hi)
                a.text(0.05, 0.92, "y = x", transform=a.transAxes, fontsize=6.2, color=MUTED,
                       va="top")
            a.set_xlabel("measured (px)", fontsize=5.4, color=MUTED, labelpad=1)
            a.set_ylabel("predicted (px)", fontsize=5.4, color=MUTED, labelpad=1)
    # threshold predicts when tiling helps: dashed feedback A -> B along the bottom
    ax.plot([CX[0] + CW / 2, CX[0] + CW / 2, CX[1] + CW / 2], [CY, 11.7, 11.7],
            color=LANE["A"][0], lw=1.3, ls=(0, (4, 2.5)), zorder=6)
    arrow(ax, (CX[1] + CW / 2, 11.7), (CX[1] + CW / 2, CY), color=LANE["A"][0], lw=1.3, ms=12,
          ls=(0, (4, 2.5)))
    ax.text((CX[0] + CX[1] + CW) / 2, 12.15, "θ predicts when tiling will help", fontsize=F_SMALL,
            color=LANE["A"][0], ha="center", va="bottom", zorder=7, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc=BAND["analysis"][1], ec="none"))

    # ══ band 4 · released ═════════════════════════════════════════════════════════════
    B4 = (1.5, 1.5, 97.0, 7.5)      # tall enough that the tab clears the first item
    band(ax, *B4, "4", "Released", "outputs")
    acc4 = BAND["outputs"][0]
    outs = [("Artifact", "annotations · predictions · metrics · harness", "cyl"),
            ("Pre-flight check", "deploy, tile, or skip, before any inference", "person"),
            ("Extensible", "a new detector or dataset in about 10 lines", "code")]
    for (t, s, g), x in zip(outs, CX):
        if g == "cyl":
            glyph_cylinder(ax, x + 1.4, 2.5, 3.4, 3.8, acc4)
        elif g == "person":
            glyph_person(ax, x + 1.0, 2.4, 4.2, acc4)
        else:
            glyph_code(ax, x + 0.8, 2.3, 4.6, acc4)
        ax.text(x + 6.6, 5.9, t, fontsize=F_TITLE, fontweight="bold", color=INK, ha="left",
                va="center", zorder=6)
        ax.text(x + 6.6, 3.7, s, fontsize=F_SUB, color=BODY, ha="left", va="center", zorder=6)

    # ══ block arrows between bands ════════════════════════════════════════════════════
    chevron(ax, 50.0, 51.5, 49.5, BAND["inputs"][0], "size-tagged data and wrappers")
    chevron(ax, 50.0, 33.5, 31.5, BAND["harness"][0], "size-stratified AP for every run")
    chevron(ax, 50.0, 10.5, 9.0, BAND["analysis"][0], "findings and tools")

    # legend: the actual glyphs, not their names
    ly, fs = 0.75, F_SMALL - 0.6
    ax.add_patch(FancyBboxPatch((1.6, ly - 0.6), 3.2, 1.2, boxstyle="round,pad=0,rounding_size=0.35",
                                fc="#fafbfc", ec=BAND["harness"][0], lw=0.9, ls=(0, (3, 2))))
    ax.text(5.5, ly, "optional run mode", fontsize=fs, color=MUTED, ha="left", va="center")
    for k, letter in enumerate("ABC"):
        dot(ax, 17.6 + 1.7 * k, ly, letter, r=0.6)
    ax.text(22.3, ly, "analysis lanes A, B, C: on a run mode, the lanes that use it",
            fontsize=fs, color=MUTED, ha="left", va="center")
    arrow(ax, (50.2, ly), (53.8, ly), color=LANE["A"][0], lw=1.3, ms=11, ls=(0, (4, 2.5)))
    ax.text(54.6, ly, "prediction", fontsize=fs, color=MUTED, ha="left", va="center")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, facecolor="white")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    apr = argparse.ArgumentParser()
    apr.add_argument("--no-title", action="store_true",
                     help="omit the in-figure title block (for the paper; caption carries it)")
    args = apr.parse_args()
    if args.no_title:
        main(with_title=False, stem="system_model_notitle")
    else:
        main()
