"""
Comprehensive Loss Visualization Script
Updated for new loss_history.pkl format (single dict with all files concatenated)
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  11,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "figure.titlesize": 14,
    "pdf.fonttype":     42,   # editable text in vector PDF
    "savefig.dpi":      300,
})

# Distinct markers per storm class, so lines are differentiable in print/grayscale
CLASS_MARKERS = {"ar": "o", "hurricane": "s", "etc": "^", "mcs": "D",
                 "total": "*", "overlap": "v"}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = ("/lustre/orion/proj-shared/cli138/z7a/Storms_data/"
               "Storms_precipitation_notebooks/WNet_parallel_global_code/"
               "ARs_Hurricanes_ETCs_MCS/Trail11")
HISTORY_PKL = os.path.join(BASE_DIR, "loss_history.pkl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "loss_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(name):
    """Save the current figure as both high-res PNG and vector PDF, then close."""
    for ext in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_loss_history(path=HISTORY_PKL):
    """
    Load the loss_history dict saved by main.py.

    Expected keys:
        total, ar, hurricane, etc, mcs, overlap   — lists, one value per epoch
        file_boundaries   — list of indices where each new file starts
        file_names        — list of nc filenames in order
        lr                — list of LR values (may be empty)

    Also supports the old per-month pkl files as a fallback.
    """
    # ── New format: single loss_history.pkl ──────────────────────────────────
    if os.path.exists(path):
        with open(path, "rb") as f:
            history = pickle.load(f)

        # Validate keys
        required = {"total", "ar", "hurricane", "etc", "mcs", "overlap"}
        if required.issubset(history.keys()):
            print(f"Loaded loss_history.pkl — "
                  f"{len(history['total'])} epochs across "
                  f"{len(history.get('file_names', []))} files")
            return history
        else:
            print("loss_history.pkl found but missing keys — trying old format")

    # ── Old format fallback: per-month loss_details_*.pkl ────────────────────
    old_dir  = os.path.join(BASE_DIR, "loss_details")
    old_files = sorted(glob.glob(os.path.join(old_dir, "loss_details_*.pkl")))

    if not old_files:
        print(f"ERROR: No loss data found at {path} or {old_dir}")
        return None

    print(f"Falling back to {len(old_files)} old-format pkl files")

    history = {
        "total": [], "ar": [], "hurricane": [], "etc": [],
        "mcs": [], "overlap": [],
        "file_boundaries": [], "file_names": [], "lr": []
    }

    for fp in old_files:
        try:
            filename = os.path.basename(fp)
            date_str = filename.split("_")[2].split(".")[0]
            with open(fp, "rb") as f:
                d = pickle.load(f)
            history["file_boundaries"].append(len(history["total"]))
            history["file_names"].append(f"era5dara_{date_str}.nc")
            for key in ["total", "ar", "hurricane", "etc", "mcs", "overlap"]:
                history[key].extend(d.get(key, []))
        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Old-format fallback: {len(history['total'])} epochs loaded")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_label(fname):
    """Extract YYYYMM from filename for axis labels."""
    try:
        return fname.split("_")[1].split(".")[0]
    except Exception:
        return fname


def _add_file_shading(ax, boundaries, file_names, values, show_labels=False):
    """
    Shade alternating file blocks. Optionally annotate each block with its
    YYYYMM label at the top (off by default — the labels clutter the figure
    and are not needed for the paper).
    """
    ymin, ymax = ax.get_ylim()
    n = len(values)
    for i, b_start in enumerate(boundaries):
        b_end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        if i % 2 == 0:
            ax.axvspan(b_start, b_end, alpha=0.07, color="gray", zorder=0)
        if show_labels:
            label = _file_label(file_names[i]) if i < len(file_names) else str(i)
            ax.text((b_start + b_end) / 2, ymax,
                    label, fontsize=5, ha="center", va="bottom",
                    rotation=90, color="dimgray")


def _smooth(values, window):
    """Simple moving average."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _epochs_per_file(history):
    """Return number of epochs trained per file (assumes uniform)."""
    boundaries = history.get("file_boundaries", [])
    if len(boundaries) < 2:
        total = len(history["total"])
        return total if total > 0 else 1
    return boundaries[1] - boundaries[0]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Overall training loss
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_training_loss(history):
    values     = history["total"]
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])

    if not values:
        print("No total loss data — skipping overall plot")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(values))

    ax.plot(x, values, linewidth=0.9, color="navy", alpha=0.85, label="Total Loss")

    # Smoothed trend
    window = max(5, len(values) // 40)
    smoothed = _smooth(values, window)
    ax.plot(np.arange(window - 1, len(values)), smoothed,
            linewidth=2.2, color="red", alpha=0.7,
            label=f"Smoothed (w={window})")

    _add_file_shading(ax, boundaries, file_names, values)

    # Mark file boundaries with vertical lines
    for b in boundaries[1:]:
        ax.axvline(b, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_yscale("log")
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Total Loss (log scale)", fontsize=12)
    ax.set_title("Overall Training Loss Across All Monthly Files", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("overall_training_loss")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Individual storm type losses
# ─────────────────────────────────────────────────────────────────────────────

def plot_individual_storm_losses(history):
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])

    storm_cfg = [
        ("ar",        "AR Loss",        "steelblue"),
        ("hurricane", "Hurricane Loss", "firebrick"),
        ("etc",       "ETC Loss",       "darkorange"),
        ("mcs",       "MCS Loss",       "forestgreen"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title, color) in zip(axes, storm_cfg):
        values = history.get(key, [])
        if not values:
            ax.set_visible(False)
            continue

        x = np.arange(len(values))
        marker = CLASS_MARKERS.get(key, "o")
        me = max(1, len(values) // 18)
        ax.plot(x, values, linewidth=0.8, color=color, alpha=0.8,
                marker=marker, markevery=me, markersize=5)

        window = max(5, len(values) // 40)
        smoothed = _smooth(values, window)
        ax.plot(np.arange(window - 1, len(values)), smoothed,
                linewidth=2.0, color="black", alpha=0.55,
                linestyle="--", label=f"Smoothed (w={window})")

        _add_file_shading(ax, boundaries, file_names, values)
        for b in boundaries[1:]:
            ax.axvline(b, color="black", linewidth=0.5, linestyle="--", alpha=0.35)

        ax.set_yscale("log")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Individual Storm Type Losses Over Training",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("individual_storm_losses")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — All loss components on one axes
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_comparison(history):
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])

    cfg = [
        ("total",     "Total",     "black",       2.0),
        ("ar",        "AR",        "steelblue",   1.2),
        ("hurricane", "Hurricane", "firebrick",   1.2),
        ("etc",       "ETC",       "darkorange",  1.2),
        ("mcs",       "MCS",       "forestgreen", 1.2),
        ("overlap",   "Overlap",   "purple",      1.0),
    ]

    fig, ax = plt.subplots(figsize=(18, 7))

    n = len(history["total"])
    me = max(1, n // 20)   # space markers out so lines stay readable
    for key, label, color, lw in cfg:
        values = history.get(key, [])
        if values:
            ax.plot(np.arange(len(values)), values,
                    linewidth=lw, color=color, alpha=0.85, label=label,
                    marker=CLASS_MARKERS.get(key, "o"),
                    markevery=me, markersize=6)

    # alternating file shading only (no file-number annotations)
    _add_file_shading(ax, boundaries, file_names, history["total"])

    ax.set_yscale("log")
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("All Loss Components During Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("loss_components_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Average loss per epoch number (epoch 1–100 averaged across files)
# ─────────────────────────────────────────────────────────────────────────────

def plot_losses_by_epoch(history):
    """
    For each epoch number (1 to num_epochs_per_file), compute the mean
    loss across all files.  This shows whether the model converges well
    within a single file's training block.
    """
    boundaries = history.get("file_boundaries", [])
    total      = history["total"]

    if not total or len(boundaries) < 1:
        print("Not enough data for per-epoch plot")
        return

    epochs_per_file = _epochs_per_file(history)
    n_files         = len(boundaries)

    # Build array: (n_files, epochs_per_file) for each loss key
    keys   = ["total", "ar", "hurricane", "etc", "mcs", "overlap"]
    titles = ["Total", "AR", "Hurricane", "ETC", "MCS", "Overlap"]
    colors = ["black", "steelblue", "firebrick", "darkorange", "forestgreen", "purple"]

    # Collect per-file epoch arrays
    file_arrays = {k: [] for k in keys}
    for i, b_start in enumerate(boundaries):
        b_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(total)
        for k in keys:
            chunk = history.get(k, [])[b_start:b_end]
            if len(chunk) == epochs_per_file:
                file_arrays[k].append(chunk)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, key, title, color in zip(axes, keys, titles, colors):
        arr = file_arrays[key]
        if not arr:
            ax.set_visible(False)
            continue

        mat  = np.array(arr)        # shape: (n_files, epochs_per_file)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        x    = np.arange(1, epochs_per_file + 1)

        marker = CLASS_MARKERS.get(key, "o")
        ax.plot(x, mean, linewidth=2.2, color=color, label="Mean across files",
                marker=marker, markersize=6)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color,
                        label="±1 std")

        # Individual file traces (faint)
        for row in mat:
            ax.plot(x, row, linewidth=0.4, color=color, alpha=0.25)

        ax.set_xlabel("Epoch within file", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{title} — avg over {len(arr)} files", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Average Loss per Epoch (averaged across all monthly files)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("losses_by_epoch")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Per-file final loss (how well each month converged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_file_final_loss(history):
    """
    For each monthly file, plot the final epoch's loss value.
    Shows which months are harder to learn.
    """
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])
    total      = history["total"]

    if not total or len(boundaries) < 2:
        print("Not enough files for per-file final loss plot")
        return

    keys   = ["total", "ar", "hurricane", "etc", "mcs", "overlap"]
    colors = ["black", "steelblue", "firebrick", "darkorange", "forestgreen", "purple"]

    # Get final epoch loss per file per key
    final_losses = {k: [] for k in keys}
    labels       = []

    for i, b_start in enumerate(boundaries):
        b_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(total)
        labels.append(_file_label(file_names[i]) if i < len(file_names) else str(i))
        for k in keys:
            chunk = history.get(k, [])[b_start:b_end]
            final_losses[k].append(chunk[-1] if chunk else np.nan)

    x    = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, key, color in zip(axes, keys, colors):
        vals = final_losses[key]
        ax.bar(x, vals, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Final Epoch Loss", fontsize=12)
        ax.set_title(f"{key.upper()} — final loss per file",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Highlight worst file
        if not all(np.isnan(vals)):
            worst_idx = np.nanargmax(vals)
            ax.bar(worst_idx, vals[worst_idx], color="red", alpha=0.9,
                   edgecolor="black", linewidth=0.6, label=f"Worst: {labels[worst_idx]}")
            ax.legend(fontsize=11)

    fig.suptitle("Final Epoch Loss per Monthly File",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("per_file_final_loss")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Seasonal variation (average loss by calendar month)
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_loss_variation(history):
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])
    total      = history["total"]

    if not total or not file_names:
        print("No file_names in history — skipping seasonal plot")
        return

    # Build per-file average losses keyed by calendar month
    monthly = {m: {k: [] for k in ["ar", "hurricane", "etc", "mcs"]}
               for m in range(1, 13)}

    for i, b_start in enumerate(boundaries):
        b_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(total)
        label = _file_label(file_names[i]) if i < len(file_names) else ""
        try:
            cal_month = int(label[4:6])
        except Exception:
            continue

        for k in ["ar", "hurricane", "etc", "mcs"]:
            chunk = history.get(k, [])[b_start:b_end]
            if chunk:
                monthly[cal_month][k].append(np.mean(chunk))

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cfg = [
        ("ar",        "AR",        "steelblue",   "o"),
        ("hurricane", "Hurricane", "firebrick",   "s"),
        ("etc",       "ETC",       "darkorange",  "^"),
        ("mcs",       "MCS",       "forestgreen", "d"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(1, 13)

    for key, label, color, marker in cfg:
        means = [np.mean(monthly[m][key]) if monthly[m][key] else np.nan
                 for m in range(1, 13)]
        ax.plot(x, means, linewidth=2.2, marker=marker, markersize=8,
                color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Calendar Month", fontsize=12)
    ax.set_ylabel("Average Loss", fontsize=12)
    ax.set_title("Seasonal Variation in Storm Detection Losses",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("monthly_loss_variation")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Loss distributions (histograms + stats table)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_statistics(history):
    keys   = ["total", "ar", "hurricane", "etc", "mcs", "overlap"]
    colors = ["black", "steelblue", "firebrick", "darkorange", "forestgreen", "purple"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    stats_rows = []
    for ax, key, color in zip(axes, keys, colors):
        values = history.get(key, [])
        if not values:
            ax.set_visible(False)
            continue

        arr = np.array(values)
        ax.hist(arr, bins=40, color=color, alpha=0.72, edgecolor="black", linewidth=0.4)
        ax.axvline(arr.mean(), color="red",   linewidth=1.6, linestyle="--", label="Mean")
        ax.axvline(np.median(arr), color="gold", linewidth=1.6, linestyle=":",  label="Median")
        ax.set_xlabel("Loss Value", fontsize=12)
        ax.set_ylabel("Frequency",  fontsize=12)
        ax.set_title(f"{key.upper()} Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax.text(0.97, 0.96,
                f"μ={arr.mean():.4f}\nσ={arr.std():.4f}\n"
                f"min={arr.min():.4f}\nmax={arr.max():.4f}",
                transform=ax.transAxes, fontsize=11,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        stats_rows.append([key.upper(),
                           f"{arr.mean():.4f}", f"{arr.std():.4f}",
                           f"{arr.min():.4f}", f"{arr.max():.4f}",
                           f"{np.median(arr):.4f}"])

    fig.suptitle("Loss Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("loss_statistics")

    # Print stats table to console as well
    header = f"{'Type':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}"
    print("\n" + "=" * 65)
    print("LOSS STATISTICS")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for row in stats_rows:
        print(f"{row[0]:<12} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Held-out validation metrics per class over passes
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_metrics(history):
    """Per-class IoU/Dice/Precision/Recall on held-out months across passes."""
    val = history.get("val_metrics", [])
    if not val:
        print("No val_metrics in history — skipping val plots")
        return

    epochs = [v["global_epoch"] for v in val]

    storm_cfg = [
        ("ar",        "AR",        "steelblue"),
        ("hurricane", "Hurricane", "firebrick"),
        ("etc",       "ETC",       "darkorange"),
        ("mcs",       "MCS",       "forestgreen"),
    ]
    metric_cfg = [
        ("iou",       "IoU"),
        ("dice",      "Dice"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (mkey, mtitle) in zip(axes, metric_cfg):
        for skey, slabel, color in storm_cfg:
            ykey = f"{skey}_{mkey}"
            vals = [v["metrics"].get(ykey, np.nan) for v in val]
            ax.plot(epochs, vals, marker=CLASS_MARKERS.get(skey, "o"),
                    markersize=6, linewidth=1.8, color=color, label=slabel)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(mtitle, fontsize=12)
        ax.set_title(f"Held-out Val {mtitle} per Storm Class",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # Overall figure title removed for the paper figure
    # fig.suptitle("Held-out Validation Metrics Across Passes",
    #              fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("val_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9 — Train loss vs val IoU per class (overfitting diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_loss_vs_val_iou(history):
    """
    Twin-axis: train loss (per class) on left, val IoU on right.
    Overfitting = train loss keeps dropping while val IoU flattens or declines.
    Marks the best-val-IoU epoch for checkpoint selection.
    """
    val = history.get("val_metrics", [])
    if not val:
        print("No val_metrics in history — skipping overfitting plot")
        return

    val_epochs = [v["global_epoch"] for v in val]

    storm_cfg = [
        ("ar",        "AR",        "steelblue"),
        ("hurricane", "Hurricane", "firebrick"),
        ("etc",       "ETC",       "darkorange"),
        ("mcs",       "MCS",       "forestgreen"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (skey, slabel, color) in zip(axes, storm_cfg):
        loss_values = history.get(skey, [])
        val_iou = [v["metrics"].get(f"{skey}_iou", np.nan) for v in val]

        if loss_values:
            ax.plot(np.arange(len(loss_values)), loss_values,
                    color=color, alpha=0.55, linewidth=0.9, label="Train loss")
            ax.set_yscale("log")
            ax.set_ylabel(f"{slabel} train loss (log)",
                          fontsize=12, color=color)
            ax.tick_params(axis="y", labelcolor=color)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(val_epochs, val_iou, marker=CLASS_MARKERS.get(skey, "o"),
                 markersize=6, linewidth=2.0, color="black", label="Val IoU")
        ax2.set_ylabel(f"{slabel} val IoU", fontsize=12)
        ax2.set_ylim(0, 1)

        ax.set_title(f"{slabel} — train loss vs held-out val IoU",
                     fontsize=13, fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper right")

    # Overall figure title removed for the paper figure
    plt.tight_layout()
    _save("train_loss_vs_val_iou")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10 — Data-ordering diagnostic (why the train loss oscillates)
# ─────────────────────────────────────────────────────────────────────────────

# The era blocks are ORDINAL — successive ~6-year slices of the record — so they
# get a single-hue sequential ramp (light→dark), not categorical hues. Keeping
# them blue also keeps them clear of the per-class colours reused in panel (c).
# Lightness is monotone (OKLCH L 0.72 → 0.50 → 0.32); all-pairs CVD ΔE ≥ 16.7.
ERA_RAMP = ["#6baed6", "#2171b5", "#08306b"]

_INK        = "#1a1a19"   # primary text
_INK_MUTED  = "#52514e"   # secondary text
_ASRUN_GREY = "#9a9a95"   # recessive "as run" trace in panel (b)


def _block_spans(history):
    """
    Return [(start_epoch, end_epoch, rank0_label), …] — one entry per file block.
    """
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])
    n = len(history["total"])
    spans = []
    for i, b_start in enumerate(boundaries):
        b_end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        label = _file_label(file_names[i]) if i < len(file_names) else str(i)
        spans.append((b_start, b_end, label))
    return spans


def _era_groups(history):
    """
    Group epochs by which slot of the pass they came from.

    Every rank owns a fixed set of monthly files and all ranks step through
    their slots in lockstep, so the slot index *is* the ~6-year block of the
    record the whole job is training on. Rank 0's file label appears exactly
    once per pass, so grouping by that label is identical to grouping by slot.
    get_rank_files() slices a chronologically sorted list, so sorting the labels
    recovers the chronological slot order.

    Returns (labels, {label: [epoch indices]}).
    """
    spans = _block_spans(history)
    if not spans:
        return [], {}
    labels = sorted({lab for _, _, lab in spans})
    groups = {lab: [] for lab in labels}
    for b_start, b_end, lab in spans:
        groups[lab].extend(range(b_start, b_end))
    return labels, groups


def _era_year_spans(labels, world_size=64, start_year=2006, end_year=2024,
                    data_dir="/lustre/orion/proj-shared/cli138/z7a/Storms_data/data"):
    """
    Human-readable year span of each block, e.g. '2006-2012'.

    A block is shared by *all* ranks at that slot, so its span covers every
    rank's slot-k file — not just rank 0's single month. Reconstructs the
    sharding with the same functions main.py used, and only returns spans if
    rank 0's reconstructed files reproduce the labels actually recorded in the
    history. Any mismatch (different world size, year range, data dir) returns
    {} and the caller falls back to bare YYYYMM labels.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from distributed_dataset import build_train_val_split, get_rank_files

        all_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("era5dara_") and f.endswith(".nc")
            and start_year <= int(f.split("_")[1][:4]) <= end_year
        )
        train_files, _ = build_train_val_split(all_files, val_fraction=0.135, seed=42)

        per_rank = [get_rank_files(train_files, r, world_size, shuffle=False)
                    for r in range(world_size)]
        rank0 = per_rank[0]
        if sorted(_file_label(f) for f in rank0) != sorted(labels):
            print("  [era spans] reconstructed sharding does not match the recorded "
                  "file labels — falling back to YYYYMM labels")
            return {}

        spans = {}
        for slot, f0 in enumerate(rank0):
            years = [int(files[slot].split("_")[1][:4])
                     for files in per_rank if slot < len(files)]
            spans[_file_label(f0)] = f"{min(years)}–{max(years)}"
        return spans
    except Exception as e:
        print(f"  [era spans] could not reconstruct sharding ({type(e).__name__}: {e}) "
              f"— falling back to YYYYMM labels")
        return {}


def _eta_squared(values, groups, burn):
    """
    Fraction of epoch-to-epoch variance in `values` explained by block identity
    (one-way ANOVA eta-squared, SSB / SST), over epochs >= burn.
    """
    per_group = [np.asarray([values[e] for e in eps if e >= burn], dtype=float)
                 for eps in groups.values()]
    per_group = [g for g in per_group if g.size > 0]
    pooled = np.concatenate(per_group) if per_group else np.array([])
    if pooled.size < 2:
        return np.nan
    grand = pooled.mean()
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in per_group)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in per_group)
    return ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan


def plot_data_ordering_diagnostic(history, focus="mcs", focus_label="MCS",
                                  focus_color="forestgreen", world_size=64):
    """
    Why the train loss oscillates: it tracks which ~6-year block of the record
    the job is training on, not optimiser instability.

    (a) as run — the loss steps in lockstep with the block cycle
    (b) the same losses at composition-complete (pass) scale — the cycle is gone
    (c) how much of each class's epoch-to-epoch variance the block identity explains

    Panel (b) is a RE-AGGREGATION of the recorded per-epoch losses at the
    composition cross-month pooling would enforce. It is not a second training
    run: it establishes that the oscillation is composition, not optimisation.
    """
    values = np.asarray(history.get(focus, []), dtype=float)
    labels, groups = _era_groups(history)
    if values.size == 0 or len(labels) < 2:
        print("Not enough file-block structure in history — skipping data-ordering plot")
        return

    epochs_per_file = _epochs_per_file(history)
    epochs_per_pass = len(labels) * epochs_per_file
    burn            = 2 * epochs_per_pass          # drop warm-up passes
    n               = values.size
    if n <= burn + epochs_per_pass:
        print("Too few epochs after warm-up — skipping data-ordering plot")
        return

    year_spans  = _era_year_spans(labels, world_size=world_size)
    era_color   = {lab: ERA_RAMP[i % len(ERA_RAMP)] for i, lab in enumerate(labels)}
    era_text    = {lab: year_spans.get(lab, lab) for lab in labels}
    block_means = {lab: np.mean([values[e] for e in groups[lab] if e >= burn])
                   for lab in labels}

    # Pass-scale re-aggregation: whole passes only, starting on a pass boundary.
    n_pass   = (n - burn) // epochs_per_pass
    tail     = burn + n_pass * epochs_per_pass
    passes   = values[burn:tail].reshape(n_pass, epochs_per_pass).mean(axis=1)
    pass_x   = burn + (np.arange(n_pass) + 0.5) * epochs_per_pass

    post     = values[burn:]
    p2p_run  = post.max() - post.min()
    p2p_pass = passes.max() - passes.min()
    eta2     = _eta_squared(values, groups, burn)
    sd_btw   = np.std(list(block_means.values()))
    sd_wtn   = np.sqrt(np.mean([
        np.var([values[e] for e in groups[lab] if e >= burn]) for lab in labels
    ]))

    # Panel (a) zooms the final few passes, where every block has settled onto a
    # flat plateau, so the step-per-block correspondence is exact rather than
    # tangled up with the early decline.
    zoom_passes = min(3, n_pass)
    zoom_lo     = tail - zoom_passes * epochs_per_pass
    zoom_hi     = tail

    fig = plt.figure(figsize=(18, 5.6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.35, 0.75],
                           left=0.045, right=0.99, bottom=0.30, top=0.90, wspace=0.26)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (ax_a, ax_b, ax_c):            # white ground so the sequential ramp reads
        ax.set_facecolor("white")
        ax.grid(True, color="#e3e3e0", linewidth=0.8)
        ax.set_axisbelow(True)

    # ── (a) as run, zoomed ───────────────────────────────────────────────────
    zoom_v    = values[zoom_lo:zoom_hi]
    zrange    = zoom_v.max() - zoom_v.min()
    ylim_a    = (zoom_v.min() - 0.10 * zrange, zoom_v.max() + 0.26 * zrange)
    zoom_mean = {lab: np.mean([values[e] for e in groups[lab]
                               if zoom_lo <= e < zoom_hi]) for lab in labels}

    for b_start, b_end, lab in _block_spans(history):
        if b_end <= zoom_lo or b_start >= zoom_hi:
            continue
        ax_a.axvspan(max(b_start, zoom_lo), min(b_end, zoom_hi),
                     color=era_color[lab], alpha=0.45, zorder=0, lw=0)

    # Direct labels live in a right-hand gutter aligned with each block's mean,
    # so identity is never colour-alone and nothing collides with the trace.
    gutter = 0.30 * (zoom_hi - zoom_lo)
    for lab in labels:
        ax_a.axhline(zoom_mean[lab], color=era_color[lab], linestyle="--",
                     linewidth=1.3, alpha=0.85, zorder=2)
        ax_a.text(zoom_hi + 0.06 * gutter, zoom_mean[lab], era_text[lab],
                  fontsize=10, ha="left", va="center", color=era_color[lab],
                  fontweight="bold", zorder=4)
    ax_a.plot(np.arange(n), values, color=focus_color, linewidth=2.0, zorder=3)

    ax_a.set_xlim(zoom_lo, zoom_hi + gutter)
    ax_a.set_ylim(*ylim_a)
    ax_a.axvspan(zoom_hi, zoom_hi + gutter, color="white", zorder=1, lw=0)
    ax_a.set_xticks([t for t in ax_a.get_xticks() if zoom_lo <= t <= zoom_hi])
    ax_a.set_xlabel(f"Global epoch (final {zoom_passes} passes)", fontsize=12, color=_INK)
    ax_a.set_ylabel(f"{focus_label} train loss", fontsize=12, color=_INK)
    ax_a.set_title("(a) As run — the loss level is set by the block",
                   fontsize=13, fontweight="bold", color=_INK)
    ax_a.text(0.03, 0.975,
              "block order is re-permuted every pass — "
              "the loss follows the block, not the position",
              transform=ax_a.transAxes, fontsize=9.5, color=_INK_MUTED,
              ha="left", va="top")

    # ── (b) composition-complete ─────────────────────────────────────────────
    brange = post.max() - post.min()
    ylim_b = (post.min() - 0.06 * brange, post.max() + 0.24 * brange)   # headroom for the legend
    ax_b.axvspan(zoom_lo, zoom_hi, color="#d9d9d4", alpha=0.55, zorder=0, lw=0)
    ax_b.text((zoom_lo + zoom_hi) / 2, post.max() + 0.02 * brange, "shown in (a)",
              fontsize=9.5, ha="center", va="bottom", color=_INK_MUTED, zorder=4)
    ax_b.plot(np.arange(n), values, color=_ASRUN_GREY, linewidth=1.0,
              alpha=0.95, zorder=2, label="As run (per epoch)")
    ax_b.plot(pass_x, passes, color=focus_color, linewidth=2.2,
              marker=CLASS_MARKERS.get(focus, "o"), markersize=8, zorder=3,
              label="Pooled (per pass)")
    ax_b.set_xlim(burn, n)
    ax_b.set_ylim(*ylim_b)
    ax_b.set_xlabel(f"Global epoch (warm-up passes 1–2 omitted)", fontsize=12, color=_INK)
    ax_b.set_ylabel(f"{focus_label} train loss", fontsize=12, color=_INK)
    ax_b.set_title("(b) Pooled across blocks — the cycle is gone",
                   fontsize=13, fontweight="bold", color=_INK)
    ax_b.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax_b.text(0.985, 0.975,
              f"peak-to-peak {p2p_run:.3f} → {p2p_pass:.3f} ({p2p_run / p2p_pass:.1f}×)\n"
              f"{passes[0]:.3f} → {passes[-1]:.3f}, declining",
              transform=ax_b.transAxes, fontsize=9.5, color=_INK,
              ha="right", va="top")

    # ── (c) eta-squared per class ────────────────────────────────────────────
    cls_cfg = [("mcs", "MCS", "forestgreen"), ("hurricane", "Hurricane", "firebrick"),
               ("ar", "AR", "steelblue"), ("total", "Total", "black"),
               ("etc", "ETC", "darkorange")]
    rows = [(lbl, _eta_squared(np.asarray(history[k], dtype=float), groups, burn), c)
            for k, lbl, c in cls_cfg if history.get(k)]
    rows.sort(key=lambda r: np.nan_to_num(r[1], nan=-1.0))

    ypos = np.arange(len(rows))
    ax_c.barh(ypos, [r[1] for r in rows], color=[r[2] for r in rows],
              height=0.6, zorder=3)
    for y, (_, val, _) in zip(ypos, rows):
        ax_c.text(val + 0.025, y, f"{val:.3f}", va="center", fontsize=11, color=_INK)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([r[0] for r in rows], fontsize=12, color=_INK)
    ax_c.set_xlim(0, 1.2)
    ax_c.set_xticks([0, 0.5, 1.0])
    ax_c.set_xlabel("Variance explained ($\\eta^2$)", fontsize=12, color=_INK)
    ax_c.set_title("(c) Not just MCS", fontsize=13, fontweight="bold", color=_INK)
    ax_c.text(0.97, 0.06,
              f"{focus_label}: between-block $\\sigma$ {sd_btw:.3f}\n"
              f"vs within-block $\\sigma$ {sd_wtn:.3f} ({sd_btw / sd_wtn:.0f}×)",
              transform=ax_c.transAxes, fontsize=9.5, color=_INK_MUTED,
              ha="right", va="bottom")

    fig.text(0.5, 0.15,
             "Each rank holds one monthly file from each of three ~6-year blocks of the record, and all "
             "ranks step through their blocks in lockstep, so the era composition of\n"
             "the global batch cycles with a period of one pass. Sample-level shuffling is already "
             "enabled within each monthly file, so it cannot mix across the block\n"
             "boundary where the loss steps — what is missing is cross-month pooling. Panel (b) "
             "re-aggregates the same recorded per-epoch losses at the composition\n"
             "that pooling across blocks would enforce; it is not a separate training run.",
             ha="center", va="top", fontsize=10, color=_INK_MUTED, linespacing=1.6)

    _save("data_ordering_not_instability")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def create_summary_report(history):
    lines = [
        "=" * 80,
        "TRAINING LOSS SUMMARY REPORT",
        "=" * 80,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total epochs logged : {len(history['total'])}",
        f"Files processed     : {len(history.get('file_names', []))}",
        "",
    ]

    for key in ["total", "ar", "hurricane", "etc", "mcs", "overlap"]:
        arr = np.array(history.get(key, []))
        if arr.size == 0:
            continue
        lines += [
            f"{key.upper()} Loss:",
            f"  Mean   : {arr.mean():.6f}",
            f"  Std    : {arr.std():.6f}",
            f"  Min    : {arr.min():.6f}",
            f"  Max    : {arr.max():.6f}",
            f"  Median : {np.median(arr):.6f}",
            f"  Final  : {arr[-1]:.6f}",
            "",
        ]

    # Per-file summary
    boundaries = history.get("file_boundaries", [])
    file_names = history.get("file_names", [])
    total      = history["total"]

    if file_names:
        lines += ["PER-FILE FINAL LOSSES", "-" * 80]
        header = f"{'File':<14} {'Total':>10} {'AR':>10} {'Hurr':>10} {'ETC':>10} {'MCS':>10}"
        lines.append(header)
        lines.append("-" * 64)
        for i, b_start in enumerate(boundaries):
            b_end  = boundaries[i + 1] if i + 1 < len(boundaries) else len(total)
            label  = _file_label(file_names[i]) if i < len(file_names) else str(i)
            row    = []
            for k in ["total", "ar", "hurricane", "etc", "mcs"]:
                chunk = history.get(k, [])[b_start:b_end]
                row.append(f"{chunk[-1]:.4f}" if chunk else "  N/A")
            lines.append(f"{label:<14} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")

    # Held-out validation metrics (if present)
    val = history.get("val_metrics", [])
    if val:
        lines += ["", "HELD-OUT VALIDATION METRICS (per pass)",
                  "-" * 80,
                  f"{'epoch':>6} {'AR_IoU':>8} {'Hurr_IoU':>9} {'ETC_IoU':>8} "
                  f"{'MCS_IoU':>8} {'mean_IoU':>9} {'mean_Dice':>10}"]
        for v in val:
            m = v["metrics"]
            lines.append(
                f"{v['global_epoch']:>6d} "
                f"{m['ar_iou']:>8.4f} {m['hurricane_iou']:>9.4f} "
                f"{m['etc_iou']:>8.4f} {m['mcs_iou']:>8.4f} "
                f"{m['mean_iou']:>9.4f} {m['mean_dice']:>10.4f}"
            )
        # Best-val-IoU pass per class (checkpoint-selection hint)
        lines += ["", "BEST VAL IoU PER CLASS:"]
        for name in ["ar", "hurricane", "etc", "mcs"]:
            ious = [v["metrics"].get(f"{name}_iou", -1.0) for v in val]
            best_i = int(np.argmax(ious))
            lines.append(f"  {name.upper():<10}  IoU={ious[best_i]:.4f}  "
                         f"@ epoch {val[best_i]['global_epoch']} "
                         f"(pass {val[best_i]['pass_idx']+1})")

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    report_text = "\n".join(lines)
    out = os.path.join(OUTPUT_DIR, "loss_summary_report.txt")
    with open(out, "w") as f:
        f.write(report_text)
    print(f"\nReport saved → {out}")
    print("\n" + report_text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def set_base_dir(base_dir):
    """
    Re-point the module at a different trial directory (e.g. to regenerate an
    older run's plots without editing the BASE_DIR literal).
    """
    global BASE_DIR, HISTORY_PKL, OUTPUT_DIR
    BASE_DIR    = base_dir
    HISTORY_PKL = os.path.join(BASE_DIR, "loss_history.pkl")
    OUTPUT_DIR  = os.path.join(BASE_DIR, "loss_plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None,
                        help="Trial directory holding loss_history.pkl "
                             f"(default: {BASE_DIR})")
    parser.add_argument("--world-size", type=int, default=64,
                        help="DDP world size of the run, used to label era blocks "
                             "in the data-ordering diagnostic (default: 64)")
    args = parser.parse_args()

    if args.base_dir:
        set_base_dir(args.base_dir)

    print(f"Loading loss history from {BASE_DIR} …")
    history = load_loss_history(HISTORY_PKL)
    if history is None:
        return

    print("\nGenerating plots …")
    plot_overall_training_loss(history)
    plot_individual_storm_losses(history)
    plot_loss_comparison(history)
    plot_losses_by_epoch(history)
    plot_per_file_final_loss(history)
    plot_monthly_loss_variation(history)
    plot_loss_statistics(history)
    plot_val_metrics(history)
    plot_train_loss_vs_val_iou(history)
    plot_data_ordering_diagnostic(history, world_size=args.world_size)

    print("\nGenerating summary report …")
    create_summary_report(history)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()