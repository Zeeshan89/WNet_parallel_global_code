#!/usr/bin/env python3
"""
phase1_verify_split.py — verify the Phase 1 split contract before training.

Three checks, all CPU, no model needed:

  A STRUCTURE   The predictor block and the label block must be disjoint. No
                predictor may be an (even approximate) copy of a label, and the
                labels must still match the legacy masks bit-for-bit so the
                science criteria are provably unchanged.

  B SCALING     Push the cache through the real dataset normalisation and report
                the standardised range per channel. The legacy contract fed the
                model an IVT channel spanning -14..+107 because `global_stats`
                paired a correct mean with the std of the global-mean time series
                rather than the pixel std; this is the check that catches that.

  C TRIVIAL BASELINE
                For every (predictor, label) pair, the best IoU obtainable by
                thresholding that ONE predictor at a single global value. This is
                the bar the network has to clear to be worth anything. It matters
                most for AR, whose label is literally `IVT > percentile` — a high
                number there is expected and is a statement about the label, not
                about the model.

Usage
  python3 phase1_verify_split.py --months 201401 201407
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ARDatasetImproved as DS

LEGACY_CACHE = DS.TENSOR_CACHE_DIR
PRED = DS.PREDICTOR_NAMES
LAB = DS.LABEL_NAMES
NP_, NL = DS.N_PREDICTORS, DS.N_LABELS
THR = 0.5


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def best_threshold_iou(field, target, n_steps=60):
    """Best IoU from a single global threshold on `field` (both directions)."""
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return float("nan"), float("nan"), ""
    qs = np.percentile(finite, np.linspace(0.5, 99.5, n_steps))
    best, best_t, best_dir = -1.0, float("nan"), ""
    for t in np.unique(qs):
        for direction in (">", "<"):
            pred = field > t if direction == ">" else field < t
            v = iou(pred, target)
            if np.isfinite(v) and v > best:
                best, best_t, best_dir = v, float(t), direction
    return best, best_t, best_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=DS.SPLIT_CACHE_DIR)
    ap.add_argument("--stats", default="predictor_stats.json")
    ap.add_argument("--months", nargs="+", default=["201401", "201407"])
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="phase1_verify.json",
                    help="where to write the report; the default overwrites the file that "
                         "records the shipped label's floors, so give a new name for a new label")
    ap.add_argument("--baseline-json-for", nargs="*", metavar="CLASS",
                    help="also write a main_phase2.py --baseline-json file (<out> with suffix "
                         "_baseline.json) in which ONLY these classes take the floor measured here "
                         "and the others keep the shipped TRIVIAL_BASELINE values, e.g. --baseline-json-for ar")
    args = ap.parse_args()

    files = [(m, os.path.join(args.cache_dir, f"split_{m}.npy")) for m in args.months]
    files = [(m, f) for m, f in files if os.path.exists(f)]
    if not files:
        print(f"No split months found in {args.cache_dir}")
        return 1

    blocks, legacy_blocks = [], []
    for m, f in files:
        a = np.load(f, mmap_mode="r")
        sel = list(range(0, a.shape[0], args.stride))
        blocks.append(np.asarray(a[sel], dtype=np.float32))
        lf = os.path.join(LEGACY_CACHE, f"tensors_{m}.npy")
        if os.path.exists(lf):
            la = np.load(lf, mmap_mode="r")
            legacy_blocks.append(np.asarray(la[sel], dtype=np.float32))
    data = np.concatenate(blocks, axis=0)
    preds, labels = data[:, :NP_], data[:, NP_:]
    n = data.shape[0]

    print("=" * 78)
    print(f" Phase 1 split-contract verification  ({n} timesteps, "
          f"{', '.join(m for m, _ in files)})")
    print("=" * 78)

    # ── A structure ──────────────────────────────────────────────────────────
    print()
    print("A  STRUCTURE")
    print(f"   cache shape        : {data.shape}  "
          f"({NP_} predictors + {NL} labels)")
    ok = data.shape[1] == NP_ + NL
    print(f"   channel count      : {'OK' if ok else 'WRONG'}")

    if legacy_blocks:
        legacy = np.concatenate(legacy_blocks, axis=0)
        d = np.abs(labels - legacy[:, DS.LEGACY_LABEL_CHANNELS]).max()
        print(f"   labels vs legacy   : max|diff| = {d:.3g} "
              f"{'(identical — science criteria unchanged)' if d == 0 else '(DIFFERS!)'}")

    print()
    print("   worst IoU between any predictor and any label")
    print("   (a near-1.0 value would mean a label is being fed to the model)")
    print()
    print(f"     {'predictor':<12}" + "".join(f"{l:>12}" for l in LAB))
    print("     " + "-" * (12 + 12 * len(LAB)))
    worst = 0.0
    for c, pname in enumerate(PRED):
        row = f"     {pname:<12}"
        for i, lname in enumerate(LAB):
            # compare each predictor's "active" support against the label
            f = preds[:, c]
            act = f > np.percentile(f[np.isfinite(f)], 90)
            v = iou(act, labels[:, i] > THR)
            worst = max(worst, v)
            row += f"{v:>12.4f}"
        print(row)
    print()
    print(f"   max predictor/label IoU (top-decile support): {worst:.4f}")
    print(f"   {'OK — no predictor is a label copy' if worst < 0.75 else 'WARNING — a predictor closely tracks a label'}")

    # ── B scaling ────────────────────────────────────────────────────────────
    print()
    print("B  SCALING (through the dataset normalisation)")
    try:
        with open(args.stats) as fh:
            stats = json.load(fh)["stats"]
    except Exception as e:
        print(f"   ! cannot read {args.stats} ({e}) — run compute_predictor_stats.py")
        return 1
    print()
    print(f"     {'predictor':<12} {'method':<12} {'min':>9} {'p1':>9} {'p99':>9} {'max':>9}")
    print("     " + "-" * 62)
    for c, pname in enumerate(PRED):
        st = stats[pname]
        z = (preds[:, c] - st["mean"]) / st["std"]
        p1, p99 = np.percentile(z, [1, 99])
        print(f"     {pname:<12} {st.get('method','?'):<12} {z.min():>9.2f} "
              f"{p1:>9.2f} {p99:>9.2f} {z.max():>9.2f}")

    # ── C trivial baseline ───────────────────────────────────────────────────
    print()
    print("C  TRIVIAL BASELINE — best IoU from thresholding ONE predictor")
    print("   The network must beat this to have learned anything non-obvious.")
    print()
    print(f"     {'predictor':<12}" + "".join(f"{l:>12}" for l in LAB))
    print("     " + "-" * (12 + 12 * len(LAB)))
    baseline = {}
    for c, pname in enumerate(PRED):
        row = f"     {pname:<12}"
        baseline[pname] = {}
        for i, lname in enumerate(LAB):
            v, t, d = best_threshold_iou(preds[:, c], labels[:, i] > THR)
            baseline[pname][lname] = {"iou": v, "threshold": t, "direction": d}
            row += f"{v:>12.4f}"
        print(row)

    print()
    print("   best single-predictor threshold per class:")
    for i, lname in enumerate(LAB):
        best_p = max(PRED, key=lambda p: baseline[p][lname]["iou"])
        b = baseline[best_p][lname]
        print(f"     {lname:<10} IoU {b['iou']:.4f}  via  {best_p} "
              f"{b['direction']} {b['threshold']:.4g}")

    floors = {lname.lower(): max(baseline[p][lname]["iou"] for p in PRED) for lname in LAB}
    with open(args.out, "w") as fh:
        json.dump({"months": [m for m, _ in files], "n_timesteps": int(n),
                   "cache_dir": args.cache_dir,
                   "max_predictor_label_iou": worst,
                   "trivial_baseline": baseline,
                   "baseline": floors}, fh, indent=2, default=float)
    print()
    print(f"   wrote {args.out}")
    if args.baseline_json_for is not None:
        # main_phase2.py reads {"baseline": {ar, hurricane, etc, mcs}} and selects best.pth on
        # skill relative to these floors, so a relabelled class needs its new floor while the
        # other three keep the values every earlier run was judged against.
        from main_phase2 import TRIVIAL_BASELINE
        mixed = dict(TRIVIAL_BASELINE)
        for c in args.baseline_json_for:
            mixed[c.lower()] = floors[c.lower()]
        bp = os.path.splitext(args.out)[0] + "_baseline.json"
        with open(bp, "w") as fh:
            json.dump({"baseline": mixed, "measured_here": args.baseline_json_for,
                       "cache_dir": args.cache_dir, "months": [m for m, _ in files]}, fh, indent=2)
        print(f"   wrote {bp}: {mixed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
