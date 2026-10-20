#!/usr/bin/env python3
"""
phase4_tc_screen_report.py — merge the Tier B label-screening partials and rank
the candidate CyTRACK criteria.

Reads screen_YYYYMM.npz written by phase4_tc_label_screen.py and pools the
contingency counts across months before recomputing the rates, so a quiet month
does not carry the same weight as an active one.

Interpretation
--------------
These are MASK scores, not model scores. They matter because the model converges
onto its teacher: the probability sweep moved model CSI 0.097 -> 0.110 and stopped
at the mask's own 0.1103. So a candidate's mask CSI is the level a retrained model
should be expected to approach, and the ranking here is the ranking of retraining
outcomes — without paying for the retraining.

The baseline row is the production call site (dataset.py:1313-1324). Every other
row changes one thing, or a stated few, so an effect is attributable to a cause.
The IBTrACS injection is absent throughout, so POD is what the physics finds.

Usage
  phase4_tc_screen_report.py --indir tc_label_screen
"""
import os
import sys
import json
import glob
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="tc_label_screen")
    ap.add_argument("--json", default="phase4_results/tc_label_screen.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "screen_*.npz")))
    if not files:
        raise SystemExit(f"no partials in {args.indir}")

    names = None
    tot = None
    steps = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        n = list(z["names"])
        if names is None:
            names = n
            tot = np.zeros((len(n), 5), dtype=np.int64)
        elif n != names:
            raise SystemExit(f"candidate list differs in {f}")
        tot += z["stats"].astype(np.int64)
        steps += int(z["n_steps"])

    months = [os.path.basename(f)[7:13] for f in files]
    print(f"\n  {len(files)} months ({months[0]}..{months[-1]}), {steps} time steps")
    print("  MASK scores vs IBTrACS, physics only (no IBTrACS injection)")
    print(f"\n  {'candidate':<26}{'det':>8}{'hits':>7}{'miss':>7}{'FA':>8}"
          f"{'POD':>7}{'FAR':>7}{'CSI':>7}{'det/step':>10}")
    print("  " + "-" * 89)

    rows = []
    for i, nm in enumerate(names):
        hits, misses, fa, ndet, ntruth = tot[i]
        pod = hits / max(hits + misses, 1)
        far = fa / max(hits + fa, 1)
        csi = hits / max(hits + misses + fa, 1)
        rows.append((csi, nm, hits, misses, fa, ndet, pod, far))
        print(f"  {nm:<26}{ndet:>8}{hits:>7}{misses:>7}{fa:>8}"
              f"{pod:>7.3f}{far:>7.3f}{csi:>7.3f}{ndet/max(steps,1):>10.2f}")

    base = next(r for r in rows if r[1].startswith("baseline"))
    print("\n  ===== ranked by CSI, with the change from baseline =====")
    print(f"  {'candidate':<26}{'CSI':>7}{'dCSI':>8}{'POD':>7}{'dPOD':>8}"
          f"{'FA':>8}{'dFA':>9}")
    print("  " + "-" * 74)
    for csi, nm, hits, misses, fa, ndet, pod, far in sorted(rows, reverse=True):
        print(f"  {nm:<26}{csi:>7.3f}{csi-base[0]:>+8.3f}{pod:>7.3f}"
              f"{pod-base[6]:>+8.3f}{fa:>8}{fa-base[4]:>+9}")

    # The decision rule: a candidate is only worth a retrain if it raises CSI
    # AND does not gut POD. Cutting false alarms by discarding real storms is the
    # trap the 96 h duration filter fell into earlier in this project.
    print("\n  ===== worth a retrain? (CSI up AND POD loss < 0.05) =====")
    good = [r for r in rows if r[0] > base[0] and (base[6] - r[6]) < 0.05
            and not r[1].startswith("baseline")]
    if good:
        for csi, nm, hits, misses, fa, ndet, pod, far in sorted(good, reverse=True):
            print(f"  {nm:<26} CSI {csi:.3f} (+{csi-base[0]:.3f})  "
                  f"POD {pod:.3f} ({pod-base[6]:+.3f})  FA {fa} ({fa-base[4]:+d})")
    else:
        print("  NONE — no candidate raises CSI without costing >0.05 POD.")
        print("  That would mean the CyTRACK formulation, not its thresholds, is")
        print("  the limit on this data, and TC needs a different label construction")
        print("  (warm-core test, SST gating) rather than retuned numbers.")

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"months": months, "n_steps": steps,
                       "candidates": {nm: dict(
                           hits=int(tot[i][0]), misses=int(tot[i][1]),
                           false_alarms=int(tot[i][2]), n_det=int(tot[i][3]),
                           n_truth=int(tot[i][4]))
                           for i, nm in enumerate(names)}}, f, indent=1)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
