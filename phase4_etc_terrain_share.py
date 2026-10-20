#!/usr/bin/env python3
"""
phase4_etc_terrain_share.py — do the cyclone tracks of each catalogue avoid high terrain?

The user's question about naec_kde_4methods.png: NAECC, TRACK and TempestExtremes show a
track-density minimum over the western cordillera, WNet does not. This measures it directly:
the share of a catalogue's QC'd track points inside the CONUS domain of the paper that fall in
the Rockies / Sierra boxes of the tracker's MOUNTAIN_MASK_BOXES, and the share of NHW-domain
points over Greenland. The references and the shipped WNet tracks are read with the paper's
own loaders (etc_naec_common/etc_paper_analysis.py: same QC, TE boundary points dropped); the
tracker-sweep variants are read from their track files and given the same QC.

Runs as a batch job (Run_etc_terrain_share.sh): the NAECC csv is 1.9 GB.

  python3 phase4_etc_terrain_share.py --years 2000 2010 [--variants S0_base S11_... ]
"""
import os
import sys
import glob
import argparse

import numpy as np

NB = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks"
sys.path.insert(0, os.path.join(NB, "etc_naec_common"))
SWEEP = os.path.join(NB, "WNet_parallel_global_code", "ARs_Hurricanes_ETCs_MCS", "Phase2",
                     "F_ibtracs_pwH50_e200", "tracker_sweep")
CONUS = (-132.5, -62.5, 12.5, 57.5)                          # the paper's CONUS domain (lon0, lon1, lat0, lat1)
NHW = (-180.0, 0.0, 20.0, 80.0)
BOXES = {"Rockies 115-100W 35-48N": (-115, -100, 35, 48), "Sierra/Basin 122-112W 36-42N": (-122, -112, 36, 42),
         "Plains lee 100-90W 35-50N": (-100, -90, 35, 50), "Greenland 55-15W 60-85N": (-55, -15, 60, 85)}
EPOCH = np.datetime64("1970-01-01T00", "h")


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def read_variant(d, years):
    """QC'd tracks (dict t, lat, lon) from ETC_tracks_YYYY_meteorological_v2.txt, the paper's QC."""
    out = []
    for y in years:
        p = os.path.join(d, "etc", f"ETC_tracks_{y}_meteorological_v2.txt")
        if not os.path.exists(p):
            continue
        cur, last = None, None
        for line in open(p):
            if line.startswith("#") or not line.strip():
                continue
            a = line.split()
            if a[0] != last:
                if cur:
                    out.append(cur)
                cur, last = dict(t=[], lat=[], lon=[]), a[0]
            cur["t"].append((np.datetime64(a[1].replace("_", "T"), "h") - EPOCH).astype(np.int64))
            cur["lon"].append(float(a[2])); cur["lat"].append(float(a[3]))
        if cur:
            out.append(cur)
    keep = []
    for tr in out:
        t, la, lo = np.array(tr["t"]), np.array(tr["lat"]), np.array(tr["lon"])
        if t.size >= 8 and t[-1] - t[0] >= 48 and hav(la[0], lo[0], la[-1], lo[-1]) >= 1000.0:
            keep.append(dict(t=t, lat=la, lon=lo))
    return keep


def inside(la, lo, box):
    return (lo >= box[0]) & (lo <= box[1]) & (la >= box[2]) & (la <= box[3])


def shares(tracks):
    la = np.concatenate([np.asarray(x["lat"]) for x in tracks])
    lo = np.concatenate([np.asarray(x["lon"]) for x in tracks])
    t = np.concatenate([np.asarray(x["t"], dtype=np.int64) for x in tracks])
    month = (EPOCH + t.astype("timedelta64[h]")).astype("datetime64[M]").astype(int) % 12 + 1
    cold = np.isin(month, (10, 11, 12, 1, 2, 3))
    conus, nhw = inside(la, lo, CONUS), inside(la, lo, NHW)
    row = {}
    for name, box in BOXES.items():
        ref = nhw if name.startswith("Greenland") else conus
        row[name] = 100.0 * (inside(la, lo, box) & ref).sum() / max(ref.sum(), 1)
    rock = inside(la, lo, BOXES["Rockies 115-100W 35-48N"])
    # summer heat lows over the plateau close a 2 hPa contour too: a warm-season excess points at
    # them, a cold-season excess at reduction artefacts or genuine lee cyclones
    row["Rockies cold Oct-Mar"] = 100.0 * (rock & conus & cold).sum() / max((conus & cold).sum(), 1)
    row["Rockies warm Apr-Sep"] = 100.0 * (rock & conus & ~cold).sum() / max((conus & ~cold).sum(), 1)
    row["CONUS points"] = int(conus.sum())
    row["tracks"] = len(tracks)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=SWEEP)
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2010])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(NB, "WNet_parallel_global_code", "phase4_results",
                                                  "etc_terrain_share.txt"))
    args = ap.parse_args()
    y0, y1 = args.years
    import etc_paper_analysis as P
    P.Y0, P.Y1 = y0, y1                                       # the loaders read the module constants
    g = P.g
    rows = [("NAECC (reference)", shares(P.load_naecc())),
            ("TRACK (reference)", shares(P.from_dicts(g.load_track_tracks_from_dir(g.TRACK_DIR, y0, y1)))),
            ("TempestExtremes (reference)", shares(P.load_te()[0])),
            ("WNet shipped tracker (paper)", shares(P.from_dicts(g.ev.load_etc_tracks_from_txt(
                g.ETC_DIR, y0, y1, pattern=g.ETC_PATTERN))))]
    dirs = sorted(glob.glob(os.path.join(args.sweep_dir, "S*")),
                  key=lambda p: int(os.path.basename(p)[1:].split("_")[0]))
    for v in dirs:
        tag = os.path.basename(v)
        if args.variants and tag not in args.variants:
            continue
        trs = read_variant(v, range(y0, y1 + 1))
        if trs:
            rows.append((tag, shares(trs)))
    cols = list(BOXES) + ["Rockies cold Oct-Mar", "Rockies warm Apr-Sep", "CONUS points", "tracks"]
    lines = [f"Share (%) of QC'd track points {y0}-{y1} inside terrain boxes: Rockies, Sierra and Plains-lee relative to "
             f"all points in the paper's CONUS domain, Greenland relative to all points in the NHW domain.",
             f"{'catalogue':<32}" + "".join(f"{c:>30}" for c in cols)]
    for name, r in rows:
        lines.append(f"{name:<32}" + "".join(f"{r[c]:>30.2f}" if isinstance(r[c], float) else f"{r[c]:>30d}" for c in cols))
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
