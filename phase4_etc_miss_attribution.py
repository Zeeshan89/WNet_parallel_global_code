#!/usr/bin/env python3
"""
phase4_etc_miss_attribution.py — why does WNet miss NAECC cyclones? Split every miss into
the stage of the pipeline that lost it, before any tracker or label is changed.

WNet finds more tracks than NAECC, yet matches only about 0.5 to 0.6 of NAECC's cyclones. A
cyclone can be lost at four different stages, and each stage has a different remedy:

  mask       the exported ETC mask never covered the cyclone          -> label / model (retrain)
  centre     the mask covered it, but no tracker centre (the centroid of a connected mask
             blob) was within 500 km: merged blobs put the centroid between two lows -> tracker
  track      centres were there, but no VALID track was: linking or the validation filters
             removed it                                               -> tracker
  fragment   valid tracks covered it, but no SINGLE track covered half of it -> gap bridging

For every NAECC cyclone that passes the shared QC (6-hourly, >= 8 points, >= 48 h, >= 1000 km
net displacement) this script measures, at the cyclone's own 6-hourly positions,

  cov_mask        share of points inside the WNet ETC mask (3 x 3 cells, any)
  cov_c300/c500   share of points with a tracker centre within 300 / 500 km at that time
  any_track       share of points with a valid WNet track point within 500 km and 6 h
  best_track      the same for the best SINGLE track; >= 0.5 is a match (the rule of
                  etc_naec_common/etc_paper_analysis.py)

and, with no detections involved at all, what the tracker's own rules would do to the NAECC
track itself (a ceiling on the detection rate that the rules allow):

  val_*           which validation criteria of ETCTracker.validate_track_meteorology it fails
  n_slow          6-hourly steps shorter than the link floor. is_realistic_movement compares
                  km/h with min_speed_kmh = 12, so a step under 72 km cannot be linked
  ceiling_ok      a linkable segment (no slow step) holding >= half of the points passes
                  validation: the cyclone was reachable under the shipped rules

NAECC centres are hourly MSLP minima on the 0.25 degree grid; WNet centres are blob centroids
and move more smoothly, so the ceiling is an estimate, not an exact replay.

Stages:
  srun -n 11 python3 phase4_etc_miss_attribution.py --years 2000 2010     one rank per year
  python3 phase4_etc_miss_attribution.py --summarize                      tables from the CSVs
"""
import os
import sys
import glob
import json
import argparse

# Production runs the tracker with the mountain mask off (Run_etc_tracks_production.sh:76-77).
os.environ.setdefault("ETC_MOUNTAIN_FRAME", "0")
os.environ.setdefault("ETC_MOUNTAIN_TRACK_FRAC", "1.0")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.dirname(HERE)
sys.path.insert(0, HERE)

NAEC_CSV = ("/lustre/orion/proj-shared/cli138/dr6/eventtype/run-tracking/StormType-Data/"
            "ETCs/NAEC/NAEC_1979_2020.csv")
DET_DIR = os.path.join(NB, "Global_pw50_e200", "ETCs_WNet_final")
TRK_DIR = os.path.join(HERE, "ARs_Hurricanes_ETCs_MCS", "Phase2_E_ibtracs", "tracks_pw50_e200", "etc")
OUT_DIR = os.path.join(HERE, "evaluations", "etc_miss_attribution")
EPOCH = np.datetime64("1970-01-01T00", "h")
TOL_KM, TOL_H = 500.0, 6
# name, lon0, lon1, lat0, lat1; first match wins, by the track mid-point
SECTORS = (("north_of_65N", -180, 0, 65, 90),
           ("gulf_of_alaska", -170, -128, 42, 65),
           ("cordillera", -128, -103, 28, 62),
           ("plains_lee", -103, -85, 28, 50),
           ("hudson_bay", -103, -65, 50, 65),
           ("east_coast_nw_atlantic", -85, -45, 28, 50),
           ("labrador_greenland", -65, -30, 50, 65),
           ("ne_atlantic", -45, 0, 28, 65),
           ("pacific_other", -180, -128, 20, 42))


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_naecc(y0, y1):
    """QC'd NAECC tracks that START in y0..y1, as in etc_paper_analysis.make_track."""
    df = pd.read_csv(NAEC_CSV, usecols=["storm", "datetime", "latitude", "longitude", "MSLPmin"])
    df = df.dropna(subset=["datetime"])
    yr = df["datetime"] // 1000000
    df = df[(yr >= y0) & (yr <= y1 + 1) & (df["datetime"] % 100 % 6 == 0)]
    ts = pd.to_datetime(df["datetime"].astype(np.int64).astype(str), format="%Y%m%d%H", errors="coerce")
    df = df.assign(_t=ts).dropna(subset=["_t"])
    out = []
    for sid, s in df.groupby("storm", sort=False):
        s = s.sort_values("_t")
        t = (s["_t"].values.astype("datetime64[h]") - EPOCH).astype(np.int64)
        la = s["latitude"].values.astype(float)
        lo = ((s["longitude"].values.astype(float) + 180) % 360) - 180
        first_year = int(str(s["_t"].values[0])[:4])
        if not (y0 <= first_year <= y1) or t.size < 8 or (t[-1] - t[0]) < 48:
            continue
        if hav(la[0], lo[0], la[-1], lo[-1]) < 1000.0:
            continue
        out.append(dict(storm=int(sid), t=t, lat=la, lon=lo, year=first_year,
                        month=int(str(s["_t"].values[0])[5:7]), minmslp=float(s["MSLPmin"].min()),
                        mslp=s["MSLPmin"].values.astype(float)))
    return out


def load_wnet_tracks(years):
    """Valid WNet tracks as flat arrays sorted by time: t (h), lat, lon, track id."""
    T, LA, LO, ID = [], [], [], []
    nid = 0
    for y in years:
        p = os.path.join(TRK_DIR, f"ETC_tracks_{y}_meteorological_v2.txt")
        if not os.path.exists(p):
            continue
        last = None
        with open(p) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                a = line.split()
                if a[0] != last:
                    nid, last = nid + 1, a[0]
                T.append(np.datetime64(a[1].replace("_", "T"), "h"))
                LO.append(float(a[2]))
                LA.append(float(a[3]))
                ID.append(nid)
    t = (np.array(T) - EPOCH).astype(np.int64)
    o = np.argsort(t, kind="stable")
    return t[o], np.array(LA)[o], np.array(LO)[o], np.array(ID)[o]


class Detections:
    """Monthly ETC masks and the tracker's centres for every frame, loaded on demand."""

    def __init__(self, tracker):
        self.tracker, self.cache = tracker, {}

    def month(self, ym):
        if ym not in self.cache:
            if len(self.cache) >= 2:                         # a cyclone spans at most two months
                self.cache.pop(next(iter(self.cache)))
            from netCDF4 import Dataset
            p = os.path.join(DET_DIR, f"era5dara_{ym}_ETCs_detection.nc")
            if not os.path.exists(p):
                self.cache[ym] = None
                return None
            with Dataset(p) as ds:
                ds.set_auto_mask(False)
                det = ds.variables["ETCs_detection"][:] > 0.5          # whole month in one read
                lat = ds.variables["latitude"][:].astype(float)
                lon = ds.variables["longitude"][:].astype(float)
                hrs = ds.variables["time"][:].astype(np.int64)
            t0 = (np.datetime64(f"{ym[:4]}-{ym[4:]}-01T00", "h") - EPOCH).astype(np.int64)
            centres = []
            for k in range(det.shape[0]):
                c = self.tracker.find_etc_centers(det[k], lon, lat)
                centres.append(np.array([(x[1], x[0]) for x in c], float).reshape(-1, 2))   # lat, lon
            self.cache[ym] = dict(det=det, lat=lat, lon=lon, index={int(t0 + h): k for k, h in enumerate(hrs)},
                                  centres=centres)
        return self.cache[ym]

    def at(self, t, la, lo):
        """(in mask, distance to nearest centre in km) for one NAECC point, or (nan, nan)."""
        ym = str(EPOCH + np.timedelta64(int(t), "h"))[:7].replace("-", "")
        m = self.month(ym)
        if m is None or int(t) not in m["index"]:
            return np.nan, np.nan
        k = m["index"][int(t)]
        iy = int(round((m["lat"][0] - la) / 0.25))
        ix = int(round((lo - m["lon"][0]) / 0.25))
        ys = np.clip(np.arange(iy - 1, iy + 2), 0, m["lat"].size - 1)
        xs = np.arange(ix - 1, ix + 2) % m["lon"].size
        inmask = bool(m["det"][k][np.ix_(ys, xs)].any())
        c = m["centres"][k]
        dist = float(hav(la, lo, c[:, 0], c[:, 1]).min()) if c.size else np.inf
        return float(inmask), dist


def validation_reasons(trk, track):
    """Every criterion of ETCTracker.validate_track_meteorology the track fails."""
    out = []
    dur = (pd.to_datetime(track[-1][0]) - pd.to_datetime(track[0][0])).total_seconds() / 3600.0
    seg = [trk.haversine_distance(track[i - 1][1], track[i - 1][2], track[i][1], track[i][2])
           for i in range(1, len(track))]
    dts = [(pd.to_datetime(track[i][0]) - pd.to_datetime(track[i - 1][0])).total_seconds() / 3600.0
           for i in range(1, len(track))]
    path = float(np.sum(seg))
    if len(track) < trk.min_track_length:
        out.append("npts")
    if dur < trk.min_hours:
        out.append("hours")
    if path < trk.min_path_km:
        out.append("path")
    sp = [d * 1000.0 / (h * 3600.0) for d, h in zip(seg, dts) if h > 0]
    if sp and not (trk.min_median_speed_ms <= np.median(sp) <= trk.max_median_speed_ms):
        out.append("median_speed")
    if dur > 0 and path * 1000.0 / (dur * 3600.0) < trk.min_avg_propagation_ms:
        out.append("propagation")
    frac, longest = trk.calculate_retrograde_stats(track)
    if longest > trk.max_retro_run_hours or frac > trk.max_retro_total_frac:
        out.append("retrograde")
    if trk.require_net_east and trk.calculate_net_zonal_displacement(track) < trk.min_net_east_km:
        out.append("net_east")
    return out, seg, dts


def sector_of(la, lo):
    for name, lo0, lo1, la0, la1 in SECTORS:
        if lo0 <= lo <= lo1 and la0 <= la <= la1:
            return name
    return "other"


def run_year(year, args):
    from updated_monthly_ETC_Tracker import ETCTracker
    trk = ETCTracker(max_distance_km=800, min_track_length=8,
                     region_bounds={"lon_min": -180.0, "lon_max": 180.0, "lat_min": -70.0, "lat_max": 70.0})
    refs = load_naecc(year, year)
    Tt, Tla, Tlo, Tid = load_wnet_tracks((year - 1, year, year + 1))
    det = Detections(trk)
    rows = []
    for r in sorted(refs, key=lambda x: x["t"][0]):                       # time order keeps the month cache hot
        n = r["t"].size
        inm, dc = np.array([det.at(t, la, lo) for t, la, lo in zip(r["t"], r["lat"], r["lon"])]).T
        ok = np.isfinite(inm)
        if ok.sum() < 4:
            continue
        # valid WNet tracks near the cyclone: any track, and the best single track
        i0, i1 = np.searchsorted(Tt, r["t"].min() - TOL_H, "left"), np.searchsorted(Tt, r["t"].max() + TOL_H, "right")
        any_tr = best_tr = 0.0
        sep = np.nan
        if i1 > i0:
            d = hav(r["lat"][:, None], r["lon"][:, None], Tla[None, i0:i1], Tlo[None, i0:i1])
            near = (d <= TOL_KM) & (np.abs(r["t"][:, None] - Tt[None, i0:i1]) <= TOL_H)
            any_tr = float(near.any(1).mean())
            ids = Tid[i0:i1]
            for u in np.unique(ids[near.any(0)]):
                f = float(near[:, ids == u].any(1).mean())
                if f > best_tr:
                    best_tr = f
                    dd = np.where(near[:, ids == u], d[:, ids == u], np.inf).min(1)
                    sep = float(dd[np.isfinite(dd)].mean())
        # what the tracker's rules would do to the NAECC track itself
        track = [(EPOCH + np.timedelta64(int(t), "h"), lo, la, 0.0) for t, la, lo in zip(r["t"], r["lat"], r["lon"])]
        reasons, seg, dts = validation_reasons(trk, track)
        kmh = np.array(seg) / np.maximum(np.array(dts), 1e-9)
        slow = kmh < trk.min_speed_kmh
        fast = kmh > trk.max_speed_kmh
        ceiling, start = False, 0
        for j in range(len(seg) + 1):                                      # segments between unlinkable steps
            if j == len(seg) or slow[j] or fast[j]:
                piece = track[start:j + 1]
                if len(piece) >= max(trk.min_track_length, int(np.ceil(0.5 * n))) and \
                        trk.validate_track_meteorology(piece):
                    ceiling = True
                start = j + 1
        mid = n // 2
        rows.append(dict(
            storm=r["storm"], year=r["year"], month=r["month"], npts=n, minmslp=r["minmslp"],
            mlat=r["lat"][mid], mlon=r["lon"][mid], sector=sector_of(r["lat"][mid], r["lon"][mid]),
            frac_gt70=float((r["lat"] > 70).mean()), frac_lt28=float((r["lat"] < 28).mean()),
            cov_mask=float(np.nanmean(inm)), cov_c300=float(np.mean(dc[ok] <= 300)),
            cov_c500=float(np.mean(dc[ok] <= 500)),
            med_centre_km=float(np.median(dc[ok][np.isfinite(dc[ok])])) if np.isfinite(dc[ok]).any() else np.nan,
            any_track=any_tr, best_track=best_tr, sep_km=sep,
            path_km=float(np.sum(seg)), net_km=float(hav(r["lat"][0], r["lon"][0], r["lat"][-1], r["lon"][-1])),
            median_ms=float(np.median(kmh) / 3.6), n_slow=int(slow.sum()), n_fast=int(fast.sum()),
            val_fail="|".join(reasons), ceiling_ok=int(ceiling)))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"cyclones_{year}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {year}: {len(rows)} NAECC cyclones -> {out}", flush=True)


def stage_of(df):
    # earliest stage that lost the cyclone wins: mask -> centre -> track -> fragment
    s = np.full(len(df), "matched", dtype=object)
    miss = (df["best_track"] < 0.5).values
    s[miss] = "fragment"
    s[miss & (df["any_track"] < 0.5).values] = "track"
    s[miss & (df["cov_c500"] < 0.5).values] = "centre"
    s[miss & (df["cov_mask"] < 0.5).values] = "mask"
    return s


def summarize(args):
    files = sorted(glob.glob(os.path.join(args.outdir, "cyclones_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["val_fail"] = df["val_fail"].fillna("")
    df["stage"] = stage_of(df)
    df["depth"] = pd.qcut(df["minmslp"], 5, labels=["deepest", "q2", "q3", "q4", "shallowest"])
    df["season"] = df["month"].map(lambda m: "cold(Oct-Mar)" if m in (10, 11, 12, 1, 2, 3) else "warm(Apr-Sep)")
    order = ["matched", "fragment", "track", "centre", "mask"]
    lines = [f"ETC miss attribution: {len(df)} NAECC cyclones, {len(files)} years "
             f"({int(df.year.min())}-{int(df.year.max())}), NAECC domain, match = one track within "
             f"{TOL_KM:.0f} km and {TOL_H} h of >= 50 % of the points", ""]

    def table(by):
        t = (df.groupby(by, observed=True)["stage"].value_counts(normalize=True).unstack(fill_value=0.0)
             .reindex(columns=order, fill_value=0.0))
        t.insert(0, "n", df.groupby(by, observed=True).size())
        t["cov_mask"] = df.groupby(by, observed=True)["cov_mask"].mean()
        t["ceiling_ok"] = df.groupby(by, observed=True)["ceiling_ok"].mean()
        return t.round(3).to_string()

    df["all"] = "all"
    for by in ("all", "depth", "season", "sector"):
        lines += [f"-- by {by}: share of cyclones by the stage that lost them; mean mask coverage; "
                  f"share reachable under the shipped tracker rules", table(by), ""]
    lines += ["-- shipped tracker rules applied to the NAECC tracks themselves (share of cyclones failing)"]
    for k in ("path", "median_speed", "propagation", "retrograde", "net_east", "hours", "npts"):
        lines.append(f"   {k:<14} {df['val_fail'].str.contains(k).mean():.3f}")
    lines += [f"   any criterion  {(df['val_fail'] != '').mean():.3f}",
              f"   >= 1 step under the 72 km link floor: {(df['n_slow'] > 0).mean():.3f} of cyclones, "
              f"{df['n_slow'].sum() / (df['npts'] - 1).sum():.3f} of steps",
              f"   reachable (ceiling_ok): {df['ceiling_ok'].mean():.3f}", "",
              "-- 'track' misses (centres present, no valid track): share explained by the shipped rules",
              f"   not reachable under the rules: {(1 - df.loc[df.stage == 'track', 'ceiling_ok'].mean()):.3f}", "",
              "-- centre offset where the mask covers the cyclone (cov_mask >= 0.5): median km to nearest centre",
              df[df.cov_mask >= 0.5].groupby("sector")["med_centre_km"].median().round(0).to_string(), ""]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    df.to_csv(os.path.join(args.outdir, "cyclones_all.csv"), index=False)
    json.dump({"n": int(len(df)), "stage_share": df["stage"].value_counts(normalize=True).round(4).to_dict(),
               "ceiling_ok": round(float(df["ceiling_ok"].mean()), 4)},
              open(os.path.join(args.outdir, "summary.json"), "w"), indent=1)


def main():
    global DET_DIR, TRK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2010])
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--det-dir", default=DET_DIR,
                    help="ETC masks to test; a LABEL written as era5dara_YYYYMM_ETCs_detection.nc gives "
                         "the label's own NAECC coverage (cov_mask) before any training")
    ap.add_argument("--trk-dir", default=TRK_DIR, help="valid WNet tracks for the track / fragment stages")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    DET_DIR, TRK_DIR = args.det_dir, args.trk_dir
    if args.summarize:
        return summarize(args)
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    for y in list(range(args.years[0], args.years[1] + 1))[rank::world]:
        if os.path.exists(os.path.join(args.outdir, f"cyclones_{y}.csv")) and not args.overwrite:
            print(f"  {y}: done", flush=True)
            continue
        run_year(y, args)


if __name__ == "__main__":
    sys.exit(main())
