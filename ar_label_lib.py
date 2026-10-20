#!/usr/bin/env python3
"""
ar_label_lib.py — the literature-based atmospheric river label ("WNet-AR-lit").

The shipped AR label is one global scalar, IVT > 400 kg/m/s, with no geometry and no
direction (dataset.py:629-633). That single number produces the tropical bands (trade
winds, monsoon jets and the Caribbean low-level jet exceed 400 most of the time), the
coastline cut-off and the empty polar caps (IVT decays inland and poleward), and the
doubled seasonal cycle. This module builds the label the way the literature defines an AR:

  intensity   IVT > max( percentile of IVT AT THAT CELL for the 5-month season centred on
              the month, absolute floor(latitude) ). The relative part follows Guan and
              Waliser (2015); it removes climatologically moist regions and the summer
              inflation. The floor removes hydrologically weak features that a purely
              relative threshold flags in dry or cold interiors: 100 kg/m/s (Guan-Waliser)
              or the hybrid 250 kg/m/s of Rutz et al. (2014) and the AR scale of Ralph et
              al. (2019), tapering to 100 in the polar regions where ARs are inherently weak.
  geometry    length >= 2000 km and length / width >= 2. Length is the largest great-circle
              distance between two cells of the object's perimeter and width is the
              object's area divided by its length, as in Guan and Waliser (2015). Nothing
              depends on a km-per-pixel constant.
  transport   mean poleward IVT > 50 kg/m/s; at least half of the cells within 45 degrees of
              the object-mean IVT direction; object-mean IVT within 45 degrees of the
              object's axis. These remove zonal trade-wind and monsoon features.
  iteration   an object that fails is thresholded again inside its own footprint at the
              87.5th, 90th, 92.5th and 95th percentile (tARget v2/v3), which separates an AR
              from a broad region of high IVT it is embedded in.
  tropics     optional: an object must reach |latitude| >= 23.5. Objects confined to the
              tropics are dropped; the tropical root of a real AR is kept.
  width       optional: width <= max_width_km. Guan-Waliser has no upper width, so a broad
              zonal band of moist westerlies (3000 km wide, length/width 3.7, mean poleward
              IVT barely over 50) passes whole. ARs are "long and NARROW": under about
              1000 km in Ralph et al. (2004) and Wick et al. (2013). Width here is
              area / length, which overstates the width of a curved or branched object, so
              the default test value is 1500 km. A too-wide object is not dropped but
              thresholded again at the next percentile, which isolates its core.

Grid: ERA5 native, latitude 90 to -90, longitude 0 to 359.75 (periodic in longitude). A regional
grid (RegCM) passes periodic=False to detect(); the default reproduces the ERA5 label exactly.
"""
import numpy as np
from scipy import ndimage

R_KM = 6371.0
LEVELS = (85.0, 87.5, 90.0, 92.5, 95.0)
COS45 = np.cos(np.radians(45.0))
DEFAULTS = dict(min_length_km=2000.0, min_lw=2.0, min_poleward=50.0, min_coherent=0.5,
                min_area_km2=1.0e5, reach_lat=23.5, max_width_km=None)
_S8 = np.ones((3, 3), bool)


def cell_area(lat, lon):
    """Grid-cell area in km2, (ny, nx)."""
    dlat, dlon = np.radians(abs(lat[1] - lat[0])), np.radians(abs(lon[1] - lon[0]))
    a = R_KM ** 2 * dlat * dlon * np.clip(np.cos(np.radians(lat)), 0.0, None)
    return np.repeat(a[:, None], lon.size, axis=1)


def floor_profile(lat, kind):
    """Absolute IVT floor by latitude, (ny, 1). 'gw': 100 everywhere. 'hybrid' (250) or
    'hybridNNN': NNN equatorward of 55, linear to 100 at 65 and poleward."""
    a = np.abs(np.asarray(lat, float))
    if kind == "gw":
        f = np.full(a.shape, 100.0)
    elif kind.startswith("hybrid"):
        top = float(kind[6:] or 250.0)
        f = np.where(a <= 55.0, top, np.where(a >= 65.0, 100.0, top - (a - 55.0) * (top - 100.0) / 10.0))
    else:
        raise ValueError(kind)
    return f[:, None].astype(np.float32)


def label_wrap(binary):
    """8-connected labels with the first and last longitude treated as neighbours."""
    lab, n = ndimage.label(binary, structure=_S8)
    if n < 2:
        return lab, n
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    left, right = lab[:, 0], lab[:, -1]
    for dy in (-1, 0, 1):                                   # 8-connectivity across the seam
        a = left[max(dy, 0):left.size + min(dy, 0)]
        b = right[max(-dy, 0):right.size + min(-dy, 0)]
        k = (a > 0) & (b > 0) & (a != b)
        for x, y in set(zip(a[k].tolist(), b[k].tolist())):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(x) for x in range(n + 1)])
    uniq, lut = np.unique(roots, return_inverse=True)       # 0 stays 0
    return lut[lab], uniq.size - 1


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _diameter(P):
    """Farthest pair of unit vectors P (n, 3) by repeated farthest-point sweeps: (i, j, km)."""
    i, best = 0, (0, 0, -1.0)
    for _ in range(6):
        d = P @ P[i]
        j = int(np.argmin(d))
        km = R_KM * float(np.arccos(np.clip(d[j], -1.0, 1.0)))
        if km <= best[2]:
            break
        best, i = (i, j, km), j
    return best


def object_metrics(sub, sl, qu, qv, ivt, lat, lon, area):
    """Geometry and transport of one object: `sub` is its boolean footprint inside slice `sl`."""
    iy, ix = np.nonzero(sub)
    la, lo = lat[sl[0]][iy], lon[sl[1]][ix]
    w = area[sl][sub]
    u, v = qu[sl][sub], qv[sl][sub]
    A = float(w.sum())
    per = sub & ~ndimage.binary_erosion(sub, structure=_S8, border_value=0)
    py, px = np.nonzero(per)
    P = _unit(lat[sl[0]][py], lon[sl[1]][px])
    i, j, length = _diameter(P)
    mu, mv = float((u * w).sum() / A), float((v * w).sum() / A)
    mnorm = max(np.hypot(mu, mv), 1e-9)
    cosang = (u * mu + v * mv) / (np.maximum(ivt[sl][sub], 1e-9) * mnorm)
    # axis = chord between the two farthest perimeter cells, in the local east/north frame
    # at the object's centre
    c = (_unit(la, lo) * w[:, None]).sum(0)
    c /= max(np.linalg.norm(c), 1e-12)
    clat, clon = np.arcsin(np.clip(c[2], -1, 1)), np.arctan2(c[1], c[0])
    east = np.array([-np.sin(clon), np.cos(clon), 0.0])
    north = np.array([-np.sin(clat) * np.cos(clon), -np.sin(clat) * np.sin(clon), np.cos(clat)])
    d = P[j] - P[i]
    ax, ay = float(d @ east), float(d @ north)
    align = abs(ax * mu + ay * mv) / (max(np.hypot(ax, ay), 1e-12) * mnorm)
    return dict(area=A, length=length, lw=length * length / max(A, 1e-9),
                poleward=float((np.sign(la) * v * w).sum() / A),
                coherent=float(((cosang >= COS45) * w).sum() / A), align=float(align),
                maxabslat=float(np.abs(la).max()))


def detect(qu, qv, thr, lat, lon, area, params=None, stats=None, periodic=True, edge=None,
           diag=None, relief=None, relief_min_km=1000.0):
    """AR masks for one time step.

    qu, qv   (ny, nx) eastward and northward IVT
    thr      (n_levels, ny, nx) intensity threshold per level, floor already applied
    periodic False for a regional grid (RegCM): plain 8-connected labels, no seam join
    edge     optional (ny, nx) bool, the cells on the domain boundary; with `diag` it counts
             objects that fail ONLY the length test while touching it (the domain edge may
             have cut them short). A diagnostic: it changes no mask.
    diag     optional dict of counters, updated in place
    relief   optional (ny, nx) bool, boundary cells through which ARs enter a regional domain
             (RegCM: its western edge). An object touching it that is at least relief_min_km
             long is not rejected for length, because the domain edge cut it short; it must
             still pass every other test (L/W, poleward, coherence, alignment).
    returns  (mask, mask_reach): all criteria, and additionally reaching |lat| >= reach_lat
    """
    p = dict(DEFAULTS, **(params or {}))
    labeller = label_wrap if periodic else (lambda b: ndimage.label(b, structure=_S8))
    ivt = np.hypot(qu, qv)
    acc = np.zeros(ivt.shape, bool)
    acc_reach = np.zeros(ivt.shape, bool)
    pending = np.ones(ivt.shape, bool)
    for k in range(thr.shape[0]):
        lab, n = labeller(pending & (ivt > thr[k]))
        pending = np.zeros(ivt.shape, bool)
        if n == 0:
            break
        areas = np.bincount(lab.ravel(), weights=area.ravel(), minlength=n + 1)
        slices = ndimage.find_objects(lab)
        for i in np.flatnonzero(areas[1:] >= p["min_area_km2"]) + 1:
            sl = slices[i - 1]
            sub = lab[sl] == i
            m = object_metrics(sub, sl, qu, qv, ivt, lat, lon, area)
            relieved = (relief is not None and m["length"] >= relief_min_km
                        and m["length"] < p["min_length_km"] and bool((relief[sl] & sub).any()))
            if m["length"] < p["min_length_km"] and not relieved:
                if diag is not None:
                    diag["short"] = diag.get("short", 0) + 1
                    if edge is not None and (edge[sl] & sub).any():
                        ok_else = (m["lw"] >= p["min_lw"] and m["poleward"] > p["min_poleward"]
                                   and m["coherent"] >= p["min_coherent"] and m["align"] >= COS45)
                        diag["short_at_edge"] = diag.get("short_at_edge", 0) + 1
                        diag["short_at_edge_only_length"] = (diag.get("short_at_edge_only_length", 0)
                                                             + int(ok_else))
                continue                                    # a higher threshold cannot lengthen it
            ok = (m["lw"] >= p["min_lw"] and m["poleward"] > p["min_poleward"]
                  and m["coherent"] >= p["min_coherent"] and m["align"] >= COS45
                  and (p["max_width_km"] is None or m["area"] / m["length"] <= p["max_width_km"]))
            if diag is not None:
                diag["relieved_accepted"] = diag.get("relieved_accepted", 0) + int(relieved and ok)
                diag["tested"] = diag.get("tested", 0) + 1
                diag["accepted"] = diag.get("accepted", 0) + int(ok)
            if ok:
                acc[sl] |= sub
                if m["maxabslat"] >= p["reach_lat"]:
                    acc_reach[sl] |= sub
                if stats is not None:
                    stats.append((k, m["area"], m["length"], m["lw"], m["poleward"], m["maxabslat"]))
            else:
                pending[sl] |= sub
    return acc, acc_reach
