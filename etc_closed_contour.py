#!/usr/bin/env python3
"""
etc_closed_contour.py — cyclone centres as closed MSLP lows, one definition shared by the ETC
tracker (ETC_CENTER_MODE=mslp) and, later, the ETC label.

The shipped label accepts a local MSLP minimum when the MAXIMUM pressure inside an 800 km box
exceeds it by 1 hPa (dataset.py:1110-1142). That is not a closed contour: almost any minimum
passes, including the artefacts of sea-level reduction over high terrain, which is why WNet
has no track-density minimum over the western cordillera while NAECC, TRACK and
TempestExtremes do. The shipped tracker then takes the CENTROID of each connected mask blob
(a union of 600 km discs), so two lows whose discs touch become one centre between them.

Here a centre is a local minimum of lightly smoothed MSLP that is enclosed by a closed contour:
every path from the minimum must climb by `depth_pa` before it has gone `radius_deg` of great
circle from it. This is the TempestExtremes criterion (--closedcontourcmd SLP,200,6,0), tested
by flood-filling the region p < p0 + depth_pa from the minimum and requiring that the filled
region stays inside the radius. Minima closer than `merge_deg` are merged, deepest first
(--mergedist 6). A lee cyclone closes a 2 hPa contour once it is off the terrain; a reduction
artefact embedded in a broad plateau low does not. Summer heat lows over plateaus CAN close a
contour, so this is a physical filter, not a terrain mask.

Grid: regular latitude/longitude, periodic in longitude; any latitude order. A regional grid
(RegCM) passes periodic=False: no longitude wrap, NaN (outside the domain) is ignored by the
smoothing instead of spreading, and a low whose closing region touches NaN or the array edge
is NOT closed. Otherwise the NaN acts as a wall and a low pressed against the domain edge (a
cyclone centred outside it) passes as closed, which is the edge artefact TempestExtremes' own
regional output shows (44-63 % of its RegCM ETC points on the boundary). The default
reproduces the ERA5 behaviour exactly.
"""
import numpy as np
from scipy import ndimage

R_KM = 6371.0
_S8 = np.ones((3, 3), bool)


def smooth(mslp, lat, sigma_km=75.0, periodic=True):
    """Gaussian smoothing, periodic in longitude. The zonal sigma is converted at 45 degrees
    so one call serves the whole field (as the label's _smooth_mslp does). periodic=False:
    no wrap, and NaN cells are left out of the average (normalised convolution) and stay NaN."""
    dlat = abs(float(lat[1] - lat[0]))
    sy = sigma_km / (111.2 * dlat)
    sx = sigma_km / (111.2 * dlat * np.cos(np.radians(45.0)))
    if periodic:
        return ndimage.gaussian_filter(mslp.astype(np.float64), (sy, sx), mode=("nearest", "wrap"))
    a = np.asarray(mslp, np.float64)
    ok = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), (sy, sx), mode="nearest")
    den = ndimage.gaussian_filter(ok.astype(np.float64), (sy, sx), mode="nearest")
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan)


def _gcd_deg(lat0, lon0, lat, lon):
    la0, lo0, la, lo = map(np.radians, (lat0, lon0, lat, lon))
    c = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(lo - lo0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def closed_lows(mslp, lat, lon, candidate=None, depth_pa=200.0, radius_deg=6.0, merge_deg=6.0,
                sigma_km=75.0, max_abs_lat=88.0, periodic=True):
    """Closed lows of one MSLP field (Pa).

    candidate   optional boolean mask; only minima inside it are tested (for the tracker: the
                filled and slightly grown mask blobs, because the wind gate of the label leaves
                a hole at the calm centre of a cyclone)
    periodic    False for a regional grid: see the module docstring
    returns     list of (lat, lon, p_min_pa), deepest first
    """
    p = smooth(mslp, lat, sigma_km, periodic) if sigma_km else np.asarray(mslp, np.float64)
    ny, nx = p.shape
    if periodic:
        is_min = p == ndimage.minimum_filter(p, size=3, mode=("nearest", "wrap"))
    else:
        pf = np.where(np.isfinite(p), p, np.inf)            # NaN can be neither a minimum nor beat one
        is_min = np.isfinite(p) & (pf == ndimage.minimum_filter(pf, size=3, mode="nearest"))
    is_min &= (np.abs(np.asarray(lat, float)) <= max_abs_lat)[:, None]
    if candidate is not None:
        is_min &= candidate
    iy, ix = np.nonzero(is_min)
    order = np.argsort(p[iy, ix])
    iy, ix = iy[order], ix[order]

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    hy = int(np.ceil((radius_deg + 1.0) / dlat))
    out = []
    for y, x in zip(iy, ix):
        la0, lo0, p0 = float(lat[y]), float(lon[x]), float(p[y, x])
        if any(_gcd_deg(la0, lo0, a, b) < merge_deg for a, b, _ in out):
            continue                                         # a deeper low already owns this one
        # a window wide enough to hold the radius at this latitude, wrapped in longitude
        hx = min(int(np.ceil((radius_deg + 1.0) / (dlon * max(np.cos(np.radians(la0)), 0.05)))), nx // 2)
        ys = np.arange(max(y - hy, 0), min(y + hy + 1, ny))
        if periodic:
            xs = np.arange(x - hx, x + hx + 1) % nx
        else:
            xs = np.arange(max(x - hx, 0), min(x + hx + 1, nx))   # clipped: the array edge is a window edge
        win = p[np.ix_(ys, xs)]
        lab, _ = ndimage.label(win < p0 + depth_pa, structure=_S8)
        region = lab == lab[y - ys[0], hx if periodic else x - xs[0]]
        dist = _gcd_deg(la0, lo0, lat[ys][:, None], lon[xs][None, :])
        # closed: the region below p0 + depth neither leaves the radius nor reaches the window edge
        edge = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        if not periodic and not edge:
            # nor touches the outside of the domain, where the pressure is unknown
            edge = bool((ndimage.binary_dilation(region, structure=_S8) & ~np.isfinite(win)).any())
        if not edge and dist[region].max() <= radius_deg:
            out.append((la0, lo0, p0))
    return out


def blob_candidate(blobs, lat, grow_deg=2.0):
    """The tracker's search region for the mask blobs: holes filled, grown by grow_deg."""
    dlat = abs(float(lat[1] - lat[0]))
    n = max(int(round(grow_deg / dlat)), 1)
    filled = ndimage.binary_fill_holes(blobs)
    return ndimage.binary_dilation(filled, structure=_S8, iterations=n)
