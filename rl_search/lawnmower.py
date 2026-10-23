"""Lawnmower baseline (stage 3): parallel lanes over a fixed search rectangle, flown as a waypoint list.

At handover the rectangle is fixed once: the ``hdr`` (80 %) highest-density region of the in-map
particles between handover and the end of the energy (KDE on a ``cell_m`` grid, weights as they are
at handover), enclosed by a rectangle aligned with the target's heading and clipped to the map.
Lanes run along the heading, ``lane_spacing_m`` apart (default: footprint width), visited with a
skip of k = max(1, ceil(2 R_eff / spacing)) lanes so every turn is flyable; the pattern repeats in
reverse once the rectangle is covered. The follower flies it like any other waypoint mission.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from rl_search.waypoints import Waypoints

EDGE_MARGIN_M = 300.0   # lanes end this far inside the map, so turnarounds stay on the map


@dataclass
class Rectangle:
    heading: float          # rad, lane direction
    u: tuple                # (min, max) along the heading
    v: tuple                # (min, max) across
    hdr_area_km2: float

    @property
    def corners(self) -> np.ndarray:
        return np.array([self.xy(u, v) for u, v in ((self.u[0], self.v[0]), (self.u[1], self.v[0]),
                                                     (self.u[1], self.v[1]), (self.u[0], self.v[1]))])

    @property
    def area_km2(self) -> float:
        return (self.u[1] - self.u[0])*(self.v[1] - self.v[0])/1e6

    def xy(self, u, v) -> np.ndarray:
        c, s = np.cos(self.heading), np.sin(self.heading)
        return np.array([u*c - v*s, u*s + v*c])


def search_rectangle(belief, t_from: float, t_to: float, heading: float, half: float, cfg, bandwidth_m: float):
    """Rectangle around the HDR of the in-map particles over [t_from, t_to]."""
    times = np.linspace(t_from, min(t_to, belief.duration_s), cfg.n_snapshots)
    xs, ws = [], []
    for t in times:
        live = belief.active(t) & (belief.w > 0)
        p = belief.positions(t)[live]
        ok = (np.abs(p) <= half).all(axis=1)
        xs.append(p[ok])
        ws.append(belief.w[live][ok])
    pts, w = np.vstack(xs), np.concatenate(ws)
    c = cfg.cell_m
    edges = np.arange(-half, half + c, c)
    hist, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[edges, edges], weights=w)
    dens = gaussian_filter(hist, bandwidth_m/c, mode="constant")
    v = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(v)
    thr = v[min(np.searchsorted(cum, cfg.hdr*cum[-1]), len(v) - 1)]
    rows, cols = np.nonzero((dens >= thr) & (dens > 0))
    centres = edges[:-1] + c/2
    cx, cy = centres[cols], centres[rows]
    ch, sh = np.cos(heading), np.sin(heading)
    u, vv = cx*ch + cy*sh, -cx*sh + cy*ch
    return Rectangle(heading, (u.min() - c/2, u.max() + c/2), (vv.min() - c/2, vv.max() + c/2),
                     hdr_area_km2=len(rows)*c*c/1e6)


def clip_segment(p0, p1, lo, hi):
    """Liang-Barsky clip of p0-p1 to the square [lo, hi]^2; None if outside."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(2):
        for pk, qk in ((-d[k], p0[k] - lo), (d[k], hi - p0[k])):
            if abs(pk) < 1e-12:
                if qk < 0:
                    return None
                continue
            r = qk/pk
            if pk < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    if t0 >= t1:
        return None
    return p0 + t0*d, p0 + t1*d


def lanes(rect: Rectangle, spacing: float, half: float, margin: float = EDGE_MARGIN_M):
    """Lane end points (n, 2, 2) across the rectangle, clipped to the map minus ``margin``."""
    n = max(1, int(np.ceil((rect.v[1] - rect.v[0])/spacing)))
    v0 = 0.5*(rect.v[0] + rect.v[1]) - 0.5*(n - 1)*spacing
    out = []
    for i in range(n):
        v = v0 + i*spacing
        seg = clip_segment(rect.xy(rect.u[0], v), rect.xy(rect.u[1], v), -half + margin, half - margin)
        if seg is not None:
            out.append(np.array(seg))
    return np.array(out).reshape(-1, 2, 2)


def visit_order(n: int, k: int) -> list[int]:
    return [j for start in range(k) for j in range(start, n, k)]


def lawnmower_path(ends: np.ndarray, k: int, start_xy, length_m: float) -> np.ndarray:
    """Boustrophedon through the lanes (skip k), repeated back and forth until ``length_m`` is flown."""
    order = visit_order(len(ends), k)
    best = None
    for seq in (order, order[::-1]):
        for flip in (False, True):
            first = ends[seq[0]][1 if flip else 0]
            d = np.hypot(*(first - start_xy))
            if best is None or d < best[0]:
                best = (d, seq, flip)
    _, seq, flip = best
    pts = [np.asarray(start_xy, float)]
    total, rev = 0.0, flip
    for _ in range(10000):
        if total >= length_m:
            break
        for i in seq:
            a, b = ends[i][::-1] if rev else ends[i]
            for p in (a, b):
                total += np.hypot(*(p - pts[-1]))
                pts.append(p)
            rev = not rev
        seq = seq[::-1]
    return np.array(pts)


def build_lawnmower(env, cfg_lawn, t_end_s: float | None = None):
    """Waypoints from the env's handover state: (Waypoints, info)."""
    sc = env.cfg
    t0 = env.t_s
    if t_end_s is None:
        t_end_s = t0 + (env.power.endurance_s(env.energy_left) if env.power is not None
                        else (sc.n_steps - env.k)*sc.dt_s)
    heading = np.deg2rad(env.scenario.heading_deg)
    rect = search_rectangle(env.belief, t0, t_end_s, heading, env.half, cfg_lawn, sc.bandwidth_m)
    spacing = cfg_lawn.lane_spacing_m or sc.fov_cross_m
    r_eff = sc.step_m/max(env.plant.max_turn, 1e-6)
    k = max(1, int(np.ceil(2*r_eff/spacing)))
    ends = lanes(rect, spacing, env.half)
    v = sc.uav_speed_mps
    length = cfg_lawn.repeat_margin*v*(t_end_s - t0)
    xy = lawnmower_path(ends, k, env.pose[:2], length) if len(ends) else np.array([env.pose[:2]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    wp = Waypoints(xy, t0 + s/v)
    info = dict(rect_corners=rect.corners.tolist(), rect_km2=rect.area_km2, hdr_km2=rect.hdr_area_km2,
                n_lanes=len(ends), skip=k, spacing_m=spacing, r_eff_m=r_eff, t_end_s=t_end_s,
                coverage_s=float(sum(np.hypot(*(e[1] - e[0])) for e in ends)/v))
    return wp, info
