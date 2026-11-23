"""Teammate channels of the multi-UAV observation, design M2 (docs/multi_agent_design.md "M1 점검과 다음 학습(M2) 설계", agreed with the user
2026-11-20 ~ 11-23; previews in scripts/multi_design/mate_channels_view.py, territory_ab_view.py). Every map is heading up with the UAV in the
centre (as ``rl_real.RealRLEnv._bin``); values are taken at the cell centres with straight distances in metres.

- small maps (side <= 6.4 km): ``claim_map`` (③ 동료 영역) and ``danger_map`` of the teammates' ``reach_fans`` (위험 구역)
- large maps: ``time_adv_map`` (안 B)
"""
from __future__ import annotations

import numpy as np

from rl_search.plants import G

FRACS = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)    # command sizes of the fan's turns (straight on is the n = 0 path)
REACH_H = 0.25                     # integration step of the fans (s): at most about 20 m from the 0.1 s flight model (11-23 check)
DANGER_LOW = 0.3                   # danger value at the map's lead (1 at the teammate now)


def cell_grid(x: float, y: float, yaw: float, side: float, px: int):
    """(forward, right) offsets of the cell centres of a heading-up map and their world coordinates, each (px, px)."""
    cell = side/px
    c = (np.arange(px) + 0.5)*cell
    F, R = np.meshgrid(side/2 - c, c - side/2, indexing="ij")
    cs, sn = np.cos(yaw), np.sin(yaw)
    return F, R, x + F*cs + R*sn, y + F*sn - R*cs


def reach_fans(plants, chi_tgts, turns, dec_s: float, T: float, h: float = REACH_H, every_s: float = 1.0, fracs=FRACS) -> list:
    """Where each UAV can be within T s: its flight model (``plants.FixedWing``, the equations of ``rl_multi.predict_paths``) flown with a
    command of each size in ``fracs`` for the first n decisions (n = 0 .. T / dec_s; a command moves the course target by size x turn) and
    straight on after. All UAVs in one integration loop (the plants share their model). Returns [(points (M, 2), times (M,))] per UAV,
    the present position first (time 0)."""
    if not len(plants):
        return []
    p0 = plants[0]
    n_dec = int(np.ceil(T/dec_s))
    A = np.array([np.zeros(n_dec)] + [np.r_[np.full(n, f), np.zeros(n_dec - n)] for f in fracs for n in range(1, n_dec + 1)])
    P = len(A)
    U = len(plants)
    chi_seq = np.concatenate([chi + np.cumsum(A, axis=1)*tu for chi, tu in zip(chi_tgts, turns)])        # (U * P, n_dec)
    x = np.repeat([float(p.pose[0]) for p in plants], P)
    y = np.repeat([float(p.pose[1]) for p in plants], P)
    yaw = np.repeat([float(p.pose[2]) for p in plants], P)
    phi = np.repeat([float(p.phi) for p in plants], P)
    r = np.repeat([float(p.r) for p in plants], P)
    v = float(p0.v)
    w2, c2 = p0.omega**2, 2*p0.zeta*p0.omega
    n, k, per_dec = int(round(T/h)), max(1, int(round(every_s/h))), int(round(dec_s/h))
    pts, ts = [], []
    for i in range(n):
        chi = chi_seq[:, min(i//per_dec, n_dec - 1)]
        err = (chi - yaw + np.pi) % (2*np.pi) - np.pi
        r = r + (w2*err - c2*r)*h
        phi_cmd = np.clip(np.arctan(r*v/G), -p0.phi_max, p0.phi_max)
        phi = phi + np.clip((phi_cmd - phi)/p0.tau, -p0.p_max, p0.p_max)*h
        yaw = yaw + G*np.tan(phi)/v*h
        x, y = x + v*np.cos(yaw)*h, y + v*np.sin(yaw)*h
        if (i + 1) % k == 0:
            pts.append(np.stack([x, y], 1).reshape(U, P, 2))
            ts.append((i + 1)*h)
    out = []
    for u, p in enumerate(plants):
        pu = np.concatenate([np.array([p.pose[:2]], float)] + [q[u] for q in pts])
        tu = np.concatenate([[0.0]] + [np.full(P, t) for t in ts])
        out.append((pu, tu))
    return out


def claim_map(x: float, y: float, yaw: float, mates_xy, side: float, px: int = 64) -> np.ndarray:
    """③ 동료 영역: every teammate j claims a cell by (d_me - d_j) / |me - j| in [-1, 1] (triangle inequality; -1 = the cell lies beyond me
    as seen from j, +1 = beyond j, 0 = on our bisector; far away it keeps the direction); the value is the strongest claim (continuous).
    All -1 with no teammate airborne (everything is mine)."""
    if not len(mates_xy):
        return -np.ones((px, px), np.float32)
    F, R, WX, WY = cell_grid(x, y, yaw, side, px)
    M = np.asarray(mates_xy, float)
    D = np.sqrt((WX[None] - M[:, 0, None, None])**2 + (WY[None] - M[:, 1, None, None])**2)
    sep = np.maximum(np.hypot(M[:, 0] - x, M[:, 1] - y), 1.0)[:, None, None]
    return np.clip(((np.sqrt(F*F + R*R)[None] - D)/sep).max(axis=0), -1.0, 1.0).astype(np.float32)


def disk_max(g: np.ndarray, rc: int) -> np.ndarray:
    """Largest value within a disk of radius rc cells around every cell: a 1-D max filter along the rows with the half-width of each disk
    row, shifted up and down (the same result as ``scipy.ndimage.grey_dilation`` with a disk footprint, about 12 times faster for rc 10)."""
    from scipy.ndimage import maximum_filter1d
    if rc <= 0:
        return g.copy()
    out = np.zeros_like(g)
    H = g.shape[0]
    rows = {}
    for dy in range(-rc, rc + 1):
        w = int(np.floor(np.sqrt(rc*rc - dy*dy)))
        if w not in rows:
            rows[w] = maximum_filter1d(g, size=2*w + 1, axis=1, mode="constant")
        m = rows[w]
        if dy >= 0:
            out[dy:] = np.maximum(out[dy:], m[:H - dy])
        else:
            out[:dy] = np.maximum(out[:dy], m[-dy:])
    return out


def danger_map(x: float, y: float, yaw: float, fans, side: float, lead: float, d: float, px: int = 64) -> np.ndarray:
    """위험 구역: the teammates' fans (``reach_fans``) up to ``lead`` s with value 1 - (1 - DANGER_LOW) x t / lead, each point spread over
    a disk of radius ``d`` (the penalty distance), the largest value per cell. 0 where no teammate can be."""
    cell = side/px
    rc = int(np.floor(d/cell))
    pad = rc + 1
    g = np.zeros((px + 2*pad, px + 2*pad), np.float32)
    cs, sn = np.cos(yaw), np.sin(yaw)
    for pts, ts in fans:
        m = ts <= lead + 1e-9
        dd = pts[m] - np.array([x, y])
        fwd, right = dd[:, 0]*cs + dd[:, 1]*sn, dd[:, 0]*sn - dd[:, 1]*cs
        row = np.floor((side/2 - fwd)/cell).astype(int) + pad
        col = np.floor((right + side/2)/cell).astype(int) + pad
        ok = (row >= 0) & (row < g.shape[0]) & (col >= 0) & (col < g.shape[1])
        val = (1.0 - (1.0 - DANGER_LOW)*ts[m]/max(lead, 1e-9)).astype(np.float32)
        np.maximum.at(g, (row[ok], col[ok]), val[ok])
    return disk_max(g, rc)[pad:pad + px, pad:pad + px]


def time_adv_map(x: float, y: float, yaw: float, mates_xy, side: float, lead: float, speed: float, px: int = 64) -> np.ndarray:
    """안 B: (d_me - d_nearest teammate) / (speed x lead), clipped to [-1, 1] (0 = we reach the cell at the same time; its size is at most
    our separation / (speed x lead), so a team close together reads 0 on a large map). All -1 with no teammate airborne."""
    if not len(mates_xy):
        return -np.ones((px, px), np.float32)
    F, R, WX, WY = cell_grid(x, y, yaw, side, px)
    M = np.asarray(mates_xy, float)
    d_m = np.sqrt(np.min((WX[None] - M[:, 0, None, None])**2 + (WY[None] - M[:, 1, None, None])**2, axis=0))
    return np.clip((np.sqrt(F*F + R*R) - d_m)/(speed*max(float(lead), 1.0)), -1.0, 1.0).astype(np.float32)
