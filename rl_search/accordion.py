"""SAROPS-style initial rectangle placement by accordion search (docs/sarops_approach_todo.md S.10, 3-1).

A rectangle (centre, length ``ell`` along the first leg, width ``wid``, orientation ``theta``) is searched by
parallel lanes. Its approximate POS is POC x (1 - exp(-W L / A)) with W the effective sweep width, L the
search path length and A the rectangle area. The accordion search starts from the smallest admissible
rectangle around the best cell and adds or removes one row/column of grid cells on any side while the
approximate POS improves (best of the 8 candidates each step).

Variants (how POC is counted; Kratzke et al. 2010 count it on a single snapshot):
  snap0   particles at the cycle start, no transit, axis-aligned  (SAROPS as written)
  mid     particles at the middle of the cycle, no transit, axis-aligned
  tavg    time-averaged POC over the search window [transit, cycle], L shortened by the transit
          (a rectangle farther than half a cycle still gets half a cycle of searching after arrival)
  tavg_pca  tavg in a grid rotated to the main axis of the top cluster
Fixed-wing: lanes n = L // (ell + pi R) (each lane end costs about a half turn of radius R), searched length
n*ell, so POD = 1 - exp(-W n ell / A). Constraints: ell >= ell_min, track spacing s = wid/n >= gamma.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rl_search.patterns import _rot, parallel_pts
from rl_search.patterns_v2 import FramedFollower, find_clusters

VARIANTS = ("snap0", "mid", "tavg", "tavg_pca")


@dataclass
class Rect:
    cx: float
    cy: float
    ell: float
    wid: float
    theta: float
    approx: float = 0.0


def sweep_width(env, n_int: int = 40) -> float:
    """Effective sweep width (m) of the env sensor model on a straight leg: integral of the cumulative POD
    across the track (kernel model: Gaussian-kernel POD update once per 5 s interval; lrc model: CPA + LRC)."""
    if getattr(env, "lrc", False):
        return float(sweep_profile_lrc(env, n_int)[1].sum()*10.0)
    c, h, v, pod = env.cfg.cell_m, env.h, env.v, env.search.pod
    dt = env.cfg.decision_dt_s
    d = np.arange(-800, 801, 10.0)
    X = n_int*dt*v/2
    logs = np.zeros_like(d)
    for k in range(n_int):
        poses = np.array([[(k*dt + s)*v, 0, 0] for s in np.arange(1, dt + 1)])
        iy, ix = np.divmod(env._cells_under(poses), env.n)
        cx, cy = env.lo + (ix + .5)*c, env.lo + (iy + .5)*c
        q = np.minimum(np.exp(-((X - cx[None])**2 + (d[:, None] - cy[None])**2)/(2*h*h)).sum(1)*c*c/(2*np.pi*h*h), 1)
        logs += np.log(1 - pod*q)
    return float((1 - np.exp(logs)).sum()*10.0)


def sweep_profile_lrc(env, n_int: int = 40, path=None):
    """Cumulative POD of stationary test particles at lateral offsets -800..800 m (10 m) after a straight leg
    (or ``path``: (S, 2) UAV positions at 1 s), using the env's own CPA + LRC code. Returns (offsets, POD)."""
    from rl_search.mission import ParticleBelief
    v, dt = env.v, env.cfg.decision_dt_s
    d = np.arange(-800, 801, 10.0)
    X = n_int*dt*v/2
    pts = np.stack([np.full_like(d, X), d], 1)
    save = getattr(env, "belief", None), getattr(env, "_pass_p", None), getattr(env, "_last_in", None)
    env.belief = ParticleBelief(np.stack([pts, pts]), 1e6, None, False)
    env._pass_p, env._last_in = np.zeros(len(d)), np.full(len(d), -np.inf)
    try:
        if path is None:
            for k in range(n_int):
                ts = k*dt + env.seg_t
                env._detect_lrc(np.stack([ts*v, np.zeros_like(ts)], 1), ts)
        else:
            for k in range(0, len(path) - 1, int(dt)):
                seg = path[k:k + int(dt) + 1]
                env._detect_lrc(seg, k + np.arange(len(seg), dtype=float))
        pod = 1 - env.belief.w
    finally:
        env.belief, env._pass_p, env._last_in = save
    return d, pod


class Accordion:
    def __init__(self, env, variant="tavg_pca", cycle_s=600.0, cell=200.0, W=None, effort=0.85,
                 ell_min=None, gamma=None, dt_snap=60.0):
        self.env, self.variant, self.T, self.g = env, variant, cycle_s, cell
        self.W = W if W is not None else sweep_width(env)
        self.effort = effort
        self.r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
        self.ell_min = ell_min if ell_min is not None else max(600.0, 4*self.r_turn)
        self.gamma = gamma if gamma is not None else 0.25*env.fp_cross
        self.taus = np.arange(0, 2*cycle_s + 1e-6, dt_snap)
        self.k_mid = int(np.argmin(np.abs(self.taus - cycle_s/2)))

    # ------------------------------------------------------------------ grid of remaining probability
    def _grid(self, theta):
        env, b = self.env, self.env.belief
        act = b.w > 0
        R = _rot(theta)
        Q, M = [], []
        for tau in self.taus:
            t = env.t + tau
            m = act & b.active(t)
            Q.append(b.positions(t) @ R)          # frame coords: x along theta
            M.append(np.where(m, b.w, 0.0)/b.n)
        Q, M = np.array(Q), np.array(M)
        sel = M.sum(0) > 0
        lo = np.quantile(Q[:, sel].reshape(-1, 2), .002, axis=0) - 2000
        hi = np.quantile(Q[:, sel].reshape(-1, 2), .998, axis=0) + 2000
        g = self.g
        ex, ey = np.arange(lo[0], hi[0] + g, g), np.arange(lo[1], hi[1] + g, g)
        H = np.stack([np.histogram2d(Q[k, :, 0], Q[k, :, 1], bins=[ex, ey], weights=M[k])[0]
                      for k in range(len(self.taus))])
        C = np.zeros((H.shape[0], H.shape[1] + 1, H.shape[2] + 1))
        C[:, 1:, 1:] = H.cumsum(1).cumsum(2)
        return dict(theta=theta, R=R, ex=ex, ey=ey, H=H, C=C, uav=env.plant.pose[:2] @ R)

    def _poc_t(self, G, i0, i1, j0, j1):
        C = G["C"]
        return C[:, i1, j1] - C[:, i0, j1] - C[:, i1, j0] + C[:, i0, j0]     # per snapshot

    def _transit(self, G, i0, i1, j0, j1):
        x0, x1, y0, y1 = G["ex"][i0], G["ex"][i1], G["ey"][j0], G["ey"][j1]
        corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        return float(np.hypot(*(corners - G["uav"]).T).min()/self.env.v)

    def approx(self, G, i0, i1, j0, j1):
        """Approximate POS of the index rectangle, or -1 if not admissible."""
        ell, wid = (i1 - i0)*self.g, (j1 - j0)*self.g
        v = self.variant
        tr = 0.0 if v in ("snap0", "mid") else self._transit(G, i0, i1, j0, j1)
        if tr > 1.5*self.T:
            return -1.0
        t_end = max(self.T, tr + self.T/2)        # far rectangles: still at least half a cycle of searching
        L = self.effort*self.env.v*(t_end - tr)
        n = L/(ell + np.pi*self.r_turn)          # lanes that fit (continuous), each lane end costs ~ a half turn
        if ell < self.ell_min or n < 1 or wid/n < self.gamma:
            return -1.0
        p = self._poc_t(G, i0, i1, j0, j1)
        if v == "snap0":
            poc = p[0]
        elif v == "mid":
            poc = p[self.k_mid]
        else:
            k = (self.taus >= tr - 1e-6) & (self.taus <= t_end + 1e-6)
            poc = p[k].mean() if k.any() else p[np.searchsorted(self.taus, tr) - 1]
        return float(poc*(1 - np.exp(-self.W*n*ell/(ell*wid))))

    # ------------------------------------------------------------------ search
    def theta(self):
        if self.variant != "tavg_pca":
            return 0.0
        cl = find_clusters(self.env, self.env.t + self.T/2, k=1)
        return cl[0].theta if cl else 0.0

    def plan(self) -> Rect | None:
        G = self._grid(self.theta())
        H = G["H"]
        nx, ny = H.shape[1:]
        dens = H[0] if self.variant == "snap0" else H[self.k_mid] if self.variant == "mid" else H[self.taus <= self.T].mean(0)
        if dens.max() <= 0:
            return None
        ci, cj = np.unravel_index(np.argmax(dens), dens.shape)
        # smallest admissible start around the best cell: ell_min long, as wide as the spacing limit needs
        L0 = self.effort*self.env.v*self.T
        a = int(np.ceil(self.ell_min/self.g))
        bw = max(1, int(np.ceil(self.gamma*L0/(self.ell_min + np.pi*self.r_turn)/self.g)))
        cur = [max(0, ci - a//2), min(nx, ci - a//2 + a), max(0, cj - bw//2), min(ny, cj - bw//2 + bw)]
        val = self.approx(G, *cur)
        for _ in range(10):       # grow until admissible (transit shortens L)
            if val >= 0:
                break
            cur = [max(0, cur[0] - 1), min(nx, cur[1] + 1), max(0, cur[2] - 1), min(ny, cur[3] + 1)]
            val = self.approx(G, *cur)
        if val < 0:
            return None
        while True:
            best, best_v = None, val
            for k in range(4):
                for d in (-1, 1):
                    c = list(cur)
                    c[k] += d if k in (1, 3) else -d       # d=+1 grows the side, -1 shrinks it
                    if not (0 <= c[0] < c[1] <= nx and 0 <= c[2] < c[3] <= ny):
                        continue
                    v = self.approx(G, *c)
                    if v > best_v + 1e-12:
                        best, best_v = c, v
            if best is None:
                break
            cur, val = best, best_v
        return self.to_rect(G, cur, val)

    def to_rect(self, G, idx, val) -> Rect:
        i0, i1, j0, j1 = idx
        cxf, cyf = (G["ex"][i0] + G["ex"][i1])/2, (G["ey"][j0] + G["ey"][j1])/2
        cx, cy = np.array([cxf, cyf]) @ G["R"].T
        return Rect(float(cx), float(cy), float((i1 - i0)*self.g), float((j1 - j0)*self.g), float(G["theta"]), val)

    def approx_rect(self, G, r: Rect) -> float:
        """Approximate POS of an arbitrary rectangle (same theta as G) by snapping it to the grid."""
        c = np.array([r.cx, r.cy]) @ G["R"]
        i0 = int(round((c[0] - r.ell/2 - G["ex"][0])/self.g))
        i1 = i0 + max(1, int(round(r.ell/self.g)))
        j0 = int(round((c[1] - r.wid/2 - G["ey"][0])/self.g))
        j1 = j0 + max(1, int(round(r.wid/self.g)))
        nx, ny = G["H"].shape[1:]
        if not (0 <= i0 < i1 <= nx and 0 <= j0 < j1 <= ny):
            return 0.0
        return max(self.approx(G, i0, i1, j0, j1), 0.0)


def lanes(env, r: Rect, cycle_s=600.0, effort=0.85):
    """Parallel lanes filling the rectangle (first leg along theta), started at the corner nearest the UAV.
    Returns (polyline in the rectangle frame, follower)."""
    r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
    x = env.plant.pose[:2]
    R = _rot(r.theta)
    corners = [np.array([sx*r.ell/2, sy*r.wid/2]) for sx in (-1, 1) for sy in (-1, 1)]
    tr = min(np.hypot(*(np.array([r.cx, r.cy]) + c @ R.T - x)) for c in corners)/env.v
    L = effort*env.v*(max(cycle_s, tr + cycle_s/2) - tr)
    n = max(1, int(L//(r.ell + np.pi*r_turn)))
    s = r.wid/n
    pts = parallel_pts(max(r.ell - s, 100.0), n, s, r_turn)
    best = None
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = pts*np.array([sx, sy])
            d = np.hypot(*(np.array([r.cx, r.cy]) + p[0] @ R.T - x))
            if best is None or d < best[0]:
                best = (d, p)
    return best[1], FramedFollower(env, best[1], np.array([r.cx, r.cy]), env.t, r.theta, np.zeros(2))


def simulate(env, r: Rect, cycle_s=600.0) -> float:
    """Probability found by flying the rectangle's lanes for one cycle from the current state (state kept)."""
    _, f = lanes(env, r, cycle_s)
    snap = env.snapshot()
    p0, t_end = env.pos_cum, env.t + cycle_s
    while env.t < t_end - 1e-6:
        _, _, term, _, _ = env.step([f.action()])
        if term:
            break
    gain = env.pos_cum - p0
    env.restore(snap)
    return gain


class AccordionPlanner:
    """policy(env) -> action: every cycle place one rectangle by accordion search and fly its lanes."""

    def __init__(self, variant="tavg_pca", cycle_s=600.0, cell=200.0, W=None, hook=None):
        self.variant, self.cycle_s, self.cell, self.W, self.hook = variant, cycle_s, cell, W, hook
        self.follower, self.t_next, self.log = None, -np.inf, []

    def __call__(self, env) -> float:
        if env.t >= self.t_next or self.follower is None:
            acc = Accordion(env, self.variant, self.cycle_s, self.cell, self.W)
            if self.hook is not None:
                self.hook(env, acc)
            r = acc.plan()
            if r is not None:
                _, self.follower = lanes(env, r, self.cycle_s)
                self.log.append(dict(t_min=round(env.t/60, 1), ell=r.ell, wid=r.wid,
                                     theta_deg=round(np.rad2deg(r.theta)), approx=round(r.approx, 4)))
            self.t_next = env.t + self.cycle_s
        if self.follower is None:
            return 0.0
        return self.follower.action()
