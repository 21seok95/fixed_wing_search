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
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter

from rl_search.patterns import _rot, parallel_pts
from rl_search.patterns_v2 import FramedFollower, find_clusters

VARIANTS = ("snap0", "mid", "tavg", "tavg_pca")
TASK_EFFORT = 0.92    # task mode: polyline budget / (v * search_s); flown length is 3-10 % over the polyline
TASK_FLOWN = 1.06     # task mode: flown length / polyline length (measured 1.03-1.11, pursuit around the corners)
TASK_FILL = 0.8       # task mode: the lanes must use at least this share of the budget (SAROPS: the pattern fills the
                      # available path length), else the rectangle is too small for the search time


@dataclass
class Rect:
    cx: float
    cy: float
    ell: float
    wid: float
    theta: float
    approx: float = 0.0
    corner: tuple | None = None    # start corner (sx, sy) of the lanes; None: the corner nearest the UAV
    spacing: float | None = None   # fixed track spacing (lane_mode "fixed"/"cover"): lanes = round(wid / spacing)
    k_max: int = 1                 # lane_mode "cover": most passes over each point the local search may choose


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
                 ell_min=None, gamma=None, dt_snap=60.0, start_smooth_m=0.0, n_starts=1, start_sep_m=1500.0,
                 pod_model="exp", search_s=None, max_transit_s=300.0, peak_timing="now", lane_mode="fill"):
        self.env, self.variant, self.g = env, variant, cell
        # task mode (search_s set, 2026-11-04 I1/I3): every rectangle gets search_s of searching after its own
        # transit, rectangles farther than max_transit_s are not admissible, the score is POS per minute of
        # (transit + search); otherwise the fixed cycle: transit and search share cycle_s
        self.search_s, self.max_transit_s = search_s, max_transit_s
        if peak_timing not in ("now", "arrival"):
            raise ValueError(peak_timing)
        self.peak_timing = peak_timing   # "arrival": start peaks from the distribution at each peak's own arrival
        self.T = search_s if search_s is not None else cycle_s
        if pod_model not in ("exp", "lanes"):
            raise ValueError(pod_model)
        self.pod_model = pod_model
        self.cover_w = 2*env.lrc_R if getattr(env, "lrc", False) else env.fp_cross   # width one lane covers
        # lane_mode "fixed" (task mode, 2026-11-04): lanes do not overlap, spacing = cover_w, the width grows by whole
        # lanes; the lane time follows from the geometry and must fit search_s; POD = single-pass POD everywhere
        # lane_mode "cover": the same with spacing cover_w / k, k = 1, 2, 3 passes over every point, k chosen per
        # rectangle by the approximation (lanes closer than 2 R are skipped, time from the polyline)
        if lane_mode not in ("fill", "fixed", "cover"):
            raise ValueError(lane_mode)
        self.lane_mode = lane_mode
        self.ks = {"fill": (), "fixed": (1,), "cover": (1, 2, 3)}[lane_mode]
        self.lanes_fixed = lane_mode in ("fixed", "cover") and search_s is not None
        self.gy = self.cover_w if self.lanes_fixed else cell   # grid step across lanes: one cover width
        self._k = {}
        self.W = W if W is not None else sweep_width(env)
        self.effort = effort
        self.r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
        self.ell_min = ell_min if ell_min is not None else max(600.0, 4*self.r_turn)
        self.gamma = gamma if gamma is not None else 0.25*env.fp_cross
        self.start_smooth_m, self.n_starts, self.start_sep_m = start_smooth_m, n_starts, start_sep_m
        horizon = 2*cycle_s if search_s is None else max_transit_s + search_s
        self.taus = np.arange(0, horizon + dt_snap/2, dt_snap)
        self.k_mid = int(np.argmin(np.abs(self.taus - self.T/2)))

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
        ex, ey = np.arange(lo[0], hi[0] + g, g), np.arange(lo[1], hi[1] + self.gy, self.gy)
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
        ell, wid = (i1 - i0)*self.g, (j1 - j0)*self.gy
        if self.search_s is not None:
            return self._approx_task(G, i0, i1, j0, j1, ell, wid)
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
        return float(poc*self.pod(wid/n))

    def _approx_task(self, G, i0, i1, j0, j1, ell, wid):
        """Task mode: approximate POS per minute = POC averaged over [transit, transit + lane time] x POD, divided
        by (transit + lane time) in minutes. Lanes: the most that fit search_s (actual polyline), and they must fill
        at least TASK_FILL of it (otherwise small rectangles finishing in 2 min win on POS per minute, measured
        2026-11-04); lane time from the polyline. -1 if farther than max_transit_s, not admissible, or spacing over
        twice a lane's cover. The detections during the transit are not in the approximation (the simulation has
        them)."""
        tr = self._transit(G, i0, i1, j0, j1)
        if tr > self.max_transit_s:
            return -1.0
        if ell < self.ell_min:
            return -1.0
        if self.lanes_fixed:
            units = j1 - j0
            p = self._poc_t(G, i0, i1, j0, j1)
            q = 1 - self.env.search.pod
            best, best_k = -1.0, 1
            for kk in self.ks:
                t_lanes = fixed_lane_time(ell, kk*units, self.cover_w/kk, self.r_turn, self.env.v)
                if t_lanes > self.search_s:
                    continue
                m = (self.taus >= tr - 1e-6) & (self.taus <= tr + t_lanes + 1e-6)
                poc = p[m].mean() if m.any() else p[np.searchsorted(self.taus, tr) - 1]
                # every task is charged transit + search_s, so a rectangle grows until its lanes fill the search time
                v = float(poc*(1 - q**kk)/((tr + self.search_s)/60.0))
                if v > best:
                    best, best_k = v, kk
            self._k[(id(G), i0, i1, j0, j1)] = best_k
            return best
        n, length = fit_lanes(float(ell), float(wid), TASK_EFFORT*self.env.v*self.search_s, float(self.r_turn), float(self.gamma))
        budget = TASK_EFFORT*self.env.v*self.search_s
        if n < 1 or wid/n > 2*self.cover_w or length < TASK_FILL*budget:
            return -1.0                          # wider gaps than a lane's cover, or too small to fill the search time
        t_lanes = TASK_FLOWN*length/self.env.v   # small rectangles finish early and are charged only their own time
        p = self._poc_t(G, i0, i1, j0, j1)
        k = (self.taus >= tr - 1e-6) & (self.taus <= tr + t_lanes + 1e-6)
        poc = p[k].mean() if k.any() else p[np.searchsorted(self.taus, tr) - 1]
        return float(poc*self.pod(wid/n)/((tr + t_lanes)/60.0))

    def pod(self, s: float) -> float:
        """Detection probability inside the rectangle for track spacing ``s``.
        exp:   random-search formula 1 - exp(-W/s) (Koopman; SAROPS initial placement)
        lanes: exact parallel lanes with a definite-range sensor: a point is covered floor(m) or ceil(m) times,
               m = covered width / s, each pass detecting with the single-pass POD"""
        if self.pod_model == "exp":
            return float(1 - np.exp(-self.W/s))
        q = 1 - self.env.search.pod
        m = self.cover_w/s
        k = np.floor(m)
        f = m - k
        return float((1 - f)*(1 - q**k) + f*(1 - q**(k + 1)))

    # ------------------------------------------------------------------ search
    def theta(self):
        if self.variant != "tavg_pca":
            return 0.0
        cl = find_clusters(self.env, self.env.t + self.T/2, k=1)
        return cl[0].theta if cl else 0.0

    def peaks(self):
        """Start peaks for the accordion. "now": peaks of the remaining distribution at t + T/2. "arrival" (task mode):
        peaks at t + tau + T/2 for transit offsets tau = 0, 2, 4, ... min up to max_transit_s, each kept only where
        its own straight-line transit is within 1 min of tau (so a far peak is taken where the targets will be when
        the UAV gets there), then the heaviest n_starts at least start_sep_m apart."""
        env, kw = self.env, dict(smooth=max(self.start_smooth_m, 250.0), sep=self.start_sep_m)
        if self.peak_timing == "now" or self.search_s is None:
            return find_clusters(env, env.t + self.T/2, k=self.n_starts, **kw)
        x = env.plant.pose[:2]
        cand = []
        for tau in np.arange(0.0, self.max_transit_s + 1.0, 120.0):
            for cl in find_clusters(env, env.t + tau + self.T/2, k=self.n_starts, **kw):
                if abs(np.hypot(*(cl.peak - x))/env.v - tau) <= 60.0:
                    cand.append(cl)
        if not cand:
            return find_clusters(env, env.t + self.T/2, k=self.n_starts, **kw)
        out = []
        for cl in sorted(cand, key=lambda c: -c.mass):
            if all(np.hypot(*(cl.peak - o.peak)) >= self.start_sep_m for o in out):
                out.append(cl)
            if len(out) == self.n_starts:
                break
        return out

    def peak_rects(self) -> list[Rect] | None:
        """One accordion search per density peak (``peaks``, each in a grid along its own main axis), best
        approximate POS first; None if there is no peak."""
        cls = self.peaks()
        if not cls:
            return None
        out = []
        for cl in cls:
            th = cl.theta if self.variant == "tavg_pca" else 0.0
            G = self._grid(th)
            q = cl.peak @ G["R"]
            ci = int(np.clip((q[0] - G["ex"][0])//self.g, 0, len(G["ex"]) - 2))
            cj = int(np.clip((q[1] - G["ey"][0])//self.gy, 0, len(G["ey"]) - 2))
            r = self._climb(G, ci, cj)
            if r is not None:
                out.append(r)
        return sorted(out, key=lambda r: -r.approx)

    def plan(self) -> Rect | None:
        """Accordion search. ``n_starts`` > 1: the peak with the best approximate POS (transit is in the
        approximation). ``start_smooth_m``: pick the start cell on a Gaussian-smoothed density instead of the
        single densest cell."""
        if self.n_starts > 1:
            rects = self.peak_rects()
            if rects is not None:
                return rects[0] if rects else None
        G = self._grid(self.theta())
        H = G["H"]
        dens = H[0] if self.variant == "snap0" else H[self.k_mid] if self.variant == "mid" else H[self.taus <= self.T].mean(0)
        if dens.max() <= 0:
            return None
        if self.start_smooth_m > 0:
            dens = gaussian_filter(dens, self.start_smooth_m/self.g)
        ci, cj = np.unravel_index(np.argmax(dens), dens.shape)
        return self._climb(G, ci, cj)

    def _climb(self, G, ci, cj) -> Rect | None:
        nx, ny = G["H"].shape[1:]
        # smallest admissible start around the start cell: ell_min long, as wide as the spacing limit needs
        L0 = self.effort*self.env.v*self.T
        a = int(np.ceil(self.ell_min/self.g))
        bw = 1 if self.lanes_fixed else \
            max(1, int(np.ceil(self.gamma*L0/(self.ell_min + np.pi*self.r_turn)/self.g)))
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
        if self.lanes_fixed:
            k = self._k.get((id(G), *idx), 1)
            return Rect(float(cx), float(cy), float((i1 - i0)*self.g), float((j1 - j0)*self.gy), float(G["theta"]), val,
                        spacing=self.cover_w/k, k_max=max(self.ks))
        return Rect(float(cx), float(cy), float((i1 - i0)*self.g), float((j1 - j0)*self.gy), float(G["theta"]), val)

    def approx_rect(self, G, r: Rect) -> float:
        """Approximate POS of an arbitrary rectangle (same theta as G) by snapping it to the grid."""
        c = np.array([r.cx, r.cy]) @ G["R"]
        i0 = int(round((c[0] - r.ell/2 - G["ex"][0])/self.g))
        i1 = i0 + max(1, int(round(r.ell/self.g)))
        j0 = int(round((c[1] - r.wid/2 - G["ey"][0])/self.gy))
        j1 = j0 + max(1, int(round(r.wid/self.gy)))
        nx, ny = G["H"].shape[1:]
        if not (0 <= i0 < i1 <= nx and 0 <= j0 < j1 <= ny):
            return 0.0
        return max(self.approx(G, i0, i1, j0, j1), 0.0)


@lru_cache(maxsize=65536)
def fit_lanes(ell: float, wid: float, budget_m: float, r_turn: float, gamma: float) -> tuple[int, float]:
    """Task mode: most lanes whose actual polyline (lanes + lane changes, ``parallel_pts``) fits ``budget_m``,
    with track spacing >= gamma. Returns (lanes, polyline length); (0, 0) if not even one lane fits."""
    n_max = max(1, int(wid//max(gamma, 1e-6)))
    for n in range(n_max, 0, -1):
        s = wid/n
        pts = parallel_pts(max(ell - s, 100.0), n, s, r_turn)
        length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        if length <= budget_m:
            return n, length
    return 0, 0.0


def fixed_lane_time(ell: float, n: int, s: float, r_turn: float, v: float) -> float:
    """Flight time of n non-overlapping lanes of length ell - s (rectangle convention ell = lane + s) at spacing s:
    lane changes are half turns of diameter max(s, 2 R), plus the straight rest when s > 2 R; with s < 2 R the
    lanes are skipped (``parallel_pts``) and the polyline is used with the measured 6 % pursuit excess."""
    if s >= 0.95*2*r_turn:                       # within 5 %: the pursuit makes the half turn
        change = np.pi*min(r_turn, s/2) + max(s - 2*r_turn, 0.0)
        return (n*(ell - s) + (n - 1)*change)/v
    pts = parallel_pts(max(ell - s, 100.0), n, s, r_turn)
    return TASK_FLOWN*float(np.hypot(*np.diff(pts, axis=0).T).sum())/v


def transit_time(env, r: Rect) -> float:
    """Straight-line time from the UAV to the rectangle's nearest corner."""
    R = _rot(r.theta)
    corners = [np.array([sx*r.ell/2, sy*r.wid/2]) for sx in (-1, 1) for sy in (-1, 1)]
    return min(np.hypot(*(np.array([r.cx, r.cy]) + c @ R.T - env.plant.pose[:2])) for c in corners)/env.v


def start_transit(env, r: Rect, cycle_s=600.0, search_s=None) -> float:
    """Straight-line time from the UAV to the lanes' actual start point (``r.corner`` or the nearest corner)."""
    pts, _ = lane_plan(env, r, cycle_s, search_s=search_s)
    p0 = np.array([r.cx, r.cy]) + pts[0] @ _rot(r.theta).T
    return float(np.hypot(*(p0 - env.plant.pose[:2]))/env.v)


def lane_count(env, r: Rect, cycle_s=600.0, effort=0.85, search_s=None) -> int:
    """Lanes that fit the search time: ``search_s`` (task mode), or what is left of the cycle after the transit
    to the rectangle's nearest corner."""
    r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
    if r.spacing is not None:
        return max(1, int(round(r.wid/r.spacing)))
    if search_s is not None:
        return max(1, fit_lanes(float(r.ell), float(r.wid), TASK_EFFORT*env.v*search_s, float(r_turn), 0.25*env.fp_cross)[0])
    else:
        tr = transit_time(env, r)
        L = effort*env.v*(max(cycle_s, tr + cycle_s/2) - tr)
    return max(1, int(L//(r.ell + np.pi*r_turn)))


def lane_plan(env, r: Rect, cycle_s=600.0, effort=0.85, search_s=None):
    """Polyline of the parallel lanes in the rectangle frame (first leg along theta) and its start corner
    (sx, sy): ``r.corner``, or the corner nearest the UAV."""
    r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
    x = env.plant.pose[:2]
    R = _rot(r.theta)
    n = lane_count(env, r, cycle_s, effort, search_s)
    s = r.wid/n
    pts = parallel_pts(max(r.ell - s, 100.0), n, s, r_turn)
    if r.corner is not None:
        return pts*np.array(r.corner), r.corner
    best = None
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = pts*np.array([sx, sy])
            d = np.hypot(*(np.array([r.cx, r.cy]) + p[0] @ R.T - x))
            if best is None or d < best[0]:
                best = (d, p, (sx, sy))
    return best[1], best[2]


def lanes(env, r: Rect, cycle_s=600.0, effort=0.85, search_s=None):
    """Parallel lanes filling the rectangle, started at ``r.corner`` or the corner nearest the UAV.
    Returns (polyline in the rectangle frame, follower)."""
    pts, _ = lane_plan(env, r, cycle_s, effort, search_s)
    return pts, FramedFollower(env, pts, np.array([r.cx, r.cy]), env.t, r.theta, np.zeros(2))


def conflict_seconds(times, xy, traffic, d_safe) -> float:
    """Seconds (5 s steps) at which the path (times, xy) is closer than ``d_safe`` to any traffic track (t, xy) at the
    same time; a track counts only inside its own time span."""
    if not traffic:
        return 0.0
    times, xy = np.asarray(times), np.asarray(xy)
    bad = np.zeros(len(times), bool)
    for tt, txy in traffic:
        m = (times >= tt[0]) & (times <= tt[-1])
        if not m.any():
            continue
        ox, oy = np.interp(times[m], tt, txy[:, 0]), np.interp(times[m], tt, txy[:, 1])
        bad[m] |= np.hypot(xy[m, 0] - ox, xy[m, 1] - oy) < d_safe
    return float(bad.sum()*5.0)


def deconflicted(score: float, conflict_s: float) -> float:
    """Planning-time separation: any conflict ranks a candidate below every conflict-free one; among conflicting
    candidates, less conflict time first, then the score."""
    return score if conflict_s <= 0 else score - 1.0 - 1e-3*conflict_s


def simulate(env, r: Rect, cycle_s=600.0, task: dict | None = None, traffic=None, d_safe: float = 300.0) -> float:
    """Probability found by flying the rectangle's lanes for one cycle from the current state (state kept).
    ``task`` (dict search_s, max_transit_s, extra_s): fly until the lanes are done once (or transit + search_s +
    extra_s), detections on the way included, and return POS per minute of flight; 0 if the rectangle is
    farther than max_transit_s."""
    if task is not None:
        tr = start_transit(env, r, cycle_s, task["search_s"])
        if tr > task["max_transit_s"]:
            return 0.0
        _, f = lanes(env, r, cycle_s, search_s=task["search_s"])
        t_cap = env.t + tr + task["search_s"] + task["extra_s"]
    else:
        _, f = lanes(env, r, cycle_s)
        t_cap = env.t + cycle_s
    snap = env.snapshot()
    skip, env.skip_cells = env.skip_cells, True      # the cell grid only feeds the area term and the kernel sensor
    p0, t0 = env.pos_cum, env.t
    times, xy = [], []
    while env.t < t_cap - 1e-6:
        _, _, term, _, _ = env.step([f.action()])
        if traffic:
            times.append(env.t)
            xy.append(env.plant.pose[:2].copy())
        if term or (task is not None and f.laps >= 1):
            break
    gain, dt = env.pos_cum - p0, env.t - t0
    env.skip_cells = skip
    env.restore(snap)
    score = gain/(max(dt, tr + task["search_s"])/60.0) if task is not None else gain   # early end charged search_s
    if traffic:
        score = deconflicted(score, conflict_seconds(times, xy, traffic, d_safe))
    return score


class AccordionPlanner:
    """policy(env) -> action: place one rectangle by accordion search and fly its lanes.

    Fixed cycle (default): re-plan every ``cycle_s``. Task mode (``search_s`` set, 2026-11-04 I1-I4): each rectangle
    is sized for ``search_s`` of searching after its transit, candidates farther than ``max_transit_s`` are dropped,
    candidates compete on POS per minute, and the next plan comes when the lanes are done (or after transit +
    search_s + ``extra_s``). With no candidate within ``max_transit_s`` the UAV flies toward the remaining-mass
    centroid at the intercept time (``lead_action``) and tries again every ``approach_check_s``."""

    def __init__(self, variant="tavg_pca", cycle_s=600.0, cell=200.0, W=None, hook=None, search_s=None,
                 max_transit_s=300.0, extra_s=120.0, approach_check_s=30.0, **acc_kw):
        self.variant, self.cycle_s, self.cell, self.W, self.hook, self.acc_kw = variant, cycle_s, cell, W, hook, acc_kw
        self.search_s, self.max_transit_s, self.extra_s, self.approach_check_s = search_s, max_transit_s, extra_s, approach_check_s
        self.follower, self.t_next, self.log = None, -np.inf, []
        self.rects = []      # (plan time, Rect, lanes in world coordinates) for plotting
        self.tasks = []      # task mode: dict per rectangle (plan time, transit estimate, end time, how it ended)
        self.traffic = None  # multi-UAV: other UAVs' committed tracks [(t, xy)] while planning (set by multi.run_team)
        self.d_safe = 300.0
        self.approach = []   # task mode: (start, end) of approach phases

    @property
    def task(self) -> dict | None:
        if self.search_s is None:
            return None
        return dict(search_s=self.search_s, max_transit_s=self.max_transit_s, extra_s=self.extra_s)

    def choose(self, env, acc) -> Rect | None:
        if not self.traffic:
            return acc.plan()
        # planning-time separation: fly each peak's rectangle and keep the best conflict-free one
        rects = acc.peak_rects() or []
        if not rects:
            return acc.plan()
        vals = [simulate(env, r, self.cycle_s, self.task, self.traffic, self.d_safe) for r in rects]
        return rects[int(np.argmax(vals))]

    def _plan(self, env) -> bool:
        acc = Accordion(env, self.variant, self.cycle_s, self.cell, self.W, search_s=self.search_s,
                        max_transit_s=self.max_transit_s, **self.acc_kw)
        if self.hook is not None:
            self.hook(env, acc)
        r = self.choose(env, acc)
        if r is None:
            return False
        pts, self.follower = lanes(env, r, self.cycle_s, search_s=self.search_s)
        self.log.append(dict(t_min=round(env.t/60, 1), ell=r.ell, wid=r.wid,
                             theta_deg=round(np.rad2deg(r.theta)), approx=round(r.approx, 4)))
        self.rects.append((env.t, r, np.array([r.cx, r.cy]) + pts @ _rot(r.theta).T))
        if self.search_s is not None:
            tr = start_transit(env, r, self.cycle_s, self.search_s)
            self.tasks.append(dict(t0=env.t, transit=tr, t_cap=env.t + tr + self.search_s + self.extra_s, end=None, how=None,
                                   lane_t=None))
        return True

    def __call__(self, env) -> float:
        if self.search_s is None:
            if env.t >= self.t_next or self.follower is None:
                self._plan(env)
                self.t_next = env.t + self.cycle_s
            return 0.0 if self.follower is None else self.follower.action()
        # task mode
        cur = self.tasks[-1] if self.tasks and self.tasks[-1]["end"] is None else None
        if cur is not None and cur["lane_t"] is None and self.follower.started:
            cur["lane_t"] = env.t
        if cur is not None and (self.follower.laps >= 1 or env.t >= cur["t_cap"]):
            cur["end"], cur["how"] = env.t, ("done" if self.follower.laps >= 1 else "cap")
            self.follower, cur = None, None
        if cur is None and env.t >= self.t_next:
            if self._plan(env):
                if self.approach and self.approach[-1][1] is None:
                    self.approach[-1] = (self.approach[-1][0], env.t)
            else:
                self.follower = None
                if not self.approach or self.approach[-1][1] is not None:
                    self.approach.append((env.t, None))
                self.t_next = env.t + self.approach_check_s
        if self.follower is None:
            from rl_search.patterns import lead_action
            return lead_action(env)
        return self.follower.action()
