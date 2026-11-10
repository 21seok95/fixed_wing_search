"""Area score for planning and evaluation (agreed with the user 2026-11-05).

H_j is the x% region of the unweighted particles at t_j = 5, 10, ..., 60 min. A cell counts as swept when its centre
comes within the sensor half width of the path while it is inside the region of that 5 min window, and it scores only
once. c_j = swept cells inside H_j / cells of H_j, and the base score is A = mean of c_j over the windows. The total is
A + w POS. A cell swept early that stays in later regions adds to every later c_j, so early sweeps of the core are
worth more.

``AreaEnv`` adds the score to what ``_detect_lrc`` or ``_detect_oblique`` returns, so ``pos_cum`` holds the total and the
existing planner rollouts (``accordion.simulate``, ``sarops.refine``) optimise it unchanged. The team shares one ``AreaScore``.
With the oblique camera (real regions, 2026-11-09) a cell counts as swept when its centre is inside the camera image at a
1 s sample, so the image's real position (ahead of the UAV, gimbal, turns) decides what is swept.
``AreaSaropsPlanner`` also starts rectangles from peaks of the unswept region and sees the cells the other UAVs are
committed to. Rectangle sizes still come from the POS approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from rl_search import multi
from rl_search.accordion import Rect
from rl_search.detection import inside_quad
from rl_search.multi import TrackedEnv
from rl_search.patterns_v2 import Cluster
from rl_search.sarops import SaropsPlanner, refine

WINDOW_S = 300.0
N_WINDOWS = 12
CELL_M = 25.0
_CORNER_H, _CORNER_V = np.array([-1.0, -1.0, 1.0, 1.0])[:, None], np.array([-1.0, 1.0, 1.0, -1.0])[:, None]   # image corners, in order round the image


def region_masks(terr, h, track, dt, x):
    """Region of each window on the terrain grid: smoothed density as rl_env._zone, threshold set so that x% of the
    particles are inside. No-go cells are excluded."""
    tc = float(terr.axis[1] - terr.axis[0])
    nt = len(terr.axis)
    out = []
    for j in range(1, N_WINDOWS + 1):
        pts = np.asarray(track[min(int(round(j*WINDOW_S/dt)), len(track) - 1)], float)
        ij = np.clip(np.round((pts - terr.axis[0])/tc).astype(int), 0, nt - 1)
        hist = np.zeros((nt, nt))
        np.add.at(hist, (ij[:, 1], ij[:, 0]), 1.0)
        dens = gaussian_filter(hist, h/tc, mode="constant")
        dens[terr.forbidden] = 0
        thr = max(np.quantile(dens[ij[:, 1], ij[:, 0]], 1 - x/100), 1e-12)
        out.append((dens >= thr) & (dens > 0))
    return out


class AreaScore:
    """Swept cells and the score of one flight or team. ``B``: bit j set when the cell is in the region of window j."""

    def __init__(self, terr, h, track, dt, x, half_width, track_curve=False):
        self.R = float(half_width)
        masks = region_masks(terr, h, track, dt, x)
        union = np.any(masks, axis=0)
        rows, cols = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        tc = float(terr.axis[1] - terr.axis[0])
        s = int(round(tc/CELL_M))
        bits = np.zeros((r1 - r0, c1 - c0), np.uint16)
        for j, m in enumerate(masks):
            bits |= m[r0:r1, c0:c1].astype(np.uint16) << j
        self.B = np.repeat(np.repeat(bits, s, axis=0), s, axis=1)
        self.x0 = float(terr.axis[c0] - tc/2)
        self.y0 = float(terr.axis[r0] - tc/2)
        self.gx = self.x0 + (np.arange(self.B.shape[1]) + .5)*CELL_M
        self.gy = self.y0 + (np.arange(self.B.shape[0]) + .5)*CELL_M
        self.size = np.array([((self.B >> j) & 1).sum() for j in range(N_WINDOWS)], float)
        w = np.where(self.size > 0, 1.0/(N_WINDOWS*np.maximum(self.size, 1.0)), 0.0)
        codes = np.arange(1 << N_WINDOWS)
        self.T = np.zeros((N_WINDOWS, len(codes)))      # value of a cell by the window it is swept in and its bits
        for k in range(N_WINDOWS):
            later = sum(((codes >> j) & 1)*w[j] for j in range(k, N_WINDOWS))
            self.T[k] = later*((codes >> k) & 1)
        self.S = np.zeros(self.B.shape, bool)
        self.log = []
        self.committed = {}
        self.planning = False
        self.count = np.zeros(N_WINDOWS) if track_curve else None

    def window(self, t: float) -> int:
        """1-based window of time t; above N_WINDOWS the sweep no longer changes the score."""
        return max(int(np.ceil(t/WINDOW_S - 1e-9)), 1)

    def mark(self, xy, ts) -> float:
        """Sweep the chord from the first to the last point. Returns the gain in A."""
        k = self.window(float(ts[-1]))
        if k > N_WINDOWS:
            return 0.0
        a, b = np.asarray(xy[0], float), np.asarray(xy[-1], float)
        R = self.R
        ix0 = max(int(np.floor((min(a[0], b[0]) - R - self.x0)/CELL_M)), 0)
        ix1 = min(int(np.floor((max(a[0], b[0]) + R - self.x0)/CELL_M)) + 1, self.B.shape[1])
        iy0 = max(int(np.floor((min(a[1], b[1]) - R - self.y0)/CELL_M)), 0)
        iy1 = min(int(np.floor((max(a[1], b[1]) + R - self.y0)/CELL_M)) + 1, self.B.shape[0])
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0
        inside = ((self.B[iy0:iy1, ix0:ix1] >> (k - 1)) & 1).astype(bool)
        if not inside.any():
            return 0.0
        X, Y = self.gx[ix0:ix1][None, :], self.gy[iy0:iy1][:, None]
        ab = b - a
        den = float(ab @ ab)
        s = np.clip(((X - a[0])*ab[0] + (Y - a[1])*ab[1])/den, 0.0, 1.0) if den > 0 else 0.0
        hit = (X - a[0] - s*ab[0])**2 + (Y - a[1] - s*ab[1])**2 <= R*R
        new = hit & inside & ~self.S[iy0:iy1, ix0:ix1]
        if not new.any():
            return 0.0
        yy, xx = np.nonzero(new)
        return self._commit(yy + iy0, xx + ix0, k)

    def mark_frame(self, x, y, frame, t, cfg) -> float:
        """Oblique camera: sweep the cells whose centre is inside the image of this sample (flat ground ``cfg.alt_m`` below).
        ``frame`` = camera axes (f, r, u), each (1, 3). Returns the gain in A."""
        k = self.window(float(t))
        if k > N_WINDOWS:
            return 0.0
        f, r, u = frame
        d = f[0] + _CORNER_H*np.tan(np.radians(cfg.hfov_deg/2))*r[0] + _CORNER_V*np.tan(np.radians(cfg.vfov_deg/2))*u[0]
        s = cfg.alt_m/np.maximum(-d[:, 2], 1e-3)
        quad = np.stack([x + s*d[:, 0], y + s*d[:, 1]], 1)
        ix0 = max(int(np.floor((quad[:, 0].min() - self.x0)/CELL_M)), 0)
        ix1 = min(int(np.floor((quad[:, 0].max() - self.x0)/CELL_M)) + 1, self.B.shape[1])
        iy0 = max(int(np.floor((quad[:, 1].min() - self.y0)/CELL_M)), 0)
        iy1 = min(int(np.floor((quad[:, 1].max() - self.y0)/CELL_M)) + 1, self.B.shape[0])
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0
        new = ((self.B[iy0:iy1, ix0:ix1] >> (k - 1)) & 1).astype(bool) & ~self.S[iy0:iy1, ix0:ix1]
        if not new.any():
            return 0.0
        yy, xx = np.nonzero(new)
        yy, xx = yy + iy0, xx + ix0
        hit = inside_quad(self.gx[xx], self.gy[yy], quad[None])[0]
        return self._commit(yy[hit], xx[hit], k) if hit.any() else 0.0

    def _commit(self, yy, xx, k: int) -> float:
        """Mark cells swept in window k; returns the gain in A."""
        codes = self.B[yy, xx]
        self.S[yy, xx] = True
        self.log.append((yy, xx))
        if self.count is not None:
            for j in range(k - 1, N_WINDOWS):
                self.count[j] += ((codes >> j) & 1).sum()
        return float(self.T[k - 1][codes].sum())

    def rollback(self, cp: int) -> list:
        undone = self.log[cp:]
        for yy, xx in undone:
            self.S[yy, xx] = False
        del self.log[cp:]
        return undone

    def apply(self, cells) -> None:
        """Mark cells as swept without scoring (other UAVs' committed lanes while planning); undone by rollback."""
        for yy, xx in cells:
            new = ~self.S[yy, xx]
            if new.any():
                self.S[yy[new], xx[new]] = True
                self.log.append((yy[new], xx[new]))

    def curve(self) -> np.ndarray:
        return self.count/np.maximum(self.size, 1.0)


class AreaEnv(TrackedEnv):
    """Team env whose detection step also sweeps the shared area score: gain = pos_w POS + area_w A."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.area, self.uav_id, self.pos_w, self.area_w = None, 0, 1.0, 1.0

    def _detect_lrc(self, xy, ts, w=None, state=None):
        gain = super()._detect_lrc(xy, ts, w, state)
        if w is not None or self.area is None:
            return gain
        return self.pos_w*gain + self.area_w*self.area.mark(xy, ts)

    def _swept(self, x, y, frame, t):
        if self.area is not None:
            self._area_gain += self.area.mark_frame(x, y, frame, t, self.detector.cfg)

    def _detect_oblique(self, samples, ts):
        self._area_gain = 0.0
        gain = super()._detect_oblique(samples, ts)
        return gain if self.area is None else self.pos_w*gain + self.area_w*self._area_gain

    def snapshot(self):
        s = super().snapshot()
        if self.area is not None:
            s["area_cp"] = len(self.area.log)
        return s

    def restore(self, s):
        super().restore(s)
        if self.area is not None:
            undone = self.area.rollback(s["area_cp"])
            if not self.area.planning:     # multi.future_factor: the cells this UAV's committed lanes will sweep
                self.area.committed[self.uav_id] = undone


_approach_track = multi.approach_track


def approach_track(env, horizon_s=300.0):
    """multi.approach_track, except that a predicted approach is not a commitment: its swept cells stay out of
    ``AreaScore.committed``, as its detections stay out of the belief shared with the other UAVs."""
    a = getattr(env, "area", None)
    if a is None:
        return _approach_track(env, horizon_s)
    a.planning = True
    try:
        return _approach_track(env, horizon_s)
    finally:
        a.planning = False


multi.approach_track = approach_track


def area_peaks(env, t_mid: float, k: int, sep: float = 1500.0, block: int = 8, smooth: float = 250.0,
               radius: float = 2000.0) -> list[Cluster]:
    """Peaks of the unswept part of the region at t_mid, as start points for rectangles (main axis from the cells)."""
    a = env.area
    w = min(a.window(t_mid), N_WINDOWS)
    open_cells = ((a.B >> (w - 1)) & 1).astype(bool) & ~a.S
    if not open_cells.any():
        return []
    ny, nx = open_cells.shape[0]//block, open_cells.shape[1]//block
    if ny < 3 or nx < 3:
        return []
    coarse = open_cells[:ny*block, :nx*block].reshape(ny, block, nx, block).sum((1, 3)).astype(float)
    g = block*CELL_M
    dens = gaussian_filter(coarse, smooth/g)
    m = max(int(sep/g), 1)
    peaks = np.argwhere((dens == maximum_filter(dens, size=2*m + 1)) & (dens > 0))
    peaks = peaks[np.argsort(-dens[peaks[:, 0], peaks[:, 1]])][:k]
    yy, xx = np.nonzero(open_cells)
    px, py = a.gx[xx], a.gy[yy]
    out = []
    for i, j in peaks:
        pk = np.array([a.x0 + (j + .5)*g, a.y0 + (i + .5)*g])
        near = np.hypot(px - pk[0], py - pk[1]) < radius
        if near.sum() < 5:
            continue
        P = np.stack([px[near], py[near]], 1)
        ev, evec = np.linalg.eigh(np.cov((P - P.mean(0)).T) + np.eye(2)*1e-6)
        theta = float(np.arctan2(evec[1, 1], evec[0, 1]))
        out.append(Cluster(pk, theta, max(np.sqrt(ev[1]), 200.0), max(np.sqrt(ev[0]), 200.0), np.zeros(2),
                           float(near.sum())*CELL_M**2/1e6))
    return out


class AreaSaropsPlanner(SaropsPlanner):
    """SAROPS planner for the total score: extra rectangle starts on the unswept region, others' committed cells
    treated as swept while planning."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.hook = self._add_peaks

    def _add_peaks(self, env, acc):
        base = acc.peaks

        def peaks():
            own = base()
            k = min(acc.n_starts, 3)                                                            # 3 extra starts at most
            if acc.peak_timing == "arrival" and acc.search_s is not None:
                # as Accordion.peaks: the region read T/2 after each arrival offset, kept where that offset is the transit
                x, cand = env.plant.pose[:2], []
                for tau in np.arange(0.0, acc.max_transit_s + 1.0, 120.0):
                    cand += [c for c in area_peaks(env, env.t + tau + acc.T/2, k, acc.start_sep_m)
                             if abs(np.hypot(*(c.peak - x))/env.v - tau) <= 60.0]
                extra = []
                for c in sorted(cand, key=lambda c: -c.mass):
                    if all(np.hypot(*(c.peak - o.peak)) >= acc.start_sep_m for o in extra):
                        extra.append(c)
                    if len(extra) == k:
                        break
            else:      # "now", or "intercept" at the time the own peaks were read
                extra = area_peaks(env, acc.t_read if acc.t_read is not None else env.t + acc.T/2, k, acc.start_sep_m)
            return own + [c for c in extra if all(np.hypot(*(c.peak - o.peak)) >= acc.start_sep_m for o in own)]
        acc.peaks = peaks

    def _plan(self, env):
        a = env.area
        cp = len(a.log)
        for v, cells in a.committed.items():
            if v != env.uav_id:
                a.apply(cells)
        a.planning = True
        try:
            return super()._plan(env)
        finally:
            a.planning = False
            a.rollback(cp)


def rects_conflict(a, b, margin: float = 0.0) -> bool:
    """Separating axis test of two rectangles, ``b`` grown by ``margin`` on every side."""
    d = np.array([b.cx - a.cx, b.cy - a.cy])
    ca, sa, cb, sb = np.cos(a.theta), np.sin(a.theta), np.cos(b.theta), np.sin(b.theta)
    ua, va, ub, vb = np.array([ca, sa]), np.array([-sa, ca]), np.array([cb, sb]), np.array([-sb, cb])
    ax, ay, bx, by = a.ell/2, a.wid/2, b.ell/2 + margin, b.wid/2 + margin
    for n in (ua, va, ub, vb):
        ra = ax*abs(ua @ n) + ay*abs(va @ n)
        rb = bx*abs(ub @ n) + by*abs(vb @ n)
        if abs(d @ n) > ra + rb:
            return False
    return True


def inside_rect(q, r, margin: float = 0.0) -> np.ndarray:
    """Points q (..., 2) inside rectangle r grown by ``margin``."""
    c, s = np.cos(r.theta), np.sin(r.theta)
    dx, dy = q[..., 0] - r.cx, q[..., 1] - r.cy
    return (np.abs(dx*c + dy*s) <= r.ell/2 + margin) & (np.abs(-dx*s + dy*c) <= r.wid/2 + margin)


class NoOverlapMixin:
    """A new rectangle must keep ``d_safe`` from every other UAV's current rectangle (requested 2026-11-05). This comes on
    top of the planner and its separation check: start peaks ignore the other rectangles, the accordion may not grow
    into them, and a rotation that breaks the rule is searched again without rotating. A rectangle may be reused once
    its UAV has finished it. If no clear rectangle exists the normal choice is flown and ``overlap_forced`` counts it.
    ``strict_overlap`` (user 2026-11-09: rectangles flown at the same time must not overlap): the search is widened to more start
    points, and if none is clear no rectangle is taken: the UAV flies towards the remaining probability and tries again 30 s later."""
    team = None            # all the planners of the team
    overlap_forced = 0
    strict_overlap = False

    def _others(self):
        return [p.rects[-1][1] for p in (self.team or ()) if p is not self and p.rects and p.tasks and p.tasks[-1]["end"] is None]

    def choose(self, env, acc):
        others = self._others()
        if not others:
            return super().choose(env, acc)
        m = self.d_safe
        base_approx, base_peaks = acc.approx, acc.peaks

        def clear(q):
            return not any(rects_conflict(q, o, m) for o in others)

        def approx(G, i0, i1, j0, j1):
            mid = np.array([(G["ex"][i0] + G["ex"][i1])/2, (G["ey"][j0] + G["ey"][j1])/2]) @ G["R"].T
            q = Rect(float(mid[0]), float(mid[1]), float((i1 - i0)*acc.g), float((j1 - j0)*acc.gy), float(G["theta"]), 0.0)
            return base_approx(G, i0, i1, j0, j1) if clear(q) else -1.0

        def peaks():
            # particles inside the others' rectangles are left out at the time each start peak is read
            # (acc.T/2 ahead with peak_timing "now", as before; at each arrival offset with "arrival")
            b = env.belief
            acc.mask = lambda t: np.any([inside_rect(b.positions(t), o, m) for o in others], axis=0)
            try:
                found = base_peaks()
            finally:
                acc.mask = None
            return [c for c in found if not any(inside_rect(c.peak, o, m) for o in others)]

        acc.approx, acc.peaks = approx, peaks
        rects = acc.peak_rects() or []
        if not rects and self.strict_overlap:
            n0, sep0 = acc.n_starts, acc.start_sep_m
            acc.n_starts, acc.start_sep_m = max(n0, 8), min(sep0, 750.0)
            rects = (acc.peak_rects() or [])[:n0]
            acc.n_starts, acc.start_sep_m = n0, sep0
        acc.approx, acc.peaks = base_approx, base_peaks
        if not rects:
            self.overlap_forced += 1
            return None if self.strict_overlap else super().choose(env, acc)
        if self.traffic and self.hard_sep:
            rects = [q for q, _ in self.feasible_starts(env, rects)]
        args = (acc.gamma, self.cycle_s, self.time_s, self.task, self.traffic, self.d_safe)
        res = []
        for r in rects:
            out = refine(env, r, self.moves, *args)
            if not clear(out[0]):
                out = refine(env, r, tuple(k for k in self.moves if k != "rotate"), *args)
            res.append(out)
        if self.strict_overlap:
            res = [q for q in res if clear(q[0])]
            if not res:
                self.overlap_forced += 1
                return None
        picked = int(np.argmax([q[1] for q in res]))
        self.search_log.append(res[picked][2] | dict(evals=sum(q[2]["evals"] for q in res), sec=sum(q[2]["sec"] for q in res),
                                                     starts=len(res), picked=picked, gain_first=res[0][1]))
        return res[picked][0]


class NoOverlapSarops(NoOverlapMixin, SaropsPlanner):
    pass


class NoOverlapAreaSarops(NoOverlapMixin, AreaSaropsPlanner):
    pass


def display_grid(a: AreaScore, first, max_cells: int = 1000) -> dict:
    """Region bits and first-sweep minute (255 = never) of an ``AreaScore`` on a grid of at most ``max_cells`` per side."""
    f = 2
    while max(a.B.shape)/f > max_cells:
        f *= 2
    ny, nx = a.B.shape[0]//f, a.B.shape[1]//f
    bits = np.bitwise_or.reduce(a.B[:ny*f, :nx*f].reshape(ny, f, nx, f), axis=(1, 3))
    block = first[:ny*f, :nx*f].reshape(ny, f, nx, f).astype(np.int32)
    block[block == 0] = 255
    return dict(nx=nx, ny=ny, cell=CELL_M*f, x0=a.x0, y0=a.y0, bits=bits, first=block.min(axis=(1, 3)).astype(np.uint8))


def score_frames(terr, h, track, dt, x, logs, cfg) -> dict:
    """A of flown oblique-camera flights from their sample logs (``env.sample_log`` rows: t, x, y, yaw, roll, f, r, u, ...),
    swept in time order: A, c_j per window, region sizes and swept area in km^2, and ``grid`` for display."""
    a = AreaScore(terr, h, track, dt, x, 0.0, track_curve=True)
    log = np.vstack(logs)
    log = log[np.argsort(log[:, 0], kind="stable")]
    first = np.zeros(a.B.shape, np.uint8)
    for row in log:
        n0 = len(a.log)
        a.mark_frame(row[1], row[2], (row[5:8][None], row[8:11][None], row[11:14][None]), row[0], cfg)
        if len(a.log) > n0:
            first[a.log[-1]] = min(int(np.ceil(row[0]/60.0)), 254)
    c = a.curve()
    return dict(A=float(c.mean()), c=c.tolist(), area_km2=(a.size*CELL_M**2/1e6).tolist(), swept_km2=float(a.S.sum())*CELL_M**2/1e6,
                grid=display_grid(a, first))


def score_flight(terr, h, track, dt, x, half_width, xy, tt) -> dict:
    """Score flown paths on a particle set: A, c_j per window, and the share of flight time inside the current region
    per 15 min. Segments are swept in time order."""
    a = AreaScore(terr, h, track, dt, x, half_width, track_curve=True)
    seg = sorted((float(t[i + 1]), u, i) for u, t in enumerate(tt) for i in range(len(t) - 1))
    for _, u, i in seg:
        a.mark(xy[u][i:i + 2], tt[u][i:i + 2])
    c = a.curve()
    inside = {}
    for lo in (0, 15, 30, 45):
        n = hit = 0
        for p, t in zip(xy, tt):
            for k in np.flatnonzero((t > lo*60) & (t <= (lo + 15)*60)):
                n += 1
                j = min(a.window(float(t[k])), N_WINDOWS) - 1
                ix, iy = int((p[k, 0] - a.x0)//CELL_M), int((p[k, 1] - a.y0)//CELL_M)
                if 0 <= ix < a.B.shape[1] and 0 <= iy < a.B.shape[0]:
                    hit += int((a.B[iy, ix] >> j) & 1)
        inside[str(lo)] = hit/n if n else 0.0
    return dict(A=float(c.mean()), c=c.tolist(), inside=inside, area_km2=(a.size*CELL_M**2/1e6).tolist())
