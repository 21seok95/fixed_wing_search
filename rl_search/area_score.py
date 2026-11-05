"""Area score for planning and evaluation (agreed with the user 2026-11-05).

H_j is the x% region of the unweighted particles at t_j = 5, 10, ..., 60 min. A cell counts as swept when its centre
comes within the sensor half width of the path while it is inside the region of that 5 min window, and it scores only
once. c_j = swept cells inside H_j / cells of H_j, and the base score is A = mean of c_j over the windows. The total is
A + w POS. A cell swept early that stays in later regions adds to every later c_j, so early sweeps of the core are
worth more.

``AreaEnv`` adds the score to what ``_detect_lrc`` returns, so ``pos_cum`` holds the total and the existing planner
rollouts (``accordion.simulate``, ``sarops.refine``) optimise it unchanged. The team shares one ``AreaScore``.
``AreaSaropsPlanner`` also starts rectangles from peaks of the unswept region and sees the cells the other UAVs are
committed to. Rectangle sizes still come from the POS approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from rl_search.multi import TrackedEnv
from rl_search.patterns_v2 import Cluster
from rl_search.sarops import SaropsPlanner

WINDOW_S = 300.0
N_WINDOWS = 12
CELL_M = 25.0


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
        yy, xx = yy + iy0, xx + ix0
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
            extra = area_peaks(env, env.t + acc.T/2, acc.n_starts, acc.start_sep_m)
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
