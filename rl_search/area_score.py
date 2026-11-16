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
_ROAD: dict = {}


def road_mask(region: str = "amboy", cell_m: float = CELL_M) -> np.ndarray:
    """Terrain cells on a drivable road (bool, [row = y, col = x], the road network's ``road_dist`` <= half a cell). Built once from the
    road network and cached in rl_search/data/real/<region>/road_mask_<cell>m.npz (git-ignored; rebuilt from the network when missing)."""
    key = (region, float(cell_m))
    if key not in _ROAD:
        from pathlib import Path
        p = Path("rl_search/data/real")/region/f"road_mask_{cell_m:g}m.npz"
        if p.exists():
            d = np.load(p)
            shape = tuple(int(v) for v in d["shape"])
            m = np.unpackbits(d["bits"])[:shape[0]*shape[1]].reshape(shape).astype(bool)
        else:
            from rl_search.vehicle import load_net
            m = load_net(region, cell_m).road_dist <= cell_m/2
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, bits=np.packbits(m), shape=np.array(m.shape))
        _ROAD[key] = m
    return _ROAD[key]
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


SEEN_CELL_M = 25.0          # region cell (the smallest map's pixel)
SEEN_RANK_CELL_M = 50.0     # density grid for ranking the cells (nested: two region cells per side)
SEEN_RANK_SIGMA_M = 150.0   # density kernel, as the window A's region (search.bandwidth_m)
SEEN_T_END_S = 3600.0       # frames up to 60 min after the LKP, the window A's span


class SeenArea:
    """A from seen particles (agreed with the user on 2026-11-11, replacing the 5 min window A for RL and SAROPS).

    Every ``dt`` (the particle save step, 10 s) up to ``t_end``, the region H(t) is the 25 m cells holding the densest x% of the active
    particles. Density: particles per 50 m cell, smoothed with sigma 150 m. Only cells with particles in them count, so empty ground beside
    a road is never region. A particle is seen once its cumulative POD since the hand-over reaches ``pod`` (survival <= 1 - pod), and the
    flag moves with the particle. Coverage c(t) = mean over the region cells of the share of the cell's particles already seen (each cell
    counts once: area, not particle count). A = mean of c(t) over the frames. Seeing particle i at time tau adds F_i(tau) = its share of
    c over the frames t >= tau, booked when it happens (as the window A books a cell's later windows when the cell is swept).
    The planner interface (``log``, ``rollback``, ``apply``, ``committed``, ``planning``) matches ``AreaScore``."""

    def __init__(self, track, exit_step, dt, x: float = 80.0, pod: float = 0.5, t_end: float = SEEN_T_END_S):
        track = np.asarray(track)
        T, N = track.shape[:2]
        self.dt, self.x, self.pod, self.n = float(dt), float(x), float(pod), N
        J = self.J = max(min(int(round(t_end/self.dt)), T - 1), 1)
        self.t = np.arange(1, J + 1)*self.dt
        ex = np.asarray(exit_step).astype(np.int64)
        a = np.zeros((J, N), np.float32)                  # share of particle i in c_j, divided by J
        self.size = np.zeros(J)                            # region cells per frame
        g, h = SEEN_RANK_CELL_M, SEEN_CELL_M
        for j in range(1, J + 1):
            idx = np.flatnonzero((ex < 0) | (ex > j))      # active at t_j (ParticleBelief.active)
            if len(idx) == 0:
                continue
            P = track[j, idx].astype(float)
            lo = np.floor(P.min(0)/g)*g - 4*SEEN_RANK_SIGMA_M
            ig = np.floor((P - lo)/g).astype(np.int64)
            ny, nx = ig[:, 1].max() + 5*int(SEEN_RANK_SIGMA_M/g) + 1, ig[:, 0].max() + 5*int(SEEN_RANK_SIGMA_M/g) + 1
            hist = np.bincount(ig[:, 1]*nx + ig[:, 0], minlength=ny*nx).reshape(ny, nx).astype(float)
            dens = gaussian_filter(hist, SEEN_RANK_SIGMA_M/g, mode="constant")
            d = dens[ig[:, 1], ig[:, 0]]
            core = d >= max(np.quantile(d, 1 - self.x/100), 1e-12)
            ic = np.floor((P[core] - lo)/h).astype(np.int64)
            _, inv, cnt = np.unique(ic[:, 1]*(2*nx + 2) + ic[:, 0], return_inverse=True, return_counts=True)
            self.size[j - 1] = len(cnt)
            a[j - 1, idx[core]] = 1.0/(J*len(cnt)*cnt[inv.ravel()])
        self.a = a
        F = np.zeros((J + 2, N), np.float32)                # F[k] = sum of a over frames k..J (k = 1..J); F[0] = F[1]
        F[1:J + 1] = np.cumsum(a[::-1], axis=0, dtype=np.float64)[::-1]
        F[0] = F[1]
        self.F = F
        self.seen = np.zeros(N, bool)
        self.t_seen = np.full(N, np.inf)
        self.log, self.gains, self.committed, self.planning = [], [], {}, False
        self.team = None            # the team's envs: seen from the team survival (product of each UAV's own), else the env's own

    def clear(self) -> None:
        self.seen[:] = False
        self.t_seen[:] = np.inf
        self.log.clear()
        self.gains.clear()
        self.committed.clear()

    def frame_of(self, t: float) -> int:
        """First frame k (1..J, J + 1 past the end) with t_k >= t."""
        return int(np.clip(np.ceil(t/self.dt - 1e-9), 1, self.J + 1))

    def frame_now(self, t: float) -> int:
        """Frame (1..J) nearest to t, for the observation."""
        return int(np.clip(np.rint(t/self.dt), 1, self.J))

    def update(self, surv: np.ndarray, t: float) -> float:
        """Particles whose survival since the hand-over fell to 1 - pod become seen at t; returns the gain in A."""
        new = ~self.seen & (surv <= 1.0 - self.pod + 1e-12)
        if not new.any():
            return 0.0
        idx = np.flatnonzero(new)
        self.seen[idx] = True
        self.t_seen[idx] = t
        self.log.append(idx)
        gain = float(self.F[self.frame_of(t), idx].sum(dtype=np.float64))
        if not self.planning:
            self.gains.append((float(t), gain, len(idx)))
        return gain

    def rollback(self, cp: int) -> list:
        undone = self.log[cp:]
        for idx in undone:
            self.seen[idx] = False
            self.t_seen[idx] = np.inf
        del self.log[cp:]
        return undone

    def apply(self, items) -> None:
        """Mark particles as seen without scoring (other UAVs' committed sweeps while planning); undone by rollback."""
        for idx in items:
            new = idx[~self.seen[idx]]
            if len(new):
                self.seen[new] = True
                self.t_seen[new] = -1.0
                self.log.append(new)

    def value_now(self, t: float) -> np.ndarray:
        """A gain of seeing each particle at t (0 for seen ones)."""
        return np.where(self.seen, 0.0, self.F[self.frame_of(t)])

    def cover_now(self, t: float) -> float:
        """c at the frame nearest to t: share of the region already seen."""
        j = self.frame_now(t)
        return float(self.a[j - 1][self.seen].sum(dtype=np.float64)*self.J)

    def curve(self) -> np.ndarray:
        """c_j per frame from the seen times."""
        out = np.zeros(self.J)
        for j0 in range(0, self.J, 60):
            sl = slice(j0, min(j0 + 60, self.J))
            hit = self.t_seen[None, :] <= self.t[sl, None]
            out[sl] = (self.a[sl]*hit).sum(1, dtype=np.float64)*self.J
        return out

    @property
    def A(self) -> float:
        return float(self.curve().mean())


RT_DENS_CELL_M = 50.0       # density grid of the real-time region (two 25 m cells per side)


class RealtimeArea:
    """The window A with its region in real time (user, 2026-11-11): the 5 min windows become 10 s frames.

    Frame j (t_j = j dt up to ``t_end`` = the end of the track, dt = the particle save step, 10 s): H_j is the x% region of the active
    unweighted particles at t_j,
    as ``region_masks`` (density smoothed with sigma ``h`` = 150 m, threshold at the particles so that x% are inside, no-go cells out), on the
    25 m cells (density on a 50 m grid around the particles, each 50 m cell giving its four 25 m cells). A 25 m cell is swept when its centre
    is inside the camera image at a 1 s sample while it is in the region of that moment, and it scores once. c_j = swept cells in H_j / cells
    of H_j; A = mean of c_j over the scoring window (``set_window``: from 0 to 60 min after the LKP by default, the window A's span; for a
    late launch, from the launch to the end of the flight). Sweeping a cell at frame j adds sum over the window frames m >= j that hold it
    of 1 / (J |H_m|), J = frames in the window, booked at once.
    The interface (``mark_frame``, ``log``, ``rollback``, ``apply``, ``committed``, ``planning``) matches ``AreaScore``.

    ``quality`` (A4, user 2026-11-11): a sweep counts by the image quality q of its 1 s sample (the detection's turn-rate factor, the same one
    that scales the POD): each cell keeps the best q so far (0-255), c_j sums those qualities, and a better sweep later adds the difference.
    Without it every sweep has q = 1 (A3).

    ``roll_max_deg`` (A4t, user 2026-11-11, method (가)): a cell is searched (q = 1) only when it is in the image at a 1 s sample whose turn
    rate is at most that of a coordinated turn at this roll (33 m/s: 20 deg = 6.2 deg/s), the gimbal's own rate included; other sweeps do
    not count, and the cell can still be searched later. Overrides ``quality``.

    ``h_off`` with ``road`` (user 2026-11-12): particles on a road cell (``road_mask``) are smoothed with ``h`` and the others with
    ``h_off`` (75 / 150 m: off-road particles spread over an area and are sparse, so the 75 m region broke into many small pieces there),
    the two densities added before the x% threshold."""

    def __init__(self, terr, h, track, exit_step, dt, x: float = 70.0, t_end: float | None = None, quality: bool = False,
                 window: tuple = (0.0, SEEN_T_END_S), roll_max_deg: float | None = None, v_mps: float = 33.0, h_off: float | None = None,
                 road=None):
        track = np.asarray(track)
        T, N = track.shape[:2]
        self.dt, self.x, self.h, self.quality = float(dt), float(x), float(h), bool(quality)
        self.h_off = float(h_off) if h_off and road is not None and float(h_off) != float(h) else 0.0
        self.roll_max_deg = roll_max_deg
        self.rate_max_dps = None if roll_max_deg is None else float(np.degrees(9.80665*np.tan(np.radians(roll_max_deg))/v_mps))
        J = self.J_all = max(min(int(round(t_end/self.dt)) if t_end is not None else T - 1, T - 1), 1)
        self.t_all = np.arange(1, J + 1)*self.dt
        tc = float(terr.axis[1] - terr.axis[0])
        self.n = n = len(terr.axis)
        self.X0 = float(terr.axis[0] - tc/2)
        f = int(round(RT_DENS_CELL_M/CELL_M))
        forb = np.asarray(terr.forbidden).ravel()
        ex = np.asarray(exit_step).astype(np.int64)
        g = RT_DENS_CELL_M
        sig = self.h/g
        sig_off = (self.h_off or self.h)/g
        pad = int(np.ceil(4*max(sig, sig_off))) + 1
        H = []
        for j in range(1, J + 1):
            idx = np.flatnonzero((ex < 0) | (ex > j))
            if len(idx) == 0:
                H.append(np.zeros(0, np.int32))
                continue
            pj = track[j, idx].astype(float)
            ig = np.floor((pj - self.X0)/g).astype(np.int64)
            lo = ig.min(0) - pad
            ij = ig - lo
            ny, nx = ij[:, 1].max() + pad + 1, ij[:, 0].max() + pad + 1
            flat = ij[:, 1]*nx + ij[:, 0]
            if self.h_off:                                     # road particles with h, off-road ones with h_off
                ic = np.clip(np.floor((pj - self.X0)/tc).astype(np.int64), 0, n - 1)
                on = road[ic[:, 1], ic[:, 0]]
                dens = gaussian_filter(np.bincount(flat[on], minlength=ny*nx).reshape(ny, nx).astype(float), sig, mode="constant") \
                    + gaussian_filter(np.bincount(flat[~on], minlength=ny*nx).reshape(ny, nx).astype(float), sig_off, mode="constant")
            else:
                hist = np.bincount(flat, minlength=ny*nx).reshape(ny, nx).astype(float)
                dens = gaussian_filter(hist, sig, mode="constant")
            thr = max(np.quantile(dens[ij[:, 1], ij[:, 0]], 1 - self.x/100), 1e-12)
            ry, rx = np.nonzero(dens >= thr)
            ry, rx = ry + lo[1], rx + lo[0]
            cy = (f*ry[:, None] + np.arange(f)[None, :]).repeat(f, 1).ravel()
            cx = np.tile(f*rx[:, None] + np.arange(f)[None, :], (1, f)).ravel()
            ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
            key = (cy[ok]*n + cx[ok]).astype(np.int64)
            key = np.unique(key[~forb[key]])
            H.append(key.astype(np.int32))
        self.H = H
        self.size_all = np.array([len(k) for k in H], float)
        self.Q = np.zeros(n*n, np.uint8)               # best image quality of each swept cell, 0 = not swept, 255 = q 1
        self.log, self.log_t, self.committed, self.planning = [], [], {}, False   # log: (keys, old Q, new Q)
        self.F = None
        self.set_window(*window)

    def set_window(self, t0: float, t1: float) -> None:
        """Scoring window: the frames with t0 <= t_j <= t1 (clipped to the track). Recomputes the booked values F_j."""
        j0 = int(np.clip(np.ceil(t0/self.dt - 1e-9), 1, self.J_all))
        j1 = int(np.clip(np.floor(t1/self.dt + 1e-9), j0, self.J_all))
        if self.F is not None and (j0, j1) == (self.j0, self.j1):
            return
        self.j0, self.j1 = j0, j1
        J = self.J = j1 - j0 + 1
        self.t = self.t_all[j0 - 1:j1]
        self.size = self.size_all[j0 - 1:j1]
        H = self.H[j0 - 1:j1]
        w = np.where(self.size > 0, 1.0/(J*np.maximum(self.size, 1.0)), 0.0)
        U = np.unique(np.concatenate(H)) if any(len(k) for k in H) else np.zeros(0, np.int32)
        acc = np.zeros(len(U))
        F = [None]*J
        for i in range(J - 1, -1, -1):                 # F_j(cell) = sum over window frames m >= j holding the cell of w_m
            pos = np.searchsorted(U, H[i])
            acc[pos] += w[i]
            F[i] = acc[pos].astype(np.float32)
        self.F = F

    def __getstate__(self):
        d = dict(self.__dict__)
        d["F"] = None                                  # rebuilt by set_window (pickled training scenarios stay small)
        if not self.log:
            d["Q"] = None                              # nothing swept yet: all zero, rebuilt on load (5.8 MB less per file)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault("h_off", 0.0)                 # regions saved before the off-road sigma (2026-11-12)
        if self.Q is None:
            self.Q = np.zeros(self.n*self.n, np.uint8)
        if self.F is None:
            j0, j1 = self.j0, self.j1
            self.j0 = self.j1 = -1
            self.set_window(j0*self.dt, j1*self.dt)

    # ---------------------------------------------------------------- state
    def clear(self) -> None:
        for keys, _, _ in self.log:
            self.Q[keys] = 0
        self.log.clear()
        self.log_t.clear()
        self.committed.clear()

    def frame_now(self, t: float) -> int:
        """Index (1..J) of the window frame nearest to t, halves rounded up (as ``curve`` counts a sweep at t in the frames with
        t < t_j + dt / 2)."""
        return int(np.clip(np.floor(t/self.dt + .5), self.j0, self.j1)) - self.j0 + 1

    def active(self, t: float) -> bool:
        return (self.j0 - .5)*self.dt <= t < (self.j1 + .5)*self.dt

    def rollback(self, cp: int) -> list:
        undone = self.log[cp:]
        for keys, old, _ in reversed(undone):
            self.Q[keys] = old
        del self.log[cp:]
        del self.log_t[cp:]
        return undone

    def apply(self, items) -> None:
        """Mark cells swept without scoring (other UAVs' committed sweeps while planning); undone by rollback."""
        for keys, _, qn in items:
            better = qn > self.Q[keys]
            if better.any():
                k = keys[better]
                self.log.append((k, self.Q[k].copy(), qn[better]))
                self.log_t.append(-1.0)
                self.Q[k] = qn[better]

    # ---------------------------------------------------------------- sweeping
    def _cells_in(self, quad):
        """Keys of the 25 m cells whose centre is inside the ground quad (4, 2)."""
        n, c = self.n, CELL_M
        ix0 = max(int(np.floor((quad[:, 0].min() - self.X0)/c)), 0)
        ix1 = min(int(np.floor((quad[:, 0].max() - self.X0)/c)) + 1, n)
        iy0 = max(int(np.floor((quad[:, 1].min() - self.X0)/c)), 0)
        iy1 = min(int(np.floor((quad[:, 1].max() - self.X0)/c)) + 1, n)
        if ix0 >= ix1 or iy0 >= iy1:
            return np.zeros(0, np.int64)
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        keys = (yy*n + xx).ravel()
        cx, cy = self.X0 + (xx.ravel() + .5)*c, self.X0 + (yy.ravel() + .5)*c
        return keys[inside_quad(cx, cy, quad[None])[0]]

    def mark_quad(self, quad, t: float, q: float = 1.0) -> float:
        """Sweep the cells inside the image quad (4, 2) at time t with image quality q (1 without ``quality``); returns the gain in A."""
        if not self.active(t):
            return 0.0
        return self._mark_keys(self._cells_in(np.asarray(quad, float)), t, q)

    def _mark_keys(self, keys, t: float, q: float = 1.0) -> float:
        qn = 255 if not self.quality else int(np.clip(round(float(q)*255), 1, 255))
        if not len(keys):
            return 0.0
        keys = keys[self.Q[keys] < qn]
        j = self.frame_now(t)
        Hj = self.H[self.j0 + j - 2]
        if not len(keys) or not len(Hj):
            return 0.0
        pos = np.clip(np.searchsorted(Hj, keys), 0, len(Hj) - 1)
        inr = Hj[pos] == keys
        if not inr.any():
            return 0.0
        new = keys[inr]
        old = self.Q[new].copy()
        self.Q[new] = qn
        self.log.append((new, old, np.full(len(new), qn, np.uint8)))
        self.log_t.append(float(t))
        return float(((qn - old.astype(np.float64))/255.0*self.F[j - 1][pos[inr]]).sum())

    def mark_frame(self, x, y, frame, t, cfg, q: float = 1.0) -> float:
        """Oblique camera (as ``AreaScore.mark_frame``): the image corners on flat ground ``cfg.alt_m`` below."""
        f, r, u = frame
        d = f[0] + _CORNER_H*np.tan(np.radians(cfg.hfov_deg/2))*r[0] + _CORNER_V*np.tan(np.radians(cfg.vfov_deg/2))*u[0]
        s = cfg.alt_m/np.maximum(-d[:, 2], 1e-3)
        quad = np.stack([x + s*d[:, 0], y + s*d[:, 1]], 1)
        img_min = float(getattr(self, "img_min", 0.0) or 0.0)
        if img_min <= 0:
            return self.mark_quad(quad, t, q)
        # (user 2026-11-12) a cell is searched only where the image-position factor of the POD (1 in the central half of each image axis,
        # 0.5 at an edge, 0.25 in a corner) is at least img_min; 0.75 keeps the central 3/4 of the width (about 141 m of 188 m)
        if not self.active(t):
            return 0.0
        keys = self._cells_in(quad)
        if not len(keys):
            return 0.0
        from rl_search.detection import image_coords
        c = self.centres(keys)
        uu, vv, front = image_coords(c[:, 0], c[:, 1], x, y, frame, cfg)
        ok = front[0] & (cfg.edge_factor(uu[0], vv[0]) >= img_min - 1e-9)
        return self._mark_keys(keys[ok], t, q)

    # ---------------------------------------------------------------- reading
    def open_now(self, t: float):
        """Region cells of the frame nearest to t that still hold value: (keys, A value of sweeping each now at q = 1)."""
        j = self.frame_now(t)
        Hj = self.H[self.j0 + j - 2]
        rem = 1.0 - self.Q[Hj]/255.0
        m = rem > 0
        return Hj[m], (self.F[j - 1][m]*rem[m]).astype(np.float32)

    def centres(self, keys):
        iy, ix = np.divmod(np.asarray(keys, np.int64), self.n)
        return np.stack([self.X0 + (ix + .5)*CELL_M, self.X0 + (iy + .5)*CELL_M], 1)

    def value_at(self, wx, wy, t: float) -> np.ndarray:
        """A value of sweeping the 25 m cells under world points now (0 outside the region or swept)."""
        j = self.frame_now(t)
        Hj = self.H[self.j0 + j - 2]
        v = np.zeros(np.shape(wx))
        if not len(Hj):
            return v
        ix = np.floor((np.asarray(wx) - self.X0)/CELL_M).astype(np.int64)
        iy = np.floor((np.asarray(wy) - self.X0)/CELL_M).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        keys = np.where(ok, iy*self.n + ix, -1)
        pos = np.clip(np.searchsorted(Hj, keys), 0, len(Hj) - 1)
        hit = ok & (Hj[pos] == keys)
        v[hit] = self.F[j - 1][pos[hit]]*(1.0 - self.Q[keys[hit]]/255.0)
        return v

    def cover_now(self, t: float) -> float:
        j = self.frame_now(t)
        Hj = self.H[self.j0 + j - 2]
        return float(self.Q[Hj].mean()/255.0) if len(Hj) else 0.0

    def curve(self) -> np.ndarray:
        """c_j per frame from the sweep log: the best quality each region cell had by t_j (sweeps before t_j + dt/2)."""
        out = np.zeros(self.J)
        if not self.log:
            return out
        keys = np.concatenate([k for k, _, _ in self.log])
        dq = np.concatenate([(n.astype(np.float64) - o)/255.0 for _, o, n in self.log])
        ts = np.concatenate([np.full(len(k), t) for (k, _, _), t in zip(self.log, self.log_t)])
        for j in range(self.J):
            Hj = self.H[self.j0 - 1 + j]
            if not len(Hj):
                continue
            m = (ts < self.t[j] + self.dt/2) & np.isin(keys, Hj)
            out[j] = dq[m].sum()/len(Hj)
        return out

    @property
    def A(self) -> float:
        return float(self.curve().mean())


class AreaEnv(TrackedEnv):
    """Team env whose detection step also scores the shared area term: gain = pos_w POS + area_w A. ``area`` is an ``AreaScore`` (5 min
    windows, swept cells) or a ``SeenArea`` (seen particles, booked once per decision from the survival since the hand-over)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.area, self.uav_id, self.pos_w, self.area_w = None, 0, 1.0, 1.0
        self._seen_on = False

    def reset(self, seed=None, options=None):
        self._seen_on = False                     # the pre-hand-over transit does not count, as for POS (run_team_launch s0)
        out = super().reset(seed, options)
        self._seen_on = True
        return out

    def _detect_lrc(self, xy, ts, w=None, state=None):
        gain = super()._detect_lrc(xy, ts, w, state)
        if w is not None or not isinstance(self.area, AreaScore):
            return gain
        return self.pos_w*gain + self.area_w*self.area.mark(xy, ts)

    def _swept(self, x, y, frame, t, rate_dps=None):
        if isinstance(self.area, RealtimeArea):
            a = self.area
            if a.rate_max_dps is not None:                        # A4t: searched only from a steady image
                if rate_dps is not None and abs(rate_dps) > a.rate_max_dps:
                    return
                q = 1.0
            else:
                q = float(self.detector.cfg.rate_factor(rate_dps)) if (a.quality and rate_dps is not None) else 1.0
            self._area_gain += a.mark_frame(x, y, frame, t, self.detector.cfg, q)
        elif isinstance(self.area, AreaScore):
            self._area_gain += self.area.mark_frame(x, y, frame, t, self.detector.cfg)

    def _detect_oblique(self, samples, ts):
        self._area_gain = 0.0
        self.last_samples = samples                 # rl_real: roll of each 1 s sample (roll-reversal cost)
        gain = super()._detect_oblique(samples, ts)
        if isinstance(self.area, SeenArea) and self._seen_on and len(ts):
            team = self.area.team
            surv = self.surv if not team or len(team) < 2 else np.prod([e.surv for e in team], axis=0)
            self._area_gain += self.area.update(surv, float(ts[-1]))
        # no area (R10 trains without A, user 2026-11-14): pos_w x POS, as with an area whose gain is 0 (pos_w is 1 unless an RL env set it)
        return self.pos_w*gain + self.area_w*self._area_gain

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
    if isinstance(a, SeenArea):
        return seen_peaks(env, t_mid, k, sep, block, smooth, radius)
    if isinstance(a, RealtimeArea):
        return realtime_peaks(env, t_mid, k, sep, block, smooth, radius)
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


def seen_peaks(env, t_mid: float, k: int, sep: float = 1500.0, block: int = 8, smooth: float = 250.0,
               radius: float = 2000.0) -> list[Cluster]:
    """``area_peaks`` for a ``SeenArea``: peaks of the unseen part of the region at t_mid. Each unseen region particle weighs 1 / (particles
    in its 25 m cell), so a block's weight is its unseen region area in cells, as the window version counts open cells."""
    a = env.area
    j = a.frame_now(t_mid)
    w = a.a[j - 1].astype(float)*a.J*a.size[j - 1]
    w[a.seen] = 0.0
    sel = w > 0
    if sel.sum() < 5:
        return []
    P, W = env.belief.positions(t_mid)[sel], w[sel]
    g = block*CELL_M
    lo = P.min(0) - 3*g
    ij = np.floor((P - lo)/g).astype(np.int64)
    ny, nx = ij[:, 1].max() + 4, ij[:, 0].max() + 4
    if ny < 3 or nx < 3:
        return []
    coarse = np.bincount(ij[:, 1]*nx + ij[:, 0], weights=W, minlength=ny*nx).reshape(ny, nx)
    dens = gaussian_filter(coarse, smooth/g)
    m = max(int(sep/g), 1)
    peaks = np.argwhere((dens == maximum_filter(dens, size=2*m + 1)) & (dens > 0))
    peaks = peaks[np.argsort(-dens[peaks[:, 0], peaks[:, 1]])][:k]
    out = []
    for i, jj in peaks:
        pk = lo + (np.array([jj, i]) + .5)*g
        near = np.hypot(P[:, 0] - pk[0], P[:, 1] - pk[1]) < radius
        if W[near].sum() < 5:
            continue
        Q, q = P[near], W[near]/W[near].sum()
        mu = (Q*q[:, None]).sum(0)
        D = Q - mu
        ev, evec = np.linalg.eigh((D*q[:, None]).T @ D + np.eye(2)*1e-6)
        theta = float(np.arctan2(evec[1, 1], evec[0, 1]))
        out.append(Cluster(pk, theta, max(np.sqrt(ev[1]), 200.0), max(np.sqrt(ev[0]), 200.0), np.zeros(2),
                           float(W[near].sum())*CELL_M**2/1e6))
    return out


def realtime_peaks(env, t_mid: float, k: int, sep: float = 1500.0, block: int = 8, smooth: float = 250.0,
                   radius: float = 2000.0) -> list[Cluster]:
    """``area_peaks`` for a ``RealtimeArea``: peaks of the unswept part of the region at the frame nearest to t_mid."""
    a = env.area
    keys, _ = a.open_now(t_mid)
    if len(keys) < 5:
        return []
    P = a.centres(keys)
    g = block*CELL_M
    lo = P.min(0) - 3*g
    ij = np.floor((P - lo)/g).astype(np.int64)
    ny, nx = ij[:, 1].max() + 4, ij[:, 0].max() + 4
    coarse = np.bincount(ij[:, 1]*nx + ij[:, 0], minlength=ny*nx).reshape(ny, nx).astype(float)
    dens = gaussian_filter(coarse, smooth/g)
    m = max(int(sep/g), 1)
    peaks = np.argwhere((dens == maximum_filter(dens, size=2*m + 1)) & (dens > 0))
    peaks = peaks[np.argsort(-dens[peaks[:, 0], peaks[:, 1]])][:k]
    out = []
    for i, jj in peaks:
        pk = lo + (np.array([jj, i]) + .5)*g
        near = np.hypot(P[:, 0] - pk[0], P[:, 1] - pk[1]) < radius
        if near.sum() < 5:
            continue
        Q = P[near]
        ev, evec = np.linalg.eigh(np.cov((Q - Q.mean(0)).T) + np.eye(2)*1e-6)
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
