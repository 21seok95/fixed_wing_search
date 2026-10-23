"""Mission layer: particle belief with map exits, camera footprint, detection and coverage.

Forked from uav_search.py. The mission layer does not know the vehicle type; it receives poses
(x, y, yaw) and an attitude from the plant. Changes against the original:

- ``ParticleBelief`` takes the per-particle exit index. Once a particle has left the map its
  remaining weight moves to ``escaped`` and its weight becomes 0, so detection, POA and planners
  ignore it. 1 = POS + remaining mass + escaped holds at every step. ``use_exits=False`` keeps the
  original behaviour (exited particles stay at their first outside position).
- The footprint comes from a sensor model (``NadirSensor``) instead of being computed in place.
- ``Coverage`` takes arrays instead of the run dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

MAX_TURN_RC_CESSNA = 133.35  # deg per 4 s step from straight flight, measured by scripts/check_plant.py


@dataclass
class SearchConfig:
    launch_delay_min: float = 0.0
    flight_min: float = 60.0
    uav_speed_mps: float = 15.0
    dt_s: float = 4.0
    fov_along_m: float = 83.0
    fov_cross_m: float = 115.0
    max_turn_deg: float | None = MAX_TURN_RC_CESSNA  # planner lattice and env clip; None: measure from the plant
    bandwidth_m: float = 150.0
    pod: float = 0.3
    particle_weights: str = "remaining"  # planner mass: "remaining" w/N or "conditional" w/sum(w)
    explore_weight: float = 0.0
    explore_density: float = 0.0
    explore_zone: str = "disc"
    explore_speed_mps: float = 0.75
    explore_hdr: float = 0.95
    explore_cell_m: float = 25.0

    @property
    def step_m(self) -> float:
        return self.uav_speed_mps*self.dt_s

    @property
    def n_steps(self) -> int:
        return int(round(self.flight_min*60/self.dt_s))


class ParticleBelief:
    """Monte Carlo particles over time with per-particle weights and map exits.

    ``track`` (T, N, 2) is sampled every ``track_dt_s`` from the LKP (t = 0). Weights start at 1 and
    are never renormalised: ``mass`` = w.sum()/N is the probability still in the map and not found.
    A particle with ``exit_step = k`` is inactive from t = k * track_dt_s on.
    """

    def __init__(self, track: np.ndarray, track_dt_s: float, exit_step: np.ndarray | None = None,
                 use_exits: bool = True):
        self.track = track
        self.track_dt_s = float(track_dt_s)
        self.n = track.shape[1]
        self.w = np.ones(self.n)
        self.escaped = 0.0
        use = use_exits and exit_step is not None
        self.exit_t = np.where(exit_step >= 0, exit_step*self.track_dt_s, np.inf) if use \
            else np.full(self.n, np.inf)
        self._gone = np.zeros(self.n, bool)

    @property
    def duration_s(self) -> float:
        return (self.track.shape[0]-1)*self.track_dt_s

    @property
    def mass(self) -> float:
        return float(self.w.sum()/self.n)

    @property
    def pos(self) -> float:
        return 1.0 - self.mass - self.escaped

    def positions(self, t_s: float) -> np.ndarray:
        """Particle positions (N, 2) at time t_s after the LKP, linearly interpolated."""
        f = np.clip(t_s/self.track_dt_s, 0, self.track.shape[0]-1)
        i = min(int(np.floor(f)), self.track.shape[0]-2)
        a = f - i
        return (1-a)*self.track[i] + a*self.track[i+1]

    def active(self, t_s: float) -> np.ndarray:
        return t_s < self.exit_t

    def update_exits(self, t_s: float) -> float:
        """Move the weight of particles that have left the map by t_s to ``escaped``; return that mass."""
        new = ~self._gone & ~self.active(t_s)
        if not new.any():
            return 0.0
        lost = float(self.w[new].sum()/self.n)
        self.escaped += lost
        self.w[new] = 0.0
        self._gone |= new
        return lost


def footprint_probs(px, py, cx, cy, yaw, along_m, cross_m, h_m):
    """Fraction of each particle's Gaussian kernel inside each rotated footprint: (M, N).

    px, py: (N,) particle positions; cx, cy, yaw: (M,) footprint centres and headings (rad).
    h_m = 0 gives the hard inside/outside indicator.
    """
    cx, cy, yaw = (np.atleast_1d(np.asarray(v, float))[:, None] for v in (cx, cy, yaw))
    dx, dy = px[None, :] - cx, py[None, :] - cy
    c, s = np.cos(yaw), np.sin(yaw)
    u = dx*c + dy*s
    v = -dx*s + dy*c
    if h_m <= 0:
        return ((np.abs(u) <= along_m/2) & (np.abs(v) <= cross_m/2)).astype(float)
    pu = ndtr((along_m/2 - u)/h_m) - ndtr((-along_m/2 - u)/h_m)
    pv = ndtr((cross_m/2 - v)/h_m) - ndtr((-cross_m/2 - v)/h_m)
    return pu*pv


@dataclass
class Footprint:
    cx: np.ndarray
    cy: np.ndarray
    yaw: np.ndarray
    along_m: float
    cross_m: float
    pod: float


class NadirSensor:
    """Gimbal-stabilised camera looking straight down: a heading-aligned rectangle under the UAV.

    The attitude (roll) is ignored as long as it stays inside the gimbal range.
    """

    def __init__(self, cfg: SearchConfig, gimbal_roll_deg: float = 45.0):
        self.cfg = cfg
        self.gimbal_roll = np.deg2rad(gimbal_roll_deg)

    def footprint(self, poses: np.ndarray, attitude: np.ndarray | None = None) -> Footprint:
        poses = np.atleast_2d(poses)
        if attitude is not None and np.any(np.abs(attitude) > self.gimbal_roll):
            warnings.warn("Roll exceeds the gimbal range; the nadir footprint is not exact")
        return Footprint(poses[:, 0], poses[:, 1], poses[:, 2], self.cfg.fov_along_m, self.cfg.fov_cross_m,
                         self.cfg.pod)


def poa(belief: ParticleBelief, t_s, cx, cy, yaw, cfg: SearchConfig):
    """POA of each footprint at time t_s: share of the total (initial) probability inside it."""
    pts = belief.positions(t_s)
    p = footprint_probs(pts[:, 0], pts[:, 1], cx, cy, yaw, cfg.fov_along_m, cfg.fov_cross_m, cfg.bandwidth_m)
    return (p*belief.w[None, :]).sum(axis=1)/belief.n


def detect(belief: ParticleBelief, t_s: float, fp: Footprint, h_m: float) -> float:
    """Apply w <- w * prod_u (1 - POD * p_u) for the footprints at time t_s; return the probability found."""
    before = belief.mass
    pts = belief.positions(t_s)
    p = footprint_probs(pts[:, 0], pts[:, 1], fp.cx, fp.cy, fp.yaw, fp.along_m, fp.cross_m, h_m)
    belief.w *= np.prod(1 - fp.pod*p, axis=0)
    return before - belief.mass


class Coverage:
    """Unsearched passable ground on a fine grid, for statistics and the exploration bonus.

    Same as uav_search.Coverage; see there for the bonus zones. ``axis``/``forbidden`` are the terrain
    grid, ``lkp`` the disc centre, ``track``/``track_dt`` the prior for the HDR zone.
    """

    def __init__(self, axis, forbidden, lkp, track, track_dt, cfg: SearchConfig, block_m: float = 300.0):
        c = cfg.explore_cell_m
        cell = float(axis[1] - axis[0])
        lo = axis[0] - cell/2
        n = int(np.ceil((axis[-1] + cell/2 - lo)/c))
        self.cfg, self.c, self.lo = cfg, c, lo
        self.centres = lo + (np.arange(n) + .5)*c
        idx = np.clip(np.round((self.centres - axis[0])/cell).astype(int), 0, len(axis)-1)
        self.u = (~forbidden[np.ix_(idx, idx)]).astype(float)
        self.passable_km2 = self.u.sum()*c*c/1e6
        self.lkp = np.asarray(lkp, float)
        self.axis, self.grid_cell, self.forbidden = axis, cell, forbidden
        self.track, self.track_dt = track, track_dt
        self._zone_k = None
        self.b = max(1, int(round(block_m/c)))
        nb = -(-n//self.b)
        self.coarse = np.zeros((nb, nb))
        self._refresh_coarse(0, n, 0, n)
        bc = lo + (np.arange(nb) + .5)*self.b*c
        self.block_x, self.block_y = np.meshgrid(bc, bc)
        self._u0, self._coarse0 = self.u.copy(), self.coarse.copy()

    def reset(self, lkp, track, track_dt):
        """Back to unsearched ground for a new scenario on the same terrain."""
        self.u[:], self.coarse[:] = self._u0, self._coarse0
        self.lkp = np.asarray(lkp, float)
        self.track, self.track_dt = track, track_dt
        self._zone_k = None

    def _refresh_coarse(self, iy0, iy1, ix0, ix1):
        b = self.b
        by0, by1, bx0, bx1 = iy0//b, -(-iy1//b), ix0//b, -(-ix1//b)
        sub = self.u[by0*b:by1*b, bx0*b:bx1*b]
        pad = np.zeros(((by1-by0)*b, (bx1-bx0)*b))
        pad[:sub.shape[0], :sub.shape[1]] = sub
        self.coarse[by0:by1, bx0:bx1] = pad.reshape(by1-by0, b, bx1-bx0, b).sum((1, 3))

    def _window(self, x, y, r):
        n = len(self.centres)
        lo = lambda v: int(np.clip(np.floor((v - r - self.lo)/self.c), 0, n))
        hi = lambda v: int(np.clip(np.ceil((v + r - self.lo)/self.c), 0, n))
        return lo(y), hi(y), lo(x), hi(x)

    def radius(self, t_s: float) -> float:
        return self.cfg.explore_speed_mps*t_s if self.cfg.explore_speed_mps > 0 else np.inf

    def cell_mass(self, t_s: float) -> float:
        """Bonus mass of one fully unseen cell at time t_s."""
        if self.cfg.explore_density > 0:
            return self.cfg.explore_density*self.c**2/1e6
        r = self.radius(t_s)
        area = np.pi*r*r if np.isfinite(r) else self.passable_km2*1e6
        return self.cfg.explore_weight*self.c**2/max(area, self.c**2)

    def _prior_zone(self, t_s: float):
        """HDR mask of the unsearched prior KDE at the nearest track snapshot: (mask, (row0, col0))."""
        k = int(round(np.clip(t_s/self.track_dt, 0, len(self.track)-1)))
        if k != self._zone_k:
            cell, axis, h = self.grid_cell, self.axis, self.cfg.bandwidth_m
            ij = np.clip(np.round((np.asarray(self.track[k]) - axis[0])/cell).astype(int), 0, len(axis)-1)
            m = int(np.ceil(4*h/cell))
            c0, c1 = max(ij[:, 0].min() - m, 0), min(ij[:, 0].max() + m + 1, len(axis))
            r0, r1 = max(ij[:, 1].min() - m, 0), min(ij[:, 1].max() + m + 1, len(axis))
            hist = np.zeros((r1 - r0, c1 - c0))
            np.add.at(hist, (ij[:, 1] - r0, ij[:, 0] - c0), 1.0)
            dens = gaussian_filter(hist, h/cell, mode="constant")
            dens[self.forbidden[r0:r1, c0:c1]] = 0
            v = np.sort(dens.ravel())[::-1]
            cum = np.cumsum(v)
            thr = v[min(np.searchsorted(cum, self.cfg.explore_hdr*cum[-1]), len(v)-1)]
            self._zone, self._zone_off, self._zone_k = (dens >= thr) & (dens > 0), (r0, c0), k
        return self._zone, self._zone_off

    def in_zone(self, x, y, t_s: float) -> np.ndarray:
        if self.cfg.explore_zone == "disc":
            return np.hypot(x - self.lkp[0], y - self.lkp[1]) < self.radius(t_s)
        zone, (r0, c0) = self._prior_zone(t_s)
        col = np.round((np.asarray(x) - self.axis[0])/self.grid_cell).astype(int) - c0
        row = np.round((np.asarray(y) - self.axis[0])/self.grid_cell).astype(int) - r0
        ok = (row >= 0) & (row < zone.shape[0]) & (col >= 0) & (col < zone.shape[1])
        out = np.zeros(np.shape(x), bool)
        out[ok] = zone[row[ok], col[ok]]
        return out

    def near_cells(self, x, y, reach, t_s):
        """Unseen cells within ``reach`` of (x, y) inside the zone: (xy (K, 2), mass (K,))."""
        iy0, iy1, ix0, ix1 = self._window(x, y, reach)
        gx, gy = np.meshgrid(self.centres[ix0:ix1], self.centres[iy0:iy1])
        u = self.u[iy0:iy1, ix0:ix1]
        ok = (u > 1e-3) & (np.hypot(gx - x, gy - y) < reach)
        ok[ok] = self.in_zone(gx[ok], gy[ok], t_s)
        return np.stack([gx[ok], gy[ok]], axis=1), u[ok]*self.cell_mass(t_s)

    def far_blocks(self, t_s):
        """Coarse unseen mass inside the zone (tested at block centres), for the terminal term."""
        ok = self.coarse > 1e-3
        ok[ok] = self.in_zone(self.block_x[ok], self.block_y[ok], t_s)
        return np.stack([self.block_x[ok], self.block_y[ok]], axis=1), self.coarse[ok]*self.cell_mass(t_s)

    def update(self, fp: Footprint):
        r = .5*np.hypot(fp.along_m, fp.cross_m) + 3*self.c
        for x, y, yaw in zip(fp.cx, fp.cy, fp.yaw):
            iy0, iy1, ix0, ix1 = self._window(x, y, r)
            gx, gy = np.meshgrid(self.centres[ix0:ix1], self.centres[iy0:iy1])
            p = footprint_probs(gx.ravel(), gy.ravel(), x, y, yaw, fp.along_m, fp.cross_m, self.c)
            self.u[iy0:iy1, ix0:ix1] *= (1 - p[0]).reshape(gx.shape)
            self._refresh_coarse(iy0, iy1, ix0, ix1)

    @property
    def searched_km2(self) -> float:
        return self.passable_km2 - self.u.sum()*self.c**2/1e6
