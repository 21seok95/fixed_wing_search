"""Detection model for the oblique EO camera and an automatic detector (YOLO + tracker).

Design agreed with the user on 2026-11-06 (docs/real_map_todo.md, "POD 설계 합의"):

- Camera at ``alt_m`` above flat ground, line of sight ``depression_deg`` below the horizon, field of view
  ``hfov_deg`` x ``vfov_deg``. The defaults give the user's trapezoid: near edge 171 m, far edge 209 m,
  depth 201 m, centre 500 m from the UAV.
- Gimbal ``2axis`` (roll / pitch): keeps the depression through roll up to ``gimbal_roll_deg``; the look
  azimuth follows the UAV heading (``look`` = forward / left / right). Roll beyond the limit tilts the line
  of sight. ``3axis``: the look azimuth is held fixed on the ground (``az_world``), roll fully compensated.
- Per particle and time slice: P = 1 - exp(-lam0 * A * g(u, v) * f(omega) * eps * dt) inside the image, 0 outside.
  A is the terrain visibility at the particle (forest, village, ...), g the image-position factor (detection
  falls towards the image edges: truncated vehicles, lens fall-off, smaller far side; u, v are image
  coordinates in [-1, 1]), f the image-stability factor of the turn rate, eps the per-pass detector
  variation (noise mode only). Particles are projected into the camera image, so the footprint test and g
  come from the same geometry. Target size (3 m x 3 m, user 2026-11-06) only matters for truncation at the
  edge (1.5 m against a ~190 m footprint) and pixels on target (27-34 px at 1920 px across): both small.
- Relative motion and the footprint sweep in turns are not factors: the footprint and the particles are
  moved every ``dt_s`` (1 s), so the dwell time follows from the geometry.
- ``Gimbal``: a roll / pitch gimbal that follows a commanded line of sight (forward 45 deg, or a ground point)
  within its roll / depression limits and slew rate; its slew rate is added to the turn rate for f.
- Expected POD (no noise) for planning and belief updates; ``noisy=True`` adds attitude-rate noise,
  pointing error and eps for "true" detections in evaluation.

Terrain height is ignored (flat ground at the altitude reference); the footprint is cast on a plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

# ESA WorldCover classes
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP, WC_BUILT, WC_BARE, WC_SNOW, WC_WATER, WC_WETLAND, WC_MANGROVE, WC_MOSS = (
    10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


@dataclass
class DetectionConfig:
    alt_m: float = 500.0
    depression_deg: float = 45.0
    hfov_deg: float = 15.12
    vfov_deg: float = 11.52
    look: str = "forward"               # forward / left / right (2-axis: relative to the heading)
    gimbal: str = "2axis"               # 2axis (roll, pitch) or 3axis (azimuth held on the ground)
    gimbal_roll_deg: float = 45.0
    lam0: float = 0.5                   # detection rate (1/s) in full view (user 2026-11-06: 1.0 was too optimistic)
    vis_open: float = 1.0               # road, crop land, bare ground, water
    vis_shrub: float = 0.85             # shrub, grass, wetland, moss
    vis_village: float = 0.5            # built-up land outside the large-city zones (and the zones)
    vis_forest_edge: float = 0.4        # forest within forest_edge_m of open land
    vis_forest: float = 0.05            # forest interior
    forest_edge_m: float = 25.0
    road_full_view: bool = True         # road cells count as open, also inside forest
    rate_ok_dps: float = 5.0            # image-stability factor: 1 up to this turn rate
    rate_floor_dps: float = 20.0        # ... falling linearly to rate_floor at this rate
    rate_floor: float = 0.2
    edge_plateau: float = 0.5           # image-position factor g: 1 for |u|, |v| up to this ...
    edge_value: float = 0.5             # ... falling linearly to this at the image edge, per axis (corner 0.25)
    image_px: int = 1920                # image width in pixels (for pixels on target only)
    target_m: float = 3.0
    dt_s: float = 1.0
    # noise, evaluation only
    att_rate_std_dps: float = 3.0
    point_err_deg: float = 0.3
    eps_log_std: float = 0.3

    def edge_factor(self, u, v):
        def h(x):
            x = np.abs(x)
            return np.where(x <= self.edge_plateau, 1.0,
                            1 - (1 - self.edge_value)*(x - self.edge_plateau)/max(1 - self.edge_plateau, 1e-9))
        return h(u)*h(v)

    def rate_factor(self, omega_dps):
        w = np.abs(np.asarray(omega_dps, float))
        f = 1 - (1 - self.rate_floor)*(w - self.rate_ok_dps)/max(self.rate_floor_dps - self.rate_ok_dps, 1e-9)
        return np.clip(f, self.rate_floor, 1.0)


def visibility_map(terr, landcover: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Terrain visibility A on the terrain grid (float32)."""
    lc = landcover
    A = np.full(lc.shape, cfg.vis_open, np.float32)
    A[np.isin(lc, [WC_SHRUB, WC_GRASS, WC_WETLAND, WC_MOSS])] = cfg.vis_shrub
    forest = np.isin(lc, [WC_TREE, WC_MANGROVE])
    depth = distance_transform_edt(forest)*terr.cell           # distance to the nearest non-forest cell
    A[forest] = cfg.vis_forest
    A[forest & (depth <= cfg.forest_edge_m)] = cfg.vis_forest_edge
    built = (lc == WC_BUILT)
    if getattr(terr, "built", None) is not None:
        built = built | terr.built
    if getattr(terr, "city", None) is not None:
        built = built | terr.city
    A[built] = cfg.vis_village
    if cfg.road_full_view and getattr(terr, "surface", None) is not None:
        A[terr.surface > 0] = cfg.vis_open
    return A


def camera_frame(yaw, roll, cfg: DetectionConfig, az_world=None, point_err=None):
    """Camera axes in world coordinates (z up): forward f, right r, up u, each (M, 3).

    yaw (rad, 0 = east, counter-clockwise), roll (rad, positive = left wing down = left turn, as in
    ``rl_search.plants``). az_world: look azimuth for the 3-axis gimbal (rad); point_err: (M, 2) extra
    azimuth / depression errors (rad). For the 2-axis gimbal, roll beyond ``gimbal_roll_deg`` tilts the view.
    """
    yaw, roll = (np.atleast_1d(np.asarray(v, float)) for v in (yaw, roll))
    M = len(yaw)
    off = {"forward": 0.0, "left": np.pi/2, "right": -np.pi/2}[cfg.look]
    if cfg.gimbal == "3axis":
        az = yaw + off if az_world is None else np.broadcast_to(np.asarray(az_world, float), (M,)).copy()
        resid = np.zeros(M)
    else:
        az = yaw + off
        resid = np.sign(roll)*np.maximum(np.abs(roll) - np.radians(cfg.gimbal_roll_deg), 0.0)
    dep = np.full(M, np.radians(cfg.depression_deg))
    if point_err is not None:
        az = az + point_err[:, 0]; dep = dep + point_err[:, 1]
    f = np.stack([np.cos(dep)*np.cos(az), np.cos(dep)*np.sin(az), -np.sin(dep)], 1)
    r = np.stack([np.sin(az), -np.cos(az), np.zeros(M)], 1)              # right of the look direction
    u = np.cross(r, f)
    if np.any(resid != 0):                                                # uncompensated roll about the heading axis
        ax = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(M)], 1)
        c, s = np.cos(-resid)[:, None], np.sin(-resid)[:, None]           # left wing down: the view swings right
        rot = lambda d: d*c + np.cross(ax, d)*s + ax*(ax*d).sum(1, keepdims=True)*(1 - c)
        f, r, u = rot(f), rot(r), rot(u)
    return f, r, u


def footprint(x, y, yaw, roll, cfg: DetectionConfig, az_world=None, point_err=None):
    """Ground corners (M, 4, 2) of the camera footprint for UAV poses (flat ground ``alt_m`` below).

    Corner order: near-left, far-left, far-right, near-right (seen along the look direction).
    """
    x, y = (np.atleast_1d(np.asarray(v, float)) for v in (x, y))
    f, r, u = camera_frame(yaw, roll, cfg, az_world, point_err)
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    corners = []
    for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):                   # near = looking lower = -v
        d = f + sh*th*r + sv*tv*u
        dz = np.minimum(d[:, 2], -1e-3)                                   # above the horizon: clamp far away
        t = cfg.alt_m/(-dz)
        corners.append(np.stack([x + t*d[:, 0], y + t*d[:, 1]], 1))
    return np.stack(corners, 1)


def image_coords(px, py, x, y, frame, cfg: DetectionConfig):
    """Normalised image coordinates (M, N) of ground points: u across (-1 left .. 1 right), v (-1 near .. 1 far),
    and whether the point is in front of the camera."""
    f, r, u = frame
    d = np.stack([px[None, :] - np.atleast_1d(x)[:, None], py[None, :] - np.atleast_1d(y)[:, None],
                  np.full((len(np.atleast_1d(x)), len(px)), -cfg.alt_m)], -1)          # (M, N, 3)
    df = (d*f[:, None, :]).sum(-1)
    front = df > 1e-6
    dfs = np.where(front, df, 1.0)
    uu = (d*r[:, None, :]).sum(-1)/dfs/np.tan(np.radians(cfg.hfov_deg/2))
    vv = (d*u[:, None, :]).sum(-1)/dfs/np.tan(np.radians(cfg.vfov_deg/2))
    return uu, vv, front


def inside_quad(px, py, quad):
    """Points (N,) inside each convex quadrilateral (M, 4, 2): bool (M, N)."""
    q = quad[:, :, None, :]                                               # (M, 4, 1, 2)
    a, b = q, np.roll(q, -1, axis=1)
    cross = (b[..., 0] - a[..., 0])*(py[None, None, :] - a[..., 1]) - (b[..., 1] - a[..., 1])*(px[None, None, :] - a[..., 0])
    return np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)


@dataclass
class Detector:
    """Per-slice detection probabilities for particles under one or more UAV footprints."""
    terr: object
    landcover: np.ndarray
    cfg: DetectionConfig = field(default_factory=DetectionConfig)

    def __post_init__(self):
        self.A = visibility_map(self.terr, self.landcover, self.cfg)

    def visibility(self, px, py):
        t = self.terr
        n = len(t.axis)
        r = np.clip(np.rint((py - t.axis[0])/t.cell).astype(int), 0, n - 1)
        c = np.clip(np.rint((px - t.axis[0])/t.cell).astype(int), 0, n - 1)
        return self.A[r, c]

    def slice_pod(self, px, py, x, y, yaw, roll, omega_dps, rng=None, az_world=None, eps=None, frame=None):
        """Detection probability of each particle in one time slice of ``cfg.dt_s`` under M UAVs: (N,).

        omega_dps: turn rate of each UAV (deg/s); with a ``Gimbal`` pass its frame and add its slew rate.
        With ``rng`` the noise model is used (attitude-rate noise, pointing error; ``eps`` per particle,
        drawn by the caller once per pass).
        """
        cfg = self.cfg
        x, y, yaw, roll, omega = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, yaw, roll, omega_dps))
        pe = None
        if rng is not None:
            omega = np.abs(omega) + np.abs(rng.normal(0, cfg.att_rate_std_dps, len(x)))
            pe = rng.normal(0, np.radians(cfg.point_err_deg), (len(x), 2))
        if frame is None:
            frame = camera_frame(yaw, roll, cfg, az_world, pe)
        uu, vv, front = image_coords(px, py, x, y, frame, cfg)
        inside = front & (np.abs(uu) <= 1) & (np.abs(vv) <= 1)             # (M, N)
        g = np.where(inside, cfg.edge_factor(uu, vv), 0.0)
        rate = cfg.rate_factor(omega)                                      # (M,)
        expo = (g*rate[:, None]).sum(axis=0)                               # several UAVs add their rates
        lam = cfg.lam0*self.visibility(px, py)*expo
        if eps is not None:
            lam = lam*eps
        return 1 - np.exp(-lam*cfg.dt_s)

    def draw_eps(self, n, rng):
        s = self.cfg.eps_log_std
        return np.exp(rng.normal(-s*s/2, s, n))                           # mean 1


@dataclass
class GimbalConfig:
    """2-axis roll / pitch gimbal limits (assumed small stabilised gimbal, user 2026-11-06: pitch 0-90 is too much)."""
    roll_lim_deg: float = 45.0           # outer roll about the body x axis
    dep_min_deg: float = 20.0            # inner pitch: depression from the body x axis
    dep_max_deg: float = 70.0
    slew_dps: float = 40.0               # rate limit of each axis


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


class Gimbal:
    """Roll / pitch gimbal that follows a commanded line of sight within its limits and slew rate.

    Body frame: x forward, y left, z up; the UAV roll (positive = left wing down) is R_x(-roll), so
    world = R_z(yaw) R_x(-roll) body. Gimbal: outer roll gamma about body x, inner depression theta from body x;
    boresight in the body frame = R_x(gamma) (cos theta, 0, -sin theta). Holding the forward 45 deg view needs
    gamma = UAV roll; beyond ``roll_lim_deg`` the view tilts, as in ``camera_frame``.
    """

    def __init__(self, n_uav: int, dcfg: DetectionConfig, gcfg: GimbalConfig | None = None):
        self.d, self.g = dcfg, gcfg or GimbalConfig()
        self.gam = np.zeros(n_uav)
        self.th = np.full(n_uav, np.radians(dcfg.depression_deg))

    def los_forward(self, yaw):
        dep = np.radians(self.d.depression_deg)
        yaw = np.atleast_1d(yaw)
        return np.stack([np.cos(dep)*np.cos(yaw), np.cos(dep)*np.sin(yaw), -np.full_like(yaw, np.sin(dep))], 1)

    def los_point(self, x, y, tx, ty):
        d = np.stack([np.atleast_1d(tx) - np.atleast_1d(x), np.atleast_1d(ty) - np.atleast_1d(y),
                      np.full(len(np.atleast_1d(x)), -self.d.alt_m)], 1)
        return d/np.linalg.norm(d, axis=1, keepdims=True)

    def step(self, yaw, roll, los_world, dt):
        """Move toward the commanded line of sight; return the camera frame (f, r, u) and the slew rate (deg/s)."""
        yaw, roll = np.atleast_1d(yaw).astype(float), np.atleast_1d(roll).astype(float)
        body = np.einsum("mij,mj->mi", _rx(roll), np.einsum("mij,mj->mi", _rz(-yaw), los_world))
        th_c = np.arccos(np.clip(body[:, 0], -1, 1))
        ga_c = np.arctan2(body[:, 1], -body[:, 2])
        g = self.g
        th_c = np.clip(th_c, np.radians(g.dep_min_deg), np.radians(g.dep_max_deg))
        ga_c = np.clip(ga_c, -np.radians(g.roll_lim_deg), np.radians(g.roll_lim_deg))
        step = np.radians(g.slew_dps)*dt
        dth = np.clip(th_c - self.th, -step, step)
        dga = np.clip(ga_c - self.gam, -step, step)
        self.th += dth
        self.gam += dga
        slew = np.degrees(np.maximum(np.abs(dth), np.abs(dga)))/max(dt, 1e-9)
        th = self.th
        f = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], 1)
        r = np.stack([np.zeros_like(th), -np.ones_like(th), np.zeros_like(th)], 1)
        u = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
        to_world = np.einsum("mij,mjk->mik", _rz(yaw), np.einsum("mij,mjk->mik", _rx(-roll), _rx(self.gam)))
        frame = tuple(np.einsum("mij,mj->mi", to_world, v) for v in (f, r, u))
        return frame, slew


def load_landcover(region: str = "chungju", cell_m: float = 25.0) -> np.ndarray:
    from pathlib import Path
    sfx = "" if cell_m == 50 else f"_{cell_m:g}m"
    return np.load(Path("rl_search/data/real")/region/f"landcover{sfx}.npz")["lc"]
