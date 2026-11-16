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
- ``Gimbal``: a roll / pitch gimbal that takes out the UAV attitude at once (within its roll / depression range)
  and turns its line of sight toward a command (forward 45 deg, or a ground point) at a limited slew rate.
- ``PassCap``: one pass over a particle finds at most ``pass_pod_max`` = 0.7.
- Resolution: pixels across a 3 m target at the slant range give a Johnson-type factor (50 % at 12 px),
  normalised to 1 at the nominal 45 deg view centre, so it only matters when the gimbal looks farther out.
- Oblique occlusion: partial cover A < 1 (forest, village, shrub) becomes A ** (sin 45 deg / sin depression), so a
  shallow view sees less under canopy (Beer-Lambert path length). Terrain masking by hills is not modelled.
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
    rate_tau_dps: float = 0.0           # > 0: exponential factor exp(-|omega| / tau) instead (user 2026-11-11, D2 in docs/rl_experiments.md)
    rate_dead_dps: float = 0.0          # with rate_tau_dps: 1 up to this turn rate, exp(-(|omega| - dead) / tau) above (D3, user 2026-11-11)
    rate_tau_low_dps: float = 0.0       # > 0: also exp(-|omega| / tau_low) from 0 on, i.e. D2 x D3 with no free zone (D4, user 2026-11-12)
    edge_plateau: float = 0.5           # image-position factor g: 1 for |u|, |v| up to this ...
    edge_value: float = 0.5             # ... falling linearly to this at the image edge, per axis (corner 0.25)
    image_px: int = 1920                # image width in pixels (for pixels on target only)
    target_m: float = 3.0
    pass_pod_max: float = 0.7           # most one pass can find (user 2026-11-06, as the earlier SAROPS sensor)
    pass_mode: str = "cap"              # "cap": min(pass POD, pass_pod_max); "scale": pass_pod_max x pass POD
    res_n50_px: float = 12.0            # resolution factor: pixels across the target at which detection halves ...
    res_exp: float = 3.0                # ... (Johnson-type curve); normalised to 1 at the 45 deg image centre
    use_resolution: bool = True
    oblique_occlusion: bool = True      # partial cover (A < 1) grows with the path through it: A ** (sin 45 / sin dep)
    pass_gap_s: float = 2.0             # a pass ends after this long out of every image
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

    def resolution_factor(self, slant_m):
        """Detection factor of the pixels across the target at this slant range, 1 at the nominal view centre."""
        ifov = np.radians(self.hfov_deg)/self.image_px

        def curve(r):
            n = self.target_m/(np.maximum(r, 1.0)*ifov)
            x = (n/self.res_n50_px)**self.res_exp
            return x/(1 + x)
        ref = curve(self.alt_m/np.sin(np.radians(self.depression_deg)))
        return np.minimum(curve(slant_m)/ref, 1.0)

    def rate_factor(self, omega_dps):
        w = np.abs(np.asarray(omega_dps, float))
        if self.rate_tau_dps > 0:
            low = w/self.rate_tau_low_dps if self.rate_tau_low_dps > 0 else 0.0
            return np.exp(-low - np.maximum(w - self.rate_dead_dps, 0.0)/self.rate_tau_dps)
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
        self.last_inside = inside.any(axis=0)
        g = np.where(inside, cfg.edge_factor(uu, vv), 0.0)
        self.last_edge = g.max(axis=0)                                      # image-position factor of each particle (best UAV)
        if cfg.use_resolution:
            slant = np.sqrt((px[None, :] - x[:, None])**2 + (py[None, :] - y[:, None])**2 + cfg.alt_m**2)
            g = g*np.where(inside, cfg.resolution_factor(slant), 0.0)
        rate = cfg.rate_factor(omega)                                      # (M,)
        expo = (g*rate[:, None]).sum(axis=0)                               # several UAVs add their rates
        A = self.visibility(px, py)
        if cfg.oblique_occlusion:
            # line of sight through canopy / buildings: transmission scales with the slant path, Beer-Lambert style
            slant = np.sqrt((px[None, :] - x[:, None])**2 + (py[None, :] - y[:, None])**2 + cfg.alt_m**2)
            k = np.sin(np.radians(cfg.depression_deg))/(cfg.alt_m/slant)          # (M, N), 1 at the nominal angle
            Ak = np.where(A[None, :] < 1, A[None, :]**k, 1.0)
            expo = (g*rate[:, None]*Ak).sum(axis=0)
            lam = cfg.lam0*expo
        else:
            lam = cfg.lam0*A*expo
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
    """Roll / pitch gimbal: fast attitude stabilisation, slow pointing.

    The pointing state is a line of sight in the world frame. Each step it turns toward the commanded line of
    sight by at most ``slew_dps`` (commanded re-pointing is slow). The UAV attitude is taken out at once
    (inertial stabilisation and the autopilot's feed-forward of planned turns are fast compared with the 1 s
    step), limited only by the mechanical range: roll ``roll_lim_deg``, depression ``dep_min_deg`` to
    ``dep_max_deg``. Beyond the range the view is clipped (forward 45 deg view at a roll above the limit tilts
    as in ``camera_frame``).

    Body frame: x forward, y left, z up; world = R_z(yaw) R_x(-roll) body (roll positive = left wing down).
    Gimbal: outer roll gamma about body x, inner depression theta from body x; boresight = R_x(gamma) (cos theta, 0,
    -sin theta). The image rate for the stability factor is max(UAV turn rate, line-of-sight rate in the world):
    the forward view in a turn turns with the heading (both equal), a ground-point stare in a turn moves little
    but still has the body motion, a re-point on a straight line moves the scene through the image.
    """

    def __init__(self, n_uav: int, dcfg: DetectionConfig, gcfg: GimbalConfig | None = None, yaw0=0.0):
        self.d, self.g = dcfg, gcfg or GimbalConfig()
        self.los = self.los_forward(np.full(n_uav, yaw0, float))
        self.gam = np.zeros(n_uav)
        self.th = np.full(n_uav, np.radians(dcfg.depression_deg))

    def los_forward(self, yaw):
        dep = np.radians(self.d.depression_deg)
        yaw = np.atleast_1d(np.asarray(yaw, float))
        return np.stack([np.cos(dep)*np.cos(yaw), np.cos(dep)*np.sin(yaw), -np.full_like(yaw, np.sin(dep))], 1)

    def los_point(self, x, y, tx, ty):
        x = np.atleast_1d(np.asarray(x, float))
        d = np.stack([np.broadcast_to(tx, x.shape) - x, np.broadcast_to(ty, x.shape) - np.atleast_1d(y),
                      np.full(len(x), -self.d.alt_m)], 1)
        return d/np.linalg.norm(d, axis=1, keepdims=True)

    def reset(self, yaw):
        self.los = self.los_forward(yaw)

    def step(self, yaw, roll, los_cmd, dt):
        """Turn toward ``los_cmd``; return the camera frame (f, r, u) and the line-of-sight rate (deg/s)."""
        yaw, roll = np.atleast_1d(yaw).astype(float), np.atleast_1d(roll).astype(float)
        old = self.los
        cosang = np.clip((old*los_cmd).sum(1), -1, 1)
        ang = np.arccos(cosang)
        step = np.radians(self.g.slew_dps)*dt
        frac = np.where(ang > 1e-9, np.minimum(1.0, step/np.maximum(ang, 1e-12)), 1.0)
        perp = los_cmd - old*cosang[:, None]
        nrm = np.linalg.norm(perp, axis=1, keepdims=True)
        perp = np.where(nrm > 1e-12, perp/np.maximum(nrm, 1e-12), 0.0)
        a = (ang*frac)[:, None]
        self.los = old*np.cos(a) + perp*np.sin(a)
        los_rate = np.degrees(ang*frac)/max(dt, 1e-9)
        # attitude taken out at once, within the mechanical range
        body = np.einsum("mij,mj->mi", _rx(roll), np.einsum("mij,mj->mi", _rz(-yaw), self.los))
        g = self.g
        self.th = np.clip(np.arccos(np.clip(body[:, 0], -1, 1)), np.radians(g.dep_min_deg), np.radians(g.dep_max_deg))
        self.gam = np.clip(np.arctan2(body[:, 1], -body[:, 2]), -np.radians(g.roll_lim_deg), np.radians(g.roll_lim_deg))
        th = self.th
        f = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], 1)
        r = np.stack([np.zeros_like(th), -np.ones_like(th), np.zeros_like(th)], 1)
        u = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
        to_world = np.einsum("mij,mjk->mik", _rz(yaw), np.einsum("mij,mjk->mik", _rx(-roll), _rx(self.gam)))
        frame = tuple(np.einsum("mij,mj->mi", to_world, v) for v in (f, r, u))
        return frame, los_rate


class PassCap:
    """Caps the probability one pass can add (``pass_pod_max``, 0.7 as in the earlier SAROPS sensor, S.11).

    A pass is a run of slices in which a particle is in an image; it ends when the particle has been out of
    every image for more than ``pass_gap_s``. Hazard beyond -ln(1 - cap) within a pass is dropped.
    """

    def __init__(self, n: int, cfg: DetectionConfig):
        self.cfg = cfg
        self.acc = np.zeros(n)          # hazard applied in the current pass
        self.raw = np.zeros(n)          # "scale": uncapped hazard of the current pass
        self.out = np.full(n, np.inf)
        self.hmax = -np.log(1 - cfg.pass_pod_max) if cfg.pass_pod_max < 1 else np.inf

    def apply(self, p, inside):
        """Probability of this slice from the uncapped one ``p`` and the in-image mask (N,).
        ``pass_mode`` "cap": a pass adds at most ``pass_pod_max``. "scale" (user 2026-11-06): a pass adds
        ``pass_pod_max`` x the model's pass POD, 1 - prod(1 - p) over its slices (never trust the model fully)."""
        dt = self.cfg.dt_s
        self.out = np.where(inside, 0.0, self.out + dt)
        new = self.out > self.cfg.pass_gap_s
        self.acc[new] = 0.0
        self.raw[new] = 0.0
        h = -np.log1p(-np.minimum(p, 1 - 1e-12))
        if self.cfg.pass_mode == "scale":
            self.raw += h
            target = -np.log1p(-self.cfg.pass_pod_max*(-np.expm1(-self.raw)))
            h = np.maximum(target - self.acc, 0.0)
        else:
            h = np.minimum(h, np.maximum(self.hmax - self.acc, 0.0))
        self.acc += h
        return -np.expm1(-h)


def load_landcover(region: str = "chungju", cell_m: float = 25.0) -> np.ndarray:
    from pathlib import Path
    sfx = "" if cell_m == 50 else f"_{cell_m:g}m"
    return np.load(Path("rl_search/data/real")/region/f"landcover{sfx}.npz")["lc"]
