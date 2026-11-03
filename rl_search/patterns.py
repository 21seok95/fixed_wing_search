"""Search methods that drive RLSearchEnv by its action (course-target change per 5 s decision), so every
method flies with the same plant, turn limit, sensor, energy and geofence as the RL policy.

IAMSAR patterns (parallel track, creeping line, track line, expanding square, spiral, sector) are built in
a pattern frame (x along the reported target heading, y to its left), placed at the predicted prior
centroid at the UAV's arrival time plus an offset, either fixed to the ground or moving with the predicted
centroid velocity, and flown by a pursuit follower. Adaptive baselines: straight, seek (remaining-mass
centroid now), lead (remaining-mass centroid at the intercept time) and greedy (POS lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from rl_search.plants import wrap

KINDS = ("parallel", "creeping", "trackline", "expanding", "spiral", "sector")


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def lane_order(n: int, k: int) -> list[int]:
    """Visit lanes k apart so a fixed-wing can reverse between consecutive lanes (k*spacing >= 2 R)."""
    return [i for r in range(k) for i in range(r, n, k)]


def parallel_pts(leg: float, n: int, spacing: float, r_turn: float) -> np.ndarray:
    """Lanes along x, stacked along y, centred on the origin; skip lanes to respect the turn radius."""
    n = max(int(n), 1)
    k = max(1, int(np.ceil(2*r_turn/spacing))) if n > 1 else 1
    ys = (np.arange(n) - (n - 1)/2)*spacing
    pts, d = [], 1.0
    for i in lane_order(n, k):
        x0, x1 = -d*leg/2, d*leg/2
        pts += [(x0, ys[i]), (x1, ys[i])]
        d = -d
    return np.array(pts)


def expanding_pts(total: float, spacing: float) -> np.ndarray:
    """Expanding square from the centre: legs 1,1,2,2,3,3,... spacing, left turns, until ``total`` length."""
    p, h, pts, length, i = np.zeros(2), 0.0, [np.zeros(2)], 0.0, 0
    while length < total:
        leg = spacing*(i//2 + 1)
        p = p + leg*np.array([np.cos(h), np.sin(h)])
        pts.append(p.copy())
        length += leg
        h += np.pi/2
        i += 1
    return np.array(pts)


def spiral_pts(r_max: float, spacing: float, r0: float) -> np.ndarray:
    """Archimedean spiral outward from r0 (minimum turn radius) to r_max, track spacing apart."""
    turns = max((r_max - r0)/spacing, 0.5)
    phi = np.linspace(0, 2*np.pi*turns, int(36*turns) + 2)
    r = r0 + spacing*phi/(2*np.pi)
    return np.stack([r*np.cos(phi), r*np.sin(phi)], axis=1)


def sector_pts(rho: float) -> np.ndarray:
    """IAMSAR sector search: three radial legs through the datum with 120 deg turns, then the same rotated 30 deg."""
    out = []
    for rot in (0.0, -np.pi/6):
        P = [rho*np.array([np.cos(rot - k*np.pi/3), np.sin(rot - k*np.pi/3)]) for k in range(6)]
        seq = [np.zeros(2), P[0], P[1], np.zeros(2), P[4], P[5], np.zeros(2), P[2], P[3], np.zeros(2)]
        out += seq if not out else seq[1:]
    return np.array(out)


@dataclass
class PatternParams:
    kind: str
    ox: float = 0.0            # datum offset along the reported heading (m)
    oy: float = 0.0            # datum offset to the left (m)
    size: float = 2000.0       # leg length / total length / radius, by kind
    n: int = 8                 # lanes (parallel, creeping, trackline)
    spacing_mult: float = 1.0  # track spacing in footprint widths
    moving: bool = True        # pattern moves with the predicted centroid

    def as_dict(self):
        return dict(kind=self.kind, ox=round(self.ox), oy=round(self.oy), size=round(self.size), n=int(self.n),
                    spacing_mult=self.spacing_mult, moving=self.moving)


def pattern_points(p: PatternParams, spacing: float, r_turn: float) -> np.ndarray:
    S = spacing*p.spacing_mult
    if p.kind == "parallel":
        pts = parallel_pts(p.size, p.n, S, r_turn)
    elif p.kind == "creeping":           # legs across the heading, creeping forward along it
        pts = parallel_pts(p.size, p.n, S, r_turn) @ _rot(np.pi/2).T
    elif p.kind == "trackline":          # few long legs along the predicted track
        pts = parallel_pts(p.size, min(p.n, 3), S, r_turn)
    elif p.kind == "expanding":
        pts = expanding_pts(p.size, S)
    elif p.kind == "spiral":
        pts = spiral_pts(p.size, S, r_turn)
    elif p.kind == "sector":
        pts = sector_pts(p.size)
    else:
        raise ValueError(p.kind)
    return pts + np.array([p.ox, p.oy])


class PriorTrack:
    """Unweighted prior centroid over time (model prediction, no observations): datum and its velocity."""

    def __init__(self, scen):
        k = np.arange(len(scen.track))
        self.t = k*scen.save_dt_s
        self.c = np.array([np.asarray(scen.track[i], float)[(scen.exit_step < 0) | (scen.exit_step > i)].mean(0)
                           if ((scen.exit_step < 0) | (scen.exit_step > i)).any() else np.full(2, np.nan)
                           for i in k])

    def at(self, t):
        return np.array([np.interp(t, self.t, self.c[:, j]) for j in (0, 1)])

    def velocity(self, t, window=600.0):
        return (self.at(t + window) - self.at(t))/window


class PatternFollower:
    """Pursuit along the pattern polyline (pattern frame), repeated when finished."""

    def __init__(self, env, p: PatternParams, prior: PriorTrack, lookahead: float = 150.0):
        self.env, self.p = env, p
        r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
        self.pts = pattern_points(p, env.fp_cross, r_turn)
        seg = np.diff(self.pts, axis=0)
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.heading = np.deg2rad(env.scen.heading_deg)
        self.R = _rot(self.heading)
        # datum: prior centroid when the UAV can arrive there
        x = env.plant.pose[:2]
        t_arr = env.t
        for _ in range(5):
            t_arr = env.t + np.hypot(*(prior.at(t_arr) - x))/env.v
        self.t_ref = t_arr
        self.O0 = prior.at(t_arr)
        self.vel = prior.velocity(t_arr) if p.moving else np.zeros(2)
        self.s, self.started, self.L = 0.0, False, lookahead

    def origin(self, t):
        return self.O0 + self.vel*(t - self.t_ref)

    def world(self, q, t):
        return self.origin(t) + q @ self.R.T

    def point_at(self, s):
        """Point(s) at arc length s (scalar or array) along the polyline."""
        s = np.clip(s, 0, self.cum[-1])
        return np.stack([np.interp(s, self.cum, self.pts[:, 0]), np.interp(s, self.cum, self.pts[:, 1])], axis=-1)

    def action(self) -> float:
        env, t = self.env, self.env.t
        x = env.plant.pose[:2]
        q = (x - self.origin(t)) @ self.R
        if not self.started:
            if np.hypot(*(q - self.pts[0])) < 2*self.L:
                self.started = True
            target = self.pts[0]
        else:
            # progress: closest point on the polyline within a forward window
            ss = np.linspace(self.s, min(self.s + 600.0, self.cum[-1]), 61)
            d = np.hypot(*(self.point_at(ss) - q).T)
            self.s = max(self.s, ss[int(np.argmin(d))])
            if self.s >= self.cum[-1] - 1.0:
                self.s, self.started = 0.0, False
                target = self.pts[0]
            else:
                target = self.point_at(self.s + self.L)
        tw = self.world(target, t + env.cfg.decision_dt_s)
        bearing = np.arctan2(tw[1] - x[1], tw[0] - x[0])
        return float(np.clip(wrap(bearing - env.chi_tgt)/env.turn, -1, 1))


def remaining_centroid(env, t):
    b = env.belief
    act = b.w > 0
    if not act.any():
        return None
    return np.average(b.positions(t)[act], axis=0, weights=b.w[act])


def steer_to(env, xy) -> float:
    x = env.plant.pose[:2]
    bearing = np.arctan2(xy[1] - x[1], xy[0] - x[0])
    return float(np.clip(wrap(bearing - env.chi_tgt)/env.turn, -1, 1))


def seek_action(env) -> float:
    c = remaining_centroid(env, env.t)
    return 0.0 if c is None else steer_to(env, c)


def lead_action(env) -> float:
    """Steer to the remaining-mass centroid at the time the UAV can reach it (intercept)."""
    x, t = env.plant.pose[:2], env.t
    tau = 0.0
    c = None
    for _ in range(4):
        c = remaining_centroid(env, t + tau)
        if c is None:
            return 0.0
        tau = np.hypot(*(c - x))/env.v
    return steer_to(env, c)


GREEDY_SEQS = [(a, 0, 0) for a in (-1, -.5, 0, .5, 1)] + [(a, a, 0) for a in (-1, -.5, .5, 1)]


def greedy_action(env) -> float:
    """Best 15 s lookahead POS among a few action sequences; if nothing is reachable, lead pursuit."""
    gains = [env.preview(seq) for seq in GREEDY_SEQS]
    i = int(np.argmax(gains))
    if gains[i] <= 1e-9:
        return lead_action(env)
    return float(GREEDY_SEQS[i][0])


ADAPTIVE = {"straight": lambda env: 0.0, "seek": seek_action, "lead": lead_action, "greedy": greedy_action}


@dataclass
class EpisodeLog:
    t: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    xy: list = field(default_factory=list)


def run_episode(env, options, policy, keep_cells=False) -> EpisodeLog:
    """policy(env) -> action; env must be in fast mode. Logs POS over time (policy part) and the cells seen
    per interval (for scoring true targets)."""
    env.reset(options=options)
    log = EpisodeLog()
    log.t.append(env.t)
    log.pos.append(0.0)
    log.xy.append(env.plant.pose[:2].copy())
    done = False
    while not done:
        _, _, term, trunc, info = env.step([policy(env)])
        log.t.append(env.t)
        log.pos.append(info["pos_policy"])
        log.xy.append(env.plant.pose[:2].copy())
        if keep_cells:
            log.cells.append((env.last_t0, env.last_cells))
        done = term or trunc
    return log
