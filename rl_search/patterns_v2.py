"""Pattern baseline v2: IAMSAR patterns re-planned every cycle (docs/sarops_approach_todo.md V.1).

Every ``cycle_s`` the planner looks at the remaining belief (weights after the unsuccessful search so far) at
the middle of the coming cycle, finds the top density peaks, measures each peak's cluster (main axis, spread,
velocity), and picks a pattern placement by cross-entropy optimisation of the probability found during the
cycle (rollouts of RLSearchEnv in fast mode). The chosen pattern is flown for one cycle by a pursuit follower,
then the planner runs again. Like SAROPS search increments, but with moving targets and a fixed-wing plant.

Also ``Greedy90``: choose among 3-segment action sequences (30 s each) by POS over 90 s, re-plan every 30 s.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from rl_search.patterns import (KINDS, _rot, expanding_pts, parallel_pts, sector_pts, spiral_pts)
from rl_search.plants import wrap


@dataclass
class Cluster:
    peak: np.ndarray
    theta: float
    s1: float
    s2: float
    vel: np.ndarray
    mass: float


def find_clusters(env, t_mid: float, k: int = 3, cell: float = 100.0, smooth: float = 250.0,
                  sep: float = 1000.0, radius: float = 2000.0) -> list[Cluster]:
    b = env.belief
    act = b.w > 0
    if not act.any():
        return []
    pts, w = b.positions(t_mid)[act], b.w[act]/b.n
    lo = np.quantile(pts, .005, axis=0) - 1500
    hi = np.quantile(pts, .995, axis=0) + 1500
    ex, ey = np.arange(lo[0], hi[0] + cell, cell), np.arange(lo[1], hi[1] + cell, cell)
    if len(ex) < 3 or len(ey) < 3:
        return []
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[ex, ey], weights=w)
    H = gaussian_filter(H, smooth/cell)
    m = int(sep/cell)
    peaks = np.argwhere((H == maximum_filter(H, size=2*m + 1)) & (H > 0))
    peaks = peaks[np.argsort(-H[peaks[:, 0], peaks[:, 1]])][:k]
    p_prev, p_next = b.positions(t_mid - 300)[act], b.positions(t_mid + 300)[act]
    out = []
    for i, j in peaks:
        pk = np.array([ex[i] + cell/2, ey[j] + cell/2])
        near = np.hypot(*(pts - pk).T) < radius
        ww = w[near]
        if ww.sum() <= 0 or near.sum() < 5:
            continue
        P = pts[near]
        mu = np.average(P, axis=0, weights=ww)
        C = np.cov((P - mu).T, aweights=ww) + np.eye(2)*1e-6
        ev, evec = np.linalg.eigh(C)
        theta = float(np.arctan2(evec[1, 1], evec[0, 1]))
        vel = np.average((p_next[near] - p_prev[near])/600.0, axis=0, weights=ww)
        out.append(Cluster(pk, theta, max(np.sqrt(ev[1]), 200.0), max(np.sqrt(ev[0]), 200.0), vel, float(ww.sum())))
    return out


def base_points(kind: str, scale: float, spacing: float, r_turn: float, path_len: float, cl: Cluster) -> np.ndarray:
    """Pattern polyline in the cluster frame (x along the main axis), sized to the cycle's path length."""
    if kind in ("parallel", "creeping"):
        major = cl.s1 if kind == "parallel" else cl.s2
        leg = float(np.clip(2.5*major*scale, 400, path_len/2))
        n = int(np.clip(round(path_len/(leg + np.pi*r_turn)), 1, 40))
        pts = parallel_pts(leg, n, spacing, r_turn)
        return pts if kind == "parallel" else pts @ _rot(np.pi/2).T
    if kind == "trackline":
        return parallel_pts(max(path_len/3*scale, 600), 3, spacing, r_turn)
    if kind == "expanding":
        return expanding_pts(path_len*scale, spacing)
    if kind == "spiral":
        return spiral_pts(r_turn + np.sqrt(path_len*spacing/np.pi)*scale, spacing, r_turn)
    if kind == "sector":
        return sector_pts(max(path_len/18*scale, 2.5*r_turn))
    raise ValueError(kind)


class FramedFollower:
    """Pursuit along a polyline given in a frame that is placed at ``origin_ref`` at ``t_ref``, rotated by
    ``theta`` and moving with ``vel``; repeats the pattern when finished."""

    def __init__(self, env, pts, origin_ref, t_ref, theta, vel, lookahead=150.0):
        self.env, self.pts = env, np.asarray(pts, float)
        seg = np.diff(self.pts, axis=0)
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.O, self.t_ref, self.R, self.vel = np.asarray(origin_ref, float), t_ref, _rot(theta), np.asarray(vel)
        self.s, self.started, self.L = 0.0, False, lookahead

    def origin(self, t):
        return self.O + self.vel*(t - self.t_ref)

    def point_at(self, s):
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
            ss = np.linspace(self.s, min(self.s + 600.0, self.cum[-1]), 61)
            self.s = max(self.s, ss[int(np.argmin(np.hypot(*(self.point_at(ss) - q).T)))])
            if self.s >= self.cum[-1] - 1.0:
                self.s, self.started = 0.0, False
                target = self.pts[0]
            else:
                target = self.point_at(self.s + self.L)
        tw = self.origin(t + env.cfg.decision_dt_s) + target @ self.R.T
        bearing = np.arctan2(tw[1] - x[1], tw[0] - x[0])
        return float(np.clip(wrap(bearing - env.chi_tgt)/env.turn, -1, 1))


@dataclass
class Choice:
    cluster: int
    kind: str
    dx: float = 0.0      # offset along the cluster axis in s1 units
    dy: float = 0.0      # offset across in s2 units
    dtheta: float = 0.0
    log_scale: float = 0.0
    spacing: float = 1.0
    moving: bool = True


class CyclicPatternPlanner:
    """policy(env) -> action. ``kinds``: one kind (re-placed each cycle) or several (CE also picks the kind)."""

    def __init__(self, kinds=KINDS, cycle_s=600.0, k_peaks=3, pop=16, iters=3, elite=0.25, seed=0, effort=0.8):
        self.kinds, self.cycle_s, self.k = tuple(kinds), cycle_s, k_peaks
        self.pop, self.iters, self.elite, self.effort = pop, iters, elite, effort
        self.rng = np.random.default_rng(seed)
        self.follower, self.t_next, self.log = None, -np.inf, []

    def build(self, env, clusters, c: Choice, t_mid):
        cl = clusters[c.cluster]
        r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
        path_len = env.v*self.cycle_s*self.effort
        pts = base_points(c.kind, float(np.exp(c.log_scale)), env.fp_cross*c.spacing, r_turn, path_len, cl)
        theta = cl.theta + c.dtheta
        origin = cl.peak + np.array([c.dx*cl.s1, c.dy*cl.s2]) @ _rot(theta).T
        return FramedFollower(env, pts, origin, t_mid, theta, cl.vel if c.moving else np.zeros(2))

    def rollout(self, env, follower) -> float:
        snap = env.snapshot()
        pos0, t_end = env.pos_cum, env.t + self.cycle_s
        while env.t < t_end - 1e-6:
            _, _, term, _, _ = env.step([follower.action()])
            if term:
                break
        gain = env.pos_cum - pos0
        env.restore(snap)
        return gain

    def sample(self, P):
        c = Choice(cluster=int(self.rng.choice(len(P["cluster"]), p=P["cluster"])),
                   kind=self.kinds[int(self.rng.choice(len(self.kinds), p=P["kind"]))],
                   moving=bool(self.rng.random() < P["moving"]))
        v = self.rng.normal(P["mu"], P["sd"])
        c.dx, c.dy, c.dtheta, c.log_scale, c.spacing = (float(np.clip(v[0], -2, 2)), float(np.clip(v[1], -2, 2)),
                                                       float(np.clip(v[2], -np.pi/4, np.pi/4)),
                                                       float(np.clip(v[3], -1, 1)), float(np.clip(v[4], 1, 2)))
        return c

    def plan(self, env):
        t_mid = env.t + self.cycle_s/2
        clusters = find_clusters(env, t_mid, self.k)
        if not clusters:
            return None
        P = dict(cluster=np.full(len(clusters), 1/len(clusters)), kind=np.full(len(self.kinds), 1/len(self.kinds)),
                 moving=0.5, mu=np.array([0, 0, 0, 0, 1.2]), sd=np.array([.7, .7, .3, .4, .3]))
        best, best_v = None, -1.0
        for _ in range(self.iters):
            cands = [self.sample(P) for _ in range(self.pop)]
            vals = np.array([self.rollout(env, self.build(env, clusters, c, t_mid)) for c in cands])
            order = np.argsort(-vals)
            if vals[order[0]] > best_v:
                best, best_v = cands[order[0]], float(vals[order[0]])
            el = [cands[i] for i in order[:max(2, int(self.elite*self.pop))]]
            a = .7
            cc = np.bincount([e.cluster for e in el], minlength=len(clusters))/len(el)
            kk = np.bincount([self.kinds.index(e.kind) for e in el], minlength=len(self.kinds))/len(el)
            P["cluster"] = a*cc + (1 - a)*P["cluster"]
            P["kind"] = a*kk + (1 - a)*P["kind"]
            P["moving"] = a*np.mean([e.moving for e in el]) + (1 - a)*P["moving"]
            V = np.array([[e.dx, e.dy, e.dtheta, e.log_scale, e.spacing] for e in el])
            P["mu"] = a*V.mean(0) + (1 - a)*P["mu"]
            P["sd"] = np.maximum(a*V.std(0) + (1 - a)*P["sd"], [.1, .1, .05, .05, .05])
            P["cluster"] /= P["cluster"].sum()
            P["kind"] /= P["kind"].sum()
        self.log.append(dict(t_min=round(env.t/60, 1), kind=best.kind, cluster=best.cluster, gain=round(best_v, 4),
                             moving=best.moving, n_clusters=len(clusters)))
        return self.build(env, clusters, best, t_mid)

    def __call__(self, env) -> float:
        if env.t >= self.t_next or self.follower is None:
            f = self.plan(env)
            self.follower = f if f is not None else self.follower
            self.t_next = env.t + self.cycle_s
        if self.follower is None:
            return 0.0
        return self.follower.action()


class Greedy90:
    """Best 3-segment (30 s each) action sequence by POS over 90 s, re-planned every 30 s."""

    def __init__(self, seg_steps=6):
        self.seg = seg_steps
        first, later = (-1, -.5, 0, .5, 1), (-1, 0, 1)
        self.seqs = [(a, b, c) for a in first for b in later for c in later]
        self.queue = []

    def __call__(self, env) -> float:
        from rl_search.patterns import lead_action
        if not self.queue:
            gains = [env.preview([a]*self.seg + [b]*self.seg + [c]*self.seg) for a, b, c in self.seqs]
            i = int(np.argmax(gains))
            if gains[i] <= 1e-9:
                return lead_action(env)
            self.queue = [self.seqs[i][0]]*self.seg
        return float(self.queue.pop(0))
