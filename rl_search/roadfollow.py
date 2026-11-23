"""Road following: "just follow the road where the distribution is high" (user 2026-11-18, feedback from colleagues), as literal
as possible, with a camera better than the RL's (docs/scenario_redesign.md "길 따라가기 규칙").

  1. once, at the start: straight to the road point under the heaviest particle bunch at arrival (``find_clusters`` at the
     intercept time, snapped to the road);
  2. from then on the UAV never leaves the road: it flies along it (pursuit, ``LOOKAHEAD_M``, so the turns stay smooth);
  3. every ``decide_s`` it looks ``look_m`` ahead along every branch (each junction both ways) and takes the road with the most
     particles under the camera at the time it passes there;
  4. when the road ahead holds less than ``turn_frac`` of what the road behind holds, it turns back (U-turn), so it reaches
     other branches through the road network;
  5. camera: the gimbal points at the road ``lead`` metres ahead along the path (the RL and SAROPS look straight ahead).
Only when no particle is within ``look_m`` of road either way does it fly straight to the heaviest bunch again (step 1).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from rl_search.accordion import AccordionPlanner
from rl_search.corridor import STEP_M, region_net, road_index
from rl_search.patrol import PatrolPlan
from rl_search.patterns_v2 import FramedFollower, find_clusters
from types import SimpleNamespace

LOOKAHEAD_M = 400.0     # pursuit look-ahead: the UAV cuts road bends tighter than its turn radius (143 m) instead of overshooting them; the gimbal keeps the camera on the road (200 m left loops, 2026-11-18)
ON_ROAD_M = 300.0       # the approach ends this close to its road point


def on_path(f: FramedFollower, env) -> None:
    """A new path that starts where the UAV already is (a re-decision on the road): carry on from the nearest path point.
    FramedFollower otherwise steers at the first point for one step; that point is usually a few tens of metres behind the
    UAV (nearest road point), so every re-decision made a full turn-back loop (found 2026-11-18 in the zoomed tracks)."""
    x = env.plant.pose[:2]
    if np.hypot(*(x - f.pts[0])) < 2*f.L:
        ss = np.linspace(0.0, min(600.0, f.cum[-1]), 61)
        f.s = float(ss[int(np.argmin(np.hypot(*(f.point_at(ss) - x).T)))])
        f.started = True


class GimbalFollower(FramedFollower):
    """Pursuit along a world polyline; the gimbal points at the path ``lead`` metres ahead of the UAV's progress."""

    def __init__(self, env, pts, lead):
        super().__init__(env, pts, np.zeros(2), env.t, 0.0, np.zeros(2), LOOKAHEAD_M)
        self.lead = lead
        on_path(self, env)

    def action(self) -> float:
        a = super().action()
        if self.lead > 0:
            self.env.gimbal_target = tuple(self.point_at(self.s + self.lead)) if self.started else None
        return a


class RoadFollowPlanner(AccordionPlanner):
    def __init__(self, region: str | None = None, net=None, decide_s=45.0, look_m=3000.0, turn_frac=0.25, gimbal=True,
                 max_paths=200, W=None, **_ignored):
        super().__init__(W=W)
        self.net = net if net is not None else region_net(region)
        self.R = road_index(self.net)
        self.decide_s, self.look_m, self.turn_frac, self.gimbal, self.max_paths = decide_s, look_m, turn_frac, gimbal, max_paths
        self.mode, self.approach_pt = "approach", None

    # ------------------------------------------------------------------ road geometry
    def _edge_pts(self, e, a):
        p = self.net.poly[self.net.off[e]:self.net.off[e + 1]]
        return p if int(self.R["u"][e]) == a else p[::-1]

    def _other(self, e, a):
        u, v = self.R["u"], self.R["v"]
        return int(v[e]) if int(u[e]) == a else int(u[e])

    def _snap(self, xy, heading=None):
        """Road point nearest ``xy``: (point, edge, [(head points, next node)] for both directions, the one along ``heading`` first)."""
        _, i = self.R["tree"].query(xy)
        e = int(self.R["eid"][i])
        q = self.R["tree"].data[i]
        p = self.net.poly[self.net.off[e]:self.net.off[e + 1]]
        k = int(np.argmin(np.hypot(*(p - q).T)))
        u0, v0 = int(self.R["u"][e]), int(self.R["v"][e])
        to_v, to_u = (np.vstack([q, p[k + 1:]]), v0, e), (np.vstack([q, p[:k][::-1]]), u0, e)
        dirs = [to_v, to_u]
        if heading is not None:
            def d(h):
                pts = h[0]
                return pts[min(len(pts) - 1, 4)] - pts[0] if len(pts) > 1 else np.zeros(2)
            hv = np.array([np.cos(heading), np.sin(heading)])
            dirs.sort(key=lambda h: -float(d(h) @ hv))
        return q, dirs

    def _paths(self, head):
        """All road paths of look_m from a direction head (points, next node, edge), branching at every junction."""
        out, stack = [], [head]
        adj = self.R["adj"]
        while stack and len(out) < self.max_paths:
            pts, node, last = stack.pop()
            L = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
            opts = [x for x in adj.get(node, []) if x != last]
            if L >= self.look_m or not opts:
                if len(pts) > 1:
                    out.append(self._resample(pts))
                continue
            for x in opts:
                stack.append((np.vstack([pts, self._edge_pts(x, node)[1:]]), self._other(x, node), x))
        return out

    def _resample(self, pts):
        seg = np.hypot(*np.diff(pts, axis=0).T)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, min(c[-1], self.look_m) + 1e-6, STEP_M)
        return np.stack([np.interp(s, c, pts[:, 0]), np.interp(s, c, pts[:, 1])], 1)

    # ------------------------------------------------------------------ particles under the camera
    def _trees(self, env):
        b = self.env_belief = env.belief
        act = b.w > 0
        ts = env.t + np.arange(0.0, 2*self.look_m/env.v + 120.0, 30.0)
        trees = []
        for t in ts:
            m = np.flatnonzero(act & b.active(t))
            trees.append((cKDTree(b.positions(t)[m]) if len(m) else None, m))
        return ts, trees

    def _mass(self, env, path, ts, trees, t0=None):
        """Particles within half a lane width of the path at the time the UAV passes each point (each once), weighted."""
        b, half = env.belief, 0.5*env.fp_cross
        t0 = env.t if t0 is None else t0
        tp = t0 + np.arange(len(path))*STEP_M/env.v
        k = np.clip(np.searchsorted(ts, tp) - 1, 0, len(ts) - 1)
        seen = set()
        for kk in np.unique(k):
            tree, m = trees[kk]
            if tree is None:
                continue
            sel = np.flatnonzero(k == kk)[::2]
            for ids in tree.query_ball_point(path[sel], half):
                seen.update(m[ids])
        return float(b.w[list(seen)].sum()/b.n) if seen else 0.0

    # ------------------------------------------------------------------ decisions
    def _go(self, env, pts, how):
        lead = float(getattr(env, "lane_lead_m", 0.0)) if self.gimbal else 0.0
        self.follower = GimbalFollower(env, pts, lead)
        if self.tasks and self.tasks[-1]["end"] is None:
            self.tasks[-1]["end"], self.tasks[-1]["how"] = env.t, "cap"
        mid = pts[len(pts)//2]
        L = (len(pts) - 1)*STEP_M
        self.rects.append((env.t, PatrolPlan(float(mid[0]), float(mid[1]), L, 120.0, 0.0, 0.0, path=pts, frame=SimpleNamespace(L=L)), pts))
        self.tasks.append(dict(t0=env.t, transit=0.0, t_cap=env.t + self.decide_s, end=None, how=None, lane_t=env.t))
        self.log.append(dict(t_min=round(env.t/60, 1), how=how, km=round(L/1000, 1)))

    def _approach(self, env) -> bool:
        x = env.plant.pose[:2]
        cls = find_clusters(env, env.t + 600.0, k=3)
        if not cls:
            return False
        best = max(cls, key=lambda c: c.mass)
        tr = float(np.hypot(*(best.peak - x)))/env.v                 # where the bunch will be when the UAV gets there
        cl2 = find_clusters(env, env.t + tr, k=1)
        peak = cl2[0].peak if cl2 else best.peak
        q, _ = self._snap(peak)
        self.approach_pt, self.mode = q, "approach"
        n = max(2, int(np.hypot(*(q - x))/STEP_M))
        self._go(env, np.linspace(x, q, n), "approach")
        return True

    def _decide(self, env) -> bool:
        x, hd = env.plant.pose[:2], float(env.plant.pose[2])
        ts, trees = self._trees(env)
        _, dirs = self._snap(x, hd)
        best = []
        for head in dirs:                                              # dirs[0]: on along the heading; dirs[1]: turned back
            paths = self._paths(head)
            if not paths:
                best.append((0.0, None))
                continue
            m = [self._mass(env, p, ts, trees) for p in paths]
            i = int(np.argmax(m))
            best.append((m[i], paths[i]))
        (fwd, pf), (back, pb) = best[0], best[1] if len(best) > 1 else (0.0, None)
        if fwd <= 0 and back <= 0:
            return self._approach(env)
        if pf is not None and fwd >= self.turn_frac*back:
            self._go(env, pf, "on")
        else:
            self._go(env, pb, "turn")
        return True

    def __call__(self, env) -> float:
        if self.mode == "approach" and self.follower is None:
            if not self._approach(env):
                return 0.0
        if self.mode == "approach" and self.approach_pt is not None and np.hypot(*(env.plant.pose[:2] - self.approach_pt)) < ON_ROAD_M:
            self.mode, self.t_next = "road", -np.inf
        if self.mode == "road" and env.t >= self.t_next:
            self._decide(env)
            self.t_next = env.t + self.decide_s
        return self.follower.action()
