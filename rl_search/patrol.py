"""Road patrol: a rule that flies along the roads where the probability will be when it gets there (user 2026-11-18,
docs/scenario_redesign.md "도로 순찰").

SAROPS and the corridor rule search an area (rectangle / corridor) and cross empty land between areas. Here there is no
area: the UAV flies on the road itself, so it searches while it moves, and every ``replan_s`` it picks the next stretch:
  1. candidates: road paths from the UAV's road point (both ways: on along its heading, or turned back), branching at
     every junction (the ``branch`` heaviest roads), ``horizon_m`` long; and from the road point nearest each of the
     ``n_peaks`` density peaks (straight transit there first), so a far heavier bunch can win over the road nearby;
  2. quick score of each: the particles within half a lane width of the camera footprint (``lead`` metres ahead on the
     path) at the time the UAV passes, each particle once, x single-pass POD, divided by the time (transit + path);
  3. the ``n_sim`` best are flown from the current state (``env.snapshot``) and the best POS per minute wins;
  4. flown with the same pursuit follower as the SAROPS lanes; the camera looks forward as for SAROPS and RL.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from rl_search.accordion import AccordionPlanner, Rect
from rl_search.corridor import STEP_M, region_net, road_index
from rl_search.patterns_v2 import FramedFollower, find_clusters

SNAP_UAV_M = 600.0      # the UAV is on the road network when a road point is this close


def _follower(env, path) -> FramedFollower:
    """Pursuit along a world path; on a re-decision on the road it carries on from the nearest point (``roadfollow.on_path``)."""
    from rl_search.roadfollow import on_path
    f = FramedFollower(env, path, np.zeros(2), env.t, 0.0, np.zeros(2))
    on_path(f, env)
    return f


@dataclass
class PatrolPlan(Rect):
    """A planned road path (world points) in the Rect fields the web payload reads (``cx``, ``cy``: its middle)."""
    path: np.ndarray | None = None
    frame: SimpleNamespace | None = None

    def outline(self, half=60.0) -> np.ndarray:
        p = self.path
        d = np.gradient(p, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], 1)/np.maximum(np.hypot(*d.T), 1e-9)[:, None]
        return np.concatenate([p + half*n, (p - half*n)[::-1], p[:1] + half*n[:1]])


class PatrolPlanner(AccordionPlanner):
    def __init__(self, region: str | None = None, net=None, replan_s=150.0, horizon_m=8000.0, branch=3, n_peaks=3, n_sim=3,
                 max_cands=300, search_s=None, max_transit_s=1800.0, extra_s=120.0, W=None, **_ignored):
        super().__init__(search_s=replan_s, max_transit_s=max_transit_s, extra_s=extra_s, W=W)
        self.net = net if net is not None else region_net(region)
        self.R = road_index(self.net)
        self.replan_s, self.H, self.branch, self.n_peaks, self.n_sim, self.max_cands = replan_s, horizon_m, branch, n_peaks, n_sim, max_cands
        self.cand_log = []

    # ------------------------------------------------------------------ road paths
    def _edge_pts(self, e, a):
        net = self.net
        p = net.poly[net.off[e]:net.off[e + 1]]
        return p if int(self.R["u"][e]) == a else p[::-1]

    def _other(self, e, a):
        u, v = self.R["u"], self.R["v"]
        return int(v[e]) if int(u[e]) == a else int(u[e])

    def _paths_from(self, xy, mass):
        """Road paths (world polylines) from the road point nearest ``xy``: both directions along its edge, then branching."""
        d, i = self.R["tree"].query(xy)
        e0 = int(self.R["eid"][i])
        p = self.net.poly[self.net.off[e0]:self.net.off[e0 + 1]]
        k = int(np.argmin(np.hypot(*(p - self.R["tree"].data[i]).T)))
        u0, v0 = int(self.R["u"][e0]), int(self.R["v"][e0])
        starts = [(np.vstack([self.R["tree"].data[i], p[k + 1:]]), v0), (np.vstack([self.R["tree"].data[i], p[:k][::-1]]), u0)]
        out = []
        adj = self.R["adj"]
        for head, node in starts:
            stack = [(head, node, e0)]
            while stack and len(out) < self.max_cands:
                pts, node, last = stack.pop()
                L = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
                opts = [x for x in adj.get(node, []) if x != last]
                if L >= self.H or not opts:
                    if len(pts) > 1:
                        out.append(pts)
                    continue
                for x in sorted(opts, key=lambda q: -mass[q])[:self.branch]:
                    stack.append((np.vstack([pts, self._edge_pts(x, node)[1:]]), self._other(x, node), x))
        return [self._cut(p) for p in out]

    def _cut(self, pts):
        seg = np.hypot(*np.diff(pts, axis=0).T)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, min(c[-1], self.H) + 1e-6, STEP_M)
        return np.stack([np.interp(s, c, pts[:, 0]), np.interp(s, c, pts[:, 1])], 1)

    # ------------------------------------------------------------------ scoring
    def _quick(self, env, cands, trees, t_slices):
        """Quick score per candidate (transit xy -> path): POS per minute (see the module doc)."""
        b, half = env.belief, 0.5*env.fp_cross
        lead = float(getattr(env, "lane_lead_m", 0.0))
        pod, x0 = env.search.pod, env.plant.pose[:2]
        out = []
        for path in cands:
            tr = float(np.hypot(*(path[0] - x0)))/env.v
            n = len(path)
            dist = np.arange(n)*STEP_M
            fp = path[np.minimum(np.arange(n) + int(round(lead/STEP_M)), n - 1)]
            tp = env.t + tr + dist/env.v
            k = np.clip(np.searchsorted(t_slices, tp) - 1, 0, len(t_slices) - 1)
            seen = set()
            for kk in np.unique(k[::4]):
                sel = np.flatnonzero(k == kk)[::4]                    # a point every 100 m
                for ids in trees[kk][0].query_ball_point(fp[sel], half):
                    seen.update(trees[kk][1][ids])
            val = float(b.w[list(seen)].sum()/b.n*pod) if seen else 0.0
            out.append(val/((tr + dist[-1]/env.v)/60.0))
        return np.array(out)

    def _sim(self, env, path) -> float:
        f = _follower(env, path)
        tr = float(np.hypot(*(path[0] - env.plant.pose[:2])))/env.v
        t_cap = env.t + tr + (len(path)*STEP_M)/env.v + 60.0
        snap = env.snapshot()
        skip, env.skip_cells = env.skip_cells, True
        p0, t0 = env.pos_cum, env.t
        while env.t < t_cap - 1e-6:
            _, _, term, _, _ = env.step([f.action()])
            if term or f.laps >= 1:
                break
        gain, dt = env.pos_cum - p0, env.t - t0
        env.skip_cells = skip
        env.restore(snap)
        return gain/(max(dt, 1.0)/60.0)

    def _plan(self, env) -> bool:
        b = env.belief
        act = b.w > 0
        if not act.any():
            return False
        horizon = self.max_transit_s + self.H/env.v + 120.0
        t_slices = env.t + np.arange(0.0, horizon, 60.0)
        trees = []
        for t in t_slices:
            m = np.flatnonzero(act & b.active(t))
            trees.append((cKDTree(b.positions(t)[m]) if len(m) else cKDTree(np.zeros((1, 2)) + 1e9), m if len(m) else np.zeros(1, int)))
        P, w = b.positions(env.t + 300.0)[act], b.w[act]/b.n
        dd, ii = self.R["tree"].query(P, distance_upper_bound=1500.0)
        ok = np.isfinite(dd)
        mass = np.bincount(self.R["eid"][ii[ok]], weights=w[ok], minlength=len(self.net.edge_main))
        cands = []
        x = env.plant.pose[:2]
        if self.R["tree"].query(x)[0] <= SNAP_UAV_M:
            cands += self._paths_from(x, mass)
        for cl in find_clusters(env, env.t + 600.0, k=self.n_peaks):     # far bunches: go there first, then patrol
            if np.hypot(*(cl.peak - x))/env.v <= self.max_transit_s:
                cands += self._paths_from(cl.peak, mass)
        cands = [c for c in cands if len(c) > 4]
        if not cands:
            return False
        q = self._quick(env, cands, trees, t_slices)
        top = np.argsort(-q)[:self.n_sim]
        sims = [self._sim(env, cands[i]) for i in top]
        best = int(top[int(np.argmax(sims))])
        path = cands[best]
        self.follower = _follower(env, path)
        tr = float(np.hypot(*(path[0] - x)))/env.v
        L = (len(path) - 1)*STEP_M
        mid = path[len(path)//2]
        r = PatrolPlan(float(mid[0]), float(mid[1]), L, 120.0, 0.0, float(q[best]), path=path, frame=SimpleNamespace(L=L))
        self.rects.append((env.t, r, path))
        self.log.append(dict(t_min=round(env.t/60, 1), cands=len(cands), quick=round(float(q[best]), 5), sim=round(float(max(sims)), 5),
                             transit_s=round(tr), km=round(L/1000, 1)))
        self.tasks.append(dict(t0=env.t, transit=tr, t_cap=env.t + tr + min(self.replan_s, L/env.v), end=None, how=None, lane_t=None, hold=0.0))
        return True
