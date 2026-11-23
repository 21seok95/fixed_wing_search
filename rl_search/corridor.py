"""Road corridor search: SAROPS in road coordinates (user 2026-11-18, docs/scenario_redesign.md "도로 회랑").

SAROPS places a rectangle (``accordion.Accordion``) and flies straight parallel lanes in it. On a curved road most of
the rectangle is empty land. Here the search area is a corridor: a stretch of road (the centre line) and a width. In the
road frame (s = distance along the centre line, d = signed distance across it) the corridor is a rectangle, so the
accordion sizing, the task-mode score and the SAROPS local search run unchanged on a (s, d) grid; the lanes are the
centre line shifted by d (curved lanes), flown with the same pursuit follower. On a straight road it is the rectangle.

Per plan (every task, as SAROPS):
  1. density peaks of the remaining particles at the intercept time (``Accordion.peaks``, as SAROPS);
  2. per peak, the centre line: the road edge with the most particles near the peak, extended at both ends along the
     adjacent road edge holding the most particles while that edge holds at least ``stop_frac`` of the cluster;
  3. the particles over the planning window projected to (s, d) of that line, accordion search from the peak cell;
  4. SAROPS local search on the flown result: the 4 start corners (which end, which side) and passes k = 1..3;
  5. the best corridor of all peaks is flown.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from rl_search.accordion import LANE_TOL, Accordion, Rect
from rl_search.patterns import parallel_pts
from rl_search.patterns_v2 import FramedFollower
from rl_search.sarops import CORNERS, SaropsPlanner

STEP_M = 25.0           # centre-line and lane point spacing
D_MAX = 2000.0          # particles farther than this from the centre line are outside the corridor frame
SNAP_M = 1500.0         # particles count for a road edge's mass within this distance
SMOOTH_M = 75.0         # centre-line smoothing (Gaussian sigma), so the shifted lanes have no kinks
MAX_LEN_M = 40000.0     # longest centre line
LOOK_M = 1000.0         # the line grows while the next LOOK_M of road (greedy chain) holds at least stop_frac of the cluster


# ---------------------------------------------------------------------- road samples (once per network)
_NETS: dict = {}


def road_index(net):
    """KD-tree over points every STEP_M along the main road edges, the edge of each point, and node -> edges."""
    key = id(net)
    if key in _NETS:
        return _NETS[key]
    me = np.flatnonzero(net.edge_main)
    pts, eid = [], []
    for e in me:
        p = net.poly[net.off[e]:net.off[e + 1]]
        seg = np.hypot(*np.diff(p, axis=0).T)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, c[-1] + 1e-6, STEP_M)
        pts.append(np.stack([np.interp(s, c, p[:, 0]), np.interp(s, c, p[:, 1])], 1))
        eid.append(np.full(len(s), e))
    pts, eid = np.concatenate(pts), np.concatenate(eid)
    u, v = net.de_from[0::2], net.de_to[0::2]
    adj: dict = {}
    for e in me:
        adj.setdefault(int(u[e]), []).append(int(e))
        adj.setdefault(int(v[e]), []).append(int(e))
    out = dict(tree=cKDTree(pts), eid=eid, u=u, v=v, adj=adj)
    _NETS[key] = out
    return out


@lru_cache(maxsize=4)
def region_net(region: str):
    from rl_search.real_scenario import load_net
    return load_net(region, 25.0)


# ---------------------------------------------------------------------- road frame
class RoadFrame:
    """A centre line (world points every STEP_M) with s, unit tangents and normals; world <-> (s, d)."""

    def __init__(self, pts: np.ndarray):
        seg = np.hypot(*np.diff(pts, axis=0).T)
        c = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, c[-1] + 1e-6, STEP_M)
        p = np.stack([np.interp(s, c, pts[:, 0]), np.interp(s, c, pts[:, 1])], 1)
        if len(p) > 5:
            p = gaussian_filter1d(p, SMOOTH_M/STEP_M, axis=0, mode="nearest")
        seg = np.hypot(*np.diff(p, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.p = p
        t = np.gradient(p, axis=0)
        self.t = t/np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        self.n = np.stack([-self.t[:, 1], self.t[:, 0]], 1)
        self.tree = cKDTree(p)
        self.L = float(self.s[-1])

    def world(self, s, d):
        s, d = np.broadcast_arrays(np.asarray(s, float), np.asarray(d, float))
        i = np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.s) - 2)
        a = np.clip((s - self.s[i])/np.maximum(self.s[i + 1] - self.s[i], 1e-9), 0, 1)[..., None]
        p = self.p[i]*(1 - a) + self.p[i + 1]*a
        nrm = self.n[i]*(1 - a) + self.n[i + 1]*a
        return p + d[..., None]*nrm

    def project(self, P):
        """(s, d) of world points P (n, 2): nearest centre-line point (STEP_M resolution) and the signed offset."""
        _, i = self.tree.query(P)
        r = P - self.p[i]
        along = np.einsum("ij,ij->i", r, self.t[i])
        s = np.clip(self.s[i] + along, 0, self.L)
        d = np.einsum("ij,ij->i", r, self.n[i])
        return s, d


def centre_line(net, P, w, peak, cluster_mass, stop_frac=0.02):
    """Road path through the particle cluster at ``peak`` (see the module doc); None if no road edge is near."""
    R = road_index(net)
    dist, idx = R["tree"].query(P, distance_upper_bound=SNAP_M)
    ok = np.isfinite(dist)
    mass = np.bincount(R["eid"][idx[ok]], weights=w[ok], minlength=len(net.edge_main))
    near = R["tree"].query_ball_point(peak, 1000.0)
    if not near:
        return None
    cand = np.unique(R["eid"][near])
    e0 = int(cand[np.argmax(mass[cand])])
    if mass[e0] <= 0:
        return None
    u, v, adj = R["u"], R["v"], R["adj"]
    stop = stop_frac*max(cluster_mass, mass[e0])
    used, total = {e0}, float(net.edge_len[e0])

    def other(e, a):
        return int(v[e]) if int(u[e]) == a else int(u[e])

    def ahead(node, first):
        """Greedy chain from ``node`` starting with edge ``first``, LOOK_M long: its edges and mass (roads are cut into short
        edges, so one edge alone holds little)."""
        chain, m, ln, a, e, seen = [], 0.0, 0.0, node, first, set(used)
        while e is not None and ln < LOOK_M:
            chain.append((e, a)); seen.add(e)
            m += mass[e]; ln += float(net.edge_len[e])
            a = other(e, a)
            opts = [k for k in adj.get(a, []) if k not in seen]
            e = max(opts, key=lambda k: mass[k]) if opts else None
        return chain, m

    def grow(node):
        nonlocal total
        path = []                      # (edge, from node)
        while total < MAX_LEN_M:
            opts = [e for e in adj.get(node, []) if e not in used]
            if not opts:
                break
            best = max((ahead(node, e) for e in opts), key=lambda x: x[1])
            if best[1] < stop:
                break
            e, a = best[0][0]
            used.add(e)
            path.append((e, a))
            total += float(net.edge_len[e])
            node = other(e, a)
        return path

    fwd = grow(int(v[e0]))
    bwd = grow(int(u[e0]))

    def poly(e, a):                    # edge e's points from node a
        p = net.poly[net.off[e]:net.off[e + 1]]
        return p if int(u[e]) == a else p[::-1]
    parts = [poly(e, a)[::-1] for e, a in reversed(bwd)] + [poly(e0, int(u[e0]))] + [poly(e, a) for e, a in fwd]
    pts = np.concatenate(parts)
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6]
    pts = pts[keep]
    return RoadFrame(pts) if len(pts) >= 2 and np.hypot(*(pts[-1] - pts[0])) > 0 else None


# ---------------------------------------------------------------------- corridor = rectangle in road coordinates
@dataclass
class Corridor(Rect):
    """A Rect in the road frame: ``s_c``, ``d_c`` its centre (s along the line, d across), ``ell`` its length along the
    line, ``wid`` its width across. ``cx``, ``cy`` (world, the centre), ``theta`` (0) keep the web payload working."""
    frame: RoadFrame | None = None
    s_c: float = 0.0
    d_c: float = 0.0

    def outline(self, n=60) -> np.ndarray:
        s = np.linspace(self.s_c - self.ell/2, self.s_c + self.ell/2, n)
        a = self.frame.world(s, np.full(n, self.d_c - self.wid/2))
        b = self.frame.world(s[::-1], np.full(n, self.d_c + self.wid/2))
        return np.concatenate([a, b, a[:1]])


def corridor_lanes(env, r: Corridor, search_s=None):
    """World polyline of the curved lanes (UAV path, camera lead applied), the footprint polyline, and the start corner."""
    from rl_search.accordion import lane_count
    r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
    n = lane_count(env, r, search_s=search_s)
    sp = r.wid/n
    base = parallel_pts(max(r.ell - sp, 100.0), n, sp, r_turn, LANE_TOL)
    x = env.plant.pose[:2]
    best = None
    for c in ([r.corner] if r.corner is not None else CORNERS):
        q = base*np.array(c)
        p0 = r.frame.world(r.s_c + q[0, 0], r.d_c + q[0, 1])
        dd = float(np.hypot(*(p0 - x)))
        if best is None or dd < best[0]:
            best = (dd, q, c)
    _, q, corner = best
    dense = [q[:1]]
    for a, b in zip(q[:-1], q[1:]):                      # points every STEP_M in the frame, then to the world
        m = max(2, int(np.hypot(*(b - a))/STEP_M) + 1)
        dense.append(np.linspace(a, b, m)[1:])
    q = np.concatenate(dense)
    fp = r.frame.world(r.s_c + q[:, 0], r.d_c + q[:, 1])
    lead = getattr(env, "lane_lead_m", 0.0)
    uav = fp
    if lead > 0:                                         # the UAV flies lead metres behind its footprint, along the path
        dv = np.gradient(fp, axis=0)
        dv /= np.maximum(np.hypot(*dv.T), 1e-9)[:, None]
        uav = fp - lead*dv
    return uav, fp, corner


def follower(env, uav_pts) -> FramedFollower:
    return FramedFollower(env, uav_pts, np.zeros(2), env.t, 0.0, np.zeros(2))


def simulate(env, r: Corridor, task: dict) -> float:
    """POS per minute of (transit + search) from flying the corridor lanes once (``accordion.simulate`` task mode);
    0 if the start is farther than max_transit_s. State kept."""
    uav, _, _ = corridor_lanes(env, r, task["search_s"])
    tr = float(np.hypot(*(uav[0] - env.plant.pose[:2]))/env.v)
    if tr > task["max_transit_s"]:
        return 0.0
    f = follower(env, uav)
    t_cap = env.t + tr + task["search_s"] + task["extra_s"]
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
    return gain/(max(dt, tr + task["search_s"])/60.0)


class CorridorAccordion(Accordion):
    """Accordion search on a (s, d) grid of one peak's road frame (cells: ``cell`` along, one lane width across)."""

    def __init__(self, env, net, stop_frac=0.02, **kw):
        super().__init__(env, **kw)
        self.net, self.stop_frac = net, stop_frac

    def _grid_frame(self, fr: RoadFrame):
        env, b = self.env, self.env.belief
        act = b.w > 0
        S, D, M = [], [], []
        for tau in self.taus:
            t = env.t + tau
            s, d = fr.project(b.positions(t))
            m = act & b.active(t) & (np.abs(d) < D_MAX)
            S.append(s); D.append(d); M.append(np.where(m, b.w, 0.0)/b.n)
        S, D, M = np.array(S), np.array(D), np.array(M)
        g = self.g
        ex = np.arange(0.0, fr.L + g, g)
        ey = np.arange(-D_MAX, D_MAX + self.gy, self.gy)
        H = np.stack([np.histogram2d(S[k], D[k], bins=[ex, ey], weights=M[k])[0] for k in range(len(self.taus))])
        C = np.zeros((H.shape[0], H.shape[1] + 1, H.shape[2] + 1))
        C[:, 1:, 1:] = H.cumsum(1).cumsum(2)
        return dict(theta=0.0, R=np.eye(2), ex=ex, ey=ey, H=H, C=C, frame=fr)

    def _climb(self, G, ci, cj):
        """Accordion growth as ``Accordion._climb`` but each side also moves by 2, 4, 8, 16 cells: along a road the particles
        come in bunches with gaps, and one-cell steps stop at the first gap."""
        r = super()._climb(G, ci, cj)
        if r is None:
            return None
        nx, ny = G["H"].shape[1:]
        i0 = int(round((r.s_c - r.ell/2 - G["ex"][0])/self.g)); i1 = i0 + int(round(r.ell/self.g))
        j0 = int(round((r.d_c - r.wid/2 - G["ey"][0])/self.gy)); j1 = j0 + int(round(r.wid/self.gy))
        cur, val = [i0, i1, j0, j1], self.approx(G, i0, i1, j0, j1)
        while True:
            best, best_v = None, val
            for k in range(4):
                for step in (1, 2, 4, 8, 16):
                    for d in (-step, step):
                        c = list(cur)
                        c[k] += d if k in (1, 3) else -d
                        if not (0 <= c[0] < c[1] <= nx and 0 <= c[2] < c[3] <= ny):
                            continue
                        v = self.approx(G, *c)
                        if v > best_v + 1e-12:
                            best, best_v = c, v
            if best is None:
                break
            cur, val = best, best_v
        return self.to_rect(G, cur, val)

    def _transit(self, G, i0, i1, j0, j1):
        s0, s1, d0, d1 = G["ex"][i0], G["ex"][i1], G["ey"][j0], G["ey"][j1]
        c = G["frame"].world(np.array([s0, s0, s1, s1]), np.array([d0, d1, d0, d1]))
        return float(np.hypot(*(c - self.env.plant.pose[:2]).T).min()/self.env.v)

    def to_rect(self, G, idx, val) -> Corridor:
        i0, i1, j0, j1 = idx
        sc, dc = (G["ex"][i0] + G["ex"][i1])/2, (G["ey"][j0] + G["ey"][j1])/2
        cx, cy = G["frame"].world(sc, dc)
        k = self._k.get((id(G), *idx), 1)
        return Corridor(float(cx), float(cy), float((i1 - i0)*self.g), float((j1 - j0)*self.gy), 0.0, val,
                        spacing=self.cover_w/k, k_max=max(self.ks), frame=G["frame"], s_c=float(sc), d_c=float(dc))

    def peak_rects(self):
        cls = self.peaks()
        if not cls:
            return None
        b = self.env.belief
        act = b.w > 0
        P, w = b.positions(self.t_read)[act], b.w[act]/b.n
        out = []
        for cl in cls:
            fr = centre_line(self.net, P, w, cl.peak, cl.mass, self.stop_frac)
            if fr is None:
                continue
            G = self._grid_frame(fr)
            s, d = fr.project(cl.peak[None])
            ci = int(np.clip(s[0]//self.g, 0, len(G["ex"]) - 2))
            cj = int(np.clip((d[0] + D_MAX)//self.gy, 0, len(G["ey"]) - 2))
            r = self._climb(G, ci, cj)
            if r is not None:
                out.append(r)
        return sorted(out, key=lambda r: -r.approx)


class CorridorPlanner(SaropsPlanner):
    """SAROPS with road corridors instead of rectangles (same task mode, peaks and local search on the flown result;
    the moves are the 4 start corners and the passes k). ``net``: the scenario's road network."""

    def __init__(self, net=None, region: str | None = None, stop_frac=0.02, **kw):
        super().__init__(**kw)
        self.net = net if net is not None else region_net(region)
        self.stop_frac = stop_frac

    def refine(self, env, r: Corridor, cover_w: float):
        """Local search on the flown result: every start corner x passes k = 1..k_max (lane spacing cover_w / k)."""
        cands = [replace(r, corner=c, spacing=cover_w/k) for c in CORNERS for k in range(1, r.k_max + 1)]
        vals = [simulate(env, q, self.task) for q in cands]
        i = int(np.argmax(vals))
        return cands[i], vals[i]

    def _plan(self, env) -> bool:
        acc = CorridorAccordion(env, self.net, self.stop_frac, variant=self.variant, cycle_s=self.cycle_s, cell=self.cell, W=self.W,
                                search_s=self.search_s, max_transit_s=self.max_transit_s, **self.acc_kw)
        rects = acc.peak_rects() or []
        if not rects:
            return False
        res = [self.refine(env, r, acc.cover_w) for r in rects]
        picked = int(np.argmax([v for _, v in res]))
        r = res[picked][0]
        uav, fp, _ = corridor_lanes(env, r, self.search_s)
        self.follower = follower(env, uav)
        self.hold_until = env.t
        self.log.append(dict(t_min=round(env.t/60, 1), ell=r.ell, wid=r.wid, line_km=round(r.frame.L/1000, 2), approx=round(r.approx, 4),
                             k=int(round(acc.cover_w/r.spacing)), picked=picked, starts=len(res)))
        self.rects.append((env.t, r, fp))
        tr = float(np.hypot(*(uav[0] - env.plant.pose[:2]))/env.v)
        self.tasks.append(dict(t0=env.t, transit=tr, t_cap=env.t + tr + self.search_s + self.extra_s, end=None, how=None, lane_t=None, hold=0.0))
        return True
