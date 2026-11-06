"""Vehicle evader particles on the real-region terrain: road graph + off-road grid.

Agreed with the user (2026-11-06, docs/real_map_todo.md):
- Vehicles only. Speed and next state follow the terrain state of each particle, not one random speed.
- On roads the particle drives along the road graph (fast / slow tier). It may leave the road with the
  car into open land (fields, grass, villages), where it drives slowly on the 25 m grid and is blocked
  by forest, water, steep ground and the city zones. Near a road it may return to it.
- Large cities are avoided (user decision 2026-11-06, option b): roads that pass through a city zone are
  removed from the graph, so routes go around the cities where a bypass exists, and off-road driving
  cannot enter them. (The first version handed particles off when they reached a city; with a 45 deg
  heading 84 % were gone after 60 min because the fastest routes ran through Chungju and Jecheon.)
- All rates are per minute and converted to the step length: p_step = 1 - (1 - p_min)^(dt/60).
- Junctions: each particle heads for the map edge in its escape direction (a goal sector of
  ``goal_sector_deg``) and picks the next road by the travel time to that goal, like a navigation
  app. With probability ``p_detour`` per junction it takes one of the other roads at random. The escape
  heading drifts slowly, so the goal can change on the way. Choosing by the local road direction alone
  made particles circle in the mountain valleys (60 min: median 4 km from the LKP), and a softmax on the
  extra travel time (60 s temperature) did the same because junctions are only ~270 m apart.

States (stored every ``save_dt_s``): FAST, SLOW (on the graph), OFF (off-road), STOP (stopped, on or
off the road), EXIT (left the map; frozen at the first outside position, like the walking model).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt

from rl_search.terrain import Terrain

FAST, SLOW, OFF, STOP, EXIT = range(5)
STATE_NAMES = ("빠른 길", "느린 길", "논밭", "정지", "지도 밖")
KMH = 1/3.6


@dataclass
class VehicleConfig:
    particles: int = 2000
    duration_s: float = 4200.0
    dt_s: float = 5.0
    save_dt_s: float = 10.0
    # speeds (km/h): mean and spread per terrain state; a per-particle driver factor keeps a fast
    # driver fast in every state
    speed_cap_kmh: float = 50.0
    fast_kmh: float = 45.0
    fast_std_kmh: float = 5.0
    slow_kmh: float = 25.0
    slow_std_kmh: float = 5.0
    off_kmh: float = 12.0
    off_std_kmh: float = 4.0
    # rates per minute
    p_leave_fast_min: float = 0.05     # fast road -> off-road (only where the roadside is open land)
    p_leave_slow_min: float = 0.05     # slow road -> off-road
    p_return_min: float = 0.30         # off-road -> road, when a road is within return_radius_m
    return_radius_m: float = 100.0
    p_stop_min: float = 0.02           # moving -> stopped
    p_go_min: float = 0.20             # stopped -> moving
    # where a car can drive off-road
    off_max_slope_deg: float = 15.0
    # direction choices
    heading_std_deg: float = 25.0      # spread of the escape headings around the scenario heading
    reorient_std_deg_min: float = 3.0  # random walk of a particle's escape heading (deg per sqrt(min))
    goal_sector_deg: float = 5.0       # width of the map-edge goal sectors seen from the LKP
    p_detour: float = 0.05             # per junction: take another road than the fastest one to the goal
    avoid_deadend: bool = True         # detours never turn into a dead-end branch (no way out but back)
    respect_oneway: bool = True
    w_off_heading: float = 1.0         # off-road: pull towards the escape heading
    w_off_keep: float = 2.0            # off-road: keep the current driving direction
    w_off_road: float = 1.0            # off-road: pull towards the nearest road (fields boxed in by forest)

    @property
    def save_every(self) -> int:
        k = self.save_dt_s/self.dt_s
        if abs(k - round(k)) > 1e-9 or k < 1:
            raise ValueError("save_dt_s must be a positive multiple of dt_s")
        return int(round(k))

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s/self.dt_s))

    def p_step(self, p_min: float) -> float:
        return 1.0 - (1.0 - min(max(p_min, 0.0), 1.0))**(self.dt_s/60.0)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "VehicleConfig":
        d = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class RoadNet:
    """Directed road graph (both directions of every edge, with a one-way flag) and the snapping grid."""

    def __init__(self, terr: Terrain, graph: dict):
        self.terr = terr
        e = graph["edges"]
        self.node_xy = graph["node_xy"].astype(float)
        self.node_city = graph["node_city"].astype(bool)
        poly = graph["edge_poly"].astype(float)
        off = graph["edge_poly_offset"].astype(np.int64)
        ne = len(e)
        seg = np.hypot(*np.diff(poly, axis=0).T)
        seg[off[1:-1] - 1] = 0.0                                  # no segment across edge boundaries
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.edge_len = (cum[off[1:] - 1] - cum[off[:-1]]).clip(min=1e-3)
        self.poly, self.off, self.cum = poly, off, cum            # cum is global; edge e starts at cum[off[e]]
        self.edge_tier = e["tier"].astype(np.uint8)

        # directed edges: 2e = u -> v (forward), 2e+1 = v -> u
        self.de_from = np.empty(2*ne, np.int64); self.de_to = np.empty(2*ne, np.int64)
        self.de_from[0::2], self.de_to[0::2] = e["u"], e["v"]
        self.de_from[1::2], self.de_to[1::2] = e["v"], e["u"]
        ow = e["oneway"]
        self.de_oneway_ok = np.empty(2*ne, bool)
        self.de_oneway_ok[0::2] = ow >= 0
        self.de_oneway_ok[1::2] = ow <= 0
        # initial direction of each directed edge (over its first ~150 m)
        p0 = poly[off[:-1]]; p1 = poly[off[1:] - 1]
        a = self._point_at(np.arange(ne), np.minimum(150.0, self.edge_len))
        b = self._point_at(np.arange(ne), np.maximum(self.edge_len - 150.0, 0.0))
        dfw, dbw = a - p0, b - p1
        self.de_dir = np.empty((2*ne, 2))
        self.de_dir[0::2] = dfw/np.maximum(np.hypot(*dfw.T), 1e-6)[:, None]
        self.de_dir[1::2] = dbw/np.maximum(np.hypot(*dbw.T), 1e-6)[:, None]
        # points every half cell along every edge: city test and snapping grid
        n, cell, lo = len(terr.axis), terr.cell, terr.axis[0]
        pts_e, pts_s = [], []
        for k in range(ne):
            L = self.edge_len[k]
            s = np.linspace(0, L, max(2, int(L/(cell/2)) + 1))
            pts_e.append(np.full(len(s), k)); pts_s.append(s)
        pe, ps = np.concatenate(pts_e), np.concatenate(pts_s)
        xy = self._point_at(pe, ps)
        r = np.rint((xy[:, 1] - lo)/cell).astype(int); c = np.rint((xy[:, 0] - lo)/cell).astype(int)
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        in_city = np.zeros(len(pe), bool)
        in_city[ok] = terr.city[r[ok], c[ok]]
        self.edge_city = np.bincount(pe[in_city], minlength=ne) > 0
        # keep only the largest connected road network without the city roads: dropping village streets
        # left ~670 pieces (the largest holds 93 % of the length, 99 % of the fast roads); particles never
        # enter the others
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        nn = len(self.node_xy)
        keep = ~self.edge_city
        g = csr_matrix((np.ones(keep.sum()), (e["u"][keep], e["v"][keep])), shape=(nn, nn))
        _, lab = connected_components(g, directed=False)
        km = np.bincount(lab[e["u"][keep]], weights=self.edge_len[keep], minlength=lab.max() + 1)
        self.edge_main = keep & (lab[e["u"]] == np.argmax(km))
        self.main_share = float(self.edge_len[self.edge_main].sum()/self.edge_len.sum())
        self.city_share = float(self.edge_len[self.edge_city].sum()/self.edge_len.sum())
        self.de_main = np.repeat(self.edge_main, 2)
        # dead-end branches: main edges outside the 2-core (iterated removal of leaf nodes). Nodes near the
        # map edge are not leaves: those roads continue outside the map.
        u, v = e["u"].astype(np.int64), e["v"].astype(np.int64)
        anchor = np.abs(self.node_xy).max(axis=1) >= -terr.axis[0] - 500.0
        self.anchor = anchor
        alive = self.edge_main & (u != v)
        while True:
            deg = np.bincount(u[alive], minlength=nn) + np.bincount(v[alive], minlength=nn)
            leaf = (deg == 1) & ~anchor
            drop = alive & (leaf[u] | leaf[v])
            if not drop.any():
                break
            alive &= ~drop
        self.edge_core = alive | (self.edge_main & (u == v))
        self.deadend_share = float(self.edge_len[self.edge_main & ~self.edge_core].sum()
                                   / self.edge_len[self.edge_main].sum())
        self.set_oneway(True)
        self._goal_cache = {}

        # snapping grid: road cells hold an edge id and the distance along it (only edges that can be driven)
        drivable = self.allowed[0::2] | self.allowed[1::2]
        ok &= drivable[pe]
        self.cell_edge = np.full((n, n), -1, np.int32)
        self.cell_s = np.zeros((n, n), np.float32)
        self.cell_edge[r[ok], c[ok]] = pe[ok]
        self.cell_s[r[ok], c[ok]] = ps[ok]
        dist, idx = distance_transform_edt(self.cell_edge < 0, return_indices=True)
        self.road_dist = (dist*cell).astype(np.float32)
        self.near_r, self.near_c = idx[0].astype(np.int32), idx[1].astype(np.int32)
        self.drive = None

    def set_oneway(self, respect: bool) -> None:
        """Out-edge lists of the drivable directed graph.

        Only the strongly connected part is kept (every node can reach every other one), with all map-edge
        nodes joined through a virtual outside node, since those roads continue outside the map. Without
        this, removing the city roads left one-way traps: at the edge of Chungju 95 of 2000 particles drove
        back and forth on one 277 m one-way piece for the whole hour.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        ok = (self.de_oneway_ok if respect else np.ones_like(self.de_oneway_ok)) & self.de_main
        nn = len(self.node_xy)
        d = np.flatnonzero(ok)
        anc = np.flatnonzero(self.anchor)
        rows = np.concatenate([self.de_from[d], anc, np.full(len(anc), nn)])
        cols = np.concatenate([self.de_to[d], np.full(len(anc), nn), anc])
        g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nn + 1, nn + 1))
        _, lab = connected_components(g, directed=True, connection="strong")
        big = lab[:nn] == lab[nn]
        allowed = ok & big[self.de_from] & big[self.de_to]
        self.allowed = allowed
        self.scc_share = float(self.edge_len[allowed[0::2] | allowed[1::2]].sum()/self.edge_len[self.edge_main].sum())
        des = np.flatnonzero(allowed)
        order = np.argsort(self.de_from[des], kind="stable")
        des = des[order]
        cnt = np.bincount(self.de_from[des], minlength=nn)
        self.max_deg = int(cnt.max())
        self.out = np.full((nn, self.max_deg), -1, np.int64)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        rank = np.arange(len(des)) - start[self.de_from[des]]
        self.out[self.de_from[des], rank] = des
        self.respect = respect

    def edge_time(self, cfg: "VehicleConfig") -> np.ndarray:
        """Travel time (s) of every undirected edge at the mean speed of its tier."""
        v = np.where(self.edge_tier == 2, min(cfg.fast_kmh, cfg.speed_cap_kmh), min(cfg.slow_kmh, cfg.speed_cap_kmh))
        return self.edge_len/np.maximum(v*KMH, 0.5)

    def goal_costs(self, lkp, cfg: "VehicleConfig") -> np.ndarray:
        """Travel time (s) from every node to the map edge in each goal sector seen from ``lkp``: (K, nodes)."""
        key = (round(float(lkp[0])), round(float(lkp[1])), cfg.fast_kmh, cfg.slow_kmh, cfg.speed_cap_kmh,
               cfg.goal_sector_deg, self.respect)
        if key in self._goal_cache:
            return self._goal_cache[key]
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        allowed = self.allowed
        des = np.flatnonzero(allowed)
        w = self.edge_time(cfg)[des >> 1]
        nn = len(self.node_xy)
        rev = csr_matrix((w + 1e-6, (self.de_to[des], self.de_from[des])), shape=(nn, nn))   # reversed: goal -> node
        half = -self.terr.axis[0]
        main_node = np.zeros(nn, bool)
        main_node[self.de_from[allowed]] = True
        edge_nodes = np.flatnonzero((np.abs(self.node_xy).max(axis=1) >= half - 500.0) & main_node)
        ang = np.arctan2(*(self.node_xy[edge_nodes] - np.asarray(lkp, float))[:, ::-1].T)
        K = int(round(360/cfg.goal_sector_deg))
        sec = np.floor((np.degrees(ang) % 360)/cfg.goal_sector_deg + 0.5).astype(int) % K
        C = np.full((K, nn), np.inf)
        for k in range(K):
            src = edge_nodes[sec == k]
            if len(src):
                C[k] = dijkstra(rev, directed=True, indices=src, min_only=True)
        for k in np.flatnonzero(~np.isfinite(C).any(axis=1)):          # empty sector: use the nearest filled one
            filled = np.flatnonzero(np.isfinite(C).any(axis=1))
            C[k] = C[filled[np.argmin(np.abs((filled - k + K/2) % K - K/2))]]
        if len(self._goal_cache) > 8:
            self._goal_cache.clear()
        self._goal_cache[key] = C
        return C

    def set_drive_mask(self, max_slope_deg: float) -> None:
        t = self.terr
        self.drive = ~(t.forest | t.water | t.fords | t.city | (t.slope_deg >= max_slope_deg))
        self.drive_slope = max_slope_deg

    def _point_at(self, edge, s):
        """Point at distance s along undirected edge (from its u end)."""
        edge = np.asarray(edge); s = np.asarray(s, float)
        key = self.cum[self.off[edge]] + np.clip(s, 0, self.edge_len[edge])
        i = np.searchsorted(self.cum, key, side="right") - 1
        i = np.clip(i, self.off[edge], self.off[edge + 1] - 2)
        seg = self.cum[i + 1] - self.cum[i]
        f = np.where(seg > 0, (key - self.cum[i])/np.where(seg > 0, seg, 1), 0.0)
        return self.poly[i] + f[:, None]*(self.poly[i + 1] - self.poly[i])

    def position(self, de, s):
        """Point at distance s from the start of directed edge de."""
        e = de >> 1
        fw = (de & 1) == 0
        return self._point_at(e, np.where(fw, s, self.edge_len[e] - s))

    def direction(self, de, s, ds=25.0):
        p = self.position(de, s)
        q = self.position(de, np.minimum(s + ds, self.edge_len[de >> 1]))
        back = np.hypot(*(q - p).T) < 1e-3
        if back.any():
            q[back] = p[back]
            p[back] = self.position(de[back], np.maximum(s[back] - ds, 0))
        d = q - p
        return d/np.maximum(np.hypot(*d.T), 1e-6)[:, None]

    def cell_of(self, xy):
        t = self.terr
        n = len(t.axis)
        r = np.clip(np.rint((xy[:, 1] - t.axis[0])/t.cell).astype(int), 0, n - 1)
        c = np.clip(np.rint((xy[:, 0] - t.axis[0])/t.cell).astype(int), 0, n - 1)
        return r, c

    def snap(self, xy, heading):
        """Nearest road point for each position: directed edge (best agreement with heading), distance along it."""
        r, c = self.cell_of(xy)
        rr, cc = self.near_r[r, c], self.near_c[r, c]
        e = self.cell_edge[rr, cc].astype(np.int64)
        s = self.cell_s[rr, cc].astype(float)
        fw, bw = 2*e, 2*e + 1
        h = np.stack([np.cos(heading), np.sin(heading)], 1)
        dfw = self.direction(fw, s); dbw = self.direction(bw, self.edge_len[e] - s)
        ok_f = self.allowed[fw]; ok_b = self.allowed[bw]
        score_f = np.where(ok_f, (dfw*h).sum(1), -9); score_b = np.where(ok_b, (dbw*h).sum(1), -9)
        use_f = score_f >= score_b
        return np.where(use_f, fw, bw), np.where(use_f, s, self.edge_len[e] - s)


def _softmax_pick(rng, scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random((len(p), 1))
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


OFF_OFFSETS = np.deg2rad(np.array([-90, -60, -35, -15, 0, 15, 35, 60, 90], float))


def simulate_vehicle(net: RoadNet, cfg: VehicleConfig, lkp, heading_deg: float, rng):
    """Particle tracks of a vehicle evader from the LKP (snapped to the nearest road).

    Returns a namespace with ``track`` (T, N, 2) float32, ``state`` (T, N) uint8, ``speed_kmh`` (T, N)
    float16, ``exit_step`` (N,) int16 (first stored index outside the map, -1 = never) and ``lkp_road``.
    """
    if net.respect != cfg.respect_oneway:
        net.set_oneway(cfg.respect_oneway)
    if net.drive is None or net.drive_slope != cfg.off_max_slope_deg:
        net.set_drive_mask(cfg.off_max_slope_deg)
    terr = net.terr
    n, dt = cfg.particles, cfg.dt_s
    esc = np.deg2rad(heading_deg + rng.normal(0, cfg.heading_std_deg, n))
    driver = rng.normal(0, 1, n)
    de, s = net.snap(np.tile(np.asarray(lkp, float), (n, 1)), esc)
    pos = net.position(de, s)
    lkp_road = pos[0].copy()
    mode = np.where(net.edge_tier[de >> 1] == 2, FAST, SLOW).astype(np.uint8)   # FAST / SLOW / OFF
    stopped = np.zeros(n, bool)
    done = np.zeros(n, np.uint8)                                                 # 0 or EXIT
    psi = np.arctan2(*net.direction(de, s)[:, ::-1].T)                           # off-road driving direction

    p_leave = {FAST: cfg.p_step(cfg.p_leave_fast_min), SLOW: cfg.p_step(cfg.p_leave_slow_min)}
    p_ret, p_stop, p_go = cfg.p_step(cfg.p_return_min), cfg.p_step(cfg.p_stop_min), cfg.p_step(cfg.p_go_min)
    cap = cfg.speed_cap_kmh*KMH
    mean = {FAST: cfg.fast_kmh*KMH, SLOW: cfg.slow_kmh*KMH, OFF: cfg.off_kmh*KMH}
    std = {FAST: cfg.fast_std_kmh*KMH, SLOW: cfg.slow_std_kmh*KMH, OFF: cfg.off_std_kmh*KMH}
    reo = np.deg2rad(cfg.reorient_std_deg_min)*np.sqrt(dt/60)
    goal = net.goal_costs(lkp_road, cfg)
    K = len(goal)
    etime = net.edge_time(cfg)

    every = cfg.save_every
    T = cfg.n_steps//every + 1
    track = np.empty((T, n, 2), np.float32)
    state = np.empty((T, n), np.uint8)
    speed_rec = np.zeros((T, n), np.float16)
    edge_rec = np.full((T, n), -1, np.int32)                                      # road edge, -1 off-road
    exit_step = np.full(n, -1, np.int16)

    def record(k, v):
        track[k] = pos
        st = mode.copy()
        st[stopped] = STOP
        st[done > 0] = done[done > 0]
        state[k] = st
        speed_rec[k] = v/KMH
        edge_rec[k] = np.where(mode != OFF, de >> 1, -1)

    record(0, np.zeros(n))
    rows = np.arange(n)
    for t in range(cfg.n_steps):
        live = done == 0
        esc[live] += rng.normal(0, reo, live.sum())
        # stop / go
        mv = live & ~stopped
        stopped[mv & (rng.random(n) < p_stop)] = True
        stopped[live & stopped & ~mv & (rng.random(n) < p_go)] = False
        moving = live & ~stopped
        v = np.zeros(n)
        for m in (FAST, SLOW, OFF):
            k = moving & (mode == m)
            v[k] = np.clip(mean[m] + std[m]*driver[k], 0.8*KMH*3, cap)

        # ---------------- on the road: leave?
        road = moving & (mode != OFF)
        if road.any():
            pl = np.where(mode == FAST, p_leave[FAST], p_leave[SLOW])
            try_leave = road & (rng.random(n) < pl)
            if try_leave.any():
                i = np.flatnonzero(try_leave)
                d = net.direction(de[i], s[i])
                side = rng.choice([-1.0, 1.0], len(i))
                perp = np.stack([-d[:, 1], d[:, 0]], 1)*side[:, None]
                q = pos[i] + perp*1.5*terr.cell + d*0.5*terr.cell
                r, c = net.cell_of(q)
                inside = terr.inside(q[:, 0], q[:, 1])
                ok = inside & net.drive[r, c]
                j = i[ok]
                pos[j] = q[ok]
                h = np.arctan2(d[ok, 1], d[ok, 0]) + side[ok]*np.deg2rad(45)
                psi[j] = h
                mode[j] = OFF
                road &= ~np.isin(rows, j)

        # ---------------- drive along the graph
        if road.any():
            i = np.flatnonzero(road)
            rem = v[i]*dt
            cur_de, cur_s = de[i], s[i] + rem
            for _ in range(50):
                L = net.edge_len[cur_de >> 1]
                arr = cur_s >= L
                if not arr.any():
                    break
                a = np.flatnonzero(arr)
                left = cur_s[a] - L[a]
                node = net.de_to[cur_de[a]]
                cand = net.out[node]                                       # (k, max_deg)
                twin = cur_de[a] ^ 1
                valid = (cand >= 0) & (cand != twin[:, None])
                dead = ~valid.any(axis=1)
                valid[dead] = cand[dead] >= 0
                cand0 = np.maximum(cand, 0)
                sec = np.floor((np.degrees(esc[i[a]]) % 360)/cfg.goal_sector_deg + 0.5).astype(int) % K
                total = etime[cand0 >> 1] + goal[sec[:, None], net.de_to[cand0]]
                total = np.where(valid, total, np.inf)
                best = total.min(axis=1, keepdims=True)
                lost = ~np.isfinite(best[:, 0])                            # goal unreachable: follow the heading
                score = np.where(np.isfinite(total), -total + 1e-3*rng.random(total.shape), -1e12)
                if lost.any():
                    hv = np.stack([np.cos(esc[i[a[lost]]]), np.sin(esc[i[a[lost]]])], 1)
                    score[lost] = (net.de_dir[cand0[lost]]*hv[:, None, :]).sum(-1)
                score = np.where(valid, score, -1e13)
                choice = np.argmax(score, axis=1)
                others = valid.copy()
                others[np.arange(len(a)), choice] = False
                if cfg.avoid_deadend:
                    others &= net.edge_core[cand0 >> 1]
                detour = others.any(axis=1) & (rng.random(len(a)) < cfg.p_detour)
                if detour.any():
                    w = others[detour]*rng.random(others[detour].shape)
                    choice[detour] = np.argmax(w, axis=1)
                none = ~valid.any(axis=1)
                pick = cand[np.arange(len(a)), choice]
                pick[none] = twin[none]                                     # true dead end: turn around
                cur_de[a] = pick
                cur_s[a] = left
            L = net.edge_len[cur_de >> 1]
            cur_s = np.minimum(cur_s, L)
            de[i], s[i] = cur_de, cur_s
            pos[i] = net.position(cur_de, cur_s)
            mode[i] = np.where(net.edge_tier[cur_de >> 1] == 2, FAST, SLOW)
            psi[i] = np.arctan2(*net.direction(cur_de, cur_s)[:, ::-1].T)

        # ---------------- off-road: return to a road, or drive on the grid
        off = moving & (mode == OFF)
        if off.any():
            i = np.flatnonzero(off)
            r, c = net.cell_of(pos[i])
            near = net.road_dist[r, c] <= cfg.return_radius_m
            back = near & (rng.random(len(i)) < p_ret)
            if back.any():
                j = i[back]
                de[j], s[j] = net.snap(pos[j], esc[j])
                pos[j] = net.position(de[j], s[j])
                mode[j] = np.where(net.edge_tier[de[j] >> 1] == 2, FAST, SLOW)
                i = i[~back]
            if len(i):
                ang = psi[i, None] + OFF_OFFSETS[None, :]
                dist = v[i]*dt
                nx = pos[i, 0, None] + dist[:, None]*np.cos(ang)
                ny = pos[i, 1, None] + dist[:, None]*np.sin(ang)
                rr, cc = net.cell_of(np.stack([nx.ravel(), ny.ravel()], 1))
                okc = net.drive[rr, cc].reshape(nx.shape) | ~terr.inside(nx, ny)
                r0, c0 = net.cell_of(pos[i])
                ax_ = terr.axis
                to_road = np.arctan2(ax_[net.near_r[r0, c0]] - pos[i, 1], ax_[net.near_c[r0, c0]] - pos[i, 0])
                score = (cfg.w_off_keep*np.cos(OFF_OFFSETS)[None, :]
                         + cfg.w_off_heading*np.cos(ang - esc[i, None])
                         + cfg.w_off_road*np.cos(ang - to_road[:, None]))
                score = np.where(okc, score, -1e9)
                stuck = ~okc.any(axis=1)
                k = _softmax_pick(rng, score)
                sel = np.arange(len(i))
                newp = np.stack([nx[sel, k], ny[sel, k]], 1)
                newp[stuck] = pos[i[stuck]]
                psi[i] = np.where(stuck, psi[i] + np.pi + rng.normal(0, .5, len(i)), ang[sel, k])
                pos[i] = newp

        # ---------------- bookkeeping
        out = (done == 0) & ~terr.inside(pos[:, 0], pos[:, 1])
        done[out] = EXIT
        k = -(-(t + 1)//every)
        exit_step[out & (exit_step < 0)] = k
        if (t + 1) % every == 0:
            record((t + 1)//every, v)
    return SimpleNamespace(track=track, state=state, speed_kmh=speed_rec, edge=edge_rec, exit_step=exit_step,
                           lkp_road=lkp_road, cfg=cfg, heading_deg=heading_deg)


def load_net(region: str = "chungju", cell_m: float = 25.0):
    from rl_search.terrain import load_terrain
    sfx = "" if cell_m == 50 else f"_{cell_m:g}m"
    terr = load_terrain(Path("rl_search/data")/f"terrain_real_{region}_60km{sfx}.npz")
    g = dict(np.load(Path("rl_search/data/real")/region/f"road_graph{sfx}.npz"))
    return RoadNet(terr, g)
