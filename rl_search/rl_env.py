"""RL search environment following the design log (todo_response.md). Numpy only; the Gymnasium/SB3
wrapper is in ``rl_gym.py``.

One step is one decision interval [t_k, t_k + 5 s] in the order of 5.3 v2:

0. fix the particle positions/weights and the expanded HDR zone at t_k (sensor integral only)
1. action -> course target: chi_tgt <- chi_tgt + a * turn_limit (4.5 v3, reference B)
2. fly FixedWingPlant for 5 s, charge energy every inner step, collect the 25 m reference cells
   under the footprint every ``sensor_dt_s`` into C_k (cell centre inside, 1.2 v2); stop at a map exit
3. detection once per interval: q_i = kernel mass of particle i over C_k, w_i <- w_i (1 - POD q_i)
   (detect_model "lrc": per particle the closest point of approach to the 1 s path segments, lateral range
   curve LRC(d), one detection chance per pass: the pass factor is 1 - LRC(min distance in the pass))
4. new area: cells of C_k not seen before and inside the expanded HDR zone; then mark C_k seen
5. reward r = w_pos * dPOS / pos_ref + w_area * dA / area_ref (1.5 v2, 1.6 v8)
6. particles that left the map by t_k+1 move to ``escaped`` (after detection)
7. terminated on a map exit or energy <= 0; truncated is always False (6.1-6.3)
8. observation at t_k+1: four 2-channel maps (unseen reward area, remaining probability) + 8-vector

Curriculum (7.2 v1, 7.5 v4): stage 1/2 start at time t0 near the prior (no transit, energy already
spent for t0), stage 3 launches on the map edge and flies the 120 s transit toward the LKP first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from rl_search.energy import make_power
from rl_search.mission import ParticleBelief
from rl_search.plants import FixedWingPlant, wrap
from rl_search.scenario import Scenario, ScenarioSampler, make_scenario
from rl_search.terrain import Terrain


@dataclass
class RLConfig:
    decision_dt_s: float = 5.0             # 4.4
    turn_limit_deg: float = 30.0           # 4.4 v6: +-30 deg per decision
    sensor_dt_s: float = 1.0               # 5.3 v2
    cell_m: float = 25.0                   # 1.2 v4 reference cell
    hdr_level: float = 0.9                 # 1.1 v4
    margin_rate: float = 0.1               # 1.1 v4: margin = min(rate * reported speed * t, max)
    margin_max_m: float = 1000.0
    w_pos: float = 1.0
    w_area: float = 1.0
    pos_ref: float = 0.15                  # 1.5 v2 placeholder: replace by the measured greedy mean
    area_ref_m2: float | None = None       # None: V * footprint width * energy budget (max new area)
    map_px: int = 64
    ego_ranges_m: tuple = (1600.0, 6400.0, 25600.0)   # 3.1 v2: near, mid, wide (UAV centred)
    pos_scale: str | tuple = "density"   # 3.4: fixed input scale per map; "density" = mass per km² of map cell
    max_sub: int = 4                       # sub-samples per map cell side for the area channel
    stage: int = 1                         # 7.5 v4 curriculum stage
    t0_s: dict = field(default_factory=lambda: {1: (120.0, 120.0), 2: (600.0, 900.0)})
    start_dist_m: tuple = (1000.0, 2000.0)  # stage 1/2: from the prior median at t0
    heading_offset_deg: float = 90.0        # stage 1/2: initial heading = bearing to the prior +- this
    reuse: int = 8                          # episodes per target simulation
    map_margin_m: float = 1000.0            # stage 1/2 start points keep this far from the map edge
    geofence_m: float = 1000.0              # 6.2 v3: inside this band, an outward course target is replaced by
                                            # a full turn toward the map centre (0 disables)
    detect_model: str = "kernel"            # "kernel": 5.4 v3 (kernel mass over the interval's cells, once per
                                            # interval); "lrc": SAROPS (Kratzke et al. 2010) closest point of
                                            # approach per particle -> lateral range curve, one chance per pass
    lrc_range_m: float | None = None        # "lrc": definite-range LRC, POD inside this range (None: half the
                                            # footprint width, so the sweep width is POD * footprint width)
    pass_gap_s: float = 10.0                # "lrc": out of range longer than this before a new pass starts


MAP_NAMES = ("near", "mid", "wide", "global")


class RLSearchEnv:
    def __init__(self, terr: Terrain, preset, cfg: RLConfig | None = None, seed: int = 0):
        self.terr, self.preset, self.cfg = terr, preset, cfg or RLConfig()
        c = self.cfg
        self.search = preset.search
        self.plant = FixedWingPlant(preset.plant, c.decision_dt_s, max_turn_deg=180.0)
        self.power = make_power(preset)
        if self.power is None:
            raise ValueError("RLSearchEnv needs the energy model (preset.energy.enabled)")
        self.v = preset.plant.FW_AIRSPD_TRIM
        self.sampler = ScenarioSampler(terr, preset.scenario)
        self.rng = np.random.default_rng(seed)
        self.turn = np.deg2rad(c.turn_limit_deg)
        n_inner = self.plant.n_inner
        every = max(1, int(round(c.sensor_dt_s/self.plant.h)))
        self.sample_idx = np.arange(every - 1, n_inner, every)
        if self.sample_idx[-1] != n_inner - 1:
            self.sample_idx = np.append(self.sample_idx, n_inner - 1)
        # 25 m reference grid over the terrain extent
        tc = float(terr.axis[1] - terr.axis[0])
        self.tcell = tc
        self.lo = float(terr.axis[0] - tc/2)
        self.half_map = float(terr.axis[-1] + tc/2)
        self.n = int(np.ceil((2*self.half_map)/c.cell_m))
        centres = self.lo + (np.arange(self.n) + .5)*c.cell_m
        tidx = np.clip(np.round((centres - terr.axis[0])/tc).astype(int), 0, len(terr.axis)-1)
        self.passable = ~terr.forbidden[np.ix_(tidx, tidx)]
        self.seen = np.zeros((self.n, self.n), bool)
        self.fp_along, self.fp_cross = self.search.fov_along_m, self.search.fov_cross_m
        self.h = self.search.bandwidth_m
        self.area_ref = c.area_ref_m2 or self.v*self.fp_cross*self.power.budget
        self._scen: Scenario | None = None
        self._uses = 0
        self.pos_cum = 0.0
        self.area_cum = 0.0
        self.fast = False      # path evaluation only: no maps, no reward zone / area (POS reward term only)
        self.skip_cells = False  # "lrc" in fast mode: no cell grid either (detection does not use it)
        if c.detect_model not in ("kernel", "lrc"):
            raise ValueError(c.detect_model)
        self.lrc = c.detect_model == "lrc"
        self.lrc_R = c.lrc_range_m if c.lrc_range_m is not None else self.fp_cross/2
        self.seg_t = np.concatenate([[0.0], (self.sample_idx + 1)*self.plant.h])   # path sample times in an interval

    # ------------------------------------------------------------------ scenario and reset
    def _next_scenario(self) -> Scenario:
        if self._scen is None or self._uses >= self.cfg.reuse:
            self._scen = make_scenario(self.sampler, int(self.rng.integers(2**62)))
            self._uses = 0
        self._uses += 1
        return self._scen

    def reset(self, seed: int | None = None, options: dict | None = None):
        """``options``: ``scenario`` (fixed scenario, e.g. evaluation), ``stage``, ``t0_s``, ``start_pose``
        (x, y, yaw) for stage 1/2."""
        options = options or {}
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scen = options.get("scenario") or self._next_scenario()
        self.scen = scen
        self.stage = int(options.get("stage", self.cfg.stage))
        self.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
        self._pass_p = np.zeros(self.belief.n)            # "lrc": LRC already applied in the current pass
        self._last_in = np.full(self.belief.n, -np.inf)   # "lrc": last time within range
        self.seen[:] = False
        self.pos_cum = self.area_cum = 0.0
        self._zone_key = None
        self.k = 0
        if self.stage in (1, 2):
            lo, hi = self.cfg.t0_s[self.stage]
            self.t = float(options.get("t0_s", self.rng.uniform(lo, hi)))
            pose = options.get("start_pose")
            if pose is None:
                pose = self._sample_start(self.t)
            self.plant.reset(pose)
            self.energy = self.power.budget - self.t*float(self.power.power(0.0, self.v))
            self.belief.update_exits(self.t)
            self._observe_now()
        else:
            t_launch = float(options.get("t_launch_s", 0.0))    # multi-UAV: staggered launch (default 0)
            self.t = t_launch
            self.plant.reset((*scen.start, np.deg2rad(scen.start_heading_deg)))
            self.energy = self.power.budget
            self.belief.update_exits(self.t)
            self._observe_now()
            t_plan = self.preset.handover.t_plan_s
            self.launch_track = [(self.t, *self.plant.pose[:2])]
            while self.t < t_launch + t_plan - 1e-9:
                bearing = np.arctan2(scen.lkp[1] - self.plant.pose[1], scen.lkp[0] - self.plant.pose[0])
                self.chi_tgt = self.plant.pose[2] + wrap(bearing - self.plant.pose[2])
                out = self._interval()
                self.launch_track.append((self.t, *self.plant.pose[:2]))
                if out["terminated"]:
                    break
        self.chi_tgt = float(self.plant.pose[2])
        self.k = 0
        self.pos_cum = self.area_cum = 0.0   # count only what the policy earns
        return self._obs(), self._info(dict(dpos=0.0, darea_m2=0.0, reward=0.0, terminated=False))

    def _sample_start(self, t):
        pts = self.belief.positions(t)[self.belief.active(t)]
        med = np.median(pts, axis=0)
        lim = self.half_map - self.cfg.map_margin_m
        for _ in range(1000):
            b = self.rng.uniform(-np.pi, np.pi)
            d = self.rng.uniform(*self.cfg.start_dist_m)
            xy = med + d*np.array([np.cos(b), np.sin(b)])
            if np.all(np.abs(xy) <= lim):
                break
        bearing = np.arctan2(med[1] - xy[1], med[0] - xy[0])
        yaw = bearing + np.deg2rad(self.rng.uniform(-self.cfg.heading_offset_deg, self.cfg.heading_offset_deg))
        return np.array([xy[0], xy[1], yaw])

    def _observe_now(self):
        """Observation at the current pose only (launch / curriculum start), no reward."""
        cells = self._cells_under(self.plant.pose[None, :])
        if self.lrc:
            xy = self.plant.pose[None, :2]
            self._detect_lrc(np.vstack([xy, xy]), np.array([self.t, self.t]))
        else:
            self._detect(cells, self.t)
        self.seen.flat[cells] = True
        self.belief.update_exits(self.t)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        a, shield = self._geofence(a)
        self.chi_tgt = self.chi_tgt + a*self.turn
        out = self._interval()
        out["shield"] = shield
        self.k += 1
        self.last_cells, self.last_t0 = out.pop("cells"), out.pop("t0")
        obs = None if self.fast else self._obs()
        return obs, out["reward"], out["terminated"], False, self._info(out)

    def _geofence(self, a: float):
        """Safety layer (6.2 v3): within ``geofence_m`` of the map edge, a course target that points outward
        is replaced by a full turn toward the map centre. Returns (action, shield active)."""
        g = self.cfg.geofence_m
        if g <= 0:
            return a, False
        x, y, _ = self.plant.pose
        lim = self.half_map - g
        out = np.array([x - np.clip(x, -lim, lim), y - np.clip(y, -lim, lim)])
        if not out.any():
            return a, False
        chi = self.chi_tgt + a*self.turn
        if np.cos(chi)*out[0] + np.sin(chi)*out[1] <= 0:     # target already points inward
            return a, False
        to_centre = np.arctan2(-y, -x)
        return float(np.sign(wrap(to_centre - self.chi_tgt)) or 1.0), True

    def _interval(self) -> dict:
        c = self.cfg
        t0 = self.t
        zone = None if self.fast else self._zone(t0)   # 0. fixed for this interval
        xy0 = self.plant.pose[:2].copy()
        hist = []
        phis = self.plant._integrate(self.chi_tgt, self.plant.n_inner, self.v, out=hist)   # 1-2
        self.plant.trace = (phis, np.full(len(phis), self.v), self.plant.h)
        self.energy -= self.power.step_energy(self.plant.trace)
        poses = np.array([hist[i][:3] for i in self.sample_idx])
        inside = self.terr.inside(poses[:, 0], poses[:, 1])
        exited = not inside.all()
        if exited:
            poses = poses[:np.argmin(inside)]          # observe until the exit
        cells = self._cells_under(poses) if len(poses) and not (self.lrc and self.skip_cells) else np.zeros(0, int)
        if self.lrc:                                   # 3 (SAROPS CPA + LRC)
            dpos = self._detect_lrc(np.vstack([xy0, poses[:, :2]]), t0 + self.seg_t[:len(poses) + 1])
        else:                                          # 3 (5.4 v3 kernel)
            dpos = self._detect(cells, t0)
        new = cells[~self.seen.flat[cells]]
        darea = 0.0 if self.fast else float(self._in_zone(new, zone).sum())*c.cell_m**2   # 4
        self.seen.flat[cells] = True
        reward = c.w_pos*dpos/c.pos_ref + c.w_area*darea/self.area_ref   # 5
        self.t = t0 + c.decision_dt_s
        self.belief.update_exits(self.t)               # 6
        self.pos_cum += dpos
        self.area_cum += darea
        terminated = exited or self.energy <= 0        # 7
        return dict(dpos=dpos, darea_m2=darea, reward=reward, terminated=terminated, exited=exited,
                    n_cells=len(cells), cells=cells, t0=t0)

    # ------------------------------------------------------------------ sensor
    def _cells_under(self, poses: np.ndarray) -> np.ndarray:
        """Flat indices of 25 m cells whose centre is inside any footprint (heading-aligned rectangle)."""
        c = self.cfg.cell_m
        r = 0.5*np.hypot(self.fp_along, self.fp_cross)
        m = int(np.ceil(r/c)) + 1
        out = []
        for x, y, yaw in poses:
            ix0 = int(np.floor((x - self.lo)/c)) - m
            iy0 = int(np.floor((y - self.lo)/c)) - m
            ii = np.arange(2*m + 1)
            ix, iy = np.clip(ix0 + ii, 0, self.n-1), np.clip(iy0 + ii, 0, self.n-1)
            gx = self.lo + (ix + .5)*c
            gy = self.lo + (iy + .5)*c
            X, Y = np.meshgrid(gx - x, gy - y)
            u = X*np.cos(yaw) + Y*np.sin(yaw)
            v = -X*np.sin(yaw) + Y*np.cos(yaw)
            ok = (np.abs(u) <= self.fp_along/2) & (np.abs(v) <= self.fp_cross/2)
            IY, IX = np.meshgrid(iy, ix, indexing="ij")
            out.append((IY*self.n + IX)[ok])
        return np.unique(np.concatenate(out)) if out else np.zeros(0, int)

    def _detect(self, cells: np.ndarray, t: float, w: np.ndarray | None = None) -> float:
        """w_i <- w_i (1 - POD q_i), q_i = Gaussian kernel mass of particle i over the cells (5.4 v3).
        Updates ``w`` in place (default: the belief weights) and returns the probability found."""
        if len(cells) == 0:
            return 0.0
        b, c, h = self.belief, self.cfg.cell_m, self.h
        w = b.w if w is None else w
        iy, ix = np.divmod(cells, self.n)
        cx, cy = self.lo + (ix + .5)*c, self.lo + (iy + .5)*c
        pts = b.positions(t)
        near = (w > 0) & (pts[:, 0] > cx.min() - 4*h) & (pts[:, 0] < cx.max() + 4*h) \
            & (pts[:, 1] > cy.min() - 4*h) & (pts[:, 1] < cy.max() + 4*h)
        idx = np.flatnonzero(near)
        if not len(idx):
            return 0.0
        d2 = (pts[idx, 0, None] - cx[None])**2 + (pts[idx, 1, None] - cy[None])**2
        q = np.minimum(np.exp(-d2/(2*h*h)).sum(1)*c*c/(2*np.pi*h*h), 1.0)
        before = w[idx].copy()
        w[idx] = before*(1 - self.search.pod*q)
        return float((before - w[idx]).sum()/b.n)

    def lrc_curve(self, d: np.ndarray) -> np.ndarray:
        """Lateral range curve: single-pass detection probability at closest-approach distance d (definite range)."""
        return np.where(d <= self.lrc_R, self.search.pod, 0.0)

    def _detect_lrc(self, xy: np.ndarray, ts: np.ndarray, w: np.ndarray | None = None, state=None) -> float:
        """SAROPS-style detection (Kratzke, Stone, Frost 2010, SIM): for each particle the closest point of approach
        (CPA) to each straight path segment (UAV and particle both moving linearly between samples), then the lateral
        range curve LRC(CPA distance). A pass is a run of segments within range (gaps shorter than ``pass_gap_s``
        merge); its failure factor is 1 - LRC(minimum distance in the pass), applied incrementally so the weight
        drops as soon as the particle comes close. ``xy`` (S, 2) UAV ground positions at times ``ts`` (S,).
        Updates ``w`` and the pass ``state`` (default: belief weights and env pass state) in place; returns the
        probability found."""
        b = self.belief
        w = b.w if w is None else w
        pp, last = (self._pass_p, self._last_in) if state is None else state
        R, gap = self.lrc_R, self.cfg.pass_gap_s
        reach = R + 15.0*(ts[-1] - ts[0]) + 50.0               # target speed cap 8.3 m/s, with room
        P0 = b.positions(ts[0])
        lo, hi = xy.min(0) - reach, xy.max(0) + reach
        idx = np.flatnonzero((w > 0) & (P0[:, 0] > lo[0]) & (P0[:, 0] < hi[0]) & (P0[:, 1] > lo[1]) & (P0[:, 1] < hi[1]))
        if not len(idx):
            return 0.0
        Q = np.stack([b.positions(t)[idx] for t in ts])        # (S, m, 2)
        before = w[idx].copy()
        ww, p, lt = w[idx], pp[idx], last[idx]
        for k in range(max(len(ts) - 1, 1)):
            k1 = min(k + 1, len(ts) - 1)
            r0, r1 = Q[k] - xy[k], Q[k1] - xy[k1]
            dr = r1 - r0
            den = (dr*dr).sum(1)
            s = np.clip(-(r0*dr).sum(1)/np.maximum(den, 1e-12), 0.0, 1.0)
            d = np.hypot(*(r0 + s[:, None]*dr).T)
            inside = d <= R
            if not inside.any():
                continue
            new = inside & (ts[k] - lt > gap)
            p[new] = 0.0
            q = self.lrc_curve(d)
            up = inside & (q > p)
            ww[up] *= (1.0 - q[up])/np.maximum(1.0 - p[up], 1e-12)
            p[up] = q[up]
            lt[inside] = ts[k1]
        w[idx], pp[idx], last[idx] = ww, p, lt
        return float((before - ww).sum()/b.n)

    def snapshot(self) -> dict:
        """Episode state for planning rollouts in fast mode (the seen-cell grid is not saved: it only feeds the
        area term, which fast mode skips)."""
        b = self.belief
        return dict(plant=self.plant.get_state(), chi=self.chi_tgt, t=self.t, k=self.k, energy=self.energy,
                    w=b.w.copy(), escaped=b.escaped, gone=b._gone.copy(), pos=self.pos_cum, area=self.area_cum,
                    pass_p=self._pass_p.copy(), last_in=self._last_in.copy())

    def restore(self, s: dict) -> None:
        b = self.belief
        self.plant.set_state(s["plant"])
        self.chi_tgt, self.t, self.k, self.energy = s["chi"], s["t"], s["k"], s["energy"]
        b.w[:], b.escaped, b._gone[:] = s["w"], s["escaped"], s["gone"]
        self.pos_cum, self.area_cum = s["pos"], s["area"]
        self._pass_p[:], self._last_in[:] = s["pass_p"], s["last_in"]

    def preview(self, actions) -> float:
        """Probability that the action sequence would find from the current state (POS term only, map exits
        ignored); the environment state is left unchanged. For lookahead planners."""
        st, chi, t = self.plant.get_state(), self.chi_tgt, self.t
        w = self.belief.w.copy()
        pst = (self._pass_p.copy(), self._last_in.copy())
        gain = 0.0
        for a in actions:
            chi = chi + float(np.clip(a, -1, 1))*self.turn
            xy0 = self.plant.pose[:2].copy()
            hist = []
            self.plant._integrate(chi, self.plant.n_inner, self.v, out=hist)
            poses = np.array([hist[i][:3] for i in self.sample_idx])
            poses = poses[self.terr.inside(poses[:, 0], poses[:, 1])]
            if len(poses):
                if self.lrc:
                    gain += self._detect_lrc(np.vstack([xy0, poses[:, :2]]), t + self.seg_t[:len(poses) + 1], w, pst)
                else:
                    gain += self._detect(self._cells_under(poses), t, w)
            t += self.cfg.decision_dt_s
        self.plant.set_state(st)
        return gain

    # ------------------------------------------------------------------ reward zone (1.1)
    def margin(self, t: float) -> float:
        return min(self.cfg.margin_rate*self.scen.speed_mps*t, self.cfg.margin_max_m)

    def _zone(self, t: float):
        """Expanded HDR of the unweighted prediction at t on the terrain grid: (mask, row0, col0)."""
        k = int(round(t/self.scen.save_dt_s))
        if self._zone_key == k:
            return self._zone_cache
        terr, tc = self.terr, self.tcell
        pts = np.asarray(self.scen.track[min(k, len(self.scen.track)-1)], float)
        act = (self.scen.exit_step < 0) | (self.scen.exit_step > k)
        pts = pts[act]
        if not len(pts):
            self._zone_cache = (np.zeros((1, 1), bool), 0, 0)
            self._zone_key = k
            return self._zone_cache
        nt = len(terr.axis)
        ij = np.clip(np.round((pts - terr.axis[0])/tc).astype(int), 0, nt-1)
        pad = int(np.ceil((4*self.h + self.margin(t))/tc)) + 1
        c0, c1 = max(ij[:, 0].min() - pad, 0), min(ij[:, 0].max() + pad + 1, nt)
        r0, r1 = max(ij[:, 1].min() - pad, 0), min(ij[:, 1].max() + pad + 1, nt)
        hist = np.zeros((r1 - r0, c1 - c0))
        np.add.at(hist, (ij[:, 1] - r0, ij[:, 0] - c0), 1.0)
        dens = gaussian_filter(hist, self.h/tc, mode="constant")
        forb = terr.forbidden[r0:r1, c0:c1]
        dens[forb] = 0
        v = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(v)
        thr = v[min(np.searchsorted(cum, self.cfg.hdr_level*cum[-1]), len(v)-1)]
        hdr = (dens >= thr) & (dens > 0)
        m = self.margin(t)
        zone = ((distance_transform_edt(~hdr)*tc <= m) & ~forb) if m > 0 else hdr
        self._zone_cache, self._zone_key = (zone, r0, c0), k
        # the same zone on the 25 m grid (crop), for the area channel
        c = self.cfg.cell_m
        y0, x0 = terr.axis[0] + (r0 - .5)*tc, terr.axis[0] + (c0 - .5)*tc
        iy0 = max(int(np.floor((y0 - self.lo)/c)), 0)
        ix0 = max(int(np.floor((x0 - self.lo)/c)), 0)
        iy1 = min(int(np.ceil((y0 + zone.shape[0]*tc - self.lo)/c)), self.n)
        ix1 = min(int(np.ceil((x0 + zone.shape[1]*tc - self.lo)/c)), self.n)
        gy = self.lo + (np.arange(iy0, iy1) + .5)*c
        gx = self.lo + (np.arange(ix0, ix1) + .5)*c
        X, Y = np.meshgrid(gx, gy)
        self._zone25 = (self._zone_at(X, Y, self._zone_cache) & self.passable[iy0:iy1, ix0:ix1], iy0, ix0)
        return self._zone_cache

    def _zone_at(self, x, y, zone) -> np.ndarray:
        mask, r0, c0 = zone
        col = np.round((x - self.terr.axis[0])/self.tcell).astype(int) - c0
        row = np.round((y - self.terr.axis[0])/self.tcell).astype(int) - r0
        ok = (row >= 0) & (row < mask.shape[0]) & (col >= 0) & (col < mask.shape[1])
        out = np.zeros(np.shape(x), bool)
        out[ok] = mask[row[ok], col[ok]]
        return out

    def _in_zone(self, cells: np.ndarray, zone) -> np.ndarray:
        iy, ix = np.divmod(cells, self.n)
        c = self.cfg.cell_m
        return self._zone_at(self.lo + (ix + .5)*c, self.lo + (iy + .5)*c, zone) & self.passable.flat[cells]

    # ------------------------------------------------------------------ observation (3.x)
    def map_specs(self):
        x, y = self.plant.pose[:2]
        ego = [(x, y, r/2) for r in self.cfg.ego_ranges_m]
        return ego + [(0.0, 0.0, self.half_map)]

    def _maps(self) -> np.ndarray:
        c, px = self.cfg, self.cfg.map_px
        self._zone(self.t)
        z25, zy0, zx0 = self._zone25
        elig_crop = z25 & ~self.seen[zy0:zy0+z25.shape[0], zx0:zx0+z25.shape[1]]
        b = self.belief
        act = b.w > 0
        pts, wts = b.positions(self.t)[act], b.w[act]/b.n
        out = np.zeros((4, 2, px, px), np.float32)
        for m, (cx, cy, half) in enumerate(self.map_specs()):
            cell = 2*half/px
            # channel 0: share of the map cell that is unseen, passable and inside the reward zone
            ns = int(np.clip(round(cell/c.cell_m), 1, c.max_sub))
            s = (np.arange(px*ns) + .5)*(cell/ns) - half
            ix = np.floor((cx + s - self.lo)/c.cell_m).astype(int) - zx0
            iy = np.floor((cy + s - self.lo)/c.cell_m).astype(int) - zy0
            okx = (ix >= 0) & (ix < elig_crop.shape[1])
            oky = (iy >= 0) & (iy < elig_crop.shape[0])
            elig = np.zeros((len(s), len(s)), bool)
            if okx.any() and oky.any():
                elig[np.ix_(oky, okx)] = elig_crop[np.ix_(iy[oky], ix[okx])]
            out[m, 0] = elig.reshape(px, ns, px, ns).mean((1, 3))
            # channel 1: remaining (not found, in map) probability per map cell, KDE-smoothed (3.4 v2)
            sig = self.h/cell
            pad = int(np.ceil(3*sig)) if sig > 0.3 else 0
            edges = cx - half - pad*cell + np.arange(px + 2*pad + 1)*cell
            edges_y = cy - half - pad*cell + np.arange(px + 2*pad + 1)*cell
            H, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[edges_y, edges], weights=wts)
            if pad:
                H = gaussian_filter(H, sig, mode="constant")[pad:pad+px, pad:pad+px]
            scale = 1e6/cell**2 if c.pos_scale == "density" else c.pos_scale[m]
            out[m, 1] = H*scale
        return out

    def vector(self) -> np.ndarray:
        x, y, yaw = self.plant.pose
        p = self.plant
        return np.array([x/self.half_map, y/self.half_map, np.cos(yaw), np.sin(yaw), p.phi/p.phi_max,
                         max(self.energy, 0.0)/self.power.budget, wrap(self.chi_tgt - yaw)/np.pi,
                         p.r/p.r_max], np.float32)

    def _obs(self) -> dict:
        maps = self._maps()
        obs = {name: maps[i] for i, name in enumerate(MAP_NAMES)}
        obs["vec"] = self.vector()
        return obs

    def _info(self, out: dict) -> dict:
        b = self.belief
        out.setdefault("shield", False)
        return dict(t_s=self.t, k=self.k, pos=b.pos, pos_policy=self.pos_cum, remaining=b.mass,
                    escaped=b.escaped, area_policy_km2=self.area_cum/1e6, energy_left=self.energy,
                    gap_deg=float(np.degrees(wrap(self.chi_tgt - self.plant.pose[2]))),
                    roll_deg=float(np.degrees(self.plant.phi)), stage=self.stage, **out)
