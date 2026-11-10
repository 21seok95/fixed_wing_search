"""RL environment on the real-region map (docs/rl_mdp_design.md, agreed with the user item by item on 2026-11-10).

The flight, the camera, the detections, the A score and the edge mask are the SAROPS reference env itself
(``web_optimize.team_env`` with ``rl_eval.sarops_ref.settings``: Amboy, 33 m/s, 49 deg per 5 s decision, oblique camera with the
gimbal facing forward, passes of 0.7 x the model POD 5 s apart, A + w POS with x 70, w 1, edge band 500 m). This module adds what the
policy needs on top:

- observation (sections 3-1 to 3-5): four UAV-centred maps with the heading up, 64 x 64 cells over 1.6 / 6.4 / 25.6 / 51.2 km, each with
  channels (1) remaining POS value now, (2) the same weights at the predicted positions 15 s / 1 min / 5 min / 10 min ahead, (3) A value
  (what sweeping the cell now adds to A), (4) ground visibility (-1 off the terrain); the 1.6 km map has (5) the image-position factor of
  the forward camera footprint. (1)-(3) are in particle units (an unsearched particle = 1, scaled to 5000 particles) and pass through
  log(1 + value); A is in the same units (A gain x 5000). Vector of 8: roll, guidance turn-rate state, course target minus heading,
  energy left, time left in the current 5 min window, probability left in the map, allowed action range (low, high).
- reward (section 4): (dA + w dPOS) x 100 per 5 s step, from the same AreaScore and belief as SAROPS (planner particles).
- starts (section 6): training scenarios drawn like the validation cases (LKP on a road in the central 3 km square, rl_eval_set) but
  with their own seeds, reused for ``reuse`` episodes with new starts; curriculum stage 1 / 2 / 3 starts mixed by ``stage_mix``.
  ``reset(options=case.options(scen))`` starts exactly like a SAROPS reference flight.
- end: energy spent = terminated (60 min flight); ``max_steps`` cuts an episode early = truncated.

The legacy env and its observation and reward (``rl_env.RLSearchEnv``, ``rl_gym``) are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import uniform_filter

from rl_search.area_score import CELL_M, N_WINDOWS, WINDOW_S, AreaScore
from rl_search.plants import wrap

MAPS = (("m1600", 1600.0, 15.0), ("m6400", 6400.0, 60.0), ("m25600", 25600.0, 300.0), ("m51200", 51200.0, 600.0))
N_REF = 5000                       # particle units are scaled to this many particles
VEC_NAMES = ("roll", "turn_rate_state", "course_minus_heading", "energy_left", "window_time_left", "prob_left", "allowed_low",
             "allowed_high")
STAGE_MIX = {1: {1: 1.0}, 2: {2: 0.75, 1: 0.25}, 3: {3: 0.7, 2: 0.15, 1: 0.15}}   # default start mix by curriculum stage


_STATIC: dict = {}


def preload(region: str = "amboy") -> dict:
    """Road network, terrain (the network's own, so it is loaded once) and scenario sampler of a region, cached per process. The terrain
    alone is about 0.5 GB; call this in the parent before starting the env workers with the "fork" start method, and the workers share
    these read-only arrays instead of loading two terrain copies each."""
    if region not in _STATIC:
        from rl_search.real_scenario import chungju_preset
        from rl_search.scenario import ScenarioSampler
        from rl_search.vehicle import load_net
        net = load_net(region)
        _STATIC[region] = dict(net=net, terr=net.terr, sampler=ScenarioSampler(net.terr, chungju_preset().scenario))
    return _STATIC[region]


@dataclass
class RealRLConfig:
    region: str = "amboy"
    map_px: int = 64
    maps: tuple = MAPS                 # (name, side in m, prediction lead in s) per map
    reward_scale: float = 100.0
    particles: int | None = None       # training particle count (None: the validation config, 5000)
    reuse: int = 4                     # episodes per generated scenario; each draws a new start
    stage: int = 1                     # curriculum stage
    stage_mix: dict | None = None      # {start stage: share}; None: STAGE_MIX[stage]
    seed_base: int = 10_000_000        # training scenario seeds (validation 70000+, test 80000+, hold-out + 1 000 000)
    max_steps: int | None = None       # cut episodes after this many decisions (truncated); None: until the energy is spent
    settings: dict = field(default_factory=dict)    # overrides of the SAROPS reference settings (web_optimize FIELDS)


class RealRLEnv:
    """One UAV on the real-region map. ``reset(seed, options)`` -> (obs, info); ``step(action)`` -> (obs, reward, terminated, truncated,
    info). ``options``: None for a curriculum start, or ``rl_eval_set.Case.options(scen)`` / dict(scenario=..., stage=..., ...) as for
    ``RLSearchEnv.reset``."""

    def __init__(self, cfg: RealRLConfig | None = None, seed: int = 0, terr=None):
        from rl_search.real_scenario import VehicleScenarios
        from rl_search.rl_eval_set import vehicle_config
        from rl_search.scripts.real_map.web_optimize import team_env
        from rl_search.scripts.rl_eval.sarops_ref import settings
        self.cfg = cfg = cfg or RealRLConfig()
        st = preload(cfg.region)
        self.terr = st["terr"] if terr is None else terr
        self.s = settings() | cfg.settings
        self.env = team_env(self.s, cfg.region, self.terr, seed, area=True)        # the SAROPS reference env (AreaEnv)
        self.env.pos_w, self.env.area_w = float(self.s["area_w"]), 1.0
        vc = vehicle_config()
        if cfg.particles is not None:
            vc = replace(vc, particles=int(cfg.particles))
        self.vs = VehicleScenarios(st["sampler"], vc, region=cfg.region, net=st["net"])
        self.rng = np.random.default_rng(seed)
        self.half = float(self.terr.axis[-1])
        self._scen = self._area = None
        self._uses = 0
        self._area_cache = {}
        self._prepare_static()

    # ------------------------------------------------------------------ static parts of the observation
    def _prepare_static(self):
        px = self.cfg.map_px
        det = self.env.detector
        self.vis_grids = []                                     # (stride, grid of cell means) per map
        for _, side, _ in self.cfg.maps:
            f = max(int(round(side/px/self.terr.cell)), 1)
            g = uniform_filter(det.A.astype(np.float32), size=f, mode="nearest")[::f, ::f] if f > 1 else det.A.astype(np.float32)
            self.vis_grids.append((f, g))
        # map frame: row 0 = farthest ahead, column 0 = farthest left; centre offsets (forward, right) per cell
        self.offsets = []
        for _, side, _ in self.cfg.maps:
            cell = side/px
            fwd = side/2 - (np.arange(px) + .5)*cell
            right = (np.arange(px) + .5)*cell - side/2
            self.offsets.append((fwd[:, None]*np.ones((1, px)), np.ones((px, 1))*right[None, :]))
        self.footprint = self._footprint_channel()

    def _footprint_channel(self) -> np.ndarray:
        """Image-position factor of the forward camera (level flight) on the 1.6 km map: 1 in the image centre, 0.5 at an edge, 0.25 in a
        corner, 0 outside (detection.DetectionConfig.edge_factor)."""
        from rl_search.detection import camera_frame, image_coords
        cfg = self.env.detector.cfg
        fwd, right = self.offsets[0]
        frame = camera_frame(np.pi/2, 0.0, cfg)                 # heading north: forward = +y, right = +x
        uu, vv, front = image_coords(right.ravel(), fwd.ravel(), 0.0, 0.0, frame, cfg)
        inside = front[0] & (np.abs(uu[0]) <= 1) & (np.abs(vv[0]) <= 1)
        return np.where(inside, cfg.edge_factor(uu[0], vv[0]), 0.0).reshape(fwd.shape).astype(np.float32)

    # ------------------------------------------------------------------ scenarios and starts
    def _new_scenario(self):
        from rl_search.rl_eval_set import center_lkp
        seed = self.cfg.seed_base + int(self.rng.integers(0, 10**9))
        r = np.random.default_rng(seed)
        lkp = center_lkp(self.vs.net, r)
        heading = float(r.uniform(0.0, 360.0))
        scen = self.vs.make_at(seed, lkp, heading, start=(0.0, 0.0))
        return scen

    def _area_score(self, scen) -> AreaScore:
        key = id(scen.track)                                    # stage 3 starts replace the scenario but keep its track
        if key not in self._area_cache:
            self._area_cache = {key: AreaScore(self.terr, self.env.h, scen.track, scen.save_dt_s, float(self.s["area_x"]), self.env.lrc_R)}
        a = self._area_cache[key]
        a.S[:] = False
        a.log.clear()
        a.committed.clear()
        return a

    def curriculum_options(self, scen, stage: int) -> dict:
        from rl_search.rl_eval_set import start_stage1, start_stage2, start_stage3
        if stage == 1:
            t0, start, info = start_stage1(scen, self.rng, self.half)
        elif stage == 2:
            t0, start, info = start_stage2(scen, self.rng, self.half)
        else:
            t0, start, info = start_stage3(np.asarray(scen.lkp, float), self.rng, self.half)
            scen = replace(scen, start=np.array(start[:2], float), start_heading_deg=float(np.degrees(start[2])))
            return dict(scenario=scen, stage=3, start_info=info)
        return dict(scenario=scen, stage=stage, t0_s=t0, start_pose=np.array(start, float), start_info=info)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options is None:
            if self._scen is None or self._uses >= self.cfg.reuse:
                self._scen, self._uses = self._new_scenario(), 0
            self._uses += 1
            mix = self.cfg.stage_mix or STAGE_MIX[self.cfg.stage]
            stages = sorted(mix)
            st = int(self.rng.choice(stages, p=np.array([mix[k] for k in stages], float)/sum(mix.values())))
            options = self.curriculum_options(self._scen, st)
        scen = options["scenario"]
        env = self.env
        env.area = self._area = self._area_score(scen)         # attached before the reset, as in the SAROPS runs (hand-over sweeps)
        env.reset(options={k: v for k, v in options.items() if k != "start_info"})
        self.start_stage = int(options.get("stage", 3))
        self.start_info = options.get("start_info", {})
        self.steps = 0
        self.A_cum = self.pos_cum = 0.0
        self._window = None
        return self.observation(), self._info(0.0, 0.0, False)

    # ------------------------------------------------------------------ step
    def step(self, action):
        env = self.env
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        _, _, terminated, _, info = env.step([a])
        d_area = float(getattr(env, "_area_gain", 0.0))
        d_total = float(info["dpos"])                          # AreaEnv: pos_w POS + area_w A of this step
        d_pos = (d_total - env.area_w*d_area)/env.pos_w
        self.A_cum += d_area
        self.pos_cum += d_pos
        self.steps += 1
        truncated = bool(not terminated and self.cfg.max_steps is not None and self.steps >= self.cfg.max_steps)
        reward = self.cfg.reward_scale*d_total
        return self.observation(), reward, bool(terminated), truncated, self._info(d_area, d_pos, info["shield"])

    def _info(self, d_area, d_pos, shield) -> dict:
        env = self.env
        return dict(t_s=env.t, dA=d_area, dPOS=d_pos, A=self.A_cum, POS=self.pos_cum, total=self.A_cum + env.pos_w*self.pos_cum,
                    shield=bool(shield), energy_left=env.energy, stage=self.start_stage, steps=self.steps)

    # ------------------------------------------------------------------ observation
    def _bin(self, pts, vals, k):
        """Sum ``vals`` of world points ``pts`` (M, 2) into the cells of map k (heading up)."""
        px = self.cfg.map_px
        side = self.cfg.maps[k][1]
        cell = side/px
        x, y, yaw = self.env.plant.pose
        d = pts - np.array([x, y])
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = d[:, 0]*c + d[:, 1]*s
        right = d[:, 0]*s - d[:, 1]*c
        row = np.floor((side/2 - fwd)/cell).astype(np.int64)
        col = np.floor((right + side/2)/cell).astype(np.int64)
        ok = (row >= 0) & (row < px) & (col >= 0) & (col < px)
        return np.bincount(row[ok]*px + col[ok], weights=vals[ok], minlength=px*px).reshape(px, px)

    def _area_cells(self):
        """A value per unswept 25 m cell for the current window, in A units: (centres (M, 2), values (M,))."""
        a = self._area
        k = a.window(self.env.t)
        if k > N_WINDOWS:
            return np.zeros((0, 2)), np.zeros(0)
        if self._window != k:
            v = a.T[k - 1][a.B]
            iy, ix = np.nonzero(v > 0)
            self._window = k
            self._cells = (iy, ix, np.stack([a.gx[ix], a.gy[iy]], 1), v[iy, ix])
        iy, ix, xy, v = self._cells
        open_ = ~a.S[iy, ix]
        return xy[open_], v[open_]

    def _area_at(self, wx, wy) -> np.ndarray:
        """A value of the 25 m cells under world points (same-resolution map: read, not binned, to avoid rotation aliasing)."""
        a = self._area
        k = a.window(self.env.t)
        v = np.zeros(wx.shape)
        if k > N_WINDOWS:
            return v
        ix = np.floor((wx - a.x0)/CELL_M).astype(np.int64)
        iy = np.floor((wy - a.y0)/CELL_M).astype(np.int64)
        ok = (ix >= 0) & (ix < a.B.shape[1]) & (iy >= 0) & (iy < a.B.shape[0])
        v[ok] = a.T[k - 1][a.B[iy[ok], ix[ok]]]*~a.S[iy[ok], ix[ok]]
        return v

    def observation(self) -> dict:
        env, cfg = self.env, self.cfg
        b, det = env.belief, env.detector
        t = env.t
        x, y, yaw = env.plant.pose
        unit = N_REF/b.n*env.pos_w                           # particle units, scaled to N_REF particles, times w
        live = b.w > 0
        P = b.positions(t)[live]
        w = b.w[live]
        val_now = w*det.visibility(P[:, 0], P[:, 1])*unit
        a_xy, a_val = self._area_cells()
        a_val = a_val*N_REF                                   # A gain in the same units as one particle (1 / N_REF of POS)
        c, s = np.cos(yaw), np.sin(yaw)
        obs = {}
        for k, (name, side, lead) in enumerate(cfg.maps):
            ch = [np.log1p(self._bin(P, val_now, k))]
            fut = live & b.active(t + lead)
            Pf = b.positions(t + lead)[fut]
            ch.append(np.log1p(self._bin(Pf, b.w[fut]*det.visibility(Pf[:, 0], Pf[:, 1])*unit, k)))
            fwd, right = self.offsets[k]
            wx, wy = x + fwd*c + right*s, y + fwd*s - right*c
            if side/cfg.map_px <= CELL_M:                     # 1.6 km map: one 25 m cell per map cell
                ch.append(np.log1p(self._area_at(wx, wy)*N_REF))
            else:                                             # larger cells: sum of the 25 m cells inside
                ch.append(np.log1p(self._bin(a_xy, a_val, k)))
            f, g = self.vis_grids[k]
            step = self.terr.cell*f
            col = np.rint((wx - self.terr.axis[0])/step).astype(int)
            row = np.rint((wy - self.terr.axis[0])/step).astype(int)
            inside = (np.abs(wx) <= self.half) & (np.abs(wy) <= self.half)
            vis = np.full(wx.shape, -1.0, np.float32)
            vis[inside] = g[np.clip(row[inside], 0, g.shape[0] - 1), np.clip(col[inside], 0, g.shape[1] - 1)]
            ch.append(vis)
            if k == 0:
                ch.append(self.footprint)
            obs[name] = np.stack(ch).astype(np.float32)
        obs["vec"] = self.vector()
        return obs

    def vector(self) -> np.ndarray:
        env = self.env
        p = env.plant
        lo, hi, _ = env.allowed_range()
        k = self._area.window(env.t) if self._area is not None else N_WINDOWS + 1
        left = (k*WINDOW_S - env.t)/WINDOW_S if k <= N_WINDOWS else 0.0
        return np.array([p.phi/p.phi_max, p.r/p.r_max, wrap(env.chi_tgt - p.pose[2])/np.pi, max(env.energy, 0.0)/env.power.budget,
                         left, env.belief.mass, lo, hi], np.float32)


def make_gym(cfg: RealRLConfig | None = None, seed: int = 0):
    """Gymnasium wrapper for SB3: Dict observation of the four maps and the vector, action Box(-1, 1, (1,))."""
    import gymnasium as gym
    from gymnasium import spaces

    class RealRLGym(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.env = RealRLEnv(cfg, seed)
            c = self.env.cfg
            px = c.map_px
            maps = {name: spaces.Box(-1.0, np.inf, (5 if k == 0 else 4, px, px), np.float32) for k, (name, _, _) in enumerate(c.maps)}
            self.observation_space = spaces.Dict({**maps, "vec": spaces.Box(-2.0, 2.0, (len(VEC_NAMES),), np.float32)})
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            return self.env.step(action)

    return RealRLGym()
