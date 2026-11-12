"""RL environment on the real-region map (docs/rl_mdp_design.md, agreed with the user item by item on 2026-11-10).

The flight, the camera, the detections, the A score and the edge mask are the SAROPS reference env itself
(``web_optimize.team_env`` with ``rl_eval.sarops_ref.settings``: Amboy, 33 m/s, 49 deg per 5 s decision, oblique camera with the
gimbal facing forward, passes of 0.7 x the model POD 5 s apart, A + w POS with x 70, w 1, edge band 500 m). This module adds what the
policy needs on top:

- observation (sections 3-1 to 3-5): four UAV-centred maps with the heading up, 64 x 64 cells over 1.6 / 6.4 / 25.6 / 51.2 km, each with
  channels (1) remaining POS value now, (2) the same weights at the predicted positions 15 s / 1 min / 5 min / 10 min ahead, (3) A value
  (what sweeping the cell now adds to A), (4) ground visibility (-1 off the terrain); the 1.6 km map has (5) the image-position factor of
  the forward camera footprint. (1)-(3) are in particle units (an unsearched particle = 1, scaled to 5000 particles) and pass through
  log(1 + value); A is in the same units (A gain x 5000). With the setting ``rel_maps`` (user 2026-11-12) every map gets a last channel
  (6) the remaining POS now as a share of the remaining belief (rescaled to 5000 particles), so the end of a flight or a searched-down
  belief reads as clearly as a fresh one. With ``fp_rate`` (user 2026-11-12) channel (5) is multiplied by the detection's turn-rate
  factor at the current roll (1 in level flight, 0.49 at 10 deg, 0.14 at 20 deg with D4). Vector of 8: roll, guidance turn-rate state, course target minus heading,
  energy left, time left in the current 5 min window, probability left in the map, allowed action range (low, high).
- A (settings ``area_def``): "window" = the 5 min window A of the first training (amboy_v1); "realtime" = ``area_score.RealtimeArea``
  (the same A with its region every 10 s, user 2026-11-11): channel (3) is what sweeping each unswept cell of the current region adds to A;
  "seen" = ``area_score.SeenArea`` (option of 2026-11-11): channel (3) is what seeing each unseen particle now adds, at the particle. With
  either new A, vector entry 4 is the share of the current region already covered (instead of the window time left). Shapes are
  unchanged, so a window-A model can be fine-tuned.
- reward (section 4): (dA + w dPOS) x 100 per 5 s step, from the same area score and belief as SAROPS (planner particles), plus the
  straight-flight bonus ``straight_bonus`` x (1 - |a|) of the applied action (user, 2026-11-11: 0 = keep the course; training only,
  reported apart as ``bonus``, never part of A or POS), minus ``roll_reversal_cost`` per roll reversal (user, 2026-11-11, B2 in
  docs/rl_experiments.md): the roll swings from beyond +``roll_reversal_deg`` to beyond -``roll_reversal_deg`` or back, counted on the 1 s
  samples; the last side is kept across steps (the roll itself is in the observation vector). Training only, reported as ``roll_rev``.
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

from rl_search.area_score import CELL_M, N_WINDOWS, WINDOW_S, AreaScore, RealtimeArea, SeenArea
from rl_search.plants import wrap

MAPS = (("m1600", 1600.0, 15.0), ("m6400", 6400.0, 60.0), ("m25600", 25600.0, 300.0), ("m51200", 51200.0, 600.0))
POOL_DIR = "rl_search/data/real/amboy/rl_pool"
POOL_DIR_DIVERSE = "rl_search/data/real/amboy/rl_pool_diverse"
POOL_DIR_MULTI = "rl_search/data/real/amboy/rl_pool_multi"      # typed scenarios, files <type>_s<seed>.pkl


def pool_dir_multi(settings: dict | None = None) -> str:
    """Typed pool of these settings: the same scenarios with the region of the off-road sigma (``<POOL_DIR_MULTI>_o150``) when it is set,
    so a running training on the plain pool is never handed regions of another definition."""
    off = float((settings or {}).get("area_sigma_off") or 0.0)
    return POOL_DIR_MULTI + (f"_o{off:g}" if off else "")
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
    pool: str | None = POOL_DIR        # pre-built training scenarios (scripts/rl_eval/make_pool.py); None or empty: generate each time
    max_steps: int | None = None       # cut episodes after this many decisions (truncated); None: until the energy is spent
    settings: dict = field(default_factory=dict)    # overrides of the SAROPS reference settings (web_optimize FIELDS)
    diverse: bool = False              # training scenarios and stage 3 starts by rl_eval_set.DIVERSE (2026-11-11); pool: POOL_DIR_DIVERSE
    straight_bonus: float = 0.0        # reward per step for flying straight: bonus x (1 - |a|), a = applied action (after the edge mask)
    roll_reversal_cost: float = 0.0    # reward taken per roll reversal (B2)
    mask_cost: float = 0.0             # reward taken per unit the edge mask cut off the policy's action (user 2026-11-11: no edge riding)
    straight_on_raw: bool = False      # straight bonus from the policy's own action (not the masked one; R2 on)
    roll_reversal_deg: float = 10.0    # a side counts once |roll| reaches this (10 deg = 3 deg/s turn at 33 m/s)
    multi: bool = False                # typed training scenarios (rl_eval_set.MULTI, R5, user 2026-11-12); pool: POOL_DIR_MULTI
    type_weights: str | None = None    # JSON {type: share} the trainer updates; read at every new scenario (None or missing: equal)


def training_scenario(vs, seed: int):
    """Training scenario of ``seed`` (``rl_eval_set`` LKP rule: a road cell in the central 3 km square, heading uniform)."""
    from rl_search.rl_eval_set import center_lkp
    r = np.random.default_rng(seed)
    lkp = center_lkp(vs.net, r)
    heading = float(r.uniform(0.0, 360.0))
    return vs.make_at(seed, lkp, heading, start=(0.0, 0.0))


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
        self.env._obs = lambda: None          # this instance only: skip the legacy maps that RLSearchEnv.reset builds (40-80 ms, unused here)
        vc = vehicle_config()
        if cfg.particles is not None:
            vc = replace(vc, particles=int(cfg.particles))
        self.vs = VehicleScenarios(st["sampler"], vc, region=cfg.region, net=st["net"])
        self.rng = np.random.default_rng(seed)
        self.half = float(self.terr.axis[-1])
        self._scen = self._area = None
        self._uses = 0
        self._area_cache = {}
        big = float(self.s.get("big_map_m", 0.0) or 0.0)             # (user 2026-11-12) side of the largest map; same key and 64 x 64 cells
        if big > 0:
            maps = list(cfg.maps)
            maps[-1] = (maps[-1][0], big, maps[-1][2])
            cfg.maps = tuple(maps)
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
    def _pool_files(self) -> list:
        if not hasattr(self, "_pool"):
            from pathlib import Path
            d = Path(self.cfg.pool) if self.cfg.pool else None
            self._pool = sorted(d.glob("*_s*.pkl" if self.cfg.multi else "s*.pkl")) if d is not None and d.is_dir() else []
        return self._pool

    def _pick_type(self) -> str:
        """Target type of the next training scenario, by the trainer's current shares (equal when there are none yet)."""
        import json
        from pathlib import Path
        from rl_search.rl_eval_set import MULTI
        types = MULTI["types"]
        w = np.ones(len(types))
        p = Path(self.cfg.type_weights) if self.cfg.type_weights else None
        if p is not None and p.exists():
            try:
                d = json.loads(p.read_text())
                w = np.array([float(d.get(t, 1.0)) for t in types])
            except (ValueError, OSError):
                pass
        return types[int(self.rng.choice(len(types), p=w/w.sum()))]

    @property
    def area_def(self) -> str:
        return self.s.get("area_def", "window")

    @property
    def seen_mode(self) -> bool:
        return self.area_def == "seen"

    def _new_scenario(self):
        files = self._pool_files()
        self._typ = self._pick_type() if self.cfg.multi else ""
        if self.cfg.multi:
            files = [f for f in files if f.name.startswith(self._typ + "_")]
        if files:                                               # pre-built: the scenario and its area score
            import pickle
            with open(files[int(self.rng.integers(len(files)))], "rb") as f:
                d = pickle.load(f)
            scen = d["scen"]
            a = d.get({"seen": "seen", "realtime": "realtime", "realtime_q": "realtime", "realtime_t": "realtime"}.get(self.area_def, "area"))
            if isinstance(a, RealtimeArea):
                self._rt_mode(a)
            ok = a is not None and (self.area_def == "window" or (a.x == float(self.s["area_x"])
                                                                and (not self.seen_mode or a.pod == float(self.s["seen_pod"]))
                                                                and (not isinstance(a, RealtimeArea) or (a.h == self.area_sigma and
                                                                     float(getattr(a, "h_off", 0.0) or 0.0) == self.area_sigma_off))))
            # a pool region built with another sigma is rebuilt for this scenario by _area_score (a few seconds, once per scenario)
            self._area_cache = {id(scen.track): a} if ok else {}
            return scen
        seed = self.cfg.seed_base + int(self.rng.integers(0, 10**9))
        if self.cfg.multi:
            from rl_search.rl_eval_set import multi_training_scenario
            return multi_training_scenario(self.vs, seed, self._typ)[0]
        if self.cfg.diverse:
            from rl_search.rl_eval_set import diverse_training_scenario
            return diverse_training_scenario(self.vs, seed)[0]
        return training_scenario(self.vs, seed)

    @property
    def area_sigma(self) -> float:
        """Blur width (m) of the real-time A region: setting ``area_sigma`` (user 2026-11-12: 75 m for R4), else the env's 150 m."""
        from rl_search.scripts.real_map.web_optimize import area_sigma
        return area_sigma(self.s, self.env.h)

    @property
    def area_sigma_off(self) -> float:
        """Blur width (m) of the off-road particles (setting ``area_sigma_off``, user 2026-11-12: 150 with 75 on the roads); 0 = the same."""
        from rl_search.scripts.real_map.web_optimize import area_sigma_off
        off = area_sigma_off(self.s)
        return off if off != self.area_sigma else 0.0

    def _rt_mode(self, a) -> None:
        """Sweep rule of a RealtimeArea for this env's A definition: A3 (any image), A4 (image quality), A4t (steady image only)."""
        a.quality = self.area_def == "realtime_q"
        roll = float(self.s.get("area_roll_max", 20.0)) if self.area_def == "realtime_t" else None
        a.roll_max_deg = roll
        a.rate_max_dps = None if roll is None else float(np.degrees(9.80665*np.tan(np.radians(roll))/self.env.v))
        a.img_min = float(self.s.get("area_img_min", 0.0) or 0.0)      # image-position gate of a searched cell (user 2026-11-12)

    def _area_score(self, scen):
        key = id(scen.track)                                    # stage 3 starts replace the scenario but keep its track
        if key not in self._area_cache:
            if self.seen_mode:
                a = SeenArea(scen.track, scen.exit_step, scen.save_dt_s, float(self.s["area_x"]), float(self.s["seen_pod"]))
            elif self.area_def in ("realtime", "realtime_q", "realtime_t"):
                from rl_search.scripts.real_map.web_optimize import realtime_kw
                a = RealtimeArea(self.terr, self.area_sigma, scen.track, scen.exit_step, scen.save_dt_s, float(self.s["area_x"]),
                                 **realtime_kw(self.s, self.cfg.region))
                self._rt_mode(a)
            else:
                a = AreaScore(self.terr, self.env.h, scen.track, scen.save_dt_s, float(self.s["area_x"]), self.env.lrc_R)
            self._area_cache = {key: a}
        a = self._area_cache[key]
        if isinstance(a, (SeenArea, RealtimeArea)):
            a.clear()
        else:
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
            from rl_search.rl_eval_set import DIVERSE, MULTI
            rule = MULTI if self.cfg.multi else DIVERSE if self.cfg.diverse else None
            ring = rule["ring_m"] if rule else (8000.0, 20000.0)
            t0, start, info = start_stage3(np.asarray(scen.lkp, float), self.rng, self.half, ring)
            scen = replace(scen, start=np.array(start[:2], float), start_heading_deg=float(np.degrees(start[2])))
            out = dict(scenario=scen, stage=3, start_info=info)
            if rule:                                            # launch 0-20 min (diverse) or 0-40 min (typed) after the report
                out["t_launch_s"] = float(self.rng.uniform(*rule["delay_s"]))
            return out
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
        if isinstance(self._area, RealtimeArea):
            from rl_search.rl_eval_set import flight_window
            self._area.set_window(*flight_window(options))      # from the launch (or start) to the end of the flight
        env.reset(options={k: v for k, v in options.items() if k != "start_info"})
        self.start_stage = int(options.get("stage", 3))
        self.start_info = options.get("start_info", {})
        self.steps = 0
        self.A_cum = self.pos_cum = self.bonus_cum = self.mask_cum = 0.0
        self.roll_rev = 0
        phi = float(env.plant.phi) if hasattr(env.plant, "phi") else 0.0
        th = np.radians(self.cfg.roll_reversal_deg)
        self._roll_side = (1 if phi > 0 else -1) if abs(phi) >= th else 0
        self._window = None
        self.transit_actions, self.transit_track = [], []
        self.transit_start = (env.t, float(env.plant.pose[0]), float(env.plant.pose[1]))
        hk = float(self.s.get("handover_km", 0.0) or 0.0)
        if hk > 0 and self.start_stage == 3:
            self._transit(hk*1000.0, float(self.s.get("transit_max_s", 1800.0)))
        return self.observation(), self._info(0.0, 0.0, False)

    def _transit(self, dist_m: float, max_s: float) -> None:
        """Distance hand-over (user 2026-11-12, instead of handing over 2 min after the launch whatever the distance): fly by rule toward
        the nearest quarter of the remaining belief (its weighted centre, recomputed every 5 s decision) until 10 % of the remaining
        belief is within ``dist_m`` or ``max_s`` has passed; then the policy takes over. The rule's actions are kept (``transit_actions``)
        so the hold-out replay flies the same path; what these steps find counts in the flight's A and POS."""
        env = self.env
        t_end = env.t + max_s
        while env.t < t_end - 1e-9:
            b = env.belief
            live = b.w > 0
            if not live.any():
                break
            P, w = b.positions(env.t)[live], b.w[live]
            x, y = env.plant.pose[:2]
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            o = np.argsort(d)
            cw = np.cumsum(w[o])/w.sum()
            if d[o][min(int(np.searchsorted(cw, 0.10)), len(o) - 1)] <= dist_m:
                break
            near = o[:min(int(np.searchsorted(cw, 0.25)), len(o) - 1) + 1]
            tx, ty = np.average(P[near], axis=0, weights=w[near])
            a = float(np.clip(wrap(np.arctan2(ty - y, tx - x) - env.chi_tgt)/env.turn, -1.0, 1.0))
            _, term, trunc, info = self._advance(np.array([a]))
            self.transit_actions.append(a)
            self.transit_track.append((env.t, float(env.plant.pose[0]), float(env.plant.pose[1]), float(info["POS"])))
            if term or trunc:
                break

    # ------------------------------------------------------------------ step
    def step(self, action):
        reward, terminated, truncated, info = self._advance(action)
        return self.observation(), reward, terminated, truncated, info

    def _advance(self, action):
        """One 5 s decision without building the observation (the rule's transit uses it directly)."""
        env = self.env
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        _, _, terminated, _, info = env.step([a])
        cut = abs(a - float(getattr(env, "a_applied", a)))       # how much the edge mask cut off
        d_area = float(getattr(env, "_area_gain", 0.0))
        d_total = float(info["dpos"])                          # AreaEnv: pos_w POS + area_w A of this step
        d_pos = (d_total - env.area_w*d_area)/env.pos_w
        self.A_cum += d_area
        self.pos_cum += d_pos
        self.steps += 1
        truncated = bool(not terminated and self.cfg.max_steps is not None and self.steps >= self.cfg.max_steps)
        a_b = a if self.cfg.straight_on_raw else float(getattr(env, "a_applied", a))
        bonus = self.cfg.straight_bonus*(1.0 - abs(a_b)) if self.cfg.straight_bonus else 0.0
        self.mask_cum += cut
        self.bonus_cum += bonus
        n_rev = self._count_reversals(getattr(env, "last_samples", None))
        self.roll_rev += n_rev
        reward = self.cfg.reward_scale*d_total + bonus - self.cfg.roll_reversal_cost*n_rev - self.cfg.mask_cost*cut
        return reward, bool(terminated), truncated, self._info(d_area, d_pos, info["shield"])

    def _count_reversals(self, samples) -> int:
        """Roll reversals in this step's 1 s samples (column 3 = roll, rad): a side is set once |roll| >= roll_reversal_deg."""
        if samples is None or not len(samples):
            return 0
        th = np.radians(self.cfg.roll_reversal_deg)
        n = 0
        for phi in np.asarray(samples)[:, 3]:
            if abs(phi) >= th:
                s = 1 if phi > 0 else -1
                if self._roll_side and s != self._roll_side:
                    n += 1
                self._roll_side = s
        return n

    def _info(self, d_area, d_pos, shield) -> dict:
        env = self.env
        return dict(t_s=env.t, dA=d_area, dPOS=d_pos, A=self.A_cum, POS=self.pos_cum, total=self.A_cum + env.pos_w*self.pos_cum,
                    bonus=self.bonus_cum, roll_rev=self.roll_rev, mask_cut=self.mask_cum, shield=bool(shield), energy_left=env.energy, stage=self.start_stage, steps=self.steps,
                    typ=getattr(self, "_typ", ""))

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
        seen = isinstance(self._area, SeenArea)
        rt = isinstance(self._area, RealtimeArea)
        if seen:                                              # what seeing each unseen particle now adds to A, at the particle
            v = self._area.value_now(t)
            m = (v > 0) & b.active(t)
            a_xy, a_val = b.positions(t)[m], v[m]
        elif rt:                                              # unswept cells of the current region and what sweeping each adds
            keys, a_val = self._area.open_now(t)
            a_xy = self._area.centres(keys)
        else:
            a_xy, a_val = self._area_cells()
        a_val = a_val*N_REF                                   # A gain in the same units as one particle (1 / N_REF of POS)
        # rel_maps (user 2026-11-12): the remaining belief rescaled to N_REF particles, so a searched-down or late belief reads as clearly
        # as a fresh one (the absolute channel above still shows how much is left)
        rel_unit = N_REF*env.pos_w/max(float(w.sum()), 1e-9) if self.s.get("rel_maps") else None
        # fp_rate (user 2026-11-12): the camera channel times the detection's turn-rate factor at the current roll, so the map shows the
        # image getting worse in a turn (level flight: unchanged)
        fp_q = float(det.cfg.rate_factor(np.degrees(9.80665*np.tan(float(env.plant.phi))/env.v))) if self.s.get("fp_rate") else 1.0
        c, s = np.cos(yaw), np.sin(yaw)
        obs = {}
        for k, (name, side, lead) in enumerate(cfg.maps):
            ch = [np.log1p(self._bin(P, val_now, k))]
            fut = live & b.active(t + lead)
            Pf = b.positions(t + lead)[fut]
            ch.append(np.log1p(self._bin(Pf, b.w[fut]*det.visibility(Pf[:, 0], Pf[:, 1])*unit, k)))
            fwd, right = self.offsets[k]
            wx, wy = x + fwd*c + right*s, y + fwd*s - right*c
            if side/cfg.map_px <= CELL_M and rt:              # 1.6 km map: one 25 m cell per map cell, read
                ch.append(np.log1p(self._area.value_at(wx, wy, t)*N_REF))
            elif side/cfg.map_px <= CELL_M and not seen:
                ch.append(np.log1p(self._area_at(wx, wy)*N_REF))
            else:                                             # larger cells: sum of the 25 m cells inside
                ch.append(np.log1p(self._bin(a_xy, a_val, k)))
            f, g = self.vis_grids[k]
            step = self.terr.cell*f
            col = np.rint((wx - self.terr.axis[0])/step).astype(int)
            row = np.rint((wy - self.terr.axis[0])/step).astype(int)
            inside = (np.abs(wx) <= self.half) & (np.abs(wy) <= self.half)
            vis = np.full(wx.shape, 0.0 if self.s.get("off_map_zero") else -1.0, np.float32)   # off the map: 0 from R6 (user 2026-11-12), -1 before
            vis[inside] = g[np.clip(row[inside], 0, g.shape[0] - 1), np.clip(col[inside], 0, g.shape[1] - 1)]
            ch.append(vis)
            if k == 0:
                ch.append(self.footprint*fp_q)
            if rel_unit is not None:                          # last channel: remaining POS now as a share of the remaining belief
                ch.append(np.log1p(self._bin(P, val_now/unit*rel_unit, k)))
            obs[name] = np.stack(ch).astype(np.float32)
        obs["vec"] = self.vector()
        return obs

    def vector(self) -> np.ndarray:
        env = self.env
        p = env.plant
        lo, hi, _ = env.allowed_range()
        if isinstance(self._area, (SeenArea, RealtimeArea)):
            left = self._area.cover_now(env.t)                # share of the current region already covered
        else:
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
            extra = 1 if self.env.s.get("rel_maps") else 0            # rel_maps: one more channel per map, the last one
            maps = {name: spaces.Box(-1.0, np.inf, ((5 if k == 0 else 4) + extra, px, px), np.float32) for k, (name, _, _) in enumerate(c.maps)}
            self.observation_space = spaces.Dict({**maps, "vec": spaces.Box(-2.0, 2.0, (len(VEC_NAMES),), np.float32)})
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            return self.env.step(action)

        def set_stage(self, stage: int):
            """Curriculum: starts from now on follow STAGE_MIX[stage] (called on every worker by the training script)."""
            self.env.cfg.stage, self.env.cfg.stage_mix = int(stage), None

    return RealRLGym()
