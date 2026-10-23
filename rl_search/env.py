"""Search environment with a reset/step interface (Gymnasium-style, no Gymnasium dependency).

One step = one decision interval. ``action[0]`` is the heading change (rad) for this step and the
optional ``action[1]`` an airspeed command (m/s, NaN = trim). The plant flies it (with wind if a
wind model is set), the energy model charges the power used, particles that have left the map move
to ``escaped``, the new footprint detects, and the reward is the probability found in this step.
A detection also happens at the launch pose in ``reset`` (reported as ``info["gain"]``), as in
uav_search.

Handover (``handover.enabled``): ``reset`` flies the transit itself, steering at the LKP with the
sensor on and energy charged, until ``t_plan_s``; the returned observation is the one at handover
and ``env.handover`` holds the handover pose and the straight-line prediction made at launch.

Episode end: ``truncated`` when the energy (or, without an energy model, the step budget) is used
up, ``terminated`` when the UAV leaves the map. Build a configured env with ``make_env(preset, terr)``.
"""
from __future__ import annotations

import numpy as np

from rl_search.mission import Coverage, NadirSensor, ParticleBelief, SearchConfig, detect
from rl_search.plants import KinematicPlant, make_plant, wrap
from rl_search.scenario import Scenario, ScenarioConfig, ScenarioGenerator
from rl_search.terrain import Terrain

LOG_KEYS = ("t_s", "poses", "weights", "gain", "escaped", "searched_km2", "roll", "r_cmd", "airspeed",
            "energy_left", "wind")


def default_search_config(scfg: ScenarioConfig) -> SearchConfig:
    return SearchConfig(flight_min=scfg.flight_s/60, uav_speed_mps=scfg.uav_speed_mps)


class SearchEnv:
    def __init__(self, terr: Terrain, scfg: ScenarioConfig | None = None, cfg: SearchConfig | None = None,
                 plant=None, sensor=None, seed: int = 0, use_exits: bool = True, record: bool = False,
                 power=None, handover=None, wind=None):
        self.terr = terr
        self.scfg = scfg or ScenarioConfig()
        self.cfg = cfg or default_search_config(self.scfg)
        self.plant = plant or KinematicPlant(self.cfg.uav_speed_mps, self.cfg.dt_s, self.cfg.max_turn_deg)
        self.sensor = sensor or NadirSensor(self.cfg)
        self.power, self.handover_cfg, self.wind = power, handover, wind
        self.use_exits, self.record = use_exits, record
        self.seed = seed
        self._gen: ScenarioGenerator | None = None
        self.coverage: Coverage | None = None
        self.belief: ParticleBelief | None = None
        self.scenario: Scenario | None = None
        self.half = float(terr.axis[-1])
        self.energy_left = np.inf
        self.handover: dict | None = None
        self.w_now = np.zeros(2)

    @property
    def t0(self) -> float:
        return self.cfg.launch_delay_min*60

    @property
    def t_s(self) -> float:
        return self.t0 + self.k*self.cfg.dt_s

    @property
    def energy_budget(self) -> float:
        return self.power.budget if self.power is not None else np.inf

    def reset(self, seed: int | None = None, options: dict | None = None):
        """New episode. ``options``: ``scenario`` fixes the scenario (evaluation set), otherwise the generator
        draws one; ``start_pose`` overrides the launch pose (x, y, yaw); ``handover=False`` skips the transit;
        ``wind_seed`` sets the wind draw (default: scenario seed)."""
        options = options or {}
        if seed is not None or self._gen is None:
            self._gen = ScenarioGenerator(self.terr, self.scfg, self.seed if seed is None else seed)
        scen = options.get("scenario") or self._gen.next()
        self.scenario = scen
        start = options.get("start_pose")
        if start is None:
            start = (*scen.start, np.deg2rad(scen.start_heading_deg))
        self.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, self.use_exits)
        if self.coverage is None:
            self.coverage = Coverage(self.terr.axis, self.terr.forbidden, scen.lkp, scen.track,
                                     scen.save_dt_s, self.cfg)
        else:
            self.coverage.reset(scen.lkp, scen.track, scen.save_dt_s)
        self.k = 0
        self.pose = self.plant.reset(start)
        self.energy_left = self.energy_budget
        if self.wind is not None:
            self.wind.reset(np.random.default_rng([options.get("wind_seed", scen.seed), 6]))
        self.w_now = np.zeros(2)
        self._log = {k: [] for k in LOG_KEYS}
        gain = self._observe()
        info = self._info(gain) | dict(seed=scen.seed, gen_s=scen.gen_s)
        self.handover = None
        hc = self.handover_cfg
        if hc is not None and hc.enabled and options.get("handover", True):
            obs, info = self._transit(hc.t_plan_s, info)
            return obs, info
        return self._obs(), info

    def _transit(self, t_plan_s: float, info: dict):
        x0, y0, yaw0 = self.pose
        d = t_plan_s - self.t_s
        predicted = np.array([x0 + self.cfg.uav_speed_mps*d*np.cos(yaw0),
                              y0 + self.cfg.uav_speed_mps*d*np.sin(yaw0), yaw0])
        found = info["gain"]
        done = False
        while self.t_s < t_plan_s - 1e-9 and not done:
            _, r, term, trunc, info = self.step([self.turn_toward(self.scenario.lkp)])
            found += r
            done = term or trunc
        self.handover = dict(t_s=self.t_s, k=self.k, pose=self.pose.copy(), predicted=predicted,
                             error_m=float(np.hypot(*(self.pose[:2] - predicted[:2]))), found=found,
                             energy_used=self.energy_budget - self.energy_left, ended=done)
        return self._obs(), info | dict(handover=self.handover)

    def turn_toward(self, xy) -> float:
        bearing = np.arctan2(xy[1] - self.pose[1], xy[0] - self.pose[0])
        return float(np.clip(wrap(bearing - self.pose[2]), -self.plant.max_turn, self.plant.max_turn))

    def step(self, action):
        a = np.ravel(action)
        turn = float(a[0])
        airspeed = float(a[1]) if len(a) > 1 and np.isfinite(a[1]) else None
        self.w_now = self.wind.sample() if self.wind is not None else np.zeros(2)
        self.pose = self.plant.step(turn, airspeed=airspeed, wind=self.w_now if self.wind is not None else None)
        self.k += 1
        if self.power is not None:
            self.energy_left -= self.power.step_energy(self.plant.trace)
        gain = self._observe()
        x, y = self.pose[:2]
        terminated = not bool(self.terr.inside(x, y))
        truncated = self.energy_left <= 0 if self.power is not None else self.k >= self.cfg.n_steps
        return self._obs(), gain, terminated, truncated, self._info(gain)

    def _observe(self) -> float:
        t = self.t_s
        self.belief.update_exits(t)
        fp = self.sensor.footprint(self.pose[None, :], self.plant.attitude)
        gain = detect(self.belief, t, fp, self.cfg.bandwidth_m)
        self.coverage.update(fp)
        if self.record:
            for k, v in (("t_s", t), ("poses", self.pose[None, :].copy()),
                         ("weights", self.belief.w.astype(np.float32)), ("gain", gain),
                         ("escaped", self.belief.escaped), ("searched_km2", self.coverage.searched_km2),
                         ("roll", self.plant.attitude), ("r_cmd", getattr(self.plant, "r", 0.0)),
                         ("airspeed", self.plant.v), ("energy_left", self.energy_left),
                         ("wind", self.w_now.copy())):
                self._log[k].append(v)
        return gain

    @property
    def left(self) -> float:
        if self.power is not None:
            return max(self.energy_left, 0.0)/self.energy_budget
        return 1 - self.k/self.cfg.n_steps

    def _obs(self) -> np.ndarray:
        x, y, yaw = self.pose
        base = np.array([x/self.half, y/self.half, np.cos(yaw), np.sin(yaw), self.left], np.float32)
        return np.concatenate([base, self.plant.obs_extra()])

    def _info(self, gain: float) -> dict:
        b = self.belief
        return dict(t_s=self.t_s, gain=gain, pos=b.pos, remaining=b.mass, escaped=b.escaped,
                    searched_km2=self.coverage.searched_km2, roll=self.plant.attitude,
                    energy_left=self.energy_left, energy_used=self.energy_budget - self.energy_left
                    if self.power is not None else np.nan, airspeed=self.plant.v)

    def snapshot(self) -> dict:
        """Mutable episode state (belief, coverage, plant, energy, wind), for planning from the current state."""
        b, c = self.belief, self.coverage
        return dict(k=self.k, pose=self.pose.copy(), energy_left=self.energy_left, w=b.w.copy(),
                    escaped=b.escaped, gone=b._gone.copy(), cov_u=c.u.copy(), cov_coarse=c.coarse.copy(),
                    plant=self.plant.get_state(), wind=None if self.wind is None else self.wind.get_state())

    def restore(self, s: dict) -> None:
        """Load a snapshot taken on an env with the same scenario (call ``reset`` with it first)."""
        b, c = self.belief, self.coverage
        self.k, self.pose, self.energy_left = s["k"], s["pose"].copy(), s["energy_left"]
        b.w[:], b.escaped, b._gone[:] = s["w"], s["escaped"], s["gone"]
        c.u[:], c.coarse[:] = s["cov_u"], s["cov_coarse"]
        self.plant.set_state(s["plant"])
        if self.wind is not None and s["wind"] is not None:
            self.wind.set_state(s["wind"])

    def log(self) -> dict:
        """Recorded episode (``record=True``) in the mission.npz layout of uav_search."""
        if not self.record:
            raise RuntimeError("Create the env with record=True to keep a log")
        out = {k: np.array(v) for k, v in self._log.items()}
        out["pos_cum"] = np.cumsum(out["gain"])
        return out


def make_env(preset, terr: Terrain, record: bool = True, plant_kind: str | None = None, wind="model",
             use_exits: bool = True, seed: int = 0) -> SearchEnv:
    """SearchEnv wired from a preset. ``wind``: "model" (preset wind model, None if disabled), None, or an
    object with reset/sample/estimate (e.g. ``wind.ConstantWind``)."""
    from rl_search.energy import make_power
    from rl_search.wind import make_wind
    cfg = preset.search
    w = make_wind(preset) if isinstance(wind, str) and wind == "model" else wind
    return SearchEnv(terr, preset.scenario, cfg, plant=make_plant(preset, plant_kind),
                     sensor=NadirSensor(cfg, preset.plant.gimbal_roll_deg), seed=seed, use_exits=use_exits,
                     record=record, power=make_power(preset), handover=preset.handover, wind=w)
