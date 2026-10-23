"""Search environment with a reset/step interface (Gymnasium-style, no Gymnasium dependency).

One step = one decision interval: the action's first element is the heading change (rad) for
this step; the plant flies it, particles that have left the map move to ``escaped``, the new
footprint detects, and the reward is the probability found in this step. A detection also
happens at the launch pose in ``reset`` (reported as ``info["gain"]``), as in uav_search.

Episode end: ``truncated`` when the flight budget is used up, ``terminated`` when the UAV
leaves the map.
"""
from __future__ import annotations

import numpy as np

from rl_search.mission import Coverage, NadirSensor, ParticleBelief, SearchConfig, detect
from rl_search.plants import KinematicPlant
from rl_search.scenario import Scenario, ScenarioConfig, ScenarioGenerator
from rl_search.terrain import Terrain


def default_search_config(scfg: ScenarioConfig) -> SearchConfig:
    return SearchConfig(flight_min=scfg.flight_s/60, uav_speed_mps=scfg.uav_speed_mps)


class SearchEnv:
    def __init__(self, terr: Terrain, scfg: ScenarioConfig | None = None, cfg: SearchConfig | None = None,
                 plant=None, sensor=None, seed: int = 0, use_exits: bool = True, record: bool = False):
        self.terr = terr
        self.scfg = scfg or ScenarioConfig()
        self.cfg = cfg or default_search_config(self.scfg)
        self.plant = plant or KinematicPlant(self.cfg.uav_speed_mps, self.cfg.dt_s, self.cfg.max_turn_deg)
        self.sensor = sensor or NadirSensor(self.cfg)
        self.use_exits, self.record = use_exits, record
        self.seed = seed
        self._gen: ScenarioGenerator | None = None
        self.coverage: Coverage | None = None
        self.belief: ParticleBelief | None = None
        self.scenario: Scenario | None = None
        self.half = float(terr.axis[-1])

    @property
    def t0(self) -> float:
        return self.cfg.launch_delay_min*60

    @property
    def t_s(self) -> float:
        return self.t0 + self.k*self.cfg.dt_s

    def reset(self, seed: int | None = None, options: dict | None = None):
        """New episode. ``options["scenario"]`` fixes the scenario (evaluation set); otherwise the
        generator draws one. ``options["start_pose"]`` overrides the launch pose (x, y, yaw)."""
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
        self._log = {k: [] for k in ("t_s", "poses", "weights", "gain", "escaped", "searched_km2", "roll")}
        gain = self._observe()
        return self._obs(), self._info(gain) | dict(seed=scen.seed, gen_s=scen.gen_s)

    def step(self, action):
        turn = float(np.ravel(action)[0])
        self.pose = self.plant.step(turn)
        self.k += 1
        gain = self._observe()
        x, y = self.pose[:2]
        terminated = not bool(self.terr.inside(x, y))
        truncated = self.k >= self.cfg.n_steps
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
                         ("roll", self.plant.attitude)):
                self._log[k].append(v)
        return gain

    def _obs(self) -> np.ndarray:
        x, y, yaw = self.pose
        left = 1 - self.k/self.cfg.n_steps
        return np.array([x/self.half, y/self.half, np.cos(yaw), np.sin(yaw), left], np.float32)

    def _info(self, gain: float) -> dict:
        b = self.belief
        return dict(t_s=self.t_s, gain=gain, pos=b.pos, remaining=b.mass, escaped=b.escaped,
                    searched_km2=self.coverage.searched_km2, roll=self.plant.attitude)

    def log(self) -> dict:
        """Recorded episode (``record=True``) in the mission.npz layout of uav_search."""
        if not self.record:
            raise RuntimeError("Create the env with record=True to keep a log")
        out = {k: np.array(v) for k, v in self._log.items()}
        out["pos_cum"] = np.cumsum(out["gain"])
        return out
