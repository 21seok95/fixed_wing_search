"""Random search scenarios on the fixed terrain.

A scenario is one target simulation (LKP, known heading, target speed -> particle tracks with
exit times) plus a UAV launch pose on the map edge 10-30 km from the LKP, heading at the LKP.
t = 0 is both the LKP observation and the UAV launch.

Training draws a fresh scenario on every ``ScenarioGenerator.next()``; with ``reuse > 1`` the
same particle tracks are reused with new launch points (tracks do not depend on the UAV). Only
the fixed-seed evaluation set is written to disk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time
import numpy as np

from rl_search.target import TargetConfig, simulate
from rl_search.terrain import Terrain


@dataclass
class ScenarioConfig:
    speed_kmh: tuple = (20.0, 40.0)        # per-scenario target mean speed, uniform
    lkp_clearance_m: float = 200.0         # LKP distance from any no-go cell
    lkp_margin_m: float = 1000.0           # LKP distance from the map edge
    start_dist_m: tuple = (10000.0, 30000.0)
    corner_margin_m: float = 1000.0        # launch points skip this much of each map corner
    perimeter_step_m: float = 10.0
    uav_speed_mps: float = 120/3.6
    flight_s: float = 3000.0               # 100 km at 120 km/h
    reuse: int = 1                         # episodes per target simulation
    target: TargetConfig = field(default_factory=TargetConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        d = dict(d)
        t = TargetConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("target").items()})
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}, target=t)


@dataclass
class Scenario:
    seed: int
    lkp: np.ndarray
    heading_deg: float
    speed_mps: float
    start: np.ndarray                     # (x, y) launch point
    start_heading_deg: float
    track: np.ndarray                     # (T, N, 2) float32 every save_dt_s
    exit_step: np.ndarray                 # (N,) int16, -1 = never leaves the map
    save_dt_s: float
    metrics: dict
    gen_s: float = 0.0

    def meta(self) -> dict:
        return dict(seed=self.seed, lkp=self.lkp.tolist(), heading_deg=self.heading_deg,
                    speed_mps=self.speed_mps, speed_kmh=self.speed_mps*3.6, start=self.start.tolist(),
                    start_heading_deg=self.start_heading_deg, save_dt_s=self.save_dt_s,
                    particles=int(self.track.shape[1]), steps=int(self.track.shape[0]),
                    metrics=self.metrics, gen_s=round(self.gen_s, 2))


class ScenarioSampler:
    """Draws targets and launch points; candidate sets are precomputed once per terrain."""

    def __init__(self, terr: Terrain, cfg: ScenarioConfig):
        self.terr, self.cfg = terr, cfg
        ok = (terr.clearance >= cfg.lkp_clearance_m)
        inner = np.abs(terr.axis) <= terr.axis[-1] - cfg.lkp_margin_m
        ok &= inner[:, None] & inner[None, :]
        self.lkp_cells = np.flatnonzero(ok)
        if not len(self.lkp_cells):
            raise ValueError("No passable LKP cell with the requested clearance")
        lo, hi = terr.axis[0], terr.axis[-1]
        s = np.arange(lo + cfg.corner_margin_m, hi - cfg.corner_margin_m + 1e-6, cfg.perimeter_step_m)
        side = lambda x, y: np.stack(np.broadcast_arrays(x, y), axis=1)
        self.perimeter = np.concatenate([side(s, lo), side(hi, s), side(s, hi), side(lo, s)]).astype(float)

    def sample_target(self, rng):
        n = len(self.terr.axis)
        r, c = np.divmod(rng.choice(self.lkp_cells), n)
        lkp = np.array([self.terr.axis[c], self.terr.axis[r]]) + rng.uniform(-.5, .5, 2)*self.terr.cell
        heading = float(rng.uniform(0, 360))
        speed = float(rng.uniform(*self.cfg.speed_kmh))/3.6
        return lkp, heading, speed

    def sample_start(self, rng, lkp):
        d = np.linalg.norm(self.perimeter - lkp, axis=1)
        lo, hi = self.cfg.start_dist_m
        cand = np.flatnonzero((d >= lo) & (d <= hi))
        start = self.perimeter[rng.choice(cand)].copy()
        heading = float(np.degrees(np.arctan2(lkp[1]-start[1], lkp[0]-start[0])))
        return start, heading


def scenario_metrics(track, exit_step, save_dt_s, start, cfg: ScenarioConfig) -> dict:
    """Escaped mass at the end of the flight, intercept time of the particle centroid, reachable mass.

    Intercept t*: first stored time at which the centroid of the in-map particles is within
    ``uav_speed * t`` of the launch point. Reachable mass: share of particles that are inside the map
    and within ``uav_speed * t`` of the launch point at some stored time t before the flight ends.
    """
    k_end = min(int(round(cfg.flight_s/save_dt_s)), len(track) - 1)
    k = np.arange(k_end + 1)
    exited = (exit_step >= 0)[None, :] & (exit_step[None, :] <= k[:, None])
    radius = cfg.uav_speed_mps*save_dt_s*k
    pts = np.asarray(track[:k_end+1], float)
    d = np.linalg.norm(pts - start, axis=2)
    reach = ((d <= radius[:, None]) & ~exited).any(axis=0)
    alive = ~exited
    cnt = np.maximum(alive.sum(axis=1), 1)
    centroid = (pts*alive[..., None]).sum(axis=1)/cnt[:, None]
    hit = np.flatnonzero(np.linalg.norm(centroid - start, axis=1) <= radius)
    return dict(escaped_end=float(exited[-1].mean()),
                intercept_s=float(hit[0]*save_dt_s) if len(hit) else None,
                reachable=float(reach.mean()),
                start_lkp_km=float(np.linalg.norm(np.asarray(track[0, 0], float) - start)/1000))


def make_scenario(sampler: ScenarioSampler, seed: int) -> Scenario:
    """Deterministic scenario for ``seed`` (target, particle tracks and launch point)."""
    t0 = time.perf_counter()
    cfg = sampler.cfg
    rng = np.random.default_rng(seed)
    lkp, heading, speed = sampler.sample_target(rng)
    track, exit_step = simulate(sampler.terr, cfg.target, lkp, heading, speed, rng)
    start, start_heading = sampler.sample_start(rng, lkp)
    metrics = scenario_metrics(track, exit_step, cfg.target.save_dt_s, start, cfg)
    return Scenario(seed, lkp, heading, speed, start, start_heading, track, exit_step,
                    cfg.target.save_dt_s, metrics, time.perf_counter() - t0)


def with_new_start(sampler: ScenarioSampler, scen: Scenario, rng) -> Scenario:
    """Same particle tracks, new launch point."""
    start, start_heading = sampler.sample_start(rng, scen.lkp)
    metrics = scenario_metrics(scen.track, scen.exit_step, scen.save_dt_s, start, sampler.cfg)
    return Scenario(scen.seed, scen.lkp, scen.heading_deg, scen.speed_mps, start, start_heading,
                    scen.track, scen.exit_step, scen.save_dt_s, metrics, 0.0)


class ScenarioGenerator:
    """Endless stream of training scenarios; one target simulation per ``reuse`` episodes."""

    def __init__(self, terr: Terrain, cfg: ScenarioConfig, seed: int = 0):
        if cfg.reuse < 1:
            raise ValueError("reuse must be >= 1")
        self.sampler = ScenarioSampler(terr, cfg)
        self.rng = np.random.default_rng(seed)
        self._current: Scenario | None = None
        self._uses = 0

    def next(self) -> Scenario:
        if self._current is None or self._uses >= self.sampler.cfg.reuse:
            self._current = make_scenario(self.sampler, int(self.rng.integers(2**62)))
            self._uses = 1
            return self._current
        self._uses += 1
        return with_new_start(self.sampler, self._current, self.rng)


def save_scenario(out: Path, scen: Scenario, cfg: ScenarioConfig, terrain_path: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"track.npy", scen.track)
    np.save(out/"exit_step.npy", scen.exit_step)
    meta = scen.meta() | dict(terrain=str(terrain_path), config=asdict(cfg))
    (out/"meta.json").write_text(json.dumps(meta, indent=2))


def load_scenario(path: Path, mmap: bool = True) -> Scenario:
    meta = json.loads((path/"meta.json").read_text())
    mode = "r" if mmap else None
    return Scenario(meta["seed"], np.array(meta["lkp"]), meta["heading_deg"], meta["speed_mps"],
                    np.array(meta["start"]), meta["start_heading_deg"],
                    np.load(path/"track.npy", mmap_mode=mode), np.load(path/"exit_step.npy"),
                    meta["save_dt_s"], meta["metrics"], meta.get("gen_s", 0.0))
