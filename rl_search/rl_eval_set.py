"""Validation and test cases for comparing the RL policy with SAROPS (docs/rl_mdp_design.md, section 6, 2026-11-10).

Amboy map, LKP on a road inside the central 3 km x 3 km square, heading uniform, vehicle settings of the user's latest SAROPS run
(``rl_eval/vehicle_config.json``, copied from web run 20261110-162140-008). Every case has a planner particle set and a hold-out set
(same LKP and heading, other random stream) for scoring. Starts by curriculum stage:

1. t0 uniform in 3-30 min, 1-3 km from one particle drawn by weight (all weights are equal before any search), heading uniform.
2. t0 uniform in 3-30 min, 3-10 km from the nearest cell of the region holding 90 % of the particles (100 m grid, 150 m smoothing).
3. launch at 0 s from a ring 8-20 km around the LKP (random bearing), heading to the LKP; the env flies the 2 min hand-over first.

Seeds: validation 70000 + 1000 stage + k, test 80000 + 1000 stage + k; hold-out seed + 1 000 000.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

REGION = "amboy"
EVAL_DIR = Path("rl_search/data/real")/REGION/"rl_eval"
SEED_BASE = {"val": 70000, "test": 80000}
HOLDOUT = 1_000_000
CENTER_HALF_M = 1500.0
START_MARGIN_M = 1500.0          # starts at least this far inside the map (the edge band is 500 m)
REGION_SHARE = 0.9


def vehicle_config():
    from rl_search.vehicle import VehicleConfig
    return VehicleConfig(**json.loads((EVAL_DIR/"vehicle_config.json").read_text()))


@dataclass
class Case:
    split: str
    stage: int
    k: int
    seed: int
    lkp: list
    heading: float
    t0_s: float
    start: list                   # x, y (m) and yaw (rad, 0 = east, counter-clockwise)
    info: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{'검증' if self.split == 'val' else '시험'} {self.stage}단계 · 시나리오 {self.k + 1}"

    def options(self, scen) -> dict:
        if self.stage in (1, 2):
            return dict(scenario=scen, stage=self.stage, t0_s=self.t0_s, start_pose=np.array(self.start, float))
        return dict(scenario=scen, stage=3)


def center_lkp(net, rng) -> np.ndarray:
    a = net.terr.axis
    near = np.abs(a) <= CENTER_HALF_M
    cells = np.flatnonzero((net.cell_edge >= 0) & near[:, None] & near[None, :])
    if not len(cells):
        raise RuntimeError("no drivable road in the central square")
    r, c = np.divmod(rng.choice(cells), len(a))
    return np.array([a[c], a[r]], float)


def _inside(p, half: float) -> bool:
    return bool(np.all(np.abs(p) <= half - START_MARGIN_M))


def _active(scen, t0: float):
    k = min(int(round(t0/scen.save_dt_s)), len(scen.track) - 1)
    ex = np.asarray(scen.exit_step)
    live = (ex < 0) | (ex > k)
    return np.asarray(scen.track[k], float)[live]


def start_stage1(scen, rng, half: float):
    t0 = float(rng.uniform(180.0, 1800.0))
    pts = _active(scen, t0)
    for _ in range(500):
        p = pts[rng.integers(len(pts))]
        d, th = rng.uniform(1000.0, 3000.0), rng.uniform(-np.pi, np.pi)
        s = p + d*np.array([np.cos(th), np.sin(th)])
        if _inside(s, half):
            return t0, [float(s[0]), float(s[1]), float(rng.uniform(-np.pi, np.pi))], dict(dist_to_particle_m=float(d))
    raise RuntimeError("no stage 1 start inside the map")


def region_mask(pts, half: float, cell: float = 100.0, sigma_m: float = 150.0, share: float = REGION_SHARE):
    n = int(np.ceil(2*half/cell))
    ij = np.clip(((pts + half)/cell).astype(int), 0, n - 1)
    hist = np.zeros((n, n))
    np.add.at(hist, (ij[:, 1], ij[:, 0]), 1.0)
    dens = gaussian_filter(hist, sigma_m/cell, mode="constant")
    thr = max(np.quantile(dens[ij[:, 1], ij[:, 0]], 1 - share), 1e-12)
    return (dens >= thr) & (dens > 0), cell


def start_stage2(scen, rng, half: float):
    """3-10 km from the nearest cell of the region holding 90 % of the particles (on roads the region is thin strips, so the distance
    to the nearest strip is used, not a ray from the mean)."""
    t0 = float(rng.uniform(180.0, 1800.0))
    pts = _active(scen, t0)
    mask, cell = region_mask(pts, half)
    dist = distance_transform_edt(~mask)*cell
    n = mask.shape[0]
    c = -half + (np.arange(n) + .5)*cell
    inner = np.abs(c) <= half - START_MARGIN_M
    ok_grid = inner[:, None] & inner[None, :]
    for _ in range(200):
        d = rng.uniform(3000.0, 10000.0)
        cand = np.flatnonzero(ok_grid.ravel() & (np.abs(dist.ravel() - d) <= cell))
        if len(cand):
            r, q = np.divmod(rng.choice(cand), n)
            s = np.array([c[q], c[r]]) + rng.uniform(-cell/2, cell/2, 2)
            return t0, [float(s[0]), float(s[1]), float(rng.uniform(-np.pi, np.pi))], dict(dist_to_region_m=float(dist[r, q]),
                                                                                          region_km2=float(mask.sum()*cell*cell/1e6))
    raise RuntimeError("no stage 2 start inside the map")


def start_stage3(lkp, rng, half: float):
    for _ in range(500):
        d, th = rng.uniform(8000.0, 20000.0), rng.uniform(-np.pi, np.pi)
        s = lkp + d*np.array([np.cos(th), np.sin(th)])
        if _inside(s, half):
            yaw = float(np.arctan2(lkp[1] - s[1], lkp[0] - s[0]))
            return 0.0, [float(s[0]), float(s[1]), yaw], dict(launch_dist_m=float(d))
    raise RuntimeError("no stage 3 launch point inside the map")


def make_case(vs, split: str, stage: int, k: int):
    """(Case, planner scenario, its simulate_vehicle result, hold-out scenario). ``vs``: real_scenario.VehicleScenarios on Amboy."""
    seed = SEED_BASE[split] + 1000*stage + k
    rng = np.random.default_rng(seed)
    lkp = center_lkp(vs.net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    half = float(vs.net.terr.axis[-1])
    if stage == 3:
        t0, start, info = start_stage3(lkp, rng, half)
    else:
        scen0 = vs.make_at(seed, lkp, heading, start=(0.0, 0.0))
        t0, start, info = (start_stage1 if stage == 1 else start_stage2)(scen0, rng, half)
    scen, vres = vs.make_at(seed, lkp, heading, start=start[:2], with_res=True)
    lkp = np.asarray(scen.lkp, float)               # snapped to the road
    hold = vs.make_at(seed + HOLDOUT, lkp, heading, start=start[:2])
    case = Case(split, stage, k, seed, [float(v) for v in lkp], heading, t0, start, info)
    return case, scen, vres, hold


def case_dict(c: Case) -> dict:
    return asdict(c) | {"label": c.label}
