"""Search scenarios on the real-region terrain with vehicle evader particles (docs/real_map_todo.md, step 2).

A scenario has the same fields as ``rl_search.scenario.Scenario`` so the search environment, the SAROPS planner
and the team runner use it unchanged: the LKP is drawn on the drivable road network (away from the map edge and
the large-city zones), the heading uniformly; the particles come from ``rl_search.vehicle.simulate_vehicle``; the
launch point is on the map edge ``start_dist_m`` from the LKP, as for the synthetic scenarios.

UAV and sensor (user decisions 2026-11-06): 33 m/s, 500 m altitude, oblique EO camera 45 deg below the horizon on
a 2-axis gimbal, YOLO detector; see ``chungju_preset`` and ``rl_search.detection``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rl_search.scenario import Scenario, scenario_metrics
from rl_search.vehicle import EXIT, OFF, STOP, VehicleConfig, load_net, simulate_vehicle

V_UAV = 33.0
OVERRIDES = [
    "plant.FW_AIRSPD_TRIM=33.0", "plant.FW_AIRSPD_MIN=25.0", "plant.FW_AIRSPD_MAX=40.0", "plant.FW_R_LIM=45.0",
    "search.uav_speed_mps=33.0", "search.fov_cross_m=190.0", "search.fov_along_m=201.0", "search.pod=0.7",
    "scenario.uav_speed_mps=33.0", "scenario.start_dist_m=(8000.0, 20000.0)", "scenario.flight_s=3600.0",
]
TERRAIN = {25.0: Path("rl_search/data/terrain_real_chungju_60km_25m.npz")}


def terrain_path(region: str = "chungju", cell_m: float = 25.0) -> Path:
    sfx = "" if cell_m == 50 else f"_{cell_m:g}m"
    return Path("rl_search/data")/f"terrain_real_{region}_60km{sfx}.npz"


# planner sensor (user 2026-11-06): lane width = the oblique camera's footprint width (trapezoid 171-209 m, 190 m
# at the centre), single-pass POD 0.7, so lanes are not packed tighter than the camera sees (fewer turns); the real
# detection is 0.7 x the model's pass POD everywhere (pass_mode "scale"), 121 m sweep width on open ground,
# against the planner's 0.7 x 190 = 133 m. (173 m = the model's effective width was tried first: chungju/step3_173.)
COVER_M, PASS_POD = 190.0, 0.7
W_PLAN = PASS_POD*COVER_M
BANK_DEG = 30.0          # standard search turn of the 33 m/s UAV: 9.8 deg/s, radius 192 m (roll limit 45 deg)


def chungju_preset(extra: list[str] | None = None):
    """rc_cessna plant and energy model at 33 m/s (roll limit 45 deg = gimbal roll range); the search fields are the
    planner's approximations of the oblique camera: lane cover 190 m (footprint width), single-pass POD 0.7 (sweep width 133 m)."""
    from rl_search.config import get_preset
    return get_preset("rc_cessna", OVERRIDES + list(extra or []))


class VehicleScenarios:
    """Draws vehicle scenarios; the road network is loaded once."""

    def __init__(self, sampler, vcfg: VehicleConfig | None = None, region: str = "chungju", cell_m: float = 25.0,
                 lkp_margin_m: float = 15000.0, net=None):
        self.sampler = sampler                       # rl_search.scenario.ScenarioSampler (launch points, config)
        self.net = load_net(region, cell_m) if net is None else net     # ``net``: a road network already loaded (shared)
        self.vcfg = vcfg or VehicleConfig()
        t = self.net.terr
        n = len(t.axis)
        drivable = self.net.cell_edge >= 0
        inner = np.abs(t.axis) <= t.axis[-1] - lkp_margin_m
        self.lkp_cells = np.flatnonzero(drivable & inner[:, None] & inner[None, :])

    def make_at(self, seed: int, lkp, heading: float, start=None, with_res: bool = False):
        """Scenario from a given LKP and heading with the particle count of ``vcfg``: the particles repeat a
        ``simulate_vehicle`` run with this seed (the web viewer's), also when that one ran for a shorter time. ``start``
        (x, y in metres, anywhere on the map; the UAV takes off towards the LKP), or None for a launch point on the map
        edge drawn from a separate random stream. ``with_res``: also return the ``simulate_vehicle`` result."""
        t0 = time.perf_counter()
        cfg = self.sampler.cfg
        vc = VehicleConfig(**{**self.vcfg.__dict__, "duration_s": max(self.vcfg.duration_s, cfg.flight_s + 600.0)})
        res = simulate_vehicle(self.net, vc, np.asarray(lkp, float), float(heading), np.random.default_rng(seed))
        if start is None:
            start, start_heading = self.sampler.sample_start(np.random.default_rng(seed + 10007), res.lkp_road)
        else:
            start = np.asarray(start, float)
            start_heading = float(np.degrees(np.arctan2(*(res.lkp_road - start)[::-1])))
        track = res.track.astype(np.float32)
        exit_step = res.exit_step.astype(np.int16)
        metrics = scenario_metrics(track, exit_step, vc.save_dt_s, start, cfg)
        scen = Scenario(seed, res.lkp_road, float(heading), vc.fast_kmh/3.6, start, start_heading, track, exit_step,
                        vc.save_dt_s, metrics, time.perf_counter() - t0)
        return (scen, res) if with_res else scen

    def make(self, seed: int, particles: int = 2000) -> Scenario:
        t0 = time.perf_counter()
        cfg = self.sampler.cfg
        rng = np.random.default_rng(seed)
        r, c = np.divmod(rng.choice(self.lkp_cells), len(self.net.terr.axis))
        lkp = np.array([self.net.terr.axis[c], self.net.terr.axis[r]], float)
        heading = float(rng.uniform(0, 360))
        vc = VehicleConfig(**{**self.vcfg.__dict__, "particles": particles, "duration_s": cfg.flight_s + 600.0})
        res = simulate_vehicle(self.net, vc, lkp, heading, rng)
        start, start_heading = self.sampler.sample_start(rng, res.lkp_road)
        track = res.track.astype(np.float32)
        exit_step = res.exit_step.astype(np.int16)
        metrics = scenario_metrics(track, exit_step, vc.save_dt_s, start, cfg)
        k_end = min(int(round(cfg.flight_s/vc.save_dt_s)), len(res.state) - 1)
        live = res.state[:k_end + 1] != EXIT
        metrics.update(off_share=float((res.state[:k_end + 1][live] == OFF).mean()),
                       stop_share=float((res.state[:k_end + 1][live] == STOP).mean()))
        return Scenario(seed, res.lkp_road, heading, vc.fast_kmh/3.6, start, start_heading, track, exit_step,
                        vc.save_dt_s, metrics, time.perf_counter() - t0)


def make_env(seed: int = 0, tracked: bool = True, lead_m: float = 0.0, prelook_s: float = 0.0, detection=None,
             bank_deg: float | None = None, lane_time: str = "flown", overrides: list[str] | None = None, region: str = "chungju", terr=None,
             area: bool = False, edge_mode: str = "mask", edge_band_m: float = 500.0):
    """Search environment on a real-region 25 m terrain with the oblique camera (fast mode: no maps, no area term).
    ``bank_deg``: course change per decision from a coordinated turn at this bank (None: the RL default 30 deg per
    5 s decision, 6 deg/s, radius 315 m). ``lane_time``: "flown" (default) or "geometry" (``fixed_lane_time``). ``terr``: terrain already
    loaded (shared by a team). ``area``: an ``area_score.AreaEnv``, for planning on A + w POS (set its ``area``, ``pos_w``, ``area_w``).
    ``edge_mode``: "mask" (default since 2026-11-10, docs/rl_mdp_design.md): within ``edge_band_m`` of the map edge no action may turn the
    course target outward; "shield": the older geofence."""
    from rl_search.multi import TrackedEnv
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.terrain import load_terrain
    terr = load_terrain(terrain_path(region)) if terr is None else terr
    cls = TrackedEnv if tracked else RLSearchEnv
    if area:
        from rl_search.area_score import AreaEnv
        cls = AreaEnv
    rc = RLConfig(stage=3, detect_model="oblique", detection=detection, oblique_region=region, edge_mode=edge_mode, edge_band_m=edge_band_m)
    if bank_deg is not None:
        rc.turn_limit_deg = float(np.degrees(9.80665*np.tan(np.radians(bank_deg))/V_UAV)*rc.decision_dt_s)
    env = cls(terr, chungju_preset(overrides), rc, seed=seed)
    env.fast = env.skip_cells = True
    env.lane_lead_m, env.prelook_s = lead_m, prelook_s
    env.lane_time_model = lane_time        # "flown" (user 2026-11-06): planner lane time by a dry flight of the follower
    return env
