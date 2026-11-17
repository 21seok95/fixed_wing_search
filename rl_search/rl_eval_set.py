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
SEED_BASE = {"val": 70000, "test": 80000, "dval": 90000, "dtest": 95000, "mval": 100000, "mtest": 110000,
             "nval": 100000, "ntest": 110000,     # nval / ntest: the mval / mtest cases with targets that never stop (NOSTOP)
             "bval": 120000, "btest": 130000,     # bval / btest: Barstow (mojave) cases, never stop, varied routes (ROUTE)
             "v2a": 140000, "v2b": 150000}        # the redesigned scenarios (V2) on amboy120 / mojave120
# Map of a split (R10 on two maps, user 2026-11-14): the Barstow cases are on the mojave map, all others on Amboy.
SPLIT_REGION = dict(bval="mojave", btest="mojave", v2a="amboy120", v2b="mojave120")
# Diverse cases (user, 2026-11-11; docs/rl_experiments.md): per case a random target model, a wider LKP area, a launch delay and a farther
# launch, so the particles are spread by hand-over and the policy also meets flights where POS cannot be collected early.
DIVERSE = dict(lkp_half_m=15000.0, p_leave_min=(0.0, 0.05), p_return_min=(0.1, 1.0), fast_kmh=(20.0, 50.0), delay_s=(0.0, 1200.0),
               ring_m=(8000.0, 30000.0), duration_s=5400.0, max_out=0.05, tries=40)
# max_out (user 2026-11-11: in practice the search map is chosen so the target rarely leaves it): a scenario whose particles leave the map
# more than this share by the end of the flight is drawn again (the next try uses seed + 10 000 000 x try).
DIVERSE_SPLITS = ("dval", "dtest")
# Typed cases (R5, user 2026-11-12): three target types, equal shares in training at first (the trainer raises the share of a type that
# does worse than SAROPS on validation a little at a time), launch delay up to 40 min so the agent also starts after the particles spread.
# road: never leaves the road; mixed: the diverse rule above; desert: leaves early, rarely returns, drives 10-25 km/h off the road.
MULTI = dict(types=("road", "mixed", "desert"), fast_kmh=(20.0, 50.0), delay_s=(0.0, 2400.0), ring_m=(8000.0, 30000.0), duration_s=6600.0,
             max_out=0.05, tries=40)
MULTI_TYPES = dict(road=dict(p_leave_min=(0.0, 0.0), p_return_min=(1.0, 1.0)),
                   mixed=dict(p_leave_min=(0.0, 0.05), p_return_min=(0.1, 1.0)),
                   desert=dict(p_leave_min=(0.05, 0.3), p_return_min=(0.0, 0.1), off_kmh=(10.0, 25.0)))
TYPE_NAMES = dict(road="도로형", mixed="혼합형", desert="사막형")
MULTI_SPLITS = ("mval", "mtest", "nval", "ntest", "bval", "btest", "v2a", "v2b")
MULTI_N = dict(mval=15, mtest=30, nval=15, ntest=30, bval=15, btest=30, v2a=15, v2b=15)   # case k has type MULTI["types"][k % 3]; mval 9 -> 15 (5 per type, user 2026-11-12)
# Targets that never stop (R10, user 2026-11-14): no stops, and each particle drives at its own steady speed: a driver factor z (one per
# particle, rl_search.vehicle) puts it at mean x (1 + 0.15 z) in every state (fast road, slow road, off road), kept within mean x (1 +- 0.3).
# nval / ntest are the mval / mtest cases (same seeds, types, LKP, launch) with these targets; the training pool is rl_pool_multi_ns.
NOSTOP = dict(std_frac=0.15, band=0.3)
NOSTOP_SPLITS = ("nval", "ntest", "bval", "btest")
SPLIT_NAMES = dict(nval="정지 없음 ", ntest="정지 없음 ", bval="바스토 ", btest="바스토 ")
# Varied routes (R10 on two maps, user 2026-11-14): every training scenario and the Barstow cases draw the route settings of the target
# model (the web tuner's group "경로·시작") uniformly in these ranges, from a stream of their own so the other draws stay as they were.
# Off-road slope limit: 30 deg for every type (vehicle_config.json, the user's maximum).
ROUTE = dict(heading_std_deg=(10.0, 60.0), reorient_std_deg_min=(0.0, 5.0), p_detour=(0.0, 0.15), goal_sector_deg=(10.0, 30.0))
ROUTE_SPLITS = ("bval", "btest")


# Redesigned scenarios (docs/scenario_redesign.md, user 2026-11-17): the 120 km maps (amboy120, mojave120), the new target speeds (paved 50 +- 10,
# unpaved 30 +- 5, off-road 10 +- 2.5 km/h, clipped at 2 std, no stops), the agreed particle rules (vehicle.NO_BACKTRACK), particles leaving the map
# going on, 70 min tracks, the launch at the report time 10-40 km from the LKP, every scenario kept (no redraw). 30 % of the training episodes
# start mid-flight from a saved state (``midflight``). Validation: 15 launch cases per map (V2_SPLITS).
V2_SPEEDS = dict(fast_kmh=50.0, fast_std_kmh=10.0, slow_kmh=30.0, slow_std_kmh=5.0, off_kmh=10.0, off_std_kmh=2.5, speed_clip_sd=2.0,
                 speed_cap_kmh=70.0, speed_band=0.0, p_stop_min=0.0)
V2 = dict(ring_m=(10000.0, 40000.0), duration_s=4200.0, midflight=0.3)
V2_SPLITS = ("v2a", "v2b")


def v2_vc(rng, typ: str, seed: int) -> dict:
    """Target model changes of a redesigned scenario of one type (against vehicle_config()): the typed leaving / returning rules with the
    other draws of ``multi_vc`` overridden by the new speeds, the route settings of ``seed``, the agreed particle rules, exits going on."""
    from rl_search.vehicle import NO_BACKTRACK
    return multi_vc(rng, typ) | V2_SPEEDS | route_vc(seed) | NO_BACKTRACK | dict(exit_straight=True, duration_s=V2["duration_s"])


def v2_training_scenario(vs, seed: int, typ: str):
    """Redesigned training scenario of one type on the map of ``vs``: (scenario, its target model changes). The launch is drawn per episode."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    rng = np.random.default_rng(seed)
    vc = v2_vc(rng, typ, seed)
    lkp = diverse_lkp(vs.net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=getattr(vs, "region", REGION), net=vs.net)
    return vs2.make_at(seed, lkp, heading, start=(0.0, 0.0)), vc


def route_vc(seed: int) -> dict:
    """Route settings of a target model (ROUTE), drawn from their own stream of ``seed``."""
    rng = np.random.default_rng([int(seed), 31])
    return {k: float(rng.uniform(*r)) for k, r in ROUTE.items()}


def split_region(split: str | None) -> str:
    return SPLIT_REGION.get(split or "", REGION)


def eval_dir(split: str | None = None) -> Path:
    """Folder of the SAROPS references of a split: rl_eval of the split's map (the target model base stays Amboy's, EVAL_DIR)."""
    return Path("rl_search/data/real")/split_region(split)/"rl_eval"


def nostop_vc(vc: dict) -> dict:
    """Changes of a typed target model (multi_vc) for targets that never stop and keep a steady speed of their own (NOSTOP)."""
    base = vehicle_config()
    f, band = NOSTOP["std_frac"], NOSTOP["band"]
    fast = float(vc.get("fast_kmh", base.fast_kmh))
    slow = float(vc.get("slow_kmh", base.slow_kmh))
    off = float(vc.get("off_kmh", base.off_kmh))
    return vc | dict(p_stop_min=0.0, fast_std_kmh=f*fast, slow_std_kmh=f*slow, off_std_kmh=f*off, speed_band=band,
                     speed_cap_kmh=(1 + band)*fast)
FLIGHT_S = 3600.0
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
    t_launch_s: float = 0.0       # stage 3: launch this long after the LKP report (diverse cases)
    vc: dict = field(default_factory=dict)     # target model changes against vehicle_config() (diverse cases)
    typ: str = ""                 # target type of a typed case (MULTI_TYPES key)
    region: str = REGION          # map of the case (SPLIT_REGION)

    @property
    def label(self) -> str:
        d = f"{TYPE_NAMES.get(self.typ, '유형별')} " if self.split in MULTI_SPLITS else ("다양 " if self.split.startswith("d") else "")
        return f"{SPLIT_NAMES.get(self.split, '')}{d}{'검증' if self.split.endswith('val') else '시험'} {self.stage}단계 · 시나리오 {self.k + 1}"

    def options(self, scen) -> dict:
        if self.stage in (1, 2):
            return dict(scenario=scen, stage=self.stage, t0_s=self.t0_s, start_pose=np.array(self.start, float))
        return dict(scenario=scen, stage=3, t_launch_s=self.t_launch_s) if self.t_launch_s else dict(scenario=scen, stage=3)


def flight_window(options: dict) -> tuple:
    """Scoring window of the real-time A: from the launch (stage 3) or the start (stage 1 / 2) to the end of the flight
    (stage 1 / 2 start with the battery of a launch at 0 s, so they end 60 min after the LKP)."""
    if int(options.get("stage", 3)) in (1, 2):
        return float(options.get("t0_s", 0.0)), FLIGHT_S
    t = float(options.get("t_launch_s", 0.0))
    return t, t + FLIGHT_S


def case_vc(case: Case):
    from dataclasses import replace
    return replace(vehicle_config(), **case.vc) if case.vc else vehicle_config()


def diverse_vc(rng) -> dict:
    """Random target model (changes against vehicle_config()): top speed, road leaving and returning, a 90 min track."""
    fast = float(rng.uniform(*DIVERSE["fast_kmh"]))
    leave = float(rng.uniform(*DIVERSE["p_leave_min"]))
    return dict(fast_kmh=fast, slow_kmh=fast*2/3, speed_cap_kmh=fast, p_leave_fast_min=leave, p_leave_slow_min=leave,
                p_return_min=float(rng.uniform(*DIVERSE["p_return_min"])), duration_s=DIVERSE["duration_s"])


def multi_vc(rng, typ: str) -> dict:
    """Random target model of one type (changes against vehicle_config()): top speed, road leaving and returning, off-road speed."""
    t = MULTI_TYPES[typ]
    fast = float(rng.uniform(*MULTI["fast_kmh"]))
    leave = float(rng.uniform(*t["p_leave_min"]))
    vc = dict(fast_kmh=fast, slow_kmh=fast*2/3, speed_cap_kmh=fast, p_leave_fast_min=leave, p_leave_slow_min=leave,
              p_return_min=float(rng.uniform(*t["p_return_min"])), duration_s=MULTI["duration_s"])
    if "off_kmh" in t:
        vc["off_kmh"] = float(rng.uniform(*t["off_kmh"]))
    return vc


def case_type(split: str, k: int) -> str:
    return MULTI["types"][k % len(MULTI["types"])] if split in MULTI_SPLITS else ""


def diverse_lkp(net, rng, half_m: float | None = None) -> np.ndarray:
    """A drivable road cell within ``half_m`` (default DIVERSE["lkp_half_m"], the central 30 km) of the map centre, both axes."""
    a = net.terr.axis
    near = np.abs(a) <= (DIVERSE["lkp_half_m"] if half_m is None else half_m)
    cells = np.flatnonzero((net.cell_edge >= 0) & near[:, None] & near[None, :])
    r, c = np.divmod(rng.choice(cells), len(a))
    return np.array([a[c], a[r]], float)


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


def _military_at(terr, p) -> bool:
    """Is the point (x, y) in military land (terrain layer ``military``, nearest 25 m cell)? False without a terrain or the layer."""
    m = getattr(terr, "military", None) if terr is not None else None
    if m is None:
        return False
    n = len(terr.axis)
    i, j = (int(np.clip(np.round((v - terr.axis[0])/terr.cell), 0, n - 1)) for v in (p[1], p[0]))
    return bool(m[i, j])


def start_stage3(lkp, rng, half: float, ring: tuple = (8000.0, 20000.0), terr=None):
    """Launch point ``ring`` from the LKP, direction uniform, inside the map. With ``terr`` (the redesigned scenarios, 2026-11-17: military
    land is closed, so a team on the ground cannot launch there) a point in military land is drawn again."""
    for _ in range(500):
        d, th = rng.uniform(*ring), rng.uniform(-np.pi, np.pi)
        s = lkp + d*np.array([np.cos(th), np.sin(th)])
        if _inside(s, half) and not _military_at(terr, s):
            yaw = float(np.arctan2(lkp[1] - s[1], lkp[0] - s[0]))
            return 0.0, [float(s[0]), float(s[1]), yaw], dict(launch_dist_m=float(d))
    raise RuntimeError("no stage 3 launch point inside the map")


def make_case(vs, split: str, stage: int, k: int):
    """(Case, planner scenario, its simulate_vehicle result, hold-out scenario). ``vs``: real_scenario.VehicleScenarios on the split's map."""
    if getattr(vs, "region", REGION) != split_region(split):
        raise ValueError(f"{split} cases are on {split_region(split)}, the scenarios given are on {vs.region}")
    seed = SEED_BASE[split] + 1000*stage + k
    rng = np.random.default_rng(seed)
    if split in DIVERSE_SPLITS:
        return make_diverse_case(vs, split, stage, k, seed, rng)
    if split in V2_SPLITS:
        return make_v2_case(vs, split, stage, k, seed, rng)
    if split in MULTI_SPLITS:
        return make_multi_case(vs, split, stage, k, seed, rng)
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


def out_share(scen, t_end_s: float) -> float:
    """Share of the particles that have left the map by t_end_s."""
    ex = np.asarray(scen.exit_step).astype(np.int64)
    return float(((ex >= 0) & (ex*scen.save_dt_s <= t_end_s)).mean())


def make_diverse_case(vs, split: str, stage: int, k: int, seed: int, rng):
    """A diverse stage 3 case: random target model, LKP in the central 30 km square, launch 0-20 min after the report from 8-30 km;
    drawn again while more than DIVERSE["max_out"] of the particles leave the map by the end of the flight."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    if stage != 3:
        raise ValueError("diverse cases are stage 3")
    for attempt in range(DIVERSE["tries"]):
        s = seed + 10_000_000*attempt
        rng = rng if attempt == 0 else np.random.default_rng(s)
        vc = diverse_vc(rng)
        lkp = diverse_lkp(vs.net, rng)
        heading = float(rng.uniform(0.0, 360.0))
        half = float(vs.net.terr.axis[-1])
        delay = float(rng.uniform(*DIVERSE["delay_s"]))
        _, start, info = start_stage3(lkp, rng, half, DIVERSE["ring_m"])
        vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=REGION, net=vs.net)
        scen, vres = vs2.make_at(s, lkp, heading, start=start[:2], with_res=True)
        out = out_share(scen, delay + FLIGHT_S)
        if out <= DIVERSE["max_out"]:
            break
    lkp = np.asarray(scen.lkp, float)
    hold = vs2.make_at(s + HOLDOUT, lkp, heading, start=start[:2])
    info = info | dict(delay_s=delay, out_share=out, tries=attempt + 1)
    case = Case(split, stage, k, s, [float(v) for v in lkp], heading, 0.0, start, info, t_launch_s=delay, vc=vc)
    return case, scen, vres, hold


def diverse_training_scenario(vs, seed: int):
    """Training scenario with the diverse rules (own seeds): (scenario, its target model changes)."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    t_end = DIVERSE["delay_s"][1] + FLIGHT_S                  # the latest launch: the start of each episode is drawn later
    for attempt in range(DIVERSE["tries"]):
        s = seed + 10_000_000*attempt
        rng = np.random.default_rng(s)
        vc = diverse_vc(rng)
        lkp = diverse_lkp(vs.net, rng)
        heading = float(rng.uniform(0.0, 360.0))
        vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=REGION, net=vs.net)
        scen = vs2.make_at(s, lkp, heading, start=(0.0, 0.0))
        if out_share(scen, t_end) <= DIVERSE["max_out"]:
            break
    return scen, vc


def make_multi_case(vs, split: str, stage: int, k: int, seed: int, rng):
    """A typed stage 3 case (R5): type MULTI["types"][k % 3], LKP in the central 30 km square, launch 0-40 min after the report from
    8-30 km; drawn again while more than MULTI["max_out"] of the particles leave the map by the end of the flight."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    if stage != 3:
        raise ValueError("typed cases are stage 3")
    typ = case_type(split, k)
    for attempt in range(MULTI["tries"]):
        s = seed + 10_000_000*attempt
        rng = rng if attempt == 0 else np.random.default_rng(s)
        vc = multi_vc(rng, typ)
        if split in NOSTOP_SPLITS:                            # no random draws: the case keeps its LKP, heading, launch and start
            vc = nostop_vc(vc)
        if split in ROUTE_SPLITS:                             # own stream: the other draws stay as they were
            vc = vc | route_vc(s)
        lkp = diverse_lkp(vs.net, rng)
        heading = float(rng.uniform(0.0, 360.0))
        half = float(vs.net.terr.axis[-1])
        delay = float(rng.uniform(*MULTI["delay_s"]))
        _, start, info = start_stage3(lkp, rng, half, MULTI["ring_m"])
        vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=getattr(vs, "region", REGION), net=vs.net)
        scen, vres = vs2.make_at(s, lkp, heading, start=start[:2], with_res=True)
        out = out_share(scen, delay + FLIGHT_S)
        if out <= MULTI["max_out"]:
            break
    lkp = np.asarray(scen.lkp, float)
    hold = vs2.make_at(s + HOLDOUT, lkp, heading, start=start[:2])
    info = info | dict(delay_s=delay, out_share=out, tries=attempt + 1, type=typ)
    case = Case(split, stage, k, s, [float(v) for v in lkp], heading, 0.0, start, info, t_launch_s=delay, vc=vc, typ=typ,
                region=getattr(vs, "region", REGION))
    return case, scen, vres, hold


def make_v2_case(vs, split: str, stage: int, k: int, seed: int, rng):
    """A redesigned validation case (V2): type MULTI["types"][k % 3], LKP on a road in the central 30 km, launch at the report time from
    10-40 km, the scenario kept whatever leaves the map."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    if stage != 3:
        raise ValueError("redesigned cases are launch (stage 3) cases")
    typ = case_type(split, k)
    vc = v2_vc(rng, typ, seed)
    lkp = diverse_lkp(vs.net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    _, start, info = start_stage3(lkp, rng, float(vs.net.terr.axis[-1]), V2["ring_m"], terr=vs.net.terr)
    vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=getattr(vs, "region", REGION), net=vs.net)
    scen, vres = vs2.make_at(seed, lkp, heading, start=start[:2], with_res=True)
    hold = vs2.make_at(seed + HOLDOUT, np.asarray(scen.lkp, float), heading, start=start[:2])
    info = info | dict(type=typ, out_share=out_share(scen, FLIGHT_S))
    case = Case(split, stage, k, seed, [float(v) for v in np.asarray(scen.lkp, float)], heading, 0.0, start, info, t_launch_s=0.0, vc=vc,
                typ=typ, region=getattr(vs, "region", REGION))
    return case, scen, vres, hold


def multi_training_scenario(vs, seed: int, typ: str, nostop: bool = False, route: bool = False):
    """Training scenario of one type with the typed rules (own seeds): (scenario, its target model changes). ``nostop``: targets that
    never stop (NOSTOP, R10); ``route``: varied route settings (ROUTE); the map is the one of ``vs``."""
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    t_end = MULTI["delay_s"][1] + FLIGHT_S                    # the latest launch: the start of each episode is drawn later
    for attempt in range(MULTI["tries"]):
        s = seed + 10_000_000*attempt
        rng = np.random.default_rng(s)
        vc = multi_vc(rng, typ)
        if nostop:
            vc = nostop_vc(vc)
        if route:
            vc = vc | route_vc(s)
        lkp = diverse_lkp(vs.net, rng)
        heading = float(rng.uniform(0.0, 360.0))
        vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=getattr(vs, "region", REGION), net=vs.net)
        scen = vs2.make_at(s, lkp, heading, start=(0.0, 0.0))
        if out_share(scen, t_end) <= MULTI["max_out"]:
            break
    return scen, vc


def case_dict(c: Case) -> dict:
    return asdict(c) | {"label": c.label}
