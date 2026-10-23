"""Presets: every setting of a run in one object, so a value or a component can be swapped without code edits.

A preset bundles the scenario, mission, plant, energy, handover, waypoint export, path following,
lawnmower and wind settings. Two presets are built in:

- ``rc_cessna`` (default, stage 4+): PX4 Gazebo gz_rc_cessna at 15 m/s, 4 s decisions, POD 0.3,
  ``FixedWingPlant`` with PX4 parameter names, energy budget, 120 s handover.
- ``fast`` (stage 0-2): 33.3 m/s, 2 s decisions, instant turns, no energy model, no handover.
  It reproduces uav_search.py and is used for the regression check.

Change any field from the command line with dotted keys::

    --preset rc_cessna --set plant.FW_R_LIM=45 search.pod=1 energy.induced_share=0.4

or write a preset to JSON (``save_preset``), edit it and pass ``--preset path/to/preset.json``.
Single sources: the cruise speed is ``plant.FW_AIRSPD_TRIM`` and the flight time is
``energy.budget_s`` (copied into the scenario and search settings by ``finalize``) while those are
active; ``search.max_turn_deg = None`` is measured from the plant (``plants.measure_max_turn``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
import ast
import copy
import json

from rl_search.mission import SearchConfig
from rl_search.scenario import ScenarioConfig
from rl_search.target import TargetConfig


@dataclass
class PlantConfig:
    kind: str = "fixed_wing"             # key of plants.PLANTS
    FW_AIRSPD_TRIM: float = 15.0         # m/s
    FW_AIRSPD_MIN: float = 11.0
    FW_AIRSPD_MAX: float = 20.0
    FW_R_LIM: float = 50.0               # deg, roll limit
    FW_R_TC: float = 0.4                 # s, roll time constant
    FW_R_RMAX: float = 56.15             # deg/s, roll rate limit
    NPFG_PERIOD: float = 12.0            # s
    NPFG_DAMPING: float = 0.7
    inner_dt_s: float = 0.1              # integration step inside one decision step
    gimbal_roll_deg: float = 45.0        # NadirSensor warns above this roll


@dataclass
class EnergyConfig:
    enabled: bool = True
    induced_share: float = 0.5           # k in P = (1-k)(V/Vt)^3 + k (Vt/V)/cos^2(phi), P_trim = 1
    budget_s: float = 3600.0             # energy in trim-flight seconds


@dataclass
class HandoverConfig:
    enabled: bool = True
    t_plan_s: float = 120.0              # transit to the LKP until this time, then the planner flies


@dataclass
class ExportConfig:
    tolerance_m: float = 15.0            # max distance of the planned path from the thinned polyline
    home_lat: float = 47.397971          # PX4 Gazebo default world home
    home_lon: float = 8.546164
    home_alt_m: float = 488.0
    altitude_m: float = 100.0            # relative to home
    origin: str = "launch"               # "launch": launch point at home, "map": map centre at home
    mission_limit: int = 500             # Pixhawk mission storage (SITL allows more)
    takeoff: bool = True
    finish: str = "land"                 # "land" (PX4 fixed-wing needs a landing: MIS_TKO_LAND_REQ 2), "rtl", "none"
    land_heading_deg: float = 0.0        # landing direction at home (0 = east, CCW); gz_rc_cessna spawns facing east
    approach_m: float = 500.0            # loiter-to-alt this far before the landing point
    approach_alt_m: float = 50.0         # then glide 50 m over 500 m = 5.7 deg < FW_LND_ANG 8 deg
    acceptance_m: float = 15.0           # NAV_ACC_RAD of gz_rc_cessna


@dataclass
class FollowConfig:
    lookahead_m: float = 90.0            # pursuit point ahead of the closest path point
    schedule: bool = False               # adjust airspeed to keep the planned ETA
    schedule_gain: float = 0.3           # m/s airspeed per s of delay
    search_ahead_m: float = 600.0        # window for the closest path point


@dataclass
class LawnmowerConfig:
    hdr: float = 0.8                     # share of in-map probability covered by the search rectangle
    lane_spacing_m: float | None = None  # None: footprint width (fov_cross_m)
    n_snapshots: int = 30                # belief snapshots between handover and energy end
    cell_m: float = 100.0                # KDE grid for the HDR
    repeat_margin: float = 1.2           # build lanes for this multiple of the remaining flight


@dataclass
class WindConfig:
    enabled: bool = False
    speed_mps: tuple = (0.0, 6.0)        # constant wind speed, uniform; direction uniform
    gust_std_mps: float = 0.8            # Ornstein-Uhlenbeck gust per axis
    gust_tau_s: float = 20.0
    replan_error_s: float = 30.0         # mode (c): replan when the schedule error exceeds this
    replan_min_gap_s: float = 120.0


@dataclass
class Preset:
    name: str = "rc_cessna"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)
    lawnmower: LawnmowerConfig = field(default_factory=LawnmowerConfig)
    wind: WindConfig = field(default_factory=WindConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Preset":
        d = copy.deepcopy(d)
        kw = dict(name=d.pop("name", "custom"))
        for k, v in d.items():
            if k == "scenario":
                kw[k] = ScenarioConfig.from_dict(v)
            elif k in _CLASSES:
                kw[k] = _build(_CLASSES[k], v)
        return cls(**kw)


_CLASSES = dict(search=SearchConfig, plant=PlantConfig, energy=EnergyConfig, handover=HandoverConfig,
                export=ExportConfig, follow=FollowConfig, lawnmower=LawnmowerConfig, wind=WindConfig)


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


def rc_cessna() -> Preset:
    return Preset(
        name="rc_cessna",
        scenario=ScenarioConfig(speed_kmh=(15.0, 30.0), start_dist_m=(5000.0, 15000.0), uav_speed_mps=15.0,
                                flight_s=3600.0, target=TargetConfig(duration_s=4200.0, speed_cap_mps=30/3.6)),
        search=SearchConfig(flight_min=60.0, uav_speed_mps=15.0, dt_s=4.0, max_turn_deg=None, pod=0.3),
    )


def fast() -> Preset:
    return Preset(
        name="fast",
        scenario=ScenarioConfig(speed_kmh=(20.0, 40.0), start_dist_m=(10000.0, 30000.0), uav_speed_mps=120/3.6,
                                flight_s=3000.0, target=TargetConfig(duration_s=3600.0, speed_cap_mps=40/3.6)),
        search=SearchConfig(flight_min=50.0, uav_speed_mps=120/3.6, dt_s=2.0, max_turn_deg=23.5, pod=1.0),
        plant=PlantConfig(kind="kinematic", FW_AIRSPD_TRIM=120/3.6, FW_AIRSPD_MIN=120/3.6, FW_AIRSPD_MAX=120/3.6),
        energy=EnergyConfig(enabled=False, budget_s=3000.0),
        handover=HandoverConfig(enabled=False),
    )


PRESETS = {"rc_cessna": rc_cessna, "fast": fast}


def parse_value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(preset: Preset, overrides: list[str] | None) -> Preset:
    """Apply ``section.field=value`` strings (``name=...`` renames the preset)."""
    preset = copy.deepcopy(preset)
    for item in overrides or []:
        key, _, text = item.partition("=")
        if not _:
            raise ValueError(f"Override {item!r} is not key=value")
        val = parse_value(text)
        if key == "name":
            preset.name = str(val)
            continue
        obj = preset
        *path, last = key.split(".")
        for p in path:
            obj = getattr(obj, p)
        if not is_dataclass(obj) or last not in {f.name for f in fields(obj)}:
            raise KeyError(f"Unknown setting {key!r}")
        setattr(obj, last, tuple(val) if isinstance(val, list) else val)
    return preset


def finalize(preset: Preset) -> Preset:
    """Tie the shared values together and measure max_turn_deg if it is None."""
    p = copy.deepcopy(preset)
    if p.plant.kind != "kinematic":
        p.search.uav_speed_mps = p.scenario.uav_speed_mps = p.plant.FW_AIRSPD_TRIM
    if p.energy.enabled:
        p.search.flight_min = p.energy.budget_s/60
        p.scenario.flight_s = p.energy.budget_s
    if p.search.max_turn_deg is None:
        from rl_search.plants import measure_max_turn
        p.search.max_turn_deg = round(measure_max_turn(p.plant, p.search.dt_s), 2)
    return p


def raw_preset(name: str | Path = "rc_cessna", overrides: list[str] | None = None) -> Preset:
    """Built-in preset name or a JSON file, with overrides, not finalized (what scenarios store)."""
    if str(name) in PRESETS:
        base = PRESETS[str(name)]()
    else:
        base = Preset.from_dict(json.loads(Path(name).read_text()))
    return apply_overrides(base, overrides)


def get_preset(name: str | Path = "rc_cessna", overrides: list[str] | None = None) -> Preset:
    return finalize(raw_preset(name, overrides))


def save_preset(path: Path, preset: Preset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preset.to_dict(), indent=2))


def preset_from_meta(meta: dict, overrides: list[str] | None = None) -> Preset:
    """Preset stored with a scenario; scenarios written before presets existed are ``fast`` ones."""
    if "preset" in meta:
        base = Preset.from_dict(meta["preset"])
    else:
        scfg = ScenarioConfig.from_dict(meta["config"])
        base = replace(fast(), scenario=scfg)
        base.search.flight_min, base.search.uav_speed_mps = scfg.flight_s/60, scfg.uav_speed_mps
    return finalize(apply_overrides(base, overrides))


def add_preset_args(ap) -> None:
    ap.add_argument("--preset", default=None, help="Built-in preset (rc_cessna, fast) or a preset JSON file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Overrides, e.g. plant.FW_R_LIM=45 search.pod=1")
