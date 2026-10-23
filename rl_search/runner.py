"""Mission runs (stages 3-6): plan -> waypoints -> follow, for any registered method.

A *method* turns the scenario into a waypoint mission. Planner methods (straight, greedy, tree, or
any ``planners.PLANNERS`` entry, e.g. a future RL policy wrapped as a Planner) roll the planner out
in a planning env from handover to the end of the energy; the planned poses, transit included, are
thinned into waypoints with ETAs. The lawnmower builds its waypoints at handover. Every mission is
then *executed* by ``PathFollower`` in a separate env (the "real" flight: same plant, true wind),
which gives the reported POS. Planned POS (the rollout itself) is kept for the planned-vs-executed
comparison.

Wind modes (stage 6): ``none`` no wind; ``unaware`` plan without wind, fly in wind; ``aware`` plan
with the wind estimate, follower corrects for it and schedules airspeed to the ETA; ``replan`` as
aware, plus a new plan from the current state when the schedule error exceeds ``replan_error_s``.
Register new methods in ``METHODS``.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from rl_search.env import make_env
from rl_search.follower import PathFollower
from rl_search.lawnmower import build_lawnmower
from rl_search.planners import PLANNERS, rollout
from rl_search.waypoints import Waypoints, max_deviation, thin_path
from rl_search.wind import ConstantWind

PLANNER_OPTS = {"straight": {}, "greedy": {},
                "tree": dict(horizon=6, terminal_weight=1.0, terminal_bw_m=600.0, beam=50)}
WIND_MODES = ("none", "unaware", "aware", "replan")


def with_wind(preset, mode: str):
    p = copy.deepcopy(preset)
    p.wind.enabled = mode != "none"
    p.follow.schedule = mode in ("aware", "replan")
    return p


def make_planner(preset, name: str, opts: dict | None = None):
    return PLANNERS[name](preset.search, 5, **(PLANNER_OPTS.get(name, {}) | (opts or {})))


def plan_planner(preset, terr, scen, name: str, wind_est=None, opts=None, start_snapshot=None):
    """Planner rollout to the end of the energy in a planning env (wind = estimate or none).

    From launch (transit included) or from ``start_snapshot``; returns (rollout, planning env, t, poses)
    where t, poses cover the whole planned flight."""
    wind = None if wind_est is None else ConstantWind(wind_est)
    env = make_env(preset, terr, record=start_snapshot is None, wind=wind)
    if start_snapshot is None:
        env.reset(options=dict(scenario=scen))
        log = env.log()
        t_pre, poses_pre = log["t_s"][:-1], log["poses"][:-1, 0]
    else:
        env.reset(options=dict(scenario=scen, handover=False))
        env.restore(start_snapshot)
        t_pre, poses_pre = np.zeros(0), np.zeros((0, 3))
    planner = make_planner(preset, name, opts)
    planner.wind = np.zeros(2) if wind_est is None else np.asarray(wind_est, float)
    ro = rollout(env, planner)
    return ro, env, np.r_[t_pre, ro["t_s"]], np.vstack([poses_pre, ro["poses"]])


def follow(env, wp: Waypoints, preset, wind_est=None, replan=None, info: dict | None = None):
    """Fly ``wp`` with PathFollower from the env's current state to the end of the episode.

    ``replan(env) -> Waypoints`` is called when |schedule error| > wind.replan_error_s."""
    pc, fc = preset.plant, preset.follow
    v_trim = preset.search.uav_speed_mps
    dt = preset.search.dt_s
    fol = PathFollower(wp.after(env.t_s), fc, v_trim, pc.FW_AIRSPD_MIN, pc.FW_AIRSPD_MAX, wind_est, dt)
    sched, xtrack, replans = [], [], []
    last_replan = env.t_s
    done = False
    while not done:
        if (replan is not None and abs(fol.schedule_error_s) > preset.wind.replan_error_s
                and env.t_s - last_replan >= preset.wind.replan_min_gap_s):
            t0 = time.perf_counter()
            new = replan(env)
            replans.append(dict(t_s=env.t_s, error_s=fol.schedule_error_s, compute_s=time.perf_counter() - t0))
            fol = PathFollower(new, fc, v_trim, pc.FW_AIRSPD_MIN, pc.FW_AIRSPD_MAX, wind_est, dt)
            last_replan = env.t_s
        turn, v = fol.act(env.pose, env.t_s)
        sched.append(fol.schedule_error_s)
        xtrack.append(fol.cross_track_m)
        _, _, term, trunc, info = env.step([turn, v])
        done = term or trunc
    return dict(info=info, schedule_error_s=np.array(sched), cross_track_m=np.array(xtrack), replans=replans,
                terminated=bool(term))


def summary(log: dict, budget: float) -> dict:
    t, pos = log["t_s"], log["pos_cum"]
    e_used = budget - log["energy_left"] if np.isfinite(budget) else np.full(len(t), np.nan)
    return dict(final_pos=float(pos[-1]), escaped=float(log["escaped"][-1]),
                bound=float(1 - log["escaped"][-1]), pos_ratio=float(pos[-1]/max(1 - log["escaped"][-1], 1e-12)),
                searched_km2=float(log["searched_km2"][-1]), t_end_s=float(t[-1]),
                energy_used=float(e_used[-1]) if np.isfinite(budget) else float("nan"),
                pos_per_energy=float(pos[-1]/e_used[-1]) if np.isfinite(budget) and e_used[-1] > 0 else float("nan"),
                max_roll_deg=float(np.degrees(np.abs(log["roll"]).max())))


def run_method(preset, terr, scen, method: str, wind_mode: str = "none", opts: dict | None = None,
               keep_logs: bool = False) -> dict:
    """One scenario, one method: executed (and for planners, planned) results."""
    preset = with_wind(preset, wind_mode)
    exec_env = make_env(preset, terr, record=True, wind="model")
    exec_env.reset(options=dict(scenario=scen))
    true_wind = exec_env.wind.estimate if exec_env.wind is not None else np.zeros(2)
    wind_est = true_wind if wind_mode in ("aware", "replan") else None
    budget = exec_env.energy_budget
    out = dict(method=method, wind_mode=wind_mode, pod=preset.search.pod, scenario=int(scen.seed),
               wind_mps=float(np.hypot(*true_wind)), handover=_handover(exec_env.handover))

    if method == "lawnmower":
        t0 = time.perf_counter()
        wp, linfo = build_lawnmower(exec_env, preset.lawnmower)
        out["compute_s"] = time.perf_counter() - t0
        out["lawnmower"] = linfo
        tr = exec_env.log()
        plan_xy, plan_t = None, None
        mission = Waypoints(np.vstack([tr["poses"][:-1, 0, :2], wp.xy]), np.r_[tr["t_s"][:-1], wp.eta_s])
        replan = None
    else:
        ro, plan_env, plan_t, plan_poses = plan_planner(preset, terr, scen, method, wind_est, opts)
        out["compute_s"] = ro["run_time_s"]
        plan_xy = plan_poses[:, :2]
        out["planned"] = dict(final_pos=float(plan_env.belief.pos), escaped=float(plan_env.belief.escaped),
                              energy_used=float(plan_env.energy_budget - plan_env.energy_left)
                              if plan_env.power is not None else float("nan"), steps=len(ro["plans"]))
        mission = thin_path(plan_xy, plan_t, preset.export.tolerance_m)
        out["thin_max_dev_m"] = max_deviation(plan_xy, mission)

        def replan_fn(env):
            _, _, t, poses = plan_planner(preset, terr, scen, method, wind_est, opts, start_snapshot=env.snapshot())
            return thin_path(poses[:, :2], t, preset.export.tolerance_m)
        replan = replan_fn if wind_mode == "replan" else None

    out["n_waypoints"] = len(mission)
    out["t_plan_s"] = preset.handover.t_plan_s if preset.handover.enabled else 0.0
    fo = follow(exec_env, mission, preset, wind_est, replan)
    log = exec_env.log()
    out |= summary(log, budget)
    se = fo["schedule_error_s"]
    out |= dict(sched_err_mean_s=float(np.mean(se)), sched_err_abs_max_s=float(np.abs(se).max()),
                sched_err_end_s=float(se[-1]), cross_track_mean_m=float(fo["cross_track_m"].mean()),
                cross_track_max_m=float(fo["cross_track_m"].max()), n_replans=len(fo["replans"]),
                terminated=fo["terminated"])
    if keep_logs:
        out["log"] = log
        out["mission"] = mission
        out["plan_xy"], out["plan_t"] = plan_xy, plan_t
    return out


def _handover(h: dict | None) -> dict | None:
    if h is None:
        return None
    return dict(t_s=float(h["t_s"]), pose=[float(v) for v in h["pose"]], error_m=h["error_m"],
                found=float(h["found"]), energy_used=float(h["energy_used"]))


METHODS = ("straight", "lawnmower", "greedy", "tree")
