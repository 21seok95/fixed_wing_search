"""Checks of the RL environment rl_search.rl_real (docs/rl_mdp_design.md, "구현 2단계"):

1. replay: SAROPS on validation case (stage, k) gives its actions; the same actions through RealRLEnv must give
   sum(reward) / 100 = A + POS of that SAROPS flight on the planner particles (no hand-over in stages 1-2), with the same path;
   observations are saved at a few times for the figure (obs_figure.py).
2. curriculum: smooth random policy episodes from curriculum starts of one stage: time per step and per reset, value ranges.

    python -m rl_search.scripts.rl_eval.env_check replay 1 1        # validation stage 1, case 2 (SAROPS once, about 5 min)
    python -m rl_search.scripts.rl_eval.env_check curriculum 2
Results go to rl_search/data/real/amboy/rl_eval/env_check/.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
OUT = Path("rl_search/data/real/amboy/rl_eval/env_check")


def replay(stage, k, snap_min=(None, 10.0, 25.0)):
    from rl_search.real_scenario import VehicleScenarios, chungju_preset, terrain_path
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import RealRLEnv
    from rl_search.scenario import ScenarioSampler
    from rl_search.scripts.real_map.web_optimize import run_case
    from rl_search.scripts.rl_eval.sarops_ref import settings
    from rl_search.terrain import load_terrain
    terr = load_terrain(terrain_path(REGION))
    vs = VehicleScenarios(ScenarioSampler(terr, chungju_preset().scenario), vehicle_config(), region=REGION)
    case, scen, vres, hold = make_case(vs, "val", stage, k)
    s = settings()
    view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
    t0 = time.perf_counter()
    out = run_case(REGION, scen, vres, vs.vcfg, view, s, options=case.options(scen), terr=terr)
    t_sarops = time.perf_counter() - t0
    acts = out["actions"][0]
    o = out["opt"]
    env = RealRLEnv(seed=0, terr=terr)
    obs, info = env.reset(options=case.options(scen))
    t_start = env.env.t
    snaps = {"start": (env.env.t, obs, env.env.plant.pose.copy(), env.env.belief.w.copy())}
    want = [m for m in snap_min if m is not None]
    R, path, steps_t = 0.0, [env.env.plant.pose[:2].copy()], []
    for a in acts:
        t1 = time.perf_counter()
        obs, r, term, trunc, info = env.step([a])
        steps_t.append(time.perf_counter() - t1)
        R += r
        path.append(env.env.plant.pose[:2].copy())
        for m in list(want):
            if env.env.t >= t_start + 60*m:
                snaps[f"{m:g}min"] = (env.env.t, obs, env.env.plant.pose.copy(), env.env.belief.w.copy())
                want.remove(m)
        if term or trunc:
            break
    sar_path = np.asarray(out["paths"][0])
    drift = float(np.hypot(*(np.asarray(path[-1]) - sar_path[-1])))
    res = dict(stage=stage, k=k, seed=case.seed, t0_min=case.t0_s/60, steps=len(path) - 1, n_actions=len(acts),
               sum_reward_div100=R/100, rl_A=info["A"], rl_POS=info["POS"], rl_total=info["total"],
               sarops_A=o["area"]["A"], sarops_POS=o["final"], sarops_total=o["area"]["total"], end_drift_m=drift,
               step_ms_mean=1000*float(np.mean(steps_t)), step_ms_p90=1000*float(np.quantile(steps_t, .9)), sarops_s=t_sarops)
    print(json.dumps(res, indent=1), flush=True)
    (OUT/f"replay_s{stage}k{k}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT/f"replay_s{stage}k{k}_obs.npz", path=np.array(path), sar_path=sar_path, lkp=np.array(case.lkp),
                        **{f"{key}__{name}": v for key, (t, ob, pose, w) in snaps.items() for name, v in ob.items()},
                        **{f"{key}__pose": pose for key, (t, ob, pose, w) in snaps.items()},
                        **{f"{key}__t": np.array(t) for key, (t, ob, pose, w) in snaps.items()},
                        **{f"{key}__w": w for key, (t, ob, pose, w) in snaps.items()},
                        track=scen.track, save_dt=np.array(scen.save_dt_s))


def curriculum(stage, n_ep=3, max_steps=120):
    from rl_search.rl_real import RealRLConfig, RealRLEnv
    env = RealRLEnv(RealRLConfig(stage=stage, stage_mix={stage: 1.0}, max_steps=max_steps, reuse=n_ep), seed=stage)
    rng = np.random.default_rng(stage)
    rows = []
    for ep in range(n_ep):
        t1 = time.perf_counter()
        obs, info = env.reset()
        t_reset = time.perf_counter() - t1
        a, R, ts = 0.0, 0.0, []
        while True:
            a = float(np.clip(0.8*a + 0.5*rng.normal(), -1, 1))         # smooth random turns
            t2 = time.perf_counter()
            obs, r, term, trunc, info = env.step([a])
            ts.append(time.perf_counter() - t2)
            R += r
            if term or trunc:
                break
        rng_ = {name: (float(v.min()), float(v.max())) for name, v in obs.items() if name != "vec"}
        rows.append(dict(stage=stage, ep=ep, start_stage=env.start_stage, start_info=env.start_info, t0_min=round((info["t_s"] - 5*info["steps"])/60, 1),
                         reset_s=round(t_reset, 2), step_ms=round(1000*float(np.mean(ts)), 1), steps=info["steps"], R=round(R, 3),
                         A=round(info["A"], 4), POS=round(info["POS"], 4), vec=[round(float(v), 3) for v in obs["vec"]], ranges=rng_))
        print(json.dumps(rows[-1]), flush=True)
    (OUT/f"curriculum_s{stage}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "replay":
        replay(int(sys.argv[2]), int(sys.argv[3]))
    else:
        curriculum(int(sys.argv[2]))
