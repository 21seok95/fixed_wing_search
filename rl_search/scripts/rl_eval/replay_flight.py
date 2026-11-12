"""Fly one evaluation case again with the policy (deterministic, the same as the evaluation) and keep every 5 s decision: pose, roll, the
policy's action and the action after the edge mask, the reward parts, and the observation maps the policy saw (POS now and A value of
the 1.6 and 6.4 km maps, heading up), plus the 1 s camera samples. For looking at single moments together with the user (2026-11-12).

    python -m rl_search.scripts.rl_eval.replay_flight --model rl_search/data/rl_real/amboy_r4/ckpt/model_500004.zip --split test --k 12 \
        --rules r4 --out rl_search/data/real/amboy/rl_eval/report/replay_test_k12_r4.npz
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

RULES = dict(r4=dict(area_def="realtime_t", area_x=70.0, area_roll_max=5.0, blur_tau=3.0, blur_dead_roll=5.0, blur_tau_low=15.0, area_sigma=75.0,
                     area_w=1.5),
             r3=dict(area_def="realtime_t", area_x=70.0, area_roll_max=5.0, blur_tau=3.0, blur_dead_roll=5.0, area_w=1.5),
             r2=dict(area_def="realtime_t", area_x=70.0, area_roll_max=10.0, blur_tau=15.0, area_w=1.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--rules", default="r4", choices=tuple(RULES))
    ap.add_argument("--config", default=None, help="a training run's config.json: fly with its settings instead of --rules (e.g. amboy_r6)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.config:
        import json
        from rl_search.scripts.rl_eval.signal_check import settings_of
        RULES["config"] = settings_of(json.load(open(args.config)))
        args.rules = "config"
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import RealRLConfig, RealRLEnv, preload
    from rl_search.scripts.real_map.web_optimize import footprints
    torch.set_num_threads(1)
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, args.split, 3, args.k)
    model = PPO.load(args.model, device="cpu")
    env = RealRLEnv(RealRLConfig(settings=dict(RULES[args.rules])), seed=0)
    e = env.env
    e.sample_log = []
    obs, info = env.reset(options=case.options(scen))
    rec = dict(t=[], x=[], y=[], yaw=[], roll=[], a=[], a_applied=[], dA=[], dPOS=[], reward=[], POS=[], A=[])
    maps = dict(m1600_pos=[], m1600_A=[], m6400_pos=[], m6400_A=[], m25600_pos=[], m25600_A=[])
    term = trunc = False
    while not (term or trunc):
        for name in ("m1600", "m6400", "m25600"):
            maps[f"{name}_pos"].append(obs[name][0].astype(np.float16)); maps[f"{name}_A"].append(obs[name][2].astype(np.float16))
        a, _ = model.predict(obs, deterministic=True)
        rec["t"].append(e.t); rec["x"].append(e.plant.pose[0]); rec["y"].append(e.plant.pose[1]); rec["yaw"].append(e.plant.pose[2])
        rec["roll"].append(float(getattr(e.plant, "phi", 0.0)))
        obs, r, term, trunc, info = env.step(a)
        rec["a"].append(float(np.ravel(a)[0])); rec["a_applied"].append(float(getattr(e, "a_applied", np.ravel(a)[0])))
        rec["dA"].append(info["dA"]); rec["dPOS"].append(info["dPOS"]); rec["reward"].append(float(r)); rec["POS"].append(info["POS"]); rec["A"].append(info["A"])
    log = np.array(e.sample_log)
    quad = footprints(log, e.detector.cfg)
    np.savez_compressed(args.out, **{k: np.asarray(v) for k, v in rec.items()}, **{k: np.stack(v) for k, v in maps.items()},
                        log=log.astype(np.float32), quad=quad.astype(np.float32), t_launch=case.t_launch_s,
                        rate_max=env._area.rate_max_dps if env._area.rate_max_dps is not None else np.inf, v=e.v, lkp=np.asarray(case.lkp))
    print(f"replayed {args.split} k{args.k} with {args.model}: steps {len(rec['t'])}, belief POS {info['POS']:.3f}, A {info['A']:.3f}, saved {args.out}")


if __name__ == "__main__":
    main()
