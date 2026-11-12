"""Does the policy use its large maps? Fly one case with the policy (deterministic) as is, and with the particle channels (POS now, POS
ahead, A value) of chosen maps emptied; compare the paths, actions and POS. Also reports, per minute of the normal flight, how far the
particles are and how much of them each map shows (user 2026-11-12: R5 does not go to far particles in desert validation case 3).

    python -m rl_search.scripts.rl_eval.obs_sensitivity --model rl_search/data/rl_real/amboy_r5/ckpt/model_1000002.zip --split mval --k 2
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

RULES = dict(area_def="realtime_t", area_x=70.0, area_roll_max=5.0, blur_tau=3.0, blur_dead_roll=5.0, blur_tau_low=15.0, area_sigma=75.0, area_w=1.5)
VARIANTS = {"그대로": (), "51.2 km 지도 비움": ("m51200",), "25.6·51.2 km 지도 비움": ("m25600", "m51200"), "1.6·6.4 km 지도 비움": ("m1600", "m6400")}


def fly(model, env, case, scen, blank, track=False):
    e = env.env
    obs, info = env.reset(options=case.options(scen))
    path, acts, rows = [], [], []
    term = trunc = False
    n = 0
    while not (term or trunc):
        if track and n % 12 == 0:               # once a minute: distance to the particles, share each map shows
            b = e.belief
            live = b.w > 0
            P, w = b.positions(e.t)[live], b.w[live]
            x, y, yaw = e.plant.pose
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            c, s = np.cos(yaw), np.sin(yaw)
            fwd, right = (P[:, 0] - x)*c + (P[:, 1] - y)*s, (P[:, 0] - x)*s - (P[:, 1] - y)*c
            shown = {name: float(w[(np.abs(fwd) < side/2) & (np.abs(right) < side/2)].sum()/max(w.sum(), 1e-12)) for name, side, _ in env.cfg.maps}
            rows.append(dict(t=e.t, near_km=float(d.min()/1000) if len(d) else np.nan, mean_km=float(np.average(d, weights=w)/1000) if len(d) else np.nan,
                             edge_km=float((env.half - max(abs(x), abs(y)))/1000), **shown))
        o = {k: v.copy() for k, v in obs.items()}
        for name in blank:
            o[name][:3] = 0.0                    # POS now, POS ahead, A value: nothing there
        a, _ = model.predict(o, deterministic=True)
        acts.append(float(np.ravel(a)[0])); path.append(e.plant.pose[:2].copy())
        obs, r, term, trunc, info = env.step(a)
        n += 1
    return np.array(path), np.array(acts), info, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="mval")
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import RealRLConfig, RealRLEnv, preload
    torch.set_num_threads(1)
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, args.split, 3, args.k)
    model = PPO.load(args.model, device="cpu")
    env = RealRLEnv(RealRLConfig(settings=dict(RULES)), seed=0)
    base = None
    for name, blank in VARIANTS.items():
        path, acts, info, rows = fly(model, env, case, scen, blank, track=not blank)
        if base is None:
            base = (path, acts)
            print(f"case {args.split} k{args.k} ({case.typ}), launch {case.t_launch_s/60:.0f} min after the report, {case.info['launch_dist_m']/1000:.0f} km away")
            print(" min | nearest particle km | weighted mean km | to map edge km | share shown by 1.6 / 6.4 / 25.6 / 51.2 km maps")
            for r in rows[::3]:
                print(f" {(r['t'] - case.t_launch_s)/60:4.0f} | {r['near_km']:6.1f} | {r['mean_km']:6.1f} | {r['edge_km']:5.1f} | "
                      f"{r['m1600']:.0%} / {r['m6400']:.0%} / {r['m25600']:.0%} / {r['m51200']:.0%}")
        m = min(len(acts), len(base[1]))
        diff = np.abs(acts[:m] - base[1][:m])
        sep = np.hypot(*(path[:m] - base[0][:m]).T)/1000
        first = int(np.argmax(diff > 0.05)) if (diff > 0.05).any() else -1
        print(f"{name:20s}: POS {info['POS']:.3f}, A {info['A']:.3f} | mean |action change| {diff.mean():.3f}, first change at decision {first}"
              f" ({first*5/60:.1f} min), path apart by {sep.max():.1f} km at most")


if __name__ == "__main__":
    main()
