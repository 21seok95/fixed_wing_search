"""Checks of the R10 set-up (docs/r10_ubuntu.md) before building the pool and training; about 1 min, prints what it finds.

1. Targets that never stop (rl_eval_set.NOSTOP): the nval cases keep the mval case's LKP, heading, launch time and start point, have no
   stopped particles, and each state's speeds stay within mean x (1 +- 0.3) with a spread of about 15 % of the mean.
2. R10 observation (rl_real.RealRLEnv.observation, obs_r10): 6 channels on the 1.6 km map and 5 on the others, a vector of 12, no A in
   the env, and the absolute POS channel never above the particle-position channel (same scale; equal where nothing was searched yet).
3. Speed of the env without A (5 s steps flown straight).

    python -m rl_search.scripts.rl_eval.check_r10
"""
from __future__ import annotations

import time
import warnings

import numpy as np

# the R10 training settings (train_rl flags in docs/r10_ubuntu.md), as signal_check.settings_of reads them from a run's config.json
R10_CONFIG = dict(area_def="realtime_t", blur_tau=3, area_roll_max=5, pos_w=1.5, blur_dead_roll=5, blur_tau_low=15, area_sigma=75,
                  area_sigma_off=150, handover_km=5, no_edge=True, area_img_min=0.75, fp_rate=True, a_weight=0, no_area=True,
                  obs_r10=True, no_stop=True)
STOP = 3


def main():
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import NOSTOP, REGION, make_case, vehicle_config
    from rl_search.rl_real import obs_layout, preload
    from rl_search.rl_real_eval import _env
    from rl_search.scripts.rl_eval.signal_check import settings_of
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    band, frac = NOSTOP["band"], NOSTOP["std_frac"]
    print(f"1. targets that never stop: speed spread {frac:.0%} of the mean, kept within +-{band:.0%}")
    ok = True
    for k in (0, 1, 2):
        cm = make_case(vs, "mval", 3, k)[0]
        cn, scen, vres, hold = make_case(vs, "nval", 3, k)
        same = cm.lkp == cn.lkp and abs(cm.heading - cn.heading) < 1e-9 and abs(cm.t_launch_s - cn.t_launch_s) < 1e-6 and cm.start == cn.start
        stops = int((vres.state == STOP).sum())
        print(f"   {cn.label}: same LKP / heading / launch / start as mval {same}, stopped particle-steps {stops}")
        ok &= same and stops == 0
        S, V = vres.state[1:], vres.speed_kmh[1:].astype(float)
        for m, key in ((0, "fast_kmh"), (1, "slow_kmh"), (2, "off_kmh")):
            mean = float(cn.vc.get(key, getattr(vehicle_config(), key)))
            v = V[(S == m) & (V > 0)]
            if len(v) < 100:
                continue
            inside = np.mean((v >= (1 - band)*mean - .1) & (v <= (1 + band)*mean + .1))
            print(f"      state {m}: mean {mean:.1f} km/h, spread {np.std(v)/mean:.0%} of the mean, inside the band {inside:.1%} "
                  f"(the rest: particles that changed state within the 5 s step before the record)")
            ok &= inside > 0.9
    s = settings_of(R10_CONFIG)
    ch, nvec = obs_layout(s)
    print(f"2. R10 observation: channels {ch}, vector {nvec}")
    env = _env(s)
    cn, scen, vres, hold = make_case(vs, "nval", 3, 0)
    obs, info = env.reset(options=cn.options(scen))
    print(f"   A in the env: {env._area}")
    ok &= env._area is None and ch == {"m1600": 6, "m6400": 5, "m25600": 5, "m51200": 5} and nvec == 12
    t0, n = time.perf_counter(), 0
    while n < 120:
        obs, r, term, trunc, info = env.step(np.array([0.0], np.float32))
        n += 1
        if term or trunc:
            break
    dt = (time.perf_counter() - t0)/n
    for name in ("m1600", "m6400", "m25600", "m51200"):
        o = obs[name]
        assert o.shape[0] == ch[name] and o.shape[1:] == (64, 64)
        below = bool((o[3] <= o[0] + 1e-6).all())
        print(f"   {name}: positions max {o[0].max():.3f}, after lead max {o[1].max():.3f}, POS normalised max {o[2].max():.3f}, "
              f"POS absolute max {o[3].max():.3f}, absolute POS <= positions everywhere {below}")
        ok &= below
    print(f"   vector: {np.round(obs['vec'], 3).tolist()}")
    print(f"3. one 5 s step without A: {dt*1000:.1f} ms (10 min straight: POS {info['POS']:.4f})")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
