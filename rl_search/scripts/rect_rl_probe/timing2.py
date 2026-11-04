"""Exploratory: cost of the full environment step over the flight, and of one observation with and without the reward zone."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import warnings

import numpy as np

from rl_search.patterns import lead_action
from rl_search.rl_env import RLConfig, RLSearchEnv
from rl_search.scenario import make_scenario
from rl_search.scripts.compare_greedy import make_env

warnings.filterwarnings("ignore")
for seed in (301, 318):
    env = make_env(seed)
    scen = make_scenario(env.sampler, seed)
    full = RLSearchEnv(env.terr, env.preset, RLConfig(stage=3, detect_model="lrc", w_area=0.0), seed=seed)
    full.reset(options=dict(scenario=scen, stage=3))
    rows = {}
    while True:
        s0 = time.perf_counter()
        _, _, term, _, _ = full.step([lead_action(full)*0.5])
        rows.setdefault(int(full.t//600), []).append(time.perf_counter() - s0)
        if term:
            break
    print(seed, "full env step ms by 10-min bin:", {k: round(float(1000*np.mean(v)), 1) for k, v in rows.items()})
    ts = []
    for _ in range(5):
        full._zone_key = None      # the reward zone is recomputed every 10 s of flight
        s0 = time.perf_counter()
        full._obs()
        ts.append(time.perf_counter() - s0)
    print(seed, "observation with the zone recomputed, ms:", round(float(1000*np.mean(ts)), 1))
    ts = []
    for _ in range(5):
        s0 = time.perf_counter()
        full._obs()
        ts.append(time.perf_counter() - s0)
    print(seed, "observation with the zone cached, ms:", round(float(1000*np.mean(ts)), 1))
