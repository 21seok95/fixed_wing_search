"""Exploratory: cost of the pieces a task-level RL environment would use, and of one 5-UAV episode with a random rectangle policy."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from rl_search.accordion import Accordion, simulate, sweep_width
from rl_search.config import get_preset
from rl_search.multi import future_factor, run_team_launch, separation_pairs
from rl_search.rl_env import RLConfig, RLSearchEnv
from rl_search.rl_gym import RLSearchGym, MultiMapExtractor
from rl_search.scenario import load_scenario, make_scenario
from rl_search.scripts.check_accordion import SENSORS
from rl_search.scripts.check_multi import make_envs
from rl_search.scripts.compare_greedy import make_env
from probe import KW, SEARCH, MAX_TR, EXTRA, CYCLE, make_policy


def T(f, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0)/n, out


seed = 301
env = make_env(seed)
dt, scen = T(lambda: make_scenario(env.sampler, seed))
print(f"make_scenario {dt:.1f} s, track {scen.track.shape}, {scen.track.nbytes/1e6:.0f} MB")
p = sorted(Path("rl_search/data/scenarios/rc_cessna/eval").iterdir())[0]
dt, _ = T(lambda: load_scenario(p))
print(f"load_scenario from disk {dt*1000:.0f} ms")
opts = dict(scenario=scen, stage=3)
dt, _ = T(lambda: env.reset(options=opts))
print(f"reset incl. 120 s transit, fast {dt*1000:.0f} ms")

W = sweep_width(env)
task = dict(search_s=SEARCH, max_transit_s=MAX_TR, extra_s=EXTRA)
dt, rects = T(lambda: Accordion(env, "tavg_pca", CYCLE, 200.0, W, **KW).peak_rects(), 3)
print(f"accordion peak_rects, 5 peaks {dt*1000:.0f} ms, {len(rects or [])} rectangles")
if rects:
    dt, _ = T(lambda: simulate(env, rects[0], CYCLE, task), 10)
    print(f"simulate one rectangle (transit + lanes) {dt*1000:.0f} ms")

full = RLSearchEnv(env.terr, env.preset, RLConfig(stage=3, detect_model="lrc", w_area=0.0), seed=seed)
full.reset(options=opts)
dt, _ = T(full._obs, 10)
print(f"observation (four maps) {dt*1000:.0f} ms")
dt, _ = T(lambda: full.step([0.0]), 20)
print(f"full env step 5 s incl. observation {dt*1000:.0f} ms")
env.reset(options=opts)
dt, _ = T(lambda: env.step([0.0]), 100)
print(f"fast env step 5 s, no observation {dt*1000:.1f} ms")

gym_env = RLSearchGym(preset="rc_cessna", overrides=SENSORS["new"], rl=dict(stage=3, detect_model="lrc", w_area=0))
ext = MultiMapExtractor(gym_env.observation_space)
obs = {k: torch.as_tensor(v)[None] for k, v in full._obs().items()}
with torch.no_grad():
    dt, _ = T(lambda: ext(obs), 50)
print(f"extractor forward, CPU 1 thread, 1 sample {dt*1000:.1f} ms; {sum(p.numel() for p in ext.parameters()):,} parameters")

for n in (1, 5):
    envs = make_envs(n, seed)
    W = sweep_width(envs[0])
    pols = [make_policy("rect_density", seed + u, W) for u in range(n)]
    t0 = time.perf_counter()
    res = run_team_launch(envs, pols, opts, 60.0, deconflict=True, hard=False)
    sec = time.perf_counter() - t0
    plans = sum(len(q.tasks) for q in pols)
    dmin, below, enc, _ = separation_pairs(res)
    print(f"{n} UAV episode, random rectangles: {sec:.1f} s, {plans} decisions, {sum(len(a) for a in res['actions'])} 5 s steps, POS {res['pos'][-1]:.3f}, "
          f"min distance {dmin:.0f} m, {below:.0f} s below 300 m")
