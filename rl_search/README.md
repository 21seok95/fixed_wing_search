# rl_search

Lawnmower vs RL heading search for a fixed-wing UAV on a fixed 60 km x 60 km terrain.
The root scripts (`uav_search.py`, `battlefield_infiltrator_demo.py`, viewers) are kept
unchanged as the reference; the parts that need changes are forked here.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r rl_search/requirements.txt
```

## Step 0: terrain

Generate the terrain once (about 20 s; output is not tracked by git):

```bash
.venv/bin/python -m rl_search.scripts.make_terrain
```

The terrain is procedural: ridged mountain ranges on uplands, rolling hills, rivers and
streams from the drainage network of the elevation (deep basins become lakes, shallow ones
sediment-filled plains), fords along the rivers, and forest on high and steep ground.
Mountains above the no-go elevation (top 8% of the map), rivers, lakes and slopes of 32
degrees or more are impassable for the target.

This writes `rl_search/data/terrain_seed7_60km.npz` (base layers only; slopes, cover and
clearance are derived on load by `rl_search.terrain.load_terrain`) and an overview figure.

## Step 1: scenarios

A scenario is one target simulation plus a UAV launch pose. The LKP is a random passable
point (200 m from any no-go cell, 1 km inside the map) with a random heading and a target
speed of 20-40 km/h. 5000 particles are simulated for 60 min at 5 s steps and stored every
10 s; particles that leave the map get an exit index and are frozen. The UAV launches from
the map edge 10-30 km from the LKP, heading at it. t = 0 is both the LKP and the launch.

Training scenarios are generated on the fly (`rl_search.scenario.ScenarioGenerator`, about
7-9 s each on one core; `reuse > 1` reuses the tracks with new launch points). Only the
fixed-seed evaluation set is saved:

```bash
.venv/bin/python -m rl_search.scripts.make_scenarios --n-eval 5 --n-stats 100   # Ubuntu: --n-eval 50
```

This writes `rl_search/data/scenarios/eval/s<seed>/` (track.npy, exit_step.npy, meta.json),
`eval_overview.png`, and the distribution check of 100 generator scenarios
(`train_stats.png`, `train_stats.json`; tracks not saved).

Check one scenario with the existing planner (prints the exact `uav_search.py` command):

```bash
.venv/bin/python -m rl_search.scripts.export_scenario rl_search/data/scenarios/eval/s1000001
```

## Step 2: environment

The mission layer (`mission.py`: belief with map exits, `NadirSensor`, detection, coverage) is
separate from the vehicle layer (`plants.py`: `reset(pose)`, `step(turn) -> pose`, `max_turn`).
`env.SearchEnv` ties them together with `reset(seed, options)` / `step(action)`; the action's
first element is the heading change (rad) for one 2 s step, the reward is the probability found
in that step. `truncated` = flight budget used, `terminated` = UAV left the map. With exits on,
1 = POS + remaining + escaped at every step.

```python
env = SearchEnv(load_terrain(path))
obs, info = env.reset()                       # fresh generated scenario
obs, reward, terminated, truncated, info = env.step([turn])
env.reset(options=dict(scenario=load_scenario(eval_dir)))   # evaluation scenario
```

Check against the original `uav_search.py` (exits off must match exactly; exits on must
conserve mass) and save both missions for the viewer:

```bash
.venv/bin/python -m rl_search.scripts.check_env rl_search/data/scenarios/eval/s1000001 --planner greedy
python mission_viewer.py rl_search/data/export/s1000001/search_env_greedy --compare ref_greedy
```
