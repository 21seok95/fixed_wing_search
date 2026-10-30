# rl_search

Lawnmower vs RL heading search for a fixed-wing UAV on a fixed 60 km x 60 km terrain.
The root scripts (`uav_search.py`, `battlefield_infiltrator_demo.py`, viewers) are kept
unchanged as the reference; the parts that need changes are forked here. Plan and results:
[PLAN.md](PLAN.md).

## Setup

```bash
source envs/activate_train.sh        # conda env fw-train (see ENV_SETUP.md), cd to the repo, PYTHONPATH set
```

All commands below run from the repository root. Outputs go to `rl_search/data/` (not tracked by git).

## Settings: presets (`config.py`)

Every setting of a run lives in one `Preset`: scenario, search (mission), plant, energy, handover,
export, follow, lawnmower and wind sections. Built-in presets:

| preset | use | values |
|---|---|---|
| `rc_cessna` (default) | stage 4+ | 15 m/s, 4 s steps, 60 min energy budget, POD 0.3, `FixedWingPlant`, 120 s handover |
| `fast` | stage 0-2, regression | 33.3 m/s, 2 s steps, instant turns, no energy model, no handover |

Change anything without editing code:

```bash
--preset rc_cessna --set plant.FW_R_LIM=45 search.pod=1 energy.induced_share=0.4 handover.t_plan_s=90
--preset my_preset.json                       # config.save_preset() writes one to edit
```

Scenarios store their (raw) preset in `meta.json`; scripts that take a scenario read it from there
(`--set` still applies). Single sources: the cruise speed is `plant.FW_AIRSPD_TRIM`, the flight time is
`energy.budget_s`, and `search.max_turn_deg = None` is measured from the plant, so changing a PX4
parameter updates the planner lattice automatically.

Swappable parts are registered in one place each: `plants.PLANTS` (`plant.kind`), `planners.PLANNERS`
(planner methods; an RL policy wrapped as a `Planner` plugs in here), `runner.METHODS` and
`runner.WIND_MODES`.

## Step 0: terrain

```bash
python -m rl_search.scripts.make_terrain
```

Procedural terrain: ridged mountain ranges, rolling hills, rivers and streams from the drainage
network (deep basins become lakes, shallow ones plains), fords, forest on high and steep ground.
Mountains above the no-go elevation (top 8%), rivers, lakes and slopes of 32 degrees or more are
impassable for the target. Writes `rl_search/data/terrain_seed7_60km.npz` and an overview figure.

## Step 1: scenarios

A scenario is one target simulation plus a UAV launch pose on the map edge `scenario.start_dist_m`
from the LKP, heading at it (rc_cessna: LKP 5-15 km away, target 15-30 km/h, 70 min of tracks;
LKPs without a launch point in range are never drawn). Training scenarios are generated on the fly
(`ScenarioGenerator`); only the fixed-seed evaluation set is saved:

```bash
python -m rl_search.scripts.make_scenarios --preset rc_cessna --n-eval 50 --n-stats 100
python -m rl_search.scripts.make_scenarios --preset fast --n-eval 5        # stage 0-2 set
```

Writes `rl_search/data/scenarios/<preset>/eval/s<seed>/` (track.npy, exit_step.npy, meta.json),
`eval_overview.png`, and the distribution check (`train_stats.png`, median and p90 per panel).

## Step 2: environment and regression

The mission layer (`mission.py`) is separate from the vehicle layer (`plants.py`). `env.SearchEnv`
ties them together with `reset(seed, options)` / `step(action)`; `action = [turn]` or
`[turn, airspeed]`, the reward is the probability found in the step. Build it from a preset:

```python
from rl_search.config import get_preset
from rl_search.env import make_env
env = make_env(get_preset("rc_cessna"), load_terrain(path))
obs, info = env.reset(options=dict(scenario=load_scenario(eval_dir)))   # returns at handover
obs, reward, terminated, truncated, info = env.step([turn])
```

Regression against the original `uav_search.py` (fast preset; must match exactly):

```bash
python -m rl_search.scripts.check_env rl_search/data/scenarios/fast/eval/s1000001 --planner greedy
python mission_viewer.py rl_search/data/export/fast/s1000001/search_env_greedy --compare ref_greedy
```

## Step 4: fixed-wing plant and energy

`FixedWingPlant` follows the PX4 fixed-wing control chain with PX4 parameter names: course setpoint
-> NPFG-like second-order guidance (`NPFG_PERIOD`, `NPFG_DAMPING`) -> roll loop (`FW_R_TC`,
`FW_R_RMAX`, `FW_R_LIM`) -> coordinated turn, integrated at 0.1 s. `energy.PowerModel` charges
P(phi, V) = (1-k)(V/V_trim)^3 + k (V_trim/V)/cos^2(phi) per inner step; the episode is truncated
when the energy budget is used up.

```bash
python -m rl_search.scripts.check_plant                       # step responses, turn radius, Kinematic vs FixedWing, max_turn
python -m rl_search.scripts.check_plant --set plant.FW_R_LIM=45
```

Writes `rl_search/data/plant_check/<preset>/` (step_response.png, steady_turn.png, kin_vs_fw.png, plant_check.json).

## Step 5: handover and waypoint export

`reset` flies the transit (auto-steer to the LKP, sensor on, energy charged) until
`handover.t_plan_s`; planners act from there. `runner.run_method` rolls a method out to the end of
the energy, thins the path into waypoints with ETAs (`waypoints.thin_path`, `export.tolerance_m`)
and flies them with `follower.PathFollower` in a fresh env, so planned and executed POS can be
compared.

```bash
python -m rl_search.scripts.export_plan rl_search/data/scenarios/rc_cessna/eval/s1000000 --method greedy
python -m rl_search.scripts.export_plan <scenario> --method lawnmower --max-minutes 14   # short mission for SITL
```

Writes `rl_search/data/plans/<preset>/<scenario>/<method>/`: `mission.csv` (x, y, eta, lat, lon),
`mission.plan` (QGC; takeoff, waypoints, PX4 fixed-wing landing pattern), `summary.txt/json`, `plan.png`,
and a viewer mission `rl_search/data/export/<preset>/<scenario>/search_<method>_exec`.

Fly the exported plan in PX4 SITL + Gazebo (fw-gz env; an isolated headless instance does not
touch an interactive session):

```bash
envs/run_sitl_headless.sh 1                                    # PX4 instance 1, MAVLink port 14541
source ~/.local/miniforge3/etc/profile.d/conda.sh && conda activate fw-gz
python rl_search/scripts/sitl_fly_plan.py rl_search/data/plans/rc_cessna/s1000000/greedy_14min/mission.plan --url udpin:0.0.0.0:14541
envs/run_sitl_headless.sh stop 1
```

## Step 3: lawnmower and method comparison

`lawnmower.build_lawnmower` fixes a search rectangle at handover (80% HDR of the in-map particles
until the end of the energy, aligned with the target heading, clipped to the map) and lays lanes
at the footprint width, skipping k = max(1, ceil(2 R_eff / spacing)) lanes. It is flown by the same
path follower as the planner missions.

```bash
python -m rl_search.scripts.compare --n 50                                            # POD 1 and 0.3
python -m rl_search.scripts.compare --methods greedy tree --tag plants \
    --configs fw:plant.kind=fixed_wing kin:plant.kind=kinematic                        # stage 4 table
python -m rl_search.scripts.compare --methods greedy lawnmower --tag wind \
    --configs pod0.3: --wind-modes none unaware aware replan                           # stage 6
```

A config is `NAME:key=value,key=value`, so any preset setting can be compared. Writes
`rl_search/data/compare/<tag>/` (table.md, table.csv, results.json, pos_time.png, pos_energy.png,
scenarios.png); `--save-missions N` adds viewer missions for the first N scenarios.

## Step 6: wind

`wind.WindModel` draws a constant wind (0-6 m/s, random direction) per scenario plus an
Ornstein-Uhlenbeck gust; ground velocity = air velocity + wind. Modes in `runner`: `unaware`
(plan without wind), `aware` (plan with the wind estimate, crab-angle correction, airspeed
scheduled to the ETA within 11-20 m/s), `replan` (aware + new plan from the current state when the
schedule error exceeds `wind.replan_error_s`).

## RL environment and PPO (design log: `todo_response.md`)

`rl_env.RLSearchEnv` implements the agreed RL design (numpy only): 5 s decisions with a course target
(+-30 deg per decision), the interval sensor (25 m cells under the footprint every 1 s, one POD update
per interval), the expanded-HDR area reward, reward `w_pos dPOS/pos_ref + w_area dA/area_ref`, four
2-channel 64x64 maps (near 1.6 km, mid 6.4 km, wide 25.6 km UAV-centred; global 60 km fixed) and an
8-vector. Curriculum stage 1/2 start near the prior at t0 = 2 min / 10-15 min; stage 3 is the full
launch + 120 s transit. `rl_gym.RLSearchGym` wraps it for Gymnasium/SB3 and `rl_gym.MultiMapExtractor`
is the 4-CNN feature extractor.

```bash
python -m rl_search.scripts.make_terrain                     # once per machine (data/ is not in git)
python -m rl_search.scripts.check_rl_env --stages 1 2 3      # step 9 checks, no torch needed
python -m rl_search.scripts.check_hdr                        # prior spread and reward zone
```

Training needs `pip install -r rl_search/requirements-rl.txt` plus torch:

- **Mac (check only):** `pip install torch`, then a short CPU run
  `python -m rl_search.scripts.train_ppo --tag smoke --n-envs 1 --n-steps 256 --total-steps 2048 --device cpu`
  and `python -m rl_search.scripts.check_rl_env --model rl_search/data/rl/smoke/model_final.zip`.
- **Ubuntu + RTX 2070:** install a CUDA build of torch (check `nvidia-smi` for the driver's CUDA version
  and pick the matching command on pytorch.org), check `python -c "import torch; print(torch.cuda.is_available())"`, then
  `python -m rl_search.scripts.train_ppo --tag stage1 --n-envs <physical cores> --total-steps 2000000 --device cuda`.
  The environment runs on the CPU and is the bottleneck (about 12-23 ms per step on one core), so
  `--n-envs` matters more than the GPU. `tensorboard --logdir rl_search/data/rl` shows episode POS,
  area, exit rate and the course-target gap.
