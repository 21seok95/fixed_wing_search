# Environment setup

Conda environments under `~/.local/miniforge3` (no system `sudo` required).

| Env | Purpose | Activate |
|-----|---------|----------|
| `fw-train` | `rl_search` scenarios, `SearchEnv`, planners, ulog fitting, future RL | `source envs/activate_train.sh` |
| `fw-gz` | Gazebo Harmonic 8 + PX4 v1.15.4 SITL (`gz_rc_cessna`, `gz_advanced_plane`, SIH) | `source envs/activate_gz.sh` |

QGroundControl v4.4.4 is installed separately; run `qgroundcontrol` from any terminal.

## Training (`fw-train`)

- Packages: numpy, scipy, matplotlib, pyulog, gymnasium, stable-baselines3, torch (CPU)
- Terrain: `rl_search/data/terrain_seed7_60km.npz`
- Eval sets (one folder per preset, see `rl_search/config.py`):
  - `rl_search/data/scenarios/rc_cessna/eval/`: 50 scenarios, stage 4+ settings (15 m/s, LKP 5-15 km)
  - `rl_search/data/scenarios/fast/eval/`: 50 scenarios, stage 0-2 settings (33.3 m/s), kept for the regression

```bash
source envs/activate_train.sh
python -m rl_search.scripts.check_env                       # stage 2 regression vs uav_search.py (fast preset)
python -m rl_search.scripts.check_plant                     # stage 4 plant checks
python -m rl_search.scripts.compare --n 5 --tag quick       # stage 3 comparison on 5 scenarios
```

`compare.py` runs one process per CPU core with one BLAS thread each (set in the script).

## PX4 + Gazebo (`fw-gz`)

```bash
source envs/activate_gz.sh
make px4_sitl gz_rc_cessna                # PLAN validation airframe, with GUI
HEADLESS=1 make px4_sitl gz_rc_cessna
make px4_sitl gz_advanced_plane           # fallback if aerodynamics need matching
make px4_sitl sihsim_airplane             # no Gazebo, fastest
```

Run these in an interactive terminal. Started in the background without stdin, the
`pxh>` console loops and writes gigabytes of log; if you must background it, feed it a
blocking stdin: `tail -f /dev/null | HEADLESS=1 make px4_sitl gz_rc_cessna > log 2>&1 &`.

MAVLink: QGC connects automatically on UDP 14550; scripts use `udpin:0.0.0.0:14540`
(instance N: 14540 + N). `pymavlink` and `mavsdk` are in `fw-gz`; note `mavsdk` 4.x is the new
native binding (the old gRPC API is `mavsdk-grpc`), so the repo scripts use `pymavlink`.
Flight logs (`.ulg`) land in `~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/` (instance N:
`instance_N/log/`) for fitting `FixedWingPlant` parameters with `pyulog`.

### Automated SITL check (stage 5 export -> PX4)

```bash
envs/run_sitl_headless.sh 1        # PX4 instance 1 + its own headless Gazebo (GZ_PARTITION fwverify_1), daemon mode
source ~/.local/miniforge3/etc/profile.d/conda.sh && conda activate fw-gz
python rl_search/scripts/sitl_fly_plan.py <plan dir>/mission.plan --url udpin:0.0.0.0:14541
envs/run_sitl_headless.sh stop 1
```

The script starts the Gazebo server first and waits for its clock (under CPU load the PX4 spawn
call otherwise times out), runs PX4 with `-d` (no `pxh>` console, so no log flood), and does not
touch an interactive instance 0. `sitl_fly_plan.py` uploads the `.plan` with the MAVLink mission
protocol, arms, switches to AUTO.MISSION, logs the track and writes `sitl_report.json`.
PX4 fixed-wing missions must end with a landing (`MIS_TKO_LAND_REQ 2` in `rc.fw_defaults`); the
exporter appends `DO_LAND_START`, an approach waypoint and `NAV_LAND` at home (`export.finish`).

Rebuild from a clean shell (the script clears paths left by other envs):

```bash
source envs/activate_gz.sh
rm -rf build/px4_sitl_default
make px4_sitl gz_rc_cessna     # gz targets ignore DONT_RUN and start the sim after building
```

Build workarounds baked into `activate_gz.sh`: `CMAKE_POLICY_VERSION_MINIMUM=3.5`
(CMake 4), and non-fatal `dangling-pointer` / `deprecated-declarations` warnings
(conda gcc 15 and protobuf vs PX4's `-Werror`).

`envs/activate_sim.sh` and the `fw-sim` env (Gazebo Classic 11) are retired: PX4's
Classic plugins crash under conda Gazebo 11, and the PLAN airframe is Harmonic-only.

## Mapping to PLAN

- Stages 0-2: `fw-train`, verified by `check_env.py` (fast preset, exact match with `uav_search.py`).
- Stage 4 (`FixedWingPlant` with PX4 parameter names): `check_plant.py`; next, fit against `gz_rc_cessna` ulogs.
- Stage 5 (QGC `.plan` waypoint export): `export_plan.py`, then QGC (File -> Open) or `sitl_fly_plan.py` on `gz_rc_cessna`.
- Stages 3 and 6 (lawnmower, wind): `compare.py` in `fw-train`.
