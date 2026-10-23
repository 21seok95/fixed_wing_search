"""Stage 2 check: SearchEnv + KinematicPlant against the original uav_search mission loop.

1. Regression: with exits off, the env flown by a forked planner must match uav_search.run_mission
   with the original planner on the same scenario and config (POS curve and poses).
2. Exits on: 1 = POS + remaining + escaped at every step; prints final POS and the bound 1 - escaped.
3. Env cost alone: one episode with random turns (no planner).

Missions are saved next to the exported scenario for mission_viewer.py:
    search_ref_<planner>   original uav_search
    search_env_<planner>   SearchEnv with exits

Run from the repository root:
    python -m rl_search.scripts.check_env rl_search/data/scenarios/eval/s1000001
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np

import uav_search as ref
from rl_search.env import SearchEnv, default_search_config
from rl_search.planners import PLANNERS, run_episode, save_mission
from rl_search.scenario import ScenarioConfig, load_scenario
from rl_search.scripts.export_scenario import export, export_dir
from rl_search.terrain import load_terrain

PLANNER_OPTS = {"greedy": {}, "straight": {}, "tree": dict(horizon=6, terminal_weight=1.0, terminal_bw_m=600.0, beam=50)}


def reference(run_dir: Path, cfg, scen, name: str):
    rcfg = ref.SearchConfig(launch_delay_min=cfg.launch_delay_min, flight_min=cfg.flight_min,
                            uav_speed_mps=cfg.uav_speed_mps, dt_s=cfg.dt_s, max_turn_deg=cfg.max_turn_deg,
                            bandwidth_m=cfg.bandwidth_m, pod=cfg.pod, start_x=float(scen.start[0]),
                            start_y=float(scen.start[1]), start_heading_deg=scen.start_heading_deg)
    data, belief, rcfg = ref.setup(run_dir, rcfg)
    planner = ref.PLANNERS[name](rcfg, 5, **PLANNER_OPTS[name])
    start = np.array([[rcfg.start_x, rcfg.start_y, np.deg2rad(rcfg.start_heading_deg)]])
    t0 = time.perf_counter()
    log = ref.run_mission(belief, rcfg, planner, start, ref.Coverage(data, rcfg))
    return log, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path, nargs="?", default=Path("rl_search/data/scenarios/eval/s1000001"))
    ap.add_argument("--planner", choices=sorted(PLANNERS), default="greedy")
    args = ap.parse_args()

    meta = json.loads((args.scenario/"meta.json").read_text())
    scfg = ScenarioConfig.from_dict(meta["config"])
    terr = load_terrain(Path(meta["terrain"]))
    scen = load_scenario(args.scenario, mmap=False)
    cfg = default_search_config(scfg)
    run_dir = export(scen, terr, export_dir(args.scenario))
    opts = PLANNER_OPTS[args.planner]
    print(f"Scenario {args.scenario.name}: {cfg.n_steps} steps of {cfg.step_m:.1f} m, planner {args.planner}")

    ref_log, ref_s = reference(run_dir, cfg, scen, args.planner)
    env = SearchEnv(terr, scfg, cfg, use_exits=False, record=True)
    planner = PLANNERS[args.planner](cfg, 5, **opts)
    log = run_episode(env, planner, options=dict(scenario=scen))
    n = min(len(log["pos_cum"]), len(ref_log["pos_cum"]))
    d_pos = np.abs(log["pos_cum"][:n] - ref_log["pos_cum"][:n]).max()
    d_xy = np.abs(log["poses"][:n, 0, :2] - ref_log["poses"][:n, 0, :2]).max()
    d_cov = abs(log["searched_km2"][n-1] - ref_log["searched_km2"][n-1])
    ok = d_pos < 1e-9 and d_xy < 1e-6 and len(log["pos_cum"]) == len(ref_log["pos_cum"])
    print(f"1. Regression (exits off): final POS env {log['pos_cum'][-1]:.6f} vs uav_search "
          f"{ref_log['pos_cum'][-1]:.6f} | max |dPOS| {d_pos:.1e}, max |dxy| {d_xy:.1e} m, "
          f"d searched {d_cov:.1e} km2 | {'OK' if ok else 'MISMATCH'}")
    print(f"   run time env {log['info']['run_time_s']:.1f} s vs uav_search {ref_s:.1f} s")
    ref_out = run_dir/f"search_ref_{args.planner}"
    ref_out.mkdir(parents=True, exist_ok=True)
    ref_meta = dict(asdict(cfg), planner=args.planner, horizon=planner.horizon,
                    n_headings=5, start_x=float(scen.start[0]), start_y=float(scen.start[1]))
    np.savez_compressed(ref_out/"mission.npz", meta=json.dumps(ref_meta), **ref_log)

    env = SearchEnv(terr, scfg, cfg, use_exits=True, record=True)
    planner = PLANNERS[args.planner](cfg, 5, **opts)
    log = run_episode(env, planner, options=dict(scenario=scen))
    remaining = log["weights"].astype(float).sum(axis=1)/scen.track.shape[1]
    err = np.abs(log["pos_cum"] + remaining + log["escaped"] - 1).max()
    info = log["info"]
    print(f"2. Exits on: final POS {info['pos']:.4f}, escaped {info['escaped']:.4f}, bound 1 - escaped "
          f"{1 - info['escaped']:.4f} | max |POS + remaining + escaped - 1| {err:.1e} "
          f"({'OK' if err < 1e-5 else 'VIOLATED'}) | terminated {info['terminated']}")
    out = run_dir/f"search_env_{args.planner}"
    save_mission(out, log, cfg, planner, dict(use_exits=True, scenario=args.scenario.name))

    env = SearchEnv(terr, scfg, cfg, use_exits=True, record=False)
    rng = np.random.default_rng(0)
    env.reset(options=dict(scenario=scen))
    t0, steps, done = time.perf_counter(), 0, False
    while not done:
        _, _, term, trunc, _ = env.step([rng.uniform(-1, 1)*env.plant.max_turn])
        steps += 1
        done = term or trunc
    dt = time.perf_counter() - t0
    print(f"3. Env alone (random turns): {steps} steps in {dt:.2f} s = {steps/dt:.0f} steps/s")
    print(f"View: python mission_viewer.py {out} --compare ref_{args.planner}")


if __name__ == "__main__":
    main()
