"""Speed and memory of the RL environment (docs/rl_mdp_design.md, implementation step "속도와 메모리 측정").

1. One process: time of the 5 s flight step and of building the observation, separately.
2. SB3 SubprocVecEnv with n parallel envs (smooth random actions, curriculum stage 3 mix): steps per second, reset time, memory of
   each worker (RSS, and PSS: shared pages split between the processes) and of the whole machine. The terrain and road network are
   loaded in the parent (rl_real.preload) and the workers are forked, so they share those read-only arrays. Larger n is skipped when
   the memory would not fit.

    python -m rl_search.scripts.rl_eval.speed_check --n 1 4 6 8 12 --steps 300
Results: rl_search/data/real/amboy/rl_eval/speed_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/speed_check.json")


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])/1024
    return float("nan")


def pss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1])/1024
    return float("nan")


def mem_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024
    return float("nan")


def make(stage: int, seed: int):
    def f():
        from stable_baselines3.common.monitor import Monitor
        from rl_search.rl_real import RealRLConfig, make_gym
        return Monitor(make_gym(RealRLConfig(stage=stage), seed=seed))
    return f


def single(steps: int, stage: int) -> dict:
    from rl_search.rl_real import RealRLConfig, RealRLEnv
    env = RealRLEnv(RealRLConfig(stage=stage), seed=123)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    env.reset()
    t_reset = time.perf_counter() - t0
    t_fly, t_obs, a = [], [], 0.0
    for _ in range(steps):
        a = float(np.clip(0.8*a + 0.5*rng.normal(), -1, 1))
        t1 = time.perf_counter()
        _, _, term, _, _ = env.env.step([a])                    # the 5 s flight, detections and A marking
        t2 = time.perf_counter()
        env.observation()
        t3 = time.perf_counter()
        t_fly.append(t2 - t1)
        t_obs.append(t3 - t2)
        if term:
            env.reset()
    return dict(reset_s=t_reset, fly_ms=1000*float(np.mean(t_fly)), obs_ms=1000*float(np.mean(t_obs)), rss_mb=rss_mb(os.getpid()))


def parallel(n: int, steps: int, stage: int) -> dict:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    t0 = time.perf_counter()
    avail0 = mem_available_mb()
    venv = SubprocVecEnv([make(stage, 1000 + i) for i in range(n)], start_method="fork")
    t_make = time.perf_counter() - t0
    t0 = time.perf_counter()
    venv.reset()
    t_reset = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    a = np.zeros((n, 1), np.float32)
    done_eps = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        a = np.clip(0.8*a + 0.5*rng.normal(size=a.shape), -1, 1).astype(np.float32)
        _, _, dones, _ = venv.step(a)
        done_eps += int(dones.sum())
    t_run = time.perf_counter() - t0
    pids = [p.pid for p in venv.processes]
    rss = [rss_mb(p) for p in pids]
    pss = [pss_mb(p) for p in pids]
    avail = mem_available_mb()
    venv.close()
    return dict(n=n, steps_per_env=steps, make_s=t_make, reset_s=t_reset, run_s=t_run, steps_per_s=n*steps/t_run,
                worker_rss_mb=rss, worker_rss_mean_mb=float(np.mean(rss)), worker_pss_mean_mb=float(np.mean(pss)),
                mem_used_mb=avail0 - avail, mem_used_per_env_mb=(avail0 - avail)/n, mem_available_mb_while_running=avail,
                episodes_ended=done_eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 4, 6, 8, 12])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stage", type=int, default=3)
    args = ap.parse_args()
    from rl_search.rl_real import preload
    preload("amboy")                                    # loaded once in the parent, shared by the forked workers
    res = dict(cpu=os.cpu_count(), single=single(args.steps, args.stage), parallel=[])
    print("single:", json.dumps(res["single"]), flush=True)
    per_env = 2000.0
    for n in args.n:
        if res["parallel"]:
            per_env = max(r["mem_used_per_env_mb"] for r in res["parallel"])
        need = n*per_env*1.2 + 1500
        if need > mem_available_mb():
            print(f"n={n}: skipped, needs about {need:.0f} MB, {mem_available_mb():.0f} MB available", flush=True)
            continue
        r = parallel(n, args.steps, args.stage)
        res["parallel"].append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "worker_rss_mb"}), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
