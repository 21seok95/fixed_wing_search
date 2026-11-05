"""Exploratory run: Greedy90 teams vs task-mode SAROPS on the centre scenarios (400-409), N = 1, 3, 5.

Nothing in the repository is changed: the team adapter for Greedy90 is patched in here.
Same scenarios, launch gap, sensor and held-out particles as rl_search/scripts/compare_center.py.
Each finished job is appended to center_greedy.jsonl.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"
OUT = HERE/"center_greedy.jsonl"
SEEDS = list(range(400, 410))


def install_team_greedy():
    """Teach rl_search.multi how a Greedy90 planner commits: it plans when its queue is empty and is committed to
    the rest of its 30 s segment. Team members see that segment through future_factor, as they see committed lanes."""
    from rl_search import multi
    from rl_search.patterns_v2 import Greedy90

    class TeamGreedy(Greedy90):
        def __init__(self):
            super().__init__()
            self.last, self.t_last, self.n_plans, self.n_lead = 0.0, -np.inf, 0, 0

        def __call__(self, env):
            planned = not self.queue
            q0 = len(self.queue)
            a = super().__call__(env)
            if planned:
                self.n_plans += 1
                if not self.queue and q0 == 0:   # no reachable mass in 90 s: lead pursuit, plan again next step
                    self.n_lead += 1
            self.last, self.t_last = a, env.t
            return a

        def committed(self, env):
            return ([self.last] if abs(self.t_last - env.t) < 1e-6 else []) + list(self.queue)

    orig_due, orig_future = multi.due, multi.future_factor

    def due(pol, env):
        return not pol.queue if isinstance(pol, TeamGreedy) else orig_due(pol, env)

    def future_factor(env, pol):
        if not isinstance(pol, TeamGreedy):
            return orig_future(env, pol)
        acts = pol.committed(env)
        if not acts:
            return np.ones(env.belief.n), None
        snap, s0 = env.snapshot(), env.surv.copy()
        skip, env.skip_cells = env.skip_cells, True
        ts, xs = [env.t], [env.plant.pose[:2].copy()]
        for a in acts:
            _, _, term, _, _ = env.step([a])
            ts.append(env.t)
            xs.append(env.plant.pose[:2].copy())
            if term:
                break
        out = np.where(s0 > 0, env.surv/np.maximum(s0, 1e-300), 1.0)
        env.skip_cells = skip
        env.restore(snap)
        return out, (np.array(ts), np.array(xs))

    multi.due, multi.future_factor = due, future_factor
    return TeamGreedy


def run_job(job):
    seed, method, n, sep = job
    warnings.filterwarnings("ignore")
    TeamGreedy = install_team_greedy()
    from rl_search import multi
    from rl_search.accordion import sweep_width
    from rl_search.sarops import SaropsPlanner
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center import center_scenario
    t0 = time.perf_counter()
    probe = make_envs(1, seed)[0]
    scen = center_scenario(probe.sampler, seed)
    opts = dict(scenario=scen, stage=3)
    hold = dict(scenario=holdout_scenario(probe, scen, seed + 10**6), stage=3)
    if method == "greedy":
        pols = [TeamGreedy() for _ in range(n)]
    else:
        W = sweep_width(probe)
        pols = [SaropsPlanner(W=W, n_starts=5 if sep else 3, big_jump=True, search_s=600.0, max_transit_s=600.0,
                              lane_mode="cover") for _ in range(n)]
    res = multi.run_team_launch(make_envs(n, seed), pols, opts, 60.0, deconflict=sep, hard=sep)
    h = multi.run_team_launch(make_envs(n, seed), None, hold, 60.0, actions=res["actions"])
    drift = max(float(np.hypot(*(a[-1] - b[-1]))) for a, b in zip(res["xy"], h["xy"]))
    if drift > 1e-6:
        raise RuntimeError(f"replay left the planned path by {drift} m")
    dmin, below, enc, _ = multi.separation_pairs(res)
    out = dict(seed=seed, method=method, n=n, sep=sep, pos=float(res["pos"][-1]), hold=float(h["pos"][-1]),
               hold_t=[float(np.interp(m*60, h["t"] - h["t"][0], h["pos"])) for m in range(0, 61, 5)],
               dmin=float(dmin) if np.isfinite(dmin) else None, below=float(below), enc=int(enc),
               heading=scen.heading_deg, speed_kmh=scen.speed_mps*3.6, start_lkp_km=scen.metrics["start_lkp_km"],
               escaped=scen.metrics["escaped_end"], lkp=scen.lkp.tolist(), start=scen.start.tolist(),
               n_plans=int(sum(getattr(p, "n_plans", 0) for p in pols)),
               n_lead=int(sum(getattr(p, "n_lead", 0) for p in pols)),
               xy=[x[::2].round(0).tolist() for x in res["xy"]], tt=[t[::2].tolist() for t in res["tt"]],
               sec=round(time.perf_counter() - t0, 1))
    with open(OUT, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"seed {seed} {method} N={n} sep={sep}: held-out {out['hold']:.3f}, plan set {out['pos']:.3f}, "
          f"close {below:.0f} s, {out['sec']:.0f} s", flush=True)
    return out


def main():
    jobs = []
    jobs += [(s, "greedy", n, False) for s in SEEDS for n in (5, 3, 1)]
    jobs += [(s, "sarops", 3, False) for s in SEEDS]
    jobs += [(s, "sarops", 3, True) for s in SEEDS]
    jobs += [(s, "sarops", n, False) for s in (400, 408) for n in (1, 5)]     # must reproduce compare_center
    jobs.sort(key=lambda j: -j[2])
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["method"], r["n"], r["sep"]))
    jobs = [j for j in jobs if tuple(j) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.perf_counter()
    with Pool(8) as pool:
        for _ in pool.imap_unordered(run_job, jobs, chunksize=1):
            pass
    print(f"done in {(time.perf_counter() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
