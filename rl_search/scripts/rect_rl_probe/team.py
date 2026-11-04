"""Exploratory: 5 UAVs, simple proposal policies with and without the existing plan-time separation check as a filter.

  <policy>        proposals flown as they are
  <policy>+check  each proposal goes through the existing check (holds 0-5 min, ``simulate`` with traffic); up to 8 proposals,
                  the first feasible one is flown; otherwise the last one, counted as a violation

Same launch (60 s apart), same traffic definition and metrics as compare_multi. Nothing here learns.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe import KW, MAX_TR, SEARCH, CYCLE, density_point, random_rect


def make_policy(name, seed, W):
    from rl_search.accordion import AccordionPlanner, SEP_HOLDS, simulate, start_transit
    rng = np.random.default_rng(seed + 991)
    base, _, chk = name.partition("+")

    class P(AccordionPlanner):
        def propose(self, env, acc):
            c = density_point(env, rng)
            if c is None:
                return None
            if base == "rect_density":
                return random_rect(env, rng, c)
            G = acc._grid(acc.theta())
            q = c @ G["R"]
            ci = int(np.clip((q[0] - G["ex"][0])//acc.g, 0, len(G["ex"]) - 2))
            cj = int(np.clip((q[1] - G["ey"][0])//acc.gy, 0, len(G["ey"]) - 2))
            return acc._climb(G, ci, cj)

        def choose(self, env, acc):
            last = None
            for _ in range(8 if chk else 1):
                r = self.propose(env, acc)
                if r is None:
                    continue
                last = r
                if not chk or not self.traffic:
                    return r
                if start_transit(env, r, CYCLE, SEARCH) > MAX_TR:
                    continue
                for h in SEP_HOLDS:
                    q = replace(r, hold_s=h)
                    if simulate(env, q, self.cycle_s, self.task, self.traffic, self.d_safe) > -0.5:
                        return q
            if last is not None and chk:
                self.violations += 1
            return last
    return P(W=W, **KW)


def run_job(job):
    seed, name, n = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import sweep_width
    from rl_search.multi import run_team_launch, separation_pairs
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.check_sarops import holdout_scenario
    probe = make_envs(1, seed)[0]
    scen = make_scenario(probe.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(probe, scen, seed + 10**6), stage=3)
    W = sweep_width(probe)
    pols = [make_policy(name, seed + 13*u, W) for u in range(n)]
    t0 = time.perf_counter()
    res = run_team_launch(make_envs(n, seed), pols, opts, 60.0, deconflict=True, hard=True)
    sec = time.perf_counter() - t0
    h = run_team_launch(make_envs(n, seed), None, hold, 60.0, actions=res["actions"])
    dmin, below, enc, _ = separation_pairs(res)
    out = dict(seed=seed, name=name, n=n, pos=float(res["pos"][-1]), hold=float(h["pos"][-1]), dmin=float(dmin), below=float(below),
               enc=int(enc), sec=sec, plans=sum(len(p.tasks) for p in pols), violations=int(sum(p.violations for p in pols)))
    print(seed, name, round(out["hold"], 3), round(below), round(dmin), out["plans"], out["violations"], round(sec, 1), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=(300, 330))
    ap.add_argument("--names", nargs="+", default=["rect_density", "rect_density+check", "loc_climb", "loc_climb+check"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    jobs = [(s, nm, a.n) for s in range(*a.seeds) for nm in a.names]
    t0 = time.perf_counter()
    with Pool(a.workers) as p:
        res = p.map(run_job, jobs, chunksize=1)
    a.output.write_text(json.dumps(res))
    print(f"done {(time.perf_counter() - t0)/60:.1f} min")
