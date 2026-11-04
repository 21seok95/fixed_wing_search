"""Exploratory: foresight with a richer candidate set (rollout algorithm over the greedy base policy, no learning).

At each decision, candidates = the accordion peak rectangles + M rectangles climbed from points drawn from the probability +
the local-search neighbours (rotate, corner, k) of the best peak rectangle. Each candidate is flown, then the episode is
finished with the greedy base policy (peak_greedy) and the candidate with the best FINAL POS is taken.
1 UAV, same task mode and scenarios as probe.py, held-out POS by replay. Lower bound on what foresight can add.
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
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe import CYCLE, EXTRA, KW, MAX_TR, SEARCH, approach, candidates, density_point, fly_task, replay

TASK = dict(search_s=SEARCH, max_transit_s=MAX_TR, extra_s=EXTRA)


def greedy_tail(env, W):
    from rl_search.accordion import simulate
    while True:
        rects = candidates(env, W, 5)
        if not rects:
            if approach(env, []):
                return env.pos_cum
            continue
        r = rects[int(np.argmax([simulate(env, q, CYCLE, TASK) for q in rects]))]
        if fly_task(env, r, []):
            return env.pos_cum


def climbed(env, acc, rng):
    c = density_point(env, rng)
    if c is None:
        return None
    G = acc._grid(acc.theta())
    q = c @ G["R"]
    ci = int(np.clip((q[0] - G["ex"][0])//acc.g, 0, len(G["ex"]) - 2))
    cj = int(np.clip((q[1] - G["ey"][0])//acc.gy, 0, len(G["ey"]) - 2))
    return acc._climb(G, ci, cj)


def rich_candidates(env, W, rng, M):
    from rl_search.accordion import Accordion, lane_plan, simulate, start_transit
    from dataclasses import replace
    from rl_search.sarops import MOVES, neighbours
    acc = Accordion(env, "tavg_pca", CYCLE, 200.0, W, **KW)
    peaks = acc.peak_rects() or []
    out = list(peaks)
    for _ in range(M):
        r = climbed(env, acc, rng)
        if r is not None:
            out.append(r)
    if peaks:
        best = peaks[int(np.argmax([simulate(env, q, CYCLE, TASK) for q in peaks]))]
        cur = replace(best, corner=lane_plan(env, best, CYCLE, search_s=SEARCH)[1])
        out += [q for _, q in neighbours(env, cur, MOVES, 1.0, acc.gamma, CYCLE, search_s=SEARCH)]
    return [r for r in out if start_transit(env, r, CYCLE, SEARCH) <= MAX_TR], len(peaks)


def rollout_policy(env, W, rng, M, stats):
    acts = []
    while True:
        cands, np_ = rich_candidates(env, W, rng, M)
        stats["peaks"].append(np_)
        stats["cands"].append(len(cands))
        if not cands:
            if approach(env, acts):
                return acts
            continue
        best, best_p = None, -1.0
        for r in cands:
            snap = env.snapshot()
            term = fly_task(env, r, [])
            p = env.pos_cum if term else greedy_tail(env, W)
            env.restore(snap)
            if p > best_p:
                best, best_p = r, p
        if fly_task(env, best, acts):
            return acts


def run_job(job):
    seed, M = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import sweep_width
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_greedy import make_env
    env = make_env(seed)
    scen = make_scenario(env.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(env, scen, seed + 10**6), stage=3)
    W = sweep_width(env)
    rng = np.random.default_rng(seed + 31)
    stats = dict(peaks=[], cands=[])
    t0 = time.perf_counter()
    env.reset(options=opts)
    acts = rollout_policy(env, W, rng, M, stats)
    sec = time.perf_counter() - t0
    pos = replay(env, opts, acts)
    out = dict(seed=seed, pos=pos, hold=replay(env, hold, acts), sec=sec, peaks=float(np.mean(stats["peaks"])), cands=float(np.mean(stats["cands"])),
               decisions=len(stats["cands"]))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=(300, 330))
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    with Pool(a.workers) as p:
        res = p.map(run_job, [(s, a.M) for s in range(*a.seeds)], chunksize=1)
    a.output.write_text(json.dumps(res))
    print(f"done {(time.perf_counter() - t0)/60:.1f} min")
