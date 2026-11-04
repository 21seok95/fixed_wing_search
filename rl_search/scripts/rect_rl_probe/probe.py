"""Exploratory only (nothing written to the repository): how does the rectangle as an action behave, without training?

Each policy is an AccordionPlanner whose ``choose`` is replaced: the same hook an RL policy would use. 1 UAV, task mode as
compare_multi (search 10 min, transit <= 10 min, cover), scenarios 300-329, held-out POS by replay.
  random_turn   flat: uniform random turn every 5 s
  rect_uniform  rectangle: uniform centre within reach, random angle, length, k, corner (an untrained spatial policy)
  rect_density  the same, centre drawn from the remaining probability at the middle of the search window
  loc_climb     centre drawn from the probability, shape from the accordion climb
  peak_random   random accordion peak rectangle
  peak_greedy   best accordion peak rectangle by simulate (no local search)
  tree          search over the accordion peak rectangles for the final POS: all sequences for ``depth`` tasks, then greedy
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

SEARCH, MAX_TR, EXTRA, CYCLE = 600.0, 600.0, 120.0, 600.0
KW = dict(search_s=SEARCH, max_transit_s=MAX_TR, lane_mode="cover", n_starts=5)


def random_rect(env, rng, centre):
    from rl_search.accordion import Rect, fixed_lane_time
    cw = 2*env.lrc_R
    r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
    k = int(rng.integers(1, 4))
    s = cw/k
    ell = float(rng.choice(np.arange(600.0, 4001.0, 200.0)))
    n = 1
    while fixed_lane_time(ell, n + 1, s, r_turn, env.v) <= SEARCH and n < 60:
        n += 1
    corner = tuple(rng.choice([-1, 1], 2).tolist())
    return Rect(float(centre[0]), float(centre[1]), ell, n*s, float(rng.uniform(0, np.pi)), corner=corner, spacing=s, k_max=3)


def density_point(env, rng):
    b = env.belief
    t = env.t + SEARCH/2
    m = (b.w > 0) & b.active(t)
    if not m.any():
        return None
    return b.positions(t)[rng.choice(np.flatnonzero(m), p=b.w[m]/b.w[m].sum())]


def make_policy(name, seed, W):
    from rl_search.accordion import AccordionPlanner, simulate
    rng = np.random.default_rng(seed + 77)

    class P(AccordionPlanner):
        def choose(self, env, acc):
            if name == "rect_uniform":
                x = env.plant.pose[:2]
                lim = env.half_map - 500.0
                for _ in range(100):
                    b, d = rng.uniform(-np.pi, np.pi), MAX_TR*env.v*np.sqrt(rng.uniform())
                    c = x + d*np.array([np.cos(b), np.sin(b)])
                    if np.all(np.abs(c) < lim):
                        return random_rect(env, rng, c)
                return None
            if name == "rect_density":
                c = density_point(env, rng)
                return None if c is None else random_rect(env, rng, c)
            if name == "loc_climb":
                c = density_point(env, rng)
                if c is None:
                    return None
                G = acc._grid(acc.theta())
                q = c @ G["R"]
                ci = int(np.clip((q[0] - G["ex"][0])//acc.g, 0, len(G["ex"]) - 2))
                cj = int(np.clip((q[1] - G["ey"][0])//acc.gy, 0, len(G["ey"]) - 2))
                return acc._climb(G, ci, cj)
            rects = acc.peak_rects() or []
            if not rects:
                return None
            if name == "peak_random":
                return rects[int(rng.integers(len(rects)))]
            vals = [simulate(env, r, self.cycle_s, self.task) for r in rects]
            return rects[int(np.argmax(vals))]
    return P(W=W, **KW)


def fly(env, opts, policy):
    env.reset(options=opts)
    acts, done = [], False
    while not done:
        a = float(policy(env))
        acts.append(a)
        _, _, term, _, _ = env.step([a])
        done = term
    return env.pos_cum, acts


def replay(env, opts, acts):
    env.reset(options=opts)
    for a in acts:
        _, _, term, _, _ = env.step([a])
        if term:
            break
    return env.pos_cum


# ---------------------------------------------------------------- tree over the accordion's peak rectangles
def candidates(env, W, K):
    from rl_search.accordion import Accordion
    return (Accordion(env, "tavg_pca", CYCLE, 200.0, W, **KW).peak_rects() or [])[:K]


def fly_task(env, r, acts):
    from rl_search.accordion import lanes, start_transit
    tr = start_transit(env, r, CYCLE, SEARCH)
    _, f = lanes(env, r, CYCLE, search_s=SEARCH)
    t_cap = env.t + tr + SEARCH + EXTRA
    while env.t < t_cap - 1e-6:
        a = float(f.action())
        acts.append(a)
        _, _, term, _, _ = env.step([a])
        if term:
            return True
        if f.laps >= 1:
            break
    return False


def approach(env, acts):
    from rl_search.patterns import lead_action
    for _ in range(6):
        a = float(lead_action(env))
        acts.append(a)
        _, _, term, _, _ = env.step([a])
        if term:
            return True
    return False


def search_tree(env, W, K, depth, stats):
    """Best (final POS, actions) from the current state; ``depth`` tasks branch on all K, then the best by simulate."""
    from rl_search.accordion import simulate
    stats["nodes"] += 1
    rects = candidates(env, W, K)
    if not rects:
        acts = []
        if approach(env, acts):
            return env.pos_cum, acts
        p, a = search_tree(env, W, K, depth, stats)
        return p, acts + a
    if depth <= 0:
        task = dict(search_s=SEARCH, max_transit_s=MAX_TR, extra_s=EXTRA)
        rects = [rects[int(np.argmax([simulate(env, r, CYCLE, task) for r in rects]))]]
    best = (-1.0, [])
    for r in rects:
        snap = env.snapshot()
        acts = []
        if fly_task(env, r, acts):
            p, a = env.pos_cum, []
        else:
            p, a = search_tree(env, W, K, depth - 1, stats)
        if p > best[0]:
            best = (p, acts + a)
        env.restore(snap)
    return best


def run_job(job):
    seed, names, K, depth = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import sweep_width
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_greedy import make_env
    env = make_env(seed)
    t0 = time.perf_counter()
    scen = make_scenario(env.sampler, seed)
    gen_s = time.perf_counter() - t0
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(env, scen, seed + 10**6), stage=3)
    W = sweep_width(env)
    out = dict(seed=seed, gen_s=gen_s, methods={})
    rng = np.random.default_rng(seed + 5)
    for name in names:
        t0 = time.perf_counter()
        n_tasks = None
        if name == "random_turn":
            pos, acts = fly(env, opts, lambda e: float(rng.uniform(-1, 1)))
        elif name.startswith("tree"):
            _, k, d = name.split("_")
            env.reset(options=opts)
            stats = dict(nodes=0)
            _, acts = search_tree(env, W, int(k), int(d), stats)
            pos = replay(env, opts, acts)
            n_tasks = stats["nodes"]
        else:
            pol = make_policy(name, seed, W)
            pos, acts = fly(env, opts, pol)
            n_tasks = len(pol.tasks)
        sec = time.perf_counter() - t0
        out["methods"][name] = dict(pos=pos, hold=replay(env, hold, acts), sec=sec, tasks=n_tasks, steps=len(acts))
    print(seed, {k: (round(v["pos"], 3), round(v["hold"], 3), round(v["sec"], 1)) for k, v in out["methods"].items()}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=(300, 330))
    ap.add_argument("--names", nargs="+", default=["random_turn", "rect_uniform", "rect_density", "loc_climb", "peak_random", "peak_greedy", "tree_5_0", "tree_3_5"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    jobs = [(s, a.names, a.K, a.depth) for s in range(*a.seeds)]
    t0 = time.perf_counter()
    with Pool(a.workers) as p:
        res = p.map(run_job, jobs, chunksize=1)
    a.output.write_text(json.dumps(res))
    print(f"done {(time.perf_counter() - t0)/60:.1f} min")
