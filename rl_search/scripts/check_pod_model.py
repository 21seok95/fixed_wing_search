"""I5 check on one scenario: accordion POD by the random-search formula (exp) vs exact parallel lanes (lanes)
(docs/sarops_approach_todo.md, 2026-11-04 improvements). New sensor + CPA/LRC.

At every 10-min decision point of an accordion episode: the rectangle each formula picks (size, spacing, simulated
gain) and how well each formula ranks a set of candidate rectangles against the simulation. Then whole episodes
(accordion only, and the SAROPS planner) with each formula, POS on the planner's and on held-out particles.

    python -m rl_search.scripts.check_pod_model --seed 301
"""
from __future__ import annotations

import argparse
import time
import warnings
import numpy as np
from scipy.stats import spearmanr

MODELS = ("exp", "lanes")


def main():
    warnings.filterwarnings("ignore")
    from rl_search.accordion import Accordion, AccordionPlanner, Rect, lane_count, simulate, sweep_width
    from rl_search.patterns import run_episode
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_greedy import make_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--n-pert", type=int, default=16)
    ap.add_argument("--no-sarops", action="store_true")
    args = ap.parse_args()
    env = make_env(args.seed)
    scen = make_scenario(env.sampler, args.seed)
    opts = dict(scenario=scen, stage=3)
    hold = dict(scenario=holdout_scenario(env, scen, args.seed + 10**6), stage=3)
    W = sweep_width(env)
    rng = np.random.default_rng(args.seed)
    rows, rho = [], {m: [] for m in MODELS}

    def hook(env, _):
        rects = {}
        for m in MODELS:
            r = Accordion(env, W=W, n_starts=3, pod_model=m).plan()
            if r is not None:
                n = lane_count(env, r)
                rects[m] = r
                rows.append(dict(t=env.t/60, model=m, ell=r.ell, wid=r.wid, s=r.wid/n, approx=r.approx, sim=simulate(env, r)))
        if not rects:
            return
        base = rects.get("exp") or next(iter(rects.values()))
        cands = list(rects.values())
        for _ in range(args.n_pert):
            cands.append(Rect(base.cx + rng.uniform(-1000, 1000), base.cy + rng.uniform(-1000, 1000),
                              float(np.clip(base.ell*np.exp(rng.uniform(-.7, .7)), 600, 5000)),
                              float(np.clip(base.wid*np.exp(rng.uniform(-.9, .9)), 200, 4000)), base.theta))
        sim = np.array([simulate(env, r) for r in cands])
        for m in MODELS:
            acc = Accordion(env, W=W, pod_model=m)
            G = {}
            ap_ = []
            for r in cands:
                if r.theta not in G:
                    G[r.theta] = acc._grid(r.theta)
                ap_.append(acc.approx_rect(G[r.theta], r))
            if sim.max() > 0:
                rho[m].append(spearmanr(ap_, sim).correlation)

    t0 = time.perf_counter()
    run_episode(env, opts, AccordionPlanner(W=W, n_starts=3, hook=hook))
    print(f"scenario {args.seed}, sweep width {W:.0f} m, decision-point checks {time.perf_counter() - t0:.0f} s\n")
    print("## Rectangle each formula picks (same state, accordion only)\n")
    print("| t (min) | formula | size (km) | spacing (m) | approx POS | simulated gain |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['t']:.0f} | {r['model']} | {r['ell']/1000:.1f} x {r['wid']/1000:.1f} | {r['s']:.0f} | {r['approx']:.3f} | {r['sim']:.3f} |")
    for m in MODELS:
        sel = [r for r in rows if r["model"] == m]
        print(f"\n{m}: mean spacing {np.mean([r['s'] for r in sel]):.0f} m, mean simulated gain {np.mean([r['sim'] for r in sel]):.4f}, "
              f"approx/sim {np.median([r['approx']/max(r['sim'], 1e-9) for r in sel]):.2f}, "
              f"rank corr. with simulation {np.nanmean(rho[m]):.2f} ({len(rho[m])} points x {args.n_pert + 2} rectangles)")

    print("\n## Whole episode\n")
    print("| planner | formula | POS (planner's particles) | held-out POS | s |")
    print("|---|---|---|---|---|")
    planners = [("accordion", lambda m: AccordionPlanner(W=W, n_starts=3, pod_model=m))]
    if not args.no_sarops:
        planners.append(("SAROPS", lambda m: SaropsPlanner(W=W, n_starts=3, big_jump=True, pod_model=m)))
    for name, make in planners:
        for m in MODELS:
            t1 = time.perf_counter()
            pol = make(m)
            acts = []
            log = run_episode(env, opts, lambda e: (acts.append(pol(e)), acts[-1])[1])
            it = iter(acts)
            hl = run_episode(env, hold, lambda e: next(it, 0.0))
            print(f"| {name} | {m} | {log.pos[-1]:.3f} | {hl.pos[-1]:.3f} | {time.perf_counter() - t1:.0f} |")


if __name__ == "__main__":
    main()
