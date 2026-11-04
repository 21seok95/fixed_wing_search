"""1-5 UAVs on the same scenario with the task-mode planner (search 10 min, transit <= 10 min, coverage k = 1-3),
sharing the belief at planning events (rl_search/multi.py). New sensor + CPA/LRC. All UAVs launch from the scenario's
launch point.

    python -m rl_search.scripts.check_multi --seed 301 --planner accordion --n 1 2 3 4 5 --plot
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
import numpy as np


def make_envs(n, seed):
    from rl_search.config import get_preset
    from rl_search.multi import TrackedEnv
    from rl_search.rl_env import RLConfig
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    envs = []
    for _ in range(n):
        e = TrackedEnv(terr, get_preset("rc_cessna", SENSORS["new"]), RLConfig(stage=3, detect_model="lrc"), seed=seed)
        e.fast = e.skip_cells = True
        envs.append(e)
    return envs


def main():
    warnings.filterwarnings("ignore")
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.multi import run_team
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_sarops import holdout_scenario

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--planner", choices=("accordion", "sarops"), default="accordion")
    ap.add_argument("--lane-mode", default="cover")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/multi"))
    args = ap.parse_args()
    probe = make_envs(1, args.seed)[0]
    scen = make_scenario(probe.sampler, args.seed)
    opts = dict(scenario=scen, stage=3)
    hold = dict(scenario=holdout_scenario(probe, scen, args.seed + 10**6), stage=3)
    W = sweep_width(probe)
    kw = dict(W=W, n_starts=3, search_s=600.0, max_transit_s=600.0, lane_mode=args.lane_mode)
    make = (lambda: AccordionPlanner(**kw)) if args.planner == "accordion" else (lambda: SaropsPlanner(big_jump=True, **kw))
    rows, flights = [], {}
    for n in args.n:
        envs = make_envs(n, args.seed)
        pols = [make() for _ in range(n)]
        events = []
        t0 = time.perf_counter()
        res = run_team(envs, pols, opts, log_plans=events)
        sec = time.perf_counter() - t0
        h = run_team(make_envs(n, args.seed), None, hold, actions=res["actions"])
        t, pos = res["t"] - res["t"][0], res["pos"]
        rows.append(dict(n=n, pos=pos[-1], hold=h["pos"][-1], p=[np.interp(m*60, t, pos) for m in (15, 30, 45)],
                         plans=len(events), sec=sec))
        flights[n] = (pols, res, [x for x in envs])
        print(f"N={n}: POS {pos[-1]:.3f}, held-out {h['pos'][-1]:.3f}, plans {len(events)}, {sec:.0f} s", flush=True)
    print(f"\nscenario {args.seed}, planner {args.planner}, task mode (search 10 min, transit <= 10 min, lanes {args.lane_mode})\n")
    print("| UAVs | POS | held-out POS | gain over 1 UAV | POS at 15 / 30 / 45 min | plans | s |")
    print("|---|---|---|---|---|---|---|")
    base = rows[0]["hold"] if rows and rows[0]["n"] == 1 else None
    for r in rows:
        g = f"{r['hold']/base:.2f}x" if base else "-"
        print(f"| {r['n']} | {r['pos']:.3f} | {r['hold']:.3f} | {g} | {r['p'][0]:.3f} / {r['p'][1]:.3f} / {r['p'][2]:.3f} | "
              f"{r['plans']} | {r['sec']:.0f} |")
    if args.plot:
        plot(flights, args, probe)


def plot(flights, args, probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.patterns import _rot
    args.out.mkdir(parents=True, exist_ok=True)
    ns = list(flights)
    fig, axs = plt.subplots(1, len(ns), figsize=(5*len(ns), 5.4), squeeze=False)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c0392b"]
    for ax, n in zip(axs[0], ns):
        pols, res, envs = flights[n]
        for u, pol in enumerate(pols):
            c = cols[u % len(cols)]
            ax.plot(res["xy"][u][:, 0]/1000, res["xy"][u][:, 1]/1000, color=c, lw=.8)
            for tp, r, L in pol.rects:
                corners = np.array([[sx*r.ell/2, sy*r.wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                ax.add_patch(Polygon((np.array([r.cx, r.cy]) + corners @ _rot(r.theta).T)/1000, fill=False, ec=c, lw=1.2))
                ax.annotate(f"{tp/60:.0f}", (r.cx/1000, r.cy/1000), color=c, fontsize=7)
        ax.plot(*np.array(envs[0].scen.lkp)/1000, "r*", ms=12)
        ax.set(title=f"{n} UAV{'s' if n > 1 else ''}: POS {res['pos'][-1]:.3f}", xlabel="km", ylabel="km", aspect="equal")
        ax.autoscale_view()
    fig.suptitle(f"scenario {args.seed}, {args.planner}, task mode: rectangles per UAV (colour), label = plan minute", fontsize=11)
    fig.tight_layout()
    f = args.out/f"multi_s{args.seed}_{args.planner}.png"
    fig.savefig(f, dpi=110)
    print(f)


if __name__ == "__main__":
    main()
