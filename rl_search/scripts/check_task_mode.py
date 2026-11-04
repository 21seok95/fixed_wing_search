"""One-scenario check of the task mode (2026-11-04 I1-I4, docs/sarops_approach_todo.md) against the fixed 10-min
cycle, for the accordion planner and the SAROPS planner. New sensor + CPA/LRC.

Task mode: search 10 min after the transit, candidates within 10 min of transit (--max-transit-s), POS per minute, next plan when the
lanes are done (cap transit + 12 min), approach with lead pursuit while nothing is within reach.

    python -m rl_search.scripts.check_task_mode --seed 301 [--plot]
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
import numpy as np


def main():
    warnings.filterwarnings("ignore")
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.patterns import run_episode
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_greedy import make_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--search-s", type=float, default=600.0)
    ap.add_argument("--max-transit-s", type=float, default=600.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--peak-timing", choices=("now", "arrival", "both"), default="now")
    ap.add_argument("--lane-mode", choices=("fill", "fixed", "both"), default="fill")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/task_mode"))
    args = ap.parse_args()
    env = make_env(args.seed)
    scen = make_scenario(env.sampler, args.seed)
    opts = dict(scenario=scen, stage=3)
    hold = dict(scenario=holdout_scenario(env, scen, args.seed + 10**6), stage=3)
    W = sweep_width(env)
    task = dict(search_s=args.search_s, max_transit_s=args.max_transit_s)
    makers = {
        "accordion, 10-min cycle": lambda: AccordionPlanner(W=W, n_starts=3),
        "SAROPS, 10-min cycle": lambda: SaropsPlanner(W=W, n_starts=3, big_jump=True),
    }
    for pt in (("now", "arrival") if args.peak_timing == "both" else (args.peak_timing,)):
        for lm in (("fill", "fixed") if args.lane_mode == "both" else (args.lane_mode,)):
            tag = ("" if pt == "now" else ", peaks at arrival") + ("" if lm == "fill" else ", lanes no overlap")
            makers[f"accordion, task{tag}"] = lambda pt=pt, lm=lm: AccordionPlanner(W=W, n_starts=3, peak_timing=pt, lane_mode=lm, **task)
            makers[f"SAROPS, task{tag}"] = lambda pt=pt, lm=lm: SaropsPlanner(W=W, n_starts=3, big_jump=True, peak_timing=pt,
                                                                               lane_mode=lm, **task)
    rows, flights = [], {}
    for name, make in makers.items():
        if args.only and not any(o in name for o in args.only):
            continue
        pol, acts, on = make(), [], []

        def rec(e):
            a = float(pol(e))
            acts.append(a)
            on.append(pol.follower is not None and pol.follower.started)
            return a

        t0 = time.perf_counter()
        log = run_episode(env, opts, rec)
        sec = time.perf_counter() - t0
        it = iter(acts)
        h = run_episode(env, hold, lambda e: next(it, 0.0))
        t = np.array(log.t)
        pos = np.array(log.pos)
        C = np.array([[r.cx, r.cy] for _, r, _ in pol.rects])
        jumps = int((np.hypot(*np.diff(C, axis=0).T) > 3000).sum()) if len(C) > 1 else 0
        on = np.array(on)
        first = next((t[i] for i in range(len(on)) if on[i]), np.nan)
        rows.append(dict(name=name, pos=pos[-1], hold=h.pos[-1], off=1 - on.mean(), first=(first - t[0])/60, plans=len(pol.rects),
                         jumps=jumps, sec=sec, pos15=np.interp(t[0] + 900, t, pos), pos30=np.interp(t[0] + 1800, t, pos)))
        flights[name] = (pol, np.array(log.xy), t, pos)
        if getattr(pol, "tasks", None):
            print(f"\n{name}: tasks (plan min, transit est. / actual min, length min, end)")
            for k in pol.tasks:
                end = k["end"] if k["end"] is not None else t[-1]
                act = f"{(k['lane_t'] - k['t0'])/60:4.1f}" if k.get("lane_t") is not None else "  - "
                print(f"  {k['t0']/60:5.1f}  {k['transit']/60:4.1f} / {act}  {(end - k['t0'])/60:5.1f}  {k['how'] or 'episode end'}")
            if pol.approach:
                print("  approach phases (min): " + ", ".join(f"{a/60:.1f}-{(b if b is not None else t[-1])/60:.1f}" for a, b in pol.approach))
    print(f"\nscenario {args.seed}, sweep width {W:.0f} m, search {args.search_s/60:.0f} min, max transit {args.max_transit_s/60:.0f} min\n")
    print("| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['pos']:.3f} | {r['hold']:.3f} | {r['pos15']:.3f} / {r['pos30']:.3f} | {r['off']:.0%} | "
              f"{r['first']:.1f} | {r['plans']} | {r['jumps']} | {r['sec']:.0f} |")
    if args.plot:
        plot(flights, args, env)


def plot(flights, args, env):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.patterns import _rot
    args.out.mkdir(parents=True, exist_ok=True)
    n = len(flights)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)
    for ax, (name, (pol, xy, t, pos)) in zip(axs[0], flights.items()):
        ax.plot(xy[:, 0]/1000, xy[:, 1]/1000, "k-", lw=.8)
        cols = plt.cm.viridis(np.linspace(0, 1, max(len(pol.rects), 1)))
        for (tp, r, L), c in zip(pol.rects, cols):
            corners = np.array([[sx*r.ell/2, sy*r.wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            ax.add_patch(Polygon((np.array([r.cx, r.cy]) + corners @ _rot(r.theta).T)/1000, fill=False, ec=c, lw=1.6))
            ax.annotate(f"{tp/60:.0f}", (r.cx/1000, r.cy/1000), color=c, fontsize=8)
        for a, b in getattr(pol, "approach", []):
            m = (t >= a) & (t <= (b if b is not None else t[-1]))
            ax.plot(xy[m, 0]/1000, xy[m, 1]/1000, color="red", lw=2, alpha=.6)
        ax.plot(*np.array(env.scen.lkp)/1000, "r*", ms=12)
        ax.plot(*xy[0]/1000, "ko", ms=5)
        ax.set(title=f"{name}: POS {pos[-1]:.3f}", xlabel="km", ylabel="km", aspect="equal")
    fig.suptitle(f"scenario {args.seed}: rectangles coloured by plan time (label = minute), red = approach phase", fontsize=11)
    fig.tight_layout()
    f = args.out/(f"task_mode_s{args.seed}_x{args.max_transit_s/60:.0f}" + ("_arrival" if args.peak_timing != "now" else "") + ("_fixed" if args.lane_mode == "fixed" else "") + ".png")
    fig.savefig(f, dpi=110)
    print(f)


if __name__ == "__main__":
    main()
