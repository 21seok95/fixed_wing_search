"""1-5 UAVs over many scenarios in parallel (docs/sarops_approach_todo.md, 2026-11-04 multi-UAV).

Task-mode planner (search 10 min, transit <= 10 min, coverage k = 1-3), launch 60 s apart from the launch point,
separation 300 m as a constraint, belief shared at planning events (rl_search/multi.py). New sensor + CPA/LRC.
One job = one (scenario, number of UAVs); jobs run in a process pool, largest first.

    python -m rl_search.scripts.compare_multi --n-scen 30 --n 1 2 3 4 5 --planner sarops --workers 10
    python -m rl_search.scripts.compare_multi --plot-only --output rl_search/data/compare_multi_sarops
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


def run_job(job):
    seed, n, a = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.multi import run_team_launch, separation_pairs
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.check_sarops import holdout_scenario
    t0 = time.perf_counter()
    probe = make_envs(1, seed)[0]
    scen = make_scenario(probe.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(probe, scen, seed + 10**6), stage=3)
    W = sweep_width(probe)
    kw = dict(W=W, n_starts=a["n_starts"], search_s=600.0, max_transit_s=600.0, lane_mode="cover")
    make = (lambda: AccordionPlanner(**kw)) if a["planner"] == "accordion" else (lambda: SaropsPlanner(big_jump=True, **kw))
    pols = [make() for _ in range(n)]
    res = run_team_launch(make_envs(n, seed), pols, opts, a["launch_gap_s"], deconflict=a["deconflict"], hard=a["hard"])
    h = run_team_launch(make_envs(n, seed), None, hold, a["launch_gap_s"], actions=res["actions"])
    dmin, below, enc, early = separation_pairs(res)
    t = res["t"] - res["t"][0]
    out = dict(seed=seed, n=n, pos=float(res["pos"][-1]), hold=float(h["pos"][-1]),
               hold_t=[float(np.interp(m*60, h["t"] - h["t"][0], h["pos"])) for m in range(0, 61, 5)],
               dmin=float(dmin), below=float(below), enc=int(enc), sec=round(time.perf_counter() - t0, 1),
               violations=int(sum(p.violations for p in pols)))
    print(f"seed {seed} N={n}: held-out {out['hold']:.3f}, close {below:.0f} s, {out['sec']:.0f} s", flush=True)
    return out


def summarize(res, args) -> str:
    ns = sorted({r["n"] for r in res})
    seeds = sorted({r["seed"] for r in res})
    by = {(r["seed"], r["n"]): r for r in res}
    L = [f"Planner {args.planner}, task mode (search 10 min, transit <= 10 min, coverage k = 1-3), launch "
         f"{args.launch_gap_s:.0f} s apart, separation 300 m {'as a constraint' if args.hard else 'off' if not args.deconflict else 'as a penalty'}. "
         f"New sensor + CPA/LRC. {len(seeds)} scenarios ({seeds[0]}-{seeds[-1]}). Held-out POS = same flights on particles "
         "the planner never saw = detection probability of a target drawn from the model.", ""]
    L.append("| UAVs | held-out POS mean ± sd | median | 10th pct | min | vs 1 UAV (paired mean) | POS at 15 / 30 / 45 min | "
             "time any pair < 300 m (s, mean / max) | closest pair (min) | plans without a feasible option | s/episode |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ns:
        rs = [by[(s, n)] for s in seeds if (s, n) in by]
        v = np.array([r["hold"] for r in rs])
        base = [by[(r["seed"], 1)]["hold"] for r in rs if (r["seed"], 1) in by]
        ratio = f"{np.mean(v[:len(base)])/np.mean(base):.2f}x" if base and n != 1 else "-"
        ht = np.mean([r["hold_t"] for r in rs], axis=0)
        cl = np.array([r["below"] for r in rs])
        dm = min(r["dmin"] for r in rs) if n > 1 else float("nan")
        L.append(f"| {n} | {v.mean():.3f} ± {v.std():.3f} | {np.median(v):.3f} | {np.quantile(v, .1):.3f} | {v.min():.3f} | {ratio} | "
                 f"{ht[3]:.3f} / {ht[6]:.3f} / {ht[9]:.3f} | {cl.mean():.0f} / {cl.max():.0f} | {dm:.0f} | "
                 f"{sum(r['violations'] for r in rs)} | {np.mean([r['sec'] for r in rs]):.0f} |")
    L += ["", "## Per scenario, held-out POS", "", "| seed | " + " | ".join(f"{n} UAV" for n in ns) + " |",
          "|---|" + "---|"*len(ns)]
    for s in seeds:
        L.append(f"| {s} | " + " | ".join(f"{by[(s, n)]['hold']:.3f}" if (s, n) in by else "-" for n in ns) + " |")
    return "\n".join(L)


def plot(res, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted({r["n"] for r in res})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tm = np.arange(0, 61, 5)
    for n in ns:
        rs = [r for r in res if r["n"] == n]
        ax[0].plot(tm, np.mean([r["hold_t"] for r in rs], axis=0), label=f"{n} UAV{'s' if n > 1 else ''}")
    ax[0].set(title="Mean held-out POS over time", xlabel="minutes after handover", ylabel="POS")
    ax[0].legend()
    ax[1].boxplot([[r["hold"] for r in res if r["n"] == n] for n in ns])
    ax[1].set_xticks(range(1, len(ns) + 1), [str(n) for n in ns])
    ax[1].set(title="Held-out POS per scenario", xlabel="UAVs", ylabel="POS")
    fig.tight_layout()
    fig.savefig(out/"summary.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--planner", choices=("accordion", "sarops"), default="sarops")
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--launch-gap-s", type=float, default=60.0)
    ap.add_argument("--no-separation", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.deconflict = args.hard = not args.no_separation
    out = args.output or Path(f"rl_search/data/compare_multi_{args.planner}")
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        res = json.loads((out/"results.json").read_text())
    else:
        a = dict(planner=args.planner, n_starts=args.n_starts, launch_gap_s=args.launch_gap_s,
                 deconflict=args.deconflict, hard=args.hard)
        jobs = sorted([(s, n, a) for s in range(args.seed0, args.seed0 + args.n_scen) for n in args.n], key=lambda j: -j[1])
        t0 = time.perf_counter()
        with Pool(args.workers) as pool:
            res = pool.map(run_job, jobs, chunksize=1)
        print(f"done in {(time.perf_counter() - t0)/60:.1f} min with {args.workers} workers")
        (out/"results.json").write_text(json.dumps(res))
    txt = summarize(res, args)
    (out/"table.md").write_text(txt)
    plot(res, args, out)
    print(txt)


if __name__ == "__main__":
    main()
