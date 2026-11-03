"""SAROPS-style planner (accordion + local search from every peak) against greedy90, on the same scenarios, with the
new sensor + CPA/LRC (docs/sarops_approach_todo.md S.10). Scenarios 300-307 tuned the planner; the rest are new.

POS is reported on the planner's particles and on a held-out set from the same target model (same actions).

    python -m rl_search.scripts.compare_greedy --n-scen 30 --workers 6
    python -m rl_search.scripts.compare_greedy --plot-only
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

from rl_search.scripts.check_sarops import holdout_scenario

OUT = Path("rl_search/data/compare_greedy")
METHODS = ("SAROPS", "greedy90", "accordion")
N_TUNE = 8
SEED_TUNE = 300
COLORS = {"SAROPS": "#2a78d6", "greedy90": "#eb6834", "accordion": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_env(seed):
    from rl_search.config import get_preset
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    env = RLSearchEnv(terr, get_preset("rc_cessna", SENSORS["new"]), RLConfig(stage=3, detect_model="lrc"), seed=seed)
    env.fast = env.skip_cells = True
    return env


def run_scenario(seed):
    warnings.filterwarnings("ignore")
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.patterns import run_episode
    from rl_search.patterns_v2 import Greedy90
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    env = make_env(seed)
    scen = make_scenario(env.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(env, scen, seed + 10**6), stage=3)
    W = sweep_width(env)
    planners = dict(SAROPS=lambda: SaropsPlanner(W=W, n_starts=3, big_jump=True), greedy90=Greedy90,
                    accordion=lambda: AccordionPlanner(W=W, n_starts=3))
    out = dict(seed=seed, methods={})
    for name in METHODS:
        pol, acts = planners[name](), []

        def record(e):
            acts.append(float(pol(e)))
            return acts[-1]

        t0 = time.perf_counter()
        log = run_episode(env, opts, record)
        sec = time.perf_counter() - t0
        it = iter(acts)
        h = run_episode(env, hold, lambda e: next(it))
        if np.hypot(*(np.array(h.xy[-1]) - np.array(log.xy[-1]))) > 1e-6:
            raise RuntimeError("replay left the planned path")
        out["methods"][name] = dict(pos=log.pos[-1], hold=h.pos[-1], sec=sec, acts=acts, t=log.t, pos_t=log.pos, hold_t=h.pos,
                                    xy=np.array(log.xy).tolist())
    print(f"seed {seed}: " + ", ".join(f"{k} {v['pos']:.3f}/{v['hold']:.3f} {v['sec']:.0f}s"
                                       for k, v in out["methods"].items()), flush=True)
    return out


def row(res, a, b):
    d = np.array([r["methods"][a]["hold"] - r["methods"][b]["hold"] for r in res])
    return f"{d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} | {int((d > 0).sum())}/{len(d)}"


def summarize(res) -> str:
    groups = [("all", res), ("tuning (300-307)", [r for r in res if r["seed"] < SEED_TUNE + N_TUNE]),
              ("new (308+)", [r for r in res if r["seed"] >= SEED_TUNE + N_TUNE])]
    L = ["New sensor + lrc. POS held-out = same actions on particles the planner never saw.", ""]
    for gname, g in groups:
        if not g:
            continue
        L += [f"## {gname}: {len(g)} scenarios", "",
              "| method | held-out POS mean ± sd | median | min | POS on planner's particles | s/episode |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            h = np.array([r["methods"][m]["hold"] for r in g])
            L.append(f"| {m} | {h.mean():.3f} ± {h.std():.3f} | {np.median(h):.3f} | {h.min():.3f} | "
                     f"{np.mean([r['methods'][m]['pos'] for r in g]):.3f} | {np.mean([r['methods'][m]['sec'] for r in g]):.0f} |")
        L += ["", f"SAROPS − greedy90 (paired, mean ± se | wins): {row(g, 'SAROPS', 'greedy90')}",
              f"SAROPS − accordion (paired, mean ± se | wins): {row(g, 'SAROPS', 'accordion')}", ""]
    L += ["## Per scenario, held-out POS", "", "| seed | SAROPS | greedy90 | accordion |", "|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['seed']} | " + " | ".join(f"{r['methods'][m]['hold']:.3f}" for m in METHODS) + " |")
    return "\n".join(L)


def curve(r, m, grid, key="hold_t"):
    x = r["methods"][m]
    return np.interp(grid, np.array(x["t"])/60, x[key])


def plot_summary(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    x = np.array([r["methods"]["greedy90"]["hold"] for r in res])
    y = np.array([r["methods"]["SAROPS"]["hold"] for r in res])
    new = np.array([r["seed"] >= SEED_TUNE + N_TUNE for r in res])
    hi = max(x.max(), y.max())*1.06
    a1.plot([0, hi], [0, hi], color=MUTED, lw=1, zorder=1)
    a1.scatter(x[new], y[new], s=70, color=COLORS["SAROPS"], edgecolor="white", lw=1.5, zorder=3, label="new scenarios (308+)")
    a1.scatter(x[~new], y[~new], s=70, facecolor="white", edgecolor=COLORS["SAROPS"], lw=1.8, zorder=3, label="tuning scenarios (300-307)")
    for i in np.argsort(np.abs(y - x))[-3:]:
        a1.annotate(str(res[i]["seed"]), (x[i], y[i]), xytext=(6, 5), textcoords="offset points", fontsize=9, color=MUTED)
    a1.text(.04, .93, f"above the line: SAROPS better\n{int((y > x).sum())} of {len(x)} scenarios", transform=a1.transAxes,
            fontsize=10, color=MUTED, va="top")
    a1.set(xlim=(0, hi), ylim=(0, hi), xlabel="greedy90 POS (held-out particles)", ylabel="SAROPS POS (held-out particles)",
           aspect="equal", title="Each scenario: SAROPS vs greedy90")
    a1.legend(loc="lower right", frameon=False, fontsize=9)
    a1.grid(color=GRID, lw=.8)
    a1.set_axisbelow(True)
    grid = np.arange(2.0, 60.01, .5)
    for m in METHODS:
        c = np.array([curve(r, m, grid) for r in res])
        mean, se = c.mean(0), c.std(0, ddof=1)/np.sqrt(len(c))
        a2.fill_between(grid, mean - se, mean + se, color=COLORS[m], alpha=.15, lw=0)
        a2.plot(grid, mean, color=COLORS[m], lw=2, label=m)
        a2.annotate(f"{m} {mean[-1]:.3f}", (grid[-1], mean[-1]), xytext=(6, dict(SAROPS=7, greedy90=-6).get(m, 0)), textcoords="offset points", va="center",
                    fontsize=10, color=INK, annotation_clip=False)
    a2.set(xlim=(0, 60), ylim=(0, None), xlabel="flight time (min)", ylabel="POS, mean of scenarios (band: ±1 se)",
           title="POS over the flight")
    a2.legend(loc="upper left", frameon=False, fontsize=10)
    a2.grid(color=GRID, lw=.8)
    a2.set_axisbelow(True)
    a2.spines[["top", "right"]].set_visible(False)
    a2.margins(x=0)
    fig.get_layout_engine().set(w_pad=.12)
    fig.suptitle(f"SAROPS vs greedy90, new sensor + LRC, {len(res)} scenarios", fontsize=13)
    fig.savefig(out/"summary.png", dpi=130)
    plt.close(fig)


def plot_paths(res, out: Path):
    """Greatest SAROPS win, greatest greedy win and the median scenario: paths at three time slices."""
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.patterns import run_episode
    from rl_search.plotting import draw_rl_compare
    from rl_search.scenario import make_scenario
    d = np.array([r["methods"]["SAROPS"]["hold"] - r["methods"]["greedy90"]["hold"] for r in res])
    order = np.argsort(d)
    picks = [("largest SAROPS advantage", order[-1]), ("largest greedy90 advantage", order[0]), ("median difference", order[len(order)//2])]
    for label, i in picks:
        r = res[i]
        env = make_env(r["seed"])
        opts = dict(scenario=make_scenario(env.sampler, r["seed"]), stage=3)
        runs = {}
        for m in ("SAROPS", "greedy90"):
            x = r["methods"][m]
            t = np.array(x["t"])
            times = [t[min(int(round(f*(len(t) - 1))), len(t) - 1)] for f in (1/3, 2/3, 1)]
            it, snaps = iter(x["acts"]), []

            def pol(e):
                return next(it)

            env.reset(options=opts)
            for k, a in enumerate(x["acts"]):
                if any(abs(env.t - s) < 1e-6 for s in times):
                    snaps.append((env.t, env.belief.w.copy(), env.pos_cum))
                env.step([a])
            snaps.append((env.t, env.belief.w.copy(), env.pos_cum))
            runs[m] = (np.array(x["xy"]), t, snaps)
        title = (f"Scenario {r['seed']}, {label}: held-out POS SAROPS {r['methods']['SAROPS']['hold']:.3f}, "
                 f"greedy90 {r['methods']['greedy90']['hold']:.3f}")
        fig = draw_rl_compare(env.terr, env.belief, runs, title)
        fig.savefig(out/f"paths_s{r['seed']}.png", dpi=85)
        plt.close(fig)
        print("wrote", out/f"paths_s{r['seed']}.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scen", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=SEED_TUNE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        t0 = time.perf_counter()
        with Pool(args.workers) as p:
            res = sorted(p.imap_unordered(run_scenario, range(args.seed0, args.seed0 + args.n_scen)), key=lambda r: r["seed"])
        (args.output/"results.json").write_text(json.dumps(res))
        print(f"total {time.perf_counter() - t0:.0f} s")
    res = json.loads((args.output/"results.json").read_text())
    table = summarize(res)
    (args.output/"table.md").write_text(table + "\n")
    print(table)
    plot_summary(res, args.output)
    plot_paths(res, args.output)


if __name__ == "__main__":
    main()
