"""Local search on the accordion rectangle (docs/sarops_approach_todo.md S.10): full move set, each move left out,
and the optional centre shift, against the accordion alone.

POS is reported twice: on the scenario's particles (the planner optimizes on them, so it is optimistic) and on
a held-out particle set from the same target model, by replaying the same actions.

    python -m rl_search.scripts.check_sarops --n-scen 8 --workers 6
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np

OUT = Path("rl_search/data/sarops_check")
FULL = ("rotate", "corner", "spacing")
VARIANTS = {"accordion": None, "rotate+corner+spacing": FULL, "no rotate": ("corner", "spacing"),
            "no corner": ("rotate", "spacing"), "no spacing": ("rotate", "corner"),
            "rotate only": ("rotate",), "corner only": ("corner",), "spacing only": ("spacing",),
            "+ shift": FULL + ("shift",), "big jump": FULL}


def holdout_scenario(env, scen, seed):
    """Same target, fresh particles (a set the planner never sees)."""
    from rl_search.target import simulate as simulate_target
    cfg = env.sampler.cfg
    track, exit_step = simulate_target(env.sampler.terr, cfg.target, scen.lkp, scen.heading_deg, scen.speed_mps,
                                       np.random.default_rng(seed))
    return replace(scen, track=track, exit_step=exit_step)


def fly(env, opts, policy):
    """Episode with a recording policy. Returns (POS, actions, final position)."""
    env.reset(options=opts)
    acts, done = [], False
    env.skip_cells = True
    while not done:
        a = policy(env)
        acts.append(a)
        _, _, term, trunc, info = env.step([a])
        done = term or trunc
    env.skip_cells = False
    return info["pos_policy"], acts, env.plant.pose[:2].copy()


def replay(env, opts, acts, xy_end):
    env.reset(options=opts)
    env.skip_cells = True
    for a in acts:
        _, _, term, trunc, info = env.step([a])
        if term or trunc:
            break
    env.skip_cells = False
    if np.hypot(*(env.plant.pose[:2] - xy_end)) > 1e-6:
        raise RuntimeError("replay left the planned path")
    return info["pos_policy"]


def run_scenario(job):
    seed, names, time_s = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.config import get_preset
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.sarops import SaropsPlanner
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    env = RLSearchEnv(terr, get_preset("rc_cessna", SENSORS["new"]), RLConfig(stage=3, detect_model="lrc"), seed=seed)
    env.fast = True
    scen = make_scenario(env.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(env, scen, seed + 10**6), stage=3)
    W = sweep_width(env)
    out = dict(seed=seed, methods={})
    for name in names:
        moves = VARIANTS[name]
        pol = AccordionPlanner(W=W, n_starts=3) if moves is None else SaropsPlanner(W=W, n_starts=3, moves=moves, time_s=time_s,
                                                                                     big_jump=name == "big jump")
        t0 = time.perf_counter()
        pos, acts, xy = fly(env, opts, pol)
        sec = time.perf_counter() - t0
        out["methods"][name] = dict(pos=pos, hold=replay(env, hold, acts, xy), sec=sec,
                                    search=getattr(pol, "search_log", []))
    print(f"seed {seed}: " + ", ".join(f"{k} {v['pos']:.3f}/{v['hold']:.3f}" for k, v in out["methods"].items()), flush=True)
    return out


def summarize(res) -> str:
    names = list(res[0]["methods"])
    base = "accordion"
    a = lambda n, k: np.array([r["methods"][n][k] for r in res])
    L = [f"Scenarios: {[r['seed'] for r in res]}. New sensor + lrc, accordion tavg_pca/200 with 3 peaks, 10-min cycle.",
         "POS on the planner's particles / on held-out particles (same actions).", "",
         "| method | POS mean ± sd | min | held-out POS mean ± sd | min | held-out vs accordion | wins | s/episode | evals/cycle | moves used |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p, h = a(n, "pos"), a(n, "hold")
        d = h - a(base, "hold")
        logs = [l for r in res for l in r["methods"][n]["search"]]
        ev = np.mean([l["evals"] for l in logs]) if logs else 0
        used = {}
        for l in logs:
            for k, v in l["used"].items():
                used[k] = used.get(k, 0) + v
        L.append(f"| {n} | {p.mean():.3f} ± {p.std():.3f} | {p.min():.3f} | {h.mean():.3f} ± {h.std():.3f} | {h.min():.3f} | "
                 f"{d.mean():+.3f} ({100*d.mean()/a(base, 'hold').mean():+.1f}%) | {int((d > 1e-9).sum())}/{len(res)} | "
                 f"{a(n, 'sec').mean():.0f} | {ev:.0f} | {used or '-'} |")
    full = [n for n in names if n != base]
    if full:
        L += ["", "Per scenario, held-out POS:", "", "| seed | " + " | ".join(names) + " |", "|---|" + "---|"*len(names)]
        for r in res:
            L.append(f"| {r['seed']} | " + " | ".join(f"{r['methods'][n]['hold']:.3f}" for n in names) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--time-s", type=float, default=None, help="Local search limit per cycle")
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    names = ["accordion"] + [v for v in args.variants if v != "accordion"]
    jobs = [(args.seed0 + i, names, args.time_s) for i in range(args.n_scen)]
    t0 = time.perf_counter()
    with Pool(args.workers) as p:
        res = sorted(p.imap_unordered(run_scenario, jobs), key=lambda r: r["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(res, indent=1))
    table = summarize(res)
    (args.output/"table.md").write_text(table + "\n")
    print(table)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
