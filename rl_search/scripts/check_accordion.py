"""Accordion search (SAROPS initial placement) on our moving-target belief (docs/sarops_approach_todo.md S.10).

Questions:
  1. Does the fast approximation POC x (1 - exp(-W L / A)) rank rectangles like the simulation does?
     Compared for the SAROPS snapshot formula and the time-averaged-with-transit formula.
  2. Which POC variant (snapshot at start / middle / time average with transit / + main-axis rotation)
     places the best rectangle for the coming 10-min cycle (simulated POS)?
  3. A planner that only uses the accordion (no simulation) each cycle vs v2 (cross-entropy, 48 rollouts).

    python -m rl_search.scripts.check_accordion --n-scen 8 --workers 2
    python -m rl_search.scripts.check_accordion --sensor new --detect lrc   # footprint 286 m, POD 0.7, SAROPS CPA+LRC
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

OUT = Path("rl_search/data/accordion_check")
PLAN_VARIANTS = [("snap0", 100), ("mid", 100), ("tavg", 100), ("tavg_pca", 100), ("tavg_pca", 200)]


SENSORS = {"old": [], "new": ["search.fov_cross_m=286", "search.fov_along_m=207", "search.pod=0.7"]}


def run_scenario(job):
    seed, n_pert, sensor, detect = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import Accordion, AccordionPlanner, Rect, simulate, sweep_width
    from rl_search.config import get_preset
    from rl_search.patterns import run_episode
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scenario import make_scenario
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter()
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    env = RLSearchEnv(terr, get_preset("rc_cessna", SENSORS[sensor]), RLConfig(stage=3, detect_model=detect), seed=seed)
    env.fast = True
    opts = dict(scenario=make_scenario(env.sampler, seed), stage=3)
    W = sweep_width(env)
    rng = np.random.default_rng(seed)
    points = []

    def hook(env, _):
        rec = dict(t_min=round(env.t/60, 1), variants={})
        rects = []
        for v, cell in PLAN_VARIANTS:
            r = Accordion(env, v, cell=cell, W=W).plan()
            if r is None:
                continue
            g = simulate(env, r)
            rec["variants"][f"{v}/{cell}"] = dict(approx=r.approx, sim=g, ell=r.ell, wid=r.wid)
            rects.append(r)
        if not rects:
            return
        base = rects[-2] if len(rects) >= 2 else rects[-1]
        th_pca = base.theta
        for _ in range(n_pert):
            th = th_pca if rng.random() < .5 else 0.0
            rects.append(Rect(base.cx + rng.uniform(-1500, 1500), base.cy + rng.uniform(-1500, 1500),
                              float(np.clip(base.ell*np.exp(rng.uniform(-.7, .7)), 600, 6000)),
                              float(np.clip(base.wid*np.exp(rng.uniform(-.7, .7)), 300, 6000)), th))
        sims = [simulate(env, r) for r in rects]
        rank = {}
        for formula in ("snap0", "tavg"):
            acc = Accordion(env, formula, cell=100, W=W)
            grids = {}
            ap = []
            for r in rects:
                if r.theta not in grids:
                    grids[r.theta] = acc._grid(r.theta)
                ap.append(acc.approx_rect(grids[r.theta], r))
            ap = np.array(ap)
            rank[formula] = dict(approx=ap.tolist())
        rec["cands"] = dict(sim=sims, **{f: rank[f]["approx"] for f in rank})
        points.append(rec)

    pol = AccordionPlanner("tavg_pca", cell=100, W=W, hook=hook)
    log = run_episode(env, opts, pol)
    out = dict(seed=seed, W=W, points=points, episodes={"tavg_pca/100 (with eval)": log.pos[-1]})
    for v, cell in PLAN_VARIANTS:
        p = AccordionPlanner(v, cell=cell, W=W)
        out["episodes"][f"{v}/{cell}"] = run_episode(env, opts, p).pos[-1]
        if (v, cell) == ("tavg_pca", 100):
            out["cycles"] = p.log
    out["sec"] = round(time.perf_counter() - t0, 1)
    print(f"seed {seed} {out['sec']} s: " + ", ".join(f"{k} {v:.3f}" for k, v in out["episodes"].items()), flush=True)
    return out


def summarize(res, v2, sensor="old", detect="kernel"):
    L = []
    nom = {"old": "0.3 x 115 m = 34.5 m", "new": "0.7 x 286 m = 200 m"}[sensor]
    L.append(f"Sensor: {sensor} ({SENSORS[sensor] or 'footprint 115 x 83 m, POD 0.3'}), detection model: {detect}\n")
    L.append(f"Effective sweep width of the env sensor (measured): {res[0]['W']:.1f} m (nominal POD x width = {nom})\n")
    # 1. ranking
    rho = {f: [] for f in ("snap0", "tavg")}
    top = {f: [] for f in ("snap0", "tavg")}
    ratio = []
    for r in res:
        for p in r["points"]:
            s = np.array(p["cands"]["sim"])
            if s.max() <= 0:
                continue
            for f in rho:
                a = np.array(p["cands"][f])
                rho[f].append(spearmanr(a, s).correlation)
                top[f].append(s[int(np.argmax(a))]/s.max())
            a = np.array(p["cands"]["tavg"])
            ok = (a > 1e-4) & (s > 1e-4)
            ratio += (s[ok]/a[ok]).tolist()
    L.append("## 1. Does the approximation rank rectangles like the simulation?\n")
    L.append(f"{len(rho['tavg'])} decision points, {len(res[0]['points'][0]['cands']['sim'])} candidate rectangles each "
             "(the planned rectangles + random shifts/resizes/rotations).\n")
    L.append("| POC formula | Spearman rank corr. (mean, median) | sim POS of approx-best / sim-best |")
    L.append("|---|---|---|")
    for f, name in (("snap0", "SAROPS: snapshot at cycle start, no transit"), ("tavg", "time average over search window, L minus transit")):
        L.append(f"| {name} | {np.nanmean(rho[f]):.2f}, {np.nanmedian(rho[f]):.2f} | {np.mean(top[f]):.0%} |")
    L.append(f"\nsim / approx (tavg) median {np.median(ratio):.2f}, IQR {np.quantile(ratio, .25):.2f}-{np.quantile(ratio, .75):.2f}\n")
    # 2. placement
    L.append("## 2. Rectangle placed by each variant: simulated POS of the next 10 min\n")
    names = list(res[0]["points"][0]["variants"])
    L.append("| variant / cell (m) | sim POS per cycle | relative to tavg_pca/100 | approx / sim | mean size (km) |")
    L.append("|---|---|---|---|---|")
    ref = []
    for r in res:
        for p in r["points"]:
            ref.append(p["variants"].get("tavg_pca/100", {}).get("sim", np.nan))
    ref = np.array(ref)
    for n in names:
        s, a, sz = [], [], []
        for r in res:
            for p in r["points"]:
                d = p["variants"].get(n)
                s.append(d["sim"] if d else np.nan)
                a.append(d["approx"] if d else np.nan)
                sz.append((d["ell"], d["wid"]) if d else (np.nan, np.nan))
        s, a, sz = np.array(s), np.array(a), np.array(sz)
        ok = ref > 0
        L.append(f"| {n} | {np.nanmean(s):.4f} | {np.nanmean(s[ok])/np.nanmean(ref[ok]):.0%} | "
                 f"{np.nanmedian(a/s):.2f} | {np.nanmean(sz[:, 0])/1000:.1f} x {np.nanmean(sz[:, 1])/1000:.1f} |")
    # 3. episodes
    L.append("\n## 3. Whole episode (one rectangle per 10-min cycle, no simulation in the planner)\n")
    L.append("| method | POS mean ± sd | min |")
    L.append("|---|---|---|")
    eps = {}
    for n in res[0]["episodes"]:
        eps[n] = [r["episodes"][n] for r in res]
    if v2 and sensor == "old" and detect == "kernel":
        by = {x["seed"]: x for x in v2}
        for m in ("v2 mixed T10", "v2 sector T10", "v2 parallel T10", "greedy90"):
            if all(r["seed"] in by and m in by[r["seed"]]["methods"] for r in res):
                eps[m + " (compare_patterns_v2)"] = [by[r["seed"]]["methods"][m]["pos"] for r in res]
    for n, v in sorted(eps.items(), key=lambda kv: -np.mean(kv[1])):
        L.append(f"| {n} | {np.mean(v):.3f} ± {np.std(v):.3f} | {np.min(v):.3f} |")
    L.append(f"\nScenarios: {[r['seed'] for r in res]}.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--n-pert", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sensor", choices=tuple(SENSORS), default="old")
    ap.add_argument("--detect", choices=("kernel", "lrc"), default="kernel")
    args = ap.parse_args()
    out = OUT if (args.sensor, args.detect) == ("old", "kernel") else OUT.parent/f"{OUT.name}_{args.sensor}_{args.detect}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, args.n_pert, args.sensor, args.detect) for s in range(args.seed0, args.seed0 + args.n_scen)]
    with Pool(args.workers) as pool:
        res = pool.map(run_scenario, jobs)
    (out/"results.json").write_text(json.dumps(res, default=float))
    v2p = Path("rl_search/data/compare_patterns_v2/results.json")
    txt = summarize(res, json.loads(v2p.read_text()) if v2p.exists() else None, args.sensor, args.detect)
    (out/"table.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
