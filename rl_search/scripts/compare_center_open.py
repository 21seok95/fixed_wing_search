"""SAROPS with the area score on the open-ground centre scenarios (rl_search/scripts/center_open.py, 2026-11-05).

Configurations (all without a separation constraint, launch 60 s apart, task mode as compare_center):
  pos    plain SAROPS, POS only (the baseline)
  cand   POS only, extra rectangle starts on the unswept region (shows what the candidates alone do)
  score  total score A + w POS, with the same extra starts

Flights are scored on held-out particles: POS by an exact action replay, A by rl_search.area_score.score_flight.

    python -m rl_search.scripts.compare_center_open --check
    python -m rl_search.scripts.compare_center_open --n 3 --x 70 --w 1 --workers 8
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time
import warnings

import numpy as np

OUT = Path("rl_search/data/center_open/first_check")
SCORE_XS = (50, 70, 85)


def make_envs(n, seed, area_env=False):
    from rl_search.area_score import AreaEnv
    from rl_search.config import get_preset
    from rl_search.multi import TrackedEnv
    from rl_search.rl_env import RLConfig
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    cls = AreaEnv if area_env else TrackedEnv
    envs = []
    for u in range(n):
        e = cls(terr, get_preset("rc_cessna", SENSORS["new"]), RLConfig(stage=3, detect_model="lrc"), seed=seed)
        e.fast = e.skip_cells = True
        e.uav_id = u
        envs.append(e)
    return envs


def run_job(job):
    seed, config, n, x, w = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import sweep_width
    from rl_search.area_score import AreaSaropsPlanner, AreaScore, score_flight
    from rl_search.multi import run_team_launch, separation_pairs
    from rl_search.sarops import SaropsPlanner
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    t0 = time.perf_counter()
    probe = make_envs(1, seed)[0]
    scen = open_scenario(probe.sampler, seed)
    hold_scen = holdout_scenario(probe, scen, seed + 10**6)
    W = sweep_width(probe)
    kw = dict(W=W, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover")
    if config == "pos":
        envs = make_envs(n, seed)
        pols = [SaropsPlanner(**kw) for _ in range(n)]
    else:
        envs = make_envs(n, seed, area_env=True)
        area = AreaScore(probe.terr, probe.h, scen.track, scen.save_dt_s, x, probe.lrc_R)
        for e in envs:
            e.area, e.pos_w, e.area_w = area, (w if config == "score" else 1.0), (1.0 if config == "score" else 0.0)
        pols = [AreaSaropsPlanner(**kw) for _ in range(n)]
    res = run_team_launch(envs, pols, dict(scenario=scen, stage=3), 60.0, deconflict=False, hard=False)
    h = run_team_launch(make_envs(n, seed), None, dict(scenario=hold_scen, stage=3), 60.0, actions=res["actions"])
    drift = max(float(np.hypot(*(a[-1] - b[-1]))) for a, b in zip(res["xy"], h["xy"]))
    if drift > 1e-6:
        raise RuntimeError(f"replay left the planned path by {drift} m")
    scores = {str(xx): score_flight(probe.terr, probe.h, hold_scen.track, hold_scen.save_dt_s, xx, probe.lrc_R,
                                    res["xy"], res["tt"]) for xx in SCORE_XS}
    dmin, below, enc, _ = separation_pairs(res)
    out = dict(seed=seed, config=config, n=n, x=x, w=w, pos=float(h["pos"][-1]),
               pos_t=[float(np.interp(m*60, h["t"] - h["t"][0], h["pos"])) for m in range(0, 61, 5)], scores=scores,
               dmin=float(dmin) if np.isfinite(dmin) else None, below=float(below), enc=int(enc),
               plans=int(sum(len(p.rects) for p in pols)), lkp=scen.lkp.tolist(), start=scen.start.tolist(),
               heading=scen.heading_deg, speed_kmh=scen.speed_mps*3.6,
               xy=[p[::2].round(0).tolist() for p in res["xy"]], tt=[t[::2].tolist() for t in res["tt"]],
               rects=[[(tp, r.cx, r.cy, r.ell, r.wid, r.theta, r.hold_s) for tp, r, _ in p.rects] for p in pols],
               actions=res["actions"], sec=round(time.perf_counter() - t0, 1))
    print(f"seed {seed} {config} N={n}: POS {out['pos']:.3f}, A at x 70 {scores['70']['A']:.3f}, "
          f"close {below:.0f} s, plans {out['plans']}, {out['sec']:.0f} s", flush=True)
    return out


def check():
    """Score arithmetic on a known path: the incremental gains must equal the mean of c_j from a direct count, a
    rollback must restore the swept cells, and committed cells must not score again."""
    warnings.filterwarnings("ignore")
    from rl_search.area_score import AreaScore, CELL_M, N_WINDOWS, region_masks
    from rl_search.scripts.center_open import open_scenario
    env = make_envs(1, 500)[0]
    s = open_scenario(env.sampler, 500)
    a = AreaScore(env.terr, env.h, s.track, s.save_dt_s, 70, env.lrc_R, track_curve=True)
    t = np.arange(0.0, 3605.0, 5.0)
    xy = s.lkp + np.stack([1500*np.cos(t/90.0)*(1 + t/3600), 1500*np.sin(t/90.0)*(1 + t/3600)], 1)
    total = sum(a.mark(xy[i:i + 2], t[i:i + 2]) for i in range(len(t) - 1))
    # direct count on the same grid
    S = np.zeros(a.B.shape, bool)
    X, Y = np.meshgrid(a.gx, a.gy)
    c = np.zeros(N_WINDOWS)
    for i in range(len(t) - 1):
        k = a.window(t[i + 1])
        p, q = xy[i], xy[i + 1]
        ab = q - p
        u = np.clip(((X - p[0])*ab[0] + (Y - p[1])*ab[1])/(ab @ ab), 0, 1)
        S |= (((X - p[0] - u*ab[0])**2 + (Y - p[1] - u*ab[1])**2) <= a.R**2) & (((a.B >> (k - 1)) & 1) > 0)
        if k <= N_WINDOWS and abs(t[i + 1] - k*300.0) < 1e-6:
            c[k - 1] = (S & (((a.B >> (k - 1)) & 1) > 0)).sum()/a.size[k - 1]
    print(f"A from gains {total:.6f}, from the curve {a.curve().mean():.6f}, from a direct count {c.mean():.6f}")
    assert abs(total - c.mean()) < 1e-3 and abs(total - a.curve().mean()) < 1e-6
    before = a.S.copy()
    cp = len(a.log)
    b = xy[100] + np.array([4000.0, -3000.0])
    a.mark(np.array([b, b + [300.0, 0.0]]), np.array([595.0, 600.0]))
    undone = a.rollback(cp)
    assert (a.S == before).all()
    a.apply(undone)
    assert a.mark(np.array([b, b + [300.0, 0.0]]), np.array([595.0, 600.0])) == 0.0
    masks = region_masks(env.terr, env.h, s.track, s.save_dt_s, 70)
    print("region km2 from the 50 m masks vs the 25 m raster:",
          [round(float(m.sum()*2500/1e6), 1) for m in masks][::3], [round(float(v*CELL_M**2/1e6), 1) for v in a.size][::3])
    print("score checks passed")


def summarize(res, x) -> str:
    seeds = sorted({r["seed"] for r in res})
    ns = sorted({r["n"] for r in res})
    by = {(r["seed"], r["config"], r["n"]): r for r in res}
    L = [f"Open-ground centre scenarios {seeds[0]}-{seeds[-1]}, no separation constraint. Total score A + w POS with region x = {x}%. "
         "All values on held-out particles.", ""]
    for n in ns:
        L += [f"## {n} UAV" + ("s" if n > 1 else ""), "",
              "| config | POS | POS at 15 / 30 / 45 min | A (x 50) | A (x 70) | A (x 85) | c at 60 min (x 70) | time inside the x 70 region, 0-15 / 15-30 / 30-45 / 45-60 min | pair < 300 m (s) | s/episode |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for cfg in ("pos", "cand", "score"):
            g = [by[(s, cfg, n)] for s in seeds if (s, cfg, n) in by]
            if not g:
                continue
            t = np.mean([r["pos_t"] for r in g], axis=0)
            ins = [np.mean([r["scores"]["70"]["inside"][str(lo)] for r in g]) for lo in (0, 15, 30, 45)]
            L.append(f"| {cfg} | {np.mean([r['pos'] for r in g]):.3f} | {t[3]:.3f} / {t[6]:.3f} / {t[9]:.3f} | "
                     + " | ".join(f"{np.mean([r['scores'][str(xx)]['A'] for r in g]):.3f}" for xx in SCORE_XS)
                     + f" | {np.mean([r['scores']['70']['c'][-1] for r in g]):.3f} | " + " / ".join(f"{v:.0%}" for v in ins)
                     + f" | {np.mean([r['below'] for r in g]):.0f} | {np.mean([r['sec'] for r in g]):.0f} |")
        L += ["", "| paired against pos | POS diff (se, higher in) | A at x 70 diff (se, higher in) |", "|---|---|---|"]
        for cfg in ("cand", "score"):
            k = [s for s in seeds if (s, cfg, n) in by and (s, "pos", n) in by]
            if not k:
                continue
            dp = np.array([by[(s, cfg, n)]["pos"] - by[(s, "pos", n)]["pos"] for s in k])
            da = np.array([by[(s, cfg, n)]["scores"]["70"]["A"] - by[(s, "pos", n)]["scores"]["70"]["A"] for s in k])
            se = lambda d: d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0.0
            L.append(f"| {cfg} | {dp.mean():+.3f} ({se(dp):.3f}, {int((dp > 0).sum())}/{len(k)}) | {da.mean():+.3f} ({se(da):.3f}, {int((da > 0).sum())}/{len(k)}) |")
        L += ["", "| seed | heading | km/h | POS pos / cand / score | A x 70 pos / cand / score | c at 60 min pos / cand / score |", "|---|---|---|---|---|---|"]
        for s in seeds:
            g = [by.get((s, cfg, n)) for cfg in ("pos", "cand", "score")]
            ref = next(r for r in g if r)
            L.append(f"| {s} | {ref['heading']:.0f} | {ref['speed_kmh']:.0f} | " + " / ".join(f"{r['pos']:.3f}" if r else "-" for r in g) + " | "
                     + " / ".join(f"{r['scores']['70']['A']:.3f}" if r else "-" for r in g) + " | "
                     + " / ".join(f"{r['scores']['70']['c'][-1]:.3f}" if r else "-" for r in g) + " |")
        L.append("")
    return "\n".join(L)


def sweep_key(r) -> tuple:
    return (r["seed"], r["config"], r["n"]) + ((float(r["x"]), float(r["w"])) if r["config"] == "score" else ())


def summarize_sweep(res) -> str:
    """Per UAV count: the baseline and every (x, w) against it, paired by scenario. A is scored at the run's own x."""
    seeds = sorted({r["seed"] for r in res})
    by = {sweep_key(r): r for r in res}
    combos = sorted({k[3:] for k in by if len(k) == 5})
    L = [f"Open-ground centre scenarios {seeds[0]}-{seeds[-1]}, SAROPS, no separation constraint, total score A + w POS. "
         "All values on held-out particles; A at the run's own x; differences are paired against POS only.", ""]
    se = lambda d: d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0.0
    for n in sorted({r["n"] for r in res}):
        base = [by.get((s, "pos", n)) for s in seeds]
        if not all(base):
            continue
        L += [f"## {n} UAV" + ("s" if n > 1 else ""), "",
              "| x | w | POS | POS diff (se, higher in) | A | A diff (se, higher in) | c at 60 min | c at 60 min, baseline | inside at 45-60 min | baseline | pair < 300 m (s) | baseline | s/episode | runs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x, w in combos:
            g = [by.get((s, "score", n, x, w)) for s in seeds]
            k = [i for i, r in enumerate(g) if r]
            if not k:
                continue
            sx = str(int(x))
            run, ref = [g[i] for i in k], [base[i] for i in k]
            dp = np.array([a["pos"] - b["pos"] for a, b in zip(run, ref)])
            da = np.array([a["scores"][sx]["A"] - b["scores"][sx]["A"] for a, b in zip(run, ref)])
            L.append(f"| {x:.0f} | {w:g} | {np.mean([r['pos'] for r in run]):.3f} | {dp.mean():+.3f} ({se(dp):.3f}, {int((dp > 0).sum())}/{len(k)}) | "
                     f"{np.mean([r['scores'][sx]['A'] for r in run]):.3f} | {da.mean():+.3f} ({se(da):.3f}, {int((da > 0).sum())}/{len(k)}) | "
                     f"{np.mean([r['scores'][sx]['c'][-1] for r in run]):.3f} | {np.mean([r['scores'][sx]['c'][-1] for r in ref]):.3f} | "
                     f"{np.mean([r['scores'][sx]['inside']['45'] for r in run]):.0%} | {np.mean([r['scores'][sx]['inside']['45'] for r in ref]):.0%} | "
                     f"{np.mean([r['below'] for r in run]):.0f} | {np.mean([r['below'] for r in ref]):.0f} | {np.mean([r['sec'] for r in run]):.0f} | {len(k)} |")
        L += ["", f"Baseline POS only: POS {np.mean([r['pos'] for r in base]):.3f}, s/episode {np.mean([r['sec'] for r in base]):.0f}.", ""]
    return "\n".join(L)


def run_sweep(args):
    """Grid of x and w. Baselines first, then w = 1, 0.5, 2 so partial results are useful. Runs from the first check are
    reused. Actions are not kept, to keep the file small (paths stay, for figures and rescoring)."""
    out = args.output if args.output != OUT else Path("rl_search/data/center_open/sweep")
    out.mkdir(parents=True, exist_ok=True)
    path = out/"results.jsonl"
    if not path.exists() and (OUT/"results.jsonl").exists():
        with open(path, "w") as f:
            for line in (OUT/"results.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["config"] in ("pos", "score"):
                    r.pop("actions", None)
                    f.write(json.dumps(r) + "\n")
    done = {sweep_key(json.loads(line)) for line in path.read_text().splitlines()} if path.exists() else set()
    seeds = range(args.seed0, args.seed0 + args.n_scen)
    jobs = [(s, "pos", n, 70.0, 1.0) for n in args.n for s in seeds]
    jobs += [(s, "score", n, float(x), float(w)) for w in args.ws for x in args.xs for n in args.n for s in seeds]
    key = lambda j: (j[0], j[1], j[2]) + ((j[3], j[4]) if j[1] == "score" else ())
    jobs = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            r.pop("actions", None)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
    print(f"done in {(time.perf_counter() - t0)/60:.1f} min", flush=True)
    res = [json.loads(line) for line in path.read_text().splitlines()]
    txt = summarize_sweep(res)
    (out/"table.md").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="grid of --xs and --ws for every --n")
    ap.add_argument("--xs", type=float, nargs="+", default=[70.0, 50.0, 85.0])
    ap.add_argument("--ws", type=float, nargs="+", default=[1.0, 0.5, 2.0])
    ap.add_argument("--smoke", action="store_true", help="one scenario, one UAV, all configurations")
    ap.add_argument("--n", type=int, nargs="+", default=[3])
    ap.add_argument("--x", type=float, default=70.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--n-scen", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.sweep:
        run_sweep(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        jobs = [(args.seed0, c, 1, args.x, args.w) for c in ("pos", "cand", "score")]
        with Pool(3) as pool:
            res = pool.map(run_job, jobs, chunksize=1)
        print(summarize(res, args.x))
        return
    done_file = args.output/"results.jsonl"
    done = {}
    if done_file.exists():
        for line in done_file.read_text().splitlines():
            r = json.loads(line)
            done[(r["seed"], r["config"], r["n"], r["x"], r["w"])] = r
    jobs = [(s, c, n, args.x, args.w) for c in ("pos", "score", "cand") for n in args.n
            for s in range(args.seed0, args.seed0 + args.n_scen)]
    jobs = [j for j in jobs if tuple(j) not in done]
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    res = list(done.values())
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            res.append(r)
            with open(done_file, "a") as f:
                f.write(json.dumps(r) + "\n")
    print(f"done in {(time.perf_counter() - t0)/60:.1f} min", flush=True)
    txt = summarize(res, args.x)
    (args.output/"table.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
