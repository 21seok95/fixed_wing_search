"""Separation cost on clean scenarios: LKP at the map centre, random target heading, launch point at a random bearing
5-15 km from the LKP (inside the map), so map exits and the geofence play no part (2026-11-04).

For each scenario: 5 UAVs with the separation constraint (300 m, holds 0-5 min, 5 peaks), 5 UAVs without it, and 1 UAV
for reference; task-mode SAROPS planner (search 10 min, transit <= 10 min, coverage k = 1-3), launch 60 s apart.
One figure per scenario: paths, rectangles, holds (thick) and close points (< 300 m, red), with / without separation.

    python -m rl_search.scripts.compare_center --n-scen 10 --workers 10
    python -m rl_search.scripts.compare_center --plot-only
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

OUT = Path("rl_search/data/compare_center")
D_SAFE = 300.0


def center_scenario(sampler, seed, dist=(5000.0, 15000.0)):
    """LKP: the admissible LKP cell nearest the map centre; heading uniform; speed as the sampler; launch at a random
    bearing and distance inside the map, heading toward the LKP."""
    from rl_search.scenario import Scenario, scenario_metrics
    from rl_search.target import simulate
    t0 = time.perf_counter()
    terr, cfg = sampler.terr, sampler.cfg
    rng = np.random.default_rng(seed)
    ok = np.argwhere(terr.clearance >= cfg.lkp_clearance_m)       # (row, col); the sampler's own LKP cells must
    xy = np.stack([terr.axis[ok[:, 1]], terr.axis[ok[:, 0]]], 1)   # be near the edge (launch on the perimeter)
    lkp = xy[int(np.argmin(np.hypot(*xy.T)))].astype(float)
    heading = float(rng.uniform(0, 360))
    speed = float(rng.uniform(*cfg.speed_kmh))/3.6
    track, exit_step = simulate(terr, cfg.target, lkp, heading, speed, rng)
    lim = terr.axis[-1] - 1500.0
    for _ in range(1000):
        b, d = rng.uniform(-np.pi, np.pi), rng.uniform(*dist)
        start = lkp + d*np.array([np.cos(b), np.sin(b)])
        if np.all(np.abs(start) <= lim):
            break
    sh = float(np.degrees(np.arctan2(lkp[1] - start[1], lkp[0] - start[0])))
    metrics = scenario_metrics(track, exit_step, cfg.target.save_dt_s, start, cfg)
    return Scenario(seed, lkp, heading, speed, start, sh, track, exit_step, cfg.target.save_dt_s, metrics,
                    time.perf_counter() - t0)


def close_points(xy, tt, d_safe=D_SAFE):
    """Midpoints of UAV pairs closer than d_safe at the same time (5 s grid, both airborne)."""
    pts = []
    n = len(xy)
    for a in range(n):
        for b in range(a + 1, n):
            t0, t1 = max(tt[a][0], tt[b][0]), min(tt[a][-1], tt[b][-1])
            if t1 <= t0:
                continue
            g = np.arange(t0, t1, 5.0)
            pa = np.stack([np.interp(g, tt[a], xy[a][:, 0]), np.interp(g, tt[a], xy[a][:, 1])], 1)
            pb = np.stack([np.interp(g, tt[b], xy[b][:, 0]), np.interp(g, tt[b], xy[b][:, 1])], 1)
            m = np.hypot(*(pa - pb).T) < d_safe
            pts += ((pa[m] + pb[m])/2).tolist()
    return pts


def run_job(job):
    seed, mode = job
    warnings.filterwarnings("ignore")
    from rl_search.accordion import sweep_width
    from rl_search.multi import run_team_launch, separation_pairs
    from rl_search.sarops import SaropsPlanner
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.check_sarops import holdout_scenario
    t0 = time.perf_counter()
    probe = make_envs(1, seed)[0]
    scen = center_scenario(probe.sampler, seed)
    opts, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(probe, scen, seed + 10**6), stage=3)
    W = sweep_width(probe)
    n = 1 if mode == "1" else 5
    sep = mode == "5 sep"
    pols = [SaropsPlanner(W=W, n_starts=5 if sep else 3, big_jump=True, search_s=600.0, max_transit_s=600.0,
                          lane_mode="cover") for _ in range(n)]
    res = run_team_launch(make_envs(n, seed), pols, opts, 60.0, deconflict=sep, hard=sep)
    h = run_team_launch(make_envs(n, seed), None, hold, 60.0, actions=res["actions"])
    dmin, below, enc, early = separation_pairs(res)
    holds = [(k["t0"], k["hold"]) for p in pols for k in p.tasks if k.get("hold", 0) > 0]
    rects = [[(tp, r.cx, r.cy, r.ell, r.wid, r.theta, r.hold_s) for tp, r, _ in p.rects] for p in pols]
    out = dict(seed=seed, mode=mode, n=n, pos=float(res["pos"][-1]), hold_pos=float(h["pos"][-1]),
               hold_t=[float(np.interp(m*60, h["t"] - h["t"][0], h["pos"])) for m in range(0, 61, 5)],
               dmin=float(dmin), below=float(below), enc=int(enc), n_holds=len(holds),
               hold_s=float(sum(x[1] for x in holds)), violations=int(sum(p.violations for p in pols)),
               plans=int(sum(len(p.rects) for p in pols)), sec=round(time.perf_counter() - t0, 1),
               lkp=scen.lkp.tolist(), start=scen.start.tolist(), heading=scen.heading_deg, speed_kmh=scen.speed_mps*3.6,
               escaped=scen.metrics["escaped_end"], start_lkp_km=scen.metrics["start_lkp_km"],
               xy=[x[::2].round(0).tolist() for x in res["xy"]], tt=[t[::2].tolist() for t in res["tt"]],
               rects=rects, holds_uav=[[(k["t0"], k["hold"]) for k in p.tasks if k.get("hold", 0) > 0] for p in pols],
               close=np.round(close_points(res["xy"], res["tt"]), 0).tolist())
    print(f"seed {seed} [{mode}]: held-out {out['hold_pos']:.3f}, close {below:.0f} s, min {dmin:.0f} m, holds "
          f"{len(holds)} ({out['hold_s']/60:.0f} min), no-option plans {out['violations']}, {out['sec']:.0f} s", flush=True)
    return out


def summarize(res) -> str:
    seeds = sorted({r["seed"] for r in res})
    by = {(r["seed"], r["mode"]): r for r in res}
    L = ["Clean scenarios: LKP at the map centre, random target heading, launch 5-15 km from the LKP inside the map. "
         "Task-mode SAROPS (search 10 min, transit <= 10 min, k = 1-3), launch 60 s apart. Held-out POS = detection "
         "probability of a target drawn from the model.", ""]
    L.append("| mode | held-out POS mean | min | POS at 15 / 30 / 45 min | time any pair < 300 m (s, mean / max) | "
             "closest pair (min) | holds per flight (min of holding) | plans without a feasible option |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mode in ("1", "5 no sep", "5 sep"):
        rs = [by[(s, mode)] for s in seeds if (s, mode) in by]
        if not rs:
            continue
        v = np.array([r["hold_pos"] for r in rs])
        ht = np.mean([r["hold_t"] for r in rs], axis=0)
        cl = np.array([r["below"] for r in rs])
        dm = min(r["dmin"] for r in rs) if mode != "1" else float("nan")
        L.append(f"| {mode} | {v.mean():.3f} | {v.min():.3f} | {ht[3]:.3f} / {ht[6]:.3f} / {ht[9]:.3f} | "
                 f"{cl.mean():.0f} / {cl.max():.0f} | {dm:.0f} | {np.mean([r['n_holds'] for r in rs]):.1f} "
                 f"({np.mean([r['hold_s'] for r in rs])/60:.1f}) | {sum(r['violations'] for r in rs)} |")
    sep = [by[(s, "5 sep")]["hold_pos"] - by[(s, "5 no sep")]["hold_pos"] for s in seeds
           if (s, "5 sep") in by and (s, "5 no sep") in by]
    if sep:
        d = np.array(sep)
        L.append(f"\nSeparation cost (5 sep - 5 no sep, paired): {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0:.4f}, "
                 f"better in {int((d > 0).sum())}/{len(d)}")
    L += ["", "| seed | heading | speed km/h | launch-LKP km | escaped | 1 UAV | 5 no sep (close s / min m) | 5 sep (close s / min m, holds) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        a, b, c = by.get((s, "1")), by.get((s, "5 no sep")), by.get((s, "5 sep"))
        ref = a or b or c
        L.append(f"| {s} | {ref['heading']:.0f} | {ref['speed_kmh']:.0f} | {ref['start_lkp_km']:.1f} | {ref['escaped']:.0%} | "
                 + (f"{a['hold_pos']:.3f}" if a else "-") + " | "
                 + (f"{b['hold_pos']:.3f} ({b['below']:.0f} / {b['dmin']:.0f})" if b else "-") + " | "
                 + (f"{c['hold_pos']:.3f} ({c['below']:.0f} / {c['dmin']:.0f}, {c['n_holds']})" if c else "-") + " |")
    return "\n".join(L)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.patterns import _rot
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c0392b"]
    by = {(r["seed"], r["mode"]): r for r in res}
    for s in sorted({r["seed"] for r in res}):
        modes = [m for m in ("5 sep", "5 no sep") if (s, m) in by]
        if not modes:
            continue
        fig, axs = plt.subplots(1, len(modes), figsize=(7.5*len(modes), 7.5), squeeze=False)
        for ax, m in zip(axs[0], modes):
            r = by[(s, m)]
            for u, (xy, tt) in enumerate(zip(r["xy"], r["tt"])):
                xy, tt = np.array(xy), np.array(tt)
                c = cols[u % len(cols)]
                ax.plot(xy[:, 0]/1000, xy[:, 1]/1000, color=c, lw=.8, label=f"UAV {u + 1}")
                for t0, h in r["holds_uav"][u]:
                    sel = (tt >= t0) & (tt <= t0 + h)
                    ax.plot(xy[sel, 0]/1000, xy[sel, 1]/1000, color=c, lw=3, alpha=.6)
                for tp, cx, cy, ell, wid, th, hs in r["rects"][u]:
                    corners = np.array([[sx*ell/2, sy*wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                    ax.add_patch(Polygon((np.array([cx, cy]) + corners @ _rot(th).T)/1000, fill=False, ec=c, lw=1, alpha=.7))
            if r["close"]:
                cp = np.array(r["close"])/1000
                ax.scatter(cp[:, 0], cp[:, 1], s=18, c="red", zorder=5, label="pair < 300 m")
            ax.plot(*np.array(r["lkp"])/1000, "k*", ms=14, label="LKP")
            ax.plot(*np.array(r["start"])/1000, "ks", ms=7, label="launch")
            ax.set(title=f"{m}: held-out POS {r['hold_pos']:.3f}, < 300 m {r['below']:.0f} s, closest {r['dmin']:.0f} m, "
                         f"holds {r['n_holds']}", xlabel="km", ylabel="km", aspect="equal")
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"scenario {s} (LKP at the centre, heading {by[(s, modes[0])]['heading']:.0f} deg, "
                     f"{by[(s, modes[0])]['speed_kmh']:.0f} km/h): thick = hold, red = pair closer than 300 m", fontsize=11)
        fig.tight_layout()
        fig.savefig(out/f"center_s{s}.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        res = json.loads((args.output/"results.json").read_text())
    else:
        order = {"5 sep": 0, "5 no sep": 1, "1": 2}
        jobs = sorted([(s, m) for s in range(args.seed0, args.seed0 + args.n_scen) for m in order], key=lambda j: order[j[1]])
        t0 = time.perf_counter()
        with Pool(args.workers) as pool:
            res = pool.map(run_job, jobs, chunksize=1)
        print(f"done in {(time.perf_counter() - t0)/60:.1f} min with {args.workers} workers")
        (args.output/"results.json").write_text(json.dumps(res))
    txt = summarize(res)
    (args.output/"table.md").write_text(txt)
    plot(res, args.output)
    print(txt)


if __name__ == "__main__":
    main()
