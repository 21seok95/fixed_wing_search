"""Compare search path methods without RL (docs/sarops_approach_todo.md V.1).

Methods: IAMSAR patterns (parallel, creeping line, track line, expanding square, spiral, sector), each with
its placement optimised per scenario (datum offset, size, lanes, spacing, ground-fixed or moving with the
predicted centroid) by random search + local refinement on a 1000-particle subsample, then evaluated on the
full belief; and the adaptive baselines straight, seek, lead (intercept) and greedy (15 s POS lookahead).
All fly RLSearchEnv (same plant, +-30 deg / 5 s, 1 s cell sensor, energy, geofence) from the stage-3 handover.

Metrics:
1. POS on the particle belief (POD of the preset, default 0.3): curve over time and final value.
2. Detection time of single true targets (POD 1): fresh trajectories from the same motion model (not the belief
   particles). The flown path does not depend on the true target, so each path scores all of them. Adaptive
   methods replan with a POD-1 belief; pattern paths do not use the belief and are reused.

    python -m rl_search.scripts.compare_patterns --n-scen 6 --workers 2
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import json
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np

T_GRID = np.arange(0, 60.1, 1.0)   # minutes after handover


def scenario_subsample(scen, n, rng):
    idx = np.sort(rng.choice(scen.track.shape[1], n, replace=False))
    return replace(scen, track=np.ascontiguousarray(scen.track[:, idx]), exit_step=scen.exit_step[idx])


def make_pattern_policy(p, prior):
    from rl_search.patterns import PatternFollower
    st = {}

    def pol(env):
        if "f" not in st:
            st["f"] = PatternFollower(env, p, prior)
        return st["f"].action()
    return pol


def sample_params(kind, rng):
    from rl_search.patterns import PatternParams
    p = PatternParams(kind=kind, ox=rng.uniform(-3000, 3000), oy=rng.uniform(-2000, 2000),
                      moving=bool(rng.random() < .5), spacing_mult=float(rng.choice([1.0, 1.5, 2.0])))
    if kind in ("parallel", "creeping"):
        p.size, p.n = rng.uniform(800, 6000), int(rng.integers(3, 31))
    elif kind == "trackline":
        p.size, p.n = rng.uniform(3000, 20000), int(rng.integers(1, 4))
    elif kind == "expanding":
        p.size = rng.uniform(5000, 60000)
    else:                                   # spiral radius / sector radius
        p.size = rng.uniform(400, 4000)
    return p


def perturb(p, rng):
    q = copy.copy(p)
    q.ox += rng.normal(0, 600)
    q.oy += rng.normal(0, 400)
    q.size = max(200.0, q.size*np.exp(rng.normal(0, .25)))
    q.n = int(np.clip(q.n + rng.integers(-3, 4), 1, 40))
    if rng.random() < .2:
        q.moving = not q.moving
    if rng.random() < .2:
        q.spacing_mult = float(rng.choice([1.0, 1.5, 2.0]))
    return q


def truth_tracks(terr, preset, scen, n, seed):
    from rl_search.target import simulate
    cfg = replace(preset.scenario.target, particles=n)
    track, exit_step = simulate(terr, cfg, scen.lkp, scen.heading_deg, scen.speed_mps, np.random.default_rng(seed))
    return track, exit_step, cfg.save_dt_s


def detection_times(env, cells_log, truths, t_handover):
    """Minutes after handover at which each true target first lies in a cell seen in that interval (POD 1)."""
    track, exit_step, dt = truths
    n = track.shape[1]
    found = np.full(n, np.inf)
    c, lo, nn = env.cfg.cell_m, env.lo, env.n
    exit_t = np.where(exit_step >= 0, exit_step*dt, np.inf)
    for t0, cells in cells_log:
        if not len(cells):
            continue
        todo = np.flatnonzero(~np.isfinite(found) & (exit_t > t0))
        if not len(todo):
            break
        hit = np.zeros(len(todo), bool)
        for tt in (t0, t0 + 2.5, t0 + 5.0):
            f = np.clip(tt/dt, 0, track.shape[0] - 1)
            i = min(int(f), track.shape[0] - 2)
            a = f - i
            pos = (1 - a)*track[i, todo] + a*track[i + 1, todo]
            ix = np.floor((pos[:, 0] - lo)/c).astype(int)
            iy = np.floor((pos[:, 1] - lo)/c).astype(int)
            ok = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
            hit[ok] |= np.isin(iy[ok]*nn + ix[ok], cells)
        found[todo[hit]] = (t0 - t_handover)/60
    return found


def curve(log, t_handover):
    tm = (np.array(log.t) - t_handover)/60
    return np.interp(T_GRID, tm, np.array(log.pos), right=log.pos[-1]).tolist()


def run_scenario(job):
    seed, args = job
    warnings.filterwarnings("ignore")
    from rl_search.config import get_preset
    from rl_search.patterns import ADAPTIVE, KINDS, PriorTrack, run_episode
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scenario import make_scenario
    from rl_search.terrain import load_terrain
    t_start = time.perf_counter()
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    preset = get_preset("rc_cessna", [f"search.pod={args.pod}"])
    preset1 = get_preset("rc_cessna", ["search.pod=1"])
    env = RLSearchEnv(terr, preset, RLConfig(stage=3), seed=seed)
    env1 = RLSearchEnv(terr, preset1, RLConfig(stage=3), seed=seed)
    for e in (env, env1):
        e.fast = True
    scen = make_scenario(env.sampler, seed)
    rng = np.random.default_rng(seed)
    scen_sub = scenario_subsample(scen, args.opt_particles, rng)
    prior = PriorTrack(scen)
    opts, opts_sub = dict(scenario=scen, stage=3), dict(scenario=scen_sub, stage=3)
    truths = truth_tracks(terr, preset, scen, args.truths, seed + 7919)
    out = dict(seed=int(seed), speed_kmh=round(scen.speed_mps*3.6, 1), methods={})
    t_h = preset.handover.t_plan_s

    def record(name, log03, log1, params=None, evals=0):
        ttd = detection_times(env1, log1.cells, truths, t_h)
        out["methods"][name] = dict(pos=round(log03.pos[-1], 4), pos_pod1=round(log1.pos[-1], 4),
                                    pos_curve=curve(log03, t_h), detect_frac=float(np.isfinite(ttd).mean()),
                                    ttd_min=[round(float(x), 2) if np.isfinite(x) else None for x in ttd],
                                    params=params, evals=evals, path=np.array(log03.xy)[::6].round(0).tolist())

    # IAMSAR patterns: optimise placement on the subsample, evaluate on the full belief
    for kind in KINDS:
        best, best_v = None, -1.0
        cands = [sample_params(kind, rng) for _ in range(args.random)]
        for p in cands:
            v = run_episode(env, opts_sub, make_pattern_policy(p, prior)).pos[-1]
            if v > best_v:
                best, best_v = p, v
        for _ in range(args.local):
            q = perturb(best, rng)
            v = run_episode(env, opts_sub, make_pattern_policy(q, prior)).pos[-1]
            if v > best_v:
                best, best_v = q, v
        log03 = run_episode(env, opts, make_pattern_policy(best, prior), keep_cells=True)
        log1 = run_episode(env1, opts, make_pattern_policy(best, prior), keep_cells=True)
        record(kind, log03, log1, best.as_dict(), args.random + args.local)
    for name, pol in ADAPTIVE.items():
        log03 = run_episode(env, opts, pol)
        log1 = run_episode(env1, opts, pol, keep_cells=True)
        record(name, log03, log1)
    out["sec"] = round(time.perf_counter() - t_start, 1)
    print(f"seed {seed} done in {out['sec']} s: " + ", ".join(f"{k} {v['pos']:.3f}" for k, v in out["methods"].items()),
          flush=True)
    return out


def summarize(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results[0]["methods"])
    rows = []
    for n in names:
        pos = np.array([r["methods"][n]["pos"] for r in results])
        pos1 = np.array([r["methods"][n]["pos_pod1"] for r in results])
        ttd = np.concatenate([[np.inf if x is None else x for x in r["methods"][n]["ttd_min"]] for r in results])
        det = np.isfinite(ttd)
        rows.append(dict(method=n, pos_mean=pos.mean(), pos_std=pos.std(), pos_pod1_mean=pos1.mean(),
                         detect=det.mean(), det30=(ttd <= 30).mean(), ttd_median=float(np.median(ttd[det])) if det.any() else np.nan,
                         wins=int(sum(r["methods"][n]["pos"] >= max(m["pos"] for m in r["methods"].values()) - 1e-12
                                      for r in results))))
    rows.sort(key=lambda r: -r["pos_mean"])
    lines = ["| method | POS (POD 0.3) mean ± sd | POS (POD 1) | true target found (POD 1) | found ≤ 30 min | median time (min) | best in scenarios |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['pos_mean']:.3f} ± {r['pos_std']:.3f} | {r['pos_pod1_mean']:.3f} | "
                     f"{r['detect']:.1%} | {r['det30']:.1%} | {r['ttd_median']:.1f} | {r['wins']}/{len(results)} |")
    (out_dir/"table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    styles = {"straight": ":", "seek": "--", "lead": "--", "greedy": "-"}
    for n in names:
        c = np.mean([r["methods"][n]["pos_curve"] for r in results], axis=0)
        ax[0].plot(T_GRID, c, styles.get(n, "-"), lw=2.2 if n in styles else 1.4, label=n)
        ttd = np.concatenate([[np.inf if x is None else x for x in r["methods"][n]["ttd_min"]] for r in results])
        ax[1].plot(T_GRID, [(ttd <= t).mean() for t in T_GRID], styles.get(n, "-"),
                   lw=2.2 if n in styles else 1.4, label=n)
    ax[0].set(title=f"Mean POS over time on the belief (POD 0.3), {len(results)} scenarios",
              xlabel="minutes after handover", ylabel="POS")
    ax[1].set(title="Single true targets found (POD 1, fresh trajectories from the same model)",
              xlabel="minutes after handover", ylabel="share found")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir/"curves.png", dpi=100)
    plt.close(fig)

    # paths of every method in the first two scenarios
    from rl_search.scenario import make_scenario  # noqa: F401  (paths are stored, prior drawn from seed below)
    fig, axs = plt.subplots(min(2, len(results)), len(names), figsize=(3.2*len(names), 3.4*min(2, len(results))),
                            squeeze=False)
    for i, r in enumerate(results[:2]):
        for j, n in enumerate(names):
            xy = np.array(r["methods"][n]["path"])/1e3
            a = axs[i, j]
            a.plot(xy[:, 0], xy[:, 1], lw=.6)
            a.plot(*xy[0], "k*", ms=6)
            a.set_title(f"s{r['seed']} {n}\nPOS {r['methods'][n]['pos']:.3f}", fontsize=8)
            a.set_aspect("equal")
            a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_dir/"paths.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scen", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--pod", type=float, default=0.3)
    ap.add_argument("--random", type=int, default=24, help="Random placements per pattern")
    ap.add_argument("--local", type=int, default=12, help="Local refinements per pattern")
    ap.add_argument("--opt-particles", type=int, default=1000)
    ap.add_argument("--truths", type=int, default=300, help="Fresh true trajectories per scenario")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/compare_patterns"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed0 + i, args) for i in range(args.n_scen)]
    with Pool(args.workers) as pool:
        results = pool.map(run_scenario, jobs)
    (args.output/"results.json").write_text(json.dumps(results))
    summarize(results, args.output)


if __name__ == "__main__":
    main()
