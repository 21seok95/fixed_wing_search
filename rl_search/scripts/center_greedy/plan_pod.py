"""Exploratory experiment: planning POD vs true POD on the centre scenarios 400-409, SAROPS and greedy, N = 1, 3, 5.

The planner lives in a world where POD = plan_pod: its belief update, its lookahead and its lane and sweep-width formulas
all read that value. The flight it produces is then scored by an exact action replay on the held-out particles under
each true POD. true POD 1.0 gives the share of particles that ever came within sensor range. No separation constraint
(as in the baseline), same scenarios, launch gap and sensor geometry as compare_center. Nothing in the repository
changes. Finished jobs are appended to plan_pod.jsonl, so the run can be resumed.

    python plan_pod.py --smoke
    python plan_pod.py --workers 10
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, HERE]
os.chdir(REPO)

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path(REPO)/"rl_search/data/center_greedy/plan_pod.jsonl"
SEEDS = list(range(400, 410))
PLAN_PODS = (0.3, 0.7, 0.5, 0.1)          # run order: the main contrast first
TRUE_PODS = (0.3, 0.5, 0.7, 0.9, 1.0)
NS = (5, 3, 1)


def make_envs(n, seed, pod):
    from pathlib import Path as P
    from rl_search.config import get_preset
    from rl_search.multi import TrackedEnv
    from rl_search.rl_env import RLConfig
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain
    terr = load_terrain(P("rl_search/data/terrain_seed7_60km.npz"))
    envs = []
    for _ in range(n):
        e = TrackedEnv(terr, get_preset("rc_cessna", SENSORS["new"] + [f"search.pod={pod}"]),
                       RLConfig(stage=3, detect_model="lrc"), seed=seed)
        if abs(e.search.pod - pod) > 1e-12:
            raise RuntimeError("POD override did not apply")
        e.fast = e.skip_cells = True
        envs.append(e)
    return envs


def run_job(job):
    seed, method, n, plan_pod = job
    warnings.filterwarnings("ignore")
    from center_greedy import install_team_greedy
    TeamGreedy = install_team_greedy()
    from rl_search import multi
    from rl_search.accordion import sweep_width
    from rl_search.sarops import SaropsPlanner
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center import center_scenario
    t0 = time.perf_counter()
    probe = make_envs(1, seed, plan_pod)[0]
    scen = center_scenario(probe.sampler, seed)
    opts = dict(scenario=scen, stage=3)
    hold = dict(scenario=holdout_scenario(probe, scen, seed + 10**6), stage=3)
    if method == "greedy":
        pols = [TeamGreedy() for _ in range(n)]
    else:
        W = sweep_width(probe)
        pols = [SaropsPlanner(W=W, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover")
                for _ in range(n)]
    res = multi.run_team_launch(make_envs(n, seed, plan_pod), pols, opts, 60.0, deconflict=False, hard=False)
    plan_s = time.perf_counter() - t0
    scored = {}
    for tp in TRUE_PODS:
        h = multi.run_team_launch(make_envs(n, seed, tp), None, hold, 60.0, actions=res["actions"])
        drift = max(float(np.hypot(*(a[-1] - b[-1]))) for a, b in zip(res["xy"], h["xy"]))
        if drift > 1e-6:
            raise RuntimeError(f"replay left the planned path by {drift} m")
        scored[str(tp)] = dict(end=float(h["pos"][-1]),
                               t=[float(np.interp(m*60, h["t"] - h["t"][0], h["pos"])) for m in range(0, 61, 5)])
    dmin, below, enc, _ = multi.separation_pairs(res)
    out = dict(seed=seed, method=method, n=n, plan_pod=plan_pod, scored=scored,
               dmin=float(dmin) if np.isfinite(dmin) else None, below=float(below), enc=int(enc),
               n_plans=int(sum(getattr(p, "n_plans", 0) or len(getattr(p, "rects", [])) for p in pols)),
               n_lead=int(sum(getattr(p, "n_lead", 0) for p in pols)),
               xy=[x[::2].round(0).tolist() for x in res["xy"]], tt=[t[::2].tolist() for t in res["tt"]],
               actions=res["actions"], plan_s=round(plan_s, 1), sec=round(time.perf_counter() - t0, 1))
    with open(OUT, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"seed {seed} {method} N={n} plan POD {plan_pod}: POS at true 0.7 {scored['0.7']['end']:.3f}, at true 0.3 "
          f"{scored['0.3']['end']:.3f}, plan {plan_s:.0f} s, total {out['sec']:.0f} s", flush=True)
    return out


def pending(jobs):
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["method"], r["n"], r["plan_pod"]))
    return [j for j in jobs if tuple(j) not in done]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.smoke:
        jobs = [(400, m, 1, p) for m in ("greedy", "sarops") for p in (0.7, 0.3)]
    else:
        jobs = [(s, m, n, p) for p in PLAN_PODS for n in NS for m in ("greedy", "sarops") for s in SEEDS]
    jobs = pending(jobs)
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(run_job, jobs, chunksize=1):
            pass
    print(f"done in {(time.perf_counter() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
