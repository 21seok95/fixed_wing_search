"""Exploratory diagnostic: what differs between the centre scenarios (400-409) and the existing ones (300-329)?"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"


def spread(pts, cell=500.0, level=.9):
    """Area (km^2) of the smallest set of 500 m cells holding ``level`` of the particles."""
    _, c = np.unique(np.floor(pts/cell).astype(int), axis=0, return_counts=True)
    c = np.sort(c)[::-1]
    return float((np.searchsorted(np.cumsum(c), level*len(pts)) + 1)*(cell/1000)**2)


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario
    p = make_envs(1, seed)[0]
    scen = (center_scenario if seed >= 400 else make_scenario)(p.sampler, seed)
    tr = np.asarray(scen.track, float)
    dt = scen.save_dt_s
    row = dict(seed=seed, edge_km=float((p.half_map - np.abs(scen.lkp).max())/1000), esc=scen.metrics["escaped_end"],
               launch_km=scen.metrics["start_lkp_km"], speed=scen.speed_mps*3.6, reach=scen.metrics["reachable"],
               intercept_s=scen.metrics["intercept_s"])
    ax = p.terr.axis
    Y, X = np.meshgrid(ax, ax, indexing="ij")
    near = np.hypot(X - scen.lkp[0], Y - scen.lkp[1]) < 15000
    row["forbidden_15km"] = float(p.terr.forbidden[near].mean())
    for m in (15, 30, 60):
        k = int(round(m*60/dt))
        alive = (scen.exit_step < 0) | (scen.exit_step > k)
        pts = tr[k][alive]
        row[f"area{m}"] = spread(pts)
        row[f"dist{m}"] = float(np.hypot(*(pts - scen.lkp).T).mean()/1000)
    return row


def main():
    with Pool(4) as pool:
        rows = pool.map(job, list(range(300, 330)) + list(range(400, 410)), chunksize=1)
    (HERE/"scenario_diag.json").write_text(json.dumps(rows))
    multi = {(r["seed"], r["n"]): r["hold"] for r in json.load(open("rl_search/data/compare_multi_sarops/results.json"))}
    center = {}
    for r in json.load(open("rl_search/data/compare_center/results.json")):
        center[(r["seed"], r["mode"])] = r["hold_pos"]
    for r in rows:
        s = r["seed"]
        r["pos1"] = multi[(s, 1)] if s < 400 else center[(s, "1")]
        r["pos5"] = multi[(s, 5)] if s < 400 else center[(s, "5 sep")]
    keys = ["edge_km", "esc", "launch_km", "speed", "reach", "forbidden_15km", "area15", "area30", "area60", "dist60",
            "pos1", "pos5"]
    print("seed | " + " | ".join(keys))
    for r in rows:
        print(r["seed"], " ".join(f"{r[k]:.3f}" if k in ("esc", "reach", "forbidden_15km", "pos1", "pos5") else f"{r[k]:.1f}" for k in keys))
    A = [r for r in rows if r["seed"] < 400]
    C = [r for r in rows if r["seed"] >= 400]
    near_edge = [r for r in A if r["edge_km"] < 5]
    inner = [r for r in A if r["edge_km"] >= 10]
    print("\nmeans:")
    print("group n " + " ".join(keys))
    for name, g in (("300-329", A), ("edge<5km", near_edge), ("edge>=10km", inner), ("centre", C)):
        print(name, len(g), " ".join(f"{np.mean([r[k] for r in g]):.3f}" for k in keys))
    print("\nSpearman with POS (N=1) over 300-329:")
    from scipy.stats import spearmanr
    for k in ("edge_km", "esc", "launch_km", "speed", "reach", "forbidden_15km", "area15", "area30", "area60", "dist60"):
        rho, p = spearmanr([r[k] for r in A], [r["pos1"] for r in A])
        print(f"  {k}: rho {rho:+.2f} (p {p:.3f})")


if __name__ == "__main__":
    main()
