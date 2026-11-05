"""Exploratory measure: how the x% region of the unweighted particles grows and moves over time, against what the UAVs can
sweep. Region: smoothed density as in rl_env._zone, threshold set so that exactly x% of the particles are inside."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import warnings
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter

SEEDS = list(range(400, 410))
XS = (50, 70, 85, 95)
MINUTES = (10, 20, 30, 40, 50, 60)
T_ALL = np.arange(5, 61, 5)
SWEEP_KM2_MIN = 15*60*286/1e6


def regions(terr, h, pts, xs):
    tc = float(terr.axis[1] - terr.axis[0])
    nt = len(terr.axis)
    ij = np.clip(np.round((pts - terr.axis[0])/tc).astype(int), 0, nt - 1)
    hist = np.zeros((nt, nt))
    np.add.at(hist, (ij[:, 1], ij[:, 0]), 1.0)
    dens = gaussian_filter(hist, h/tc, mode="constant")
    dens[terr.forbidden] = 0
    at = dens[ij[:, 1], ij[:, 0]]
    out = {}
    for x in xs:
        out[x] = (dens >= max(np.quantile(at, 1 - x/100), 1e-12)) & (dens > 0)
    return out, tc


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario
    env = make_envs(1, seed)[0]
    scen = center_scenario(env.sampler, seed)
    row = {}
    union = {x: None for x in XS}
    reg_at = {}
    for m in T_ALL:
        pts = np.asarray(scen.track[int(m*60/scen.save_dt_s)], float)
        reg, tc = regions(env.terr, env.h, pts, XS)
        reg_at[int(m)] = reg
        for x in XS:
            union[x] = reg[x] if union[x] is None else union[x] | reg[x]
    km2 = tc*tc/1e6
    for x in XS:
        row[x] = dict(area={m: float(reg_at[m][x].sum()*km2) for m in MINUTES}, union=float(union[x].sum()*km2),
                      keep_30_60=float((reg_at[30][x] & reg_at[60][x]).sum()/max(reg_at[60][x].sum(), 1)),
                      keep_50_60=float((reg_at[50][x] & reg_at[60][x]).sum()/max(reg_at[60][x].sum(), 1)))
    return seed, row


def main():
    with Pool(3) as pool:
        res = dict(pool.map(job, SEEDS, chunksize=1))
    print(f"sweep capacity: {SWEEP_KM2_MIN:.3f} km2 per minute per UAV; in 60 min 1 UAV {SWEEP_KM2_MIN*60:.1f}, 3 UAVs {3*SWEEP_KM2_MIN*60:.1f}, 5 UAVs {5*SWEEP_KM2_MIN*60:.1f} km2")
    print("\nx% | mean region km2 at " + " ".join(f"{m}min" for m in MINUTES) + " | range at 60 min | union 5-60 min | region at 60 that was in region at 30 / at 50")
    for x in XS:
        a = {m: np.mean([res[s][x]["area"][m] for s in SEEDS]) for m in MINUTES}
        e = [res[s][x]["area"][60] for s in SEEDS]
        print(f"{x} | " + " ".join(f"{a[m]:.1f}" for m in MINUTES) + f" | {min(e):.1f}-{max(e):.1f} | {np.mean([res[s][x]['union'] for s in SEEDS]):.1f} | "
              f"{np.mean([res[s][x]['keep_30_60'] for s in SEEDS]):.2f} / {np.mean([res[s][x]['keep_50_60'] for s in SEEDS]):.2f}")
    print("\nmean growth km2 per minute between 30 and 60 min vs sweep capacity (1 UAV 0.257, 3 UAVs 0.772, 5 UAVs 1.287):")
    for x in XS:
        g = [(res[s][x]["area"][60] - res[s][x]["area"][30])/30 for s in SEEDS]
        print(f"  {x}%: mean {np.mean(g):+.3f}, range {min(g):+.3f} to {max(g):+.3f}, scenarios with growth above 1 UAV capacity {sum(v > .257 for v in g)}/10, above 5 UAVs {sum(v > 1.287 for v in g)}/10")
    print("\nper scenario area at 60 min, x = 50/70/85/95:")
    for s in SEEDS:
        print(s, " ".join(f"{res[s][x]['area'][60]:.1f}" for x in XS))


if __name__ == "__main__":
    main()
