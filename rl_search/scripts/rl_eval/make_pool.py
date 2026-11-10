"""Pre-built training scenarios for the RL environment (docs/rl_mdp_design.md, "구현 3단계"): a new vehicle scenario takes about
5 s (vehicle simulation 2 s, A score regions 3 s) and every parallel env waits for a reset, so the scenarios are made once.

Scenario i has seed 10 000 000 + i (rl_real.training_scenario: LKP rule of the validation cases, its own seeds). Each file holds the
scenario and its AreaScore (x 70, the SAROPS reference settings), pickled; the env loads one per new scenario. Existing files are
kept, so the pool can be extended or a stopped run resumed. Not in git (about 19 MB per scenario); rebuild on another PC with:

    python -m rl_search.scripts.rl_eval.make_pool --n 1000 --workers 6
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

_W = {}


def _init(region):
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.web_optimize import team_env
    from rl_search.scripts.rl_eval.sarops_ref import settings
    st = preload(region)
    s = settings()
    env = team_env(s, region, st["terr"], 0, area=True)
    _W.update(st=st, s=s, h=env.h, R=env.lrc_R, vs=VehicleScenarios(st["sampler"], vehicle_config(), region=region, net=st["net"]))


def job(args):
    seed, out = args
    from rl_search.area_score import AreaScore
    from rl_search.rl_real import training_scenario
    t0 = time.perf_counter()
    scen = training_scenario(_W["vs"], seed)
    area = AreaScore(_W["st"]["terr"], _W["h"], scen.track, scen.save_dt_s, float(_W["s"]["area_x"]), _W["R"])
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(scen=scen, area=area, seed=seed), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return seed, time.perf_counter() - t0


def main():
    from rl_search.rl_real import POOL_DIR, RealRLConfig, preload
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path(POOL_DIR))
    ap.add_argument("--region", default="amboy")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = RealRLConfig().seed_base
    todo = [(base + i, args.out/f"s{base + i}.pkl") for i in range(args.n) if not (args.out/f"s{base + i}.pkl").exists()]
    print(f"{args.n - len(todo)} already there, making {len(todo)}", flush=True)
    preload(args.region)                                    # shared with the forked workers
    t0 = time.perf_counter()
    with get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.region,)) as pool:
        for i, (seed, sec) in enumerate(pool.imap_unordered(job, todo, chunksize=1), 1):
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"{i}/{len(todo)} done, {el/60:.1f} min, last {sec:.1f} s, about {el/i*(len(todo) - i)/60:.0f} min left", flush=True)
    print("pool:", args.out)


if __name__ == "__main__":
    main()
