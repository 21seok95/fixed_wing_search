"""Pre-built training scenarios for the RL environment (docs/rl_mdp_design.md, "구현 3단계"): a new vehicle scenario takes about
5 s (vehicle simulation 2 s, A score regions 3 s) and every parallel env waits for a reset, so the scenarios are made once.

Scenario i has seed 10 000 000 + i (rl_real.training_scenario: LKP rule of the validation cases, its own seeds). Each file holds the
scenario and its area score for the SAROPS reference settings, pickled: ``seen`` (area_score.SeenArea, x 80, POD 0.5, the A from
2026-11-11) or, with the window A, ``area`` (AreaScore, x 70). The env loads one per new scenario and builds a missing seen A itself
(1.5 s), so a window-A pool still works. Existing files are kept, so the pool can be extended or a stopped run resumed. Not in git
(about 19 MB per scenario, 33 MB with the seen A); rebuild on another PC with:

    python -m rl_search.scripts.rl_eval.make_pool --n 1000 --workers 6

R10 (targets that never stop, no A, about 26 MB per scenario, 600 = about 16 GB):

    python -m rl_search.scripts.rl_eval.make_pool --multi --no-stop --no-area --n 600 --workers 8
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


def _init(region, area_def="window", diverse=False, sigma=None, sigma_off=0.0, src=None, no_stop=False, no_area=False):
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.web_optimize import team_env
    from rl_search.scripts.rl_eval.sarops_ref import settings
    st = preload(region)
    s = settings() | dict(area_def=area_def)
    env = team_env(s, region, st["terr"], 0, area=True)
    _W.update(st=st, s=s, h=env.h, R=env.lrc_R, vs=VehicleScenarios(st["sampler"], vehicle_config(), region=region, net=st["net"]),
              diverse=diverse, sigma=float(sigma) if sigma else env.h, sigma_off=float(sigma_off or 0.0), src=src, region=region,
              no_stop=no_stop, no_area=no_area)


def job(args):
    seed, out = args[:2]
    typ = args[2] if len(args) > 2 else ""
    from rl_search.area_score import AreaScore, RealtimeArea, SeenArea
    from rl_search.rl_real import training_scenario
    t0 = time.perf_counter()
    old = {}
    if out.exists():                          # an existing scenario: keep it and its other area scores, add this one
        with open(out, "rb") as f:
            old = pickle.load(f)
    elif _W.get("src") is not None and (_W["src"]/out.name).exists():     # the same scenario from another pool (--src): only the region is new
        with open(_W["src"]/out.name, "rb") as f:
            o = pickle.load(f)
        old = {k: o[k] for k in ("scen", "vc", "type", "seed") if k in o}
    vc = old.get("vc")
    if "scen" in old:
        scen = old["scen"]
    elif typ:                                 # typed scenario (R5)
        from rl_search.rl_eval_set import multi_training_scenario
        scen, vc = multi_training_scenario(_W["vs"], seed, typ, nostop=bool(_W.get("no_stop")))
    elif _W.get("diverse"):
        from rl_search.rl_eval_set import diverse_training_scenario
        scen = diverse_training_scenario(_W["vs"], seed)[0]
    else:
        scen = training_scenario(_W["vs"], seed)
    s = _W["s"]
    if _W.get("no_area"):                     # R10: the scenario only (the env trains without A), about 26 MB
        d = dict(scen=scen, seed=seed)
    elif s.get("area_def", "window") == "seen":
        d = dict(scen=scen, seen=SeenArea(scen.track, scen.exit_step, scen.save_dt_s, float(s["area_x"]), float(s["seen_pod"])), seed=seed)
    elif s.get("area_def", "window") in ("realtime", "realtime_q"):     # one region for both (the quality is applied when sweeping)
        from rl_search.area_score import road_mask
        kw = dict(h_off=_W["sigma_off"], road=road_mask(_W["region"])) if _W.get("sigma_off") else {}
        d = dict(scen=scen, realtime=RealtimeArea(_W["st"]["terr"], _W["sigma"], scen.track, scen.exit_step, scen.save_dt_s, float(s["area_x"]),
                                                  **kw), seed=seed)
    else:
        d = dict(scen=scen, area=AreaScore(_W["st"]["terr"], _W["h"], scen.track, scen.save_dt_s, float(s["area_x"]), _W["R"]), seed=seed)
    d = {**old, **d} | ({"type": typ, "vc": vc} if typ else {})
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return seed, time.perf_counter() - t0


def main():
    from rl_search.rl_real import POOL_DIR, POOL_DIR_DIVERSE, POOL_DIR_MULTI, RealRLConfig, pool_dir_multi, preload
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path(POOL_DIR))
    ap.add_argument("--region", default="amboy")
    ap.add_argument("--diverse", action="store_true", help="diverse training scenarios (rl_eval_set.DIVERSE) into POOL_DIR_DIVERSE")
    ap.add_argument("--multi", action="store_true", help="typed scenarios (rl_eval_set.MULTI, R5), the types in turn, into POOL_DIR_MULTI")
    ap.add_argument("--area-sigma", type=float, default=None, help="blur width (m) of the real-time A region (default 150)")
    ap.add_argument("--area-def", default="window", choices=("window", "realtime", "seen"), help="area score stored with each scenario (realtime also serves realtime_q)")
    ap.add_argument("--area-sigma-off", type=float, default=0.0,
                    help="blur width (m) of the off-road particles in the real-time A region (0 = same); with --multi the pool goes to "
                         "rl_pool_multi_o<value> and takes its scenarios from rl_pool_multi (only the region is rebuilt)")
    ap.add_argument("--src", type=Path, default=None, help="take the scenarios of the same file names from this pool, rebuild only the region")
    ap.add_argument("--no-stop", action="store_true",
                    help="R10 (user 2026-11-14): typed targets that never stop and keep a steady speed of their own (rl_eval_set.NOSTOP); "
                         "with --multi the pool goes to rl_pool_multi_ns")
    ap.add_argument("--no-area", action="store_true", help="store the scenario only, no A region (R10 trains without A)")
    args = ap.parse_args()
    if args.diverse and str(args.out) == POOL_DIR:
        args.out = Path(POOL_DIR_DIVERSE)
    if args.multi and str(args.out) == POOL_DIR:
        args.out = Path(pool_dir_multi({"area_sigma_off": args.area_sigma_off, "no_stop": args.no_stop}))
        if args.area_sigma_off and args.src is None and not args.no_stop:
            args.src = Path(POOL_DIR_MULTI)
    args.out.mkdir(parents=True, exist_ok=True)
    base = RealRLConfig().seed_base
    if args.multi:                           # <type>_s<seed>.pkl, the types in turn so every type has n / 3
        from rl_search.rl_eval_set import MULTI
        types = MULTI["types"]
        todo = [(base + i, args.out/f"{types[i % len(types)]}_s{base + i}.pkl", types[i % len(types)]) for i in range(args.n)
                if not (args.out/f"{types[i % len(types)]}_s{base + i}.pkl").exists()]
    else:
        todo = [(base + i, args.out/f"s{base + i}.pkl") for i in range(args.n) if not (args.out/f"s{base + i}.pkl").exists()]
    if args.area_def != "window" and not args.multi:     # scenarios of an older pool get the new area score added in place
        import pickle as _p
        todo += [(base + i, args.out/f"s{base + i}.pkl") for i in range(args.n) if (args.out/f"s{base + i}.pkl").exists()
                 and args.area_def not in _p.load(open(args.out/f"s{base + i}.pkl", "rb"))]
    print(f"{args.n - len(todo)} already there, making {len(todo)}", flush=True)
    preload(args.region)                                    # shared with the forked workers
    t0 = time.perf_counter()
    if args.area_sigma_off and not args.no_area:
        from rl_search.area_score import road_mask
        road_mask(args.region)                              # build the road cells once before the workers fork
    with get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.region, args.area_def, args.diverse, args.area_sigma,
                                                                             args.area_sigma_off, args.src, args.no_stop, args.no_area)) as pool:
        for i, (seed, sec) in enumerate(pool.imap_unordered(job, todo, chunksize=1), 1):
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"{i}/{len(todo)} done, {el/60:.1f} min, last {sec:.1f} s, about {el/i*(len(todo) - i)/60:.0f} min left", flush=True)
    print("pool:", args.out)


if __name__ == "__main__":
    main()
