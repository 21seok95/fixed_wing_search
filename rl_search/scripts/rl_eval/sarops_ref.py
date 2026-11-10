"""SAROPS references on the RL validation or test cases (rl_search.rl_eval_set, docs/rl_mdp_design.md).

One flight per case on the planner particles with the web defaults (A + POS, x 70, w 1, 1 UAV) and the gimbal fixed forward
(pre-look 0 s), scored on the hold-out particles by replaying the same actions. Each flight is saved as a web run (run list filter
"검증 n단계") and the scores go to rl_search/data/real/amboy/rl_eval/sarops_ref_<split>.json.

    python -m rl_search.scripts.rl_eval.sarops_ref --split val --workers 5
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import get_context

import numpy as np

OVERRIDE = dict(cap="A", score="area", n_uav=1, prelook_s=0)


def settings() -> dict:
    from rl_search.scripts.real_map.web_optimize import FIELDS
    return {f["name"]: f["value"] for f in FIELDS} | OVERRIDE


def run_job(job):
    split, stage, k = job
    warnings.filterwarnings("ignore")
    from dataclasses import asdict
    from rl_search.area_score import score_frames
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import VehicleScenarios, chungju_preset, terrain_path
    from rl_search.rl_eval_set import REGION, case_dict, make_case, vehicle_config
    from rl_search.scenario import ScenarioSampler
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import run_case, team_env
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter()
    terr = load_terrain(terrain_path(REGION))
    vc = vehicle_config()
    vs = VehicleScenarios(ScenarioSampler(terr, chungju_preset().scenario), vc, region=REGION)
    case, scen, vres, hold = make_case(vs, split, stage, k)
    s = settings()
    view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
    out = run_case(REGION, scen, vres, vc, view, s, options=case.options(scen), terr=terr)
    env = team_env(s, REGION, terr, case.seed)                       # the same actions on the hold-out particles
    env.sample_log = []
    h = run_team_launch([env], None, case.options(hold), 60.0, actions=out["actions"])
    drift = float(np.hypot(*(np.asarray(h["xy"][0][-1]) - np.asarray(out["paths"][0][-1]))))
    a_h = score_frames(terr, env.h, hold.track, hold.save_dt_s, float(s["area_x"]), [np.array(env.sample_log)], env.detector.cfg)
    pos_h = float(h["pos"][-1])
    holdout = dict(pos=pos_h, A=float(a_h["A"]), total=float(a_h["A"] + float(s["area_w"])*pos_h))
    o = out["opt"]
    rid = f"eval-{split}-s{stage}-k{k}-sarops"
    meta = vw.run_meta(o, rid, f"{case.label} · SAROPS", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split=split, stage=stage,
                       case=k, method="SAROPS", holdout=holdout, case_info=case_dict(case))
    vw.save_run(rid, dict(opt=o, res=out["res"]), meta)
    return dict(id=rid, case=case_dict(case), belief=dict(pos=o["final"], A=o["area"]["A"], total=o["area"]["total"]), holdout=holdout,
                plans=meta["plans"], drift_m=drift, sec=round(time.perf_counter() - t0, 1))


def main():
    from rl_search.rl_eval_set import EVAL_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--cases", type=int, default=None, help="Cases per stage (default 5 for val, 30 for test)")
    ap.add_argument("--workers", type=int, default=5, help="Processes at once (each about 1.2 GB)")
    args = ap.parse_args()
    n = args.cases or (5 if args.split == "val" else 30)
    jobs = [(args.split, st, k) for st in args.stages for k in range(n)]
    out = EVAL_DIR/f"sarops_ref_{args.split}.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {r["id"] for r in rows}
    rows = [r for r in rows if r["id"] not in {f"eval-{args.split}-s{j[1]}-k{j[2]}-sarops" for j in jobs}]
    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            rows.append(r)
            c, hh = r["case"], r["holdout"]
            print(f"{r['id']}: t0 {c['t0_s']/60:.1f} min, hold-out A + POS {hh['total']:.3f} (POS {hh['pos']:.3f}, A {hh['A']:.3f}), "
                  f"belief {r['belief']['total']:.3f}, plans {r['plans']}, replay drift {r['drift_m']:.2g} m, {r['sec']:.0f} s", flush=True)
            out.write_text(json.dumps(sorted(rows, key=lambda x: x["id"]), ensure_ascii=False, indent=1))
    print("saved", out, "previously", len(done))


if __name__ == "__main__":
    main()
