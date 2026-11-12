"""Old flights scored with the new A (step 1 of the A change agreed with the user on 2026-11-11, docs/rl_mdp_design.md).

The 30 test cases (stage 3) flown as before: the RL policy amboy_v1 (6.2M steps, window-A observation) and SAROPS planning on the window
A (x 70, as sarops_ref). Each flight is replayed on the hold-out particles and scored with both A's: the window A (x 70, cells swept in
5 min windows) and the seen-particle A (``area_score.SeenArea``: x 80, seen = POD since the hand-over >= 0.5), with POS. It also records
the turn rate of every 1 s sample and when each A and POS was gained, to see how much came while turning. SAROPS flights are saved as
web runs (eval-test-s3-k<k>-sarops), since only their scores came over from the Ubuntu PC.

    python -m rl_search.scripts.rl_eval.rescore_area --workers 5

Another policy on the same cases (rows and files under its label, e.g. a fine-tuning run of 2026-11-11):

    python -m rl_search.scripts.rl_eval.rescore_area --model rl_search/data/rl_real/amboy_t1/best_stage3.zip --label T1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

MODEL = "rl_search/data/rl_real/amboy_v1/best_stage3.zip"
OUT = Path("rl_search/data/real/amboy/rl_eval/rescore")
X_OLD, X_NEW, POD = 70.0, 80.0, 0.5
TURN_DPS = 5.0                     # the blur factor is 1 up to 5 deg/s


def old_settings() -> dict:
    from rl_search.scripts.rl_eval.sarops_ref import settings
    return settings() | dict(area_def="window", area_x=X_OLD)


def window_gains(terr, h, scen, x, log, cfg):
    """Window A of a sample log (as ``score_frames``) and the gain of each row."""
    from rl_search.area_score import AreaScore
    a = AreaScore(terr, h, scen.track, scen.save_dt_s, x, 0.0, track_curve=True)
    g = np.zeros(len(log))
    for i, row in enumerate(log):
        g[i] = a.mark_frame(row[1], row[2], (row[5:8][None], row[8:11][None], row[11:14][None]), row[0], cfg)
    return float(a.curve().mean()), g


def fly_rl(case, scen, model_path=MODEL, settings=None):
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_real import RealRLConfig, RealRLEnv
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    env = RealRLEnv(RealRLConfig(settings=dict(settings or dict(area_def="window", area_x=X_OLD))), seed=0)
    obs, _ = env.reset(options=case.options(scen))
    acts, term, trunc = [], False, False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(float(np.ravel(a)[0]))
        obs, _, term, trunc, _ = env.step(a)
    return [acts], env.s


def fly_sarops(case, scen, vres, terr):
    from dataclasses import asdict
    from rl_search.rl_eval_set import REGION, case_dict, vehicle_config
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import run_case
    s = old_settings()
    vc = vehicle_config()
    view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
    out = run_case(REGION, scen, vres, vc, view, s, options=case.options(scen), terr=terr)
    o = out["opt"]
    rid = f"eval-test-s3-k{case.k}-sarops"
    meta = vw.run_meta(o, rid, f"{case.label} · SAROPS", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split="test", stage=3,
                       case=case.k, method="SAROPS", case_info=case_dict(case))
    vw.save_run(rid, dict(opt=o, res=out["res"]), meta)
    return out["actions"], s


def run_job(job):
    method, k = job[:2]
    model_path, settings = (job[2], job[3]) if len(job) > 2 else (MODEL, None)
    warnings.filterwarnings("ignore")
    from rl_search.area_score import SeenArea
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, case_dict, make_case, vehicle_config
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.web_optimize import team_env
    t0 = time.perf_counter()
    st = preload(REGION)
    terr = st["terr"]
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, "test", 3, k)
    actions, s = fly_rl(case, scen, model_path, settings) if method != "sarops" else fly_sarops(case, scen, vres, terr)
    # the same actions on the hold-out particles, both A's
    eh = team_env(s, REGION, terr, case.seed, area=True)
    eh.area = SeenArea(hold.track, hold.exit_step, hold.save_dt_s, X_NEW, POD)
    eh.pos_w, eh.area_w = 1.0, 1.0
    eh.sample_log = []
    h = run_team_launch([eh], None, case.options(hold), 60.0, actions=actions)
    log = np.array(eh.sample_log)
    log = log[np.argsort(log[:, 0], kind="stable")]
    cfg = eh.detector.cfg
    a_old, g_old = window_gains(terr, eh.h, hold, X_OLD, log, cfg)
    sa = eh.area
    a_new = sa.A
    g_new = np.array(sa.gains, float).reshape(-1, 3)
    pos = float(h["pos"][-1])
    t_ho = float(eh.launch_track[-1][0])
    om = np.degrees(9.80665*np.tan(log[:, 4])/eh.v)                     # turn rate of each 1 s sample (deg/s)
    after = log[:, 0] > t_ho + 1e-6
    turning = np.abs(om) > TURN_DPS
    acts = np.asarray(actions[0], float)
    big = acts[np.abs(acts) >= 0.05]
    rev = int((np.sign(big[1:]) != np.sign(big[:-1])).sum()) if len(big) > 1 else 0
    fly_min = max((log[after, 0].max() - t_ho)/60.0, 1e-9) if after.any() else 1e-9

    def step_turning(t_end):
        m = (log[:, 0] > t_end - 5.0 + 1e-6) & (log[:, 0] <= t_end + 1e-6)
        return bool(np.abs(om[m]).mean() > TURN_DPS) if m.any() else False

    new_turn = np.array([step_turning(t) for t in g_new[:, 0]], bool) if len(g_new) else np.zeros(0, bool)
    pt, pp = np.asarray(h["t"], float), np.asarray(h["pos"], float)
    dpos = np.diff(np.concatenate([[0.0], pp]))
    pos_turn = np.array([step_turning(t) for t in pt], bool)
    row = dict(method=method, k=k, case=case_dict(case), pos=pos, A_old=a_old, A_new=a_new, total_old=a_old + pos, total_new=a_new + pos,
               turn=dict(share=float(turning[after].mean()) if after.any() else 0.0, mean_abs=float(np.abs(om[after]).mean()) if after.any() else 0.0,
                         reversals_per_min=rev/fly_min),
               gained_turning=dict(A_old=float(g_old[turning].sum()/max(g_old.sum(), 1e-12)),
                                   A_new=float(g_new[new_turn, 1].sum()/max(g_new[:, 1].sum(), 1e-12)) if len(g_new) else 0.0,
                                   pos=float(dpos[pos_turn].sum()/max(dpos.sum(), 1e-12))),
               seen_share=float(sa.seen.mean()), region_km2=float(sa.size.mean()*25.0**2/1e6), t_handover=t_ho, steps=len(acts),
               sec=round(time.perf_counter() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    from rl_search.scripts.real_map.web_optimize import footprints
    np.savez_compressed(OUT/f"{method}_k{k}.npz", log=log[:, :5].astype(np.float32), quad=footprints(log, cfg).astype(np.float32),
                        g_old=g_old.astype(np.float32), g_new=g_new,
                        t_seen=sa.t_seen.astype(np.float32), pos_t=pt, pos=pp, actions=acts.astype(np.float32), curve=sa.curve(),
                        size=sa.size)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--methods", nargs="*", default=["sarops", "rl"])
    ap.add_argument("--model", default=None, help="another policy: flown under --label instead of the methods above")
    ap.add_argument("--label", default=None)
    ap.add_argument("--area-def", default="window", choices=("window", "seen"), help="A of the policy's observation (its training A)")
    args = ap.parse_args()
    out = OUT.parent/"rescore_area_test.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["method"], r["k"]) for r in rows}
    if args.model:
        obs = dict(area_def="seen", area_x=X_NEW) if args.area_def == "seen" else dict(area_def="window", area_x=X_OLD)
        jobs = [(args.label, k, args.model, obs) for k in range(args.cases) if (args.label, k) not in done]
    else:
        jobs = [(m, k) for m in args.methods for k in range(args.cases) if (m, k) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done before)", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            rows.append(r)
            print(f"{r['method']:6s} k{r['k']:02d}: POS {r['pos']:.3f}, A old {r['A_old']:.3f} -> new {r['A_new']:.3f}, "
                  f"turning {r['turn']['share']:.0%}, A old while turning {r['gained_turning']['A_old']:.0%}, "
                  f"new {r['gained_turning']['A_new']:.0%}, {r['sec']:.0f} s", flush=True)
            out.write_text(json.dumps(sorted(rows, key=lambda x: (x["method"], x["k"])), ensure_ascii=False, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
