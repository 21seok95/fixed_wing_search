"""Evaluation of an RL policy on the validation / test cases, scored like the SAROPS references (docs/rl_mdp_design.md, section 6).

One deterministic flight on the planner particles (the policy sees this belief), then the same actions replayed on the hold-out particles,
A from the camera frames (``area_score.score_frames``) and POS from the team runner, exactly as ``scripts/rl_eval/sarops_ref.py`` does
for SAROPS. Each flight is saved as a web run (``eval-<split>-s<stage>-k<k>-rl-<step>``), shown with the run list filter "검증 n단계".
Cases and envs are cached per process, so an evaluation worker builds each case once.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

_CACHE: dict = {}


def sarops_refs(split: str = "val") -> dict:
    """SAROPS reference scores by (stage, k): {"holdout": {...}, "belief": {...}}."""
    from rl_search.rl_eval_set import EVAL_DIR
    rows = json.loads((EVAL_DIR/f"sarops_ref_{split}.json").read_text())
    return {(r["case"]["stage"], r["case"]["k"]): r for r in rows}


def _case(split: str, stage: int, k: int):
    key = ("case", split, stage, k)
    if key not in _CACHE:
        from rl_search.real_scenario import VehicleScenarios
        from rl_search.rl_eval_set import REGION, make_case, vehicle_config
        from rl_search.rl_real import preload
        st = preload(REGION)
        vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
        _CACHE[key] = make_case(vs, split, stage, k)
    return _CACHE[key]


def _env():
    if "env" not in _CACHE:
        from rl_search.rl_real import RealRLEnv
        _CACHE["env"] = RealRLEnv(seed=0)
    return _CACHE["env"]


def _model(path: str):
    if _CACHE.get("model_path") != path:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        _CACHE["model"], _CACHE["model_path"] = PPO.load(path, device="cpu"), path
    return _CACHE["model"]


def evaluate_case(model_path: str, split: str, stage: int, k: int, step: int, save_web: bool = True) -> dict:
    """Fly the policy on one case; returns the belief and hold-out scores (A, POS, total) and the SAROPS reference of the case."""
    from rl_search.area_score import score_frames
    from rl_search.multi import run_team_launch
    from rl_search.rl_eval_set import REGION, case_dict
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import b64, footprints, particles_payload, team_env
    t_start = time.perf_counter()
    case, scen, vres, hold = _case(split, stage, k)
    model, env = _model(model_path), _env()
    e = env.env
    e.sample_log = []
    obs, info = env.reset(options=case.options(scen))
    s = env.s
    x, w = float(s["area_x"]), float(s["area_w"])
    acts, tt, xy, pos_t, pos = [], [e.t], [e.plant.pose[:2].copy()], [e.t], [0.0]
    rec_t, rec_w = [e.t], [np.round(e.surv*255).astype(np.uint8)]
    term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(float(np.ravel(a)[0]))
        obs, r, term, trunc, info = env.step(a)
        tt.append(e.t)
        xy.append(e.plant.pose[:2].copy())
        pos_t.append(e.t)
        pos.append(info["POS"])
        rec_t.append(e.t)
        rec_w.append(np.round(e.surv*255).astype(np.uint8))
    log = np.array(e.sample_log)
    e.sample_log = None
    terr, cfg = env.terr, e.detector.cfg
    a_b = score_frames(terr, e.h, scen.track, scen.save_dt_s, x, [log], cfg)
    belief = dict(pos=float(info["POS"]), A=float(a_b["A"]), total=float(a_b["A"] + w*info["POS"]))
    # the same actions on the hold-out particles (as sarops_ref)
    eh = team_env(s, REGION, terr, case.seed)
    eh.sample_log = []
    h = run_team_launch([eh], None, case.options(hold), 60.0, actions=[acts])
    a_h = score_frames(terr, eh.h, hold.track, hold.save_dt_s, x, [np.array(eh.sample_log)], eh.detector.cfg)
    pos_h = float(h["pos"][-1])
    holdout = dict(pos=pos_h, A=float(a_h["A"]), total=float(a_h["A"] + w*pos_h))
    try:
        ref = sarops_refs(split).get((stage, k))
    except FileNotFoundError:                                  # SAROPS references of this split not made yet
        ref = None
    out = dict(split=split, stage=stage, k=k, step=step, belief=belief, holdout=holdout, steps=len(acts),
               sarops=ref["holdout"] if ref else None, sec=time.perf_counter() - t_start)
    if save_web:
        dt, n = scen.save_dt_s, e.belief.n
        T = min(int(np.ceil(tt[-1]/dt)) + 1, len(scen.track))
        rt, rw = np.array(rec_t), np.array(rec_w)
        W = np.full((T, n), 255, np.uint8)
        for j in range(T):
            i = np.searchsorted(rt, j*dt, side="right") - 1
            if i >= 0:
                W[j] = rw[i]
        g = a_b.pop("grid")
        uav = dict(rects=[], tasks=[], tt=b64(np.asarray(tt, np.float32)), xy=b64(np.asarray(xy, np.float32)), n_path=len(tt),
                   log_t=b64(log[:, 0].astype(np.float32)), log_xyyaw=b64(log[:, 1:4].astype(np.float32)),
                   quad=b64(footprints(log, cfg).astype(np.float32)), target=b64(log[:, 14:16].astype(np.float32)), n_log=len(log))
        opts = case.options(scen)
        start = opts["start_pose"][:2] if opts.get("stage", 3) in (1, 2) else scen.start
        opt = dict(T=T, N=int(n), dt_s=float(dt), weight=b64(W), uavs=[uav], start=[float(v) for v in start], lkp=[float(v) for v in scen.lkp],
                   pos_t=np.round(np.asarray(pos_t), 1).tolist(), pos=np.round(np.asarray(pos), 4).tolist(), final=belief["pos"],
                   escaped=float(e.belief.escaped), no_clear=0,
                   area={**a_b, "on": True, "x": x, "w": w, "total": belief["total"], "nx": g["nx"], "ny": g["ny"], "cell": g["cell"],
                         "x0": g["x0"], "y0": g["y0"], "bits": b64(g["bits"].astype("<u2")), "first": b64(g["first"])},
                   sec=round(out["sec"], 1), settings=s)
        view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
        rid = f"eval-{split}-s{stage}-k{k}-rl-{step}"
        vc = _CACHE.get("vc")
        if vc is None:
            from rl_search.rl_eval_set import vehicle_config
            vc = _CACHE["vc"] = vehicle_config()
        name = f"RL {step:,}" if isinstance(step, int) else f"RL {step}"
        meta = vw.run_meta(opt, rid, f"{case.label} · {name}", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split=split,
                           stage=stage, case=k, method=name, holdout=holdout, case_info=case_dict(case))
        vw.save_run(rid, dict(opt=opt, res=particles_payload(vres, vc, case.heading, start, 0.0)), meta)
        out["web_run"] = rid
    return out


def eval_job(args):
    """Pool entry: (model_path, split, stage, k, step)."""
    import warnings
    warnings.filterwarnings("ignore")
    return evaluate_case(*args)
