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


def sarops_refs(split: str = "val", key: str = "window") -> dict:
    """SAROPS reference scores by (stage, k): {"holdout": {...}, "belief": {...}}, for a reference set (sarops_ref.ref_key)."""
    from rl_search.scripts.rl_eval.sarops_ref import ref_path
    rows = json.loads(ref_path(split, key).read_text())
    return {(r["case"]["stage"], r["case"]["k"]): r for r in rows}


def run_area_def(run_dir) -> str:
    """Reference set of a training run from its config.json (A definition and turn-rate factor); "window" for runs made before 2026-11-11."""
    from pathlib import Path
    from rl_search.scripts.rl_eval.sarops_ref import ref_key
    cfg = Path(run_dir)/"config.json"
    if not cfg.exists():
        return "window"
    c = json.loads(cfg.read_text())
    tau = float(c.get("blur_tau", 0.0) or 0.0)
    dead = float(c.get("blur_dead_roll", 0.0) or 0.0) if tau else 0.0
    return ref_key(c.get("area_def", "window"), tau, float(c.get("area_roll_max", 20.0) or 20.0), float(c.get("pos_w", 1.0) or 1.0), dead,
                   float(c.get("blur_tau_low", 0.0) or 0.0) if dead else 0.0, c.get("area_sigma") or None, float(c.get("area_img_min", 0.0) or 0.0))


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


def _env(settings: dict | None = None):
    key = ("env", json.dumps(settings or {}, sort_keys=True))
    if key not in _CACHE:
        from rl_search.rl_real import RealRLConfig, RealRLEnv
        _CACHE[key] = RealRLEnv(RealRLConfig(settings=dict(settings or {})), seed=0)
    return _CACHE[key]


def _model(path: str):
    if _CACHE.get("model_path") != path:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        _CACHE["model"], _CACHE["model_path"] = PPO.load(path, device="cpu"), path
    return _CACHE["model"]


def evaluate_case(model_path: str, split: str, stage: int, k: int, step: int, save_web: bool = True, settings: dict | None = None,
                  launch_s: float | None = None, built: tuple | None = None, web: bool = False, progress=None) -> dict:
    """Fly the policy on one case; returns the belief and hold-out scores (A, POS, total) and the SAROPS reference of the case.
    ``settings``: overrides of the SAROPS reference settings (e.g. dict(area_def="window", area_x=70) for a window-A model).
    ``launch_s``: launch this long after the report instead of the case's own delay (same particles, target model and start point;
    user 2026-11-14: wider spread at the launch). No SAROPS reference then (the references are for the case's own launch).
    ``built``: (case, scenario, simulation result, hold-out scenario, VehicleConfig) instead of a validation / test case (the web viewer's
    scenario, scripts/real_map/web_rl.py); ``web``: return the replay payload as ``out["data"]`` and the run-list fields as
    ``out["meta_extra"]`` instead of saving a web run; ``progress(stage, minutes)`` is called once a simulated minute."""
    from rl_search.area_score import RealtimeArea, SeenArea, score_frames
    from rl_search.rl_eval_set import REGION, case_dict
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import X_WINDOW_REF, b64, footprints, particles_payload, realtime_payload, seen_payload
    from rl_search.scripts.rl_eval.sarops_ref import holdout_replay
    t_start = time.perf_counter()
    case, scen, vres, hold = built[:4] if built is not None else _case(split, stage, k)
    if launch_s is not None:
        from dataclasses import replace
        case = replace(case, t_launch_s=float(launch_s), info=dict(case.info) | {"delay_s": float(launch_s), "delay_case_s": float(case.t_launch_s)})
    model, env = _model(model_path), _env(settings)
    e = env.env
    e.sample_log = []
    obs, info = env.reset(options=case.options(scen))
    s = env.s
    x, w = float(s["area_x"]), float(s["area_w"])
    acts, tt, xy, pos_t, pos = [], [e.t], [e.plant.pose[:2].copy()], [e.t], [0.0]
    if getattr(env, "transit_actions", None):           # distance hand-over: the rule's transit comes first (path, POS, hold-out replay)
        t0_, x0_, y0_ = env.transit_start
        tr = env.transit_track
        acts = list(env.transit_actions)
        tt, xy = [t0_] + [r[0] for r in tr], [np.array([x0_, y0_])] + [np.array(r[1:3]) for r in tr]
        pos_t, pos = [t0_] + [r[0] for r in tr], [0.0] + [r[3] for r in tr]
    rec_t, rec_w = [e.t], [np.round(e.surv*255).astype(np.uint8)]
    end_rec = []          # once a minute: t, POS, share of the remaining belief within 10 km, distance to the map edge (end-of-flight check)
    term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(float(np.ravel(a)[0]))
        obs, r, term, trunc, info = env.step(a)
        tt.append(e.t)
        xy.append(e.plant.pose[:2].copy())
        pos_t.append(e.t)
        pos.append(info["POS"])
        if progress is not None and len(acts) % 12 == 0:
            progress("RL 비행", (e.t - float(case.t_launch_s))/60.0)
        if len(acts) % 12 == 0:
            b = e.belief
            Pb, wb = b.positions(e.t), b.w
            dd = np.hypot(Pb[:, 0] - e.plant.pose[0], Pb[:, 1] - e.plant.pose[1])
            end_rec.append((e.t, float(info["POS"]), float(wb[dd < 10000.0].sum()/max(wb.sum(), 1e-12)),
                            float(env.half - max(abs(e.plant.pose[0]), abs(e.plant.pose[1])))))
        rec_t.append(e.t)
        rec_w.append(np.round(e.surv*255).astype(np.uint8))
    log = np.array(e.sample_log)
    e.sample_log = None
    terr, cfg = env.terr, e.detector.cfg
    from rl_search.scripts.rl_eval.sarops_ref import ref_key_of
    seen = isinstance(env._area, (SeenArea, RealtimeArea))
    area_def = s.get("area_def", "window")
    key = ref_key_of(s)
    a_b = score_frames(terr, e.h, scen.track, scen.save_dt_s, X_WINDOW_REF if seen else x, [log], cfg)
    a_bel = env._area.A if seen else a_b["A"]
    aw = float(s.get("a_weight", 1.0))                        # score a_weight x A + w x POS (a_weight 0: POS only, user 2026-11-13)
    belief = dict(pos=float(info["POS"]), A=float(a_bel), total=float(aw*a_bel + w*info["POS"]), A_window=float(a_b["A"]))
    # the same actions on the hold-out particles (as sarops_ref)
    if progress is not None:
        progress("별도 particle 채점", 60.0)
    pos_h, a_h, a_win, _ = holdout_replay(s, terr, case, hold, [acts])
    holdout = dict(pos=pos_h, A=float(a_h), total=float(aw*a_h + w*pos_h), A_window=float(a_win), area_def=area_def, ref_key=key)
    try:
        ref = sarops_refs(split, key).get((stage, k)) if launch_s is None and built is None else None
    except FileNotFoundError:                                  # SAROPS references of this split not made yet
        ref = None
    er = np.array(end_rec) if end_rec else np.zeros((0, 4))
    end = dict(pos_last15=float(info["POS"]) - (float(np.interp(er[-1, 0] - 900.0, er[:, 0], er[:, 1])) if len(er) else 0.0),
               far_min=float((er[:, 2] < 0.05).sum()), edge_min=float((er[:, 3] < 1000.0).sum())) if len(er) else {}
    # end-of-flight check (user 2026-11-12): POS gained in the last 15 min, minutes with < 5 % of the remaining belief within 10 km,
    # minutes within 1 km of the map edge
    out = dict(split=split, stage=stage, k=k, step=step, belief=belief, holdout=holdout, steps=len(acts),
               sarops=ref["holdout"] if ref else None, sec=time.perf_counter() - t_start, end=end)
    if save_web or web:
        dt, n = scen.save_dt_s, e.belief.n
        T = min(int(np.ceil(tt[-1]/dt)) + 1, len(scen.track))
        rt, rw = np.array(rec_t), np.array(rec_w)
        W = np.full((T, n), 255, np.uint8)
        for j in range(T):
            i = np.searchsorted(rt, j*dt, side="right") - 1
            if i >= 0:
                W[j] = rw[i]
        g = a_b.pop("grid")
        if isinstance(env._area, RealtimeArea):
            area = realtime_payload(env._area, True, x, w, belief["pos"], a_b["A"])
        elif seen:
            area = seen_payload(env._area, True, x, w, belief["pos"], a_b["A"])
        else:
            area = {**a_b, "def": "window", "on": True, "x": x, "w": w, "total": belief["total"], "nx": g["nx"], "ny": g["ny"], "cell": g["cell"],
                    "x0": g["x0"], "y0": g["y0"], "bits": b64(g["bits"].astype("<u2")), "first": b64(g["first"])}
        uav = dict(rects=[], tasks=[], tt=b64(np.asarray(tt, np.float32)), xy=b64(np.asarray(xy, np.float32)), n_path=len(tt),
                   log_t=b64(log[:, 0].astype(np.float32)), log_xyyaw=b64(log[:, 1:4].astype(np.float32)),
                   quad=b64(footprints(log, cfg).astype(np.float32)), target=b64(log[:, 14:16].astype(np.float32)), n_log=len(log))
        opts = case.options(scen)
        start = opts["start_pose"][:2] if opts.get("stage", 3) in (1, 2) else scen.start
        opt = dict(T=T, N=int(n), dt_s=float(dt), weight=b64(W), uavs=[uav], start=[float(v) for v in start], lkp=[float(v) for v in scen.lkp],
                   pos_t=np.round(np.asarray(pos_t), 1).tolist(), pos=np.round(np.asarray(pos), 4).tolist(), final=belief["pos"],
                   escaped=float(e.belief.escaped), no_clear=0, area=area, sec=round(out["sec"], 1), settings=s)
        view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
        rid = f"eval-{split}-s{stage}-k{k}-rl-{step}"
        from rl_search.rl_eval_set import case_vc
        vc = built[4] if built is not None and len(built) > 4 else case_vc(case)    # the case's own target model (diverse cases)
        name = f"RL {step:,}" if isinstance(step, int) else f"RL {step}"
        if web:                                               # the web viewer's job queue saves the run (vehicle_web.Jobs)
            out["data"] = dict(opt=opt, res=particles_payload(vres, vc, case.heading, start, 0.0))
            out["meta_extra"] = dict(method=name, holdout=holdout)
            return out
        meta = vw.run_meta(opt, rid, f"{case.label} · {name}", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split=split,
                           stage=stage, case=k, method=name, holdout=holdout, case_info=case_dict(case))
        vw.save_run(rid, dict(opt=opt, res=particles_payload(vres, vc, case.heading, start, 0.0)), meta)
        out["web_run"] = rid
    return out


def eval_job(args):
    """Pool entry: (model_path, split, stage, k, step[, save_web, settings, launch_s])."""
    import warnings
    warnings.filterwarnings("ignore")
    return evaluate_case(*args)
