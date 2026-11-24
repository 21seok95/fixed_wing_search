"""RL policy flights from the web viewer (vehicle_web, user 2026-11-14: check RL results on the web like SAROPS).

A trained checkpoint flies the scenario on the page (any built map since 2026-11-17: on maps it was not trained on, such as Chungju,
the flight checks how it generalises), with the env settings of its training
run (config.json) and scored like the validation flights (rl_search.rl_real_eval): one deterministic flight on the page's particles,
then the same actions on hold-out particles (seed + rl_eval_set.HOLDOUT). The launch can be delayed (minutes after the report), as in
training (0-40 min). Each flight runs in its own process from the viewer's job queue and is saved as a web run.
"""
from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

RL_DIR = Path("rl_search/data/rl_real")
RL_REGION = "amboy"                 # the map the first models were trained on (the default when the page sends none)


def run_dir(model: Path) -> Path:
    """Training run folder of a checkpoint (<run>/ckpt/model_<step>.zip or <run>/best_stage3.zip)."""
    return model.parent.parent if model.parent.name == "ckpt" else model.parent


def model_kind(config: dict) -> tuple[str, list[int]]:
    """(kind, team sizes trained with) of a training run's config: "single" = a single-UAV run (R15 and before: its own inputs only),
    "solo" = one-UAV worlds of the multi-UAV env (R16: the no-teammate inputs), "team" = a team model (R17: teammate inputs live)."""
    teams = config.get("teams")
    if not teams:
        return "single", [1]
    sizes = sorted({int(k) for k in teams})
    return ("solo", [1]) if sizes == [1] else ("team", sizes)


def rl_models() -> list[dict]:
    """Checkpoints of the training runs that saved their settings (config.json), newest run first, then highest step first; with the
    team sizes each was trained with (``teams``, the page recommends the most common one) and its kind (model_kind)."""
    out = []
    runs = [d for d in RL_DIR.iterdir() if d.is_dir() and (d/"config.json").exists()] if RL_DIR.exists() else []
    for d in sorted(runs, key=lambda p: (p/"config.json").stat().st_mtime, reverse=True):
        try:
            c = json.loads((d/"config.json").read_text())
        except (OSError, ValueError):
            c = {}
        kind, sizes = model_kind(c)
        teams = c.get("teams") or [1]
        rec = max(sizes, key=lambda k: (list(teams).count(k), k))              # the size most of its worlds had
        info = dict(run=d.name, kind=kind, teams=sizes, recommend=rec)
        for z in sorted((d/"ckpt").glob("model_*.zip"), key=lambda p: int(p.stem.split("_")[1]), reverse=True):
            step = int(z.stem.split("_")[1])
            out.append(dict(path=str(z), step=step, label=f"{d.name} · {step:,} 스텝", **info))
        if (d/"best_stage3.zip").exists():
            out.append(dict(path=str(d/"best_stage3.zip"), step=None, label=f"{d.name} · 검증 최고", **info))
    return out


def model_label(model: Path) -> str:
    d = run_dir(model)
    return f"{d.name} {int(model.stem.split('_')[1]):,}" if model.stem.startswith("model_") else f"{d.name} 검증 최고"


def _run(p: dict, q) -> dict:
    if int(p.get("n_uav", 1) or 1) > 1:
        return _run_team(p, q)
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import HOLDOUT, Case
    from rl_search.rl_real import preload
    from rl_search.rl_real_eval import evaluate_case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    from rl_search.vehicle import VehicleConfig
    view, model = p["view"], Path(p["model"])
    launch_s = 60.0*float(p.get("launch_min", 0.0) or 0.0)
    q.put(dict(kind="progress", stage="particle 계산", t_min=0.0, plans=0))
    region = view.get("region") or RL_REGION
    st = preload(region)
    vc = VehicleConfig(**p["config"])
    vc = replace(vc, duration_s=max(float(vc.duration_s), launch_s + 3600.0 + 600.0))     # the particles must outlast the flight
    vs = VehicleScenarios(st["sampler"], vc, region=region, net=st["net"])
    seed, heading = int(view["seed"]), float(view["heading"])
    scen, vres = vs.make_at(seed, view["lkp"], heading, view.get("start"), with_res=True)
    hold = vs.make_at(seed + HOLDOUT, np.asarray(scen.lkp, float), heading, start=np.asarray(scen.start, float)[:2])
    hold = replace(hold, start_heading_deg=scen.start_heading_deg)     # the replay takes off exactly as the flight did
    case = Case("web", 3, 0, seed, [float(v) for v in scen.lkp], heading, 0.0,
                [float(scen.start[0]), float(scen.start[1]), float(np.radians(scen.start_heading_deg))], {"web": True}, t_launch_s=launch_s,
                region=region)
    settings = settings_of(json.loads((run_dir(model)/"config.json").read_text()))
    out = evaluate_case(str(model), "web", 3, 0, model_label(model), save_web=False, settings=settings, built=(case, scen, vres, hold, vc),
                        web=True, progress=lambda stage, t: q.put(dict(kind="progress", stage=stage, t_min=round(float(t), 1), plans=0)))
    meta = out["meta_extra"] | dict(model=str(model), launch_min=launch_s/60.0, case_info=dict(t_launch_s=launch_s))
    return dict(opt=out["data"]["opt"], res=out["data"]["res"], meta_extra=meta)


def _built(p: dict, q) -> tuple:
    """The page's scenario as a case: (case, scenario, simulation result, hold-out scenario, VehicleConfig, launch delay s)."""
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import HOLDOUT, Case
    from rl_search.rl_real import preload
    from rl_search.vehicle import VehicleConfig
    view = p["view"]
    launch_s = 60.0*float(p.get("launch_min", 0.0) or 0.0)
    q.put(dict(kind="progress", stage="particle 계산", t_min=0.0, plans=0))
    region = view.get("region") or RL_REGION
    st = preload(region)
    vc = VehicleConfig(**p["config"])
    vc = replace(vc, duration_s=max(float(vc.duration_s), launch_s + 3600.0 + 600.0))     # the particles must outlast the flight
    vs = VehicleScenarios(st["sampler"], vc, region=region, net=st["net"])
    seed, heading = int(view["seed"]), float(view["heading"])
    scen, vres = vs.make_at(seed, view["lkp"], heading, view.get("start"), with_res=True)
    hold = vs.make_at(seed + HOLDOUT, np.asarray(scen.lkp, float), heading, start=np.asarray(scen.start, float)[:2])
    hold = replace(hold, start_heading_deg=scen.start_heading_deg)     # the replay takes off exactly as the flight did
    case = Case("web", 3, 0, seed, [float(v) for v in scen.lkp], heading, 0.0,
                [float(scen.start[0]), float(scen.start[1]), float(np.radians(scen.start_heading_deg))], {"web": True}, t_launch_s=launch_s,
                region=region)
    return case, scen, vres, hold, vc, launch_s


def team_gaps(p: dict, k: int) -> list[float]:
    """Launch gaps (s) between consecutive UAVs: the page's (one value for all, or one per gap), else 5-10 s drawn from the seed (as the
    m2 / m3 training draws them, rl_multi.DESIGNS gap_range)."""
    g = p.get("gaps")
    if g:
        g = [float(x) for x in g]
        return (g*(k - 1))[:k - 1] if len(g) == 1 else (g + [g[-1]]*(k - 1))[:k - 1]
    rng = np.random.default_rng(int(p["view"]["seed"]) + 2027)
    return [round(float(x), 2) for x in rng.uniform(5.0, 10.0, k - 1)]


def _run_team(p: dict, q) -> dict:
    """A team of ``n_uav`` UAVs flies the page's scenario (user 2026-11-24): every UAV with the model, launching one after another from the
    start point (gaps ``team_gaps``), the multi-UAV env's safety filter on (rl_multi.MultiRealEnv, scripts/multi_design/ma_eval.fly).
    A team model keeps its run's design and decision order; a single-UAV model (R15) and a one-UAV model of the multi env (R16) fly every
    UAV on its own inputs under the current design m3 (user 11-24: "충돌 규칙은 그대로"). Scored like the validation team flights: the
    planner particles, then the same applied commands on the hold-out particles."""
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.scripts.multi_design.ma_eval import fly, web_payload
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    model_path, k = Path(p["model"]), int(p["n_uav"])
    c = json.loads((run_dir(model_path)/"config.json").read_text())
    kind, trained = model_kind(c)
    design = c.get("ma_design", "m3") if kind == "team" else "m3"
    seq = bool(c.get("sequential", False)) and kind == "team"
    s = settings_of(c)
    case, scen, vres, hold, vc, launch_s = _built(p, q)
    gaps = team_gaps(p, k)
    seed = int(p["view"]["seed"])
    model = PPO.load(str(model_path), device="cpu")
    t0 = time.perf_counter()
    prog = lambda m: q.put(dict(kind="progress", stage=f"RL {k}대 비행", t_min=round(float(m), 1), plans=0))
    r = fly(k, 0, s, model=model, single=kind == "single", web=True, design=design, sequential=seq, solo=kind == "solo",
            built=(case, scen, vres, hold, vc), gaps=gaps, seed=seed, progress=prog)
    applied = r.pop("applied")
    q.put(dict(kind="progress", stage="채점용 particle로 다시 비행", t_min=0.0, plans=0))
    h = fly(k, 0, s, actions=applied, design=design, built=(case, scen, vres, hold, vc), gaps=gaps, seed=seed)
    opt, res, _, holdout = web_payload(s, r, h["found"], time.perf_counter() - t0)
    each = " (각자 비행)" if kind != "team" else ""
    off = kind == "team" and k not in trained                   # a one-UAV model in a team flies each UAV alone, as intended
    name = f"RL {model_label(model_path)} · {k}대{each}" + (" · 학습 대수와 다름" if off else "")
    share = np.array(r["share"], float)
    num = lambda x: None if x is None or not np.isfinite(float(x)) else round(float(x), 4)      # meta.json must stay valid JSON (no NaN)
    meta = dict(method=name, holdout=holdout, model=str(model_path), launch_min=launch_s/60.0, case_info=dict(t_launch_s=launch_s),
                team=k, trained_teams=trained, team_off=off, design=design, sequential=seq, gaps=gaps, launch_t=[num(t) for t in r["launch_t"]],
                close1km=num(r["close1km"]), dmin=num(r["dmin"]), cuts=int(r["cuts"]), viol=num(r["viol"]),
                shares=(share/max(share.sum(), 1e-12)).round(3).tolist(), circling=num(r["circling"]), act_mean=num(r["act_mean"]))
    return dict(opt=opt, res=res, meta_extra=meta)


def worker(params: dict, q) -> None:
    try:
        t0 = time.perf_counter()
        data = _run(params, q)
        data["opt"]["sec"] = round(time.perf_counter() - t0, 1)
        q.put(dict(kind="result", data=data))
    except Exception:
        q.put(dict(kind="error", msg=traceback.format_exc(limit=4)))
