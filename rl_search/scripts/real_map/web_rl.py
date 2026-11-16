"""RL policy flights from the web viewer (vehicle_web, user 2026-11-14: check RL results on the web like SAROPS).

A trained checkpoint flies the scenario on the page (Amboy only, where the models were trained), with the env settings of its training
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
RL_REGION = "amboy"


def run_dir(model: Path) -> Path:
    """Training run folder of a checkpoint (<run>/ckpt/model_<step>.zip or <run>/best_stage3.zip)."""
    return model.parent.parent if model.parent.name == "ckpt" else model.parent


def rl_models() -> list[dict]:
    """Checkpoints of the training runs that saved their settings (config.json), newest run first, then highest step first."""
    out = []
    runs = [d for d in RL_DIR.iterdir() if d.is_dir() and (d/"config.json").exists()] if RL_DIR.exists() else []
    for d in sorted(runs, key=lambda p: (p/"config.json").stat().st_mtime, reverse=True):
        for z in sorted((d/"ckpt").glob("model_*.zip"), key=lambda p: int(p.stem.split("_")[1]), reverse=True):
            step = int(z.stem.split("_")[1])
            out.append(dict(path=str(z), run=d.name, step=step, label=f"{d.name} · {step:,} 스텝"))
        if (d/"best_stage3.zip").exists():
            out.append(dict(path=str(d/"best_stage3.zip"), run=d.name, step=None, label=f"{d.name} · 검증 최고"))
    return out


def model_label(model: Path) -> str:
    d = run_dir(model)
    return f"{d.name} {int(model.stem.split('_')[1]):,}" if model.stem.startswith("model_") else f"{d.name} 검증 최고"


def _run(p: dict, q) -> dict:
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
    st = preload(RL_REGION)
    vc = VehicleConfig(**p["config"])
    vc = replace(vc, duration_s=max(float(vc.duration_s), launch_s + 3600.0 + 600.0))     # the particles must outlast the flight
    vs = VehicleScenarios(st["sampler"], vc, region=RL_REGION, net=st["net"])
    seed, heading = int(view["seed"]), float(view["heading"])
    scen, vres = vs.make_at(seed, view["lkp"], heading, view.get("start"), with_res=True)
    hold = vs.make_at(seed + HOLDOUT, np.asarray(scen.lkp, float), heading, start=np.asarray(scen.start, float)[:2])
    hold = replace(hold, start_heading_deg=scen.start_heading_deg)     # the replay takes off exactly as the flight did
    case = Case("web", 3, 0, seed, [float(v) for v in scen.lkp], heading, 0.0,
                [float(scen.start[0]), float(scen.start[1]), float(np.radians(scen.start_heading_deg))], {"web": True}, t_launch_s=launch_s)
    settings = settings_of(json.loads((run_dir(model)/"config.json").read_text()))
    out = evaluate_case(str(model), "web", 3, 0, model_label(model), save_web=False, settings=settings, built=(case, scen, vres, hold, vc),
                        web=True, progress=lambda stage, t: q.put(dict(kind="progress", stage=stage, t_min=round(float(t), 1), plans=0)))
    meta = out["meta_extra"] | dict(model=str(model), launch_min=launch_s/60.0, case_info=dict(t_launch_s=launch_s))
    return dict(opt=out["data"]["opt"], res=out["data"]["res"], meta_extra=meta)


def worker(params: dict, q) -> None:
    try:
        t0 = time.perf_counter()
        data = _run(params, q)
        data["opt"]["sec"] = round(time.perf_counter() - t0, 1)
        q.put(dict(kind="result", data=data))
    except Exception:
        q.put(dict(kind="error", msg=traceback.format_exc(limit=4)))
