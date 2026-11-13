"""SAROPS references on the RL validation or test cases (rl_search.rl_eval_set, docs/rl_mdp_design.md).

One flight per case on the planner particles with the web defaults (A + POS, w 1, 1 UAV) and the gimbal fixed forward (pre-look 0 s),
scored on the hold-out particles by replaying the same actions. Each flight is saved as a web run (run list filter "검증 n단계") and the
scores go to rl_search/data/real/amboy/rl_eval/sarops_ref_<split>.json (window A, x 70, the default). With ``--area-def seen`` (the
seen-particle A of 2026-11-11, x 80) they go to sarops_ref_<split>_seen.json and the web runs are eval-<split>-s<stage>-k<k>-sarops-seen.

    python -m rl_search.scripts.rl_eval.sarops_ref --split val --workers 5
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import get_context

import numpy as np

OVERRIDE = dict(cap="A", score="area", n_uav=1, prelook_s=0, area_def="window")   # the seen A (x 80, POD 0.5) is an option since 2026-11-11
SEEN = dict(area_def="seen", area_x=80.0)
REALTIME = dict(area_def="realtime", area_x=70.0)
DEFS = dict(window=dict(area_def="window"), seen=SEEN, realtime=REALTIME, realtime_q=dict(area_def="realtime_q", area_x=70.0),
            realtime_t=dict(area_def="realtime_t", area_x=70.0))


def settings() -> dict:
    from rl_search.scripts.real_map.web_optimize import FIELDS
    return {f["name"]: f["value"] for f in FIELDS} | OVERRIDE


def ref_key(area_def: str = "window", blur_tau: float = 0.0, roll_max: float | None = None, pos_w: float = 1.0, blur_dead: float = 0.0,
            blur_low: float = 0.0, sigma: float | None = None, img_min: float = 0.0, sigma_off: float = 0.0, a_weight: float = 1.0) -> str:
    """Name of a reference set: the A definition (with its roll limit for realtime_t), d<tau> with the exponential turn-rate factor (D2),
    then z<roll> when that factor is 1 up to the turn rate of this roll (D3) and l<tau_low> when it also falls below it (D4),
    s<sigma> for a real-time A region blurred with sigma != 150 m, o<sigma> for the off-road particles' own sigma, i<min> for the
    image-position gate, w<w> when the score is A + w POS with w != 1; "window" alone = amboy_v1's.
    Examples: realtime_t5_d3z5_w1.5 (R3), realtime_t5_d3z5l15_s75_w1.5 (R4), realtime_t5_d3z5l15_s75_o150_i0.75_w1.5 (next bundle)."""
    name = area_def + (f"{roll_max:g}" if area_def == "realtime_t" else "")
    d = (f"_d{blur_tau:g}" + (f"z{blur_dead:g}" if blur_dead else "") + (f"l{blur_low:g}" if blur_dead and blur_low else "")) if blur_tau else ""
    sg = f"_s{sigma:g}" if sigma and float(sigma) != 150.0 and area_def.startswith("realtime") else ""
    sg += f"_o{float(sigma_off):g}" if sigma_off and float(sigma_off) != float(sigma or 150.0) and area_def.startswith("realtime") else ""
    im = f"_i{float(img_min):g}" if img_min and area_def.startswith("realtime") else ""     # image-position gate of a searched cell
    aw = f"_a{float(a_weight):g}" if float(a_weight) != 1.0 else ""            # A weight (a0: plan and score on POS only, user 2026-11-13)
    return name + d + sg + im + aw + (f"_w{pos_w:g}" if pos_w != 1 else "")


def ref_key_of(s: dict) -> str:
    return ref_key(s.get("area_def", "window"), float(s.get("blur_tau", 0.0)), float(s.get("area_roll_max", 20.0)), float(s.get("area_w", 1.0)),
                   float(s.get("blur_dead_roll", 0.0) or 0.0), float(s.get("blur_tau_low", 0.0) or 0.0), s.get("area_sigma"),
                   float(s.get("area_img_min", 0.0) or 0.0), float(s.get("area_sigma_off", 0.0) or 0.0), float(s.get("a_weight", 1.0)))


def ref_path(split: str, key: str = "window"):
    from rl_search.rl_eval_set import EVAL_DIR
    return EVAL_DIR/(f"sarops_ref_{split}.json" if key == "window" else f"sarops_ref_{split}_{key}.json")


def holdout_replay(s: dict, terr, case, hold, actions):
    """The same actions on the hold-out particles: (POS, A, A_window x 70, replay run). With the seen A (``s["area_def"] == "seen"``) the
    replay env carries a ``SeenArea`` of the hold-out particles; the window A comes from the camera frames either way."""
    from rl_search.area_score import RealtimeArea, SeenArea, score_frames
    from rl_search.multi import run_team_launch
    from rl_search.rl_eval_set import REGION
    from rl_search.scripts.real_map.web_optimize import X_WINDOW_REF, team_env
    adef = s.get("area_def", "window")
    seen = adef != "window"
    env = team_env(s, REGION, terr, case.seed, area=seen)
    if adef == "seen":
        env.area = SeenArea(hold.track, hold.exit_step, hold.save_dt_s, float(s["area_x"]), float(s["seen_pod"]))
    elif adef in ("realtime", "realtime_q", "realtime_t"):
        from rl_search.rl_eval_set import flight_window
        from rl_search.scripts.real_map.web_optimize import area_sigma, realtime_kw
        env.area = RealtimeArea(terr, area_sigma(s, env.h), hold.track, hold.exit_step, hold.save_dt_s, float(s["area_x"]), quality=adef == "realtime_q",
                                window=flight_window(case.options(hold)),
                                roll_max_deg=float(s.get("area_roll_max", 20.0)) if adef == "realtime_t" else None, **realtime_kw(s, REGION))
        env.area.img_min = float(s.get("area_img_min", 0.0) or 0.0)
    if seen:
        env.pos_w, env.area_w = 1.0, 1.0
    env.sample_log = []
    h = run_team_launch([env], None, case.options(hold), 60.0, actions=actions)
    log = np.array(env.sample_log)
    aw = score_frames(terr, env.h, hold.track, hold.save_dt_s, X_WINDOW_REF if seen else float(s["area_x"]), [log], env.detector.cfg)
    pos = float(h["pos"][-1])
    return pos, (env.area.A if seen else aw["A"]), aw["A"], h


def run_job(job):
    split, stage, k, area_def = job[:4]
    blur_tau = job[4] if len(job) > 4 else 0.0
    roll_max = job[5] if len(job) > 5 else None
    pos_w = job[6] if len(job) > 6 else None
    blur_dead = job[7] if len(job) > 7 else 0.0
    blur_low = job[8] if len(job) > 8 else 0.0
    sigma = job[9] if len(job) > 9 else None
    img_min = job[10] if len(job) > 10 else 0.0
    sigma_off = job[11] if len(job) > 11 else 0.0
    a_weight = job[12] if len(job) > 12 else 1.0
    warnings.filterwarnings("ignore")
    from dataclasses import asdict
    from rl_search.area_score import score_frames
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import VehicleScenarios, chungju_preset, terrain_path
    from rl_search.rl_eval_set import REGION, case_dict, case_vc, make_case, vehicle_config
    from rl_search.scenario import ScenarioSampler
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import run_case, team_env
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter()
    terr = load_terrain(terrain_path(REGION))
    vc = vehicle_config()
    vs = VehicleScenarios(ScenarioSampler(terr, chungju_preset().scenario), vc, region=REGION)
    case, scen, vres, hold = make_case(vs, split, stage, k)
    vc = case_vc(case)                                                # the case's own target model (diverse cases)
    s = settings() | DEFS[area_def] | ({"blur_tau": blur_tau} if blur_tau else {}) | ({"area_roll_max": roll_max} if roll_max else {}) \
        | ({"area_w": pos_w} if pos_w else {}) | ({"blur_dead_roll": blur_dead} if blur_dead else {}) \
        | ({"blur_tau_low": blur_low} if blur_low else {}) | ({"area_sigma": sigma} if sigma and float(sigma) != 150.0 else {}) \
        | ({"area_img_min": img_min} if img_min else {}) | ({"area_sigma_off": sigma_off} if sigma_off else {}) \
        | ({"a_weight": float(a_weight)} if float(a_weight) != 1.0 else {}) | ({"score": "pos"} if float(a_weight) == 0.0 else {})
    view = dict(region=REGION, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
    out = run_case(REGION, scen, vres, vc, view, s, options=case.options(scen), terr=terr)
    pos_h, a_h, a_win, h = holdout_replay(s, terr, case, hold, out["actions"])          # the same actions on the hold-out particles
    drift = float(np.hypot(*(np.asarray(h["xy"][0][-1]) - np.asarray(out["paths"][0][-1]))))
    holdout = dict(pos=pos_h, A=float(a_h), total=float(float(s.get("a_weight", 1.0))*a_h + float(s["area_w"])*pos_h), A_window=float(a_win), area_def=s.get("area_def", "window"),
                   ref_key=ref_key_of(s))
    o = out["opt"]
    key = ref_key_of(s)
    rid = f"eval-{split}-s{stage}-k{k}-sarops" + ("" if key == "window" else f"-{key}")
    method = {"window": "SAROPS", "seen": "SAROPS (본 particle A)", "realtime": "SAROPS (실시간 A)", "realtime_q": "SAROPS (실시간 A + 화면 품질)",
              "realtime_t": "SAROPS (실시간 A, 롤 기준)"}[area_def]
    if area_def == "realtime_t":
        method += f" {float(s['area_roll_max']):g}°"
    if blur_tau:
        method += (f" · 흔들림 D4 (롤 {blur_dead:g}°까지 ω₀ {blur_low:g}, 그 위 τ {blur_tau:g})" if blur_dead and blur_low else
                   f" · 흔들림 ω₀ {blur_tau:g}" + (f" (롤 {blur_dead:g}°까지 1)" if blur_dead else ""))
    if sigma and float(sigma) != 150.0:
        method += f" · σ {float(sigma):g} m"
    if sigma_off:
        method += f" · 도로 밖 σ {float(sigma_off):g} m"
    if float(a_weight) != 1.0:
        method += " · POS만 (A 끔)" if float(a_weight) == 0.0 else f" · A 가중치 {float(a_weight):g}"
    if img_min:
        method += f" · 화면 위치 {float(img_min):g} 이상"
    if float(s.get("area_w", 1.0)) != 1:
        method += f" · A + {float(s['area_w']):g} POS"
    meta = vw.run_meta(o, rid, f"{case.label} · {method}", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split=split, stage=stage,
                       case=k, method=method, holdout=holdout, case_info=case_dict(case))
    vw.save_run(rid, dict(opt=o, res=out["res"]), meta)
    return dict(id=rid, case=case_dict(case), belief=dict(pos=o["final"], A=o["area"]["A"], total=o["area"]["total"]), holdout=holdout,
                plans=meta["plans"], drift_m=drift, sec=round(time.perf_counter() - t0, 1))


def main():
    from rl_search.rl_eval_set import EVAL_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=("val", "test", "dval", "dtest", "mval", "mtest"))
    ap.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--cases", type=int, default=None, help="Cases per stage (default 5 for val, 30 for test)")
    ap.add_argument("--workers", type=int, default=5, help="Processes at once (each about 1.2 GB)")
    ap.add_argument("--blur-tau", type=float, default=0.0, help="turn-rate factor: 0 = linear (D1), > 0 = exp(-w/tau) (D2)")
    ap.add_argument("--area-roll-max", type=float, default=20.0, help="realtime_t: roll limit (deg) for a searched cell")
    ap.add_argument("--pos-w", type=float, default=1.0, help="score A + w POS")
    ap.add_argument("--blur-dead-roll", type=float, default=0.0, help="with --blur-tau: no turn-rate loss up to this roll's turn rate (D3)")
    ap.add_argument("--blur-tau-low", type=float, default=0.0, help="with --blur-dead-roll: exp(-w/value) below that rate too (D4 = D2 x D3)")
    ap.add_argument("--area-sigma", type=float, default=None, help="blur width (m) of the real-time A region (default 150)")
    ap.add_argument("--area-img-min", type=float, default=0.0, help="a cell is searched only where the image-position factor is at least this")
    ap.add_argument("--area-sigma-off", type=float, default=0.0, help="blur width (m) of the off-road particles in the real-time A region (0 = same)")
    ap.add_argument("--a-weight", type=float, default=1.0, help="score a_weight x A + w x POS; 0 = plan and score on POS only (A still reported)")
    ap.add_argument("--tag", default=None, help="suffix of the reference file and web runs (default: the A definition, plus d<tau> with --blur-tau)")
    ap.add_argument("--area-def", default="window", choices=("window", "realtime", "realtime_q", "realtime_t", "seen"),
                    help="A of the plan and the score (realtime: the window A every 10 s; seen: the particle option, x 80)")
    args = ap.parse_args()
    from rl_search.rl_eval_set import MULTI_N
    n = args.cases or MULTI_N.get(args.split) or (5 if args.split in ("val", "dval") else 30)
    jobs = [(args.split, st, k, args.area_def, args.blur_tau, args.area_roll_max, args.pos_w, args.blur_dead_roll, args.blur_tau_low, args.area_sigma,
             args.area_img_min, args.area_sigma_off, args.a_weight) for st in args.stages for k in range(n)]
    key = ref_key(args.area_def, args.blur_tau, args.area_roll_max, args.pos_w, args.blur_dead_roll, args.blur_tau_low, args.area_sigma,
                  args.area_img_min, args.area_sigma_off, args.a_weight)
    out = ref_path(args.split, key)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {r["id"] for r in rows}
    sfx = "" if key == "window" else f"-{key}"
    rows = [r for r in rows if r["id"] not in {f"eval-{args.split}-s{j[1]}-k{j[2]}-sarops{sfx}" for j in jobs}]
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
