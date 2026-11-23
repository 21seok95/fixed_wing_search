"""Multi-UAV evaluation (docs/multi_agent_design.md; M1, user 2026-11-20): a model flies the three Jirisan validation cases (v2j, launch
10-40 km, never trained on) with teams of 1-4 UAVs from one base (launch gaps fixed per case, as ma_step1), deterministic actions, safety filter
on. Scored twice: on the planner particles the policy saw, and on the case's hold-out particles by flying the same applied commands (after
the filter) again with every UAV (rl_multi.MultiRealEnv, filter off, no observations): the multi-UAV form of the single-UAV hold-out score.

Models: a multi-UAV checkpoint (teammate inputs), or R15 flown by every UAV on its own inputs only (``--single``: the baseline before
multi-UAV training). Per model and team size: team POS (planner / hold-out), share of steps the nearest pair is within 1 km, minimum distance,
filter cuts, violations, each UAV's own share (to spot an idle UAV). Every flight is also saved as a web run (all UAVs' paths and
camera footprints, team survival as particle weights), listed with the case's validation runs: ``eval-v2j-s3-k<case>-ma<k>-<label>``.

    python -m rl_search.scripts.multi_design.ma_eval run --model rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip --label r15_6.8M --single
    python -m rl_search.scripts.multi_design.ma_eval run --model rl_search/data/rl_real/m1_multi/ckpt/model_400000.zip --label m1_0.4M
    python -m rl_search.scripts.multi_design.ma_eval figure --labels r15_6.8M m1_0.4M ...
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"ma_eval"
RUN_CFG = "rl_search/data/rl_real/m1_multi/config.json"
R15_CFG = "rl_search/data/rl_real/r15_chungju/config.json"
TEAMS = (1, 2, 3, 4)
CASES = (0, 1, 2)


def _gaps(case_k: int, k: int, design: str = "m1") -> list:
    """Launch gaps of a case, fixed per case: m1 drawn from 5 / 10 / 15 s, m2 uniform in 5-10 s (rl_multi.DESIGNS)."""
    rng = np.random.default_rng(1000 + case_k)
    if design in ("m2", "m3"):
        return [float(x) for x in rng.uniform(5.0, 10.0, 3)][:k - 1]
    return [float(x) for x in rng.choice((5.0, 10.0, 15.0), 3)][:k - 1]


def run_config(model_path: str, single: bool) -> tuple:
    """(config of the model's run, its multi-UAV design): the run folder's config.json (ckpt/..), else M1's (R15's with ``single``);
    the design is the config's ``ma_design`` (runs before 2026-11-23: m1)."""
    c_path = Path(model_path).parent.parent/"config.json"
    if not c_path.exists():
        c_path = Path(R15_CFG if single else RUN_CFG)
    c = json.loads(c_path.read_text())
    return c, c.get("ma_design", "m1")


def _surv8(W, W0) -> np.ndarray:
    """Team survival of every particle since the launch (W / W at the launch) as 0-255, the web run's particle weights."""
    m = W0 > 0
    s = np.ones(len(W0))
    s[m] = W[m]/W0[m]
    return np.round(np.clip(s, 0.0, 1.0)*255).astype(np.uint8)


def fly(k: int, case_k: int, s: dict, model=None, single: bool = False, actions=None, web: bool = False, design: str = "m1",
        sequential: bool = False, solo: bool = False) -> dict:
    """One team flight on case ``case_k``: with ``model`` (planner particles, filter on) or replaying ``actions`` (hold-out particles, filter
    off, the commands as applied). ``web``: also keep what a web run needs (camera samples, team survival per step) under ``out["web"]``.
    ``sequential``: the model decides one UAV after another within each step (rl_multi sequential=True, user 11-23); a replay flies the
    applied commands together (the filter is off, so each path is the same; the belief after a step does not depend on the order)."""
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    case, scen, vres, hold = _case("v2j", 3, case_k)
    replay = actions is not None
    seq = bool(sequential) and not replay
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s), seed=case_k, safety=not replay, design=design, sequential=seq)
    env.return_obs = not replay
    env.solo_view = bool(solo)                          # (11-24) a one-UAV model of the multi env (R16) in a team: no-teammate inputs
    if web:
        for u in env.uavs:
            u.env.sample_log = []                         # every real 1 s sample with the camera axes (footprints, A)
    env.reset(options=case.options(hold if replay else scen), gaps=_gaps(case_k, k, design))
    W_init = env.W.copy()
    rec_t, rec_w = [env.T], [np.full(env.N, 255, np.uint8)]
    applied, n, act_hist = [], 0, []
    def act_one(i, o):                                    # sequential: the UAV whose turn it is
        a, _ = model.predict({key: v[None] for key, v in o.items()}, deterministic=True)
        return np.atleast_1d(float(np.ravel(a)[0]))

    while not env.done:
        if seq:
            _, _, info = env.seq_step(act_one)
        else:
            if replay:
                acts = actions[n] if n < len(actions) else [0.0]*k
            else:
                obs = [env.base_observation(i) if single else env.observation(i) for i in range(k)]
                batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
                acts, _ = model.predict(batch, deterministic=True)
                acts = np.ravel(acts).tolist()
            _, _, _, info = env.step([np.atleast_1d(a) for a in acts])
        applied.append([float(x) for x in info["applied"]])
        act_hist.append(np.asarray(info["active"], bool))
        n += 1
        if web:
            rec_t.append(env.T)
            rec_w.append(_surv8(env.W, W_init))
    lg = env.log
    d = np.array(lg["dist"], float)
    fin = d[np.isfinite(d)]
    share = np.array(lg["share"]).sum(0)
    # turning habit (11-23, M2 0.4M flew springs): the mean applied command and its share to the plus side over the active steps, and the
    # share of minutes in a circle (heading turned one way by >= 330 deg over 3 min, as rl_eval/circle_check) per UAV
    A, M = np.array(applied, float), np.array(act_hist, bool)
    circ = []
    for h in env.hist:
        h = np.array(h, float)
        if len(h) > 200:
            t1 = np.arange(h[0, 0], h[-1, 0], 1.0)
            hd = np.unwrap(np.arctan2(np.diff(np.interp(t1, h[:, 0], h[:, 2])), np.diff(np.interp(t1, h[:, 0], h[:, 1]))))
            mins = np.arange(0, len(hd) - 180, 60)
            circ.append(float(np.mean([abs(hd[m + 180] - hd[m]) >= np.radians(330) for m in mins])) if len(mins) else np.nan)
    turn = dict(act_mean=float(A[M].mean()) if M.any() else 0.0, act_plus=float((A[M] > 0).mean()) if M.any() else 0.0,
                circling=float(np.nanmean(circ)) if circ else float("nan"))
    out = dict(found=float(env.found), curve=[float(x) for x in lg["found"]], t=[float(x) for x in lg["t"]], applied=applied, **turn,
               close1km=float((fin < 1000).mean()) if len(fin) else 0.0, dmin=float(fin.min()) if len(fin) else float("nan"),
               cuts=int((np.array(lg["cut"]) > 0).sum()), viol=float(np.array(lg["viol"]).sum()/2), share=share.tolist(),
               hist=[np.array(h)[::5].tolist() for h in env.hist] if not replay else None, launch_t=env.launch_t.tolist())
    if web:
        out["web"] = dict(env=env, case=case, scen=scen, vres=vres, rec_t=rec_t, rec_w=rec_w, t0=float(rec_t[0]))
    return out


def save_web(label: str, k: int, case_k: int, s: dict, r: dict, holdout_pos: float, single: bool, sec: float) -> str:
    """Save a team flight as a web run (the viewer replays every UAV's path and camera footprints over the particles, team survival as
    the particle weights), listed with the case's validation runs ("v2j 3단계", case k): ``eval-v2j-s3-k<case>-ma<k>-<label>``."""
    from dataclasses import asdict
    from rl_search.area_score import score_frames
    from rl_search.rl_eval_set import case_dict, case_vc
    from rl_search.scripts.real_map import vehicle_web as vw
    from rl_search.scripts.real_map.web_optimize import b64, footprints, particles_payload
    from rl_search.scripts.rl_eval.sarops_ref import ref_key_of
    wd = r.pop("web")
    env, case, scen, vres = wd["env"], wd["case"], wd["scen"], wd["vres"]
    dt, n = scen.save_dt_s, env.N
    T = min(int(np.ceil(env.T/dt)) + 1, len(scen.track))
    rt = np.array(wd["rec_t"])
    W = np.full((T, n), 255, np.uint8)
    for j in range(T):
        i = np.searchsorted(rt, j*dt, side="right") - 1
        if i >= 0:
            W[j] = wd["rec_w"][i]
    uavs, logs = [], []
    for i, u in enumerate(env.uavs):
        e = u.env
        log = np.array(e.sample_log, float) if e.sample_log else np.zeros((0, 16))
        e.sample_log = None
        logs.append(log)
        h = np.array(env.hist[i], float)
        uavs.append(dict(rects=[], tasks=[], tt=b64(h[:, 0].astype(np.float32)), xy=b64(h[:, 1:3].astype(np.float32)), n_path=len(h),
                         log_t=b64(log[:, 0].astype(np.float32)), log_xyyaw=b64(log[:, 1:4].astype(np.float32)),
                         quad=b64(footprints(log, e.detector.cfg).astype(np.float32)), target=b64(log[:, 14:16].astype(np.float32)),
                         n_log=len(log)))
    u0 = env.uavs[0]
    x, w, aw = float(s["area_x"]), float(s["area_w"]), float(s.get("a_weight", 1.0))
    a = score_frames(u0.terr, u0.env.h, scen.track, dt, x, [lg for lg in logs if len(lg)], u0.env.detector.cfg)   # the 5 min window A (reference)
    g = a.pop("grid")
    final = float(env.found)
    area = {**a, "def": "window", "on": aw > 0, "x": x, "w": w, "total": aw*a["A"] + w*final, "nx": g["nx"], "ny": g["ny"], "cell": g["cell"],
            "x0": g["x0"], "y0": g["y0"], "bits": b64(g["bits"].astype("<u2")), "first": b64(g["first"])}
    start = scen.start
    pos_t = [wd["t0"]] + [float(t) for t in env.log["t"]]
    pos = [0.0] + [float(v) for v in env.log["found"]]
    opt = dict(T=T, N=int(n), dt_s=float(dt), weight=b64(W), uavs=uavs, start=[float(v) for v in start[:2]], lkp=[float(v) for v in scen.lkp],
               pos_t=np.round(pos_t, 1).tolist(), pos=np.round(pos, 4).tolist(), final=final, escaped=float(u0.env.belief.escaped), no_clear=0,
               area=area, sec=round(sec, 1), settings=s)
    vc = case_vc(case)
    name = f"RL {label}{' (각자 비행)' if single else ''} · {k}대"
    holdout = dict(pos=float(holdout_pos), A=None, total=float(w*holdout_pos), A_window=None, area_def=s.get("area_def", "window"),
                   ref_key=ref_key_of(s))
    view = dict(region=case.region, seed=case.seed, lkp=case.lkp, heading=case.heading, start=case.start[:2])
    rid = f"eval-v2j-s3-k{case_k}-ma{k}-{label.replace('.', 'p')}"
    meta = vw.run_meta(opt, rid, f"{case.label} · {name}", time.strftime("%m-%d %H:%M"), view, asdict(vc), group="eval", split="v2j", stage=3,
                       case=case_k, method=name, holdout=holdout, case_info=case_dict(case), team=k, launch_t=env.launch_t.tolist(),
                       close1km=r["close1km"], cuts=r["cuts"], viol=r["viol"])
    vw.save_run(rid, dict(opt=opt, res=particles_payload(vres, vc, case.heading, start, 0.0)), meta)
    return rid


def run(model_path: str, label: str, single: bool, web: bool = True, design: str | None = None, teams=TEAMS, holdout: bool = True,
        solo: bool = False):
    """``design``: the env's multi-UAV design (gaps, safety distances, teammate inputs); default the model run's (a single-UAV model: m1,
    as the 11-20 baseline; give m2 to compare with an m2 run). ``teams``: team sizes to fly; ``holdout`` False: no replay on the hold-out
    particles (the planner score is stored as both; 11-23: the two agreed within 0.01 in every evaluation so far)."""
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    c, run_design = run_config(model_path, single)
    design = design or ("m1" if single else run_design)
    seq = bool(c.get("sequential", False)) and not single              # a run trained with sequential decisions flies them (11-23)
    s = settings_of(c)
    print(f"{label}: design {design}{', sequential decisions' if seq else ''}", flush=True)
    model = PPO.load(model_path, device="cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for case_k in CASES:
        for k in teams:
            t1 = time.perf_counter()
            r = fly(k, case_k, s, model=model, single=single, web=web, design=design, sequential=seq, solo=solo)
            applied = r.pop("applied")
            h = fly(k, case_k, s, actions=applied, design=design) if holdout else dict(found=r["found"], curve=r["curve"])
            if web:
                save_web(label, k, case_k, s, r, h["found"], single, time.perf_counter() - t1)
            rows.append(dict(k=k, case=case_k, belief=r["found"], holdout=h["found"], holdout_replayed=bool(holdout), curve=r["curve"],
                             curve_h=h["curve"], t=r["t"],
                             act_mean=r["act_mean"], act_plus=r["act_plus"], circling=r["circling"],
                             close1km=r["close1km"], dmin=r["dmin"], cuts=r["cuts"], viol=r["viol"], share=r["share"], hist=r["hist"],
                             launch_t=r["launch_t"]))
            print(f"{label} case {case_k + 1} k={k}: team POS {r['found']:.3f} (hold-out {h['found']:.3f}), 1 km {r['close1km']*100:.0f}%, "
                  f"min {r['dmin']:.0f} m, cuts {r['cuts']}, violations {r['viol']:.0f}, circling {r['circling']*100:.0f}%, mean command {r['act_mean']:+.2f}, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
    (WORK/f"{label}.json").write_text(json.dumps(dict(label=label, model=model_path, single=single, design=design, sequential=seq, teams=list(teams),
                                                      holdout=bool(holdout), rows=rows)))


def figure(labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    data = [json.loads((WORK/f"{lb}.json").read_text()) for lb in labels]
    cols = ["#8a8f98", "#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad", "#c0392b", "#16a085", "#d4ac0d"]
    fig, axs = plt.subplots(2, 3, figsize=(21, 11))

    def per_k(d, key):
        return [np.mean([r[key] for r in d["rows"] if r["k"] == k]) for k in TEAMS]

    for ax, key, ylab, title in ((axs[0, 0], "holdout", "팀 POS (채점용 입자, 사례 3개 평균)", "(가) 대수별 팀 POS (채점용 입자)"),
                                 (axs[0, 1], "belief", "팀 POS (계획용 입자)", "(나) 대수별 팀 POS (계획용 입자)"),
                                 (axs[0, 2], "close1km", "가장 가까운 두 기체 1 km 안 (비행 시간 중)", "(다) 붙어 다니는 정도"),
                                 (axs[1, 0], "cuts", "필터가 자른 결정 (비행마다)", "(라) 안전 필터에 기댄 정도"),
                                 (axs[1, 1], "viol", "150 m 안 걸음 (0이어야 함)", "(마) 위반")):
        for d, c in zip(data, cols):
            ax.plot(TEAMS, per_k(d, key), "o-", color=c, label=d["label"])
        ax.set_xticks(TEAMS)
        ax.set_xlabel("대수")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 2]
    for j, (d, c) in enumerate(zip(data, cols)):
        sh = [np.array(r["share"])/max(sum(r["share"]), 1e-9) for r in d["rows"] if r["k"] == 4]
        m = np.mean(np.sort(np.array(sh), axis=1)[:, ::-1], axis=0)
        ax.bar(np.arange(4) + j*0.8/len(data), m, 0.8/len(data), color=c, label=d["label"])
    ax.set_xticks(np.arange(4) + 0.4 - 0.4/len(data))
    ax.set_xticklabels(["가장 많이 찾은 기체", "2위", "3위", "가장 적게"])
    ax.set_ylabel("자기 몫의 비율 (4대 팀)")
    ax.set_title("(바) 4대 팀에서 기체별 몫: 노는 기체가 있나", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("다수 기체 평가: 지리산 검증 3개 사례(학습에 안 쓴 지도), 1~4대, 같은 기지 순차 이륙, 결정은 무작위성 없이", fontsize=13)
    fig.savefig(OUT/"ma_eval.png", dpi=105, bbox_inches="tight")
    L = ["# 다수 기체 평가 (지리산 v2j 3사례)", "", "| 모델 | 대수 | 팀 POS 채점용 | 계획용 | 1 km 안 | 최소 거리 (m) | 잘린 결정 | 위반 |", "|---|---|---|---|---|---|---|---|"]
    for d in data:
        for k in TEAMS:
            rr = [r for r in d["rows"] if r["k"] == k]
            L.append(f"| {d['label']} | {k} | {np.mean([r['holdout'] for r in rr]):.3f} | {np.mean([r['belief'] for r in rr]):.3f} | "
                     f"{np.mean([r['close1km'] for r in rr])*100:.0f}% | {np.nanmin([r['dmin'] for r in rr]) if k > 1 else float('nan'):.0f} | "
                     f"{np.mean([r['cuts'] for r in rr]):.0f} | {np.sum([r['viol'] for r in rr]):.0f} |")
    (OUT/"ma_eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--single", action="store_true", help="the model is a single-UAV one (R15): every UAV flies on its own inputs only")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--no-web", action="store_true", help="do not save the flights as web runs")
    ap.add_argument("--design", choices=("m1", "m2", "m3"), default=None, help="env design (default: the model run's; single models m1)")
    ap.add_argument("--teams", type=int, nargs="+", default=list(TEAMS), help="team sizes to fly (default 1 2 3 4)")
    ap.add_argument("--no-holdout", action="store_true", help="skip the replay on the hold-out particles (planner score only)")
    ap.add_argument("--solo", action="store_true", help="every UAV flies on its own inputs with the multi-env layout (no-teammate channels, "
                                                        "1 / k = 1): a one-UAV model of the multi env (R16) as the team baseline")
    a = ap.parse_args()
    if a.what == "run":
        run(a.model, a.label, a.single, web=not a.no_web, design=a.design, teams=tuple(a.teams), holdout=not a.no_holdout, solo=a.solo)
    else:
        figure(a.labels)


if __name__ == "__main__":
    main()
