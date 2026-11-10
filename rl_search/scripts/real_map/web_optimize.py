"""SAROPS on the particles shown in the web viewer (vehicle_web): 1 to 5 UAVs, 60 min, run in its own process.

The settings default to the ones agreed on 2026-11-09 (docs/real_map_todo.md): lead + gimbal 8 s, each pass finds
0.7 x the model's pass POD, pass gap 5 s, lane width 190 m, 30 deg bank, lane time flown, transit limit 30 min, fix A.
The plan maximises A + w POS (rl_search.area_score, the swept cells come from the camera image) or POS alone; A is reported
either way. Two or more UAVs launch 60 s apart from one point; rectangles held at the same time stay 300 m apart (never overlapping:
a UAV with no clear rectangle flies on and tries again).
"""
from __future__ import annotations

import base64
import time
import traceback
import warnings

import numpy as np

SCORE_GROUP = "탐색 점수"
FIELDS = [dict(name="n_uav", group="무인기", label="무인기 수", min=1, max=5, step=1, value=1,
               help="같은 출발점에서 60초 간격으로 이륙한다. 같은 시간에 맡는 사각형은 서로 300 m 이상 떨어뜨린다."),
          dict(name="area_x", group=SCORE_GROUP, label="영역 x (%)", min=50, max=90, step=5, value=70,
               help="H(t): 시각 t의 particle 중 x%가 들어가는 영역 (5분 간격). 이 안에서 카메라 화면에 들어온 칸을 훑은 칸으로 센다."),
          dict(name="area_w", group=SCORE_GROUP, label="POS 가중치 w", min=.25, max=4, step=.25, value=1,
               help="총점 = A + w × POS. 작을수록 아직 안 본 영역을 더 중시한다."),
          dict(name="pass_pod", group="탐지", label="통과 1회 POD 배율", min=.3, max=1.0, step=.05, value=.7,
               help="한 번 지나갈 때 모델 POD에 곱하는 값. 계획기의 통과 POD이기도 하다."),
          dict(name="pass_gap_s", group="탐지", label="통과 간격 (초)", min=1, max=60, step=1, value=5,
               help="화면 밖에 이 시간보다 오래 있다가 다시 보이면 새 통과로 센다."),
          dict(name="max_transit_min", group="계획", label="이동 제한 (분)", min=5, max=60, step=5, value=30,
               help="계획기가 고를 수 있는 가장 먼 사각형까지의 이동 시간."),
          dict(name="search_min", group="계획", label="탐색 시간 (분)", min=5, max=20, step=1, value=10,
               help="사각형 하나의 레인 시간 예산. 사각형 크기를 정한다."),
          dict(name="cover_m", group="계획", label="레인 폭 (m)", min=100, max=250, step=5, value=190,
               help="레인 하나가 덮는다고 계획기가 보는 폭."),
          dict(name="lead_m", group="비행", label="발자국 앞섬 (m)", min=0, max=700, step=10, value=510,
               help="화면 중심이 레인 위를 지나도록 기체 경로를 뒤로 미는 거리."),
          dict(name="prelook_s", group="비행", label="짐벌 앞당김 (초)", min=0, max=20, step=1, value=8,
               help="레인이 끝나기 이 시간 전부터 다음 레인 시작점을 본다."),
          dict(name="bank_deg", group="비행", label="선회 롤 (°)", min=10, max=45, step=5, value=30,
               help="레인을 바꿀 때의 선회 경사. 30°면 반경 약 192 m.")]
CAPS = [dict(value="A", label="12분 상한 + 수정 A"), dict(value="now", label="12분 상한만"), dict(value="none", label="상한 없음")]
SCORES = [dict(value="area", label="A + w × POS (미탐색 영역 포함)"), dict(value="pos", label="POS만")]


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


POS_UNIT_M = 10.0          # positions are sent as int16 in 10 m units (+-327 km)


def particles_payload(res, cfg, heading: float, auto_start, sim_s: float) -> dict:
    """What the page needs to draw and play back one particle simulation."""
    from rl_search.scripts.real_map.vehicle_tuner import summarize
    shares, pct = summarize(res)
    pos = np.clip(np.rint(res.track/POS_UNIT_M), -32767, 32767).astype("<i2")
    return {"T": int(res.track.shape[0]), "N": int(res.track.shape[1]), "dt_s": cfg.save_dt_s,
            "pos_unit_m": POS_UNIT_M, "pos": b64(pos), "state": b64(res.state.astype(np.uint8)),
            "speed": b64(np.clip(np.rint(res.speed_kmh.astype(float)), 0, 255).astype(np.uint8)),
            "shares": shares.round(4).tolist(), "pct": np.nan_to_num(pct, nan=-1).round(3).tolist(),
            "lkp_road": [float(v) for v in res.lkp_road], "heading": float(heading), "sim_s": round(sim_s, 2),
            "speed_cap_kmh": cfg.speed_cap_kmh, "auto_start": [float(v) for v in auto_start]}


def footprints(log, cfg) -> np.ndarray:
    """Ground corners of the image at each logged second: (n, 4, 2)."""
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    x, f, r, u = log[:, 1:3], log[:, 5:8], log[:, 8:11], log[:, 11:14]
    out = []
    for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        d = f + sh*th*r + sv*tv*u
        s = np.minimum(cfg.alt_m/np.maximum(-d[:, 2], 1e-3), 5000.0)
        out.append(x + s[:, None]*d[:, :2])
    return np.stack(out, 1)


def team_env(s: dict, region: str, terr, seed: int, area: bool = False):
    """Search env of one UAV with the web settings ``s`` (detection belief, lane width, lead, gimbal pre-look, bank)."""
    from rl_search.real_scenario import make_env
    over = [f"search.fov_cross_m={float(s['cover_m'])}", f"search.pod={float(s['pass_pod'])}"]
    return make_env(seed, lead_m=float(s["lead_m"]), prelook_s=float(s["prelook_s"]), bank_deg=float(s["bank_deg"]), overrides=over, region=region, terr=terr, area=area,
                    detection=dict(pass_mode="scale", pass_pod_max=float(s["pass_pod"]), pass_gap_s=float(s["pass_gap_s"])))


def run_case(region: str, scen, vres, vc, view: dict, s: dict, q=None, options: dict | None = None, terr=None, t_sim: float = 0.0,
             t_start: float | None = None) -> dict:
    """SAROPS (1 to 5 UAVs) on one scenario with the settings ``s`` (FIELDS names, ``cap``, ``score``). ``options`` go to env.reset
    (default: launch at 0 s from ``scen.start``; stage 1 / 2 with ``t0_s`` and ``start_pose`` start mid-flight). Returns
    dict(opt, res, actions): the web replay payload and the action lists (for replays on other particles)."""
    from rl_search.area_score import AreaSaropsPlanner, AreaScore, NoOverlapAreaSarops, NoOverlapSarops, score_frames
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import terrain_path
    from rl_search.sarops import SaropsPlanner
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter() if t_start is None else t_start
    options = options or dict(scenario=scen, stage=3)
    put = (lambda d: q.put(d)) if q is not None else (lambda d: None)
    n_uav, area_on, x, w = int(s["n_uav"]), s["score"] == "area", float(s["area_x"]), float(s["area_w"])
    terr = load_terrain(terrain_path(region)) if terr is None else terr
    envs = [team_env(s, region, terr, view["seed"], area_on) for _ in range(n_uav)]
    area = AreaScore(terr, envs[0].h, scen.track, scen.save_dt_s, x, envs[0].lrc_R) if area_on else None
    for u, e in enumerate(envs):
        e.uav_id, e.sample_log, e.spacing_budget = u, [], s["cap"] == "A"
        if area is not None:
            e.area, e.pos_w, e.area_w = area, w, 1.0

    def team_surv():
        return np.round(np.prod([e.surv for e in envs], axis=0)*255).astype(np.uint8)

    base = (NoOverlapAreaSarops if area_on else NoOverlapSarops) if n_uav > 1 else (AreaSaropsPlanner if area_on else SaropsPlanner)
    rec_t, rec_w = [], []

    class Planner(base):
        def __call__(self, e):
            if not rec_t or e.t > rec_t[-1] + 1e-6:
                rec_t.append(e.t)
                rec_w.append(team_surv())
            return super().__call__(e)

        def _plan(self, e):
            ok = super()._plan(e)
            put(dict(kind="progress", stage="비행·계획", t_min=round(e.t/60, 1), plans=sum(len(k.rects) for k in pols)))
            return ok

    pols = [Planner(W=float(s["pass_pod"])*float(s["cover_m"]), n_starts=3, big_jump=True, search_s=60.0*float(s["search_min"]),
                    max_transit_s=60.0*float(s["max_transit_min"]), lane_mode="cover", peak_timing="intercept",
                    extra_s=7200.0 if s["cap"] == "none" else 120.0) for _ in range(n_uav)]
    for k in pols:
        k.team, k.strict_overlap = pols, n_uav > 1 and bool(s.get("strict", 1))
    res = run_team_launch(envs, pols, options, 60.0)

    dt, n = scen.save_dt_s, envs[0].belief.n
    t_end = max(float(t[-1]) for t in res["tt"])
    T = min(int(np.ceil(t_end/dt)) + 1, len(scen.track))
    rt, rw = np.array(rec_t), np.array(rec_w)
    W = np.full((T, n), 255, np.uint8)
    for k in range(T):
        i = np.searchsorted(rt, k*dt, side="right") - 1
        if i >= 0:
            W[k] = rw[i]
    W[max(0, int(np.floor(t_end/dt))):] = team_surv()
    uavs, logs = [], []
    for u, (e, pol) in enumerate(zip(envs, pols)):
        log = np.array(e.sample_log)
        logs.append(log)
        cw = e.fp_cross
        rects = [dict(tp=float(tp), cx=float(r.cx), cy=float(r.cy), ell=float(r.ell), wid=float(r.wid), theta=float(r.theta),
                      k=int(round(cw/r.spacing)) if r.spacing else None, lanes=np.round(pts, 1).tolist()) for tp, r, pts in pol.rects]
        tasks = [dict(t0=float(k["t0"]), lane_t=k["lane_t"], end=k["end"], how=k["how"]) for k in pol.tasks]
        uavs.append(dict(rects=rects, tasks=tasks, tt=b64(np.asarray(res["tt"][u], np.float32)), xy=b64(np.asarray(res["xy"][u], np.float32)), n_path=len(res["tt"][u]),
                         log_t=b64(log[:, 0].astype(np.float32)), log_xyyaw=b64(log[:, 1:4].astype(np.float32)), quad=b64(footprints(log, e.detector.cfg).astype(np.float32)),
                         target=b64(log[:, 14:16].astype(np.float32)), n_log=len(log)))
    final = float(res["pos"][-1])
    a = score_frames(terr, envs[0].h, scen.track, scen.save_dt_s, x, logs, envs[0].detector.cfg)
    g = a.pop("grid")
    start = options.get("start_pose", scen.start)[:2] if options.get("stage", 3) in (1, 2) else scen.start
    opt = dict(T=T, N=int(n), dt_s=float(dt), weight=b64(W), uavs=uavs, start=[float(v) for v in start], lkp=[float(v) for v in scen.lkp],
               pos_t=np.round(np.asarray(res["t"]), 1).tolist(), pos=np.round(np.asarray(res["pos"]), 4).tolist(), final=final, escaped=float(envs[0].belief.escaped),
               no_clear=int(sum(getattr(k, "overlap_forced", 0) for k in pols)),
               area={**a, "on": area_on, "x": x, "w": w, "total": a["A"] + w*final, "nx": g["nx"], "ny": g["ny"], "cell": g["cell"], "x0": g["x0"], "y0": g["y0"],
                     "bits": b64(g["bits"].astype("<u2")), "first": b64(g["first"])},
               sec=round(time.perf_counter() - t0, 1), settings=s)
    return dict(opt=opt, res=particles_payload(vres, vc, view["heading"], start, t_sim), actions=res["actions"], paths=res["xy"])


def _run(p: dict, q) -> dict:
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios, chungju_preset, terrain_path
    from rl_search.scenario import ScenarioSampler
    from rl_search.terrain import load_terrain
    from rl_search.vehicle import VehicleConfig
    s = {f["name"]: f["value"] for f in FIELDS} | {"cap": "A", "score": "area"} | p.get("opt", {})
    t0 = time.perf_counter()
    view = p["view"]
    region = view.get("region", "chungju")
    terr = load_terrain(terrain_path(region))
    q.put(dict(kind="progress", stage="particle 계산", t_min=0.0, plans=0))
    vc = VehicleConfig(**p["config"])
    t_sim = time.perf_counter()
    sampler = ScenarioSampler(terr, chungju_preset().scenario)
    scen, vres = VehicleScenarios(sampler, vc, region=region).make_at(view["seed"], view["lkp"], view["heading"], view.get("start"), with_res=True)
    t_sim = time.perf_counter() - t_sim
    out = run_case(region, scen, vres, vc, view, s, q, terr=terr, t_sim=t_sim, t_start=t0)
    out.pop("actions"); out.pop("paths")
    return out


def worker(params: dict, q) -> None:
    try:
        q.put(dict(kind="result", data=_run(params, q)))
    except Exception:
        q.put(dict(kind="error", msg=traceback.format_exc(limit=4)))
