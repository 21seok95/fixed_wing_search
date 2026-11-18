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
               help="H(t): 시각 t의 particle 중 x%가 들어가는 영역. 5분 창 A는 5분 간격으로 150 m 흐린 띠(x 70), 본 particle A는 10초마다 particle이 있는 25 m 칸만(시험한 값 x 80)."),
          dict(name="area_roll_max", group=SCORE_GROUP, label="탐색 완료 롤 기준 (°)", min=5, max=45, step=1, value=20,
               help="'롤 기준 이하에서 본 칸만' A에서: 롤이 이 각도 이하(화면이 안정)일 때 화면에 들어온 칸만 탐색 완료로 센다. 20°면 선회율 6.2°/s."),
          dict(name="seen_pod", group=SCORE_GROUP, label="'봤음' 기준 POD", min=.1, max=.9, step=.05, value=.5,
               help="본 particle A에서만: particle의 누적 POD가 이 값 이상이면 본 것으로 친다 (표시는 particle을 따라다님). 한 통과의 POD는 최대 0.7이다."),
          dict(name="area_w", group=SCORE_GROUP, label="POS 가중치 w", min=.25, max=4, step=.25, value=1,
               help="총점 = A + w × POS. 작을수록 아직 안 본 영역을 더 중시한다."),
          dict(name="pass_pod", group="탐지", label="통과 1회 POD 배율", min=.3, max=1.0, step=.05, value=.7,
               help="한 번 지나갈 때 모델 POD에 곱하는 값. 계획기의 통과 POD이기도 하다."),
          dict(name="pass_gap_s", group="탐지", label="통과 간격 (초)", min=1, max=60, step=1, value=5,
               help="화면 밖에 이 시간보다 오래 있다가 다시 보이면 새 통과로 센다."),
          dict(name="blur_tau", group="탐지", label="흔들림 지수 ω₀ (°/s, 0 = 기존 직선)", min=0, max=40, step=1, value=0,
               help="0이면 5°/s까지 1, 20°/s에서 0.2(직선). 0보다 크면 exp(−선회율/ω₀): 조금이라도 돌면 POD가 낮아진다. A(화면 품질)에도 같은 값."),
          dict(name="blur_dead_roll", group="탐지", label="흔들림 없는 롤 (°, 0 = 없음)", min=0, max=20, step=1, value=0,
               help="ω₀가 0보다 클 때만: 이 롤의 선회율(33 m/s에서 5° = 1.5°/s)까지는 POD를 줄이지 않고, 그 위로 exp(−(선회율 − 그 값)/ω₀). "
                    "'롤 기준 이하에서 본 칸만' A의 롤 기준과 같게 두면 A에 세는 화면 = POD를 다 주는 화면이 된다."),
          dict(name="blur_tau_low", group="탐지", label="흔들림 없는 롤 아래의 지수 (°/s, 0 = 그 아래는 1)", min=0, max=40, step=1, value=0,
               help="흔들림 없는 롤과 함께: 0보다 크면 그 아래에서도 exp(−선회율/값)으로 줄어든다(손해 없는 구간 없음). "
                    "15면 D4 = D2(ω₀ 15) × D3: 롤 5°까지 조금씩, 그 위로 빠르게."),
          dict(name="area_sigma", group=SCORE_GROUP, label="실시간 A 영역의 흐림 σ (m)", min=25, max=300, step=25, value=150,
               help="실시간 A(10초마다 영역)를 정할 때 particle을 흐리는 가우시안 폭. 작을수록 영역이 particle에 붙는다(150 m: 영역 칸의 90%가 "
                    "particle에서 약 120 m 안, 75 m: 약 80 m 안)."),
          dict(name="area_sigma_off", group=SCORE_GROUP, label="도로 밖 particle의 σ (m, 0 = 위와 같음)", min=0, max=300, step=25, value=0,
               help="도로 칸 위의 particle은 위의 σ로, 도로 밖 particle은 이 σ로 흐린다(사용자 2026-11-12: 75 / 150). 도로 밖 particle은 "
                    "넓게 흩어져 75 m로는 영역이 잘게 조각나므로."),
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
AREA_DEFS = [dict(value="window", label="A: 5분 창의 훑은 칸 (기본)"), dict(value="realtime", label="A: 10초마다 실시간 영역의 훑은 칸"),
             dict(value="realtime_q", label="A: 실시간 영역 + 화면 품질(흔들림 계수)"),
             dict(value="realtime_t", label="A: 실시간 영역, 롤 기준 이하에서 본 칸만 탐색 완료"),
             dict(value="seen", label="A: 본 particle 기준, 10초마다 (시험)")]
X_WINDOW_REF = 70.0        # the window A is also reported with its own x for comparison with the runs before 2026-11-11


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
    from rl_search.real_scenario import V_UAV, make_env
    over = [f"search.fov_cross_m={float(s['cover_m'])}", f"search.pod={float(s['pass_pod'])}"]
    dead = float(s.get("blur_dead_roll", 0.0) or 0.0)                      # D3: no turn-rate loss up to the rate of this roll
    return make_env(seed, lead_m=float(s["lead_m"]), prelook_s=float(s["prelook_s"]), bank_deg=float(s["bank_deg"]), overrides=over, region=region, terr=terr, area=area,
                    edge_mode=str(s.get("edge_mode", "mask")),       # "none" (RL from R6, user 2026-11-12): fly off the map without ending the flight
                    detection=dict(pass_mode="scale", pass_pod_max=float(s["pass_pod"]), pass_gap_s=float(s["pass_gap_s"]),
                                   rate_tau_dps=float(s.get("blur_tau", 0.0)),
                                   rate_dead_dps=float(np.degrees(9.80665*np.tan(np.radians(dead))/V_UAV)) if dead > 0 else 0.0,
                                   rate_tau_low_dps=float(s.get("blur_tau_low", 0.0) or 0.0)))


def area_sigma(s: dict, default: float) -> float:
    """Blur width (m) of the real-time A region: the setting ``area_sigma``, else the env's bandwidth (150 m)."""
    v = s.get("area_sigma")
    return float(v) if v else float(default)


def area_sigma_off(s: dict) -> float:
    """Blur width (m) of the off-road particles in the real-time A region (user 2026-11-12: 150 with the road particles at 75); 0 = all
    particles at ``area_sigma``."""
    return float(s.get("area_sigma_off") or 0.0)


def realtime_kw(s: dict, region: str) -> dict:
    """Extra ``RealtimeArea`` arguments of the settings: the off-road sigma and the road cells, when the off-road sigma is set."""
    off = area_sigma_off(s)
    if not off:
        return {}
    from rl_search.area_score import road_mask
    return dict(h_off=off, road=road_mask(region))


def run_case(region: str, scen, vres, vc, view: dict, s: dict, q=None, options: dict | None = None, terr=None, t_sim: float = 0.0,
             t_start: float | None = None) -> dict:
    """SAROPS (1 to 5 UAVs) on one scenario with the settings ``s`` (FIELDS names, ``cap``, ``score``). ``options`` go to env.reset
    (default: launch at 0 s from ``scen.start``; stage 1 / 2 with ``t0_s`` and ``start_pose`` start mid-flight). Returns
    dict(opt, res, actions): the web replay payload and the action lists (for replays on other particles)."""
    from rl_search.area_score import AreaSaropsPlanner, AreaScore, NoOverlapAreaSarops, NoOverlapSarops, RealtimeArea, SeenArea, score_frames
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import terrain_path
    from rl_search.sarops import SaropsPlanner
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter() if t_start is None else t_start
    options = options or dict(scenario=scen, stage=3)
    put = (lambda d: q.put(d)) if q is not None else (lambda d: None)
    n_uav, area_on, x, w = int(s["n_uav"]), s["score"] == "area", float(s["area_x"]), float(s["area_w"])
    seen_def = s.get("area_def", "window") == "seen"
    rt_def = s.get("area_def", "window") in ("realtime", "realtime_q", "realtime_t")
    terr = load_terrain(terrain_path(region)) if terr is None else terr
    envs = [team_env(s, region, terr, view["seed"], area_on or seen_def or rt_def) for _ in range(n_uav)]
    if seen_def:            # also with POS-only planning (A weight 0), so the seen A can be reported
        area = SeenArea(scen.track, scen.exit_step, scen.save_dt_s, x, float(s.get("seen_pod", 0.5)))
        if n_uav > 1:
            area.team = envs
    elif rt_def:
        from rl_search.rl_eval_set import flight_window
        area = RealtimeArea(terr, area_sigma(s, envs[0].h), scen.track, scen.exit_step, scen.save_dt_s, x, quality=s.get("area_def") == "realtime_q",
                            window=flight_window(options),
                            roll_max_deg=float(s.get("area_roll_max", 20.0)) if s.get("area_def") == "realtime_t" else None,
                            **realtime_kw(s, region))
        area.img_min = float(s.get("area_img_min", 0.0) or 0.0)
    else:
        area = AreaScore(terr, envs[0].h, scen.track, scen.save_dt_s, x, envs[0].lrc_R) if area_on else None
    for u, e in enumerate(envs):
        e.uav_id, e.sample_log, e.spacing_budget = u, [], s["cap"] == "A"
        if area is not None:
            e.area, e.pos_w, e.area_w = area, (w if area_on else 1.0), (float(s.get("a_weight", 1.0)) if area_on else 0.0)

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
    if seen_def or rt_def:
        aw = score_frames(terr, envs[0].h, scen.track, scen.save_dt_s, X_WINDOW_REF, logs, envs[0].detector.cfg)
        area_out = (seen_payload if seen_def else realtime_payload)(area, area_on, x, w, final, aw["A"])
    else:
        a = score_frames(terr, envs[0].h, scen.track, scen.save_dt_s, x, logs, envs[0].detector.cfg)
        g = a.pop("grid")
        area_out = {**a, "def": "window", "on": area_on, "x": x, "w": w, "total": a["A"] + w*final, "nx": g["nx"], "ny": g["ny"], "cell": g["cell"],
                    "x0": g["x0"], "y0": g["y0"], "bits": b64(g["bits"].astype("<u2")), "first": b64(g["first"])}
    start = options.get("start_pose", scen.start)[:2] if options.get("stage", 3) in (1, 2) else scen.start
    opt = dict(T=T, N=int(n), dt_s=float(dt), weight=b64(W), uavs=uavs, start=[float(v) for v in start], lkp=[float(v) for v in scen.lkp],
               pos_t=np.round(np.asarray(res["t"]), 1).tolist(), pos=np.round(np.asarray(res["pos"]), 4).tolist(), final=final, escaped=float(envs[0].belief.escaped),
               no_clear=int(sum(getattr(k, "overlap_forced", 0) for k in pols)), area=area_out,
               sec=round(time.perf_counter() - t0, 1), settings=s)
    return dict(opt=opt, res=particles_payload(vres, vc, view["heading"], start, t_sim), actions=res["actions"], paths=res["xy"])


def realtime_payload(area, on: bool, x: float, w: float, final: float, a_window: float | None = None) -> dict:
    """What the page needs for a ``RealtimeArea``: A, c per 10 s frame, region size, the region cells once a minute (keys of the 25 m
    terrain grid, ``off`` = where each minute starts) and the sweep time of every swept cell. ``a_window``: the 5 min window A (x 70)."""
    step = max(int(round(60.0/area.dt)), 1)
    frames = list(range(step - 1, area.J, step))                   # window-relative, one a minute
    keys = [area.H[area.j0 - 1 + j] for j in frames]
    off = np.cumsum([0] + [len(k) for k in keys])
    sw = [(e[0], e[2], t) for e, t in zip(area.log, area.log_t) if t >= 0]          # (keys, new quality 0-255, time), in time order
    sk = np.concatenate([k for k, _, _ in sw]).astype("<i4") if sw else np.zeros(0, "<i4")
    sq = np.concatenate([q for _, q, _ in sw]).astype(np.uint8) if sw else np.zeros(0, np.uint8)
    st = np.concatenate([np.full(len(k), t) for k, _, t in sw]).astype("<f4") if sw else np.zeros(0, "<f4")
    return {"def": "realtime", "on": on, "x": x, "w": w, "A": area.A, "total": area.A + w*final, "dt": area.dt, "J": area.J, "j0": area.j0,
            "min_t0": float(area.t[frames[0]]) if frames else 0.0,
            "c": np.round(area.curve(), 4).tolist(), "size_km2": np.round(area.size*25.0**2/1e6, 4).tolist(),
            "swept_km2": float(len(sk))*25.0**2/1e6, "n": area.n, "X0": area.X0, "cell": 25.0, "minutes": len(frames),
            "keys": b64(np.concatenate(keys).astype("<i4") if keys else np.zeros(0, "<i4")), "off": b64(off.astype("<i4")),
            "sweep_keys": b64(sk), "sweep_t": b64(st), "sweep_q": b64(sq), "quality": bool(area.quality), "A_window": a_window,
            "x_window": X_WINDOW_REF}


def seen_payload(area, on: bool, x: float, w: float, final: float, a_window: float | None = None) -> dict:
    """What the page needs for a ``SeenArea``: A, c per 10 s frame, region size, the seen time of each particle (-1 = never) and the
    region membership of each particle per frame (bits, np.packbits order). ``a_window``: the window A (x 70) of the same flight."""
    t_seen = np.where(np.isfinite(area.t_seen), area.t_seen, -1.0).astype("<f4")
    return {"def": "seen", "on": on, "x": x, "w": w, "pod": area.pod, "A": area.A, "total": area.A + w*final, "dt": area.dt, "J": area.J,
            "c": np.round(area.curve(), 4).tolist(), "size_km2": np.round(area.size*25.0**2/1e6, 4).tolist(),
            "seen_share": float(area.seen.mean()), "t_seen": b64(t_seen), "core": b64(np.packbits(area.a > 0, axis=1)),
            "A_window": a_window, "x_window": X_WINDOW_REF}


def _run(p: dict, q) -> dict:
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios, chungju_preset, terrain_path
    from rl_search.scenario import ScenarioSampler
    from rl_search.terrain import load_terrain
    from rl_search.vehicle import VehicleConfig
    s = {f["name"]: f["value"] for f in FIELDS} | {"cap": "A", "score": "area", "area_def": "window"} | p.get("opt", {})
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
