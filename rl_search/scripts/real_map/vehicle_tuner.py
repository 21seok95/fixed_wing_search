"""Slider viewer for tuning the vehicle evader model on the real-region terrain.

Run from the repository root:
    python -m rl_search.scripts.real_map.vehicle_tuner
    python -m rl_search.scripts.real_map.vehicle_tuner --snapshot rl_search/data/real/chungju/vehicle_default.png

Window:
- Left: the map with the particles coloured by state at the time on the time slider.
  Left-click on the map moves the LKP (snapped to the nearest road) and runs again.
- Right: state shares over time, distance from the LKP over time, speed histogram at the slider time.
- "재생" (or the space key) plays the time slider; the left / right keys step one minute.
- Bottom: parameter sliders in three groups (speed / per-minute rates / route and start). With
  "자동 실행" on, every slider change runs the particles again; otherwise press "실행".
  "저장" writes the settings to rl_search/data/real/vehicle_params.json (loaded at start if present).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from rl_search.vehicle import (EXIT, FAST, OFF, SLOW, STATE_NAMES, STOP, VehicleConfig, load_net,
                               simulate_vehicle)

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

PARAMS = Path("rl_search/data/real/vehicle_params.json")
STATE_COLORS = {FAST: "#111111", SLOW: "#c0702f", OFF: "#2a9d8f", STOP: "#7b2cbf", EXIT: "#888888"}
DEFAULT_LKP = (5000.0, 5000.0)
DEFAULT_HEADING = 45.0

# (field, label, min, max, step) per group; heading / particles / duration / seed are viewer settings. Reorganised 2026-11-18 (user: the
# web viewer should follow the current scenarios): every field has a help line with the range training draws from (rl_eval_set V2 /
# MOUNTAIN); "논밭" in a label reads "도로 밖" on the desert maps (vehicle_web). Bool fields are shown as check boxes on the web page.
GROUPS = {
    "속도 (km/h)": [("fast_kmh", "포장(빠른 길) 평균", 10, 90, 1), ("fast_std_kmh", "포장 퍼짐", 0, 20, .5),
                  ("slow_kmh", "비포장(느린 길) 평균", 5, 60, 1), ("slow_std_kmh", "비포장 퍼짐", 0, 20, .5),
                  ("off_kmh", "논밭 평균", 2, 40, .5), ("off_std_kmh", "논밭 퍼짐", 0, 15, .5),
                  ("speed_clip_sd", "속도 자르기 (퍼짐의 배)", 0, 3, .5), ("speed_cap_kmh", "최대 속도", 20, 100, 1)],
    "도로 이탈·복귀": [("p_leave_fast_min", "포장→논밭 (분당)", 0, .5, .005), ("p_leave_slow_min", "비포장→논밭 (분당)", 0, .5, .005),
                  ("p_return_min", "논밭→도로 복귀 (분당)", 0, 1, .01), ("return_radius_m", "복귀 거리 (m)", 25, 500, 25),
                  ("off_max_slope_deg", "논밭 경사 한계 (°)", 3, 30, 1),
                  ("shortcut_m", "지름길 최대 거리 (m, 0 = 끔)", 0, 3000, 100), ("shortcut_factor", "지름길 시간 한계 (도로의 배)", 1, 3, .1)],
    "경로·시작": [("heading", "도주 방향 (°)", 0, 359, 5), ("heading_std_deg", "방향 퍼짐 (°)", 0, 120, 1),
              ("reorient_std_deg_min", "방향 흔들림 (°/√분)", 0, 20, .5),
              ("p_detour", "갈림길 우회 확률", 0, .5, .01), ("goal_sector_deg", "목표 구역 폭 (°)", 2, 30, 1),
              ("particles", "입자 수", 200, 5000, 100), ("duration_min", "시간 (분)", 10, 120, 5),
              ("seed", "seed", 0, 50, 1)],
    # user 2026-11-17 (docs/scenario_redesign.md): on/off rules (0 / 1 here; the web page shows them as check boxes, with HELP)
    "되돌아옴·지도 밖": [("start_by_route", "목표 쪽으로 출발·복귀", 0, 1, 1), ("no_return", "지나온 길을 거꾸로 안 감", 0, 1, 1),
                   ("deadend_offroad", "막다른 길 끝에서 도로 밖으로", 0, 1, 1), ("off_follow", "도로 밖 장애물은 옆으로 돌아감", 0, 1, 1),
                   ("detour_deadend_deg", "우회로 막다른 길 (도주 방향 ±°)", 0, 90, 5), ("detour_max_deg", "우회 길 방향 한계 (±°)", 30, 180, 5),
                   ("detour_keep_tier", "우회는 같은 급 이상의 길로만", 0, 1, 1),
                   ("exit_straight", "지도 밖으로 나가면 계속 이동", 0, 1, 1),
                   ("exit_trend_min", "지도 밖: 나가기 전 평균 속도를 잴 시간 (분)", 0, 15, 1),
                   ("exit_trend_max_deg", "지도 밖: 처음 방향에서 벗어날 수 있는 각도 (°)", 0, 180, 5)],
    "고급 (지금 학습에서 안 씀)": [("p_stop_min", "정지 (분당)", 0, .3, .005), ("p_go_min", "재출발 (분당)", 0, 1, .01),
                            ("w_off_road", "논밭에서 도로 쪽 끌림", 0, 5, .1), ("speed_band", "운전 성향 폭 (평균의 ±배)", 0, .5, .05)],
}
# shown when the pointer rests on a setting's name in the web page (학습 = the range the training scenarios draw from)
HELP = {
    "fast_kmh": "포장도로(고속·국도·지방도 등)에서의 평균 속도. 학습: 50 (사막·산악 지형 모두).",
    "fast_std_kmh": "입자마다 운전 성향이 달라 생기는 속도 퍼짐(표준편차). 학습: 사막 10, 산악 5~15(시나리오마다 0.5~1.5배).",
    "slow_kmh": "비포장·농로·임도의 평균 속도. 학습: 30.",
    "slow_std_kmh": "비포장 속도 퍼짐. 학습: 사막 5, 산악 2.5~7.5(포장과 같은 배율).",
    "off_kmh": "도로 밖(들판, 논밭)의 평균 속도. 학습: 10.",
    "off_std_kmh": "도로 밖 속도 퍼짐. 학습: 사막 2.5, 산악 1.25~3.75(같은 배율).",
    "speed_clip_sd": "속도를 평균 ± 퍼짐 × 이 값 안으로 자른다(예: 포장 50 ± 2×10 = 30~70). 0 = 자르지 않음. 학습: 2.",
    "speed_cap_kmh": "어떤 길에서도 넘지 않는 최대 속도. 학습: 70.",
    "p_leave_fast_min": "포장도로에서 1분에 도로를 떠날 확률. 학습: 도로형 0, 혼합형 0~0.05, 들판형 0.05~0.3, 산악형 0.05~0.2(지름길이 있을 때만 건넘).",
    "p_leave_slow_min": "비포장에서 1분에 도로를 떠날 확률. 학습: 포장과 같음.",
    "p_return_min": "도로 밖에서 도로가 복귀 거리 안에 있을 때 1분에 도로로 돌아올 확률. 학습: 도로형 1, 혼합형 0.1~1, 들판형 0~0.1. 지름길에는 쓰이지 않음.",
    "return_radius_m": "이 거리 안에 도로가 있어야 도로로 돌아올 수 있다.",
    "off_max_slope_deg": "도로 밖에서 이보다 가파른 땅은 못 간다(숲·물·큰 도시·군사 구역도 못 감). 학습: 15.",
    "shortcut_m": "0보다 크면 도로 밖은 지름길로만: 이 거리 안의 다른 도로로 논밭을 곧게 건너 합류할 수 있고 목표에 가까워질 때만 떠난다. "
                  "도로 밖을 자유롭게 다니지 않으므로 갇히지 않음. 학습: 산악형 2000, 그 밖 0.",
    "shortcut_factor": "지름길(도로 밖 시간 + 그 뒤 도로 시간)이 도로로만 갈 때의 이 배 안일 때만 건넌다. 학습: 1.5.",
    "heading": "도주 방향(입자들이 향하는 쪽, 0 = 동쪽, 90 = 북쪽). 지도 오른쪽 위 원판으로도 바꿀 수 있음.",
    "heading_std_deg": "입자마다 도주 방향이 이만큼 흩어진다(표준편차). 갈래 수를 정하는 주된 값(충주 120 km: 10° 한 갈래 ~ 90° 유효 갈래 57). 학습: 사막 10~60, 산악 10~90.",
    "reorient_std_deg_min": "가는 동안 도주 방향이 조금씩 바뀌는 정도. 크면 갈래와 함께 고리 돌기도 늘어남. 학습: 사막 0~5, 산악 0~3.",
    "p_detour": "갈림길에서 가장 빠른 길 대신 다른 길로 갈 확률. 갈래는 거의 안 늘고 작은 고리 돌기가 늘어남(충주 120 km: 0.15에서 31%). 학습: 사막 0~0.15, 산악 0~0.05.",
    "goal_sector_deg": "도주 방향의 지도 가장자리를 이 각도 폭의 구역으로 나눠, 그 구역까지 가장 빠른 길로 간다. 영향 작음. 학습: 사막 10~30, 산악 20.",
    "particles": "입자 수(많을수록 매끄럽지만 느림). 학습: 5000.",
    "duration_min": "입자를 움직일 시간. 학습: 70분(비행 60분 + 예측 10분).",
    "seed": "같은 설정에서 다른 무작위 결과를 보려면 바꿈. '학습처럼 뽑기'도 이 seed를 씀.",
    "start_by_route": "LKP에서 출발할 때와 도로 밖에서 도로로 돌아올 때, 목표(도주 방향의 지도 가장자리)까지 시간이 짧은 쪽으로 간다. "
                      "우회 확률만큼은 도주 방향이 가리키는 쪽으로 출발한다(막다른 길이면 도주 방향 ±45° 안일 때만). 학습: 켬.",
    "no_return": "갈림길에서 최근 지나온 도로 12개를 거꾸로 달리지 않고, 막다른 길로 방향을 돌리지도 않는다. 남은 길 중 목표까지 가장 빠른 길로 "
                 "(내비게이션의 경로 재탐색처럼). 거꾸로 가는 길밖에 없을 때만 허용. 학습: 켬.",
    "deadend_offroad": "막다른 길 끝에서 돌아서지 않고 도주 방향 쪽 도로 밖으로 간다. 떠나온 막다른 길로는 다시 올라타지 않는다. "
                       "지름길을 켜면 쓰이지 않음(막다른 길 끝에서 돌아섬). 학습: 켬.",
    "off_follow": "도로 밖에서 앞쪽 ±90°가 모두 못 가는 땅이면 돌아서지 않고 장애물 가장자리를 따라 돌아간다(Bug 알고리즘). 도주 방향이 열리고 "
                  "막혔던 곳보다 나아가면 다시 도주 방향으로. 사방이 막혔을 때만 돌아선다. 학습: 켬.",
    "detour_deadend_deg": "무작위 우회 때 도주 방향 ± 이 각도 안을 향한 막다른 길에 들어갈 수 있다(끝에서 도로 밖으로). 0 = 들어가지 않음(예전). 학습: 45.",
    "detour_max_deg": "무작위 우회로 고르는 길을 도주 방향 ± 이 각도 안을 향한 길로 제한한다(추적자 쪽으로 우회하지 않음). 180 = 제한 없음(예전). 학습: 90.",
    "detour_keep_tier": "갈림길 우회 때 포장도로에서 비포장으로는 빠지지 않는다(같은 급 이상의 길로만). 작은 마을 길로 빠져 고리를 도는 입자가 약 절반으로 "
                        "줄어듦. 학습: 산악 지형 켬, 사막 지형 끔.",
    "exit_straight": "지도 밖으로 나간 입자가 나갈 때의 방향·속도로 계속 간다(믿음에 남음). 끄면 지도 밖에서 멈춤(예전). 학습: 켬.",
    "exit_trend_min": "0이면 나가는 순간의 방향·속도로 똑바로 감. 0보다 크면 나가기 전 이 시간 동안 실제로 나아간 방향으로, 나갈 때의 속도(시나리오 속도 설정)로 출발해 도주 방향 표류(방향 흔들림)만큼 방향이 천천히 바뀜(사용자 11-18: 너무 일자는 인위적). 학습: 5분.",
    "exit_trend_max_deg": "지도 밖 입자의 방향이 처음 방향에서 이 각도 넘게 돌지 않음(되돌아오지 않게). 학습: 90°.",
    "p_stop_min": "1분에 멈출 확률(예전 학습). 지금 학습: 0(멈추지 않음).",
    "p_go_min": "멈춘 입자가 1분에 다시 출발할 확률(예전 학습).",
    "w_off_road": "도로 밖에서 가까운 도로 쪽으로 끌리는 정도(예전 설정).",
    "speed_band": "입자의 운전 성향을 평균의 ±이 배 안에 둔다(R10~R14의 정지 없는 표적). 지금 학습: 0(대신 속도 자르기).",
}


def load_settings():
    cfg, view = VehicleConfig(particles=1000), {"heading": DEFAULT_HEADING, "lkp": list(DEFAULT_LKP), "seed": 0}
    if PARAMS.exists():
        d = json.loads(PARAMS.read_text())
        cfg = VehicleConfig(**{k: v for k, v in d.get("config", {}).items() if k in asdict(cfg)})
        view.update(d.get("view", {}))
    return cfg, view


def background(ax, net):
    t = net.terr
    a = t.axis/1000
    s = 2                                                     # draw at 2 cells for speed
    hs = LightSource(azdeg=315, altdeg=45).hillshade(t.z[::s, ::s], vert_exag=2, dx=t.cell*s, dy=t.cell*s)
    rgb = np.ones(hs.shape + (3,))*np.array([.96, .95, .90])
    rgb[t.forest[::s, ::s]] = (.80, .88, .78)
    rgb[t.built[::s, ::s]] = (.90, .80, .82)
    if getattr(t, "military", None) is not None:
        rgb[t.military[::s, ::s]] = (.66, .68, .55)                  # military land (closed to the targets)
    rgb[t.city[::s, ::s]] = (.86, .55, .60)
    rgb[t.water[::s, ::s]] = (.55, .72, .95)
    rgb *= (.75 + .25*hs)[..., None]
    h = t.cell/2000
    ax.imshow(rgb, origin="lower", extent=(a[0] - h, a[-1] + h, a[0] - h, a[-1] + h), interpolation="nearest")
    segs_f, segs_s, segs_x = [], [], []
    for k in range(len(net.edge_len)):
        xy = net.poly[net.off[k]:net.off[k + 1]]/1000
        if not net.edge_main[k]:
            segs_x.append(xy)                                 # city roads and cut-off pieces: not driven
        else:
            (segs_f if net.edge_tier[k] == 2 else segs_s).append(xy)
    ax.add_collection(LineCollection(segs_x, colors="#9a9a9a", linewidths=.3, alpha=.5, linestyles=":"))
    ax.add_collection(LineCollection(segs_s, colors="#b9905f", linewidths=.35, alpha=.7))
    ax.add_collection(LineCollection(segs_f, colors="#555555", linewidths=.6, alpha=.8))
    ax.set_xlim(a[0], a[-1]); ax.set_ylim(a[0], a[-1]); ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")


def map_image(net, px: int = 2000):
    """Terrain and roads drawn once off-screen into one RGBA image (redrawing 27 k road lines was half
    of every frame); returns the image and its extent in km."""
    fig = Figure(figsize=(px/100, px/100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    background(ax, net)
    for col, lw in zip(ax.collections, (.6, .6, 1.0)):
        col.set_linewidth(lw)
    ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    return img, (*ax.get_xlim(), *ax.get_ylim())


def summarize(res):
    st = res.state
    shares = np.stack([(st == s).mean(axis=1) for s in range(len(STATE_NAMES))], axis=1)
    d = np.hypot(*(res.track - res.lkp_road).transpose(2, 0, 1))/1000
    live = st != EXIT
    pct = np.full((len(st), 3), np.nan)
    for k in range(len(st)):
        if live[k].any():
            pct[k] = np.percentile(d[k, live[k]], [10, 50, 90])
    return shares, pct


def snapshot(net, cfg, view, out: Path):
    res = simulate_vehicle(net, cfg, view["lkp"], view["heading"], np.random.default_rng(view["seed"]))
    shares, pct = summarize(res)
    tmin = np.arange(len(res.state))*cfg.save_dt_s/60
    times = [5, 15, 30, 60]
    fig = plt.figure(figsize=(20, 11.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
    for j, tm in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        background(ax, net)
        k = min(int(round(tm*60/cfg.save_dt_s)), len(res.state) - 1)
        for s_, c in STATE_COLORS.items():
            m = res.state[k] == s_
            ax.scatter(res.track[k, m, 0]/1000, res.track[k, m, 1]/1000, s=3, c=c, lw=0, zorder=5)
        ax.plot(*(res.lkp_road/1000), marker="*", ms=14, mfc="yellow", mec="black", zorder=6)
        ax.set_title(f"{tm}분 뒤", fontsize=12)
    ax = fig.add_subplot(gs[1, :2])
    ax.stackplot(tmin, shares.T, colors=[STATE_COLORS[s] for s in range(len(STATE_NAMES))], labels=STATE_NAMES, alpha=.85)
    ax.set_xlim(0, tmin[-1]); ax.set_ylim(0, 1); ax.set_xlabel("시간 (분)"); ax.set_ylabel("particle 비율")
    ax.set_title("상태별 비율"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1), fontsize=9)
    ax = fig.add_subplot(gs[1, 2:])
    ax.fill_between(tmin, pct[:, 0], pct[:, 2], color="#4a6fa5", alpha=.25, label="10–90%")
    ax.plot(tmin, pct[:, 1], color="#4a6fa5", lw=2, label="중앙값")
    ax.set_xlim(0, tmin[-1]); ax.set_xlabel("시간 (분)"); ax.set_ylabel("LKP에서 직선거리 (km)")
    ax.set_title("LKP에서 거리 (지도 밖 제외)"); ax.legend()
    fig.suptitle(f"차량 도주 particle {cfg.particles}개, LKP ({view['lkp'][0]/1000:.1f}, {view['lkp'][1]/1000:.1f}) km, "
                 f"방향 {view['heading']:.0f}°, 기본 설정", fontsize=14)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return res, shares, pct


class Tuner:
    def __init__(self, net, cfg, view):
        self.net, self.cfg, self.view = net, cfg, view
        self.fig = plt.figure(figsize=(19, 11))
        self.ax_map = self.fig.add_axes([0.03, 0.30, 0.50, 0.66])
        self.ax_share = self.fig.add_axes([0.58, 0.72, 0.33, 0.24])
        self.ax_dist = self.fig.add_axes([0.58, 0.45, 0.33, 0.20])
        self.ax_speed = self.fig.add_axes([0.58, 0.30, 0.33, 0.10])
        img, ext = map_image(net)
        self.ax_map.imshow(img, extent=ext, origin="upper", interpolation="antialiased")
        a = net.terr.axis/1000
        self.ax_map.set_xlim(a[0], a[-1]); self.ax_map.set_ylim(a[0], a[-1]); self.ax_map.set_aspect("equal")
        self.ax_map.set_xlabel("동쪽 (km)"); self.ax_map.set_ylabel("북쪽 (km)")
        self.colors = np.array([to_rgba(STATE_COLORS[s]) for s in range(len(STATE_NAMES))])
        self.scat = self.ax_map.scatter([], [], s=4, lw=0, zorder=5)
        self.lkp_mark, = self.ax_map.plot([], [], marker="*", ms=15, mfc="yellow", mec="black", zorder=6)
        self.arrow = None
        self.ax_map.legend(handles=[Line2D([], [], marker="o", ls="", color=STATE_COLORS[s], label=STATE_NAMES[s])
                                    for s in range(len(STATE_NAMES))], loc="upper left", fontsize=8, framealpha=.9)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        ax_t = self.fig.add_axes([0.06, 0.255, 0.45, 0.025])
        self.s_time = Slider(ax_t, "시간 (분)", 0, cfg.duration_s/60, valinit=min(30, cfg.duration_s/60), valstep=1/6)
        self.s_time.drawon = False                                 # redrawn by blitting in draw_time
        self.s_time.on_changed(lambda _: self.draw_time())
        self.speed_bins = np.arange(0, 102.5, 2.5)
        self.stairs = {s_: self.ax_speed.stairs(np.zeros(len(self.speed_bins) - 1), self.speed_bins, color=STATE_COLORS[s_],
                                                lw=1.8) for s_ in (FAST, SLOW, OFF)}
        self.ax_speed.set_yticks([]); self.ax_speed.set_xlabel("속도 (km/h)", fontsize=8)
        self.speed_title = self.ax_speed.set_title("", fontsize=9)
        self.bg, self.anim = None, []
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)
        self.param_timer = self.fig.canvas.new_timer(interval=400)  # run once the slider stops moving
        self.param_timer.single_shot = True
        self.param_timer.add_callback(self.run)

        self.radio = RadioButtons(self.fig.add_axes([0.01, 0.03, 0.09, 0.15]), list(GROUPS), active=0)
        self.radio.on_clicked(self.show_group)
        self.sliders, self.slider_axes = {}, {}
        for g, items in GROUPS.items():
            for idx, (name, label, lo, hi, step) in enumerate(items):
                col, row = divmod(idx, 4)
                ax = self.fig.add_axes([0.20 + col*0.36, 0.19 - row*0.045, 0.22, 0.025])
                val = self.value_of(name)
                s = Slider(ax, label, lo, hi, valinit=float(np.clip(val, lo, hi)), valstep=step)
                s.on_changed(lambda _, n=name: self.on_param(n))
                self.sliders[name], self.slider_axes[name] = s, (g, ax)
        self.checks = CheckButtons(self.fig.add_axes([0.89, 0.06, 0.10, 0.12]), ["자동 실행", "일방통행 지킴"],
                                   [True, cfg.respect_oneway])
        self.checks.on_clicked(self.on_check)
        self.b_run = Button(self.fig.add_axes([0.92, 0.20, 0.07, 0.04]), "실행")
        self.b_run.on_clicked(lambda _: self.run())
        self.b_save = Button(self.fig.add_axes([0.92, 0.25, 0.07, 0.04]), "저장")
        self.b_save.on_clicked(lambda _: self.save())
        self.b_play = Button(self.fig.add_axes([0.545, 0.25, 0.035, 0.035]), "재생")
        self.b_play.on_clicked(lambda _: self.toggle_play())
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.status = self.fig.text(0.58, 0.985, "", fontsize=10)
        self.show_group(list(GROUPS)[0])
        self.run()

    def value_of(self, name):
        if name == "heading":
            return self.view["heading"]
        if name == "seed":
            return self.view["seed"]
        if name == "duration_min":
            return self.cfg.duration_s/60
        return getattr(self.cfg, name)

    def show_group(self, group):
        for name, (g, ax) in self.slider_axes.items():
            ax.set_visible(g == group)
        self.fig.canvas.draw_idle()

    def read_sliders(self):
        kw = {}
        for name, s in self.sliders.items():
            v = s.val
            if name == "heading":
                self.view["heading"] = float(v)
            elif name == "seed":
                self.view["seed"] = int(v)
            elif name == "duration_min":
                kw["duration_s"] = float(v)*60
            elif name == "particles":
                kw["particles"] = int(v)
            else:
                kw[name] = float(v)
        on = self.checks.get_status()
        kw["respect_oneway"] = on[1]
        self.cfg = replace(self.cfg, **kw)

    def on_param(self, name):
        if self.checks.get_status()[0]:
            self.param_timer.stop()
            self.param_timer.start()

    def on_check(self, label):
        if label != "자동 실행" and self.checks.get_status()[0]:
            self.run()

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if ev.inaxes is self.ax_map and ev.button == 1 and (tb is None or not tb.mode):
            self.view["lkp"] = [ev.xdata*1000, ev.ydata*1000]
            self.run()

    def run(self):
        self.read_sliders()
        self.status.set_text("실행 중 ...")
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        self.res = simulate_vehicle(self.net, self.cfg, self.view["lkp"], self.view["heading"],
                                    np.random.default_rng(self.view["seed"]))
        self.shares, self.pct = summarize(self.res)
        self.tmin = np.arange(len(self.res.state))*self.cfg.save_dt_s/60
        self.s_time.valmax = self.cfg.duration_s/60
        self.s_time.ax.set_xlim(0, self.s_time.valmax)
        lk = self.res.lkp_road/1000
        self.lkp_mark.set_data([lk[0]], [lk[1]])
        if self.arrow is not None:
            self.arrow.remove()
        h = np.deg2rad(self.view["heading"])
        self.arrow = self.ax_map.annotate("", xy=(lk[0] + 4*np.cos(h), lk[1] + 4*np.sin(h)), xytext=lk,
                                          arrowprops=dict(arrowstyle="->", lw=2, color="black"), zorder=6)
        ax = self.ax_share
        ax.clear()
        ax.stackplot(self.tmin, self.shares.T, colors=[STATE_COLORS[s] for s in range(len(STATE_NAMES))], alpha=.85)
        ax.set_xlim(0, self.tmin[-1]); ax.set_ylim(0, 1); ax.set_title("상태별 비율", fontsize=10)
        self.vl_share = ax.axvline(0, color="red", lw=1)
        ax = self.ax_dist
        ax.clear()
        ax.fill_between(self.tmin, self.pct[:, 0], self.pct[:, 2], color="#4a6fa5", alpha=.25)
        ax.plot(self.tmin, self.pct[:, 1], color="#4a6fa5", lw=2)
        ax.set_xlim(0, self.tmin[-1]); ax.set_title("LKP에서 거리 km (중앙값, 10–90%)", fontsize=10)
        self.vl_dist = ax.axvline(0, color="red", lw=1)
        # speed histograms of every stored time, as particle shares per bin
        n = self.cfg.particles
        sp = self.res.speed_kmh.astype(float)
        idx = np.clip(np.digitize(sp, self.speed_bins) - 1, 0, len(self.speed_bins) - 2)
        self.hist = {}
        for s_ in (FAST, SLOW, OFF):
            m = self.res.state == s_
            h = np.zeros((len(sp), len(self.speed_bins) - 1))
            rows = np.broadcast_to(np.arange(len(sp))[:, None], sp.shape)
            np.add.at(h, (rows[m], idx[m]), 1.0/n)
            self.hist[s_] = h
        top = max(h[1:].max() for h in self.hist.values())          # t = 0 holds every particle at 0 km/h
        self.ax_speed.set_ylim(0, max(top, 1e-3)*1.05)
        self.ax_speed.set_xlim(0, self.cfg.speed_cap_kmh + 5)
        sl = self.s_time
        self.anim = [self.scat, self.vl_share, self.vl_dist, *self.stairs.values(), self.speed_title,
                     sl.poly, sl.valtext, *([sl._handle] if hasattr(sl, "_handle") else [])]
        for art in self.anim:
            art.set_animated(True)
        self.status.set_text(f"particle {self.cfg.particles}개, LKP ({lk[0]:.1f}, {lk[1]:.1f}) km, "
                             f"방향 {self.view['heading']:.0f}°, seed {self.view['seed']}")
        self.set_time_artists()
        self.bg = None
        self.fig.canvas.draw_idle()

    def set_time_artists(self):
        k = min(int(round(self.s_time.val*60/self.cfg.save_dt_s)), len(self.res.state) - 1)
        st = self.res.state[k]
        self.scat.set_offsets(self.res.track[k]/1000)
        self.scat.set_facecolor(self.colors[st])
        tm = self.s_time.val
        self.vl_share.set_xdata([tm, tm]); self.vl_dist.set_xdata([tm, tm])
        for s_, art in self.stairs.items():
            art.set_data(self.hist[s_][k])
        self.speed_title.set_text(f"{tm:.1f}분 속도 분포: " + ", ".join(
            f"{STATE_NAMES[s]} {(st == s).mean():.0%}" for s in range(len(STATE_NAMES))))

    def on_draw(self, _ev):
        """After a full redraw: keep the static picture and draw the moving artists on top."""
        c = self.fig.canvas
        self.bg = c.copy_from_bbox(self.fig.bbox)
        for art in self.anim:
            self.fig.draw_artist(art)

    def draw_time(self):
        self.set_time_artists()
        c = self.fig.canvas
        if self.bg is None or not getattr(c, "supports_blit", False):
            c.draw_idle()
            return
        c.restore_region(self.bg)
        for art in self.anim:
            self.fig.draw_artist(art)
        c.blit(self.fig.bbox)
        c.flush_events()

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("정지" if self.playing else "재생")
        (self.timer.start if self.playing else self.timer.stop)()
        self.fig.canvas.draw_idle()

    def tick(self):
        s = self.s_time
        nxt = s.val + 0.5                                       # 30 s of simulated time per frame
        s.set_val(s.valmin if nxt > s.valmax + 1e-9 else nxt)

    def on_key(self, ev):
        s = self.s_time
        if ev.key in ("right", "left"):
            s.set_val(float(np.clip(s.val + (1 if ev.key == "right" else -1), s.valmin, s.valmax)))
        elif ev.key == " ":
            self.toggle_play()

    def save(self):
        PARAMS.write_text(json.dumps({"config": asdict(self.cfg), "view": self.view}, ensure_ascii=False, indent=1))
        self.status.set_text(f"저장: {PARAMS}")
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="chungju")
    ap.add_argument("--snapshot", type=Path, help="Write a figure with the saved/default settings and exit")
    args = ap.parse_args()
    net = load_net(args.region, 25.0)
    cfg, view = load_settings()
    if args.snapshot:
        matplotlib.use("Agg")
        snapshot(net, replace(cfg, particles=max(cfg.particles, 2000)), view, args.snapshot)
        print(f"Saved {args.snapshot}")
        return
    tuner = Tuner(net, cfg, view)   # keep a reference: matplotlib holds widget callbacks only weakly
    plt.show()
    return tuner


if __name__ == "__main__":
    main()
