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
from matplotlib.colors import LightSource
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

# (field, label, min, max, step) per group; heading / particles / duration / seed are viewer settings
GROUPS = {
    "속도 (km/h)": [("speed_cap_kmh", "최대 속도", 20, 100, 1), ("fast_kmh", "빠른 길 평균", 10, 90, 1),
                  ("fast_std_kmh", "빠른 길 퍼짐", 0, 20, .5), ("slow_kmh", "느린 길 평균", 5, 60, 1),
                  ("slow_std_kmh", "느린 길 퍼짐", 0, 20, .5), ("off_kmh", "논밭 평균", 2, 40, .5),
                  ("off_std_kmh", "논밭 퍼짐", 0, 15, .5)],
    "전이 (분당)": [("p_leave_fast_min", "빠른 길→논밭", 0, .5, .005), ("p_leave_slow_min", "느린 길→논밭", 0, .5, .005),
                ("p_return_min", "논밭→도로 복귀", 0, 1, .01), ("return_radius_m", "복귀 거리 (m)", 25, 500, 25),
                ("p_stop_min", "정지", 0, .3, .005), ("p_go_min", "재출발", 0, 1, .01),
                ("off_max_slope_deg", "논밭 경사 한계 (°)", 3, 30, 1)],
    "경로·시작": [("heading", "도주 방향 (°)", 0, 359, 5), ("heading_std_deg", "방향 퍼짐 (°)", 0, 120, 1),
              ("reorient_std_deg_min", "방향 흔들림 (°/√분)", 0, 20, .5),
              ("p_detour", "교차로 다른 길 확률", 0, .5, .01), ("goal_sector_deg", "목적지 구간 (°)", 2, 30, 1),
              ("particles", "particle 수", 200, 5000, 100), ("duration_min", "시간 (분)", 10, 120, 5),
              ("seed", "seed", 0, 50, 1)],
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
        background(self.ax_map, net)
        self.scat = self.ax_map.scatter([], [], s=4, lw=0, zorder=5)
        self.lkp_mark, = self.ax_map.plot([], [], marker="*", ms=15, mfc="yellow", mec="black", zorder=6)
        self.arrow = None
        self.ax_map.legend(handles=[Line2D([], [], marker="o", ls="", color=STATE_COLORS[s], label=STATE_NAMES[s])
                                    for s in range(len(STATE_NAMES))], loc="upper left", fontsize=8, framealpha=.9)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        ax_t = self.fig.add_axes([0.06, 0.255, 0.45, 0.025])
        self.s_time = Slider(ax_t, "시간 (분)", 0, cfg.duration_s/60, valinit=min(30, cfg.duration_s/60), valstep=1/6)
        self.s_time.on_changed(lambda _: self.draw_time())

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
        self.b_play = Button(self.fig.add_axes([0.53, 0.25, 0.04, 0.035]), "재생")
        self.b_play.on_clicked(lambda _: self.toggle_play())
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.status = self.fig.text(0.58, 0.97, "", fontsize=10)
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
            self.run()

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
        self.status.set_text(f"particle {self.cfg.particles}개, LKP ({lk[0]:.1f}, {lk[1]:.1f}) km, "
                             f"방향 {self.view['heading']:.0f}°, seed {self.view['seed']}")
        self.draw_time()

    def draw_time(self):
        k = min(int(round(self.s_time.val*60/self.cfg.save_dt_s)), len(self.res.state) - 1)
        st = self.res.state[k]
        self.scat.set_offsets(self.res.track[k]/1000)
        self.scat.set_color([STATE_COLORS[s] for s in st])
        tm = self.s_time.val
        self.vl_share.set_xdata([tm, tm]); self.vl_dist.set_xdata([tm, tm])
        ax = self.ax_speed
        ax.clear()
        bins = np.arange(0, self.cfg.speed_cap_kmh + 5, 2.5)
        for s_ in (FAST, SLOW, OFF):
            m = st == s_
            if m.any():
                ax.hist(self.res.speed_kmh[k, m].astype(float), bins=bins, color=STATE_COLORS[s_], alpha=.7)
        ax.set_title(f"{tm:.1f}분 속도 분포 (km/h): " + ", ".join(
            f"{STATE_NAMES[s]} {(st == s).mean():.0%}" for s in range(len(STATE_NAMES))), fontsize=9)
        ax.set_yticks([])
        self.fig.canvas.draw_idle()

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
