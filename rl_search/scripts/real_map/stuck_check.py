"""Particles that barely move in the last 20 minutes, with and without the off-road pull towards roads.
Figure: rl_search/data/real/chungju/vehicle_stuck.png

A particle counts as stuck when it is less than 500 m from where it was 20 minutes earlier and has not
left the map. "Off-road" stuck: >= 80 % of that time off-road; "road": <= 20 %.

Run from the repository root:
    python -m rl_search.scripts.real_map.stuck_check
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.vehicle_tuner import map_image
from rl_search.vehicle import EXIT, OFF, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LKPS = {"A (5, 5)": (5000.0, 5000.0), "B (−9.6, 15)": (-9600.0, 15000.0), "C (−20, −15)": (-20000.0, -15000.0)}
PULLS = (0.0, 1.0)


def stuck(r):
    T = r.track.shape[0] - 1
    k0 = T - int(20*60/r.cfg.save_dt_s)
    moved = np.hypot(*(r.track[T] - r.track[k0]).T)
    off = (r.state[k0:] == OFF).mean(axis=0)
    s = (moved < 500) & (r.state[T] != EXIT)
    return s & (off >= .8), s & (off <= .2), s


def main():
    net = load_net()
    base = VehicleConfig(particles=2000)
    res = {}
    for name, lkp in LKPS.items():
        for w in PULLS:
            res[(name, w)] = simulate_vehicle(net, replace(base, w_off_road=w), lkp, 45.0, np.random.default_rng(0))
    img, ext = map_image(net, 1600)
    fig = plt.figure(figsize=(19, 11), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    for j, w in enumerate(PULLS):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, extent=ext, origin="upper")
        for (name, lkp), c in zip(LKPS.items(), ("#2a6fdb", "#e07a1f", "#3a9d5d")):
            r = res[(name, w)]
            so, sr, _ = stuck(r)
            p = r.track[-1]/1000
            ax.scatter(p[so, 0], p[so, 1], s=16, marker="s", c=c, edgecolor="black", lw=.4)
            ax.scatter(p[sr, 0], p[sr, 1], s=22, marker="^", c=c, edgecolor="black", lw=.4)
            ax.plot(*(r.lkp_road/1000), marker="*", ms=13, mfc=c, mec="black")
        ax.set_title(f"논밭→도로 끌림 {w:g}: 멈춘 particle (■ 논밭, ▲ 도로, ★ LKP)")
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(LKPS))
    for j, (w, c) in enumerate(zip(PULLS, ("#bbbbbb", "#2a4d7a"))):
        v = [stuck(res[(n, w)])[2].mean() for n in LKPS]
        ax.barh(y + (j - .5)*.38, np.array(v)*100, height=.36, color=c, label=f"끌림 {w:g}")
        for yy, vv in zip(y + (j - .5)*.38, v):
            ax.text(vv*100 + .05, yy, f"{vv:.1%}", va="center", fontsize=9)
    ax.set_yticks(y, [f"LKP {n} km" for n in LKPS]); ax.invert_yaxis()
    ax.set_xlabel("마지막 20분 동안 500 m 안에 머문 비율 (%)")
    ax.set_title("멈춘 particle 비율 (도로 함정은 두 경우 모두 고친 뒤)"); ax.legend()
    ax = fig.add_subplot(gs[1, :])
    tmin = np.arange(base.n_steps//base.save_every + 1)*base.save_dt_s/60
    for (name, lkp), c in zip(LKPS.items(), ("#2a6fdb", "#e07a1f", "#3a9d5d")):
        for w, ls in zip(PULLS, (":", "-")):
            r = res[(name, w)]
            ax.plot(tmin, (r.state == OFF).mean(axis=1), color=c, ls=ls, lw=1.8, label=f"{name}, 끌림 {w:g}")
    ax.set_xlabel("시간 (분)"); ax.set_ylabel("논밭에 있는 비율"); ax.set_xlim(0, tmin[-1])
    ax.set_title("논밭에 있는 particle 비율: 끌림이 없으면 숲에 둘러싸인 논밭에 갇힌 차가 쌓인다"); ax.legend(ncol=3, fontsize=9)
    fig.suptitle(f"멈춘 particle 확인 (particle 2,000개, 방향 45°, 쓰는 도로 중 일방통행 기준 서로 오갈 수 있는 부분 {net.scc_share:.1%})",
                 fontsize=14)
    fig.savefig(DATA/"chungju"/"vehicle_stuck.png", dpi=110)
    plt.close(fig)
    for name in LKPS:
        print(name, " | ".join(f"끌림 {w:g}: 멈춤 {stuck(res[(name, w)])[2].mean():.3f} (논밭 {stuck(res[(name, w)])[0].sum()}, "
                               f"도로 {stuck(res[(name, w)])[1].sum()}), 논밭 시간 {np.mean(res[(name, w)].state[1:] == OFF):.3f}"
                               for w in PULLS))


if __name__ == "__main__":
    main()
