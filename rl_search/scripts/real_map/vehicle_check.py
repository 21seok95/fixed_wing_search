"""Checks of the vehicle evader model: which noise source slows the spread, the junction detour sweep,
and the spread with the default settings (cities avoided). Figures go to rl_search/data/real/chungju/.

Run from the repository root:
    python -m rl_search.scripts.real_map.vehicle_check
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.vehicle import EXIT, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

LKPS = {"A (5, 5) km": (5000.0, 5000.0), "B (−9.6, 15) km": (-9600.0, 15000.0), "C (−20, −15) km": (-20000.0, -15000.0)}
HEADING = 45.0
N = 2000


def dist_km(r):
    return np.hypot(*(r.track - r.lkp_road).transpose(2, 0, 1))/1000


def main():
    out = DATA/"chungju"
    net = load_net()
    base = VehicleConfig(particles=N)
    quiet = replace(base, p_leave_fast_min=0, p_leave_slow_min=0, p_stop_min=0, heading_std_deg=0,
                    reorient_std_deg_min=0, p_detour=0, fast_std_kmh=0, slow_std_kmh=0)
    tests = [("잡음 없음", {}), ("속도 퍼짐 ±5", dict(fast_std_kmh=5, slow_std_kmh=5)), ("정지 2%/분", dict(p_stop_min=.02)),
             ("논밭 이탈 5%/분", dict(p_leave_fast_min=.05, p_leave_slow_min=.05)), ("방향 퍼짐 25°", dict(heading_std_deg=25)),
             ("방향 흔들림 3°/√분", dict(reorient_std_deg_min=3)), ("교차로 다른 길 5%", dict(p_detour=.05)),
             ("모두 (기본값)", dict(fast_std_kmh=5, slow_std_kmh=5, p_stop_min=.02, p_leave_fast_min=.05,
                              p_leave_slow_min=.05, heading_std_deg=25, reorient_std_deg_min=3, p_detour=.05))]
    k60 = int(60*60/base.save_dt_s)
    abl = np.zeros((len(tests), len(LKPS), 3))
    for i, (_, kw) in enumerate(tests):
        for j, lkp in enumerate(LKPS.values()):
            d = dist_km(simulate_vehicle(net, replace(quiet, **kw), lkp, HEADING, np.random.default_rng(0)))[k60]
            abl[i, j] = np.percentile(d, [10, 50, 90])
    sweep_p = [0.0, 0.05, 0.1, 0.2, 0.3]
    curves = {}
    for p in sweep_p:
        for name, lkp in LKPS.items():
            r = simulate_vehicle(net, replace(base, p_detour=p), lkp, HEADING, np.random.default_rng(0))
            curves[(p, name)] = np.median(dist_km(r), axis=1)
    tmin = np.arange(base.n_steps//base.save_every + 1)*base.save_dt_s/60
    spread = {}
    for name, lkp in LKPS.items():
        r = simulate_vehicle(net, base, lkp, HEADING, np.random.default_rng(0))
        d = dist_km(r)
        d[r.state == EXIT] = np.nan
        spread[name] = (np.nanpercentile(d, [10, 50, 90], axis=1), (r.state == EXIT).mean(axis=1))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6.2), layout="constrained")
    ax = axs[0]
    y = np.arange(len(tests))
    cols = ["#2a6fdb", "#e07a1f", "#3a9d5d"]
    for j, (name, c) in enumerate(zip(LKPS, cols)):
        yy = y + (j - 1)*.26
        ax.errorbar(abl[:, j, 1], yy, xerr=[abl[:, j, 1] - abl[:, j, 0], abl[:, j, 2] - abl[:, j, 1]], fmt="o",
                    color=c, ms=6, capsize=3, label=f"LKP {name}")
    ax.set_yticks(y, [t for t, _ in tests]); ax.invert_yaxis()
    ax.set_xlabel("60분 뒤 LKP에서 직선거리 (km, 중앙값과 10–90%)")
    ax.set_title("잡음을 하나씩 켰을 때"); ax.legend(fontsize=8, loc="lower left")
    ax = axs[1]
    shades = plt.get_cmap("viridis")(np.linspace(0, .85, len(sweep_p)))
    for p, c in zip(sweep_p, shades):
        for name, ls in zip(LKPS, ("-", "--", ":")):
            ax.plot(tmin, curves[(p, name)], color=c, ls=ls, lw=1.8, label=f"{p:.0%}" if name == list(LKPS)[0] else None)
    ax.set_xlabel("시간 (분)"); ax.set_ylabel("LKP에서 직선거리 중앙값 (km)")
    ax.set_title("교차로에서 다른 길로 갈 확률 (선: A 실선, B 점선, C 점)")
    ax.legend(title="다른 길 확률", fontsize=8)
    ax = axs[2]
    for (name, (q, ex)), c in zip(spread.items(), cols):
        ax.fill_between(tmin, q[0], q[2], color=c, alpha=.15)
        ax.plot(tmin, q[1], color=c, lw=2, label=f"{name} (70분 지도 밖 {ex[-1]:.0%})")
    ax.set_xlabel("시간 (분)"); ax.set_ylabel("LKP에서 직선거리 (km, 중앙값과 10–90%)")
    ax.set_title(f"기본값 퍼짐, 큰 도시는 피함 (도시 도로 {net.city_share:.1%} 제외)"); ax.legend(fontsize=8)
    fig.suptitle(f"차량 도주 모델 검사 (particle {N}개, 방향 {HEADING:.0f}°, 25 m 격자)", fontsize=14)
    fig.savefig(out/"vehicle_check.png", dpi=115)
    plt.close(fig)

    print("| 잡음 | " + " | ".join(f"LKP {n} 60분 p10/50/90 (km)" for n in LKPS) + " |")
    print("|---|" + "---|"*len(LKPS))
    for (nm, _), row in zip(tests, abl):
        print(f"| {nm} | " + " | ".join(f"{a:.1f} / {b:.1f} / {c:.1f}" for a, b, c in row) + " |")
    k30 = int(30*60/base.save_dt_s)
    print(f"main network {net.main_share:.3f} of road length, city roads {net.city_share:.3f}")
    for name, (q, ex) in spread.items():
        print(f"{name}: 30분 p10/50/90 {q[0, k30]:.1f}/{q[1, k30]:.1f}/{q[2, k30]:.1f} km, "
              f"60분 {q[0, k60]:.1f}/{q[1, k60]:.1f}/{q[2, k60]:.1f} km, 70분 지도 밖 {ex[-1]:.2f}")
    for p in sweep_p:
        print(f"다른 길 {p:.0%}: " + ", ".join(f"{n} 60분 {curves[(p, n)][k60]:.1f} km" for n in LKPS))


if __name__ == "__main__":
    main()
