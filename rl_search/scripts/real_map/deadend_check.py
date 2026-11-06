"""Dead-end branches of the road network and how much time the vehicle particles spend on them,
with and without the dead-end rule for junction detours. Figure: rl_search/data/real/chungju/vehicle_deadend.png

Run from the repository root:
    python -m rl_search.scripts.real_map.deadend_check
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from rl_search.real_terrain import DATA
from rl_search.vehicle import VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LKPS = {"A (5, 5)": (5000.0, 5000.0), "B (−9.6, 15)": (-9600.0, 15000.0), "C (−20, −15)": (-20000.0, -15000.0)}


def main():
    net = load_net()
    base = VehicleConfig(particles=2000)
    rows = []
    for name, lkp in LKPS.items():
        for av in (False, True):
            r = simulate_vehicle(net, replace(base, avoid_deadend=av), lkp, 45.0, np.random.default_rng(0))
            e = r.edge[1:]
            on = e >= 0
            dead = on & ~net.edge_core[np.maximum(e, 0)]
            # moving towards the core: distance along the edge to its core end shrinks; approximated by
            # whether the particle is on a dead-end edge at the next stored time as well (time spent there)
            d = np.hypot(*(r.track - r.lkp_road).transpose(2, 0, 1))[360]/1000
            rows.append((name, av, dead.sum()/on.sum(), np.percentile(d, [10, 50, 90])))

    fig, axs = plt.subplots(1, 2, figsize=(17, 7.6), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    segs = {"core": [], "dead": []}
    for k in np.flatnonzero(net.edge_main):
        segs["core" if net.edge_core[k] else "dead"].append(net.poly[net.off[k]:net.off[k + 1]]/1000)
    ax.add_collection(LineCollection(segs["core"], colors="#555555", linewidths=.45))
    ax.add_collection(LineCollection(segs["dead"], colors="#d1495b", linewidths=.6))
    a = net.terr.axis/1000
    ax.contourf(a, a, net.terr.water.astype(float), levels=[.5, 2], colors=["#8fb7ef"])
    ax.set_xlim(a[0], a[-1]); ax.set_ylim(a[0], a[-1]); ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    ax.set_title(f"막다른 가지 (빨강): 쓰는 도로망 길이의 {net.deadend_share:.1%}")
    ax.legend(handles=[Line2D([], [], color="#555555", lw=2, label="빠져나갈 길이 있는 도로"),
                       Line2D([], [], color="#d1495b", lw=2, label="막다른 가지 (되돌아 나와야 함)")], loc="lower left")
    ax = axs[1]
    y = np.arange(len(LKPS))
    for j, (av, c, lab) in enumerate(((False, "#bbbbbb", "다른 길 후보에 막다른 길 포함 (이전)"),
                                      (True, "#2a4d7a", "다른 길 후보에서 막다른 길 제외 (지금)"))):
        v = [r[2] for r in rows if r[1] == av]
        ax.barh(y + (j - .5)*.38, np.array(v)*100, height=.36, color=c, label=lab)
        for yy, vv, r in zip(y + (j - .5)*.38, v, [r for r in rows if r[1] == av]):
            ax.text(vv*100 + .15, yy, f"{vv:.1%}  (60분 중앙값 {r[3][1]:.1f} km)", va="center", fontsize=9)
    ax.set_yticks(y, [f"LKP {n} km" for n in LKPS]); ax.invert_yaxis()
    ax.set_xlim(0, 14); ax.set_xlabel("도로 위 시간 중 막다른 가지 위에 있던 비율 (%)")
    ax.set_title("particle 2,000개, 방향 45°, 70분"); ax.legend(loc="lower right", fontsize=9)
    fig.savefig(DATA/"chungju"/"vehicle_deadend.png", dpi=115)
    plt.close(fig)
    print(f"dead-end share of the main network: {net.deadend_share:.3f}")
    for name, av, share, q in rows:
        print(f"{name:14s} {'avoid' if av else 'old  '} dead-end time {share:.3f}, 60 min p10/50/90 {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} km")


if __name__ == "__main__":
    main()
