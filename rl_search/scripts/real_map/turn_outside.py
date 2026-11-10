"""Why does the UAV turn so far outside the search rectangle? One rectangle flown on a copy of the plant, without and with the 510 m lead.

Figure: rl_search/data/real/chungju/turn_outside.png

    python -m rl_search.scripts.real_map.turn_outside
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rl_search.accordion import LANE_TOL, _DryEnv, lead_shift
from rl_search.patterns import parallel_pts
from rl_search.patterns_v2 import FramedFollower
from rl_search.real_scenario import BANK_DEG, make_env
from rl_search.real_terrain import DATA

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
ELL, WID = 3000.0, 570.0


def fly(lead: float, k: int):
    env = make_env(1, lead_m=lead, bank_deg=BANK_DEG)
    s, n = 190.0/k, int(round(WID/(190.0/k)))
    R = env.v/(env.turn/env.cfg.decision_dt_s)
    plan = parallel_pts(max(ELL - s, 100.0), n, s, R, LANE_TOL)
    pts = lead_shift(plan, lead) if lead > 0 else plan
    d = _DryEnv(env)
    u = (pts[1] - pts[0])/np.hypot(*(pts[1] - pts[0]))
    p0 = pts[0] - 400*u
    d.plant.reset(np.array([p0[0], p0[1], np.arctan2(u[1], u[0])]))
    d.chi_tgt = float(np.arctan2(u[1], u[0]))
    f = FramedFollower(d, pts, np.zeros(2), 0.0, 0.0, np.zeros(2))
    path, on = [], False
    while f.laps < 1 and d.t < 7200:
        a = f.action()
        on = on or f.started
        d.chi_tgt += float(np.clip(a, -1, 1))*d.turn
        hist = []
        d.plant._integrate(d.chi_tgt, d.plant.n_inner, d.v, out=hist)
        d.t += env.cfg.decision_dt_s
        if on:
            path += [h[:3] for h in hist[::10]]
    p = np.array(path)
    ox, oy = np.abs(p[:, 0]) - ELL/2, np.abs(p[:, 1]) - WID/2
    return dict(p=p, plan=plan, t=d.t/60, ox=float(ox.max()), oy=float(max(oy.max(), 0.0)), out=float(((ox > 0) | (oy > 0)).mean()), n=n)


def main():
    warnings.filterwarnings("ignore")
    res = {(L, k): fly(L, k) for L in (0.0, 510.0) for k in (1, 2, 3)}
    fig = plt.figure(figsize=(21, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .8], height_ratios=[1, 1])
    for j, L in enumerate((0.0, 510.0)):
        ax = fig.add_subplot(gs[:, j])
        r = res[(L, 2)]
        ax.add_patch(Rectangle((-ELL/2, -WID/2), ELL, WID, fill=True, fc="#e8f0f9", ec="black", lw=1.6, label="탐색 사각형"))
        P = r["plan"]
        for i in range(0, len(P) - 1, 2):
            ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color="#2a4d7a", ls="--", lw=1, label="계획한 레인" if i == 0 else None)
        ax.plot(r["p"][:, 0], r["p"][:, 1], color="#c0392b", lw=1.5, label="무인기 경로")
        ax.set_xlim(-ELL/2 - 1100, ELL/2 + 1100); ax.set_ylim(-1000, 1000); ax.set_aspect("equal")
        ax.set_title(f"앞섬 {L:.0f} m (겹침 2회): 사각형 밖 {r['ox']:.0f} m까지, 사각형 밖 시간 {r['out']:.0%}, 전체 {r['t']:.1f}분", fontsize=11)
        ax.set_xlabel("레인 방향 (m)"); ax.set_ylabel("레인 가로 (m)"); ax.legend(loc="upper left", fontsize=9)
    x = np.arange(3)
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x - .2, [res[(0.0, k)]["ox"] for k in (1, 2, 3)], .4, color="#7f8c8d", label="앞섬 0 m")
    ax.bar(x + .2, [res[(510.0, k)]["ox"] for k in (1, 2, 3)], .4, color="#c0392b", label="앞섬 510 m")
    ax.set_xticks(x); ax.set_xticklabels(["겹침 1회", "2회", "3회"]); ax.set_ylabel("m"); ax.legend(fontsize=9); ax.set_title("사각형 끝에서 밖으로 나가는 거리", fontsize=11)
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x - .2, [100*res[(0.0, k)]["out"] for k in (1, 2, 3)], .4, color="#7f8c8d")
    ax.bar(x + .2, [100*res[(510.0, k)]["out"] for k in (1, 2, 3)], .4, color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(["겹침 1회", "2회", "3회"]); ax.set_ylabel("%"); ax.set_title("레인 작업 중 기체가 사각형 밖에 있는 시간", fontsize=11)
    fig.suptitle(f"사각형 밖 선회의 원인: 3.0 × 0.57 km 사각형, 롤 {BANK_DEG:.0f}° (반경 192 m)", fontsize=13)
    p = DATA/"chungju"/"turn_outside.png"
    fig.savefig(p, dpi=95)
    print("Saved", p)


if __name__ == "__main__":
    main()
