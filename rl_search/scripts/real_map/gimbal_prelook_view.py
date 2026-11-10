"""What gimbal pre-pointing does at a lane change, seen from above: camera footprints before, during and after the turn, with and
without pointing at the next lane's start. Flight on a copy of the plant with the real follower and gimbal (no particles).

Figure: rl_search/data/real/chungju/gimbal_prelook_view.png

    python -m rl_search.scripts.real_map.gimbal_prelook_view
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon, Rectangle

from rl_search.accordion import LANE_TOL, LeadFollower, _DryEnv, lead_shift
from rl_search.detection import Gimbal
from rl_search.patterns import parallel_pts
from rl_search.real_scenario import BANK_DEG, make_env
from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.web_optimize import footprints

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
ELL, WID, K, LEAD = 3000.0, 570.0, 2, 510.0
COL = {"lane": "#2a4d7a", "pre": "#e67e22", "turn": "#c0392b", "next": "#27ae60"}
NAMES = {"lane": "레인 직선 (앞쪽 45°)", "pre": "레인 끝 직전 (앞당김)", "turn": "레인 바꾸는 중", "next": "다음 레인 (앞쪽 45°)"}


def fly(prelook: float):
    env = make_env(1, lead_m=LEAD, prelook_s=prelook, bank_deg=BANK_DEG)
    s, n = 190.0/K, int(round(WID/(190.0/K)))
    R = env.v/(env.turn/env.cfg.decision_dt_s)
    plan = parallel_pts(max(ELL - s, 100.0), n, s, R, LANE_TOL)
    uav = lead_shift(plan, LEAD)
    d = _DryEnv(env)
    d.gimbal_target = None
    u = (uav[1] - uav[0])/np.hypot(*(uav[1] - uav[0]))
    p0 = uav[0] - 400*u
    d.plant.reset(np.array([p0[0], p0[1], np.arctan2(u[1], u[0])]))
    d.chi_tgt = float(np.arctan2(u[1], u[0]))
    f = LeadFollower(d, plan, uav, np.zeros(2), 0.0, 0.0, np.zeros(2), prelook_s=prelook)
    cfg, gim, rows = env.detector.cfg, None, []
    while f.laps < 1 and d.t < 7200:
        a = f.action()
        seg = int(np.clip(np.searchsorted(f.cum, f.s, side="right") - 1, 0, len(f.pts) - 2))
        tgt = d.gimbal_target
        d.chi_tgt += float(np.clip(a, -1, 1))*d.turn
        hist = []
        d.plant._integrate(d.chi_tgt, d.plant.n_inner, d.v, out=hist)
        for j in env.sample_idx:
            x, y, yaw, phi, _ = hist[j]
            if gim is None:
                gim = Gimbal(1, cfg, yaw0=float(yaw))
            los = gim.los_point(x, y, *tgt) if tgt is not None else gim.los_forward(yaw)
            frame, _ = gim.step(yaw, phi, los, 1.0)
            rows.append([d.t + (j + 1)*d.plant.h, x, y, yaw, phi, *frame[0][0], *frame[1][0], *frame[2][0], *(tgt if tgt is not None else (np.nan, np.nan)),
                         seg if f.started else -1, np.degrees(gim.th[0]), np.degrees(gim.gam[0])])
        d.t += env.cfg.decision_dt_s
    log = np.array(rows, float)
    return dict(log=log, quad=footprints(log, cfg), plan=plan, alt=cfg.alt_m)


def window(r, S):
    log = r["log"]
    seg, t = log[:, 16], log[:, 0]
    t_end, t_next = t[np.flatnonzero(seg == S)[0]], t[np.flatnonzero(seg == S + 1)[0]]
    m = (t >= t_end - 24) & (t <= t_next + 16)
    return np.flatnonzero(m), t_end, t_next


def phase(row, S):
    if row[16] == S:
        return "turn"
    if row[16] == S + 1:
        return "next"
    return "pre" if np.isfinite(row[14]) else "lane"


def draw(ax, r, S, title):
    log, Q, plan = r["log"], r["quad"], r["plan"]
    idx, t_end, t_next = window(r, S)
    ax.add_patch(Rectangle((-ELL/2, -WID/2), ELL, WID, fc="#eef3f9", ec="black", lw=1.6, zorder=0))
    for i in range(0, len(plan) - 1, 2):
        ax.plot(plan[i:i + 2, 0], plan[i:i + 2, 1], color="#8aa2bf", ls="--", lw=.8, zorder=1)
    for k in idx[::3]:
        c = COL[phase(log[k], S)]
        ax.add_patch(Polygon(Q[k], fc=c, ec=c, alpha=.32, lw=.8, zorder=2))
    ax.plot(log[idx, 1], log[idx, 2], color="black", lw=1.8, zorder=4)
    for k in idx[::6]:
        f = log[k, 5:8]
        s_ = r["alt"]/max(-f[2], 1e-3)
        ax.plot([log[k, 1], log[k, 1] + s_*f[0]], [log[k, 2], log[k, 2] + s_*f[1]], color="#777", lw=.6, zorder=3)
    for tm, lab in ((t_end - 8, "8초 전"), (t_end, "레인 끝"), (t_next, "다음 레인")):
        k = int(np.argmin(np.abs(log[:, 0] - tm)))
        ax.plot(log[k, 1], log[k, 2], "o", ms=7, mfc="white", mec="black", zorder=6)
        ax.annotate(lab, (log[k, 1], log[k, 2]), textcoords="offset points", xytext=(6, 7), fontsize=10, weight="bold", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1))
    tx, ty = plan[S + 1]
    ax.plot(tx, ty, "x", color="#00a651", ms=14, mew=3.5, zorder=8)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=12); ax.set_xlabel("레인 방향 (m)"); ax.set_ylabel("레인 가로 (m)")


def bounds(rs, S):
    pts = []
    for r in rs:
        idx, _, _ = window(r, S)
        pts += [r["quad"][idx].reshape(-1, 2), r["log"][idx, 1:3]]
    p = np.vstack(pts)
    lo, hi = p.min(0) - 90, p.max(0) + 90
    half = max(hi - lo)/2
    c = (lo + hi)/2
    return c[0] - half, c[0] + half, c[1] - half, c[1] + half


def main():
    warnings.filterwarnings("ignore")
    a, b = fly(0.0), fly(8.0)
    cases = [(1, "넓게 도는 레인 변경 (간격 475 m)"), (5, "옆 레인으로 가는 레인 변경 (간격 95 m, 고리)")]
    fig = plt.figure(figsize=(21, 14.5), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .85])
    for row, (S, name) in enumerate(cases):
        x0, x1, y0, y1 = bounds((a, b), S)
        for col, (r, t) in enumerate(((a, "앞당김 없음: 계속 앞쪽 45°"), (b, "짐벌 앞당김 8초: 초록 ×를 바라봄"))):
            ax = fig.add_subplot(gs[row, col])
            draw(ax, r, S, f"{name}\n{t}")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            if row == 0 and col == 0:
                ax.legend(handles=[Patch(fc=COL[k], alpha=.5, label=NAMES[k]) for k in COL], loc="lower left", fontsize=10)
        ax = fig.add_subplot(gs[row, 2])
        for r, ls, lab in ((a, ":", "앞당김 없음"), (b, "-", "앞당김 8초")):
            idx, t_end, t_next = window(r, S)
            t = r["log"][idx, 0] - t_end
            ax.plot(t, r["log"][idx, 17], color="#2a4d7a", ls=ls, lw=2, label=f"짐벌 피치 ({lab})")
            ax.plot(t, r["log"][idx, 18], color="#c0392b", ls=ls, lw=2, label=f"짐벌 롤 ({lab})")
            if ls == "-":
                ax.axvspan(-8, 0, color=COL["pre"], alpha=.15); ax.axvspan(0, t_next - t_end, color=COL["turn"], alpha=.1)
        for v, c in ((70, "#2a4d7a"), (20, "#2a4d7a"), (45, "#c0392b"), (-45, "#c0392b")):
            ax.axhline(v, color=c, lw=.8, ls="--")
        ax.set_xlabel("레인 끝을 기준으로 한 시각 (초)"); ax.set_ylabel("각도 (°)"); ax.legend(fontsize=9, loc="lower right")
        ax.set_title("짐벌 각도 (점선 가로줄 = 한계: 피치 20~70°, 롤 ±45°)", fontsize=12)
        idx, t_end, t_next = window(b, S)
        k = (b["log"][idx, 0] >= t_end) & (b["log"][idx, 0] < t_next)
        print(name, "lane change", round(t_next - t_end), "s; at pitch limit 70:", round(float((b["log"][idx, 17][k] >= 69.9).mean()), 2), "; roll at limit:", round(float((np.abs(b["log"][idx, 18][k]) >= 44.9).mean()), 2))
    fig.suptitle("짐벌 앞당김: 레인 끝 8초 전부터 레인을 바꾸는 동안 다음 레인 시작점(초록 ×)을 바라본다 · 사다리꼴 = 3초마다 카메라 화면 · 회색 선 = 기체에서 화면 중심", fontsize=14)
    p = DATA/"chungju"/"gimbal_prelook_view.png"
    fig.savefig(p, dpi=90)
    print("Saved", p)


if __name__ == "__main__":
    main()
