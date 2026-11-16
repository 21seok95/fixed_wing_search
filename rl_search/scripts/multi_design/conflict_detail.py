"""Explanatory figure for the multi-UAV design talk (docs/multi_agent_design.md): how the predicted-conflict penalty is computed.
Kinematics only (not the RL env): 33 m/s, heading change at most 49 deg per 5 s decision, decisions every 5 s, positions every 1 s.

    python rl_search/scripts/multi_design/conflict_detail.py rl_search/data/real/amboy/rl_eval/report
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(sys.argv[1])
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#d9d8d3", "#fcfcfb"
V, WMAX, DMIN, LOOK, STEP = 33.0, np.radians(49.0/5.0), 100.0, 20.0, 5.0


def vel(psi):
    return V*np.array([np.cos(psi), np.sin(psi)])


def cpa(r, w):
    """Straight flight of both: (seconds to the closest approach, distance then). w = 0: never closer."""
    w2 = float(w @ w)
    if w2 < 1e-9:
        return np.inf, float(np.linalg.norm(r))
    t = -float(r @ w)/w2
    return t, float(np.linalg.norm(r + w*max(t, 0.0)))


def conflict(r, w, c=1.0):
    t, d = cpa(r, w)
    return c*(1 - t/LOOK)*(1 - d/DMIN) if (np.isfinite(t) and 0 < t <= LOOK and d < DMIN) else 0.0


def encounter(avoid: bool, gap=2000.0, T=40.0, dt=0.1):
    """Head-on: me at (0,0) going north, the other at (0, gap) going south. avoid: at each decision where the penalty of the
    state now is > 0, both turn right with the largest command for that 5 s step; otherwise straight."""
    p = [np.zeros(2), np.array([0.0, gap])]
    psi = [np.pi/2, -np.pi/2]
    t, rows, dist = 0.0, [], []
    while t < T - 1e-9:
        r, w = p[1] - p[0], vel(psi[1]) - vel(psi[0])
        cmd = -1.0 if (avoid and conflict(r, w) > 0) else 0.0          # -1 = right turn for both (each in its own frame)
        for _ in range(int(STEP/dt)):
            for i in (0, 1):
                psi[i] += cmd*WMAX*dt
                p[i] = p[i] + vel(psi[i])*dt
            t += dt
            if abs(t - round(t)) < dt/2:
                dist.append((round(t), float(np.linalg.norm(p[1] - p[0]))))
        r, w = p[1] - p[0], vel(psi[1]) - vel(psi[0])
        tt, dd = cpa(r, w)
        inside = min(d for s, d in dist if s > round(t) - STEP) < DMIN
        rows.append(dict(t=round(t), cmd=cmd, dist=float(np.linalg.norm(r)), tcpa=tt, dcpa=dd, pen=conflict(r, w), inside=inside))
    return rows, np.array(dist)


fig, axs = plt.subplots(1, 3, figsize=(21, 6.6), facecolor=BG, gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.15]))
# (가) the geometry, seen from me (I stand still, the other moves with the relative velocity)
ax = axs[0]
r0, psi_o = np.array([343.0, 420.0]), np.radians(200.0)          # closest approach 40 m after 10 s
w = vel(psi_o) - vel(np.pi/2)
tc, dc = cpa(r0, w)
line = np.array([r0 + w*s for s in np.linspace(0, 1.25*tc, 50)])
ax.plot(line[:, 0], line[:, 1], ls="--", color=MUTED, lw=1.2)
ax.add_patch(Circle((0, 0), DMIN, fill=False, ec="#7a1f1f", lw=1.4))
ax.plot(0, 0, marker=(3, 0, 0), ms=11, color=INK)
ax.annotate("", xy=(0, 260), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(25, 200, "내 속도\n(33 m/s, 북쪽)", fontsize=8.5, color=INK)
ax.plot(*r0, "o", color="#2a78d6", ms=8)
ax.annotate("", xy=r0 + vel(psi_o)*8, xytext=r0, arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.6))
ax.text(r0[0] - 40, r0[1] + 45, "상대: 지금 위치 r, 속도", fontsize=8.5, color="#2a78d6")
pc = r0 + w*tc
ax.plot(*pc, "o", color="#d6336c", ms=7)
ax.plot([0, pc[0]], [0, pc[1]], color="#d6336c", lw=1.4)
ax.text(pc[0] + 20, pc[1] - 70, f"가장 가까워지는 점\nt = {tc:.1f}초 뒤, d = {dc:.0f} m", fontsize=9, color="#d6336c")
ax.text(-280, 640, "점선: 나를 멈춰 놓고 본 상대의 움직임\n(상대 속도 − 내 속도 방향으로 곧게)", fontsize=8.5, color=MUTED)
ax.text(-90, -150, "d_min 100 m", fontsize=8.5, color="#7a1f1f")
ax.set_xlim(-300, 750); ax.set_ylim(-300, 750); ax.set_aspect("equal")
ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8); [s.set_color(GRID) for s in ax.spines.values()]
pen0 = conflict(r0, w)
ax.set_title(f"(가) 5초마다 계산하는 두 값 (기체 쌍마다)\n벌점 = c × (1 − t/20) × (1 − d/100) = c × {1 - tc/LOOK:.2f} × {1 - dc/DMIN:.2f} = {pen0:.2f}c",
             loc="left", fontsize=10, color=INK)
ax.set_xlabel("m (나 기준, 기수가 위)", fontsize=8.5, color=MUTED)
# (나) a head-on encounter, with and without one avoiding turn
ax = axs[1]
cases = [("그대로 직진", False, "#7a1f1f"), ("벌점이 처음 걸릴 때 둘 다 오른쪽으로 한 번", True, "#2a78d6")]
summ = []
for name, avoid, col in cases:
    rows, dist = encounter(avoid)
    ax.plot(dist[:, 0], dist[:, 1], color=col, lw=2, label=name)
    tot = sum(r["pen"] for r in rows)
    viol = sum(r["inside"] for r in rows)
    summ.append((name, rows, tot, viol, dist[:, 1].min()))
ax.axhline(DMIN, color="#7a1f1f", lw=1, ls=":")
ax.text(1, DMIN + 30, "d_min 100 m", fontsize=8.5, color="#7a1f1f")
ax.set_xlabel("시간 (초), 정면 2 km에서 시작, 결정은 5초마다", fontsize=8.5, color=MUTED)
ax.set_ylabel("두 기체 사이 거리 (m)", fontsize=8.5, color=MUTED)
ax.set_ylim(0, 2100); ax.set_xlim(0, 40)
ax.grid(color=GRID, lw=.6); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8); [s.set_color(GRID) for s in ax.spines.values()]
ax2 = ax.twinx()
for k, (name, rows, tot, viol, dmin) in enumerate(summ):
    ts = np.array([r["t"] for r in rows]); pens = np.array([r["pen"] for r in rows])
    ax2.bar(ts - 1.0 + 2.0*k, pens, width=1.8, color=cases[k][2], alpha=.35)
ax2.set_ylim(0, 1.6); ax2.set_ylabel("5초마다 받는 예측 충돌 벌점 (× c), 막대", fontsize=8.5, color=MUTED); ax2.tick_params(colors=MUTED, labelsize=8)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.set_title("(나) 정면으로 마주 올 때: 선 = 거리, 막대 = 그 5초 비행이 끝난 뒤 받는 벌점", loc="left", fontsize=10, color=INK)
# (다) numbers
ax = axs[2]
ax.axis("off")
lines = ["(다) 숫자로 보기", "", "정면 2 km에서 시작 (위 (나)):"]
for name, rows, tot, viol, dmin in summ:
    lines.append(f"  {name}")
    lines.append(f"    예측 벌점 합 {tot:.2f}c, 100 m 안 들어간 5초 구간 {viol}개, 최근접 {dmin:.0f} m")
lines += ["    직진: " + ", ".join(f"{r['t']}초 {r['pen']:.2f}c" for r in summ[0][1] if r["pen"] > 0)]
ex = [("정면 1 km", np.array([0, 1000.0]), np.radians(-90)), ("정면 0.5 km", np.array([0, 500.0]), np.radians(-90)),
      ("정면 1 km, 옆으로 150 m", np.array([150.0, 1000.0]), np.radians(-90)),
      ("직각 (오른쪽 앞 0.5 km씩, 서쪽으로)", np.array([500.0, 500.0]), np.radians(180)),
      ("나란히 150 m (같은 방향)", np.array([150.0, 0.0]), np.radians(90)),
      ("멀어지는 중 (뒤 0.3 km, 남쪽으로)", np.array([0, -300.0]), np.radians(-90))]
lines += ["", "다른 경우 (나는 북쪽으로 33 m/s):", "  상황 → t, d → 벌점"]
for name, r, ps in ex:
    t, d = cpa(r, vel(ps) - vel(np.pi/2))
    tt = "다가오지 않음" if not np.isfinite(t) else ("멀어짐" if t <= 0 else f"{t:.1f}초, {d:.0f} m")
    lines.append(f"  {name} → {tt} → {conflict(r, vel(ps) - vel(np.pi/2)):.2f}c")
lines += ["", "규칙:", "  · 0 < t ≤ 20초 이고 d < 100 m 일 때만 벌점",
          "  · 5초 비행이 끝난 뒤의 위치·방향으로 계산 → 꺾은 결정이 바로 그 보상에 반영",
          "  · 실제 거리가 100 m 안이면 그 5초마다 큰 벌점 P 따로",
          "  · 쌍의 두 기체 모두 받음, 여러 기체면 쌍마다 더함"]
ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9.5, color=INK, linespacing=1.55, transform=ax.transAxes)
fig.suptitle("충돌 벌점 계산 자세히 (설명용, 두 기체 33 m/s, 바람 없음)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT/"ma_conflict_detail.png", dpi=95, facecolor=BG)
print("saved", OUT/"ma_conflict_detail.png")
for name, rows, tot, viol, dmin in summ:
    print(name, round(tot, 3), viol, round(dmin), [(r["t"], r["cmd"], round(r["dist"]), round(r["tcpa"], 1) if np.isfinite(r["tcpa"]) else None,
                                                     round(r["dcpa"]), round(r["pen"], 3)) for r in rows])
print("geometry", round(tc, 2), round(dc, 1), round(pen0, 3))
