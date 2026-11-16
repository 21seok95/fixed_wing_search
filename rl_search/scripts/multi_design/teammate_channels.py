"""Explanatory figures for the multi-UAV design talk (not the RL env): how teammates would be drawn on the ego maps, and where the
predicted-conflict penalty would apply. Kinematics only: 33 m/s, heading change at most 49 deg per 5 s decision (9.8 deg/s).

    python rl_search/scripts/multi_design/teammate_channels.py rl_search/data/real/amboy/rl_eval/report
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#d9d8d3", "#fcfcfb"
V, WMAX, DMIN, LOOK = 33.0, np.radians(49.0/5.0), 100.0, 20.0
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#8a5cd1"}


def fly(p0, psi0, rates, dt):
    """Positions after each dt for a turn-rate sequence (rad/s), from p0 with heading psi0 (math angle)."""
    psi = psi0 + np.cumsum(rates*dt, axis=-1)
    x = p0[0] + np.cumsum(V*np.cos(psi)*dt, axis=-1)
    y = p0[1] + np.cumsum(V*np.sin(psi)*dt, axis=-1)
    return np.stack([x, y], -1), psi


def past_track(end, psi_end, rates_back, dt=1.0):
    """Track over the past len(rates_back) s (oldest first) that ends at ``end`` with heading ``psi_end``: (ages s, positions)."""
    pos, psi = fly(np.zeros(2), 0.0, np.asarray(rates_back, float), dt)
    pos = np.vstack([np.zeros(2), pos])
    psi = np.concatenate([[0.0], psi])
    rot = psi_end - psi[-1]
    c, s = np.cos(rot), np.sin(rot)
    pos = pos @ np.array([[c, s], [-s, c]])
    pos += end - pos[-1]
    ages = dt*np.arange(len(pos))[::-1]
    return ages, pos


def raster(L, xy, val):
    """64 x 64 map of side L (m) centred on the own UAV (heading up = +y): max of val per cell."""
    n, cell = 64, L/64
    img = np.zeros((n, n))
    j = np.floor((xy[:, 0] + L/2)/cell).astype(int)
    i = np.floor((xy[:, 1] + L/2)/cell).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    np.maximum.at(img, (i[ok], j[ok]), val[ok])
    return img


def reachable(p0, psi0, H, n=30000, seed=0, dt=0.25):
    """Points a UAV can be at within H s (any command every 5 s), with the earliest time each is reached."""
    rng = np.random.default_rng(seed)
    nd = int(np.ceil(H/5.0))
    a = rng.uniform(-1, 1, (n, nd))
    a[:8] = np.array([[-1], [-0.5], [0], [0.5], [1], [-1], [1], [0]])[:, :1].repeat(nd, 1)
    per = int(5.0/dt)
    rates = np.repeat(a*WMAX, per, axis=1)[:, :int(H/dt)]
    pos, _ = fly(p0, psi0, rates, dt)
    t = dt*np.arange(1, rates.shape[1] + 1)
    return pos.reshape(-1, 2), np.broadcast_to(t, rates.shape).reshape(-1)


# the scene: own UAV at the origin heading north (map up), three teammates
OWN, PSI_OWN = np.zeros(2), np.pi/2
deg = np.radians
team = {
    "A": dict(end=np.array([-350.0, 350.0]), psi=deg(-20), rates=np.zeros(600), note="왼쪽에서 비스듬히 다가옴", lab=(-11.5, -3.0)),
    "B": dict(end=np.array([2300.0, 400.0]), psi=deg(90), rates=np.r_[np.zeros(400), np.full(10, WMAX*0.9), np.zeros(190)],
              note="오른쪽에서 나란히 북쪽으로", lab=(4.0, -4.0)),
    "C": dict(end=np.array([-2000.0, 2500.0]), psi=deg(0), rates=np.r_[np.zeros(480), np.full(120, -WMAX)], note="지금 원 선회 중",
              lab=(-12.5, 3.2)),
    "D": dict(end=np.array([-7000.0, 8000.0]), psi=deg(180), rates=np.zeros(600), note="멀리서 서쪽으로", lab=(-12.5, 10.3)),
}
for k, m in team.items():
    m["ages"], m["track"] = past_track(m["end"], m["psi"], m["rates"])

MAPS = [(1600.0, 15.0, 15.0), (6400.0, 60.0, 30.0), (25600.0, 300.0, None), (51200.0, 600.0, None)]   # side, track window, reach horizon


def track_map(L, W):
    xy, val = [], []
    for m in team.values():
        keep = m["ages"] <= W
        xy.append(m["track"][keep]); val.append(1.0 - m["ages"][keep]/W)
    return raster(L, np.vstack(xy), np.concatenate(val))


def reach_map(L, H):
    xy, val = [], []
    for k, m in team.items():
        if np.abs(m["end"]).max() > L/2 + V*H:
            continue
        p, t = reachable(m["end"], m["psi"], H, seed=ord(k))
        xy += [p, m["end"][None]]; val += [1.0 - t/H, np.ones(1)]
    return raster(L, np.vstack(xy), np.concatenate(val)) if xy else np.zeros((64, 64))


def cpa(p_rel, v_rel):
    w2 = float(v_rel @ v_rel)
    if w2 < 1e-9:
        return np.inf, float(np.linalg.norm(p_rel))
    t = -float(p_rel @ v_rel)/w2
    return t, float(np.linalg.norm(p_rel + v_rel*max(t, 0.0)))


def vel(psi):
    return V*np.array([np.cos(psi), np.sin(psi)])


def style(ax, title, L):
    ax.set_facecolor(BG)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.plot(0, 0, marker=(3, 0, 0), ms=9, color=INK, zorder=5)
    ax.text(0.02, 0.02, f"한 변 {L/1000:g} km, 칸 {L/64:g} m", transform=ax.transAxes, fontsize=8, color=MUTED)


def show(ax, img, L, cmap, title):
    ax.imshow(img, origin="lower", extent=[-L/2, L/2, -L/2, L/2], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    style(ax, title, L)


fig, axs = plt.subplots(2, 4, figsize=(21, 11), facecolor=BG)
# (a) the scene
ax = axs[0, 0]
ax.set_facecolor(BG)
for k, m in team.items():
    tr = m["track"]
    ax.plot(tr[:, 0]/1000, tr[:, 1]/1000, color=COL[k], lw=1.3)
    ax.plot(*(m["end"]/1000), "o", color=COL[k], ms=5, zorder=4)
    ax.annotate(f"동료 {k}: {m['note']}", xy=m["end"]/1000, xytext=m["lab"], fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))
ax.plot([0, 0], [-6, 0], color=INK, lw=1.3)
ax.plot(0, 0, marker=(3, 0, 0), ms=10, color=INK, zorder=5)
ax.annotate("나 (기수가 위)", xy=(0, 0), xytext=(1.0, -8.0), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
for L, ls in ((1600, "-"), (6400, "--"), (25600, ":")):
    ax.add_patch(Rectangle((-L/2000, -L/2000), L/1000, L/1000, fill=False, ls=ls, lw=1, ec=MUTED))
ax.set_xlim(-13.5, 13.5); ax.set_ylim(-13.5, 13.5); ax.set_aspect("equal")
ax.set_title("(가) 예시 장면: 지난 10분 경로와 지금 위치 (km)\n지도 범위: 실선 1.6 km · 파선 6.4 km · 점선 25.6 km", loc="left", fontsize=10, color=INK)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ax.spines.values():
    s.set_color(GRID)
# the base design (user 2026-11-16): one tail channel per map (now and the past, no future drawn); reach_map stays only as an option
show(axs[0, 1], track_map(1600, 15), 1600, "Blues", "(나) 1.6 km 지도 · 꼬리 15초\nA: 왼쪽 위에서 오는 중 (밝은 끝 = 지금)")
show(axs[0, 2], track_map(6400, 60), 6400, "Blues", "(다) 6.4 km 지도 · 꼬리 1분\nB는 나란한 선, C는 고리(원 선회 중)")
show(axs[0, 3], track_map(25600, 300), 25600, "Blues", "(라) 25.6 km 지도 · 꼬리 5분\n이동은 선, C의 원 선회는 한두 칸에 모임")
show(axs[1, 0], track_map(51200, 600), 51200, "Blues", "(마) 51.2 km 지도 · 꼬리 10분")
show(axs[1, 1], track_map(102400, 600), 102400, "Blues", "(바) 102.4 km 지도 · 꼬리 10분")
ax = axs[1, 3]
ax.axis("off")
ax.text(0.0, 1.0, "\n".join(["(아) 꼬리 하나에서 읽히는 것", "", "· 지금 위치: 가장 밝은 칸",
                             "· 가는 방향: 흐린 쪽 → 밝은 쪽", "· 최근 선회: 꼬리가 휜 정도",
                             "· 원 선회 중: 고리, 또는 한두 칸에 모임", "· 속도: 모두 33 m/s라 꼬리 길이가 일정",
                             "", "그리지 않는 것", "· 앞으로의 경로(알 수 없음)", "· 아직 뜨지 않은 기체",
                             "· 지난 탐색 자체(이미 공유 믿음에 들어 있음)"]),
        va="top", fontsize=10, color=INK, linespacing=1.6, transform=ax.transAxes)
# (사) the vector part
ax = axs[1, 2]
ax.axis("off")
rows = []
for k, m in team.items():
    t, d = cpa(m["end"] - OWN, vel(m["psi"]) - vel(PSI_OWN))
    rows.append((k, np.linalg.norm(m["end"] - OWN), t, d))
lines = ["(사) 벡터에 더하는 값", "", f"동료 수: {len(team)}", "", "지금 이대로 곧게 날면 가장 가까워지는 거리와 그때까지 시간:"]
for k, dist, t, d in rows:
    when = "같은 속도로 나란히, 가까워지지 않음" if not np.isfinite(t) else ("멀어지는 중" if t <= 0 else f"{t:.1f}초 뒤 {d:.0f} m")
    if np.isfinite(t) and t > LOOK:
        when += " (20초 밖, 해당 없음)"
    lines.append(f"  동료 {k}: 지금 {dist/1000:.2f} km → {when}")
worst = min((r for r in rows if np.isfinite(r[2]) and 0 < r[2] <= LOOK), key=lambda r: r[3], default=None)
lines += ["", "가장 위험한 기체 하나만 넣음:"]
if worst:
    lines.append(f"  동료 {worst[0]}: 지금 {worst[1]:.0f} m, 최근접 {worst[3]:.0f} m, {worst[2]:.1f}초 뒤")
lines += ["", "지도 가운데 삼각형 = 나 (채널에는 그리지 않음)"]
ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9.5, color=INK, linespacing=1.6, transform=ax.transAxes)
fig.suptitle("동료를 기체 중심 지도에 그리는 법: 지도마다 꼬리 채널 하나 (지금 위치 + 지난 경로, 설명용 예시, 기수가 위)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT/"ma_teammate_channels.png", dpi=95, facecolor=BG)
print("saved", OUT/"ma_teammate_channels.png")
for k, dist, t, d in rows:
    print(k, round(dist), round(t, 1) if np.isfinite(t) else None, round(d))

# ---- figure 2: where the predicted-conflict penalty applies
def penalty(rel, v_rel):
    t, d = cpa(rel, v_rel)
    if np.linalg.norm(rel) < DMIN:
        return np.nan                     # inside d_min: the large penalty instead
    if not np.isfinite(t) or t <= 0 or t > LOOK or d >= DMIN:
        return 0.0
    return (1 - t/LOOK)*(1 - d/DMIN)


g = np.linspace(-1500, 1500, 241)
cases = [("정면 (상대가 남쪽으로)", deg(-90)), ("직각 (상대가 서쪽으로)", deg(180)), ("나란히 (상대도 북쪽으로)", deg(90))]
fig, axs = plt.subplots(1, 4, figsize=(21, 5.6), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
for ax, (name, psi) in zip(axs[:3], cases):
    img = np.array([[penalty(np.array([x, y]), vel(psi) - vel(PSI_OWN)) for x in g] for y in g])
    ax.imshow(img, origin="lower", extent=[g[0]/1000, g[-1]/1000, g[0]/1000, g[-1]/1000], cmap="Reds", vmin=0, vmax=1)
    ax.add_patch(Circle((0, 0), DMIN/1000, color="#7a1f1f", zorder=4))
    ax.plot(0, 0, marker=(3, 0, 0), ms=9, color="white", zorder=5)
    ax.set_title(f"{name}\n색 = 예측 충돌 벌점 크기, 진한 원 = d_min 100 m 안(큰 벌점)", loc="left", fontsize=10, color=INK)
    ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlabel("상대의 지금 위치 (km, 나 기준, 기수 위)", fontsize=8.5, color=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
ax = axs[3]
ax.set_facecolor(BG)
dist = np.linspace(2000, 120, 400)
for off, col in ((0, "#7a1f1f"), (50, "#d6336c"), (90, "#f08ca0")):
    pen = [penalty(np.array([off, y]), vel(deg(-90)) - vel(PSI_OWN)) for y in dist]
    ax.plot(dist/1000, pen, color=col, lw=2, label=f"옆으로 {off} m 비켜서 정면 접근")
ax.axvline(LOOK*2*V/1000, color=MUTED, lw=1, ls="--")
ax.text(LOOK*2*V/1000 + 0.04, 0.45, "20초 × 66 m/s = 1.32 km:\n여기서부터 벌점", fontsize=8.5, color=MUTED, ha="right")
ax.invert_xaxis()
ax.set_xlabel("지금 거리 (km), 오른쪽으로 갈수록 가까워짐", fontsize=8.5, color=MUTED)
ax.set_ylabel("5초당 벌점 (× c)", fontsize=8.5, color=MUTED)
ax.set_title("정면으로 다가올 때 벌점이 커지는 모양\n(1 − 남은 시간/20초) × (1 − 최근접 거리/100 m)", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, lw=.6)
for s in ax.spines.values():
    s.set_color(GRID)
fig.suptitle("예측 충돌 벌점: 지금 거리가 아니라 '곧게 날면 가장 가까워질 거리'와 '그때까지 시간'으로 (설명용, 두 기체 33 m/s)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT/"ma_conflict_penalty.png", dpi=95, facecolor=BG)
print("saved", OUT/"ma_conflict_penalty.png")
