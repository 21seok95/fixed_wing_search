"""Figure of the RL observation (rl_search.rl_real) at a moment saved by env_check replay: the whole map with the particles and the
four map extents, and every channel of the four heading-up maps.

    python -m rl_search.scripts.rl_eval.obs_figure rl_search/data/real/amboy/rl_eval/env_check/replay_s1k1_obs.npz 10min
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from matplotlib import font_manager as _fm
_NAMES = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = [n for n in ("AppleGothic", "Noto Sans CJK KR") if n in _NAMES][:1] + ["sans-serif"]   # Mac or Ubuntu Korean font
plt.rcParams["axes.unicode_minus"] = False
src, key = Path(sys.argv[1]), sys.argv[2]
out_dir = src.parent
d = np.load(src)
MAPS = (("m1600", 1.6, "1.6 km (칸 25 m), 예측 15초"), ("m6400", 6.4, "6.4 km (100 m), 예측 1분"),
        ("m25600", 25.6, "25.6 km (400 m), 예측 5분"), ("m51200", 51.2, "51.2 km (800 m), 예측 10분"))
CH = ("① POS 가치 (지금)", "② POS 가치 (예측)", "③ A 가치", "④ 땅 가시도", "⑤ 화면 탐지 세기")
CMAP = ("Blues", "Blues", "Greens", "Greys_r", "Purples")
INK, MUTED = "#1f1f1e", "#6b6b67"

t, pose, w = float(d[f"{key}__t"]), d[f"{key}__pose"], d[f"{key}__w"]
track, dt = d["track"], float(d["save_dt"])
k = min(int(round(t/dt)), len(track) - 1)
P = track[k]
fig = plt.figure(figsize=(21, 11.5), layout="constrained")
gs = fig.add_gridspec(4, 7, width_ratios=[2.6, .05, 1, 1, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0])
path = d["path"]
n_now = int(round((t - float(d["start__t"]))/5.0))
live = w > 0
sc = ax.scatter(P[live, 0]/1000, P[live, 1]/1000, c=w[live], s=2, cmap="Oranges", vmin=0, vmax=1, zorder=2)
ax.plot(path[:n_now + 1, 0]/1000, path[:n_now + 1, 1]/1000, color="#2a6db0", lw=1.2, zorder=3, label="지금까지 경로 (SAROPS 행동 재생)")
ax.plot(*np.asarray(d["lkp"])/1000, marker="*", ms=14, color="#f2c94c", mec=INK, zorder=4, label="LKP")
x, y, yaw = pose
c, s = np.cos(yaw), np.sin(yaw)
for (name, side, _), ls in zip(MAPS, ("-", "--", ":", "-.")):
    h = side*500
    corners = np.array([[h, -h], [h, h], [-h, h], [-h, -h]])        # (forward, right) corners
    wx = x + corners[:, 0]*c + corners[:, 1]*s
    wy = y + corners[:, 0]*s - corners[:, 1]*c
    ax.add_patch(Polygon(np.stack([wx, wy], 1)/1000, closed=True, fill=False, ec=MUTED, lw=1, ls=ls, zorder=1))
ax.annotate("", xy=((x + 1500*c)/1000, (y + 1500*s)/1000), xytext=(x/1000, y/1000),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5), zorder=5)
ax.plot(x/1000, y/1000, "o", color=INK, ms=5, zorder=5)
ax.set(xlim=(-30, 30), ylim=(-30, 30), aspect="equal", xlabel="x (km)", ylabel="y (km)",
       title=f"지도 전체 (t = {t/60:.1f}분): particle 색 = 남은 가중치, 사각형 = 지도 4장 범위")
ax.legend(loc="lower left", fontsize=9, frameon=False)
fig.colorbar(sc, ax=ax, shrink=.4, label="남은 가중치")
for i, (name, side, label) in enumerate(MAPS):
    o = d[f"{key}__{name}"]
    for j in range(5):
        a = fig.add_subplot(gs[i, 2 + j])
        a.set_xticks([]); a.set_yticks([])
        if j >= len(o):
            a.axis("off")
            continue
        img = o[j]
        if j == 3:
            im = a.imshow(np.where(img < 0, np.nan, img), cmap=CMAP[j], vmin=0, vmax=1)
            if (img < 0).any():
                a.imshow(np.where(img < 0, 1.0, np.nan), cmap="Reds", vmin=0, vmax=2, alpha=.6)
        else:
            im = a.imshow(img, cmap=CMAP[j], vmin=0, vmax=max(float(img.max()), 1e-6))
        a.plot(31.5, 31.5, marker="^", color="#c0392b", ms=6)
        if i == 0:
            a.set_title(CH[j], fontsize=11)
        if j == 0:
            a.set_ylabel(label, fontsize=10)
        a.text(1, 62, f"최대 {img.max():.2f}" if j != 3 else ("빨강 = 지형 밖" if (img < 0).any() else ""), fontsize=8, color=MUTED, va="bottom")
fig.suptitle(f"새 관측 (기체 중심, 진행 방향이 위, ▲ = 기체): {src.stem.replace('_obs', '')}, {key}.  ①②③은 log(1 + particle 단위 값)",
             fontsize=14)
out = out_dir/f"{src.stem.replace('_obs', '')}_{key}.png"
fig.savefig(out, dpi=80)
print("saved", out)
