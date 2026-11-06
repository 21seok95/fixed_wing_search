"""Ground POD map of one lane + U-turn + lane over Chungju with the new detection model.

Stationary ground points every 25 m; cumulative POD = 1 - exp(-sum lam0 * A * f(omega) * dt) over the
flight, for the 2-axis gimbal (look azimuth follows the heading) and the 3-axis option (azimuth held north).
Figure: rl_search/data/real/chungju/detection_check.png

Run from the repository root:
    python -m rl_search.scripts.real_map.detection_check
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon, Patch

from rl_search.detection import DetectionConfig, Detector, footprint, load_landcover
from rl_search.real_terrain import DATA
from rl_search.terrain import load_terrain

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
V, G = 33.0, 9.81
ROLL = 30.0
X0, Y0, LANE = -24500.0, -10500.0, 3000.0          # first lane goes north from (X0, Y0)


def flight():
    """1 s samples of a lane north, a left U-turn at ROLL, a lane south: x, y, yaw, roll, turn rate (deg/s)."""
    R = V**2/(G*np.tan(np.radians(ROLL))); w = V/R
    out = []
    t1 = LANE/V; t2 = np.pi/w
    for t in np.arange(0, 2*t1 + t2, 1.0):
        if t < t1:
            out.append((X0, Y0 + V*t, np.pi/2, 0.0, 0.0))
        elif t < t1 + t2:
            a = w*(t - t1)
            cx = X0 - R
            out.append((cx + R*np.cos(a), Y0 + LANE + R*np.sin(a), np.pi/2 + a, np.radians(ROLL), np.degrees(w)))
        else:
            out.append((X0 - 2*R, Y0 + LANE - V*(t - t1 - t2), -np.pi/2, 0.0, 0.0))
    return np.array(out), R


def main():
    terr = load_terrain(Path("rl_search/data/terrain_real_chungju_60km_25m.npz"))
    lc = load_landcover()
    path, R = flight()
    win = (X0 - 1400, X0 + 900, Y0 - 300, Y0 + LANE + 1100)
    a = terr.axis
    ci = np.flatnonzero((a >= win[0]) & (a <= win[1])); ri = np.flatnonzero((a >= win[2]) & (a <= win[3]))
    GX, GY = np.meshgrid(a[ci], a[ri])
    px, py = GX.ravel(), GY.ravel()
    res = {}
    for name, cfg in (("2축 (방위가 기체를 따라 돎)", DetectionConfig()),
                      ("3축 (방위를 북쪽에 고정)", DetectionConfig(gimbal="3axis"))):
        det = Detector(terr, lc, cfg)
        expo = np.zeros(len(px))
        dwell = np.zeros(len(px))
        for x, y, yaw, roll, om in path:
            p = det.slice_pod(px, py, x, y, yaw, roll, om, az_world=np.pi/2 if cfg.gimbal == "3axis" else None)
            expo += -np.log1p(-np.minimum(p, 1 - 1e-12))
            q = footprint(x, y, yaw, roll, cfg, az_world=np.pi/2 if cfg.gimbal == "3axis" else None)
            from rl_search.detection import inside_quad
            dwell += inside_quad(px, py, q)[0]*cfg.dt_s
        res[name] = (1 - np.exp(-expo)).reshape(GX.shape), dwell.reshape(GX.shape), det
    A = res[list(res)[0]][2].A[np.ix_(ri, ci)]

    ext = (a[ci[0]]/1000, a[ci[-1]]/1000, a[ri[0]]/1000, a[ri[-1]]/1000)
    fig = plt.figure(figsize=(21, 12.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, 0])
    levels = [0.05, 0.4, 0.5, 0.85, 1.0]
    cmapA = ListedColormap(["#1b5e20", "#66bb6a", "#d99aa5", "#c5d86d", "#f6f1e3"])
    ax.imshow(A, origin="lower", extent=ext, cmap=cmapA, norm=BoundaryNorm([0, .2, .45, .7, .95, 1.05], 5),
              interpolation="nearest")
    ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="#2a4d7a", lw=2)
    for k in (0, len(path)//4, int(LANE/V) + 6, len(path) - 1):
        q = footprint(*path[k, :4], DetectionConfig())[0]/1000
        ax.add_patch(Polygon(q, fill=False, ec="#c0392b", lw=1.5))
    ax.set_title("가시성 A + 비행 경로 + 2축 발자국 예시", fontsize=11)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ["#f6f1e3", "#c5d86d", "#d99aa5", "#66bb6a", "#1b5e20"],
        ["열린 땅·도로 1.0", "관목·초지 0.85", "마을 0.5", "숲 가장자리 0.4", "숲 안 0.05"])], loc="lower left", fontsize=8)
    for j, (name, (pod, dwell, det)) in enumerate(res.items()):
        ax = fig.add_subplot(gs[0, 1 + j])
        im = ax.imshow(pod, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="#4fc3f7", lw=1.2)
        ax.set_title(f"{name}\n정지 표적 누적 POD", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.7)
    ax = fig.add_subplot(gs[0, 3])
    d2, d3 = res[list(res)[0]][1], res[list(res)[1]][1]
    im = ax.imshow(d2, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=12, interpolation="nearest")
    ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="white", lw=1)
    ax.set_title("2축: 발자국 안에 머문 시간 (초)", fontsize=11); fig.colorbar(im, ax=ax, shrink=.7)
    for axx in fig.axes[:4]:
        if hasattr(axx, "set_xlabel") and axx.get_title():
            axx.set_xlabel("동쪽 (km)"); axx.set_ylabel("북쪽 (km)")

    # POD by visibility class inside the straight-lane swath (first lane, away from the turn)
    ax = fig.add_subplot(gs[1, :2])
    # ground fully crossed by a straight-lane footprint (within +-80 m of the lane; the near edge is 171 m wide)
    lane1 = (np.abs(GX - X0) <= 80) & (GY > Y0 + 520) & (GY < Y0 + LANE)
    lane2 = (np.abs(GX - (X0 - 2*R)) <= 80) & (GY > Y0 - 500) & (GY < Y0 + LANE - 800)
    first = lane1 | lane2
    cls = [(1.0, "열린 땅·도로"), (0.85, "관목·초지"), (0.5, "마을"), (0.4, "숲 가장자리"), (0.05, "숲 안")]
    cols = ["#2a4d7a", "#c0392b"]
    pod2, dwell2, _ = res[list(res)[0]]
    v = [pod2[first & np.isclose(A, val)].mean() for val, _ in cls]
    lam0 = DetectionConfig().lam0
    theory = [1 - np.exp(-lam0*val*dwell2[first].mean()) for val, _ in cls]
    xs = np.arange(len(cls))
    ax.bar(xs - .19, v, width=.36, color=cols[0], label="2축, 시뮬레이션 (지형 칸별 평균)")
    ax.bar(xs + .19, theory, width=.36, color="#bbbbbb", label=f"이론 1 − exp(−λ₀ A × {dwell2[first].mean():.1f}초), 화면 위치 배율 없음")
    for xx, vv in zip(xs - .19, v):
        ax.text(xx, vv + .02, f"{vv:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(cls)), [f"{n}\n(A {val})" for val, n in cls]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("평균 누적 POD"); ax.set_title("직선 레인 아래(레인 중심 ±80 m, 두 레인) 지형별 POD. 직선에서는 2축과 3축이 같다", fontsize=11)
    ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[1, 2:])
    turn = GY > Y0 + LANE - 100
    for j, (name, (pod, dwell, det)) in enumerate(res.items()):
        open_ = turn & (A >= .95) & (dwell > 0)
        h, b = np.histogram(dwell[open_], bins=np.arange(0, 40, 1))
        ax.step(b[:-1], h/h.sum(), where="post", color=cols[j], lw=2,
                label=f"{name}: 열린 땅 평균 POD {pod[open_].mean():.2f}, 머문 시간 중앙값 {np.median(dwell[open_]):.1f}초")
    ax.set_xlabel("U턴 구역에서 발자국 안에 머문 시간 (초)"); ax.set_ylabel("땅 칸 비율")
    ax.set_title(f"U턴 구역 (롤 {ROLL:.0f}°, 반경 {R:.0f} m)", fontsize=11); ax.legend(fontsize=9)
    fig.suptitle(f"충주 남서부 농경지, 33 m/s · 고도 500 m · 45° 아래 앞쪽 보기, λ₀ = {DetectionConfig().lam0}/s, 1초 단위 적분", fontsize=14)
    fig.savefig(DATA/"chungju"/"detection_check.png", dpi=105)
    plt.close(fig)

    for name, (pod, dwell, det) in res.items():
        print(name)
        for val, n in cls:
            m = first & np.isclose(A, val)
            if m.any():
                print(f"   straight swath {n}: POD {pod[m].mean():.3f}, dwell {dwell[m].mean():.1f}s, cells {m.sum()}")
        open_ = turn & (A >= .95) & (dwell > 0)
        print(f"   U-turn open land: POD {pod[open_].mean():.3f}, dwell median {np.median(dwell[open_]):.1f}s, "
              f"area seen {(turn & (dwell > 0)).sum()*terr.cell**2/1e6:.2f} km2")


if __name__ == "__main__":
    main()
