"""Image-position factor of the detection model and the gimbal roll needed to look inside a turn.
Figure: rl_search/data/real/chungju/detection_edge.png

Run from the repository root:
    python -m rl_search.scripts.real_map.detection_edge
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.detection import DetectionConfig, camera_frame, footprint, image_coords
from rl_search.real_terrain import DATA

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
V, G = 33.0, 9.81


def lateral_range(cfg, A, with_edge=True, dt=0.2):
    """POD of one straight pass (north at 33 m/s) for ground points at each cross-track offset."""
    xs = np.arange(-125, 126, 2.5)
    ys = np.linspace(800, 1800, 41)                       # ground well inside the pass
    X, Y = np.meshgrid(xs, ys)
    px, py = X.ravel(), Y.ravel()
    expo = np.zeros(len(px))
    for t in np.arange(0, 2600/V, dt):
        frame = camera_frame(np.pi/2, 0.0, cfg)
        uu, vv, fr = image_coords(px, py, 0.0, V*t, frame, cfg)
        inside = fr[0] & (np.abs(uu[0]) <= 1) & (np.abs(vv[0]) <= 1)
        g = cfg.edge_factor(uu[0], vv[0]) if with_edge else 1.0
        expo += np.where(inside, cfg.lam0*A*g*dt, 0.0)
    return xs, (1 - np.exp(-expo)).reshape(X.shape).mean(axis=0)


def main():
    cfg = DetectionConfig()
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.6), layout="constrained")
    # (a) g on the ground footprint, UAV at the origin heading north
    ax = axs[0]
    X, Y = np.meshgrid(np.arange(-130, 131, 2.0), np.arange(380, 640, 2.0))
    uu, vv, fr = image_coords(X.ravel(), Y.ravel(), 0.0, 0.0, camera_frame(np.pi/2, 0.0, cfg), cfg)
    inside = fr[0] & (np.abs(uu[0]) <= 1) & (np.abs(vv[0]) <= 1)
    g = np.where(inside, cfg.edge_factor(uu[0], vv[0]), np.nan).reshape(X.shape)
    im = ax.pcolormesh(X, Y, g, cmap="viridis", vmin=.4, vmax=1, shading="auto")
    q = footprint(0, 0, np.pi/2, 0, cfg)[0]
    ax.plot(*np.vstack([q, q[:1]]).T, color="black", lw=1.2)
    ifov = np.radians(cfg.hfov_deg)/cfg.image_px
    for (yy, R, lab) in ((q[0, 1], 646, "가까운 변"), (q[1, 1], 790, "먼 변")):
        ax.text(0, yy + (-14 if lab == "가까운 변" else 6), f"{lab}: 사거리 {R} m, 표적 3 m ≈ {cfg.target_m/(R*ifov):.0f}픽셀",
                ha="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8, label="화면 위치 배율 g")
    ax.set_aspect("equal"); ax.set_xlabel("옆 방향 (m)"); ax.set_ylabel("기체에서 앞쪽 거리 (m)")
    ax.set_title(f"화면 위치 배율 g: 축마다 |좌표| ≤ {cfg.edge_plateau}에서 1, 가장자리 {cfg.edge_value}\n(모서리 {cfg.edge_value**2:.2f})", fontsize=11)
    # (b) lateral range curves
    ax = axs[1]
    for A, c, lab in ((1.0, "#2a4d7a", "열린 땅 A 1.0"), (0.4, "#2e7d32", "숲 가장자리 A 0.4"), (0.05, "#8d6e63", "숲 안 A 0.05")):
        xs, p1 = lateral_range(cfg, A, False)
        _, p2 = lateral_range(cfg, A, True)
        ax.plot(xs, p1, color=c, ls=":", lw=1.5)
        ax.plot(xs, p2, color=c, lw=2.2, label=f"{lab}: 중심 {p2[len(xs)//2]:.2f}, ±80 m {p2[np.argmin(abs(xs-80))]:.2f}")
    ax.axvspan(-85.5, 85.5, color="#eeeeee", zorder=0); ax.text(0, .03, "가까운 변 폭 171 m", ha="center", fontsize=8)
    ax.set_xlabel("레인 중심에서 옆 방향 거리 (m)"); ax.set_ylabel("한 번 지날 때 POD (정지 표적)")
    ax.set_ylim(0, 1.03); ax.legend(fontsize=8, loc="center", bbox_to_anchor=(.5, .45))
    ax.set_title("횡거리 곡선: 점선 = 배율 없음, 실선 = 화면 위치 배율 적용", fontsize=11)
    # (c) gimbal roll to keep looking at the centre of the orbit
    ax = axs[2]
    R = np.linspace(80, 1500, 300)
    phi = np.degrees(np.arctan(V**2/(G*R)))
    dep = np.degrees(np.arctan(cfg.alt_m/R))
    need = (90 - dep) + phi
    ax.plot(R, need, color="#c0392b", lw=2.2, label="선회 중심을 보는 데 필요한 짐벌 롤")
    ax.plot(R, phi, color="#2a4d7a", lw=1.6, ls="--", label="기체 롤 (앞쪽 45° 보기 때 보정량)")
    ax.plot(R, 90 - dep, color="#7f8c8d", lw=1.2, ls=":", label="수직에서 안쪽으로 꺾는 각 (90° − 내려보는 각)")
    ax.axhline(cfg.gimbal_roll_deg, color="black", lw=1); ax.text(1480, cfg.gimbal_roll_deg + 1.5, "짐벌 롤 한계 45° (지금 가정)", ha="right", fontsize=9)
    i = np.argmin(need)
    ax.plot(R[i], need[i], "o", color="#c0392b"); ax.annotate(f"최소 {need[i]:.0f}° (반경 {R[i]:.0f} m)", (R[i], need[i]),
                                                              xytext=(20, -25), textcoords="offset points", fontsize=9)
    ax.set_xlabel("선회 반경 (m)"); ax.set_ylabel("각도 (°)"); ax.set_ylim(0, 95)
    ax.set_title(f"롤·피치 짐벌로 선회 중심 응시 ({V:.0f} m/s, 고도 {cfg.alt_m:.0f} m)", fontsize=11); ax.legend(fontsize=8, loc="upper right")
    fig.savefig(DATA/"chungju"/"detection_edge.png", dpi=110)
    plt.close(fig)
    for A in (1.0, 0.4, 0.05):
        xs, p1 = lateral_range(cfg, A, False); _, p2 = lateral_range(cfg, A, True)
        sw1, sw2 = np.trapezoid(p1, xs), np.trapezoid(p2, xs)
        print(f"A {A}: centre {p1[len(xs)//2]:.3f}->{p2[len(xs)//2]:.3f}, 80 m {p1[np.argmin(abs(xs-80))]:.3f}->{p2[np.argmin(abs(xs-80))]:.3f}, "
              f"sweep width {sw1:.0f}->{sw2:.0f} m")
    for r_ in (200, 300, 500, 800):
        j = np.argmin(abs(R - r_)); print(f"orbit R {r_}: bank {phi[j]:.1f}, inward {90-dep[j]:.1f}, gimbal roll {need[j]:.1f}")


if __name__ == "__main__":
    main()
