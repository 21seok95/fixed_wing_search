"""POD inside the oblique camera's footprint: where in the trapezoid it is high, and what one pass adds across the lane.

Straight level flight, forward 45 deg view, 500 m above flat ground, full detection model (image-position factor,
resolution factor, canopy path for A < 1, per-pass 0.7 x model). Figure: rl_search/data/real/chungju/fov_pod_map.png

    python -m rl_search.scripts.real_map.fov_pod_map
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.detection import DetectionConfig, camera_frame, image_coords
from rl_search.real_terrain import DATA

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
V = 33.0
LANDS = [("열린 땅 / 도로", 1.0, "#2a4d7a"), ("마을", 0.5, "#e67e22"), ("숲 가장자리", 0.4, "#27ae60"), ("숲 안", 0.05, "#8d6e63")]


def factors(cfg, px, py, x=0.0, y=0.0, frame=None):
    """Image-position and resolution factors of ground points for a UAV at (x, y) flying east; 0 outside the image."""
    fr = camera_frame(0.0, 0.0, cfg) if frame is None else frame
    uu, vv, front = image_coords(px, py, x, y, fr, cfg)
    uu, vv, front = uu[0], vv[0], front[0]
    ins = front & (np.abs(uu) <= 1) & (np.abs(vv) <= 1)
    slant = np.sqrt((px - x)**2 + (py - y)**2 + cfg.alt_m**2)
    return ins, np.where(ins, cfg.edge_factor(uu, vv), 0.0), np.where(ins, cfg.resolution_factor(slant), 0.0), slant, uu, vv


def pass_curve(cfg, A, resolution=True):
    """Model POD of one pass against the distance from the lane (m), 1 s samples averaged over the sampling phase."""
    ys = np.arange(-130.0, 130.1, 2.5)
    x0 = np.arange(0.0, V, 3.3)                 # target position along the lane: the phase of the 1 s samples
    X, Y = np.meshgrid(x0 + 2000.0, ys)
    px, py = X.ravel(), Y.ravel()
    fr = camera_frame(0.0, 0.0, cfg)
    expo = np.zeros(len(px))
    for t in np.arange(0.0, 120.0, cfg.dt_s):
        ins, g, res, slant, _, _ = factors(cfg, px, py, V*t, 0.0, fr)
        k = np.sin(np.radians(cfg.depression_deg))/(cfg.alt_m/slant)
        Ak = (A**k if A < 1 and cfg.oblique_occlusion else A)
        expo += cfg.lam0*g*(res if resolution else 1.0)*Ak*cfg.dt_s
    return ys, (1 - np.exp(-expo)).reshape(X.shape).mean(1)


def main():
    cfg = DetectionConfig()
    x = np.arange(380.0, 640.0, 1.0)
    y = np.arange(-125.0, 125.5, 1.0)
    GX, GY = np.meshgrid(x, y)
    ins, g, res, slant, uu, vv = factors(cfg, GX.ravel(), GY.ravel())
    g, res, ins = g.reshape(GX.shape), res.reshape(GX.shape), ins.reshape(GX.shape)
    p1 = 1 - np.exp(-cfg.lam0*g*res*cfg.dt_s)
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    fr = camera_frame(0.0, 0.0, cfg)
    f, r, u = fr[0][0], fr[1][0], fr[2][0]
    quad = []
    for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        d = f + sh*th*r + sv*tv*u
        s = cfg.alt_m/(-d[2])
        quad.append((s*d[0], s*d[1]))
    quad = np.array(quad + quad[:1])

    fig, axs = plt.subplots(2, 3, figsize=(21, 12.5), layout="constrained")
    panels = [(g, "화면 위치 배율 g(u, v)", "viridis", 0, 1), (res, "해상도 배율 (거리에 따른 차량 픽셀 수)", "viridis", 0, 1),
              (p1, "열린 땅에서 1초 POD = 1 − exp(−λ₀ · g · 해상도)", "magma", 0, .4)]
    for ax, (Z, title, cmap, lo, hi) in zip(axs[0], panels):
        im = ax.imshow(np.where(ins, Z, np.nan), origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.plot(quad[:, 0], quad[:, 1], color="black", lw=1.5)
        cs = ax.contour(GX, GY, np.where(ins, Z, np.nan), levels=np.linspace(lo, hi, 9)[1:-1], colors="white", linewidths=.7)
        ax.clabel(cs, fontsize=7, fmt="%.2f")
        ax.set_xlabel("기체 앞쪽 거리 (m)"); ax.set_ylabel("레인에서 옆으로 (m)"); ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.8)

    ax = axs[1, 0]
    us = np.linspace(-1, 1, 401)
    U, W = np.meshgrid(us, us)
    im = ax.imshow(cfg.edge_factor(U, W), origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=0, vmax=1)
    cs = ax.contour(U, W, cfg.edge_factor(U, W), levels=[.25, .5, .75, .99], colors="white", linewidths=.8)
    ax.clabel(cs, fontsize=8, fmt="%.2f")
    ax.set_xlabel("화면 가로 u"); ax.set_ylabel("화면 세로 v (−1 가까운 쪽, 1 먼 쪽)"); ax.set_aspect("equal")
    ax.set_title("카메라 화면에서 본 배율: 가운데 정사각형(|u|, |v| ≤ 0.5)이 1", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=.8)

    ax = axs[1, 1]
    rows = []
    for lab, A, c in LANDS:
        ys, p = pass_curve(cfg, A)
        ys0, p0 = pass_curve(cfg, A, resolution=False)
        dy = ys[1] - ys[0]
        rows.append((lab, p[len(ys)//2], .7*p[len(ys)//2], p.sum()*dy, .7*p.sum()*dy, p0.sum()*dy))
        ax.plot(ys, p, color=c, ls=":", lw=1.5)
        ax.plot(ys, .7*p, color=c, lw=2.4, label=f"{lab}: 중심 {.7*p[len(ys)//2]:.2f}")
    ax.step([-130, -95, 95, 130], [0, .7, 0, 0], where="post", color="black", lw=1.3, label="계획기 가정 (폭 190 m, 0.7)")
    ax.set_xlabel("레인에서 옆으로 떨어진 거리 (m)"); ax.set_ylabel("통과 1회 POD"); ax.set_ylim(0, 1); ax.legend(fontsize=9)
    ax.set_title("직선으로 한 번 지날 때 POD (실선 = 0.7 × 모델, 점선 = 모델)", fontsize=11)

    ax = axs[1, 2]
    xp = np.arange(len(rows))
    ax.bar(xp - .2, [r[3] for r in rows], .4, color="#7f8c8d", label="모델")
    ax.bar(xp + .2, [r[4] for r in rows], .4, color="#2a4d7a", label="0.7 × 모델 (지금 믿음)")
    ax.axhline(133, color="black", lw=1.2, ls="--", label="계획기 133 m")
    for i, r in enumerate(rows):
        ax.text(i - .2, r[3] + 2, f"{r[3]:.0f}", ha="center", fontsize=9); ax.text(i + .2, r[4] + 2, f"{r[4]:.0f}", ha="center", fontsize=9)
    ax.set_xticks(xp); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("실효 탐색 폭 (m)"); ax.legend(fontsize=9)
    ax.set_title("땅 종류별 실효 탐색 폭", fontsize=11)
    fig.suptitle(f"사다리꼴 발자국 안의 POD (고도 {cfg.alt_m:.0f} m, 앞쪽 45°, 가까운 변 {abs(quad[0, 1] - quad[3, 1]):.0f} m, 먼 변 {abs(quad[1, 1] - quad[2, 1]):.0f} m, 앞쪽 {quad[0, 0]:.0f}~{quad[1, 0]:.0f} m)", fontsize=14)
    out = DATA/"chungju"/"fov_pod_map.png"
    fig.savefig(out, dpi=95)

    def at(a, b):
        i = np.argmin(np.abs(x - a)); j = np.argmin(np.abs(y - b))
        return g[j, i], res[j, i], p1[j, i]
    print("quad", np.round(quad[:4], 1).tolist())
    for name, a, b in (("centre", 506, 0), ("near mid", 411, 0), ("far mid", 610, 0), ("side mid", 506, 92), ("near corner", 411, 84), ("far corner", 609, 102)):
        print(name, [round(float(v), 3) for v in at(a, b)])
    print("share of footprint: g=1 %.3f, 1 s POD >= 0.3 %.3f" % ((g[ins] >= .999).mean(), (p1[ins] >= .3).mean()))
    for r in rows:
        print(r[0], "centre model %.2f x0.7 %.2f | width model %.0f x0.7 %.0f (without resolution %.0f)" % r[1:])
    print("Saved", out)


if __name__ == "__main__":
    main()
