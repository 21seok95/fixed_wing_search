"""Gimbal steering on the predicted target distribution: one straight lane over Chungju vehicle particles.

Modes: fixed forward 45 deg; greedy steering every second to the reachable line of sight with the largest
expected detected mass (particles predicted 2 s ahead, remaining weights, per-pass cap 0.7); the same steering
without the resolution factor (to show why it is needed). Figure: rl_search/data/real/chungju/gimbal_steer.png

Run from the repository root:
    python -m rl_search.scripts.real_map.gimbal_steer_demo
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from rl_search.detection import DetectionConfig, Detector, Gimbal, GimbalConfig, PassCap, _rx, _rz, load_landcover
from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.vehicle_tuner import map_image
from rl_search.vehicle import VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
V = 33.0
LKP, HEADING = (5000.0, 5000.0), 45.0
T0 = 15*60.0                # UAV starts its lane 15 min after the LKP
LANE = 8000.0
N = 1500
GAMMAS = np.radians(np.arange(-45, 45.1, 7.5))
DEPS = np.radians(np.arange(20, 70.1, 10))


def positions(track, t, dt_save=10.0):
    k = t/dt_save
    k0 = int(np.clip(np.floor(k), 0, len(track) - 2))
    f = np.clip(k - k0, 0, 1)
    return (1 - f)*track[k0] + f*track[k0 + 1]


def frame_of(yaw, roll, gam, th):
    """Camera frame in the world for body gimbal angles (arrays of candidates, one UAV attitude)."""
    m = len(gam)
    f = np.stack([np.cos(th), np.zeros(m), -np.sin(th)], 1)
    r = np.stack([np.zeros(m), -np.ones(m), np.zeros(m)], 1)
    u = np.stack([np.sin(th), np.zeros(m), np.cos(th)], 1)
    to_world = np.einsum("ij,mjk->mik", _rz(np.array(yaw)) @ _rx(np.array(-roll)), _rx(gam))
    return tuple(np.einsum("mij,mj->mi", to_world, v) for v in (f, r, u))


def corners(x, y, frame, cfg):
    f, r, u = (a[0] for a in frame)
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    out = []
    for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        d = f + sh*th*r + sv*tv*u
        t = cfg.alt_m/max(-d[2], 1e-3)
        out.append((x + t*d[0], y + t*d[1]))
    return np.array(out)


def fly(det, det_true, track, path, mode):
    """``det`` plans the gimbal; ``det_true`` decides what is found."""
    cfg = det_true.cfg
    gim = Gimbal(1, cfg, GimbalConfig(), yaw0=path[0, 2])
    cap = PassCap(N, cfg)
    w = np.ones(N)
    cand = np.array([(g, t) for g in GAMMAS for t in DEPS])
    pos_t, polys, hist = [], [], []
    cur = None
    for k, (t, x, y, yaw) in enumerate(path):
        p_now = positions(track, t)
        if mode == "fixed":
            los = gim.los_forward(yaw)
        else:
            p_ahead = positions(track, t + 2.0)
            frames = frame_of(yaw, 0.0, cand[:, 0], cand[:, 1])
            score = np.zeros(len(cand))
            remain = -np.expm1(-np.maximum(cap.hmax - cap.acc, 0))
            for j in range(len(cand)):
                pj = det.slice_pod(p_ahead[:, 0], p_ahead[:, 1], x, y, yaw, 0.0, 0.0,
                                   frame=tuple(a[j:j + 1] for a in frames))
                score[j] = (w*np.minimum(pj, remain)).sum()
            best = int(np.argmax(score))
            if cur is None or score[best] > 1.15*score[cur] + 1e-9:
                cur = best
            fr = frame_of(yaw, 0.0, cand[cur:cur + 1, 0], cand[cur:cur + 1, 1])
            los = fr[0]
        frame, los_rate = gim.step(yaw, 0.0, los, 1.0)
        p = det_true.slice_pod(p_now[:, 0], p_now[:, 1], x, y, yaw, 0.0, los_rate[0], frame=frame)
        w *= 1 - cap.apply(p, det_true.last_inside)
        pos_t.append(1 - w.mean())
        polys.append(corners(x, y, frame, cfg))
        hist.append((t, np.degrees(gim.gam[0]), np.degrees(gim.th[0]), los_rate[0]))
    return np.array(pos_t), polys, np.array(hist), w


def main():
    net = load_net()
    res = simulate_vehicle(net, VehicleConfig(particles=N, duration_s=T0 + LANE/V + 60), LKP, HEADING,
                           np.random.default_rng(1))
    track = res.track.astype(float)
    lc = load_landcover()
    # lane through the centroid of the particles at mid-lane, along their main axis
    tm = T0 + LANE/V/2
    pm = positions(track, tm)
    c = pm.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pm - c).T))
    ax_ = evec[:, -1]
    yaw = np.arctan2(ax_[1], ax_[0])
    ts = T0 + np.arange(0, LANE/V, 1.0)
    s = (ts - tm)*V
    path = np.column_stack([ts, c[0] + s*np.cos(yaw), c[1] + s*np.sin(yaw), np.full(len(ts), yaw)])
    base = DetectionConfig()
    det_true = Detector(net.terr, lc, base)
    naive = Detector(net.terr, lc, replace(base, use_resolution=False, oblique_occlusion=False))
    runs = {"fixed": ("앞쪽 45° 고정", det_true, "fixed"),
            "steer": ("예측 분포로 짐벌 조향 (해상도·비스듬한 가림 반영)", det_true, "steer"),
            "steer_nores": ("조향, 계획이 해상도·가림 무시", naive, "steer")}
    out = {k: fly(det, det_true, track, path, m) for k, (_, det, m) in runs.items()}

    img, ext = map_image(net, 1600)
    fig = plt.figure(figsize=(22, 12.5), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    pad = 4500
    win = (c[0] - pad, c[0] + pad, c[1] - pad, c[1] + pad)
    cols = {"fixed": "#7f8c8d", "steer": "#2a4d7a", "steer_nores": "#e67e22"}
    for j, (k, (name, det, m)) in enumerate(runs.items()):
        pos_t, polys, hist, w = out[k]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, extent=ext, origin="upper", alpha=.55)
        ax.add_collection(PolyCollection([p/1000 for p in polys[::3]], facecolors=cols[k], edgecolors="none", alpha=.08))
        pe = positions(track, ts[-1])
        found = 1 - w
        sc = ax.scatter(pe[:, 0]/1000, pe[:, 1]/1000, c=found, cmap="inferno_r", vmin=0, vmax=.7, s=6, lw=0)
        ax.plot(path[:, 1]/1000, path[:, 2]/1000, color="#00bcd4", lw=2)
        ax.plot(path[0, 1]/1000, path[0, 2]/1000, ">", color="#00bcd4", ms=9)
        ax.set_xlim(win[0]/1000, win[1]/1000); ax.set_ylim(win[2]/1000, win[3]/1000); ax.set_aspect("equal")
        ax.set_title(f"{name}\n레인 끝 POS {pos_t[-1]:.3f} (색: particle별 발견 확률, 칠한 곳: 본 범위)", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
        if j == 2:
            fig.colorbar(sc, ax=ax, shrink=.7, label="발견 확률")
    ax = fig.add_subplot(gs[1, 0])
    for k, (name, _, _) in runs.items():
        ax.plot(out[k][0]*0 + (out[k][2][:, 0] - T0), out[k][0], color=cols[k], lw=2.2, label=f"{name}: {out[k][0][-1]:.3f}")
    ax.set_xlabel("레인 시작 후 시간 (초)"); ax.set_ylabel("누적 POS"); ax.legend(fontsize=9); ax.set_title("누적 POS", fontsize=11)
    ax = fig.add_subplot(gs[1, 1:])
    for k in ("steer", "steer_nores"):
        h = out[k][2]
        ax.plot(h[:, 0] - T0, h[:, 1], color=cols[k], lw=2, label=f"짐벌 롤: {runs[k][0]}")
        ax.plot(h[:, 0] - T0, h[:, 2], color=cols[k], lw=1.6, ls="--", label=f"내려보는 각: {runs[k][0]}")
    ax.axhline(45, color="#bbb", lw=.8); ax.axhline(-45, color="#bbb", lw=.8); ax.axhline(20, color="#bbb", lw=.8); ax.axhline(70, color="#bbb", lw=.8)
    ax.set_xlabel("레인 시작 후 시간 (초)"); ax.set_ylabel("각도 (°)"); ax.set_ylim(-50, 75)
    ax.legend(fontsize=8, ncol=2); ax.set_title("조향한 짐벌 각도 (롤 ±45°, 내려보기 20~70°, 시선 변경 40°/s)", fontsize=11)
    fig.suptitle(f"충주 차량 particle {N}개 (LKP {LKP[0]/1000:.0f}, {LKP[1]/1000:.0f} km, 방향 {HEADING:.0f}°), "
                 f"LKP 후 {T0/60:.0f}분부터 {LANE/1000:.0f} km 직선 레인, 33 m/s · 고도 500 m · 통과 상한 0.7\n"
                 "발견 판정은 모두 해상도·비스듬한 가림을 반영한 모델. 산에 의한 가림(지형 높이)은 아직 없음", fontsize=13)
    fig.savefig(DATA/"chungju"/"gimbal_steer.png", dpi=105)
    plt.close(fig)
    for k, (name, _, _) in runs.items():
        h = out[k][2]
        print(f"{k}: POS {out[k][0][-1]:.4f}, mean depression {h[:, 2].mean():.1f}, mean |roll| {np.abs(h[:, 1]).mean():.1f}, "
              f"time slewing >5 deg/s {np.mean(h[:, 3] > 5):.2f}")


if __name__ == "__main__":
    main()
