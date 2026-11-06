"""Lower POD defaults and the roll / pitch gimbal: forward 45 deg vs pointing at the next lane during a U-turn.
Figure: rl_search/data/real/chungju/gimbal_check.png

Run from the repository root:
    python -m rl_search.scripts.real_map.gimbal_check
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.detection import (DetectionConfig, Detector, Gimbal, GimbalConfig, PassCap, camera_frame, image_coords,
                                 load_landcover)
from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.detection_check import LANE, ROLL, V, X0, Y0
G = 9.81
ROLL_RATE = 45.0          # deg/s, roll-in and roll-out (PX4 FW_R_RMAX 56 deg/s with a 0.4 s time constant)


def flight(dt=0.05):
    """1 s samples of lane north -> left U-turn at ROLL with a rate-limited roll -> lane south.

    Returns rows (x, y, yaw, roll, turn rate deg/s, phase) with phase 0 lane 1, 1 turn, 2 lane 2, and the
    nominal turn radius.
    """
    x, y, yaw, roll, t = X0, Y0, np.pi/2, 0.0, 0.0
    phase, out, nxt = 0, [], 0.0
    yaw_end = np.pi/2 + np.pi
    while True:
        if phase == 0 and y >= Y0 + LANE:
            phase = 1
        cmd = np.radians(ROLL) if phase == 1 else 0.0
        if phase == 1:
            w = G*np.tan(roll)/V
            lead = (roll/np.radians(ROLL_RATE))*w/2          # heading still gained while rolling out
            if yaw >= yaw_end - lead:
                phase = 2
        roll += np.clip(cmd - roll, -np.radians(ROLL_RATE)*dt, np.radians(ROLL_RATE)*dt)
        w = G*np.tan(roll)/V
        yaw += w*dt
        x += V*np.cos(yaw)*dt; y += V*np.sin(yaw)*dt
        t += dt
        if t >= nxt - 1e-9:
            out.append((x, y, yaw, roll, np.degrees(w), phase)); nxt += 1.0
        if phase == 2 and y <= Y0 and abs(roll) < 1e-6:
            break
    R = V**2/(G*np.tan(np.radians(ROLL)))
    return np.array(out), R
from rl_search.terrain import load_terrain

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OLD = DetectionConfig(lam0=1.0, edge_plateau=0.6, edge_value=0.7)
NEW = DetectionConfig()


def lateral(cfg, A, dt=0.2):
    xs = np.arange(-125, 126, 2.5); ys = np.linspace(800, 1800, 41)
    X, Y = np.meshgrid(xs, ys); px, py = X.ravel(), Y.ravel()
    expo = np.zeros(len(px))
    fr = camera_frame(np.pi/2, 0.0, cfg)
    for t in np.arange(0, 2600/V, dt):
        uu, vv, front = image_coords(px, py, 0.0, V*t, fr, cfg)
        ins = front[0] & (np.abs(uu[0]) <= 1) & (np.abs(vv[0]) <= 1)
        expo += np.where(ins, cfg.lam0*A*cfg.edge_factor(uu[0], vv[0])*dt, 0)
    return xs, (1 - np.exp(-expo)).reshape(X.shape).mean(0)


def run(det, path, R, mode, gcfg, lead_s=0.0):
    """Cumulative ground POD on the detection_check window for one gimbal mode; also the gimbal history."""
    terr = det.terr
    a = terr.axis
    win = (X0 - 1400, X0 + 900, Y0 - 300, Y0 + LANE + 1100)
    ci = np.flatnonzero((a >= win[0]) & (a <= win[1])); ri = np.flatnonzero((a >= win[2]) & (a <= win[3]))
    GX, GY = np.meshgrid(a[ci], a[ri]); px, py = GX.ravel(), GY.ravel()
    gim = Gimbal(1, det.cfg, gcfg, yaw0=path[0, 2])
    cap = PassCap(len(px), det.cfg)
    target = (X0 - 2*R, Y0 + LANE - 350)                    # first part of the return lane (the blind gap)
    turn = np.flatnonzero(path[:, 5] == 1)
    t_turn0, t_turn1 = turn[0], turn[-1] + 1
    miss = np.ones(len(px)); hist = []
    for k, (x, y, yaw, roll, om, ph) in enumerate(path):
        t = float(k)
        point = mode == "prelook" and t_turn0 - lead_s <= t < t_turn1
        los = gim.los_point(x, y, *target) if point else gim.los_forward(yaw)
        frame, los_rate = gim.step(yaw, roll, los, det.cfg.dt_s)
        p = det.slice_pod(px, py, x, y, yaw, roll, max(abs(om), los_rate[0]), frame=frame)
        miss *= 1 - cap.apply(p, det.last_inside)
        hist.append((t, np.degrees(gim.gam[0]), np.degrees(gim.th[0]), los_rate[0], np.degrees(roll)))
    return (1 - miss).reshape(GX.shape), GX, GY, np.array(hist), target, (a[ci[0]], a[ci[-1]], a[ri[0]], a[ri[-1]])


def main():
    terr = load_terrain(Path("rl_search/data/terrain_real_chungju_60km_25m.npz"))
    lc = load_landcover()
    path, R = flight()
    det = Detector(terr, lc, NEW)
    gcfg = GimbalConfig()
    modes = {"forward": ("앞쪽 45° 유지", "forward", 0.0), "prelook": ("U턴 시작부터 다음 레인 지향", "prelook", 0.0),
             "lead": ("8초 앞당겨 다음 레인 지향", "prelook", 8.0)}
    res = {k: run(det, path, R, m, gcfg, lead) for k, (_, m, lead) in modes.items()}
    turn_t = np.flatnonzero(path[:, 5] == 1)
    t0, t1 = turn_t[0], turn_t[-1] + 1

    fig = plt.figure(figsize=(22, 12.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, 0])
    for A, c, lab in ((1.0, "#2a4d7a", "열린 땅"), (0.4, "#2e7d32", "숲 가장자리"), (0.05, "#8d6e63", "숲 안")):
        xs, po = lateral(OLD, A); _, pn = lateral(NEW, A)
        pc = np.minimum(pn, NEW.pass_pod_max)
        ax.plot(xs, po, color=c, ls=":", lw=1.3)
        ax.plot(xs, pn, color=c, ls="--", lw=1.3)
        ax.plot(xs, pc, color=c, lw=2.4, label=f"{lab}: 중심 {pc[len(xs)//2]:.2f}, ±70 m {pc[np.argmin(abs(xs-70))]:.2f}")
    ax.axhline(NEW.pass_pod_max, color="#c0392b", lw=1); ax.text(-120, NEW.pass_pod_max + .02, "통과 1회 상한 0.7", color="#c0392b", fontsize=9)
    ax.set_ylim(0, 1.03); ax.set_xlabel("레인 중심에서 옆 방향 (m)"); ax.set_ylabel("한 번 지날 때 POD")
    ax.set_title(f"점선: 이전 (λ₀ 1.0) · 파선: λ₀ {NEW.lam0}, 가장자리 {NEW.edge_value}\n실선: + 통과 1회 상한 {NEW.pass_pod_max}", fontsize=10)
    ax.legend(fontsize=8, loc="lower center")
    gap = (np.abs(res["forward"][1] - (X0 - 2*R)) <= 80) & (res["forward"][2] > Y0 + LANE - 700) & (res["forward"][2] < Y0 + LANE)
    for j, (k, (name, _, _)) in enumerate(modes.items()):
        pod, GX, GY, hist, target, ext = res[k]
        ax = fig.add_subplot(gs[0, 1 + j])
        im = ax.imshow(pod, origin="lower", extent=np.array(ext)/1000, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="#4fc3f7", lw=1.2)
        if k != "forward":
            ax.plot(target[0]/1000, target[1]/1000, "x", color="#4fc3f7", ms=10, mew=2)
        ax.set_title(f"{name}\n새 레인 첫 700 m 평균 POD {pod[gap].mean():.2f}", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
        if j == 2:
            fig.colorbar(im, ax=ax, shrink=.7, label="누적 POD")
    ax = fig.add_subplot(gs[1, :3])
    cols = {"forward": "#999999", "prelook": "#e67e22", "lead": "#2a4d7a"}
    for k, (name, _, _) in modes.items():
        h = res[k][3]
        ax.plot(h[:, 0], h[:, 1], color=cols[k], lw=2, label=f"짐벌 롤: {name}")
        ax.plot(h[:, 0], h[:, 2], color=cols[k], lw=1.6, ls="--", label=f"내려보는 각: {name}")
    h = res["forward"][3]
    ax.plot(h[:, 0], h[:, 4], color="black", lw=1, ls=":", label="기체 롤 (초당 45°로 기울임)")
    for v, lab in ((gcfg.roll_lim_deg, "롤 한계 45°"), (-gcfg.roll_lim_deg, ""), (gcfg.dep_min_deg, "내려보기 최소 20°"), (gcfg.dep_max_deg, "내려보기 최대 70°")):
        ax.axhline(v, color="#bbb", lw=.8)
        if lab:
            ax.text(t1 + 58, v + 1.2, lab, ha="right", fontsize=8, color="#777")
    ax.axvspan(t0, t1, color="#f6e7c1", zorder=0); ax.text((t0 + t1)/2, 76, "U턴", ha="center")
    ax.axvspan(t0 - 8, t0, color="#dfe9f5", zorder=0); ax.text(t0 - 4, 76, "앞당김", ha="center", fontsize=8)
    ax.set_xlim(t0 - 30, t1 + 60); ax.set_ylim(-55, 82); ax.set_xlabel("시간 (초)"); ax.set_ylabel("각도 (°)")
    ax.set_title(f"짐벌 각도: 자세 보정은 즉시(범위 안), 시선 변경은 {gcfg.slew_dps:.0f}°/s", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax = fig.add_subplot(gs[1, 3])
    turn = res["forward"][2] > Y0 + LANE - 100
    labels = ["새 레인\n첫 700 m", "U턴 구역", "그림 전체"]
    xs = np.arange(3)
    for j, (k, (name, _, _)) in enumerate(modes.items()):
        v = [res[k][0][gap].mean(), res[k][0][turn].mean(), res[k][0].mean()]
        ax.bar(xs + (j - 1)*.27, v, .25, color=cols[k], label=name)
        for xx, vv in zip(xs + (j - 1)*.27, v):
            ax.text(xx, vv + .01, f"{vv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, labels); ax.set_ylim(0, .8); ax.set_ylabel("평균 누적 POD (땅 칸)"); ax.legend(fontsize=8)
    ax.set_title("구역별 평균 POD", fontsize=11)
    fig.suptitle(f"충주 남서부: 레인 3 km → 롤 {ROLL:.0f}° U턴 (반경 {R:.0f} m) → 레인, 33 m/s, 고도 500 m, 2축 짐벌 "
                 f"(롤 ±{gcfg.roll_lim_deg:.0f}°, 내려보기 {gcfg.dep_min_deg:.0f}~{gcfg.dep_max_deg:.0f}°)", fontsize=14)
    fig.savefig(DATA/"chungju"/"gimbal_check.png", dpi=105)
    plt.close(fig)
    for A in (1.0, 0.4, 0.05):
        xs, po = lateral(OLD, A); _, pn = lateral(NEW, A)
        print(f"A {A}: centre {po[len(xs)//2]:.3f}->{pn[len(xs)//2]:.3f} (capped {min(pn[len(xs)//2], .7):.3f}), "
              f"sweep width capped {np.trapezoid(np.minimum(pn, .7), xs):.0f} m")
    for k in modes:
        h = res[k][3]
        print(k, "gap/turn/all", [round(float(res[k][0][m].mean()), 3) for m in (gap, turn, np.ones_like(gap))],
              "gimbal roll max", round(np.abs(h[:, 1]).max(), 1), "dep", round(h[:, 2].min(), 1), round(h[:, 2].max(), 1),
              "los rate max", round(h[:, 3].max(), 1))


if __name__ == "__main__":
    main()
