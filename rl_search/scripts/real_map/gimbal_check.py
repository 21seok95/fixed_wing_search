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

from rl_search.detection import DetectionConfig, Detector, Gimbal, GimbalConfig, camera_frame, image_coords, load_landcover
from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.detection_check import LANE, ROLL, V, X0, Y0, flight
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


def run(det, path, R, mode, gcfg):
    """Cumulative ground POD on the detection_check window for one gimbal mode; also the gimbal history."""
    terr = det.terr
    a = terr.axis
    win = (X0 - 1400, X0 + 900, Y0 - 300, Y0 + LANE + 1100)
    ci = np.flatnonzero((a >= win[0]) & (a <= win[1])); ri = np.flatnonzero((a >= win[2]) & (a <= win[3]))
    GX, GY = np.meshgrid(a[ci], a[ri]); px, py = GX.ravel(), GY.ravel()
    gim = Gimbal(1, det.cfg, gcfg)
    target = (X0 - 2*R, Y0 + LANE - 350)                    # first part of the return lane (the blind gap)
    t_turn0, t_turn1 = LANE/V, LANE/V + np.pi*R/V
    expo = np.zeros(len(px)); hist = []
    for k, (x, y, yaw, roll, om) in enumerate(path):
        t = float(k)
        los = gim.los_point(x, y, *target) if (mode == "prelook" and t_turn0 <= t < t_turn1) else gim.los_forward(yaw)
        frame, slew = gim.step(yaw, roll, los, det.cfg.dt_s)
        p = det.slice_pod(px, py, x, y, yaw, roll, om + slew[0], frame=frame)
        expo += -np.log1p(-np.minimum(p, 1 - 1e-12))
        hist.append((t, np.degrees(gim.gam[0]), np.degrees(gim.th[0]), slew[0], np.degrees(roll)))
    return (1 - np.exp(-expo)).reshape(GX.shape), GX, GY, np.array(hist), target, (a[ci[0]], a[ci[-1]], a[ri[0]], a[ri[-1]])


def main():
    terr = load_terrain(Path("rl_search/data/terrain_real_chungju_60km_25m.npz"))
    lc = load_landcover()
    path, R = flight()
    det = Detector(terr, lc, NEW)
    gcfg = GimbalConfig()
    res = {m: run(det, path, R, m, gcfg) for m in ("forward", "prelook")}

    fig = plt.figure(figsize=(21, 12), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, 0])
    for A, c, lab in ((1.0, "#2a4d7a", "열린 땅"), (0.4, "#2e7d32", "숲 가장자리"), (0.05, "#8d6e63", "숲 안")):
        xs, po = lateral(OLD, A); _, pn = lateral(NEW, A)
        ax.plot(xs, po, color=c, ls=":", lw=1.5)
        ax.plot(xs, pn, color=c, lw=2.2, label=f"{lab}: 중심 {pn[len(xs)//2]:.2f}, ±70 m {pn[np.argmin(abs(xs-70))]:.2f}")
    ax.set_ylim(0, 1.03); ax.set_xlabel("레인 중심에서 옆 방향 (m)"); ax.set_ylabel("한 번 지날 때 POD")
    ax.set_title(f"POD 낮춤: 점선 이전 (λ₀ {OLD.lam0}, 가장자리 {OLD.edge_value})\n실선 새 기본값 (λ₀ {NEW.lam0}, |좌표| ≤ {NEW.edge_plateau}에서 1, 가장자리 {NEW.edge_value})", fontsize=10)
    ax.legend(fontsize=8, loc="lower center")
    names = {"forward": "앞쪽 45° 유지", "prelook": "U턴 동안 다음 레인 시작점 지향"}
    gap = None
    for j, m in enumerate(("forward", "prelook")):
        pod, GX, GY, hist, target, ext = res[m]
        ax = fig.add_subplot(gs[0, 1 + j])
        im = ax.imshow(pod, origin="lower", extent=np.array(ext)/1000, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="#4fc3f7", lw=1.2)
        ax.plot(target[0]/1000, target[1]/1000, "x", color="#4fc3f7", ms=10, mew=2)
        g = (np.abs(GX - (X0 - 2*R)) <= 80) & (GY > Y0 + LANE - 700) & (GY < Y0 + LANE)
        gap = g
        ax.set_title(f"{names[m]}\n새 레인 첫 700 m 평균 POD {pod[g].mean():.2f}", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
        fig.colorbar(im, ax=ax, shrink=.7)
    ax = fig.add_subplot(gs[0, 3])
    pf, pp = res["forward"][0], res["prelook"][0]
    im = ax.imshow(pp - pf, origin="lower", extent=np.array(res["forward"][5])/1000, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="black", lw=.8)
    ax.set_title("차이 (지향 − 유지): 파랑 = 지향이 더 찾음", fontsize=11); fig.colorbar(im, ax=ax, shrink=.7)
    ax.set_xlabel("동쪽 (km)")
    ax = fig.add_subplot(gs[1, :3])
    h = res["prelook"][3]; hf = res["forward"][3]
    ax.plot(h[:, 0], h[:, 1], color="#c0392b", lw=2, label="짐벌 롤 (지향)")
    ax.plot(h[:, 0], h[:, 2], color="#2a4d7a", lw=2, label="짐벌 내려보는 각 (지향)")
    ax.plot(hf[:, 0], hf[:, 1], color="#c0392b", lw=1, ls=":", label="짐벌 롤 (유지)")
    ax.plot(h[:, 0], h[:, 4], color="#999", lw=1, ls="--", label="기체 롤")
    for v, lab in ((gcfg.roll_lim_deg, "롤 한계"), (-gcfg.roll_lim_deg, ""), (gcfg.dep_min_deg, "내려보기 최소"), (gcfg.dep_max_deg, "내려보기 최대")):
        ax.axhline(v, color="#bbb", lw=.8); ax.text(LANE/V + np.pi*R/V + 58, v + 1, lab, ha="right", fontsize=8, color="#777")
    t0, t1 = LANE/V, LANE/V + np.pi*R/V
    ax.axvspan(t0, t1, color="#f6e7c1", zorder=0); ax.text((t0 + t1)/2, 75, "U턴", ha="center")
    ax2 = ax.twinx(); ax2.plot(h[:, 0], h[:, 3], color="#8e44ad", lw=1.2, alpha=.7); ax2.set_ylabel("짐벌 회전 속도 (°/s, 보라)")
    ax2.set_ylim(0, 60)
    ax.set_xlim(t0 - 30, t1 + 60); ax.set_ylim(-60, 80); ax.set_xlabel("시간 (초)"); ax.set_ylabel("각도 (°)")
    ax.set_title(f"짐벌 각도 (롤 ±{gcfg.roll_lim_deg:.0f}°, 내려보기 {gcfg.dep_min_deg:.0f}~{gcfg.dep_max_deg:.0f}°, 회전 속도 {gcfg.slew_dps:.0f}°/s)", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[1, 3])
    labels = ["새 레인\n첫 700 m", "U턴 구역", "그림 전체"]
    turn = res["forward"][2] > Y0 + LANE - 100
    vals = {m: [res[m][0][gap].mean(), res[m][0][turn].mean(), res[m][0].mean()] for m in ("forward", "prelook")}
    xs = np.arange(3)
    ax.bar(xs - .19, vals["forward"], .36, color="#bbbbbb", label=names["forward"])
    ax.bar(xs + .19, vals["prelook"], .36, color="#2a4d7a", label=names["prelook"])
    for i in range(3):
        ax.text(xs[i] - .19, vals["forward"][i] + .01, f"{vals['forward'][i]:.2f}", ha="center", fontsize=9)
        ax.text(xs[i] + .19, vals["prelook"][i] + .01, f"{vals['prelook'][i]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs, labels); ax.set_ylabel("평균 누적 POD (땅 칸)"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
    ax.set_title("구역별 평균 POD", fontsize=11)
    fig.suptitle(f"충주 남서부: 레인 3 km → 롤 {ROLL:.0f}° U턴 (반경 {R:.0f} m) → 레인, 33 m/s, 고도 500 m, 2축 롤·피치 짐벌", fontsize=14)
    fig.savefig(DATA/"chungju"/"gimbal_check.png", dpi=105)
    plt.close(fig)
    for A in (1.0, 0.4, 0.05):
        xs, po = lateral(OLD, A); _, pn = lateral(NEW, A)
        print(f"A {A}: centre {po[len(xs)//2]:.3f}->{pn[len(xs)//2]:.3f}, 70 m {po[np.argmin(abs(xs-70))]:.3f}->{pn[np.argmin(abs(xs-70))]:.3f}, "
              f"sweep width {np.trapezoid(po, xs):.0f}->{np.trapezoid(pn, xs):.0f} m")
    for m in ("forward", "prelook"):
        h = res[m][3]
        print(m, "gap/turn/all", [round(v, 3) for v in vals[m]], "max gimbal roll", round(np.abs(h[:, 1]).max(), 1),
              "dep range", round(h[:, 2].min(), 1), round(h[:, 2].max(), 1), "max slew", round(h[:, 3].max(), 1))


if __name__ == "__main__":
    main()
