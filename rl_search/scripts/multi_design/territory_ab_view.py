"""Two ways to show the teammates on the seven maps (user 2026-11-23, after territory_final_view: the separation-normalised ④ may mislead on
the large maps), one moment per figure, laid out to be read (user: label the panels, a context panel, explain the layout):

- left, "전체 상황": the area seen from above, north up, centred on the observing UAV: the particles (grey, darker = more remaining
  probability), every UAV (number = launch order, arrow = heading), the observing UAV in black, dashed squares = the maps that fit
- row A (안 A): small maps (1.6 / 3.2 / 6.4 km) ③ (distance difference over the separation, the strongest teammate claim, -1..1);
  large maps (12.8 km and up) only a mark at each teammate (and a short line along its heading), no division
- row B (안 B): every map the arrival-time advantage over the map's lead: (d_me - d_nearest teammate) / (33 m/s x lead), clipped to -1..1
  (lead 15 / 30 / 60 / 150 / 300 / 600 / 600 s = 0.5 / 1 / 2 / 5 / 10 / 20 / 20 km)

Colours on the maps: orange = mine (-1), light grey = not divided (0), blue = a teammate's (+1). Each map: heading up, black triangle = the
observing UAV in the centre, coloured circles with numbers = the teammates' true positions. Panels are labelled A1..A7 / B1..B7.

    python -m rl_search.scripts.multi_design.territory_ab_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M \
        --minute 3 --what "줄지어 이동"
    python -m rl_search.scripts.multi_design.territory_ab_view --model rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip --label r15_6.8M \
        --single --minute 45 --what "멀리 흩어짐"
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, RUN_CFG, _gaps
from rl_search.scripts.multi_design.mate_channels_view import _cells
from rl_search.scripts.multi_design.territory_final_view import territory_diff

SPEED = 33.0
SMALL = 3                          # the first three maps (1.6 / 3.2 / 6.4 km) are "small"
MARK_CELLS, HEAD_CELLS = 1.5, 4


def territory_time(x, y, yaw, mates_xy, side, lead, px=64) -> np.ndarray:
    """안 B: (d_me - d_nearest teammate) / (SPEED x lead), clipped to -1..1; all -1 with no teammate airborne."""
    if not len(mates_xy):
        return -np.ones((px, px), np.float32)
    F, R, WX, WY = _cells(x, y, yaw, side, px)
    d_m = np.sqrt(np.min([(WX - mx)**2 + (WY - my)**2 for mx, my in mates_xy], axis=0))
    return np.clip((np.sqrt(F*F + R*R) - d_m)/(SPEED*max(float(lead), 1.0)), -1.0, 1.0).astype(np.float32)


def marks_channel(x, y, yaw, mates_pose, side, px=64) -> np.ndarray:
    """안 A large maps: 1 in a disk of MARK_CELLS cells at each teammate, 0.5 along a short heading line, else 0."""
    g = np.zeros((px, px), np.float32)
    cell = side/px
    rr, cc = np.mgrid[0:px, 0:px]
    cs, sn = np.cos(yaw), np.sin(yaw)

    def rc(p):
        d = np.asarray(p) - np.array([x, y])
        fwd, right = d[0]*cs + d[1]*sn, d[0]*sn - d[1]*cs
        return (side/2 - fwd)/cell, (right + side/2)/cell

    for mx, my, myaw in mates_pose:
        for q in np.linspace(0, HEAD_CELLS*cell, 4*HEAD_CELLS):
            r_, c_ = rc((mx + q*np.cos(myaw), my + q*np.sin(myaw)))
            ir, ic = int(np.floor(r_)), int(np.floor(c_))
            if 0 <= ir < px and 0 <= ic < px:
                g[ir, ic] = max(g[ir, ic], 0.5)
        r0, c0 = rc((mx, my))
        g = np.maximum(g, (((rr + 0.5 - r0)**2 + (cc + 0.5 - c0)**2) <= MARK_CELLS**2).astype(np.float32))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--single", action="store_true", help="a single-UAV model (R15) flown by every UAV on its own inputs")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--uav", type=int, default=1, help="the observing UAV (0 = first launched)")
    ap.add_argument("--minute", type=float, required=True, help="minutes after the first launch")
    ap.add_argument("--what", default="", help="what happens at that moment (title)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(1)
    c = json.loads(Path(RUN_CFG).read_text())
    s = settings_of(c) | ({"look_w": float(c["look_w"])} if c.get("look_w") else {})
    model = PPO.load(a.model, device="cpu")
    case, scen, _, _ = _case("v2j", 3, a.case)
    env = MultiRealEnv(a.k, RealRLConfig(region="jirisan120", settings=s, mask_cost=float(c.get("mask_cost") or 0.0)), seed=a.case)
    obs = env.reset(options=case.options(scen), gaps=_gaps(a.case, a.k))
    t0 = env.T
    while (env.T - t0)/60.0 < a.minute - 1e-9 and not env.done:
        if a.single:
            obs = [env.base_observation(i) for i in range(a.k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
    i = a.uav
    x, y, yaw = env.uavs[i].env.plant.pose
    mates = [j for j in range(a.k) if j != i and env.active[j]]
    mates_xy = [tuple(env.uavs[j].env.plant.pose[:2]) for j in mates]
    mates_pose = [tuple(env.uavs[j].env.plant.pose) for j in mates]
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    cmap = LinearSegmentedColormap.from_list("mine_mates", ["#d9480f", "#fbe3d4", "#f2f2f2", "#d6e4fa", "#1c4fb0"])

    fig = plt.figure(figsize=(31, 9.6))
    gs = fig.add_gridspec(2, 9, width_ratios=[2.1] + [1]*7 + [0.12], wspace=0.12, hspace=0.28)
    # 전체 상황 (north up, centred on the observer)
    ax = fig.add_subplot(gs[:, 0])
    far = max([np.hypot(px_ - x, py_ - y) for px_, py_ in mates_xy] + [800.0])
    half = 1.3*far
    e = env.uavs[i].env
    P = e.belief.positions(e.t)
    w = env.W
    keep = (np.abs(P[:, 0] - x) < half) & (np.abs(P[:, 1] - y) < half) & (w > 0)
    if keep.any():
        ww = w[keep]/max(float(w.max()), 1e-12)
        ax.scatter((P[keep, 0] - x)/1000, (P[keep, 1] - y)/1000, s=2, c=0.15 + 0.85*ww, cmap="Greys", vmin=0, vmax=1, linewidths=0)
    for k, (name, side, lead) in enumerate(env.maps):
        if side/2 > half:
            continue
        cs, sn = np.cos(yaw), np.sin(yaw)
        corners = [(f, r_) for f, r_ in ((side/2, -side/2), (side/2, side/2), (-side/2, side/2), (-side/2, -side/2))]
        poly = [((f*cs + r_*sn)/1000, (f*sn - r_*cs)/1000) for f, r_ in corners]
        ax.add_patch(Polygon(poly, closed=True, fill=False, ls="--", lw=1.0, ec="#555555"))
        ax.text(poly[1][0], poly[1][1], f" {side/1000:g} km 지도", fontsize=8, color="#555555")
    for j in range(a.k):
        if not env.active[j]:
            continue
        px_, py_, pyaw = env.uavs[j].env.plant.pose
        dx, dy = (px_ - x)/1000, (py_ - y)/1000
        colr = "#111111" if j == i else cols[j]
        ax.plot(dx, dy, "^" if j == i else "o", color=colr, ms=11 if j == i else 9, mec="white", mew=1.0)
        ax.annotate("", xy=(dx + 0.09*half/1000*np.cos(pyaw), dy + 0.09*half/1000*np.sin(pyaw)), xytext=(dx, dy),
                    arrowprops=dict(arrowstyle="->", color=colr, lw=1.8))
        ax.text(dx, dy + 0.035*half/1000, f"{j + 1}번째" + (" (나)" if j == i else ""), color=colr, fontsize=10, ha="center", va="bottom",
                fontweight="bold")
    ax.set_xlim(-half/1000, half/1000)
    ax.set_ylim(-half/1000, half/1000)
    ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (km, 나 기준)")
    ax.set_ylabel("북쪽 (km, 나 기준)")
    ax.set_title("전체 상황 (위에서 본 그림, 북쪽이 위)\n회색 점 = 입자(진할수록 남은 확률 큼), 화살표 = 기수 방향,\n점선 네모 = 오른쪽 지도들이 덮는 넓이(기수 방향으로 돌아감)", fontsize=10)

    def draw_marks(ax, side):
        cs, sn = np.cos(yaw), np.sin(yaw)
        for j in mates:
            d = env.uavs[j].env.plant.pose[:2] - np.array([x, y])
            fwd, right = d[0]*cs + d[1]*sn, d[0]*sn - d[1]*cs
            if abs(fwd) <= side/2 and abs(right) <= side/2:
                ax.plot(right/1000, fwd/1000, "o", mfc="none", mec=cols[j], ms=12, mew=2.0)
                ax.text(right/1000, fwd/1000, str(j + 1), color=cols[j], fontsize=8, ha="center", va="center", fontweight="bold")
        ax.plot(0, 0, marker="^", color="#111111", ms=10, mec="white", mew=0.8)

    im = None
    for row, plan in enumerate(("A", "B")):
        for k, (name, side, lead) in enumerate(env.maps):
            ax = fig.add_subplot(gs[row, 1 + k])
            if plan == "A":
                v = territory_diff(x, y, yaw, mates_xy, side) if k < SMALL else marks_channel(x, y, yaw, mates_pose, side)
            else:
                v = territory_time(x, y, yaw, mates_xy, side, lead)
            im = ax.imshow(v, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", extent=(-side/2000, side/2000, -side/2000, side/2000))
            draw_marks(ax, side)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.97, f"{plan}{k + 1}", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999"))
            kind = ("③ 나누기" if k < SMALL else "동료 점만") if plan == "A" else f"기준 {SPEED*lead/1000:g} km"
            ax.set_title(f"{side/1000:g} km 지도 · {kind}", fontsize=10)
            if k == 0:
                ax.set_ylabel("안 A\n(작은 지도 ③, 큰 지도 점)" if plan == "A" else "안 B\n(지도마다 시간 기준으로 나누기)", fontsize=12)
    cax = fig.add_subplot(gs[:, 8])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["-1 내 몫\n(주황)", "0 안 나뉨\n(회색)", "+1 동료 몫·동료\n(파랑)"], fontsize=10)
    fig.suptitle(f"{a.minute:g}분: {a.what} - {i + 1}번째 이륙 기체가 받는 동료 입력 ({a.label}, 지리산 사례 {a.case + 1}, {a.k}대). "
                 f"오른쪽 지도: 위 = 기수 방향, 가운데 검은 삼각형 = 나, 숫자 동그라미 = 동료(이륙 순서)", fontsize=13)
    path = OUT/f"territory_ab_{a.label}_m{a.minute:g}_u{i + 1}.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    dists = {j + 1: round(float(np.hypot(px_ - x, py_ - y))/1000, 2) for j, (px_, py_) in zip(mates, mates_xy)}
    print(path, "distances to teammates (km):", dists)


if __name__ == "__main__":
    main()
