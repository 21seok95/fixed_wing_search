"""The user's idea for the large maps (2026-11-23): the cell of each teammate 1, falling off around it ("동료 도달 범위"): value =
max over teammates of (1 - d / (33 m/s x map lead)), clipped at 0 = 1 - (the time the teammate needs to get there) / (the map's lead):
1 at the teammate, 0 where it cannot get within the lead (12.8 / 25.6 / 51.2 / 102.4 km maps: 4.95 / 9.9 / 19.8 / 19.8 km). It does not
compare with this UAV (unlike plan B): it only says where teammates are and how far they can get.

One figure, three moments (rows): 3 min in a column, 35 min clustered (M1 1.2M), 45 min spread (R15 flown by each UAV on its own). Each row:
"전체 상황" from above (north up), then the four large maps as the observing UAV gets them (heading up). Panels labelled 1a..3d.
Grey dots on the maps = the particles (drawn for reference only, not part of this input).

    python -m rl_search.scripts.multi_design.mate_reach_view
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, RUN_CFG, _gaps
from rl_search.scripts.multi_design.mate_channels_view import _cells

SPEED = 33.0
LARGE = (3, 4, 5, 6)
M1 = "rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip"
R15 = "rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip"
MOMENTS = ((M1, False, 3.0, "줄지어 이동 (M1 1.2M)"), (M1, False, 35.0, "뭉쳐 맴돎 (M1 1.2M)"), (R15, True, 45.0, "멀리 흩어짐 (R15를 각자 날림)"))


def reach_map(x, y, yaw, mates_xy, side, lead, px=64) -> np.ndarray:
    """max over teammates of 1 - d / (SPEED x lead), at least 0 (0 everywhere with no teammate airborne)."""
    if not len(mates_xy):
        return np.zeros((px, px), np.float32)
    _, _, WX, WY = _cells(x, y, yaw, side, px)
    d = np.sqrt(np.min([(WX - mx)**2 + (WY - my)**2 for mx, my in mates_xy], axis=0))
    return np.clip(1.0 - d/(SPEED*max(float(lead), 1.0)), 0.0, 1.0).astype(np.float32)


def fly_to(model_path, single, minute, k=4, case_k=0):
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    c = json.loads(Path(RUN_CFG).read_text())
    s = settings_of(c) | ({"look_w": float(c["look_w"])} if c.get("look_w") else {})
    model = PPO.load(model_path, device="cpu")
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s, mask_cost=float(c.get("mask_cost") or 0.0)), seed=case_k)
    obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, k))
    t0 = env.T
    while (env.T - t0)/60.0 < minute - 1e-9 and not env.done:
        if single:
            obs = [env.base_observation(i) for i in range(k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
    return env


def main():
    warnings.filterwarnings("ignore")
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(1)
    i = 1                                                       # the observing UAV: the second launched
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    fig = plt.figure(figsize=(26, 19))
    gs = fig.add_gridspec(3, 6, width_ratios=[1.35, 1, 1, 1, 1, 0.08], wspace=0.15, hspace=0.32)
    im = None
    for r, (mp, single, minute, what) in enumerate(MOMENTS):
        env = fly_to(mp, single, minute)
        x, y, yaw = env.uavs[i].env.plant.pose
        mates = [j for j in range(env.k) if j != i and env.active[j]]
        mates_xy = [tuple(env.uavs[j].env.plant.pose[:2]) for j in mates]
        e = env.uavs[i].env
        P = e.belief.positions(e.t)
        w = env.W
        # 전체 상황
        ax = fig.add_subplot(gs[r, 0])
        half = 1.3*max([np.hypot(px_ - x, py_ - y) for px_, py_ in mates_xy] + [3000.0])
        keep = (np.abs(P[:, 0] - x) < half) & (np.abs(P[:, 1] - y) < half) & (w > 0)
        if keep.any():
            ax.scatter((P[keep, 0] - x)/1000, (P[keep, 1] - y)/1000, s=2, c=0.15 + 0.85*w[keep]/max(float(w.max()), 1e-12), cmap="Greys",
                       vmin=0, vmax=1, linewidths=0)
        for j in range(env.k):
            if not env.active[j]:
                continue
            px_, py_, pyaw = env.uavs[j].env.plant.pose
            dx, dy = (px_ - x)/1000, (py_ - y)/1000
            colr = "#111111" if j == i else cols[j]
            ax.plot(dx, dy, "^" if j == i else "o", color=colr, ms=11 if j == i else 9, mec="white", mew=1.0)
            ax.annotate("", xy=(dx + 0.09*half/1000*np.cos(pyaw), dy + 0.09*half/1000*np.sin(pyaw)), xytext=(dx, dy),
                        arrowprops=dict(arrowstyle="->", color=colr, lw=1.8))
            ax.text(dx, dy + 0.04*half/1000, f"{j + 1}번째" + (" (나)" if j == i else ""), color=colr, fontsize=10, ha="center", va="bottom")
        ax.set_xlim(-half/1000, half/1000)
        ax.set_ylim(-half/1000, half/1000)
        ax.set_aspect("equal")
        ax.set_xlabel("동쪽 (km, 나 기준)")
        ax.set_ylabel(f"{minute:g}분: {what}\n\n북쪽 (km, 나 기준)", fontsize=11)
        if r == 0:
            ax.set_title("전체 상황 (위에서 본 그림, 북쪽이 위)\n회색 점 = 입자, 화살표 = 기수 방향", fontsize=10)
        dist = ", ".join(f"{j + 1}번째 {np.hypot(px_ - x, py_ - y)/1000:.1f} km" for j, (px_, py_) in zip(mates, mates_xy))
        ax.text(0.0, -0.2, f"나와 동료 거리: {dist}", transform=ax.transAxes, fontsize=9)
        for q, k in enumerate(LARGE):
            name, side, lead = env.maps[k]
            ax = fig.add_subplot(gs[r, 1 + q])
            v = reach_map(x, y, yaw, mates_xy, side, lead)
            im = ax.imshow(v, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", extent=(-side/2000, side/2000, -side/2000, side/2000))
            cs, sn = np.cos(yaw), np.sin(yaw)
            d = P[w > 0] - np.array([x, y])
            fwd, right = d[:, 0]*cs + d[:, 1]*sn, d[:, 0]*sn - d[:, 1]*cs
            ok = (np.abs(fwd) < side/2) & (np.abs(right) < side/2)
            ax.scatter(right[ok]/1000, fwd[ok]/1000, s=1, c="#9a9a9a", linewidths=0, alpha=0.6)
            for j in mates:
                dd = env.uavs[j].env.plant.pose[:2] - np.array([x, y])
                f_, r_ = dd[0]*cs + dd[1]*sn, dd[0]*sn - dd[1]*cs
                if abs(f_) <= side/2 and abs(r_) <= side/2:
                    ax.plot(r_/1000, f_/1000, "o", mfc="none", mec=cols[j], ms=11, mew=1.8)
                    ax.text(r_/1000, f_/1000, str(j + 1), color=cols[j], fontsize=8, ha="center", va="center")
            ax.plot(0, 0, marker="^", color="#111111", ms=10, mec="white", mew=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.97, f"{r + 1}{'abcd'[q]}", transform=ax.transAxes, fontsize=12, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999"))
            ax.set_title(f"{side/1000:g} km 지도 (예측 {lead:.0f}초 → 반경 {SPEED*lead/1000:.1f} km)", fontsize=10)
    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0 동료가 예측 시간 안에\n못 오는 곳", "0.5 예측 시간의\n절반이면 옴", "1 동료가 있는 곳"], fontsize=10)
    fig.suptitle("큰 지도의 동료 표시 (사용자 안): 동료가 있는 칸 1, 멀어질수록 줄어 '그 지도의 예측 시간 안에 못 오는 거리'에서 0. "
                 "2번째 이륙 기체가 받는 입력, 지리산 사례 1, 4대\n오른쪽 지도: 위 = 기수 방향, 검은 삼각형 = 나, 숫자 동그라미 = 동료(이륙 순서), "
                 "회색 점 = 입자(참고로 겹쳐 그림, 이 입력에는 없음)", fontsize=13)
    path = OUT/"mate_reach_large.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
