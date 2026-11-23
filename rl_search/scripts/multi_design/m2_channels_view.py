"""The teammate channels as the M2 env builds them (rl_multi design "m2", rl_search.mate_maps), read straight from ``MultiRealEnv.observation``
to check the implementation against the previews (territory_ab_view, mate_channels_view). The team is flown by R15 on its own inputs (an m2
model is not trained yet) on a Jirisan validation case with the m2 launch gaps; one figure per moment, laid out to be read:

- left "전체 상황": north up, centred on the observing UAV (black), teammates numbered by launch order with heading arrows, particles grey
- row 가 (가1..가7): the "who gets there first" channel of every map: ③ claim on the small maps (가1..가3), plan B on the large maps (가4..가7)
- row 나 (나1..나3): the danger zone of the small maps (where a teammate can be within the map's lead, widened 250 m)

    python -m rl_search.scripts.multi_design.m2_channels_view --minute 3 --what "줄지어 이동"
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, R15_CFG, RUN_CFG, _gaps

R15 = "rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute", type=float, required=True)
    ap.add_argument("--what", default="")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--uav", type=int, default=1)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from rl_search.rl_multi import SMALL_SIDE, MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(1)
    s = settings_of(json.loads(Path(RUN_CFG).read_text()))
    model = PPO.load(R15, device="cpu")
    case, scen, _, _ = _case("v2j", 3, a.case)
    env = MultiRealEnv(a.k, RealRLConfig(region="jirisan120", settings=s, mask_cost=0.1), seed=a.case, design="m2")
    env.reset(options=case.options(scen), gaps=_gaps(a.case, a.k, "m2"))
    t0 = env.T
    while (env.T - t0)/60.0 < a.minute - 1e-9 and not env.done:
        obs = [env.base_observation(i) for i in range(a.k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        env.step([np.atleast_1d(x) for x in np.ravel(acts)])
    i = a.uav
    o = env.observation(i)                                     # exactly what the network gets (float32 here)
    x, y, yaw = env.uavs[i].env.plant.pose
    mates = [j for j in range(a.k) if j != i and env.active[j]]
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    div = LinearSegmentedColormap.from_list("mine_mates", ["#d9480f", "#fbe3d4", "#f2f2f2", "#d6e4fa", "#1c4fb0"])
    fig = plt.figure(figsize=(31, 9.6))
    gs = fig.add_gridspec(2, 9, width_ratios=[2.1] + [1]*7 + [0.12], wspace=0.12, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    far = max([np.hypot(*(env.uavs[j].env.plant.pose[:2] - np.array([x, y]))) for j in mates] + [800.0])
    half = 1.3*far
    e = env.uavs[i].env
    P, w = e.belief.positions(e.t), env.W
    keep = (np.abs(P[:, 0] - x) < half) & (np.abs(P[:, 1] - y) < half) & (w > 0)
    if keep.any():
        ax.scatter((P[keep, 0] - x)/1000, (P[keep, 1] - y)/1000, s=2, c=0.15 + 0.85*w[keep]/max(float(w.max()), 1e-12), cmap="Greys",
                   vmin=0, vmax=1, linewidths=0)
    for j in range(a.k):
        if not env.active[j]:
            continue
        px_, py_, pyaw = env.uavs[j].env.plant.pose
        dx, dy = (px_ - x)/1000, (py_ - y)/1000
        colr = "#111111" if j == i else cols[j]
        ax.plot(dx, dy, "^" if j == i else "o", color=colr, ms=11 if j == i else 9, mec="white", mew=1.0)
        ax.annotate("", xy=(dx + 0.09*half/1000*np.cos(pyaw), dy + 0.09*half/1000*np.sin(pyaw)), xytext=(dx, dy),
                    arrowprops=dict(arrowstyle="->", color=colr, lw=1.8))
        ax.text(dx, dy + 0.035*half/1000, f"{j + 1}번째" + (" (나)" if j == i else ""), color=colr, fontsize=10, ha="center", va="bottom")
    ax.set_xlim(-half/1000, half/1000)
    ax.set_ylim(-half/1000, half/1000)
    ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (km, 나 기준)")
    ax.set_ylabel("북쪽 (km, 나 기준)")
    dist = ", ".join(f"{j + 1}번째 {np.hypot(*(env.uavs[j].env.plant.pose[:2] - np.array([x, y])))/1000:.1f} km" for j in mates)
    ax.set_title(f"전체 상황 (북쪽 위)\n나와 동료 거리: {dist}", fontsize=10)

    def marks(ax, side):
        cs, sn = np.cos(yaw), np.sin(yaw)
        for j in mates:
            d = env.uavs[j].env.plant.pose[:2] - np.array([x, y])
            f_, r_ = d[0]*cs + d[1]*sn, d[0]*sn - d[1]*cs
            if abs(f_) <= side/2 and abs(r_) <= side/2:
                ax.plot(r_/1000, f_/1000, "o", mfc="none", mec=cols[j], ms=12, mew=2.0)
                ax.text(r_/1000, f_/1000, str(j + 1), color=cols[j], fontsize=8, ha="center", va="center")
        ax.plot(0, 0, marker="^", color="#111111", ms=10, mec="white", mew=0.8)

    im1 = im2 = None
    q = 0
    for k, (name, side, lead) in enumerate(env.maps):
        small = side <= SMALL_SIDE
        ch = o[name]
        first = ch[-2] if small else ch[-1]                    # ③ claim (small) or plan B (large)
        ax = fig.add_subplot(gs[0, 1 + k])
        im1 = ax.imshow(first, cmap=div, vmin=-1, vmax=1, interpolation="nearest", extent=(-side/2000, side/2000, -side/2000, side/2000))
        marks(ax, side)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.03, 0.97, f"가{k + 1}", transform=ax.transAxes, fontsize=12, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999"))
        ax.set_title(f"{side/1000:g} km · " + ("③ 동료 영역" if small else f"안 B (기준 {env.uavs[i].env.v*lead/1000:.1f} km)"), fontsize=10)
        if k == 0:
            ax.set_ylabel("가: 누가 먼저 닿나\n(주황 내 몫 · 회색 경계 · 파랑 동료 몫)", fontsize=11)
        if small:
            ax = fig.add_subplot(gs[1, 1 + k])
            im2 = ax.imshow(ch[-1], cmap="Reds", vmin=0, vmax=1, interpolation="nearest", extent=(-side/2000, side/2000, -side/2000, side/2000))
            marks(ax, side)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.97, f"나{k + 1}", transform=ax.transAxes, fontsize=12, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999"))
            ax.set_title(f"{side/1000:g} km · 위험 구역 ({lead:.0f}초)", fontsize=10)
            if k == 0:
                ax.set_ylabel("나: 위험 구역\n(동료가 곧 올 수 있는 곳 + 250 m)", fontsize=11)
            q += 1
    cax = fig.add_subplot(gs[0, 8])
    cb = fig.colorbar(im1, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["-1 내 몫", "0 경계", "+1 동료 몫"], fontsize=10)
    cax2 = fig.add_subplot(gs[1, 8])
    cb2 = fig.colorbar(im2, cax=cax2, ticks=[0, 0.3, 1])
    cb2.ax.set_yticklabels(["0 동료가\n못 옴", "0.3 예측 시간\n끝에 옴", "1 지금 동료"], fontsize=10)
    fig.suptitle(f"M2 환경이 실제로 만드는 동료 채널: {a.minute:g}분 {a.what} - {i + 1}번째 이륙 기체의 입력 (R15를 각자 날린 비행, 이륙 간격 5~10초, "
                 f"지리산 사례 {a.case + 1}, {a.k}대). 오른쪽 지도: 위 = 기수 방향, 검은 삼각형 = 나, 숫자 동그라미 = 동료", fontsize=13)
    path = OUT/f"m2_channels_m{a.minute:g}.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    print(path, "| vec", o["vec"].shape, "| map channels", {nm: o[nm].shape[0] for nm, _, _ in env.maps})


if __name__ == "__main__":
    main()
