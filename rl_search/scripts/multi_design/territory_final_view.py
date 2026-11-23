"""④ 동료 영역 in the form agreed on 2026-11-20 (user: check once more how it looks before it is built): soft, p = 1, centred,
value = 2 x d_me / (d_me + d_mate) - 1 in [-1, 1] (-1 = only I am near, 0 = I and the nearest teammate are as near, +1 = only a teammate
is near), on all seven maps, at the two moments of mate_channels_view (3 min in a column, 35 min clustered), for two UAVs, with the same
value seen from above (north up) around the UAV.

    python -m rl_search.scripts.multi_design.territory_final_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M
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
from rl_search.scripts.multi_design.mate_channels_view import MOMENTS, territory_map


def territory_signed(x, y, yaw, mates_xy, side, px=64) -> np.ndarray:
    """-1 .. +1 form of the soft ④ (all -1 with no teammate airborne: everything is mine)."""
    if not len(mates_xy):
        return -np.ones((px, px), np.float32)
    return 2.0*territory_map(x, y, yaw, mates_xy, side, px, soft=True) - 1.0


def territory_diff(x, y, yaw, mates_xy, side, px=64, nearest: bool = False) -> np.ndarray:
    """④ as the distance difference over the separation. Each teammate j claims a cell by c_j = (d_me - d_j) / |me - j| in [-1, 1]
    (triangle inequality): -1 = the cell lies beyond me as seen from j, +1 = beyond j, 0 = on our bisector; far from both it keeps the
    direction (cos of the angle to the me -> j axis) instead of fading to 0. The value is the strongest claim, max_j c_j (continuous;
    ``nearest``: the claim of the teammate nearest to the cell, which jumps where the nearest teammate changes). All -1 with no teammate."""
    from rl_search.scripts.multi_design.mate_channels_view import _cells
    if not len(mates_xy):
        return -np.ones((px, px), np.float32)
    F, R, WX, WY = _cells(x, y, yaw, side, px)
    M = np.array(mates_xy, float)
    D = np.sqrt((WX[None] - M[:, 0, None, None])**2 + (WY[None] - M[:, 1, None, None])**2)       # (n_mates, px, px)
    sep = np.maximum(np.hypot(M[:, 0] - x, M[:, 1] - y), 1.0)[:, None, None]
    C = (np.sqrt(F*F + R*R)[None] - D)/sep
    if nearest:
        j = np.argmin(D, axis=0)
        return np.clip(np.take_along_axis(C, j[None], 0)[0], -1.0, 1.0).astype(np.float32)
    return np.clip(C.max(axis=0), -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--uavs", type=int, nargs="+", default=[1], help="whose maps (0 = first launched)")
    ap.add_argument("--above-km", type=float, default=12.8, help="size of the north-up view")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
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
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    cmap = LinearSegmentedColormap.from_list("mine_mates", ["#d9480f", "#fbe3d4", "#f4f4f4", "#d6e4fa", "#1c4fb0"])
    forms = (("비율", territory_signed), ("거리 차 ÷ 기체 사이, 가장 가까운 동료", lambda *q: territory_diff(*q, nearest=True)),
             ("거리 차 ÷ 기체 사이, 가장 강한 주장", territory_diff))
    rows = [(m, w, i, f) for (m, w) in MOMENTS for i in a.uavs for f in forms]
    fig, axs = plt.subplots(len(rows), 8, figsize=(30, 4.3*len(rows)), gridspec_kw=dict(width_ratios=[1.25] + [1]*7))
    done_minute = None
    im = None
    for r, (minute, what, i, (fname, ffun)) in enumerate(rows):
        if done_minute != minute:
            while (env.T - t0)/60.0 < minute - 1e-9 and not env.done:
                batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
                acts, _ = model.predict(batch, deterministic=True)
                obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
            done_minute = minute
        x, y, yaw = env.uavs[i].env.plant.pose
        mates = [j for j in range(a.k) if j != i and env.active[j]]
        mates_xy = [tuple(env.uavs[j].env.plant.pose[:2]) for j in mates]

        def marks(ax, side, north=False):
            fy = np.pi/2 if north else yaw
            for j in mates:
                d = env.uavs[j].env.plant.pose[:2] - np.array([x, y])
                fwd, right = d[0]*np.cos(fy) + d[1]*np.sin(fy), d[0]*np.sin(fy) - d[1]*np.cos(fy)
                if abs(fwd) <= side/2 and abs(right) <= side/2:
                    ax.plot(right/1000, fwd/1000, "o", mfc="none", mec=cols[j], ms=10, mew=1.8)
            ax.plot(0, 0, marker=(3, 0, (np.degrees(yaw - np.pi/2) if north else 0)), color="#111111", ms=11)

        side = a.above_km*1000
        ax = axs[r, 0]
        im = ax.imshow(ffun(x, y, np.pi/2, mates_xy, side), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest",
                       extent=(-side/2000, side/2000, -side/2000, side/2000))
        marks(ax, side, north=True)
        ax.set_xlabel("동쪽 (km)", fontsize=8)
        ax.set_ylabel(f"{minute:g}분: {what}\n{i + 1}번째 기체 · {fname}\n북쪽 (km)", fontsize=11)
        ax.set_title(f"위에서 본 같은 값 (북쪽 위), {a.above_km:g} km", fontsize=9)
        for k, (name, sd, _) in enumerate(env.maps):
            v = ffun(x, y, yaw, mates_xy, sd)
            ax = axs[r, 1 + k]
            ax.imshow(v, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", extent=(-sd/2000, sd/2000, -sd/2000, sd/2000))
            marks(ax, sd)
            ax.set_xticks([])
            ax.set_yticks([])
            mine, mid, mate = (v < -0.2).mean(), (np.abs(v) <= 0.2).mean(), (v > 0.2).mean()
            ax.set_title(f"{sd/1000:g} km 지도 (입력, 위 = 기수 방향)\n내 쪽 {mine*100:.0f}% · 비슷 {mid*100:.0f}% · 동료 쪽 {mate*100:.0f}%", fontsize=9)
    cb = fig.colorbar(im, ax=axs[:, -1], fraction=0.025, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["-1 나만 가까움\n(내 몫)", "-0.5", "0 나와 동료가\n비슷하게 가까움", "+0.5", "+1 동료만 가까움\n(동료 몫)"], fontsize=9)
    fig.suptitle(f"④ 동료 영역 두 가지 정규화: 비율 = 2 × 내 거리 ÷ (내 거리 + 가장 가까운 동료 거리) - 1, 거리 차 ÷ 기체 사이 = (내 거리 - 동료 거리) ÷ (나와 그 동료 사이 거리)를 그 칸에 가장 가까운 동료로, 또는 동료 중 가장 큰 값으로.\n{a.label}, 지리산 사례 {a.case + 1}, {a.k}대. "
                 f"검은 삼각형 = 자기, 동그라미 = 동료 (1번째 파랑, 2번째 주황, 3번째 초록, 4번째 보라). "
                 f"'비슷' = -0.2~+0.2. 떠 있는 동료가 없으면 지도 전체가 -1", fontsize=12)
    path = OUT/f"territory_forms_{a.label}_k{a.k}.png"
    fig.savefig(path, dpi=75, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
