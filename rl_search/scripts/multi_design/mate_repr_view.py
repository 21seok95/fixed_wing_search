"""Ways to show a UAV where its teammates are (user 2026-11-20: the tail picture was hard to read; "동료 위치를 잘 파악할 수 있는 방법"):
one moment of a team flight (Jirisan validation case, a model, deterministic), one UAV's view on a few of its maps, the same teammates drawn
as candidate input channels side by side:

- 지금 꼬리: the M1 tail channel (teammates' 1 s positions over the map's lead time, 1 - age / window)
- 위치 + 방향: a mark of a fixed size in cells at each teammate (visible on every map) and a short line along its heading
- 앞으로 갈 길: each teammate flown on with its current course command for the map's lead time (the particles' prediction channel uses the
  same lead), 1 now fading to 0.3 at the lead
- 동료 영역: the cells nearer to a teammate than to this UAV (who reaches it first)

    python -m rl_search.scripts.multi_design.mate_repr_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M \
        --k 4 --case 0 --uav 1 --minute 15
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

MAP_IDS = (0, 2, 4, 6)          # 1.6, 6.4, 25.6, 102.4 km
MARK_CELLS = 1.5                # radius of the position mark in cells
HEAD_CELLS = 5                  # length of the heading line in cells


def cell_world(x, y, yaw, side, px):
    """World coordinates of the cell centres of a heading-up map (px, px)."""
    cell = side/px
    fwd = side/2 - (np.arange(px) + 0.5)*cell
    right = (np.arange(px) + 0.5)*cell - side/2
    F, R = np.meshgrid(fwd, right, indexing="ij")
    c, s = np.cos(yaw), np.sin(yaw)
    return x + F*c + R*s, y + F*s - R*c


def to_cells(pts, x, y, yaw, side, px):
    d = np.atleast_2d(pts) - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    fwd, right = d[:, 0]*c + d[:, 1]*s, d[:, 0]*s - d[:, 1]*c
    cell = side/px
    return (side/2 - fwd)/cell, (right + side/2)/cell              # fractional row, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--uav", type=int, default=1)
    ap.add_argument("--minute", type=float, default=15.0)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_multi import MultiRealEnv, predict_paths
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
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
    i = a.uav
    me = env.uavs[i].env.plant
    x, y, yaw = me.pose
    mates = [j for j in range(a.k) if j != i and env.active[j]]
    o = obs[i]
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    heads = ["남은 확률 (지금, 기존 채널)", "① 지금: 동료 꼬리", "② 위치 + 방향 표시", "③ 앞으로 갈 길 (지금 명령 유지)", "④ 동료 영역 (나보다 동료가 가까운 곳)"]
    fig, axs = plt.subplots(len(MAP_IDS), 5, figsize=(19, 3.9*len(MAP_IDS)))
    for r, k in enumerate(MAP_IDS):
        name, side, lead = env.maps[k]
        px = o[name].shape[-1]
        ext = (-side/2000, side/2000, -side/2000, side/2000)
        imgs = [o[name][2], o[name][-1]]
        mark = np.zeros((px, px))
        rr, cc = np.mgrid[0:px, 0:px]
        for j in mates:
            pj = env.uavs[j].env.plant
            row, col = to_cells(pj.pose[:2], x, y, yaw, side, px)
            mark = np.maximum(mark, ((rr + 0.5 - row[0])**2 + (cc + 0.5 - col[0])**2 <= MARK_CELLS**2)*1.0)
            for q in np.linspace(0, HEAD_CELLS*side/px, 4*HEAD_CELLS):
                hx, hy = pj.pose[0] + q*np.cos(pj.pose[2]), pj.pose[1] + q*np.sin(pj.pose[2])
                hr, hc = to_cells((hx, hy), x, y, yaw, side, px)
                ir, ic = int(np.floor(hr[0])), int(np.floor(hc[0]))
                if 0 <= ir < px and 0 <= ic < px:
                    mark[ir, ic] = max(mark[ir, ic], 0.6)
        imgs.append(mark)
        fut = np.zeros((px, px))
        for j in mates:
            pj = env.uavs[j].env.plant
            path = predict_paths(pj, [env.uavs[j].env.chi_tgt], max(float(lead), 5.0), every_s=max(1.0, lead/200))[0]
            val = np.linspace(1.0, 0.3, len(path))
            row, col = to_cells(path, x, y, yaw, side, px)
            ir, ic = np.floor(row).astype(int), np.floor(col).astype(int)
            ok = (ir >= 0) & (ir < px) & (ic >= 0) & (ic < px)
            np.maximum.at(fut, (ir[ok], ic[ok]), val[ok])
        imgs.append(fut)
        WX, WY = cell_world(x, y, yaw, side, px)
        d_me = np.hypot(WX - x, WY - y)
        d_m = np.min([np.hypot(WX - env.uavs[j].env.plant.pose[0], WY - env.uavs[j].env.plant.pose[1]) for j in mates], axis=0) if mates \
            else np.full_like(d_me, np.inf)
        imgs.append((d_m < d_me)*1.0)
        for cix, img in enumerate(imgs):
            ax = axs[r, cix]
            ax.imshow(img, cmap="Greys" if cix == 0 else "magma" if cix < 4 else "Blues", vmin=0, vmax=max(float(img.max()), 1e-6) if cix == 0 else 1.0,
                      interpolation="nearest", extent=ext)
            ax.plot(0, 0, marker=(3, 0, 0), color="#00b4d8", ms=9)
            if cix in (0, 4):
                for j in mates:
                    d = env.uavs[j].env.plant.pose[:2] - np.array([x, y])
                    fwd, right = d[0]*np.cos(yaw) + d[1]*np.sin(yaw), d[0]*np.sin(yaw) - d[1]*np.cos(yaw)
                    if abs(fwd) <= side/2 and abs(right) <= side/2:
                        ax.plot(right/1000, fwd/1000, "o", mfc="none", mec=cols[j], ms=9, mew=1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(heads[cix], fontsize=10)
            if cix == 0:
                ax.set_ylabel(f"{side/1000:g} km 지도\n(칸 {side/px:.0f} m, 예측 {lead:.0f} s)", fontsize=10)
            if 0 < cix < 4:
                ax.text(0.02, 0.02, f"켜진 칸 {int((img > 0).sum())}", color="w", fontsize=8, transform=ax.transAxes)
    fig.suptitle(f"동료 위치를 지도에 넣는 방법 비교: {a.label}, 지리산 사례 {a.case + 1}, {a.k}대, 첫 이륙 후 {a.minute:g}분, {i + 1}번째 이륙 기체가 보는 지도\n"
                 f"(위 = 기수 방향, 하늘색 삼각형 = 자기, 동그라미 = 동료 실제 위치: 1번째 파랑, 2번째 주황, 3번째 초록, 4번째 보라)", fontsize=12)
    path = OUT/f"mate_repr_{a.label}_k{a.k}_u{i + 1}_m{a.minute:g}.png"
    fig.savefig(path, dpi=85, bbox_inches="tight")
    print(path, "speed", round(float(me.v), 1), "m/s")


if __name__ == "__main__":
    main()
