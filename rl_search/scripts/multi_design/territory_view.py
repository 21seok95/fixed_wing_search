"""What the teammate-area channel (④ 동료 영역) is (user 2026-11-20: "4 동료영역이 무엇인지는 확인이 필요"): at two moments of a team flight
(Jirisan validation case, a model, deterministic) the area around the team seen from above (north up), every cell coloured by the UAV nearest
to it, and next to it the channel two UAVs would get (their own heading-up map: 0 = nearer to me, 1 = nearer to a teammate), with the
current tail channel of the same two UAVs for comparison: how different the inputs of two UAVs flying together are.

    python -m rl_search.scripts.multi_design.territory_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M
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
from rl_search.scripts.multi_design.mate_repr_view import cell_world

MOMENTS = ((3.0, 2, "줄지어 이동 중"), (35.0, 4, "뭉쳐 맴도는 중"))       # (minutes after the first launch, map index, what happens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1], help="the two UAVs whose inputs are compared (0 = first launched)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
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
    snaps = []
    for minute, k_map, what in MOMENTS:
        while (env.T - t0)/60.0 < minute - 1e-9 and not env.done:
            batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
            acts, _ = model.predict(batch, deterministic=True)
            obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
        snaps.append(dict(minute=minute, k_map=k_map, what=what, poses=[u.env.plant.pose.copy() for u in env.uavs],
                          active=env.active.copy(), tails={i: obs[i][env.maps[k_map][0]][-1].copy() for i in a.pair}))
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    fig, axs = plt.subplots(len(snaps), 5, figsize=(24, 5.4*len(snaps)), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1, 1]))
    for r, sn in enumerate(snaps):
        name, side, _ = env.maps[sn["k_map"]]
        px = 64
        act = [j for j in range(a.k) if sn["active"][j]]
        P = np.array([sn["poses"][j][:2] for j in act])
        # (가) from above, north up: every cell coloured by the nearest UAV, around the team's centre
        ax = axs[r, 0]
        cx, cy = P.mean(0)
        half = side/2
        g = np.linspace(-half, half, 200)
        GX, GY = np.meshgrid(cx + g, cy + g)
        D = np.stack([np.hypot(GX - p[0], GY - p[1]) for p in P])
        near = np.array(act)[np.argmin(D, axis=0)]
        ax.imshow(near, origin="lower", extent=((cx - half)/1000, (cx + half)/1000, (cy - half)/1000, (cy + half)/1000),
                  cmap=ListedColormap(cols[:a.k]), vmin=-0.5, vmax=a.k - 0.5, alpha=0.35, interpolation="nearest")
        for j in act:
            x, y, yaw = sn["poses"][j]
            ax.plot(x/1000, y/1000, "o", color=cols[j], ms=8)
            ax.annotate("", xy=((x + 0.06*side*np.cos(yaw))/1000, (y + 0.06*side*np.sin(yaw))/1000), xytext=(x/1000, y/1000),
                        arrowprops=dict(arrowstyle="->", color=cols[j], lw=1.6))
            ax.text(x/1000, y/1000 + side/40000, f"{j + 1}번째", color=cols[j], fontsize=9, ha="center", va="bottom")
        ax.set_aspect("equal")
        ax.set_xlabel("동쪽 (km)")
        ax.set_ylabel("북쪽 (km)")
        ax.set_title(f"(가) 위에서 본 그림 (북쪽이 위), 첫 이륙 후 {sn['minute']:g}분: {sn['what']}\n색 = 그 칸에 가장 가까운 기체 (화살표 = 기수 방향)", fontsize=10)
        # (나)(다) the channel of the two UAVs, heading up; (라)(마) their tail channel now
        terr = {}
        for col, i in enumerate(a.pair):
            x, y, yaw = sn["poses"][i]
            WX, WY = cell_world(x, y, yaw, side, px)
            d_me = np.hypot(WX - x, WY - y)
            mates = [j for j in act if j != i]
            d_m = np.min([np.hypot(WX - sn["poses"][j][0], WY - sn["poses"][j][1]) for j in mates], axis=0) if mates else np.full_like(d_me, np.inf)
            terr[i] = (d_m < d_me)*1.0
            for cc, img, cmap, title in ((1 + col, terr[i], "Blues", f"④ 동료 영역: {i + 1}번째 기체의 입력"),
                                         (3 + col, sn["tails"][i], "magma", f"지금 꼬리 채널: {i + 1}번째 기체의 입력")):
                ax = axs[r, cc]
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", extent=(-side/2000, side/2000, -side/2000, side/2000))
                ax.plot(0, 0, marker=(3, 0, 0), color="#00b4d8" if cc < 3 else "#00e5ff", ms=10)
                ax.set_xticks([])
                ax.set_yticks([])
                lit = float((img > 0).mean())
                ax.set_title(f"{title}\n({side/1000:g} km 지도, 위 = 기수 방향, 켜진 칸 {lit*100:.0f}%)", fontsize=10)
        i0, i1 = a.pair
        dif_t = float(np.abs(terr[i0] - terr[i1]).mean())
        dif_tail = float(np.abs(sn["tails"][i0] - sn["tails"][i1]).mean())
        axs[r, 1].text(0.0, -0.09, f"두 기체 입력이 다른 칸: ④ {dif_t*100:.0f}%  vs  꼬리 채널 평균 차이 {dif_tail*100:.1f}%", transform=axs[r, 1].transAxes,
                       fontsize=10, color="#c0392b")
    fig.suptitle(f"④ 동료 영역 채널이란: 지도 칸마다 '나보다 동료가 더 가까운가' (흰색 = 내 몫, 파랑 = 동료 몫). {a.label}, 지리산 사례 {a.case + 1}, {a.k}대", fontsize=13)
    path = OUT/f"territory_view_{a.label}_k{a.k}.png"
    fig.savefig(path, dpi=85, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
