"""What a UAV sees of its teammates (user 2026-11-20: "먼저 이 채널이 어떻게 보이는지 가시화 해서 확인해보자"): one team flight on a
Jirisan validation case (a model, deterministic), and at chosen minutes the observation of one UAV on all seven maps: the particle channel
(where the particles are now, unweighted) next to the teammate tail channel (teammates' 1 s positions over the map's lead time, value
1 - age / window), with the teammates' true positions marked (they should sit at the bright end of each tail) and the vector entries.

    python -m rl_search.scripts.multi_design.tail_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M \
        --k 4 --case 0 --uav 1 --minutes 3 15 35
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--uav", type=int, default=1, help="whose observation (0 = first launched)")
    ap.add_argument("--minutes", type=float, nargs="+", default=[3.0, 15.0, 35.0], help="minutes after the first launch")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_multi import EXTRA_NAMES, MultiRealEnv
    from rl_search.rl_real import VEC_NAMES, RealRLConfig
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
    want = sorted(a.minutes)
    snaps = []
    while not env.done and want:
        if (env.T - t0)/60.0 >= want[0] - 1e-9:
            i = a.uav
            snaps.append(dict(minute=want.pop(0), obs={key: v.copy() for key, v in obs[i].items()}, pose=env.uavs[i].env.plant.pose.copy(),
                              mates=[(j, env.uavs[j].env.plant.pose[:2].copy(), bool(env.active[j])) for j in range(a.k) if j != i],
                              active=bool(env.active[i])))
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
    maps = env.maps
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    nm = len(maps)
    fig, axs = plt.subplots(2*len(snaps), nm + 1, figsize=(3.0*(nm + 1), 3.25*2*len(snaps)),
                            gridspec_kw=dict(width_ratios=[1]*nm + [1.35]))
    vec_names = list(VEC_NAMES) + [f"map{k + 1}_scale" for k in range(nm)] + list(EXTRA_NAMES)
    for r, sn in enumerate(snaps):
        x, y, yaw = sn["pose"]
        cs, sn_ = np.cos(yaw), np.sin(yaw)
        for k, (name, side, lead) in enumerate(maps):
            o = sn["obs"][name]
            px = o.shape[-1]
            for row, (img, title, cmap) in enumerate(((o[0], "입자 (지금, 가중치 무시)", "Greys"), (o[-1], f"동료 꼬리 (최근 {lead:.0f} s)", "magma"))):
                ax = axs[2*r + row, k]
                ax.imshow(img, cmap=cmap, vmin=0, vmax=max(float(img.max()), 1e-6) if row == 0 else 1.0, interpolation="nearest",
                          extent=(-side/2000, side/2000, -side/2000, side/2000))
                ax.plot(0, 0, marker=(3, 0, 0), color="#00b4d8", ms=7)
                for j, p, act in sn["mates"]:
                    d = p - np.array([x, y])
                    fwd, right = d[0]*cs + d[1]*sn_, d[0]*sn_ - d[1]*cs
                    if abs(fwd) <= side/2 and abs(right) <= side/2 and act:
                        ax.plot(right/1000, fwd/1000, "o", mfc="none", mec=cols[j], ms=8, mew=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0 or row == 0:
                    ax.set_title(f"{side/1000:g} km · {title}" if row == 0 else title, fontsize=8)
                if k == 0:
                    ax.set_ylabel(f"{sn['minute']:g}분\n" + ("입자" if row == 0 else "꼬리"), fontsize=10)
                if row == 1:
                    lit = int((img > 0).sum())
                    ax.text(0.02, 0.02, f"켜진 칸 {lit}/{px*px}", color="w", fontsize=7, transform=ax.transAxes)
        ax = axs[2*r, nm]
        v = sn["obs"]["vec"]
        ax.barh(np.arange(len(v)), v, color=["#bbb"]*(len(v) - len(EXTRA_NAMES)) + ["#e07b39"]*len(EXTRA_NAMES))
        ax.set_yticks(np.arange(len(v)))
        ax.set_yticklabels(vec_names[:len(v)], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(-1.1, 1.1)
        ax.set_title("벡터 (주황 = 동료 항목)", fontsize=8)
        axs[2*r + 1, nm].axis("off")
        axs[2*r + 1, nm].text(0, 0.5, "\n".join(f"{n}: {val:.2f}" for n, val in zip(EXTRA_NAMES, v[-len(EXTRA_NAMES):])), fontsize=8,
                              va="center", transform=axs[2*r + 1, nm].transAxes)
    fig.suptitle(f"동료 꼬리 채널: {a.label}, 지리산 사례 {a.case + 1}, {a.k}대, {a.uav + 1}번째 이륙 기체의 관측 (위가 기수 방향, 하늘색 삼각형 = 자기, "
                 f"동그라미 = 동료의 실제 위치: 1번째 파랑, 2번째 주황, 3번째 초록, 4번째 보라)", fontsize=11)
    path = OUT/f"tail_view_{a.label}_k{a.k}_u{a.uav + 1}.png"
    fig.savefig(path, dpi=85, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
