"""How big is each reward term under the reward agreed for the next multi-UAV run (user 2026-11-23)? No training.

Agreed search reward of UAV i per step: reward_scale x pos_w x (team found + BETA x found by i) / remaining probability (before the step),
gamma 0.999; the other terms unchanged: look reward A' (reward_scale x look_w x i's look count), safety penalty (from 250 m, while closing),
violations (P_HARD each), filter cut (mask_cost x cut). Teams of 2, 3, 4 on the three Jirisan cases flown by R15 6.8M on its own inputs
(deterministic, so clumped: a hard test for the penalties), the M2 environment (rl_multi design "m2", filter on).
Per term: the share of the summed size of all terms over a flight (undiscounted and discounted from the launch with gamma 0.999), the
largest single-step value, and the same for the old M2 reward (own share, not divided, gamma 0.995) for comparison.

    python -m rl_search.scripts.multi_design.reward_balance --workers 3
Writes report/reward_balance.png and .json.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
R15 = Path("rl_search/data/rl_real/r15_chungju")
BETA = 0.25
TERMS = ("탐색: 팀", "탐색: 내 몫 × 0.25", "주시 A′", "안전 벌점", "위반", "잘린 양")


def fly(job):
    case_k, k = job
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import P_HARD, MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.multi_design.ma_eval import _gaps
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    c = json.loads((R15/"config.json").read_text())
    s = settings_of(c) | {"look_w": float(c.get("look_w") or 0.0)}     # settings_of leaves out look_w (train_rl adds it): A' on as in training
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s), seed=case_k, design="m2")
    env.reset(options=case.options(scen), gaps=_gaps(case_k, k, "m2"))
    model = PPO.load(str(R15/"ckpt/model_6800000.zip"), device="cpu")
    cfg = env.uavs[0].cfg
    sc, pw, lw, mc = float(cfg.reward_scale), float(env.uavs[0].env.pos_w), float(env.uavs[0].look_w), float(cfg.mask_cost)
    rows = []                                      # per step: t, active (k), team, rem, seen (k), share (k), look (k), pen (k), viol (k), cut (k)
    while not env.done:
        rem = float(env.W.sum()/env.N)
        obs = [env.base_observation(i) for i in range(k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        _, _, _, info = env.step([np.atleast_1d(a) for a in np.ravel(acts)])
        rows.append(dict(t=float(env.T), act=np.asarray(info["active"], bool), team=float(info["team"]), rem=rem,
                         seen=np.asarray(info["seen"], float), share=np.asarray(info["share"], float), look=np.asarray(info["look"], float),
                         pen=np.asarray(info["penalty"], float), viol=np.asarray(info["violation"], float), cut=np.asarray(info["cut"], float)))
    t0 = min(r["t"] for r in rows)
    out = dict(case=case_k, k=k, terms_new=[], terms_old=[], t=[], found=float(env.found))
    for r in rows:
        a = r["act"]
        new = np.stack([sc*pw*r["team"]/max(r["rem"], 1e-9)*a, sc*pw*BETA*r["seen"]/max(r["rem"], 1e-9)*a, sc*lw*r["look"]*a,
                        -r["pen"]*a, -P_HARD*r["viol"]*a, -mc*r["cut"]*a])                               # (terms, k)
        old = np.stack([np.zeros(k), sc*pw*r["share"]*a, sc*lw*r["look"]*a, -r["pen"]*a, -P_HARD*r["viol"]*a, -mc*r["cut"]*a])
        out["terms_new"].append(new.tolist()); out["terms_old"].append(old.tolist()); out["t"].append(r["t"] - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(c, k) for c in (0, 1, 2) for k in (2, 3, 4)]
    with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(fly, jobs, chunksize=1)
    summary = {}
    for kind, gamma in (("new", 0.999), ("old", 0.995)):
        for k in (2, 3, 4):
            Rs = [r for r in res if r["k"] == k]
            und, dis, peak = [], [], []
            for r in Rs:
                T = np.array(r[f"terms_{kind}"])                 # (steps, terms, k)
                t = np.array(r["t"])
                disc = gamma**(t/5.0)
                und.append(np.abs(T).sum((0, 2))); dis.append((np.abs(T)*disc[:, None, None]).sum((0, 2))); peak.append(np.abs(T).max((0, 2)))
            und, dis = np.mean(und, 0), np.mean(dis, 0)
            summary[f"{kind} {k}대"] = dict(share_undiscounted=(und/und.sum()).round(4).tolist(), share_discounted=(dis/dis.sum()).round(4).tolist(),
                                           per_uav_return=float(np.mean([np.array(r[f"terms_{kind}"]).sum((0,)).sum(0).mean() for r in Rs])),
                                           peak=np.max(peak, 0).round(3).tolist())
    (OUT/"reward_balance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    figure(res, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def figure(res, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = ["#2a78d6", "#7fb2ea", "#E69F00", "#d6336c", "#7a0f3a", "#6b6b67"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5), facecolor=BG)
    for a_, (kind, key, title) in zip(ax[:2], (("new", "share_discounted", "새 보상 (γ 0.999로 할인한 크기의 몫)"),
                                                ("old", "share_discounted", "지금 M2 보상 (자기 몫, γ 0.995)"))):
        bottom = np.zeros(3)
        for j, term in enumerate(TERMS):
            v = np.array([S[f"{kind} {k}대"][key][j]*100 for k in (2, 3, 4)])
            a_.bar(range(3), v, 0.6, bottom=bottom, color=COL[j], label=term)
            for x, (b, h) in enumerate(zip(bottom, v)):
                if h >= 3:
                    a_.text(x, b + h/2, f"{h:.0f}%", ha="center", va="center", fontsize=9, color="white" if j in (0, 3, 4) else INK)
            bottom += v
        a_.set_xticks(range(3), ["2대", "3대", "4대"]); a_.set_ylim(0, 100); a_.set_ylabel("보상 항 크기의 몫 (%)", color=MUTED)
        a_.set_title(title, loc="left", fontsize=11.5, color=INK); a_.legend(frameon=False, fontsize=8.5, loc="lower right")
    a_ = ax[2]
    r = next(x for x in res if x["k"] == 4 and x["case"] == 0)
    T = np.array(r["terms_new"]); t = np.array(r["t"])/60
    for j, term in enumerate(TERMS):
        y = T[:, j, :].sum(1)
        if np.abs(y).max() > 0:
            a_.plot(t, y, color=COL[j], lw=1.2, label=term)
    a_.set_xlabel("이륙 뒤 분", color=MUTED); a_.set_ylabel("걸음 보상 (팀 4대 합)", color=MUTED)
    a_.set_title("새 보상의 걸음별 크기 (지리산 1번, 4대)", loc="left", fontsize=11.5, color=INK); a_.legend(frameon=False, fontsize=8.5)
    for x in ax:
        x.set_facecolor(BG); x.grid(color=GRID, lw=.5); x.tick_params(colors=MUTED)
        for sp in x.spines.values():
            sp.set_color(GRID)
    fig.suptitle("보상 항의 균형: 새 보상 (팀 + 0.25 × 내 몫, ÷ 남은 확률) 대 지금 M2 (R15 680만 각자, 지리산 3개, 학습 없음)", x=.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"reward_balance.png", dpi=78, facecolor=BG)


if __name__ == "__main__":
    main()
