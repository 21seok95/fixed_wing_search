"""Jirisan 120 km extra validation cases, kept apart from the training-time evaluation (user 2026-11-18).

The training evaluates on v2j cases k = 0-2 (rl_eval_set.MULTI_N["v2j"] = 3). For a larger check when needed, this freezes 15 more cases of
the same split rule (make_case(vs, "v2j", 3, k), road type, mountain-terrain rules, launch at the report time 10-40 km) with k = 3-17, so they
never overlap the training-time ones. Each case is pickled as (case, scen, vres, hold), the ``built`` argument of
rl_real_eval.evaluate_case, so later code changes do not alter them; cases.json lists them and a figure shows them.

    python -m rl_search.scripts.rl_eval.v2j_extra make --workers 5
    python -m rl_search.scripts.rl_eval.v2j_extra figure
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SPLIT, STAGE = "v2j", 3
KS = tuple(range(3, 18))
OUT = Path("rl_search/data/real/jirisan120/rl_eval/v2j_extra")


def _vs():
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import split_region, vehicle_config
    from rl_search.rl_real import preload
    region = split_region(SPLIT)
    st = preload(region)
    return VehicleScenarios(st["sampler"], vehicle_config(), region=region, net=st["net"])


_VS = {}


def _init():
    _VS["vs"] = _vs()


def one(k):
    warnings.filterwarnings("ignore")
    from rl_search.rl_eval_set import case_dict, make_case
    from rl_search.scripts.rl_eval.branch_check import branches
    built = make_case(_VS["vs"], SPLIT, STAGE, k)
    case, scen = built[0], built[1]
    with open(OUT/f"case_k{k:02d}.pkl", "wb") as f:
        pickle.dump(built, f, protocol=pickle.HIGHEST_PROTOCOL)
    tr, dt = scen.track, float(scen.save_dt_s)
    L = np.asarray(scen.lkp, float)
    m = {}
    for mn in (30, 60):
        P = tr[min(int(round(mn*60/dt)), tr.shape[0] - 1)].astype(float)
        b, e = branches(P)
        m[str(mn)] = dict(big=b, eff=e, r95=float(np.percentile(np.hypot(*(P - L).T), 95)/1000),
                          out=float(1 - (np.abs(P) <= 60000).all(axis=1).mean()))
    snap = tr[min(int(round(3600/dt)), tr.shape[0] - 1), ::25].astype(np.float32).tolist()
    return dict(case=case_dict(case), m=m, snap=snap)


def make(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    _init()
    with get_context("fork").Pool(workers) as pool:
        rows = sorted(pool.map(one, KS, chunksize=1), key=lambda r: r["case"]["k"])
    (OUT/"cases.json").write_text(json.dumps(dict(split=SPLIT, stage=STAGE, ks=list(KS), note="k 0-2 are the training-time v2j cases",
                                                  rows=rows), ensure_ascii=False))
    for r in rows:
        c = r["case"]
        print(c["k"], c["typ"], "seed", c["seed"], "launch %.1f km" % (c["info"]["launch_dist_m"]/1000), "60 min", r["m"]["60"], flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from rl_search.rl_real import preload
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    d = json.loads((OUT/"cases.json").read_text())
    rows = d["rows"]
    net = preload("jirisan120")["net"]
    xy = net.node_xy/1000
    segs = np.stack([xy[net.de_from[0::2]], xy[net.de_to[0::2]]], 1)
    fig = plt.figure(figsize=(22, 15), layout="constrained")
    gs = fig.add_gridspec(3, 6)
    ax = fig.add_subplot(gs[:2, :2])
    ax.add_collection(LineCollection(segs, colors="#c8c8c8", linewidths=.25))
    for r in rows:
        c = r["case"]; L = np.array(c["lkp"])/1000; S = np.array(c["start"][:2])/1000
        ax.plot([S[0], L[0]], [S[1], L[1]], color="#888", lw=.8, ls="--")
        ax.plot(*L, "*", color="#d6336c", ms=11); ax.plot(*S, "^", color="#2a78d6", ms=7)
        ax.text(L[0] + 1, L[1] + 1, str(c["k"]), fontsize=9, weight="bold")
    ax.add_patch(plt.Rectangle((-15, -15), 30, 30, fill=False, color="k", lw=1, ls=":"))
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect(1)
    ax.set_title("사례 위치 (★ LKP, ▲ 이륙점, 점선 상자 = LKP 범위 중앙 30 km)", loc="left")
    ax = fig.add_subplot(gs[2, 0])
    ax.hist([r["case"]["info"]["launch_dist_m"]/1000 for r in rows], bins=np.arange(10, 42, 3), color="#2a78d6")
    ax.set_xlabel("LKP에서 이륙점까지 (km)"); ax.set_title("이륙 거리", loc="left")
    ax = fig.add_subplot(gs[2, 1])
    ks = [r["case"]["k"] for r in rows]
    ax.bar(np.arange(len(rows)) - .2, [r["m"]["60"]["big"] for r in rows], .4, color="#d6336c", label="갈래 (5% 넘는 조각)")
    ax.bar(np.arange(len(rows)) + .2, [r["m"]["60"]["eff"] for r in rows], .4, color="#f0a6c0", label="유효 갈래")
    ax.set_xticks(range(len(rows)), ks, fontsize=8); ax.set_xlabel("사례 k"); ax.legend(fontsize=8); ax.set_title("60분 갈래", loc="left")
    for j, r in enumerate(rows[:12]):
        ax = fig.add_subplot(gs[j//4, 2 + j % 4])
        c = r["case"]; L = np.array(c["lkp"])/1000; S = np.array(c["start"][:2])/1000; P = np.array(r["snap"])/1000
        ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=.3))
        ax.scatter(P[:, 0], P[:, 1], s=3, color="#d6336c")
        ax.plot(*L, "k*", ms=10); ax.plot(*S, "^", color="#2a78d6", ms=8)
        ax.set_xlim(L[0] - 50, L[0] + 50); ax.set_ylim(L[1] - 50, L[1] + 50); ax.set_aspect(1); ax.tick_params(labelsize=7)
        m = r["m"]["60"]
        ax.set_title(f"k{c['k']}: 이륙 {c['info']['launch_dist_m']/1000:.0f} km, 갈래 {m['big']} (유효 {m['eff']:.1f}), R95 {m['r95']:.0f} km", fontsize=9)
    fig.suptitle("지리산 120 km 별도 검증 사례 v2j k=3~17 (도로형, 산악 지형 규칙, 신고 시각 이륙 10~40 km; 작은 그림 = 60분 입자, 앞 12개)", fontsize=14)
    fig.savefig(OUT/"v2j_extra.png", dpi=80)
    print("saved", OUT/"v2j_extra.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("make", "figure"))
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    make(a.workers) if a.cmd == "make" else figure()
