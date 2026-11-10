"""Figure of the SAROPS references on the RL validation cases: per case the particles at the start time, the LKP, the start and the
flown path, with the hold-out score in the title. Reads the saved web runs (rl_eval.sarops_ref) and writes
rl_search/data/real/amboy/rl_eval/sarops_ref_<split>.png.

    python -m rl_search.scripts.rl_eval.ref_figure --split val
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rl_search.rl_eval_set import EVAL_DIR, REGION
from rl_search.scripts.real_map.vehicle_web import RUNS

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, BG = "#1f1f1e", "#6b6b67", "#fcfcfb"
PATH_C, PART_C = "#2a6db0", "#c9772e"
STAGE_NAME = {1: "1단계: 분포 근처 1~3 km에서 시작", 2: "2단계: 분포 밖 3~10 km에서 시작", 3: "3단계: 8~20 km 고리에서 이륙"}


def load(rid):
    d = json.load(gzip.open(RUNS/rid/"run.json.gz", "rt"))
    return d["opt"], d["res"], d["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    rows = {r["id"]: r for r in json.loads((EVAL_DIR/f"sarops_ref_{args.split}.json").read_text())}
    stages = sorted({r["case"]["stage"] for r in rows.values()})
    ks = sorted({r["case"]["k"] for r in rows.values()})
    g = np.load(f"rl_search/data/real/{REGION}/road_graph_25m.npz")
    p_, o_ = g["edge_poly"].astype(float)/1000, g["edge_poly_offset"]
    roads = [p_[o_[k]:o_[k + 1]] for k in range(len(o_) - 1)]
    fig, axs = plt.subplots(len(stages), len(ks), figsize=(3.6*len(ks), 3.9*len(stages) + .6), layout="constrained", facecolor=BG, squeeze=False)
    for i, st in enumerate(stages):
        for j, k in enumerate(ks):
            ax = axs[i, j]
            rid = f"eval-{args.split}-s{st}-k{k}-sarops"
            if rid not in rows:
                ax.axis("off")
                continue
            r = rows[rid]
            opt, res, meta = load(rid)
            c = r["case"]
            T, N = res["T"], res["N"]
            P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(T, N, 2).astype(float)*res["pos_unit_m"]/1000
            S = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(T, N)
            kt = min(int(round(c["t0_s"]/res["dt_s"])), T - 1)
            pts = P[kt][S[kt] != 4]
            U = opt["uavs"][0]
            xy = np.frombuffer(base64.b64decode(U["xy"]), np.float32).reshape(-1, 2)/1000
            box = np.vstack([pts, xy, [c["lkp"][0]/1000, c["lkp"][1]/1000], [c["start"][0]/1000, c["start"][1]/1000]])
            lo, hi = np.percentile(box, 1, axis=0), np.percentile(box, 99, axis=0)
            mid, half = (lo + hi)/2, max((hi - lo).max()/2 + 1.0, 2.0)
            ax.set_facecolor(BG)
            ax.add_collection(LineCollection(roads, colors="#d9d8d3", linewidths=.7, zorder=1))
            ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=PART_C, alpha=.35, lw=0, zorder=2)
            ax.plot(xy[:, 0], xy[:, 1], color=PATH_C, lw=1.1, zorder=3)
            ax.plot(*np.array(c["lkp"])/1000, marker="*", ms=12, color="#ffd400", mec=INK, mew=.8, zorder=5)
            ax.plot(c["start"][0]/1000, c["start"][1]/1000, marker="^", ms=8, color=INK, mec="white", mew=.8, zorder=5)
            ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half); ax.set_aspect("equal")
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
            for sp in ax.spines.values():
                sp.set_color("#d9d8d3")
            h = r["holdout"]
            ax.set_title(f"시나리오 {k + 1} · 시작 {c['t0_s']/60:.0f}분\nA + POS {h['total']:.3f} (POS {h['pos']:.3f}, A {h['A']:.3f})", fontsize=10, color=INK, loc="left")
        axs[i, 0].set_ylabel(STAGE_NAME.get(st, f"{st}단계"), fontsize=11, color=INK)
    fig.suptitle(f"RL 검증 세트의 SAROPS 기준값 (암보이, 1대, A + POS, 짐벌 정면 고정)\n주황 = 시작 시각의 particle, 파랑 = SAROPS 경로, ★ = LKP, ▲ = 시작점. 점수는 계획에 쓰지 않은 별도 particle로 잼",
                 fontsize=13, color=INK)
    out = EVAL_DIR/f"sarops_ref_{args.split}.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
