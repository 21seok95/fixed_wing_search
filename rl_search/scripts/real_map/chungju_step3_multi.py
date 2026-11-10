"""Step 3: several chungju_step3 runs on the same scenarios, paired against the first.

    python -m rl_search.scripts.real_map.chungju_step3_multi --runs step3_flown=10분 step3_flown_tr15=15분 \\
        step3_flown_tr30=30분 --name compare_transit --title "이동 제한 (계획기가 고를 수 있는 가장 먼 사각형)"

Figure: rl_search/data/real/chungju/<last run>/<name>.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
COL = ["#7f8c8d", "#2a4d7a", "#c0392b", "#27ae60", "#8e44ad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dir=label, the first is the reference")
    ap.add_argument("--name", default="compare")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    S = {lab: json.loads((DATA/"chungju"/d/"summary.json").read_text()) for d, lab in runs}
    labs = [lab for _, lab in runs]
    seeds = sorted(int(s) for s in S[labs[0]])
    grid = np.arange(0, 3601, 60)
    curves = {}
    for d, lab in runs:
        cs = []
        for s in seeds:
            o = np.load(DATA/"chungju"/d/f"run_{s}.npz")
            cs.append(np.interp(grid, o["pos_t"], o["pos"], left=0))
        curves[lab] = np.array(cs)

    fig = plt.figure(figsize=(21, 12), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    for c, lab in zip(COL, labs):
        ax.plot(grid/60, curves[lab].mean(0), color=c, lw=2.5, label=f"{lab}: {curves[lab].mean(0)[-1]:.3f}")
    ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel("평균 누적 POS"); ax.legend(); ax.set_title("평균 누적 POS", fontsize=11)
    ax = fig.add_subplot(gs[0, 1])
    ref = curves[labs[0]]
    for c, lab in zip(COL[1:], labs[1:]):
        d = curves[lab] - ref
        m, se = d.mean(0), d.std(0, ddof=1)/np.sqrt(len(seeds))
        ax.plot(grid/60, m, color=c, lw=2.2, label=lab)
        ax.fill_between(grid/60, m - se, m + se, color=c, alpha=.2)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel(f"POS 차이 ({labs[0]} 대비)"); ax.legend()
    ax.set_title(f"시각별 POS 차이 (띠 = 표준오차, 시나리오 {len(seeds)}개 짝지음)", fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    w = .8/(len(labs) - 1)
    for j, (c, lab) in enumerate(zip(COL[1:], labs[1:])):
        d = curves[lab][:, -1] - ref[:, -1]
        ax.bar(np.arange(len(seeds)) + (j - (len(labs) - 2)/2)*w, d, w, color=c,
               label=f"{lab}: {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}, {(d > 0).sum()}/{len(d)}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds); ax.set_xlabel("시나리오"); ax.legend(fontsize=9)
    ax.set_title(f"최종 POS 차이 ({labs[0]} 대비)", fontsize=11)
    keys = [("first_lane_min", "첫 레인 시작 (분)", None), ("lane", "비행 중 레인 비율", 1),
            ("capped", "12분 상한에 걸린 작업 비율", 1)]
    w = .8/len(labs)
    for j, (key, t, ylim) in enumerate(keys):
        ax = fig.add_subplot(gs[1, j])
        for k, (c, lab) in enumerate(zip(COL, labs)):
            v = [S[lab][str(s)][key] for s in seeds]
            ax.bar(np.arange(len(seeds)) + (k - (len(labs) - 1)/2)*w, v, w, color=c, label=f"{lab}: 평균 {np.mean(v):.2f}")
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds); ax.set_xlabel("시나리오"); ax.legend(fontsize=9)
        if ylim:
            ax.set_ylim(0, ylim)
        ax.set_title(t, fontsize=11)
    fig.suptitle(f"{args.title} (앞섬 + 짐벌 8초, 0.7 × 모델 POD, 폭 190 m, 롤 30°, 레인 시간 모의 비행, 시나리오 {len(seeds)}개)", fontsize=13)
    out = DATA/"chungju"/runs[-1][0]/f"{args.name}.png"
    fig.savefig(out, dpi=95)
    print("| 지표 | " + " | ".join(labs) + " |\n|---|" + "---|"*len(labs))
    print("| 최종 POS | " + " | ".join(f"{curves[l][:, -1].mean():.3f}" for l in labs) + " |")
    for m in (15, 30, 45):
        print(f"| POS {m}분 | " + " | ".join(f"{curves[l][:, m].mean():.3f}" for l in labs) + " |")
    for key, t, _ in keys + [("turn_in_lane", "레인 작업 중 선회", 1), ("fp_in_rect", "발자국 중심이 사각형 안", 1)]:
        print(f"| {t} | " + " | ".join(f"{np.mean([S[l][str(s)][key] for s in seeds]):.2f}" for l in labs) + " |")
    for lab in labs[1:]:
        d = curves[lab][:, -1] - ref[:, -1]
        print(lab, "diff", np.round(d, 3), f"{d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
