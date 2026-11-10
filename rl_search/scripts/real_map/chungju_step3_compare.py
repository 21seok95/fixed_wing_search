"""Step 3: two runs of chungju_step3 on the same scenarios, paired per scenario.

    python -m rl_search.scripts.real_map.chungju_step3_compare                      (173 m vs 190 m lane width)
    python -m rl_search.scripts.real_map.chungju_step3_compare --a step3 --b step3_flown \
        --la "레인 시간 기하 근사" --lb "레인 시간 모의 비행" --name compare_flown --title "계획기 레인 시간: 기하 근사 대 모의 비행"
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
COL = ["#7f8c8d", "#c0392b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="step3_173"); ap.add_argument("--b", default="step3")
    ap.add_argument("--la", default="173 m (모델 실효 폭)"); ap.add_argument("--lb", default="190 m (카메라 발자국 폭)")
    ap.add_argument("--name", default="compare_173_190")
    ap.add_argument("--title", default="계획기 레인 폭 173 m 대 190 m")
    args = ap.parse_args()
    RUNS = {args.la: DATA/"chungju"/args.a, args.lb: DATA/"chungju"/args.b}
    S = {k: json.loads((d/"summary.json").read_text()) for k, d in RUNS.items()}
    seeds = sorted(int(s) for s in next(iter(S.values())))
    names = list(RUNS)
    fig = plt.figure(figsize=(20, 11), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    grid = np.arange(0, 3601, 60)
    for c, (k, d) in zip(COL, RUNS.items()):
        cs = []
        for s in seeds:
            o = np.load(d/f"run_{s}.npz")
            cs.append(np.interp(grid, o["pos_t"], o["pos"], left=0))
            ax.plot(grid/60, cs[-1], color=c, lw=.7, alpha=.35)
        ax.plot(grid/60, np.mean(cs, 0), color=c, lw=3, label=f"{k}: 평균 {np.mean(cs, 0)[-1]:.3f}")
    ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel("누적 POS"); ax.legend()
    ax.set_title("누적 POS (가는 선 = 시나리오, 굵은 선 = 평균)", fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    d = np.array([S[names[1]][str(s)]["pos"] - S[names[0]][str(s)]["pos"] for s in seeds])
    ax.bar(range(len(seeds)), d, color=np.where(d >= 0, COL[1], COL[0]))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds); ax.set_xlabel("시나리오")
    ax.set_title(f"최종 POS 차이 (뒤 − 앞): {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}, 뒤가 나은 시나리오 {(d > 0).sum()}/{len(d)}",
                 fontsize=10)
    keys = [("lane", "비행 중 레인 비율"), ("turn_in_lane", "레인 작업 중 선회 비율"),
            ("fp_in_rect", "발자국 중심이 사각형 안"), ("capped", "12분 상한에 걸린 작업")]
    for j, (key, lab) in enumerate(keys):
        ax = fig.add_subplot(gs[1, j] if j < 3 else gs[1, 2])
        if j == 3:
            ax.cla()
        a = [S[names[0]][str(s)][key] for s in seeds]; b = [S[names[1]][str(s)][key] for s in seeds]
        x = np.arange(len(seeds))
        ax.bar(x - .2, a, .4, color=COL[0], label=f"{names[0]}: 평균 {np.mean(a):.2f}")
        ax.bar(x + .2, b, .4, color=COL[1], label=f"{names[1]}: 평균 {np.mean(b):.2f}")
        ax.set_xticks(x); ax.set_xticklabels(seeds); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title(lab, fontsize=11)
        if j == 2:
            break
    fig.suptitle(f"{args.title} (실제 탐지는 통과마다 0.7 × 모델 POD, 앞섬 + 짐벌 8초, 롤 30°, 시나리오 {len(seeds)}개)", fontsize=13)
    out = DATA/"chungju"/args.b/f"{args.name}.png"
    fig.savefig(out, dpi=95)
    print(f"| 지표 | {names[0]} | {names[1]} |\n|---|---|---|")
    for key, lab in [("pos", "최종 POS")] + keys + [("first_lane_min", "첫 레인 시작 (분)")]:
        a = np.mean([S[names[0]][str(s)][key] for s in seeds]); b = np.mean([S[names[1]][str(s)][key] for s in seeds])
        print(f"| {lab} | {a:.3f} | {b:.3f} |")
    print("per-seed diff", np.round(d, 3), f"Saved {out}")


if __name__ == "__main__":
    main()
