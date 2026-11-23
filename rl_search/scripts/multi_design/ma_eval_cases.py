"""Per-case view of multi-UAV evaluations at one team size (user 2026-11-23, M2c: "is it going well?"). No flying: reads ma_eval's
report/ma_eval/<label>.json. Panels: team POS per Jirisan case (hold-out particles, the planner score in the legend table), share of the
flight with two UAVs within 1 km, and team POS over time per case (hold-out).

    python -m rl_search.scripts.multi_design.ma_eval_cases --k 3 --labels r15_6.8M_m2 m2_0.8M m2b_0.4M m2c_0.2M --out m2c_eval_0.2M
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"ma_eval"
NAME = {"r15_6.8M_m2": "R15 680만 각자 (기준선)"}
COL = ["#8a8f98", "#E69F00", "#7fb2ea", "#2a78d6", "#0b4ea2", "#3a9a5b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="ma_eval_cases")
    ap.add_argument("--ramp", action="store_true", help="the first label grey, the rest checkpoints of one run in light -> dark blue")
    a = ap.parse_args()
    global COL
    if a.ramp:
        COL = ["#8a8f98"] + ["#a9c8f0", "#5a9be3", "#2a78d6", "#0b4ea2", "#062f63"][-(len(a.labels) - 1):] if len(a.labels) <= 6 else COL
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    rows = {}
    for lb in a.labels:
        d = json.loads((WORK/f"{lb}.json").read_text())
        rows[lb] = {r["case"]: r for r in d["rows"] if r["k"] == a.k}
    cases = sorted(set.intersection(*(set(r) for r in rows.values())))
    n = len(a.labels)
    fig = plt.figure(figsize=(19, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    axp, axc = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2])
    axt = [fig.add_subplot(gs[1, j]) for j in range(len(cases[:3]))]
    w = 0.8/n
    for j, lb in enumerate(a.labels):
        hold = [rows[lb][c]["holdout"] for c in cases]
        bel = [rows[lb][c]["belief"] for c in cases]
        close = [rows[lb][c]["close1km"] for c in cases]
        xs = np.arange(len(cases) + 1) + (j - (n - 1)/2)*w
        vals = hold + [float(np.mean(hold))]
        axp.bar(xs, vals, w*0.92, color=COL[j % len(COL)], label=f"{NAME.get(lb, lb)}  (평균 {np.mean(hold):.3f}, 계획용 {np.mean(bel):.3f})")
        for x, v in zip(xs, vals):
            axp.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        axc.bar(np.arange(len(cases)) + (j - (n - 1)/2)*w, np.array(close)*100, w*0.92, color=COL[j % len(COL)])
        for ci, c in enumerate(cases[:3]):
            r = rows[lb][c]
            t = np.array(r["t"])/60.0
            axt[ci].plot(t - t[0], r["curve_h"], color=COL[j % len(COL)], lw=2.4 if j == n - 1 else 1.3)
    axp.set_xticks(range(len(cases) + 1), [f"지리산 {c + 1}" for c in cases] + ["평균"])
    axp.set_ylabel("팀 POS (채점용 입자)", color=MUTED); axp.set_ylim(0, 0.8)
    axp.set_title(f"① 사례별 팀 POS, {a.k}대 (채점용 입자로 같은 명령 다시 비행)", loc="left", fontsize=11.5, color=INK)
    axp.legend(frameon=False, fontsize=9.5, loc="upper right")
    axc.set_xticks(range(len(cases)), [f"지리산 {c + 1}" for c in cases]); axc.set_ylim(0, 105)
    axc.set_ylabel("비행 시간 중 비율 (%)", color=MUTED)
    axc.set_title("② 가장 가까운 두 기체가 1 km 안에 있는 시간", loc="left", fontsize=11.5, color=INK)
    for ci, c in enumerate(cases[:3]):
        axt[ci].set_title(f"③ 지리산 {c + 1}: 시간에 따른 팀 POS (채점용)", loc="left", fontsize=11, color=INK)
        axt[ci].set_xlabel("첫 이륙 뒤 분", color=MUTED); axt[ci].set_ylim(0, 0.8)
    axt[0].set_ylabel("팀 POS (누적)", color=MUTED)
    for x_ in [axp, axc, *axt]:
        x_.set_facecolor(BG); x_.grid(color=GRID, lw=.5); x_.tick_params(colors=MUTED)
        for sp in x_.spines.values():
            sp.set_color(GRID)
    fig.suptitle(f"다수 기체 평가 {a.k}대: 지리산 검증 3사례(학습에 안 쓴 지도), 같은 기지 순차 이륙, 무작위성 없는 결정", x=.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/f"{a.out}.png", dpi=72, facecolor=BG)
    print(OUT/f"{a.out}.png")


if __name__ == "__main__":
    main()
