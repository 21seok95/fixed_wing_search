"""Two training runs side by side at the same steps from their train logs (user 2026-11-24: "R16 학습과정과 비교해 R17은 어떤가"). No training.

Per logged update: launch distance (the curriculum stage), training flight POS (one UAV for R16, the team for R17), value fit, update size
(approx_kl, clip fraction), action noise (std); R17's team records (time within 1 km, the least-finding UAV's share, safety penalty /
search reward). Thin lines = every update, thick = mean of the last 10 updates; dashed verticals = the launch-stage switches.

    python -m rl_search.scripts.multi_design.run_compare --runs r16_chungju r17_chungju --names "R16 (1대)" "R17 (2대, 팀)" --xmax 420000
Writes report/<--out>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.train_curves import LOGS, OUT, parse, series

COL = ["#8a8f98", "#2a78d6", "#E69F00"]


def roll(y, n=10):
    if len(y) < 2:
        return y
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0.0))
    out = np.empty(len(y))
    for i in range(len(y)):
        j = max(0, i - n + 1)
        out[i] = (c[i + 1] - c[j])/(i + 1 - j)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["r16_chungju", "r17_chungju"])
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--out", default="r16_r17_train")
    a = ap.parse_args()
    names = a.names or a.runs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    data = {t: parse(t) for t in a.runs}
    xmax = a.xmax or min(max(r["time/total_timesteps"] for r in rows) for rows in data.values())
    sw = {}
    for t in a.runs:
        try:
            st = json.loads((LOGS/t/"state.json").read_text())
            sw[t] = [s["step"] for s in st.get("ring", {}).get("switches", [])]
        except (OSError, ValueError):
            sw[t] = []
    panels = [("episode/launch_km", "① 이륙 거리 (km, 끝난 비행 평균) = 커리큘럼 단계", None, False),
              ("episode/POS", "② 학습 비행 POS (R16 = 1대, R17 = 2대 팀)", (0, 0.8), False),
              ("train/explained_variance", "③ 가치 설명력 (1에 가까울수록 보상을 잘 맞힘)", (0, 1.02), True),
              ("train/approx_kl", "④ 업데이트 크기 approx_kl (상한 0.02의 1.5배에서 멈춤)", (0, 0.035), True),
              ("train/clip_fraction", "⑤ 잘린 비율 clip_fraction (정책이 크게 바뀐 몫)", (0, 0.3), True),
              ("train/std", "⑥ 행동 잡음 std (탐색 폭)", (0, 0.4), False),
              ("team_k2/close1km", "⑦ R17: 두 기체가 1 km 안에 있는 시간 비율", (0, 1), False),
              ("team_k2/own_min", "⑧ R17: 덜 찾은 기체의 몫 (0.5면 똑같이)", (0, 0.55), False),
              ("team_k2/pen_share", "⑨ R17: 안전 벌점 ÷ 탐색 보상", (0, 0.6), False)]
    fig, ax = plt.subplots(3, 3, figsize=(19, 11.5), facecolor=BG)
    for x_, (key, title, ylim, smooth) in zip(ax.ravel(), panels):
        for j, t in enumerate(a.runs):
            xs, ys = series(data[t], key)
            m = xs <= xmax
            xs, ys = xs[m], ys[m]
            if not len(xs):
                continue
            c = COL[j % len(COL)]
            if smooth:
                x_.plot(xs/1e4, ys, color=c, lw=0.6, alpha=0.35)
                x_.plot(xs/1e4, roll(ys), color=c, lw=2.2, label=names[j])
            else:
                x_.plot(xs/1e4, ys, color=c, lw=2.0, drawstyle="steps-post" if key.startswith(("episode/", "team_k")) else "default", label=names[j])
            for s in sw[t]:
                if s <= xmax:
                    x_.axvline(s/1e4, color=c, lw=0.9, ls="--", alpha=0.8)
        if ylim:
            x_.set_ylim(*ylim)
        if key == "team_k2/own_min":
            x_.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        x_.set_title(title, loc="left", fontsize=11, color=INK)
        x_.set_facecolor(BG); x_.grid(color=GRID, lw=.5); x_.tick_params(colors=MUTED)
        for sp in x_.spines.values():
            sp.set_color(GRID)
    for x_ in ax[-1]:
        x_.set_xlabel("학습 스텝 (만)", color=MUTED)
    h, l = ax[0, 1].get_legend_handles_labels()
    fig.suptitle(f"같은 스텝에서 학습 과정 비교 (0 ~ {xmax/1e4:.0f}만 스텝; 점선 = 이륙 거리 단계가 넘어간 때)", x=.01, ha="left", fontsize=13.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.legend(h, l, frameon=False, fontsize=11, loc="upper right", ncol=len(h), bbox_to_anchor=(.995, .995))
    fig.savefig(OUT/f"{a.out}.png", dpi=72, facecolor=BG)
    print(OUT/f"{a.out}.png")
    rows = []
    for t in a.runs:                                       # numbers per window for the reply
        for lo, hi in ((0, 1e5), (1e5, 2e5), (2e5, 3.1e5), (3.1e5, xmax)):
            r = {"run": t, "from": lo, "to": hi}
            for key, *_ in panels:
                xs, ys = series(data[t], key)
                m = (xs > lo) & (xs <= hi)
                r[key] = float(np.mean(ys[m])) if m.any() else None
            rows.append(r)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
