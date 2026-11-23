"""Training curves of multi-UAV runs side by side from their train logs (user 2026-11-23, M2c: "is it going well?"). No training.

Per logged update: the policy's mean turn command (multi/act_mean, + = left), the share beyond the action range (act_sat), the value
fit (train/explained_variance), the update size (approx_kl, clip_fraction), the action noise (std), and the finished-flight summaries
(team_k3/POS, ep_rew_mean, own_min, close1km, pen_share; these change only when flights end).

    python -m rl_search.scripts.multi_design.train_curves --runs m2_multi m2b_multi m2c_multi --xmax 1300000
Writes report/<--out>.png.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LOGS = Path("rl_search/data/rl_real")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
LABEL = {"m2_multi": "M2 (2·3·4대 9세계, γ 0.999)", "m2b_multi": "M2b (M2 80만에서, γ 0.995)", "m2c_multi": "M2c (3대 20세계, γ 0.995)"}
COL = {"m2_multi": "#E69F00", "m2b_multi": "#7fb2ea", "m2c_multi": "#2a78d6"}


def parse(tag):
    rows, sec, cur = [], "", {}
    for line in (LOGS/f"{tag}_train.log").read_text(errors="ignore").splitlines():
        if line.startswith("-----"):
            if "time/total_timesteps" in cur:
                rows.append(cur)
            cur, sec = {}, ""
            continue
        m = re.match(r"\|\s*(\S+)\s*\|\s*(\S*)\s*\|", line)
        if not m:
            continue
        key, val = m.groups()
        if key.endswith("/"):
            sec = key
        elif val:
            try:
                cur[sec + key] = float(val)
            except ValueError:
                pass
    seen, out = set(), []
    for r in rows:                                   # SB3 prints the same table twice after a checkpoint line: keep the first
        t = r["time/total_timesteps"]
        if t not in seen:
            seen.add(t); out.append(r)
    return out


def series(rows, key):
    x = np.array([r["time/total_timesteps"] for r in rows if key in r])
    y = np.array([r[key] for r in rows if key in r])
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["m2_multi", "m2b_multi", "m2c_multi"])
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--out", default="m2c_train_curves")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    data = {t: parse(t) for t in a.runs}
    panels = [("multi/act_mean", "① 평균 선회 명령 (+ 왼쪽, − 오른쪽)", (-0.6, 0.6)),
              ("multi/act_sat", "② 범위(±1) 밖 비율 (20%면 자동 멈춤)", (0, 0.3)),
              ("train/explained_variance", "③ 가치 설명력 (1에 가까울수록 보상을 잘 맞힘)", (0, 1)),
              ("train/approx_kl", "④ 업데이트 크기 approx_kl (상한 0.02 × 1.5)", (0, 0.035)),
              ("train/std", "⑤ 행동 잡음 std", (0, 0.35)),
              ("team_k3/POS", "⑥ 3대 팀 POS (학습 중, 끝난 비행 평균)", (0, 0.8)),
              ("team_k3/close1km", "⑦ 3대: 동료와 1 km 안에 있는 시간 비율", (0, 1)),
              ("team_k3/own_min", "⑧ 3대: 가장 적게 찾은 기체의 몫 (1/3이면 고르게)", (0, 0.4)),
              ("team_k3/pen_share", "⑨ 3대: 안전 벌점 ÷ 탐색 보상", (0, 0.5))]
    fig, ax = plt.subplots(3, 3, figsize=(19, 11.5), facecolor=BG)
    for x_, (key, title, ylim) in zip(ax.ravel(), panels):
        for t, rows in data.items():
            xs, ys = series(rows, key)
            if a.xmax is not None:
                ys, xs = ys[xs <= a.xmax], xs[xs <= a.xmax]
            if len(xs):
                step = key.startswith("team_k") or key.startswith("rollout/")
                x_.plot(xs/1e6, ys, color=COL.get(t, INK), lw=2.4 if t == a.runs[-1] else 1.3, drawstyle="steps-post" if step else "default",
                        label=LABEL.get(t, t), alpha=1 if t == a.runs[-1] else .85)
        if key == "multi/act_mean":
            x_.axhspan(-0.15, 0.15, color=GRID, alpha=.6, lw=0); x_.axhline(0, color=MUTED, lw=.6)
        if key == "multi/act_sat":
            x_.axhline(0.2, color="#d6336c", lw=1, ls="--")
        if key == "team_k3/own_min":
            x_.axhline(1/3, color=MUTED, lw=.8, ls=":")
        x_.set_ylim(*ylim); x_.set_title(title, loc="left", fontsize=11, color=INK)
        x_.set_facecolor(BG); x_.grid(color=GRID, lw=.5); x_.tick_params(colors=MUTED)
        for sp in x_.spines.values():
            sp.set_color(GRID)
    for x_ in ax[-1]:
        x_.set_xlabel("학습 스텝 (백만, 각 학습의 시작부터)", color=MUTED)
    h, l = ax[1, 2].get_legend_handles_labels()                 # the POS panel has every run (M2 logged no act_mean / act_sat)
    fig.suptitle("다수 기체 학습 곡선: M2c(굵은 파랑)와 앞선 M2·M2b (모두 R15 또는 그 이어받기에서 출발, 학습 로그)", x=.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.legend(h, l, frameon=False, fontsize=10.5, loc="upper right", ncol=len(h), bbox_to_anchor=(.995, .995))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f"{a.out}.png", dpi=72, facecolor=BG)
    print(OUT/f"{a.out}.png")


if __name__ == "__main__":
    main()
