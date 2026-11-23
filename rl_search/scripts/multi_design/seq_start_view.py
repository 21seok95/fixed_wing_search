"""Figure for choosing where the sequential + symmetric run starts (user 2026-11-23: "순차 결정과 좌우 대칭을 같이 넣어서 이어서 학습 ...
끝나는 시간 보고 어느 모델에 이어서 할지 결정"): (A) the candidate checkpoints, 4 UAVs on the three Jirisan cases, as trained and with the
symmetric policy (stored mirror_check flights); (B) training speed of one configuration each, two 3-UAV worlds (the speed check of 11-23);
(C) when 1 / 2 / 3 million steps are reached if the run starts now (M2c stopped) or after M2c's 3 million.

    python -m rl_search.scripts.multi_design.seq_start_view --speeds 41 40 30 --full-fps 147 --now 18:20 --m2c-end 22:42
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT

MIRROR = OUT/"mirror_check"
CANDS = (("R15", "R15 680만\n(1대 학습)"), ("M2b_0.6M", "M2b 60만"), ("M2b_1.0M", "M2b 100만"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs=3, required=True, help="fps: plain, symmetric, sequential + symmetric (two worlds)")
    ap.add_argument("--full-fps", type=float, required=True, help="M2c's training speed (20 worlds)")
    ap.add_argument("--now", required=True)
    ap.add_argument("--m2c-end", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.2, 0.9, 1.4]))

    # (A) candidates: raw vs symmetric, per case dots and the mean
    x = np.arange(len(CANDS))
    for off, mode, col, name in ((-0.18, "raw", "#9aa5b1", "학습된 그대로"), (0.18, "sym", "#2b6cb0", "좌우 대칭 정책으로")):
        vals = [[json.loads((MIRROR/f"{lab}_{mode}_k4_c{c}.json").read_text())["found"] for c in range(3)] for lab, _ in CANDS]
        m = [np.mean(v) for v in vals]
        ax[0].bar(x + off, m, 0.34, color=col, label=name)
        for xi, v in zip(x + off, vals):
            ax[0].scatter([xi]*3, v, s=14, color="k", zorder=3)
        for xi, mi in zip(x + off, m):
            ax[0].text(xi, 0.03, f"{mi:.2f}", ha="center", fontsize=10, color="white", fontweight="bold")
    ax[0].set_xticks(x, [n for _, n in CANDS])
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel("팀 POS (4대, 지리산 3사례)")
    ax[0].set_title("(A) 시작 후보 모델: 막대 = 3사례 평균, 점 = 사례 하나")
    ax[0].legend(loc="upper right", fontsize=9)

    # (B) speed
    names = ["지금 방식\n(M2c와 같음)", "좌우 대칭만", "순차 결정\n+ 좌우 대칭"]
    cols = ["#9aa5b1", "#7aa2c7", "#2b6cb0"]
    ax[1].bar(range(3), a.speeds, color=cols)
    for i, v in enumerate(a.speeds):
        ax[1].text(i, v + 0.8, f"초당 {v:.0f}\n({v/a.speeds[0]*100:.0f}%)", ha="center", fontsize=9)
    ax[1].set_xticks(range(3), names)
    ax[1].set_ylim(0, max(a.speeds)*1.3)
    ax[1].set_ylabel("학습 속도 (초당 스텝, 3대 세계 2개 시험)")
    ax[1].set_title("(B) 속도: 같은 조건 짧은 시험")

    # (C) timeline
    fps = a.full_fps*a.speeds[2]/a.speeds[0]
    today = datetime.now().date()
    t_now = datetime.combine(today, datetime.strptime(a.now, "%H:%M").time())
    t_end = datetime.combine(today, datetime.strptime(a.m2c_end, "%H:%M").time())
    rows = (("지금 시작\n(M2c 멈춤)", t_now), ("M2c 끝난 뒤\n시작", t_end))
    for r, (name, t0) in enumerate(rows):
        start = t0 + timedelta(minutes=5)
        ends = [start + timedelta(seconds=n*1e6/fps) for n in (1, 2, 3)]
        ax[2].plot([t0, ends[-1]], [r, r], color="#2b6cb0", lw=6, solid_capstyle="butt", alpha=0.35)
        for n, e in zip((1, 2, 3), ends):
            ax[2].scatter([e], [r], color="#2b6cb0", s=40, zorder=3)
            ax[2].text(e, r + 0.18, f"{n}00만\n{e:%H:%M}" if e.date() == today else f"{n}00만\n익일 {e:%H:%M}", ha="center", fontsize=9)
    ax[2].axvline(t_end, color="#c05621", ls="--", lw=1)
    ax[2].text(t_end, -0.55, f"M2c 300만 끝 {a.m2c_end}", color="#c05621", ha="center", fontsize=9)
    ax[2].set_yticks(range(len(rows)), [n for n, _ in rows])
    ax[2].set_ylim(-0.8, len(rows) - 0.2)
    import matplotlib.dates as mdates
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[2].set_title(f"(C) 일정: 예상 속도 초당 약 {fps:.0f} (M2c {a.full_fps:.0f} x {a.speeds[2]/a.speeds[0]:.2f})")
    ax[2].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT/"seq_start_choice.png"
    fig.savefig(path, dpi=110)
    print(path, f"estimated fps {fps:.0f}")


if __name__ == "__main__":
    main()
