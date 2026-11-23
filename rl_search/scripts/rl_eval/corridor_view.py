"""One validation case flown by SAROPS, by the road-corridor rule (rl_search/corridor.py) and by RL models: the tracks over the
particles at the launch and at the end, the corridors the rule planned (outline, numbered in plan order), and the POS over the
flight. Web runs eval-<split>-s3-k<k>-sarops-<KEY>[-corridor] and eval-<split>-s3-k<k>-rl-<label>.

    python -m rl_search.scripts.rl_eval.corridor_view --split bval --k 0 --runs "R14 도로형만=r14_road-3200000"
Writes rl_search/data/real/amboy/rl_eval/report/corridor/case_<split>_k<k>.png.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rl_search.scripts.rl_eval.case_tracks import KEY, SPLIT_NAME, TN, flight
from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

PLAN = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7A5195", "#B8860B", "#2F9FD8", "#A23B2A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bval")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--runs", nargs="*", default=[], help='"name=<RL web run label>"')
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report/corridor"))
    args = ap.parse_args()
    from rl_search.scripts.real_map.vehicle_web import RUNS
    korean_font()
    base = f"eval-{args.split}-s3-k{args.k}"
    sets = [("SAROPS (사각형)", RUNS/f"{base}-sarops-{KEY}"), ("도로 회랑 규칙", RUNS/f"{base}-sarops-{KEY}-corridor"),
            ("도로 순찰 규칙", RUNS/f"{base}-sarops-{KEY}-patrol"), ("길 따라가기 규칙", RUNS/f"{base}-sarops-{KEY}-roadfollow")]
    sets += [(n, RUNS/f"{base}-rl-{lab}") for n, lab in (s.split("=", 1) for s in args.runs)]
    F = [(n, flight(d), d) for n, d in sets if (d/"run.json.gz").exists()]
    ci = F[0][1]["ci"]
    allxy = np.vstack([np.vstack([f["xy"], f["P0"], f["P1"]]) for _, f, _ in F])
    lo, hi = np.percentile(allxy, 1, axis=0) - 1.5, np.percentile(allxy, 99, axis=0) + 1.5
    fig, axes = plt.subplots(1, len(F) + 1, figsize=(5.4*(len(F) + 1), 5.8), layout="constrained")
    for a, (n, f, d) in zip(axes, F):
        a.scatter(f["P0"][:, 0], f["P0"][:, 1], s=1.5, color="#c9772e", alpha=.35, lw=0, zorder=2)
        a.scatter(f["P1"][:, 0], f["P1"][:, 1], s=1.5, color="#7a52c7", alpha=.25, lw=0, zorder=2)
        rects = json.load(gzip.open(d/"run.json.gz", "rt"))["opt"]["uavs"][0]["rects"]
        for i, r in enumerate(rects):
            c = PLAN[i % len(PLAN)]
            if "outline" in r and "회랑" in n:               # corridors only (the road rules plan every 45 s-2.5 min: too many to number)
                o = np.array(r["outline"])/1000
                a.fill(o[:, 0], o[:, 1], color=c, alpha=.12, zorder=1)
                a.plot(o[:, 0], o[:, 1], color=c, lw=.8, zorder=1)
                m = o[len(o)//4]
                a.text(m[0], m[1], str(i + 1), color=c, fontsize=10, fontweight="bold", zorder=5)
        seg = np.stack([f["xy"][:-1], f["xy"][1:]], axis=1)
        a.add_collection(LineCollection(seg, colors=np.where(f["w"][1:] > 1.49, "#d6336c", "#2a78d6"), lw=1.2, zorder=3))
        a.plot(*f["xy"][0], "o", color=INK, ms=6, zorder=4)
        a.plot(*(np.array(ci["lkp"])/1000), "*", color=INK, ms=12, zorder=4)
        a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="동쪽 (km)", ylabel="북쪽 (km)",
              title=f"{n}: 별도 particle POS {f['hold']:.2f}" + (f"\n회랑 {len(rects)}개 (번호 = 계획 순서)" if "회랑" in n else ""))
        if "순찰" in n:                                    # patrol: the planned paths, not areas
            pass
        style(a)
    a = axes[-1]
    for (n, f, _), c in zip(F, [MUTED, "#e6007e", "#E69F00", "#009E73", "#2a78d6"]):
        x = np.linspace(0, f["t"][-1] - ci["t_launch_s"], len(f["curve"]))/60
        a.plot(x, f["curve"], color=c, lw=2, label=f"{n} ({f['hold']:.2f})")
    a.set(xlabel="이륙 뒤 시간 (분)", ylabel="계획 particle 누적 POS", title="POS가 쌓이는 과정 (괄호: 별도 particle 채점)", ylim=(0, None))
    a.legend(frameon=False, fontsize=9, loc="upper left")
    style(a)
    fig.suptitle(f"{SPLIT_NAME.get(args.split, args.split)} {args.k + 1}번 {TN.get(ci['typ'], ci['typ'])}: 주황 = 이륙 때 particle, 보라 = 비행 끝 particle, "
                 f"색 띠 = 회랑, 파랑 = 비행 경로, 빨강 = 빠른 선회", fontsize=12, color=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out/f"case_{args.split}_k{args.k}.png"
    fig.savefig(p, dpi=90)
    print("saved", p, "|", ", ".join(f"{n} {f['hold']:.3f}" for n, f, _ in F))


if __name__ == "__main__":
    main()
