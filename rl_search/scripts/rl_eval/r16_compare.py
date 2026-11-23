"""R16 against R15 (user 2026-11-24 "학습 결과는 어떤거 같아?"): the same training A (Chungju 120 km road type, the same pool order, launch-distance
stages, Jirisan v2j validation every 400k); R16 = the multi-UAV env (design m2) with one UAV, the reward / remaining probability x 1.25, 8-bit
maps, the symmetric policy. Four panels: (A) Jirisan validation, ratio to SAROPS, with the launch-stage switches; (B) the three validation
cases' POS; (C) training launch POS per block; (D) the policy's action noise std and R16's mean action (one-sided turning).

    python -m rl_search.scripts.rl_eval.r16_compare
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RUNS = (("r15_chungju", "R15 (1대 환경, 대칭 없음)", "#8a94a0", "--"), ("r16_chungju", "R16 (다수 기체 환경 1대, 대칭)", "#2b6cb0", "-"))
OUT = Path("rl_search/data/real/amboy/rl_eval/report")


def _tb(run: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    xs, ys = [], []
    for d in sorted(glob.glob(f"rl_search/data/rl_real/{run}/tb/ppo_*")):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                xs.append(e.step)
                ys.append(e.value)
    o = np.argsort(xs)
    return np.array(xs)[o]/1e6, np.array(ys)[o]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    a, b, c, d = ax.ravel()
    rows = {}
    for run, name, col, ls in RUNS:
        s = json.loads(Path(f"rl_search/data/rl_real/{run}/state.json").read_text())
        h = [x for x in s["history"]["3"] if x["step"] <= 6_000_000 + 1000]
        st = np.array([x["step"] for x in h])/1e6
        ratio = np.array([x["ratio"] for x in h])
        rl = np.array([x["rl"] for x in h])
        rows[run] = dict(st=st, ratio=ratio, rl=rl)
        a.plot(st, ratio, ls, color=col, marker="o", ms=4, label=name)
        for sw in s["ring"]["switches"]:
            a.axvline(sw["step"]/1e6, color=col, ls=ls, lw=1, alpha=0.6)
        for k, mk in zip(range(3), ("o", "s", "^")):
            b.plot(st, rl[:, k], ls, color=col, marker=mk, ms=4, label=f"{name.split(' ')[0]} 사례 {k + 1}")
        bl = [x for x in s["ring"]["blocks"] if x["step"] <= 6_000_000 + 1000]
        c.plot([x["step"]/1e6 for x in bl], [x["pos"] for x in bl], ls, color=col, marker="o", ms=4, label=name)
        for x in bl:
            if x["stage"] < 2:
                c.annotate(f"{x['stage'] + 1}단계", (x["step"]/1e6, x["pos"]), textcoords="offset points", xytext=(0, 7), fontsize=8, color=col, ha="center")
        xs, ys = _tb(run, "train/std")
        d.plot(xs, ys, ls, color=col, label=f"{name.split(' ')[0]} 행동 잡음 std")
    xs, ys = _tb("r16_chungju", "multi/act_mean")
    d.plot(xs, ys, color="#c05621", lw=0.8, label="R16 평균 행동 (+ 왼쪽, - 오른쪽)")
    xs, ys = _tb("r16_chungju", "multi/act_sat")
    d.plot(xs, ys, color="#38a169", lw=0.8, label="R16 범위 밖 평균 행동 비율")
    d.axhline(0, color="k", lw=0.5)
    a.set_title("(A) 지리산 검증 3사례: SAROPS 대비 비율 (세로선 = 이륙 거리 단계 넘김)")
    a.set_ylabel("RL POS ÷ SAROPS POS")
    a.legend(fontsize=9, loc="lower right")
    b.set_title("(B) 같은 검증의 사례별 POS (실선 R16, 점선 R15)")
    b.set_ylabel("POS (채점용 입자)")
    b.legend(fontsize=8, ncol=2, loc="lower right")
    c.set_title("(C) 학습 비행의 평균 POS (블록마다 약 700비행; 1·2단계는 10 km / 10~20 km 이륙)")
    c.set_ylabel("학습 비행 POS")
    c.legend(fontsize=9, loc="lower right")
    d.set_title("(D) 행동 잡음과 R16의 한쪽 쏠림")
    d.set_ylim(-0.1, 0.4)
    d.set_xlim(-0.1, 6.2)
    d.legend(fontsize=8, loc="upper right")
    for x in ax.ravel():
        x.set_xlabel("학습 스텝 (백만)")
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT/"r16_compare.png", dpi=110)
    # table: the same steps, from 1.6M (both in the last stage)
    r15, r16 = rows["r15_chungju"], rows["r16_chungju"]
    common = sorted(set(np.round(r15["st"], 1)) & set(np.round(r16["st"], 1)))
    print("step  R15 ratio  R16 ratio  diff | R15 cases            | R16 cases")
    for x in common:
        i, j = list(np.round(r15["st"], 1)).index(x), list(np.round(r16["st"], 1)).index(x)
        print(f"{x:4.1f}  {r15['ratio'][i]:.3f}      {r16['ratio'][j]:.3f}      {r16['ratio'][j] - r15['ratio'][i]:+.3f} | "
              f"{np.round(r15['rl'][i], 3)} | {np.round(r16['rl'][j], 3)}")
    print(OUT/"r16_compare.png")


if __name__ == "__main__":
    main()
