"""Figures of an RL training run (scripts/rl_eval/train_rl.py): validation score against SAROPS over training with the curriculum stages,
training episode curves, per-case scores of the last evaluation, and SAROPS / RL paths on the validation cases of the last stage.

    python -m rl_search.scripts.rl_eval.train_report --tag amboy_v1
Writes <out>/<tag>/report_*.png.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager as _fm
_NAMES = {f.name for f in _fm.fontManager.ttflist}


def korean_font():
    """Mac or Ubuntu Korean font (again after importing the web code, whose tuner sets AppleGothic only)."""
    plt.rcParams["font.family"] = [n for n in ("AppleGothic", "Noto Sans CJK KR") if n in _NAMES][:1] + ["sans-serif"]


korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_RL, C_SAR = "#eb6834", "#2a78d6"
STAGE_BG = {1: "#f3f2ee", 2: "#e9f1fb", 3: "#eef7f2"}
STAGE_NAME = {1: "1단계: 분포 근처 1~3 km", 2: "2단계: 분포 밖 3~10 km", 3: "3단계: 8~20 km 고리에서 이륙"}


def style(a):
    a.grid(color=GRID, lw=.8)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)


def stage_spans(state, last_step):
    edges = [0] + [s["step"] for s in state["switches"]] + [last_step]
    stages = [1] + [s["to_stage"] for s in state["switches"]]
    return list(zip(stages, edges[:-1], edges[1:]))


def curve_figure(state, out, tb_dir):
    hist = state["history"]
    last = max(h["step"] for v in hist.values() for h in v)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    (a1, a2), (a3, a4) = axs
    for a in (a1, a2, a3, a4):
        for st, lo, hi in stage_spans(state, last + 25_000):
            a.axvspan(lo/1e6, hi/1e6, color=STAGE_BG[st], zorder=0)
    for st, rows in sorted(hist.items(), key=lambda kv: int(kv[0])):
        x = np.array([r["step"] for r in rows])/1e6
        ratio = np.array([r["ratio"] for r in rows])
        a1.plot(x, ratio, "-o", color=C_RL, ms=5, lw=1.8, label="RL / SAROPS (검증 5개 평균)" if st == "1" else None)
        a2.plot(x, [np.mean(r["rl"]) for r in rows], "-o", color=C_RL, ms=5, lw=1.8, label="RL" if st == "1" else None)
        a2.plot([x[0] - .02, x[-1] + .02], [np.mean(rows[0]["sarops"])]*2, "--", color=C_SAR, lw=2, label="SAROPS (그 단계 검증 5개)" if st == "1" else None)
        a3.plot(x, [np.mean(r["pos"]) for r in rows], "-o", color=C_RL, ms=4, lw=1.5, label="RL POS" if st == "1" else None)
        a3.plot(x, [np.mean(r["A"]) for r in rows], "-s", color="#1baf7a", ms=4, lw=1.5, label="RL A" if st == "1" else None)
    a1.axhline(1.0, color=C_SAR, lw=1.5, ls="--")
    a1.axhline(0.9, color=MUTED, lw=1, ls=":")
    a1.text(0.01, 1.005, "SAROPS와 같음", color=C_SAR, fontsize=9, va="bottom")
    a1.text(0.01, 0.905, "단계 통과선 0.9", color=MUTED, fontsize=9, va="bottom")
    for s in state["switches"]:
        a1.annotate(f"{s['from_stage']}→{s['to_stage']}단계", (s["step"]/1e6, 0.86), fontsize=9, color=MUTED, ha="center")
    a1.set(xlabel="학습 스텝 (백만)", ylabel="A + POS 비율 (RL ÷ SAROPS)", title="검증 점수: SAROPS 대비 (별도 particle로 채점)", ylim=(0.8, max(1.25, a1.get_ylim()[1])))
    a2.set(xlabel="학습 스텝 (백만)", ylabel="A + POS (별도 particle)", title="검증 점수: 단계마다 검증 경우가 다르다")
    a3.set(xlabel="학습 스텝 (백만)", ylabel="값", title="RL 검증 점수의 두 항")
    for a in (a1, a2, a3):
        a.legend(frameon=False, fontsize=9, loc="lower right")
        style(a)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        runs = sorted(tb_dir.glob("ppo_*"))
        ea = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        for tag, color, label in (("episode/total", C_RL, "에피소드 A + POS (학습, 계획 particle)"), ("episode/POS", "#7a5af8", "POS"), ("episode/A", "#1baf7a", "A")):
            if tag in ea.Tags()["scalars"]:
                s = ea.Scalars(tag)
                a4.plot([e.step/1e6 for e in s], [e.value for e in s], color=color, lw=1.5, label=label)
        a4.legend(frameon=False, fontsize=9, loc="lower right")
    except Exception as exc:                       # the figure stays useful without the TensorBoard panel
        a4.text(.5, .5, f"TensorBoard 읽기 실패: {exc}", ha="center", transform=a4.transAxes)
    a4.set(xlabel="학습 스텝 (백만)", ylabel="최근 100 에피소드 평균", title="학습 에피소드 (커리큘럼 시작, 단계가 바뀌면 시작이 달라진다)")
    style(a4)
    handles = [matplotlib.patches.Patch(color=STAGE_BG[k], label=STAGE_NAME[k]) for k in (1, 2, 3)]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=10)
    fig.savefig(out/"report_curve.png", dpi=100)
    plt.close(fig)


def load_run(rid):
    from rl_search.scripts.real_map.vehicle_web import RUNS
    p = RUNS/rid/"run.json.gz"
    return json.load(gzip.open(p, "rt")) if p.exists() else None


def cases_figure(state, out, refs):
    hist = state["history"]
    st = max(int(k) for k in hist)
    last = hist[str(st)][-1]
    n = len(last["rl"])
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(n)
    w = .36
    sar = [refs[(st, k)]["holdout"] for k in range(n)]
    ax.bar(x - w/2, [s["pos"] for s in sar], w, color=C_SAR, label="SAROPS POS")
    ax.bar(x - w/2, [s["A"] for s in sar], w, bottom=[s["pos"] for s in sar], color="#9cc3ee", label="SAROPS A")
    ax.bar(x + w/2, last["pos"], w, color=C_RL, label="RL POS")
    ax.bar(x + w/2, last["A"], w, bottom=last["pos"], color="#f6b89a", label="RL A")
    for i in range(n):
        ax.text(i - w/2, sar[i]["total"] + .02, f"{sar[i]['total']:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(i + w/2, last["rl"][i] + .02, f"{last['rl'][i]:.2f}", ha="center", fontsize=9, color=INK)
    ax.set(xticks=x, xticklabels=[f"경우 {k + 1}" for k in range(n)], ylabel="A + POS (별도 particle)",
           title=f"{STAGE_NAME[st]} 검증 5개: SAROPS와 RL {last['step']:,} 스텝 (막대 = POS + A)")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.set_ylim(0, max(max(last["rl"]), max(s["total"] for s in sar))*1.25)
    style(ax)
    fig.savefig(out/"report_cases.png", dpi=100)
    plt.close(fig)


def decode(b, dtype, shape):
    return np.frombuffer(base64.b64decode(b), dtype).reshape(shape)


def paths_figure(state, out):
    from rl_search.scripts.real_map import vehicle_web  # noqa: F401  (imported before the figure: it resets the font)
    korean_font()
    hist = state["history"]
    st = max(int(k) for k in hist)
    step = hist[str(st)][-1]["step"]
    n = len(hist[str(st)][-1]["rl"])
    fig, axs = plt.subplots(2, n, figsize=(4.2*n, 8.6), layout="constrained")
    for k in range(n):
        for row, (rid, color, name) in enumerate(((f"eval-val-s{st}-k{k}-sarops", C_SAR, "SAROPS"), (f"eval-val-s{st}-k{k}-rl-{step}", C_RL, f"RL {step:,}"))):
            a = axs[row, k]
            d = load_run(rid)
            if d is None:
                a.text(.5, .5, f"{rid} 없음", ha="center", transform=a.transAxes)
                continue
            o, r, m = d["opt"], d["res"], d["meta"]
            T, N = int(r["T"]), int(r["N"])
            P = decode(r["pos"], "<i2", (T, N, 2)).astype(float)*float(r["pos_unit_m"])
            W = decode(o["weight"], np.uint8, (int(o["T"]), int(o["N"])))
            u = o["uavs"][0]
            xy = decode(u["xy"], np.float32, (-1, 2))
            j = min(int(o["T"]) - 1, T - 1)
            live = W[j] > 0
            sc = a.scatter(P[j, live, 0]/1000, P[j, live, 1]/1000, c=W[j, live]/255, s=1.5, cmap="Greys", vmin=0, vmax=1, zorder=1)
            a.plot(xy[:, 0]/1000, xy[:, 1]/1000, color=color, lw=1.1, zorder=2)
            a.plot(*np.array(o["start"])/1000, marker="^", color=INK, ms=7, zorder=3)
            a.plot(*np.array(o["lkp"])/1000, marker="*", color="#f2c94c", mec=INK, ms=12, zorder=3)
            h = m["holdout"]
            a.set_title(f"경우 {k + 1} · {name}\nA + POS {h['total']:.3f} (POS {h['pos']:.3f}, A {h['A']:.3f})", fontsize=10)
            a.set_aspect("equal")
            a.tick_params(labelsize=8)
            if k == 0:
                a.set_ylabel("y (km)")
    fig.suptitle(f"{STAGE_NAME[st]} 검증 비행: 위 SAROPS, 아래 RL. 점 = 비행 끝 시각의 particle (진할수록 남은 가중치 큼), ★ LKP, ▲ 이륙점", fontsize=12)
    fig.savefig(out/"report_paths.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/rl_real"))
    args = ap.parse_args()
    from rl_search.rl_real_eval import sarops_refs
    d = args.out/args.tag
    state = json.loads((d/"state.json").read_text())
    curve_figure(state, d, d/"tb")
    cases_figure(state, d, sarops_refs("val"))
    paths_figure(state, d)
    print("saved", sorted(p.name for p in d.glob("report_*.png")))


if __name__ == "__main__":
    main()
