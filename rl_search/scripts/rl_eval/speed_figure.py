"""Figure of rl_eval.speed_check: environment steps per second and memory per env by the number of parallel envs.

    python -m rl_search.scripts.rl_eval.speed_figure
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["AppleGothic", "Noto Sans CJK KR", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
SRC = Path("rl_search/data/real/amboy/rl_eval/speed_check.json")
INK, MUTED, GRID, C1, C2 = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"

d = json.loads(SRC.read_text())
P = d["parallel"]
n = np.array([r["n"] for r in P])
sps = np.array([r["steps_per_s"] for r in P])
mem = np.array([r["mem_used_per_env_mb"] for r in P])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
a1.plot(n, sps, "-o", color=C1, lw=2, ms=7)
for x, y in zip(n, sps):
    a1.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
a1.axhline(91, color=C2, lw=1.5, ls="--")
a1.text(n[-1], 96, "참고: 지난 5초 RL 학습 전체 속도 91 (옛 환경, 신경망·학습 포함)", ha="right", va="bottom", fontsize=9, color=C2)
a1.set(xlabel="병렬 환경 수", ylabel="환경 스텝 / 초 (5초 비행 1번 = 1스텝)", title="속도: 무작위 행동, 신경망·학습 시간 제외", ylim=(0, max(sps)*1.18), xticks=n)
a2.bar(["지형 따로 읽기\n(고치기 전, 4개)", "부모에서 한 번 읽고\n나눠 쓰기 (12개)"], [1750, mem[-1]], color=[C2, C1], width=.5)
for i, v in enumerate([1750, mem[-1]]):
    a2.text(i, v + 30, f"약 {v:.0f} MB", ha="center", fontsize=10, color=INK)
a2.set(ylabel="환경 하나가 실제로 더 쓰는 메모리 (MB)", title="메모리: 16 GB PC에서 고치기 전 4개, 고친 뒤 12개까지", ylim=(0, 2100))
for a in (a1, a2):
    a.grid(axis="y", color=GRID, lw=.8)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"RL 환경 속도와 메모리 (Ubuntu i5-10400 6코어, 1개일 때 한 스텝: 비행 {d['single']['fly_ms']:.1f} ms + 관측 {d['single']['obs_ms']:.1f} ms)", fontsize=12)
out = SRC.with_suffix(".png")
fig.savefig(out, dpi=110)
print("saved", out)
