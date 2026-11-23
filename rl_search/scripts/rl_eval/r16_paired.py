"""R15 vs R16 training flights at the same step (user 2026-11-24 "r15보다 환경수가 더 많았나?"): same 20 envs, same pool order, same flight
length, so about the same target scenarios at the same step (the launch points are drawn apart). Run: python -m rl_search.scripts.rl_eval.r16_paired"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_search.scripts.rl_eval.r16_compare import _tb
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
grid = np.arange(0.05, 6.0, 0.01)
D = {}
for run in ("r15_chungju", "r16_chungju"):
    D[run] = {t: np.interp(grid, *_tb(run, t)) for t in ("episode/POS", "episode/launch_km")}
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for run, name, col, ls in (("r15_chungju", "R15", "#8a94a0", "--"), ("r16_chungju", "R16", "#2b6cb0", "-")):
    ax[0].plot(grid, D[run]["episode/launch_km"], ls, color=col, label=name)
    ax[1].plot(grid, D[run]["episode/POS"], ls, color=col, label=name)
dp = D["r16_chungju"]["episode/POS"] - D["r15_chungju"]["episode/POS"]
ax[2].plot(grid, dp, color="#2b6cb0")
ax[2].axhline(0, color="k", lw=0.6)
both3 = grid >= 2.1
ax[2].axvspan(2.1, 6.0, color="#2b6cb0", alpha=0.06)
ax[2].text(4.0, ax[2].get_ylim()[1]*0.85, f"둘 다 10~40 km 이륙인 구간 평균 차이 {dp[both3].mean():+.3f}", ha="center")
ax[0].set_title("(A) 학습 비행의 평균 이륙 거리 (km): 이륙 거리 단계가 언제 넘어갔는지")
ax[1].set_title("(B) 학습 비행 POS (최근 100비행 평균, 같은 스텝 = 거의 같은 시나리오)")
ax[2].set_title("(C) 같은 스텝의 차이 R16 - R15 (시나리오 탓 출렁임이 빠진 값)")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend(); ax[1].legend()
ax[2].set_xlabel("학습 스텝 (백만)")
fig.tight_layout()
out = "rl_search/data/real/amboy/rl_eval/report/r16_r15_paired.png"
fig.savefig(out, dpi=110)
corr = np.corrcoef(D["r15_chungju"]["episode/POS"][both3], D["r16_chungju"]["episode/POS"][both3])[0, 1]
print(out, f"corr (both 10-40 km) {corr:.2f}, mean diff {dp[both3].mean():+.4f}, by 1M:",
      [f"{a}-{a+1}M {dp[(grid >= a) & (grid < a + 1)].mean():+.3f}" for a in range(2, 6)])
