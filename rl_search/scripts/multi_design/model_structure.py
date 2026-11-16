"""Diagram of the current RL policy network (R13: rl_gym.RealMapExtractor + SB3 MultiInputPolicy heads, train_rl.py) with the parameter
count of each block, and where type-specific changes would go. Counts are computed from the layer sizes (total 1,659,811 = R13).

    python rl_search/scripts/multi_design/model_structure.py rl_search/data/real/amboy/rl_eval/report
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
OUT = Path(sys.argv[1])
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#d9d8d3", "#fcfcfb"
FILL = {"in": "#eef3fb", "cnn": "#e6f2ec", "joint": "#f4eee4", "head": "#f6e9ee", "opt": "#fff4e5"}


def conv(cin, cout):
    return cin*cout*9 + cout


def lin(a, b):
    return a*b + b


fine = conv(6, 16) + conv(16, 16) + conv(16, 32) + conv(32, 32) + lin(2048, 128)
coarse = conv(5, 16) + conv(16, 32) + conv(32, 32) + lin(2048, 128)
vec = lin(13, 32) + lin(32, 32)
joint = lin(5*128 + 32, 256)
pi = lin(256, 128) + lin(128, 128) + lin(128, 1) + 1
vf = lin(256, 128) + lin(128, 128) + lin(128, 1)
total = fine + 4*coarse + vec + joint + pi + vf
assert total == 1_659_811, total


def box(ax, x, y, w, h, text, kind, fs=9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=FILL[kind],
                                ec=MUTED if kind != "opt" else "#d9822b", lw=1.0, ls=ls))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))


fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
rows = [("1.6 km 지도 64×64, 채널 6", "합성곱 3×3: 16(보폭 1) → 16 → 32 → 32\n64 → 64 → 32 → 16 → 8칸, 펼쳐 2048 → 128", fine),
        ("6.4 km 지도 64×64, 채널 5", "합성곱 3×3: 16 → 32 → 32 (보폭 2)\n64 → 32 → 16 → 8칸, 펼쳐 2048 → 128", coarse),
        ("25.6 km 지도 64×64, 채널 5", "같은 모양, 가중치는 따로", coarse),
        ("51.2 km 지도 64×64, 채널 5", "같은 모양, 가중치는 따로", coarse),
        ("102.4 km 지도 64×64, 채널 5 (R13)", "같은 모양, 가중치는 따로", coarse)]
ys = [0.80, 0.665, 0.53, 0.395, 0.26]
for (inp, net, n), y in zip(rows, ys):
    box(ax, 0.01, y, 0.20, 0.10, inp, "in", fs=9.5)
    box(ax, 0.27, y, 0.25, 0.10, net + f"\n파라미터 {n:,}", "cnn", fs=8.8)
    arrow(ax, 0.21, y + 0.05, 0.27, y + 0.05)
    arrow(ax, 0.52, y + 0.05, 0.585, 0.53)
box(ax, 0.01, 0.075, 0.20, 0.15, "벡터 13개\n롤, 선회 상태, 목표−실제 방향,\n남은 에너지, 창 남은 시간, 남은 확률,\n허용 범위 하·상 + 지도별 POS 크기 5", "in", fs=8.6)
box(ax, 0.27, 0.11, 0.25, 0.08, f"13 → 32 → 32 (ReLU)\n파라미터 {vec:,}", "cnn", fs=8.8)
arrow(ax, 0.21, 0.15, 0.27, 0.15); arrow(ax, 0.52, 0.15, 0.585, 0.47)
ax.text(0.01, 0.955, "지도 채널: ① 지금 입자 위치  ② 예측 시각의 입자 위치  ③ POS(지도마다 정규화)  ④ POS(실제 값)  ⑤ 땅 가시도  ⑥ 카메라 화면(1.6 km만)",
        fontsize=10, color=INK)
ax.text(0.01, 0.925, "모든 지도는 기체 중심, 기수가 위. 유형(도로·혼합·사막)은 입력에 없음 → 입자 모양(선 / 퍼짐)과 예측 입자로 알아봐야 함", fontsize=10, color=MUTED)
box(ax, 0.585, 0.40, 0.14, 0.20, f"이어 붙이기\n5 × 128 + 32 = 672\n→ 256 (ReLU)\n파라미터 {joint:,}\n\n정책과 가치가 함께 씀", "joint", fs=9.2)
box(ax, 0.79, 0.58, 0.20, 0.17, f"정책 머리\n128 → 128 (tanh)\n→ 행동 평균 a (−1~1, × 49°)\n+ 상태와 무관한 표준편차\n파라미터 {pi:,}", "head", fs=9)
box(ax, 0.79, 0.25, 0.20, 0.15, f"가치 머리\n128 → 128 (tanh)\n→ 가치 V\n파라미터 {vf:,}", "head", fs=9)
arrow(ax, 0.725, 0.52, 0.79, 0.665); arrow(ax, 0.725, 0.48, 0.79, 0.325)
# where type-specific changes would go
box(ax, 0.27, 0.005, 0.25, 0.085, "선택지 ①: 이동 모델 변수(도로 이탈 확률,\n도로 밖 속도 등)를 벡터에 더함", "opt", fs=8.6, ls="--")
box(ax, 0.79, 0.79, 0.20, 0.10, "선택지 ②: 유형별 정책 머리 3개\n(CNN·합치기는 공유)", "opt", fs=8.6, ls="--")
box(ax, 0.565, 0.17, 0.18, 0.15, "선택지 ③: 정책·가치 CNN 분리\n선택지 ④: 기억(GRU)\n(둘 다 MDP 설계 7절·3-7에서\n'문제가 보이면 검토'로 남김)", "opt", fs=8.6, ls="--")
ax.text(0.79, 0.19, f"합계 {total:,}개 (R13)\n기억 없음: 5초마다 지금 관측만 보고 결정", fontsize=10, color=INK, va="top")
fig.suptitle("지금 RL 모델 구조 (R13, 학습: PPO)  ·  주황 점선 = 유형별 대응을 바꾼다면 손댈 곳", x=0.01, ha="left", fontsize=14, color=INK)
fig.savefig(OUT/"model_structure_r13.png", dpi=95, facecolor=BG, bbox_inches="tight")
print("saved", OUT/"model_structure_r13.png", total)
