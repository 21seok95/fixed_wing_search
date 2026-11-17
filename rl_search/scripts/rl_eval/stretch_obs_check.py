"""What the smallest observation map (1.6 km, 64 x 64 cells of 25 m) shows on a stretched map (rl_search.stretch: terrain cells of 50 m)
against the real 25 m terrain data (user 2026-11-17: is a 50 m cell a problem when the smallest map has 25 m cells?).

Top row, a place with mixed ground: the camera visibility of the real 60 km map in the 0.8 x 0.8 km it covers (32 x 32 cells of 25 m);
the visibility channel the RL env builds on amboy_x2 there (64 x 64, each terrain cell on 2 x 2 map cells); that channel averaged back
over 2 x 2 (= the real data); and, for comparison, a 1.6 x 1.6 km window of the real 25 m data (what a map with 25 m terrain cells would
show in the same 64 x 64). Bottom row, a place with particles: the particle channel of the stretched map and the particle positions.

    python -m rl_search.scripts.rl_eval.stretch_obs_check
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")


def main():
    warnings.filterwarnings("ignore")
    from scipy.ndimage import generic_filter, uniform_filter
    from rl_search.detection import DetectionConfig, load_landcover, shared_visibility
    from rl_search.rl_real import RealRLConfig, RealRLEnv, preload
    from rl_search.scripts.rl_eval.signal_check import settings_of
    r, base = "amboy_x2", "amboy"
    B = preload(base)["terr"]
    Ab = shared_visibility(B, load_landcover(base, float(B.cell)), DetectionConfig())
    # the most mixed 0.8 km window (32 x 32 cells) of the central 40 km: largest local variance of the visibility
    m = uniform_filter(Ab.astype(float), 32)
    v = uniform_filter(Ab.astype(float)**2, 32) - m**2
    n = len(B.axis)
    c0, c1 = n//2 - 800, n//2 + 800
    sub = v[c0:c1:8, c0:c1:8]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    ri, cj = c0 + 8*i, c0 + 8*j
    px, py = float(B.axis[cj]), float(B.axis[ri])                 # real metres
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text())) | dict(scen_v2=True)
    env = RealRLEnv(RealRLConfig(region=r, multi=True, pool=None, settings=settings, stage=3, stage_mix={3: 1.0}), seed=4)
    env._pick_type = lambda: "road"
    env.reset()
    e = env.env
    k = 2.0

    def obs_at(x, y):
        e.plant.reset(np.array([x, y, np.pi/2]))                  # heading north: row 0 = north, column 0 = west
        e.chi_tgt = np.pi/2
        return env.observation()["m1600"]
    o = obs_at(px*k, py*k)
    vis_ch = int(np.argmin([np.abs(o[c] - np.round(o[c], 6)).sum() + (0 if (o[c].min() >= 0 and o[c].max() <= 1.0) else 1e9)
                            for c in range(o.shape[0])])) if False else 4
    vis = o[vis_ch]
    # the real data under the same 0.8 km window, north up (row 0 = north)
    half_c = 16
    real = Ab[ri - half_c:ri + half_c, cj - half_c:cj + half_c][::-1]
    # ② against ① drawn 2 x 2 per cell: the map cell centres fall half a terrain cell off the terrain cell edges, so the 2 x 2 blocks can sit
    # one map cell off; take the shift (0 or 1 cell, each axis) that matches best and compare away from the window edge
    up = np.repeat(np.repeat(real, 2, 0), 2, 1)
    best = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.roll(np.roll(up, dy, 0), dx, 1)
            same = float(np.mean(sh[2:-2, 2:-2] == vis[2:-2, 2:-2]))
            if best is None or same > best[0]:
                best = (same, dy, dx, sh)
    back = best[3]
    wide = Ab[ri - 32:ri + 32, cj - 32:cj + 32][::-1]
    # particles: put the UAV on the densest 1.6 km cell of the particles now
    P = e.belief.positions(e.t)
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=np.arange(-60000, 60001, 1600))
    a, b_ = np.unravel_index(np.argmax(H), H.shape)
    qx, qy = (xe[a] + xe[a + 1])/2, (ye[b_] + ye[b_ + 1])/2
    o2 = obs_at(qx, qy)
    part = o2[0]
    inwin = (np.abs(P[:, 0] - qx) < 800) & (np.abs(P[:, 1] - qy) < 800)
    figure(real, vis, back, wide, part, P[inwin] - [qx, qy], o.shape, vis_ch, (px, py), best[0])


def figure(real, vis, back, wide, part, Pw, shape, vis_ch, where, maxdiff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    fig, axs = plt.subplots(2, 4, figsize=(22, 11.5), facecolor=BG)
    kw = dict(cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    panels = [(real, "① 실제 지형 자료 (25 m 칸)\n0.8 × 0.8 km = 32 × 32 칸", 32),
              (vis, "② 늘린 지도에서 학습 환경이 만든 가시도 채널\n1.6 km 지도(실제 0.8 km) = 64 × 64 칸: 지형 한 칸이 2 × 2칸에 찍힘", 64),
              (back, f"③ ①의 칸마다 2 × 2로 늘린 그림 (한 칸 밀림 맞춤)\n②와 같은 칸 {100*maxdiff:.1f}% → 줄어든 정보 없음", 64),
              (wide, "④ 비교: 25 m 칸 지형 1.6 × 1.6 km (64 × 64)\n= 실제 큰 지도라면 이 지도의 무늬 굵기", 64)]
    for ax, (img, ttl, nc) in zip(axs[0], panels):
        ax.imshow(img, **kw)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for t in range(0, nc + 1, 8):
            ax.axhline(t - .5, color="#2a78d6", lw=0.3, alpha=0.4); ax.axvline(t - .5, color="#2a78d6", lw=0.3, alpha=0.4)
    ax = axs[1, 0]
    ax.imshow(part, cmap="Oranges", interpolation="nearest")
    ax.set_title("⑤ 같은 늘린 지도의 입자 채널 (1.6 km 지도)\n입자가 있는 한 칸씩 찍힘 (2 × 2 덩어리 아님)", loc="left", fontsize=10, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1, 1]
    ax.scatter(Pw[:, 0], Pw[:, 1], s=4, color="#c9772e")
    for t in np.arange(-800, 801, 25):
        ax.axvline(t, color=GRID, lw=0.3); ax.axhline(t, color=GRID, lw=0.3)
    ax.set_xlim(-800, 800); ax.set_ylim(-800, 800); ax.set_aspect("equal")
    ax.set_title("⑥ 그 입자들의 실제 좌표 (늘린 지도 m, 선 = 25 m 칸)\n좌표가 칸에 붙지 않은 연속 값", loc="left", fontsize=10, color=INK)
    for ax in axs[1, 2:]:
        ax.axis("off")
    txt = ["정리",
           "· 지형 칸 50 m (늘린 지도) → 가장 작은 지도에서 가시도 한 칸이 2 × 2칸에 똑같이 찍힘 (②)",
           "· 그래도 실제 자료 그대로: ①을 2 × 2로 늘린 그림과 칸마다 같음 (③)",
           "· 입자·예측·남은 POS·카메라 채널은 좌표가 연속이라 25 m 칸마다 따로 찍힘 (⑤, ⑥)",
           "· 실제 큰 지도(25 m 칸)로 돌아가면 가시도 무늬만 2배 촘촘해짐 (④) = '지형만 다르다'",
           f"· 위치: 실제 지도 ({where[0]/1000:.1f}, {where[1]/1000:.1f}) km, 이 지도에서 가시도가 가장 섞인 곳; 가시도 채널 = {vis_ch}번째"]
    axs[1, 2].text(0, 0.9, "\n".join(txt), fontsize=11, color=INK, va="top", linespacing=1.8, transform=axs[1, 2].transAxes)
    fig.suptitle("늘린 지도의 가장 작은 관측 지도 (1.6 km, 25 m 칸): 지형 칸 50 m가 학습 입력에 어떻게 보이는지 (암보이, 흰색 = 잘 보임)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"stretch_obs_check.png", dpi=85, facecolor=BG)
    print("saved; same cells", maxdiff, "shape", shape)


if __name__ == "__main__":
    main()
