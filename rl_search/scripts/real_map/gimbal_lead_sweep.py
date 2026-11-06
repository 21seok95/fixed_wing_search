"""Gimbal pre-pointing from the UAV's own planned path: how many seconds before a U-turn to start pointing at
the next lane. Figure: rl_search/data/real/chungju/gimbal_lead.png

Run from the repository root:
    python -m rl_search.scripts.real_map.gimbal_lead_sweep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.detection import Detector, GimbalConfig, load_landcover
from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.gimbal_check import LANE, NEW, X0, Y0, flight, run
from rl_search.terrain import load_terrain

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LEADS = [None, 0, 4, 8, 12, 16, 20]
SHOW = [None, 8, 16]


def main():
    terr = load_terrain(Path("rl_search/data/terrain_real_chungju_60km_25m.npz"))
    det = Detector(terr, load_landcover(), NEW)
    path, R = flight()
    g = GimbalConfig()
    rows, maps = [], {}
    for lead in LEADS:
        pod, GX, GY, h, tgt, ext = run(det, path, R, "forward" if lead is None else "prelook", g, 0.0 if lead is None else lead)
        gap = (np.abs(GX - (X0 - 2*R)) <= 80) & (GY > Y0 + LANE - 700) & (GY < Y0 + LANE)
        lanes = ((np.abs(GX - X0) <= 80) | (np.abs(GX - (X0 - 2*R)) <= 80)) & (GY > Y0 + 500) & (GY < Y0 + LANE + 600)
        outside = (GX < X0 - 2*R - 120) | (GY > Y0 + LANE + 650)
        cell = terr.cell**2/1e6
        rows.append((lead, pod[gap].mean(), pod[lanes].mean(), pod.sum()*cell, pod[outside].sum()*cell))
        maps[lead] = (pod, ext, tgt)
    rows = np.array([(np.nan if r[0] is None else r[0],) + r[1:] for r in rows], float)

    fig = plt.figure(figsize=(20, 9), layout="constrained")
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .08, 1.6])
    names = {None: "앞쪽 45° 유지", 8: "8초 앞당겨 다음 레인 지향", 16: "16초 앞당겨 다음 레인 지향"}
    for j, lead in enumerate(SHOW):
        pod, ext, tgt = maps[lead]
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(pod, origin="lower", extent=np.array(ext)/1000, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.plot(path[:, 0]/1000, path[:, 1]/1000, color="#4fc3f7", lw=1.2)
        if lead is not None:
            ax.plot(tgt[0]/1000, tgt[1]/1000, "x", color="#4fc3f7", ms=10, mew=2)
        r = rows[LEADS.index(lead)]
        ax.set_title(f"{names[lead]}\n새 레인 빈 곳 {r[1]:.2f} · 레인 띠 {r[2]:.2f}", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), label="누적 POD")
    ax = fig.add_subplot(gs[0, 4])
    L = rows[1:, 0]
    ax.plot(L, rows[1:, 1], "o-", color="#2a4d7a", lw=2, label="새 레인 첫 700 m (빈 곳)")
    ax.plot(L, rows[1:, 2], "s-", color="#2e7d32", lw=2, label="두 레인 띠 (탐색 구역 안)")
    ax.axhline(rows[0, 1], color="#2a4d7a", ls=":", lw=1); ax.axhline(rows[0, 2], color="#2e7d32", ls=":", lw=1)
    ax.set_xlabel("U턴 전 앞당김 시간 (초)"); ax.set_ylabel("평균 누적 POD")
    ax2 = ax.twinx()
    ax2.plot(L, rows[1:, 4], "^-", color="#c0392b", lw=1.6, label="레인 바깥에서 쓴 POD (km² 환산)")
    ax2.axhline(rows[0, 4], color="#c0392b", ls=":", lw=1); ax2.set_ylabel("레인 바깥 POD 합 (km²)", color="#c0392b")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
    ax.set_title("앞당김 시간별 (점선 = 앞쪽 45° 유지)", fontsize=11)
    fig.suptitle("무인기 자기 경로 기반 짐벌 앞당김: U턴 전에 다음 레인 시작점을 지향 (2축, 롤 ±45°, 내려보기 20~70°, 40°/s)", fontsize=14)
    fig.savefig(DATA/"chungju"/"gimbal_lead.png", dpi=105)
    plt.close(fig)
    print("| 앞당김 | 새 레인 빈 곳 | 레인 띠 | 전체 POD 합 (km²) | 레인 바깥 (km²) |")
    print("|---|---|---|---|---|")
    for lead, r in zip(LEADS, rows):
        print(f"| {'유지' if lead is None else f'{lead}초'} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} |")


if __name__ == "__main__":
    main()
