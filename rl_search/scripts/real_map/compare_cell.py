"""Compare the real-region terrain at 50 m and 25 m cells: close-ups of the vehicle-relevant layers and stats.

Run from the repository root after building both grids:
    python -m rl_search.scripts.make_real_terrain --region chungju --cell 50
    python -m rl_search.scripts.make_real_terrain --region chungju --cell 25
    python -m rl_search.scripts.real_map.compare_cell
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from rl_search.real_terrain import DATA
from rl_search.terrain import load_terrain

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

REGION = "chungju"
FILES = {50: Path("rl_search/data")/f"terrain_real_{REGION}_60km.npz",
         25: Path("rl_search/data")/f"terrain_real_{REGION}_60km_25m.npz"}
WINDOWS = [(-26, -22, -10, -6, "남서쪽 농경지 (4 km)"),
           (-17, -13, -6, -2, "충주 남한강과 다리 (4 km)"),
           (7, 11, -17, -13, "산간 계곡 도로 (4 km)")]
CLASSES = [("열린 땅 (논밭·초지·마을 밖)", "#f3efe2"), ("숲", "#2e6b34"), ("급경사 ≥32° (숲 밖)", "#8d6e63"),
           ("물", "#2f80ed"), ("다리 칸", "#ff6f00"), ("시가지 (가림)", "#d99aa5"), ("큰 도시 (인계)", "#8b1e2d"),
           ("느린 길", "#c58a4a"), ("빠른 길", "#111111")]


def classes(t):
    c = np.zeros(t.z.shape, np.uint8)
    c[t.forest] = 1
    c[(t.slope_deg >= t.cfg.max_slope_deg) & ~t.forest] = 2
    c[t.built] = 5
    c[t.surface == 1] = 7
    c[t.surface == 2] = 8
    c[t.city] = 6
    c[t.water] = 3
    c[t.fords] = 4
    return c


def main():
    out = DATA/REGION
    terr = {k: load_terrain(p) for k, p in FILES.items()}
    stats = {k: json.loads((out/("terrain_stats.json" if k == 50 else f"terrain_stats_{k}m.json")).read_text())
             for k in FILES}
    cmap = ListedColormap([c for _, c in CLASSES])
    norm = BoundaryNorm(np.arange(-.5, len(CLASSES)), len(CLASSES))

    fig, axs = plt.subplots(2, 3, figsize=(17, 11.6), layout="constrained")
    for row, (cell, t) in enumerate(terr.items()):
        a = t.axis/1000
        h = t.cell/2000
        ext = (a[0] - h, a[-1] + h, a[0] - h, a[-1] + h)
        hs = LightSource(azdeg=315, altdeg=45).hillshade(t.z, vert_exag=2, dx=t.cell, dy=t.cell)
        cl = classes(t)
        for col, (x0, x1, y0, y1, nm) in enumerate(WINDOWS):
            ax = axs[row, col]
            ax.imshow(cl, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
            ax.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, alpha=.18)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_title(f"{cell} m 격자 · {nm}", fontsize=11)
            ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    fig.legend(handles=[Patch(color=c, label=n) for n, c in CLASSES], loc="outside lower center", ncol=9, fontsize=9)
    fig.suptitle("같은 장소를 50 m(위)와 25 m(아래) 격자로 본 레이어", fontsize=14)
    fig.savefig(out/"cell_compare_zoom.png", dpi=120)
    plt.close(fig)

    # stats side by side
    rows = [("물", "water"), ("다리 칸 비율", None), ("급경사 ≥32°", "steep_nogo"), ("통과 불가", "forbidden"),
            ("빠른 길 칸", "surface_fast"), ("느린 길 칸", "surface_slow"), ("시가지 (가림)", "built_outside_city"),
            ("큰 도시", "city")]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), layout="constrained")
    ax = axs[0]
    y = np.arange(len(rows))
    cols = {50: "#7a8ca8", 25: "#2a4d7a"}
    for i, cell in enumerate(FILES):
        s, t = stats[cell], terr[cell]
        v = [t.fords.mean() if k is None else s[k] for _, k in rows]
        ax.barh(y + (i - .5)*.38, np.array(v)*100, height=.36, color=cols[cell], label=f"{cell} m")
        for yy, vv in zip(y + (i - .5)*.38, v):
            ax.text(vv*100 + .1, yy, f"{vv*100:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y, [r for r, _ in rows]); ax.invert_yaxis(); ax.set_xlabel("지도 면적 비율 (%)")
    ax.set_xlim(0, 16); ax.legend(); ax.set_title("레이어 면적 비율")
    ax = axs[1]
    bins = np.arange(0, 61, 1)
    for cell, t in terr.items():
        hst, _ = np.histogram(t.slope_deg[~t.forest & ~t.water], bins=bins, density=True)
        ax.plot(bins[:-1] + .5, hst, color=cols[cell], lw=2, label=f"{cell} m, 숲·물 밖")
        hst, _ = np.histogram(t.slope_deg[t.forest], bins=bins, density=True)
        ax.plot(bins[:-1] + .5, hst, color=cols[cell], lw=1.3, ls="--", label=f"{cell} m, 숲")
    ax.axvline(15, color="#b5703a", ls=":"); ax.text(15.5, ax.get_ylim()[1]*.92, "차량 논밭 경사 한계 15° (초기값)", fontsize=8)
    ax.axvline(32, color="#555", ls=":"); ax.text(32.5, ax.get_ylim()[1]*.8, "도보 한계 32°", fontsize=8)
    ax.set_xlabel("경사 (°)"); ax.set_ylabel("밀도"); ax.legend(fontsize=8)
    ax.set_title("경사 분포: 25 m에서 산비탈(숲)이 더 가파르게 보이고 논밭은 그대로")
    fig.savefig(out/"cell_compare_stats.png", dpi=120)
    plt.close(fig)

    for cell in FILES:
        s = stats[cell]
        print(cell, {k: s[k] for k in ("water", "steep_nogo", "forbidden", "surface_fast", "surface_slow",
                                       "bridge_groups", "land_parts_km2")})


if __name__ == "__main__":
    main()
