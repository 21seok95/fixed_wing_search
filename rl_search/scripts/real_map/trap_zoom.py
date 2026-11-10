"""Close look at where off-road particles get trapped, with the settings of a saved web run: their tracks on the 25 m drivable mask,
what blocks each cell, and how much of the mask's edge comes from very small obstacles.

Figure: rl_search/data/real/chungju/trap_zoom.png

    python -m rl_search.scripts.real_map.trap_zoom --run 20261109-143455-001
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

from rl_search.real_terrain import DATA
from rl_search.vehicle import EXIT, OFF, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20261109-143455-001")
    ap.add_argument("--half", type=float, default=1500.0)
    args = ap.parse_args()
    m = json.loads((DATA/"chungju"/"web_runs"/args.run/"meta.json").read_text())
    cfg, v = VehicleConfig(**m["config"]), m["view"]
    print("run", m["label"], {k: m["config"][k] for k in ("particles", "p_leave_fast_min", "p_leave_slow_min", "p_return_min", "return_radius_m", "off_max_slope_deg", "w_off_road", "w_off_keep", "w_off_heading")})
    net = load_net("chungju", 25.0)
    r = simulate_vehicle(net, cfg, v["lkp"], v["heading"], np.random.default_rng(v["seed"]))
    dt = cfg.save_dt_s; T = min(int(3600/dt), len(r.track) - 1); k0 = T - int(1200/dt)
    tr = np.asarray(r.track[:T + 1], float)
    moved = np.hypot(*(tr[T] - tr[k0]).T)
    live = r.state[T] != EXIT
    off = (r.state[k0:T + 1] == OFF).mean(0)
    stuck = np.flatnonzero(live & (moved < 500) & (off >= .8))
    print(f"stuck (500 m in 20 min, off-road >= 80 %): {len(stuck)} of {int(live.sum())} = {len(stuck)/live.sum():.1%}")

    t, axis = net.terr, net.terr.axis
    H, ex, ey = np.histogram2d(tr[T, stuck, 0], tr[T, stuck, 1], bins=[np.arange(-30000, 30001, 1500)]*2)
    i, j = np.unravel_index(H.argmax(), H.shape)
    cx, cy = (ex[i] + ex[i + 1])/2, (ey[j] + ey[j + 1])/2
    near = stuck[(np.abs(tr[T, stuck, 0] - cx) < args.half) & (np.abs(tr[T, stuck, 1] - cy) < args.half)]
    print(f"densest spot ({cx/1000:.1f}, {cy/1000:.1f}) km: {len(near)} stuck particles")
    c0, c1 = np.searchsorted(axis, [cx - args.half, cx + args.half]); r0, r1 = np.searchsorted(axis, [cy - args.half, cy + args.half])
    ext = [axis[c0]/1000, axis[c1 - 1]/1000, axis[r0]/1000, axis[r1 - 1]/1000]
    why = np.zeros(net.drive.shape, np.uint8)
    why[t.slope_deg >= cfg.off_max_slope_deg] = 1
    why[t.forest] = 2
    why[t.water | t.fords] = 3
    why[t.city] = 4
    why[net.drive] = 0
    cmap = ListedColormap(["#f4f1e6", "#c9a27a", "#6f9a68", "#7fb2dc", "#d7a1ad"])
    rd = net.road_dist[r0:r1, c0:c1] <= 12.5

    fig, axs = plt.subplots(1, 3, figsize=(21, 7.4), layout="constrained")
    for ax, (mask, ttl) in zip(axs[:2], ((why[r0:r1, c0:c1], "지금 지도 (25 m 칸)"), (None, "100 m 칸으로 뭉갠 지도 (칸의 절반 이상이 막히면 막힘)"))):
        if mask is None:
            bl = (~net.drive[r0:r1, c0:c1]).astype(float)
            n = (bl.shape[0]//4, bl.shape[1]//4)
            cg = bl[:n[0]*4, :n[1]*4].reshape(n[0], 4, n[1], 4).mean((1, 3)) >= .5
            ax.imshow(cg, origin="lower", extent=ext, cmap=ListedColormap(["#f4f1e6", "#6f9a68"]), interpolation="nearest")
        else:
            ax.imshow(mask, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~rd, rd), origin="lower", extent=ext, cmap=ListedColormap(["#333333"]), interpolation="nearest")
        for k, p in enumerate(near[:10]):
            ax.plot(tr[k0:T + 1, p, 0]/1000, tr[k0:T + 1, p, 1]/1000, lw=1.1)
            ax.plot(tr[T, p, 0]/1000, tr[T, p, 1]/1000, "o", ms=4, color="red", mec="white", mew=.5)
        ax.set_title(ttl + " · 선 = 갇힌 차의 마지막 20분", fontsize=11); ax.set_xlabel("km"); ax.set_ylabel("km"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    blocked = ~net.drive
    lab, n = label(blocked)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    size[0] = 0
    cell_ha = (t.cell/100)**2
    d = net.drive
    edge = lambda a, b: np.stack([a, b], 0)
    # drivable cells next to a blocked cell, and the size of that blocked piece
    sizes = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(lab, (dr, dc), (0, 1))
        sizes.append(size[nb[d & (nb > 0)]])
    sizes = np.concatenate(sizes)*cell_ha
    ax = axs[2]
    edges = [0, 1, 4, 25, 100, 1e12]
    labs = ["1 ha 미만\n(100 m 칸 하나)", "1~4 ha", "4~25 ha", "25 ha~1 km²", "1 km² 이상"]
    share = [100*float(((sizes >= a) & (sizes < b)).mean()) for a, b in zip(edges[:-1], edges[1:])]
    cnt = [int(((size*cell_ha >= a) & (size*cell_ha < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
    ax.bar(range(5), share, color="#2a4d7a")
    for i_, (y, c) in enumerate(zip(share, cnt)):
        ax.text(i_, y, f"{y:.0f}%\n{c:,}개", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_ylabel("막힌 땅과 맞닿은 경계 중 %"); ax.set_ylim(0, max(share)*1.25)
    ax.set_title("막힌 땅 조각의 크기별: 차가 부딪히는 벽의 몇 %인가 (지도 전체)", fontsize=11)
    fig.suptitle(f"갇힌 곳 확대: {m['label']} (나가기 {cfg.p_leave_fast_min}/분, 복귀 {cfg.p_return_min}/분), 빨간 점 = 60분 위치 · 왼쪽 색: 베이지 = 다닐 수 있음, 갈색 = 급경사, 초록 = 숲, 파랑 = 물, 분홍 = 도시, 검정 = 도로", fontsize=12)
    p = DATA/"chungju"/"trap_zoom.png"
    fig.savefig(p, dpi=95)
    print("blocked pieces", n, "by size", dict(zip([l.replace("\n", " ") for l in labs], cnt)), "edge share %", [round(x, 1) for x in share]); print("Saved", p)


if __name__ == "__main__":
    main()
