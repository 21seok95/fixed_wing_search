"""Which open land counts as a "maze" (narrow gaps, small patches)? Shown on the map, before anything is changed in the model.

Closed = drivable cells that vanish under a morphological opening with a disk of diameter W, plus what is left in pieces smaller
than A. Particles use the settings of a saved web run. Figure: rl_search/data/real/chungju/maze_mask.png

    python -m rl_search.scripts.real_map.maze_mask --run 20261109-143455-001
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.offroad_closure import closed
from rl_search.vehicle import EXIT, OFF, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
WIDTHS = (100.0, 150.0, 200.0)
AREA_KM2 = .05
HALF = 1500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20261109-143455-001")
    args = ap.parse_args()
    m = json.loads((DATA/"chungju"/"web_runs"/args.run/"meta.json").read_text())
    cfg, v = VehicleConfig(**m["config"]), m["view"]
    net = load_net("chungju", 25.0)
    r = simulate_vehicle(net, cfg, v["lkp"], v["heading"], np.random.default_rng(v["seed"]))
    dt = cfg.save_dt_s; T = min(int(3600/dt), len(r.track) - 1); k0 = T - int(1200/dt)
    tr = np.asarray(r.track[:T + 1], float)
    moved = np.hypot(*(tr[T] - tr[k0]).T)
    live = r.state[T] != EXIT
    off = (r.state[k0:T + 1] == OFF).mean(0)
    stuck = np.flatnonzero(live & (moved < 500) & (off >= .8))
    H, ex, ey = np.histogram2d(tr[T, stuck, 0], tr[T, stuck, 1], bins=[np.arange(-30000, 30001, 3000)]*2)
    spots = []
    for idx in np.argsort(H.ravel())[::-1][:2]:
        i, j = np.unravel_index(idx, H.shape)
        pts = tr[T, stuck][(tr[T, stuck, 0] >= ex[i]) & (tr[T, stuck, 0] < ex[i + 1]) & (tr[T, stuck, 1] >= ey[j]) & (tr[T, stuck, 1] < ey[j + 1])]
        spots.append(np.median(pts, axis=0))
    t, axis, base = net.terr, net.terr.axis, net.drive.copy()
    masks = {W: closed(base, float(t.cell), W/2, AREA_KM2) for W in WIDTHS}
    ro, co = net.cell_of(tr[T, stuck])
    # all particle-time spent off-road, to see how much of it would be affected
    s = r.state[:T + 1]
    oi, oj = np.nonzero(s == OFF)
    orr, occ = net.cell_of(tr[oi, oj])
    stats = {}
    for W, keep in masks.items():
        gone = base & ~keep
        stats[W] = dict(closed_land=float(gone.sum()/base.sum()), stuck_in_closed=float(gone[ro, co].mean()), off_time_in_closed=float(gone[orr, occ].mean()))
    fig, axs = plt.subplots(2, 3, figsize=(20, 13.2), layout="constrained")
    cmap = ListedColormap(["#6f9a68", "#f4f1e6", "#e4572e"])
    for a, c in enumerate(spots):
        c0, c1 = np.searchsorted(axis, [c[0] - HALF, c[0] + HALF]); r0, r1 = np.searchsorted(axis, [c[1] - HALF, c[1] + HALF])
        ext = [axis[c0]/1000, axis[c1 - 1]/1000, axis[r0]/1000, axis[r1 - 1]/1000]
        rd = net.road_dist[r0:r1, c0:c1] <= 12.5
        near = stuck[(np.abs(tr[T, stuck, 0] - c[0]) < HALF) & (np.abs(tr[T, stuck, 1] - c[1]) < HALF)]
        for b, W in enumerate(WIDTHS):
            ax = axs[a, b]
            img = np.zeros(rd.shape, np.uint8)
            img[base[r0:r1, c0:c1]] = 2
            img[masks[W][r0:r1, c0:c1]] = 1
            ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            ax.imshow(np.ma.masked_where(~rd, rd), origin="lower", extent=ext, cmap=ListedColormap(["#333333"]), interpolation="nearest")
            for p in near[:12]:
                ax.plot(tr[k0:T + 1, p, 0]/1000, tr[k0:T + 1, p, 1]/1000, lw=.8, color="#1f4e8c", alpha=.75)
            ax.plot(tr[T, near, 0]/1000, tr[T, near, 1]/1000, "o", ms=3.2, color="#111", mec="white", mew=.3)
            st = stats[W]
            ax.set_title(f"폭 {W:.0f} m 미만 + {AREA_KM2*100:.0f} ha 미만 닫기 · 위치 ({c[0]/1000:.1f}, {c[1]/1000:.1f}) km\n닫히는 땅 {st['closed_land']:.0%} · 갇힌 차 중 닫히는 땅에 있던 비율 {st['stuck_in_closed']:.0%} · 도로 밖 시간 중 그 땅 {st['off_time_in_closed']:.0%}", fontsize=10)
            ax.set_xlabel("km"); ax.set_ylabel("km"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.suptitle(f"닫을 땅 후보 (빨강 = 닫히는 땅, 베이지 = 남는 땅, 초록 = 원래 못 가는 땅, 검정 = 도로, 파란 선과 점 = 갇힌 차 {len(stuck)}개 중 이곳의 차) · 설정: {m['label']}, 나가기 {cfg.p_leave_fast_min}/분, 복귀 {cfg.p_return_min}/분", fontsize=13)
    p = DATA/"chungju"/"maze_mask.png"
    fig.savefig(p, dpi=90)
    print(len(stuck), "stuck of", int(live.sum())); print({W: {k: round(x, 3) for k, x in s_.items()} for W, s_ in stats.items()}); print("Saved", p)


if __name__ == "__main__":
    main()
