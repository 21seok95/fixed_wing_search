"""Do vehicles get trapped off-road when they leave the roads more often? Where, and can they get back to a road?

For each LKP and setting: particles that stayed within D metres in the last 20 minutes and were off-road for at least 80 % of them.
Road access is measured by walking on drivable cells only (``net.drive``): the walking distance to the nearest cell from
which a car may return to the road (within ``return_radius_m``). Infinite = a pocket with no way back.

Figure and numbers: rl_search/data/real/chungju/offroad_trap.png / .json

    python -m rl_search.scripts.real_map.offroad_trap
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.vehicle_tuner import map_image
from rl_search.vehicle import EXIT, OFF, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LKPS = {"A (5, 5)": (5000.0, 5000.0), "B (−9.6, 15)": (-9600.0, 15000.0), "C (−20, −15)": (-20000.0, -15000.0)}
SETS = {"지금 설정": {}, "논밭 많이": dict(p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1), "논밭 아주 많이": dict(p_leave_fast_min=.5, p_leave_slow_min=.5, p_return_min=.05)}
SEEDS = (0, 1)


def walk_distance(net, radius: float) -> np.ndarray:
    """Metres walked on drivable cells to the nearest return cell (inf if none), 4-neighbour steps."""
    drive, cell = net.drive, float(net.terr.cell)
    n, m = drive.shape
    dist = np.full(drive.shape, np.inf, np.float32)
    front = np.flatnonzero((drive & (net.road_dist <= radius)).ravel())
    dist.ravel()[front] = 0.0
    step = 0
    while len(front):
        step += 1
        r, c = np.divmod(front, m)
        nb = np.concatenate([(r[r > 0] - 1)*m + c[r > 0], (r[r < n - 1] + 1)*m + c[r < n - 1], r[c > 0]*m + c[c > 0] - 1, r[c < m - 1]*m + c[c < m - 1] + 1])
        nb = np.unique(nb)
        nb = nb[drive.ravel()[nb] & ~np.isfinite(dist.ravel()[nb])]
        dist.ravel()[nb] = step*cell
        front = nb
    return dist


def main():
    net = load_net("chungju", 25.0)
    rows, pts = [], {}
    geo = None
    for sn, kw in SETS.items():
        for ln, lkp in LKPS.items():
            for seed in SEEDS:
                r = simulate_vehicle(net, VehicleConfig(particles=2000, **kw), lkp, 45.0, np.random.default_rng(seed))
                if geo is None:
                    geo = walk_distance(net, r.cfg.return_radius_m)
                T = min(int(3600/r.cfg.save_dt_s), len(r.track) - 1)
                k0 = T - int(1200/r.cfg.save_dt_s)
                moved = np.hypot(*(r.track[T] - r.track[k0]).T)
                offs = (r.state[k0:T + 1] == OFF).mean(0)
                live = r.state[T] != EXIT
                ro, co = net.cell_of(np.asarray(r.track[T], float))
                g, rd = geo[ro, co], net.road_dist[ro, co]
                offn = live & (r.state[T] == OFF)
                st5, st2 = live & (moved < 500) & (offs >= .8), live & (moved < 2000) & (offs >= .8)
                # how long each particle has been off-road without a break, at the end
                streak = np.zeros(r.track.shape[1]); s = (r.state[:T + 1] == OFF)
                last_on = T - np.argmax((~s)[::-1], axis=0)
                streak = np.where(s.all(0), T + 1, T - last_on)*r.cfg.save_dt_s/60
                rows.append(dict(setting=sn, lkp=ln, seed=seed, off_share=float((r.state[:T + 1][r.state[:T + 1] != EXIT] == OFF).mean()), off_now=float(offn.sum()/live.sum()),
                                 stuck500=float(st5.sum()/live.sum()), stuck2k=float(st2.sum()/live.sum()),
                                 long_off=float((offn & (streak >= 30)).sum()/live.sum()),
                                 walk_med=float(np.median(g[st2])/1000) if st2.any() else np.nan, road_med=float(np.median(rd[st2])/1000) if st2.any() else np.nan,
                                 no_way=float((~np.isfinite(g) & offn).sum()/live.sum()), far_walk=float((np.isfinite(g) & (g > 1000) & offn).sum()/live.sum())))
                pts.setdefault(sn, []).append((np.asarray(r.track[T], float)[st2]/1000, np.asarray(r.lkp_road)/1000))
    img, ext = map_image(net, 1600)
    fig = plt.figure(figsize=(21, 11), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
    names = list(SETS)
    for j, sn in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, extent=ext, origin="upper", alpha=.55)
        for p, l in pts[sn]:
            ax.scatter(p[:, 0], p[:, 1], s=9, c="#c0392b", lw=0)
            ax.plot(*l, "*", ms=12, mfc="yellow", mec="black")
        m = [r for r in rows if r["setting"] == sn]
        ax.set_title(f"{sn}: 20분 동안 2 km 안에 머문 논밭 particle (전체의 {100*np.mean([r['stuck2k'] for r in m]):.1f}%)", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    x = np.arange(len(names))
    keys = [("off_share", "도로 밖 시간 비율", "#27ae60"), ("long_off", "30분 넘게 도로 밖에 있는 particle", "#e67e22"), ("stuck2k", "20분 동안 2 km 안에 머묾", "#c0392b"), ("stuck500", "20분 동안 500 m 안에 머묾", "#7a1f1f")]
    ax = fig.add_subplot(gs[1, :2]); w = .2
    for k, (key, lab, c) in enumerate(keys):
        v = [100*np.mean([r[key] for r in rows if r["setting"] == s]) for s in names]
        ax.bar(x + (k - 1.5)*w, v, w, color=c, label=lab)
        for xx, y in zip(x + (k - 1.5)*w, v):
            ax.text(xx, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("%"); ax.legend(fontsize=9); ax.set_title("60분 시점 (LKP 3곳, seed 2개 평균)", fontsize=11)
    ax = fig.add_subplot(gs[1, 2])
    v = [(np.nanmean([r["walk_med"] for r in rows if r["setting"] == s]), np.nanmean([r["road_med"] for r in rows if r["setting"] == s])) for s in names]
    ax.bar(x - .2, [a for a, _ in v], .4, color="#2a4d7a", label="논밭길로 걸어 복귀 지점까지")
    ax.bar(x + .2, [b for _, b in v], .4, color="#bbb", label="직선으로 가장 가까운 도로까지")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("km (중앙값)"); ax.legend(fontsize=9); ax.set_title("갇힌 particle에서 도로까지", fontsize=11)
    fig.suptitle("도로 밖으로 많이 나갈 때 갇히는 particle", fontsize=14)
    p = DATA/"chungju"/"offroad_trap.png"
    fig.savefig(p, dpi=95)
    summ = {s: {k: float(np.nanmean([r[k] for r in rows if r["setting"] == s])) for k in rows[0] if k not in ("setting", "lkp", "seed")} for s in names}
    p.with_suffix(".json").write_text(json.dumps(dict(summary=summ, rows=rows, no_way_cells=float((~np.isfinite(geo) & net.drive).sum()/net.drive.sum())), indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False)); print("drivable cells with no way back", (~np.isfinite(geo) & net.drive).sum(), "of", net.drive.sum())


if __name__ == "__main__":
    main()
