"""Off-road particles that get stuck: are they standing still or circling, how soon can that be told, and would closing small or
narrow open land remove them? Settings: "논밭 많이" (leave 0.2/min, return 0.1/min). Vehicle simulation only (no search).

Figure and numbers: rl_search/data/real/chungju/offroad_closure.png / .json

    python -m rl_search.scripts.real_map.offroad_closure
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, label

from rl_search.real_terrain import DATA
from rl_search.vehicle import EXIT, OFF, STOP, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LKPS = ((5000.0, 5000.0), (-9600.0, 15000.0), (-20000.0, -15000.0))
SEEDS = (0, 1)
KW = dict(particles=2000, p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1)
WINDOWS = (10, 30, 60, 120, 180)
RATIO = .3
VARIANTS = [("닫지 않음", 0, 0.0), ("폭 100 m 미만 닫기", 50, 0.0), ("폭 200 m 미만 닫기", 100, 0.0), ("폭 300 m 미만 닫기", 150, 0.0),
            ("0.25 km² 미만 닫기", 0, .25), ("1 km² 미만 닫기", 0, 1.0), ("폭 200 m + 1 km²", 100, 1.0), ("폭 300 m + 4 km²", 150, 4.0)]


def disk(r_cells):
    y, x = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    return x*x + y*y <= r_cells*r_cells


def closed(drive, cell, r_m, area):
    m = drive.copy()
    if r_m > 0:
        m &= binary_opening(drive, structure=disk(int(round(r_m/cell))))
    if area > 0:
        lab, n = label(m)
        size = np.bincount(lab.ravel(), minlength=n + 1)*(cell/1000)**2
        keep = size >= area
        keep[0] = False
        m = keep[lab]
    return m


def main():
    net = load_net("chungju", 25.0)
    r0 = simulate_vehicle(net, VehicleConfig(particles=50), LKPS[0], 45.0, np.random.default_rng(0))
    base, cell = net.drive.copy(), float(net.terr.cell)
    dt = r0.cfg.save_dt_s
    rows, detail = [], []
    for name, rm, area in VARIANTS:
        net.drive = closed(base, cell, rm, area)
        for lkp in LKPS:
            for seed in SEEDS:
                r = simulate_vehicle(net, VehicleConfig(**KW), lkp, 45.0, np.random.default_rng(seed))
                T = int(3600/dt); k0 = T - int(1200/dt)
                tr = np.asarray(r.track[:T + 1], float)
                moved = np.hypot(*(tr[T] - tr[k0]).T)
                live = r.state[T] != EXIT
                s = r.state[k0:T + 1]
                offs = (s == OFF).mean(0)
                st5, st2 = live & (moved < 500) & (offs >= .8), live & (moved < 2000) & (offs >= .8)
                L = np.hypot(*(tr[T] - tr[0]).T)
                rows.append(dict(name=name, off=float((r.state[:T + 1][r.state[:T + 1] != EXIT] == OFF).mean()), st5=float(st5.sum()/live.sum()), st2=float(st2.sum()/live.sum()),
                                 d60=float(np.median(L[live])/1000), kept=float(net.drive.sum()/base.sum())))
                if name == VARIANTS[0][0]:
                    step = np.hypot(*(tr[k0 + 1:T + 1] - tr[k0:T]).transpose(2, 0, 1))
                    sel = np.flatnonzero(st5)
                    ok = (r.state[k0 + 1:T + 1] == OFF) & (r.state[k0:T] == OFF)
                    item = dict(path_km=[float(step[:, i].sum()/1000) for i in sel], still=float((ok[:, sel] & (step[:, sel] < 1.0)).sum()/max(ok[:, sel].sum(), 1)) if len(sel) else np.nan)
                    moving = live & (r.state[T] == OFF) & (moved >= 2000)
                    d = {}
                    for W in WINDOWS:
                        w = int(round(W/dt))
                        frames = range(k0 + w, T + 1, max(1, w))
                        hit = {"stuck": [0, 0], "moving": [0, 0]}
                        for f in frames:
                            win = r.state[f - w:f + 1]
                            good = (win == OFF).all(0)
                            v = r.speed_kmh[f - w:f + 1].astype(float).mean(0)/3.6
                            net_d = np.hypot(*(tr[f] - tr[f - w]).T)
                            flag = good & (net_d < RATIO*v*w*dt)
                            for key, mask in (("stuck", st5), ("moving", moving)):
                                hit[key][0] += int((flag & mask).sum()); hit[key][1] += int((good & mask).sum())
                        d[W] = {k: v[0]/max(v[1], 1) for k, v in hit.items()}
                    item["win"] = d
                    detail.append(item)
    net.drive = base
    paths = np.concatenate([d["path_km"] for d in detail])
    still = float(np.nanmean([d["still"] for d in detail]))
    win = {W: {k: float(np.mean([d["win"][W][k] for d in detail])) for k in ("stuck", "moving")} for W in WINDOWS}
    summ = {n: {k: float(np.mean([r[k] for r in rows if r["name"] == n])) for k in ("off", "st5", "st2", "d60", "kept")} for n, _, _ in VARIANTS}
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5), layout="constrained")
    ax = axs[0]
    ax.hist(paths, bins=np.arange(0, 6.1, .25), color="#2a4d7a")
    ax.axvline(4, color="#c0392b", lw=1.4, label="12 km/h로 20분 직진 = 4 km")
    ax.set_xlabel("마지막 20분 동안 실제로 달린 거리 (km)"); ax.set_ylabel("particle 수"); ax.legend()
    ax.set_title(f"500 m 안에 머문 particle: 맴돌고 있나 (중앙값 {np.median(paths):.1f} km, 제자리 10초 구간 {100*still:.1f}%)", fontsize=11)
    ax = axs[1]
    x = np.arange(len(WINDOWS))
    ax.bar(x - .2, [100*win[W]["stuck"] for W in WINDOWS], .4, color="#c0392b", label="갇힌 particle을 잡는 비율")
    ax.bar(x + .2, [100*win[W]["moving"] for W in WINDOWS], .4, color="#bbb", label="잘 가는 particle을 잘못 잡는 비율")
    ax.set_xticks(x); ax.set_xticklabels([f"{w}초" for w in WINDOWS]); ax.set_ylabel("% (판단 시점 기준)"); ax.legend()
    ax.set_title(f"판단 시간별: 직선거리 < 달릴 수 있는 거리의 {RATIO:.0%}", fontsize=11)
    ax = axs[2]
    names = [v[0] for v in VARIANTS]; y = np.arange(len(names))
    ax.barh(y - .2, [100*summ[n]["st2"] for n in names], .4, color="#c0392b", label="20분간 2 km 안")
    ax.barh(y + .2, [100*summ[n]["off"] for n in names], .4, color="#27ae60", label="도로 밖 시간")
    for i, n in enumerate(names):
        ax.text(100*summ[n]["off"] + .5, i + .2, f"남는 땅 {100*summ[n]['kept']:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlabel("%"); ax.legend(fontsize=9)
    ax.set_title("좁거나 작은 논밭을 닫았을 때", fontsize=11)
    fig.suptitle("논밭에서 갇히는 particle 진단 (논밭 많이: 나가기 0.2/분, 복귀 0.1/분, LKP 3곳, seed 2개)", fontsize=13)
    p = DATA/"chungju"/"offroad_closure.png"
    fig.savefig(p, dpi=95)
    p.with_suffix(".json").write_text(json.dumps(dict(path_median=float(np.median(paths)), still=still, windows=win, variants=summ), indent=1, ensure_ascii=False))
    print("stuck500 path km: median %.2f, p10 %.2f, p90 %.2f | zero-move 10 s steps %.3f" % (np.median(paths), *np.percentile(paths, [10, 90]), still))
    for W in WINDOWS:
        print(W, "s: detect %.3f, false alarm on moving %.3f" % (win[W]["stuck"], win[W]["moving"]))
    for n in names:
        print(n, {k: round(v, 3) for k, v in summ[n].items()})
    print("Saved", p)


if __name__ == "__main__":
    main()
