"""Off-road vehicles on open desert (no forest, water, steep ground or cities) against Chungju, with the same settings.

Stuck = stayed within D metres in the last 20 minutes while off-road for at least 80 % of them.
Figure and numbers: rl_search/data/real/mojave/desert_compare.png / .json

    python -m rl_search.scripts.real_map.desert_compare
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.vehicle import EXIT, OFF, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
MAPS = {"충주": ("chungju", ((5000.0, 5000.0), (-9600.0, 15000.0), (-20000.0, -15000.0))),
        "모하비 사막": ("mojave", ((0.0, -3000.0), (-8000.0, 12000.0), (12000.0, -9000.0)))}
SETS = {"기본 (나가기 0.05/분, 복귀 0.3/분)": {}, "도로 밖 많이 (0.2/분, 0.1/분)": dict(p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1),
        "사용자 설정 (0.5/분, 0.3/분)": dict(p_leave_fast_min=.5, p_leave_slow_min=.5, p_return_min=.3)}
SEEDS = (0, 1)
HEADINGS = (45.0, 200.0)


def main():
    rows = []
    for mn, (key, lkps) in MAPS.items():
        net = load_net(key, 25.0)
        print(mn, "main road share %.3f, dead-end share %.3f" % (net.main_share, net.deadend_share), flush=True)
        simulate_vehicle(net, VehicleConfig(particles=20), lkps[0], 45.0, np.random.default_rng(0))
        print("  drivable off-road share at slope limit 15 deg: %.3f" % net.drive.mean(), flush=True)
        for sn, kw in SETS.items():
            for lkp in lkps:
                for seed in SEEDS:
                    r = simulate_vehicle(net, VehicleConfig(particles=2000, **kw), lkp, HEADINGS[seed], np.random.default_rng(seed))
                    dt = r.cfg.save_dt_s; T = min(int(3600/dt), len(r.track) - 1); k0 = T - int(1200/dt)
                    tr = np.asarray(r.track[:T + 1], float)
                    live = r.state[T] != EXIT
                    moved = np.hypot(*(tr[T] - tr[k0]).T)
                    off = (r.state[k0:T + 1] == OFF).mean(0)
                    s = r.state[:T + 1]
                    step = np.hypot(*(tr[k0 + 1:T + 1] - tr[k0:T]).transpose(2, 0, 1)).sum(0)
                    st = live & (moved < 500) & (off >= .8)
                    rows.append(dict(map=mn, setting=sn, off=float((s[s != EXIT] == OFF).mean()), st5=float(st.sum()/live.sum()), st2=float((live & (moved < 2000) & (off >= .8)).sum()/live.sum()),
                                     path=float(np.median(step[st])/1000) if st.any() else np.nan, d60=float(np.median(np.hypot(*(tr[T] - np.asarray(r.lkp_road)).T)[live])/1000),
                                     exit=float((~live).mean())))
    sets, maps = list(SETS), list(MAPS)
    summ = {m: {s: {k: float(np.nanmean([r[k] for r in rows if r["map"] == m and r["setting"] == s])) for k in ("off", "st5", "st2", "path", "d60", "exit")} for s in sets} for m in maps}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.3), layout="constrained")
    x = np.arange(len(sets)); w = .36; cols = {"충주": "#2a7a4d", "모하비 사막": "#c98a2b"}
    for ax, (key, ttl) in zip(axs, (("st2", "20분 동안 2 km 안에 머문 도로 밖 차 (%)"), ("st5", "20분 동안 500 m 안에 머문 도로 밖 차 (%)"), ("off", "도로 밖에 있는 시간 (%)"))):
        for j, m in enumerate(maps):
            v = [100*summ[m][s][key] for s in sets]
            ax.bar(x + (j - .5)*w, v, w, color=cols[m], label=m)
            for xx, y in zip(x + (j - .5)*w, v):
                ax.text(xx, y, f"{y:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" (", "\n(") for s in sets]); ax.set_ylabel("%"); ax.legend(); ax.set_title(ttl, fontsize=12)
    fig.suptitle("막힌 곳이 거의 없는 사막과 충주의 비교 (같은 표적 설정, 지도별 LKP 3곳, seed 2개, 60분 시점)", fontsize=14)
    p = DATA/"mojave"/"desert_compare.png"
    fig.savefig(p, dpi=95)
    p.with_suffix(".json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    for m in maps:
        print(m)
        for s in sets:
            print("  ", s, {k: round(v, 3) for k, v in summ[m][s].items()})


if __name__ == "__main__":
    main()
