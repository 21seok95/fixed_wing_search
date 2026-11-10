"""Off-road movement is a choice among nine headings every 5 s. How much do vehicles circle, and does holding the course help?

Same maps and stuck measure as desert_compare.py; "도로 밖 많이" setting (leave 0.2/min, return 0.1/min).
Figure and numbers: rl_search/data/real/mojave/offroad_keep.png / .json

    python -m rl_search.scripts.real_map.offroad_keep
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.desert_compare import HEADINGS, MAPS, SEEDS
from rl_search.vehicle import EXIT, OFF, OFF_OFFSETS, VehicleConfig, load_net, simulate_vehicle

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
KEEPS = (2.0, 4.0, 8.0, 16.0)
KW = dict(particles=2000, p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1)


def main():
    print("heading change every 5 s if only the keep term acts (rms deg):",
          {k: round(float(np.degrees(np.sqrt(np.sum(np.exp(k*np.cos(OFF_OFFSETS))*OFF_OFFSETS**2)/np.sum(np.exp(k*np.cos(OFF_OFFSETS)))))), 1) for k in KEEPS})
    out = {}
    for mn, (key, lkps) in MAPS.items():
        net = load_net(key, 25.0)
        for k in KEEPS:
            rows = []
            for lkp in lkps:
                for seed in SEEDS:
                    r = simulate_vehicle(net, VehicleConfig(w_off_keep=k, **KW), lkp, HEADINGS[seed], np.random.default_rng(seed))
                    dt = r.cfg.save_dt_s; T = min(int(3600/dt), len(r.track) - 1); k0 = T - int(1200/dt)
                    tr = np.asarray(r.track[:T + 1], float)
                    live = r.state[T] != EXIT
                    moved = np.hypot(*(tr[T] - tr[k0]).T)
                    off = (r.state[k0:T + 1] == OFF).mean(0)
                    s = r.state[:T + 1]
                    rows.append((float((s[s != EXIT] == OFF).mean()), float((live & (moved < 500) & (off >= .8)).sum()/live.sum()), float((live & (moved < 2000) & (off >= .8)).sum()/live.sum()),
                                 float(np.median(np.hypot(*(tr[T] - np.asarray(r.lkp_road)).T)[live])/1000)))
            out.setdefault(mn, {})[k] = dict(zip(("off", "st5", "st2", "d60"), np.mean(rows, axis=0).tolist()))
    maps = list(MAPS); x = np.arange(len(KEEPS)); w = .36; cols = dict(zip(maps, ("#2a7a4d", "#c98a2b")))
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.8), layout="constrained")
    for ax, (key, t, f) in zip(axs, (("st2", "20분 동안 2 km 안에 머묾 (%)", 100), ("st5", "20분 동안 500 m 안에 머묾 (%)", 100), ("off", "도로 밖 시간 (%)", 100), ("d60", "60분 뒤 LKP에서 거리, 중앙값 (km)", 1))):
        for j, m in enumerate(maps):
            v = [f*out[m][k][key] for k in KEEPS]
            ax.bar(x + (j - .5)*w, v, w, color=cols[m], label=m)
            for xx, y in zip(x + (j - .5)*w, v):
                ax.text(xx, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"{k:g}" for k in KEEPS]); ax.set_xlabel("진행 방향 유지 가중치 (w_off_keep, 지금 2)"); ax.legend(fontsize=9); ax.set_title(t, fontsize=11)
    fig.suptitle("도로 밖에서 방향을 더 유지하게 하면 (나가기 0.2/분, 복귀 0.1/분, 지도별 LKP 3곳, seed 2개, 60분 시점)", fontsize=13)
    p = DATA/"mojave"/"offroad_keep.png"
    fig.savefig(p, dpi=95)
    p.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for m in maps:
        print(m)
        for k in KEEPS:
            print("  keep", k, {a: round(b, 3) for a, b in out[m][k].items()})
    print("Saved", p)


if __name__ == "__main__":
    main()
