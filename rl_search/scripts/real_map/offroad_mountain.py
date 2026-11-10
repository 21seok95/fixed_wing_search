"""What if cars could cross the mountains (forest and steep ground), slowly? Vehicle simulation only (no search).

Figure and numbers: rl_search/data/real/chungju/offroad_mountain.png / .json

    python -m rl_search.scripts.real_map.offroad_mountain
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
LKPS = ((5000.0, 5000.0), (-9600.0, 15000.0), (-20000.0, -15000.0))
SEEDS = (0, 1)
TARGETS = {"기본 (나가기 0.05/분)": {}, "논밭 많이 (나가기 0.2/분)": dict(p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1)}
MODES = [("지금 (산은 못 지나감)", False, 1.0), ("산 통과, 속도 그대로", True, 1.0), ("산 통과, 산에서 속도 1/3", True, 1/3)]


def main():
    net = load_net("chungju", 25.0)
    simulate_vehicle(net, VehicleConfig(particles=50), LKPS[0], 45.0, np.random.default_rng(0))
    t, base = net.terr, net.drive.copy()
    mountain = t.forest | (t.slope_deg >= 15.0)
    opened = ~(t.water | t.fords | t.city)
    rows = []
    for tn, kw in TARGETS.items():
        for mn, op, f in MODES:
            net.drive = opened if op else base
            fac = np.ones(base.shape, np.float32); fac[mountain] = f
            net.off_factor = fac if op else None
            for lkp in LKPS:
                for seed in SEEDS:
                    r = simulate_vehicle(net, VehicleConfig(particles=2000, **kw), lkp, 45.0, np.random.default_rng(seed))
                    dt = r.cfg.save_dt_s; T = int(3600/dt); k0 = T - int(1200/dt)
                    tr = np.asarray(r.track[:T + 1], float)
                    moved = np.hypot(*(tr[T] - tr[k0]).T)
                    live = r.state[T] != EXIT
                    off = (r.state[k0:T + 1] == OFF).mean(0)
                    ro, co = net.cell_of(tr[T])
                    s = r.state[:T + 1]
                    rows.append(dict(target=tn, mode=mn, off=float((s[s != EXIT] == OFF).mean()), st5=float((live & (moved < 500) & (off >= .8)).sum()/live.sum()),
                                     st2=float((live & (moved < 2000) & (off >= .8)).sum()/live.sum()), d60=float(np.median(np.hypot(*(tr[T] - tr[0]).T)[live])/1000),
                                     in_mountain=float((live & mountain[ro, co]).sum()/live.sum())))
    net.drive, net.off_factor = base, None
    summ = {tn: {mn: {k: float(np.mean([r[k] for r in rows if r["target"] == tn and r["mode"] == mn])) for k in ("off", "st5", "st2", "d60", "in_mountain")} for mn, _, _ in MODES} for tn in TARGETS}
    fig, axs = plt.subplots(1, 2, figsize=(17, 6), layout="constrained")
    keys = [("off", "도로 밖 시간", "#27ae60"), ("in_mountain", "60분에 산(숲·급경사)에 있는 비율", "#8d6e63"), ("st2", "20분간 2 km 안에 머묾", "#c0392b"), ("st5", "20분간 500 m 안에 머묾", "#7a1f1f")]
    for ax, tn in zip(axs, TARGETS):
        x = np.arange(len(MODES)); w = .2
        for k, (key, lab, c) in enumerate(keys):
            v = [100*summ[tn][m[0]][key] for m in MODES]
            ax.bar(x + (k - 1.5)*w, v, w, color=c, label=lab)
            for xx, y in zip(x + (k - 1.5)*w, v):
                ax.text(xx, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"{m[0]}\n60분 거리 {summ[tn][m[0]]['d60']:.1f} km" for m in MODES]); ax.set_ylabel("%"); ax.legend(fontsize=9); ax.set_title(tn, fontsize=12)
    fig.suptitle("차가 산을 지나갈 수 있다면 (LKP 3곳, seed 2개, 60분 시점)", fontsize=14)
    p = DATA/"chungju"/"offroad_mountain.png"
    fig.savefig(p, dpi=95)
    p.with_suffix(".json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    for tn, d in summ.items():
        print(tn)
        for mn, v in d.items():
            print("  ", mn, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
