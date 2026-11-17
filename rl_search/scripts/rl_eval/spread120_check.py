"""How far the particles spread on the 120 km maps with everything decided for the new scenarios (docs/scenario_redesign.md, user
2026-11-17): the new target speeds (spread_check.NEW), the agreed particle rules (vehicle.NO_BACKTRACK), particles that leave the map
going on straight (exit_straight), 70 min tracks (launch at the report time, 60 min of flight, 10 min of prediction), the LKP on a road in
the central 30 km. Compared with the same speeds on the 60 km maps under the earlier rules (spread_check.py, report/spread_check.json).

Per scenario and minute: the share of particles outside the 120 km map, and R50 / R95 = radius around the LKP holding 50 / 95 % of the
particles (the ones outside count where they are, since they keep moving). Also the map half-width a scenario needs at 60 / 70 min: the
LKP's distance from the map centre (Chebyshev) plus the farthest 1 % of the particles from it, in x or y.

    python -m rl_search.scripts.rl_eval.spread120_check --n 12 --workers 6
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
REG = {"amboy120": "암보이", "mojave120": "바스토"}
OLD = {"amboy120": "amboy", "mojave120": "mojave"}
MINUTES = np.arange(0, 71, 2)
_NET = {}


def _init(regions):
    from rl_search.rl_real import preload
    for r in regions:
        _NET[r] = preload(r)["net"]


def job(a):
    region, typ, i = a
    warnings.filterwarnings("ignore")
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, route_vc, vehicle_config
    from rl_search.scripts.rl_eval.spread_check import NEW as SPEEDS
    from rl_search.vehicle import NO_BACKTRACK, simulate_vehicle
    net = _NET[region]
    s = 7_000_000 + 1000*TYPES.index(typ) + i + (500 if region.startswith("mojave") else 0)
    rng = np.random.default_rng(s)
    vc = multi_vc(rng, typ) | SPEEDS | route_vc(s) | NO_BACKTRACK
    lkp = diverse_lkp(net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=4200.0, exit_straight=True)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    tr, ex, dt = res.track, res.exit_step.astype(np.int64), float(cfg.save_dt_s)
    L = np.asarray(res.lkp_road, float)
    half = float(net.terr.axis[-1])
    out = dict(region=region, typ=typ, i=i, lkp_cheb_km=float(np.abs(L).max()/1000), outside=[], r50=[], r95=[], need=[])
    for m in MINUTES:
        k = min(int(round(m*60/dt)), tr.shape[0] - 1)
        P = tr[k].astype(float)
        d = np.hypot(*(P - L).T)/1000
        out["outside"].append(float((np.abs(P).max(axis=1) > half).mean()))
        out["r50"].append(float(np.percentile(d, 50)))
        out["r95"].append(float(np.percentile(d, 95)))
        far = np.percentile(np.abs(P - L), 99, axis=0).max()/1000          # x or y extent of 99 % of the particles from the LKP
        out["need"].append(float(np.abs(L).max()/1000 + far))
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=["amboy120", "mojave120"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    _init(args.regions)
    jobs = [(r, t, i) for r in args.regions for t in TYPES for i in range(args.n)]
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    (args.out/"spread120_check.json").write_text(json.dumps(rows))
    old = json.loads((args.out/"spread_check.json").read_text())
    figure(rows, old, args)


def figure(rows, old, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = {"amboy120": "#2a78d6", "mojave120": "#eb6834"}
    OM = np.arange(0, 101, 2)
    fig, axs = plt.subplots(3, 3, figsize=(21, 15.5), facecolor=BG)
    L = [f"# 120 km 지도에서 입자가 퍼지는 정도 (새 시나리오 설정, 지도·유형마다 {args.n}개 × 입자 2,000개)", "",
         "| 지도 | 유형 | 지도 밖 60분 / 70분 | R50 60분 (km) | R95 60분 / 70분 (km) | 필요 반폭 60분 / 70분 (중앙값, 최대) | 60 km 지도·예전 규칙: 밖 60분, 필요 반폭 60분 |",
         "|---|---|---|---|---|---|---|"]
    i60, i70 = list(MINUTES).index(60), list(MINUTES).index(70)
    for j, t in enumerate(TYPES):
        for r in REG:
            rr = [x for x in rows if x["region"] == r and x["typ"] == t]
            if not rr:
                continue
            oo = [x for x in old if x["region"] == OLD[r] and x["typ"] == t and x["variant"] == "new"]
            outside = np.mean([x["outside"] for x in rr], 0)
            r95 = np.median([x["r95"] for x in rr], 0)
            need = np.array([x["need"] for x in rr])
            axs[0, j].plot(MINUTES, 100*outside, color=COL[r], lw=2.2, label=f"{REG[r]} 120 km · 새 설정")
            if oo:
                axs[0, j].plot(OM, 100*np.mean([x["exit"] for x in oo], 0), color=COL[r], lw=1.4, ls=":", label=f"{REG[r]} 60 km · 새 속도, 예전 규칙")
            axs[1, j].plot(MINUTES, r95, color=COL[r], lw=2.2, label=f"{REG[r]}: R95 (중앙값)")
            axs[1, j].plot(MINUTES, np.median([x["r50"] for x in rr], 0), color=COL[r], lw=1.4, ls="--", label=f"{REG[r]}: R50 (중앙값)")
            axs[2, j].plot(MINUTES, np.median(need, 0), color=COL[r], lw=2.2, label=f"{REG[r]}: 중앙값")
            axs[2, j].fill_between(MINUTES, np.percentile(need, 10, 0), need.max(0), color=COL[r], alpha=0.15, lw=0, label=f"{REG[r]}: 10% ~ 최대")
            old60 = f"{100*np.mean([x['exit'][list(OM).index(60)] for x in oo]):.1f}%, {np.median([x['lkp_cheb_km'] + x['v95_km_min']*60 for x in oo]):.0f} km" if oo else "-"
            L.append(f"| {REG[r]} | {TN[t]} | {100*outside[i60]:.1f}% / {100*outside[i70]:.1f}% | {np.median([x['r50'][i60] for x in rr]):.1f} | "
                     f"{r95[i60]:.1f} / {r95[i70]:.1f} | {np.median(need[:, i60]):.0f} / {np.median(need[:, i70]):.0f} ({need[:, i60].max():.0f}, {need[:, i70].max():.0f}) | {old60} |")
        axs[0, j].set_title(f"{TN[t]}: 지도 밖에 있는 입자 몫 (%)", loc="left", fontsize=11.5, color=INK)
        axs[1, j].set_title(f"{TN[t]}: LKP에서 입자 50% / 95%가 드는 반경 (km)", loc="left", fontsize=11.5, color=INK)
        axs[2, j].set_title(f"{TN[t]}: 필요한 지도 반폭 (km; 120 km 지도 = 60)", loc="left", fontsize=11.5, color=INK)
        axs[2, j].axhline(60, color=INK, lw=1, ls="--"); axs[2, j].axhline(30, color=MUTED, lw=1, ls=":")
        axs[2, j].text(1, 61, "120 km 지도의 반폭", fontsize=8.5, color=INK); axs[2, j].text(1, 31, "60 km 지도의 반폭", fontsize=8.5, color=MUTED)
        for ax in axs[:, j]:
            ax.set_xlabel("도주 시작 뒤 분", fontsize=9, color=MUTED); ax.set_xlim(0, 100 if ax is axs[0, j] else 70)
            ax.grid(color=GRID, lw=.6); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
            for s_ in ax.spines.values():
                s_.set_color(GRID)
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        axs[0, j].set_ylim(0, 100)
    fig.suptitle("120 km 지도에서 입자가 퍼지는 정도: 새 속도 + 합의한 입자 규칙 + 지도 밖 직진, 70분 궤적 (점선 = 60 km 지도, 새 속도, 예전 규칙)",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(args.out/"spread120_check.png", dpi=85, facecolor=BG)
    L += ["", "필요 반폭: 시나리오마다 'LKP의 지도 중심에서의 거리(가로·세로 중 큰 값) + LKP에서 입자 99%가 드는 가로·세로 거리'. 예전 열은 spread_check의 어림(R95 증가 속도로 늘림)."]
    (args.out/"spread120_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
