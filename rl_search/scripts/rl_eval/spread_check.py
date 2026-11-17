"""How far the particles spread with the new target speeds (user 2026-11-17) against the R10-R13 model, on the 60 km maps.

New speeds: paved (fast roads) mean 50 km/h, std 10 (30-70); unpaved (slow roads) 30, std 5 (20-40); fields and mountains (off-road) 10,
std 2.5 (5-15); a per-particle driver tendency, clipped at 2 std (VehicleConfig.speed_clip_sd); targets never stop. The R10-R13 model:
fast roads 20-50 km/h per scenario, slow 2/3 of it, per-particle 15 %, clipped at +-30 % (rl_eval_set.nostop_vc). Both use the same LKP
(central 30 km square, on a road), heading, type rules (road / mixed / desert leaving and returning) and route variables. Scenarios that
leave the map are kept (the share that leaves is what is measured).

Per scenario and minute: the share of particles that left the map, and R95 = the radius around the LKP holding 95 % of the particles
(particles that left count as farther than any radius). The map size a scenario needs at time T is estimated as the LKP's distance from
the map centre (Chebyshev) plus R95 extrapolated with its growth rate before particles leave.

    python -m rl_search.scripts.rl_eval.spread_check --n 12 --workers 8
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
from rl_search.rl_eval_set import V2_SPEEDS as NEW      # the new target speeds (now kept with the redesigned scenarios)
MINUTES = np.arange(0, 101, 2)
_W = {}


def _init(regions, particles):
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import preload
    from dataclasses import replace
    for r in regions:
        st = preload(r)
        _W[r] = VehicleScenarios(st["sampler"], replace(vehicle_config(), particles=particles), region=r, net=st["net"])


def job(a):
    region, typ, i, variant = a
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, nostop_vc, route_vc
    warnings.filterwarnings("ignore")
    vs = _W[region]
    s = 7_000_000 + 1000*TYPES.index(typ) + i + (500 if region == "mojave" else 0)
    rng = np.random.default_rng(s)
    vc = multi_vc(rng, typ)
    vc = (nostop_vc(vc) if variant == "now" else vc | NEW) | route_vc(s)      # NEW also replaces the desert type's off-road speed
    lkp = diverse_lkp(vs.net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=region, net=vs.net)
    scen = vs2.make_at(s, lkp, heading, start=(0.0, 0.0))
    tr, ex, dt = scen.track, np.asarray(scen.exit_step).astype(np.int64), float(scen.save_dt_s)
    L = np.asarray(scen.lkp, float)[:2]
    out = dict(region=region, typ=typ, i=i, variant=variant, lkp_cheb_km=float(np.abs(L).max()/1000), exit=[], r95=[], r50=[])
    for m in MINUTES:
        k = min(int(round(m*60/dt)), tr.shape[0] - 1)
        gone = (ex >= 0) & (ex*dt <= m*60)
        d = np.hypot(*(tr[k] - L).T)/1000
        d[gone] = np.inf
        out["exit"].append(float(gone.mean()))
        out["r95"].append(float(np.percentile(d, 95)))
        out["r50"].append(float(np.percentile(d, 50)))
    r = np.array(out["r95"])
    ok = np.isfinite(r) & (MINUTES >= 10)
    out["v95_km_min"] = float(np.polyfit(MINUTES[ok], r[ok], 1)[0]) if ok.sum() >= 3 else float("nan")
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=["amboy", "mojave"])
    ap.add_argument("--n", type=int, default=12, help="scenarios per map and type")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    jobs = [(r, t, i, v) for r in args.regions for t in TYPES for i in range(args.n) for v in ("now", "new")]
    _init(args.regions, args.particles)                        # in the parent: the forked workers share the terrain
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"spread_check.json").write_text(json.dumps(rows))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = {"amboy": "#2a78d6", "mojave": "#eb6834"}
    RN = {"amboy": "암보이", "mojave": "바스토"}
    VN = {"now": "지금 (R10~R13)", "new": "새 속도"}
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), facecolor=BG)
    L = [f"# 새 도망자 속도에서 입자가 퍼지는 정도 (60 km 지도, 지도·유형마다 시나리오 {args.n}개, 입자 {args.particles}개)", "",
         "| 지도 | 유형 | 속도 | 30분 벗어남 | 60분 벗어남 | 100분 벗어남 | 30분 R95 (km) | 60분 R95 (km) | R95 증가 (km/분) | 필요 반폭 60분 / 100분 (km, 중앙값) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for j, t in enumerate(TYPES):
        for r in args.regions:
            for v, ls in (("now", "--"), ("new", "-")):
                rr = [x for x in rows if x["region"] == r and x["typ"] == t and x["variant"] == v]
                ex = np.mean([x["exit"] for x in rr], 0)
                r95 = np.median(np.array([x["r95"] for x in rr]), 0)
                axs[0, j].plot(MINUTES, 100*ex, ls=ls, color=COL[r], lw=2, label=f"{RN[r]} · {VN[v]}")
                axs[1, j].plot(MINUTES, np.where(np.isfinite(r95), r95, np.nan), ls=ls, color=COL[r], lw=2, label=f"{RN[r]} · {VN[v]}")
                v95 = np.array([x["v95_km_min"] for x in rr])
                need = lambda T: np.median([x["lkp_cheb_km"] + x["v95_km_min"]*T for x in rr])
                at = lambda key, m: np.mean([x[key][list(MINUTES).index(m)] for x in rr])
                med = lambda key, m: np.median([x[key][list(MINUTES).index(m)] for x in rr])
                f = lambda z: "> 지도" if not np.isfinite(z) else f"{z:.1f}"
                L.append(f"| {RN[r]} | {TN[t]} | {VN[v]} | {100*at('exit', 30):.1f}% | {100*at('exit', 60):.1f}% | {100*at('exit', 100):.1f}% | "
                         f"{f(med('r95', 30))} | {f(med('r95', 60))} | {np.nanmedian(v95):.2f} | {need(60):.0f} / {need(100):.0f} |")
        axs[0, j].set_title(f"{TN[t]}: 60 km 지도를 벗어난 입자 몫", loc="left", fontsize=11, color=INK)
        axs[0, j].set_ylabel("%", fontsize=9, color=MUTED); axs[0, j].set_ylim(0, 100)
        axs[1, j].set_title(f"{TN[t]}: LKP에서 입자 95%가 들어가는 반경 R95 (중앙값, 5% 넘게 벗어나면 끊김)", loc="left", fontsize=11, color=INK)
        axs[1, j].set_ylabel("km", fontsize=9, color=MUTED)
        for ax in axs[:, j]:
            ax.set_xlabel("도주 시작 뒤 분", fontsize=9, color=MUTED)
            ax.grid(color=GRID, lw=.6); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("새 도망자 속도(포장 50 · 비포장 30 · 들판·산 10 km/h 평균)에서 입자가 퍼지는 정도: 실선 = 새 속도, 점선 = 지금(R10~R13)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(args.out/"spread_check.png", dpi=95, facecolor=BG)
    L += ["", "R95: LKP에서 입자 95%가 들어가는 반경(지도를 벗어난 입자는 무한히 먼 것으로 셈, 5% 넘게 벗어나면 '> 지도').",
          "필요 반폭: 시나리오마다 'LKP의 지도 중심에서의 거리(가로·세로 중 큰 값) + R95 증가 속도 × 시간', 지도를 벗어나기 전의 증가로 늘려 잡은 어림."]
    (args.out/"spread_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
