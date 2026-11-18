"""How many particles leave the map on the maps the next training and its validation use (user 2026-11-18: "on 120 km they do not
leave?"). Scenarios are made exactly as training makes them (rl_eval_set.V2 rules, LKP by diverse_lkp, 70 min tracks): the stretched maps
amboy_x2 / mojave_x2 (made on the 60 km map at half speed, then scaled, rl_search.stretch), the 120 km maps amboy120 / mojave120, and
Chungju 120 km with the mountain-terrain types (rl_eval_set.scenario_vc). For reference, the 60 km maps with full speed (what the
earlier training and the Chungju web flights used).

Per scenario: the share of particles outside the map at 30 / 60 / 70 min and the LKP's distance to the nearest map edge.

    python -m rl_search.scripts.rl_eval.exit_share_check --n 12 --workers 6
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"exit_share_check"
MAPS = (("amboy_x2", ("desert:road", "desert:mixed", "desert:desert"), "암보이 늘린 지도 (학습)"),
        ("mojave_x2", ("desert:road", "desert:mixed", "desert:desert"), "바스토 늘린 지도 (학습)"),
        ("amboy120", ("desert:road", "desert:mixed", "desert:desert"), "암보이 120 km"),
        ("mojave120", ("desert:road", "desert:mixed", "desert:desert"), "바스토 120 km"),
        ("chungju120", ("mountain:road", "mountain:mountain"), "충주 120 km"),
        ("mojave", ("desert:road", "desert:mixed", "desert:desert"), "바스토 60 km (참고)"),
        ("chungju", ("mountain:road", "mountain:mountain"), "충주 60 km (참고)"))
TYPE_NAMES = {"road": "도로형", "mixed": "혼합형", "desert": "들판형", "mountain": "산악형"}
_ST = {}


def _vs(region):
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import preload
    st = preload(region)
    return VehicleScenarios(st.get("base_sampler", st["sampler"]), vehicle_config(), region=st.get("base", region), net=st["net"]), \
        float(st.get("k", 1.0)), st["terr"]


def _init():
    for region, _, _ in MAPS:
        _ST[region] = _vs(region)


def job(a):
    warnings.filterwarnings("ignore")
    region, kind, i = a
    from dataclasses import replace
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import diverse_lkp, scenario_vc
    vs, k, terr = _ST[region]
    m = next(m for m in MAPS if m[0] == region)
    seed = 9_300_000 + 1000*MAPS.index(m) + 100*m[1].index(kind) + i
    if k != 1.0:                                                  # stretched: training's own path (base map at speed / k, scaled)
        from rl_search.stretch import stretched_training_scenario
        scen = stretched_training_scenario(vs, seed, kind.split(":")[1], k)[0]
    else:
        rng = np.random.default_rng(seed)
        vc = scenario_vc(rng, kind, seed)
        lkp = diverse_lkp(vs.net, rng)
        heading = float(rng.uniform(0.0, 360.0))
        vs2 = VehicleScenarios(vs.sampler, replace(vs.vcfg, **vc), region=vs.region, net=vs.net)
        scen = vs2.make_at(seed, lkp, heading, start=(0.0, 0.0))
    ex = np.asarray(scen.exit_step, int)
    dt = float(scen.save_dt_s)
    half = float(terr.axis[-1])
    out = {f"out{m}": float(((ex >= 0) & (ex*dt <= m*60)).mean()) for m in (30, 60, 70)}
    lkp = np.asarray(scen.lkp, float)
    return dict(region=region, kind=kind, i=i, half_km=half/1000, lkp_edge_km=float(half - np.abs(lkp).max())/1000, **out)


def run(n, workers):
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(r, kd, i) for r, kinds, _ in MAPS for kd in kinds for i in range(n)]
    rows = []
    with get_context("fork").Pool(workers) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec)
            print(f"{rec['region']:10s} {rec['kind']:18s} {rec['i']:2d}: out 60 min {rec['out60']:.1%}, 70 min {rec['out70']:.1%}", flush=True)
    (WORK/"rows.json").write_text(json.dumps(rows))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    rows = json.loads((WORK/"rows.json").read_text())
    colors = {"road": "#2a6fdb", "mixed": "#e07b39", "desert": "#3a9a5b", "mountain": "#8e44ad"}
    fig, ax = plt.subplots(figsize=(14, 5.5))
    xt, xl = [], []
    x = 0
    for region, kinds, name in MAPS:
        for kd in kinds:
            typ = kd.split(":")[1]
            v = np.array([r["out60"] for r in rows if r["region"] == region and r["kind"] == kd])*100
            ax.scatter(np.full(len(v), x) + np.random.default_rng(x).uniform(-.15, .15, len(v)), v, s=14, color=colors[typ], alpha=.6, lw=0)
            ax.plot([x - .3, x + .3], [v.mean()]*2, color=colors[typ], lw=2.5)
            ax.text(x, v.max() + 1.5, f"{v.mean():.1f}%", ha="center", fontsize=7, color="#333")
            x += 1
        xt.append(x - (len(kinds) + 1)/2)
        xl.append(name)
        ax.axvline(x - .5 + .5, color="#ddd", lw=.8)
        x += 1
    ax.set_xticks(xt, xl, fontsize=9)
    ax.set_ylabel("60분까지 지도 밖으로 나간 입자 (%)")
    for typ, c in colors.items():
        ax.plot([], [], "s", color=c, label=TYPE_NAMES[typ])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color="#eee")
    ax.set_title("시나리오마다 60분까지 지도 밖으로 나간 입자 몫 (점 = 시나리오, 선 = 평균; 학습과 같은 생성 규칙 V2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/"exit_share_check.png", dpi=120)
    L = ["# 지도 밖으로 나가는 입자 몫 (학습 생성 규칙 V2)", "", "| 지도 | 유형 | 시나리오 | 30분 | 60분 평균 (최대) | 70분 평균 (최대) | 60분에 5% 넘게 나간 시나리오 |",
         "|---|---|---|---|---|---|---|"]
    for region, kinds, name in MAPS:
        for kd in kinds:
            rr = [r for r in rows if r["region"] == region and r["kind"] == kd]
            f = lambda key: np.array([r[key] for r in rr])*100
            L.append(f"| {name} | {TYPE_NAMES[kd.split(':')[1]]} | {len(rr)} | {f('out30').mean():.2f}% | {f('out60').mean():.2f}% ({f('out60').max():.1f}%) | "
                     f"{f('out70').mean():.2f}% ({f('out70').max():.1f}%) | {int((f('out60') > 5).sum())} |")
    (OUT/"exit_share_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=("all", "run", "figure"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.what in ("all", "run"):
        run(a.n, a.workers)
    if a.what in ("all", "figure"):
        figure()


if __name__ == "__main__":
    main()
