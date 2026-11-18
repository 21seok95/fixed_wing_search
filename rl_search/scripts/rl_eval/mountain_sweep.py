"""Mountain terrain, road type: what each scenario variable does (user 2026-11-18: prepare the training on mountain terrain road type;
fewer detours from big into small roads, a dead end turned at). On Chungju 120 km, the same 20 scenarios (seed, LKP in the central 30 km,
heading) with one variable changed at a time from a base (the training middle values: heading spread 35, detour 0.075, drift 2.5,
goal sector 20, speeds 50 +- 10 / 30 +- 5, the agreed rules with dead ends turned at, 70 min, 2,000 particles).

Per scenario at 30 / 60 min: R95 around the LKP, branches (branch_check.branches), 1 km cells with a particle, share outside the map,
and "circling": particles that within 3 min drove 1 km or more and ended within 150 m of where they were (small loops, back and forth).

    python -m rl_search.scripts.rl_eval.mountain_sweep run --n 20 --workers 8
    python -m rl_search.scripts.rl_eval.mountain_sweep figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REGION = "chungju120"
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"mountain_sweep"
BASE = dict(heading_std_deg=35.0, p_detour=0.075, reorient_std_deg_min=2.5, goal_sector_deg=20.0)
SWEEPS = {
    "detour": ("갈림길 우회 확률", "p_detour", [0.0, 0.02, 0.05, 0.1, 0.15, 0.3]),
    "detour_tier": ("갈림길 우회 확률 (같은 급 이상의 길로만)", "p_detour", [0.02, 0.05, 0.1, 0.15, 0.3]),
    "spread": ("방향 퍼짐 (°)", "heading_std_deg", [0.0, 20.0, 40.0, 60.0, 90.0, 120.0]),
    "drift": ("방향 흔들림 (°/√분)", "reorient_std_deg_min", [0.0, 2.5, 5.0, 10.0]),
    "sector": ("목표 구역 폭 (°)", "goal_sector_deg", [5.0, 10.0, 20.0, 30.0]),
    "speed": ("포장 / 비포장 평균 (km/h)", "speed", [(50, 30), (40, 25), (60, 35)]),
}
_NET = {}


def _init():
    from rl_search.rl_real import preload
    _NET[REGION] = preload(REGION)["net"]


def circling_share(tr, st, dt):
    from rl_search.vehicle import FAST, SLOW
    k = int(round(180/dt))
    T, N = st.shape
    hit = np.zeros(N, bool)
    step = np.hypot(*np.diff(tr, axis=0).transpose(2, 0, 1))          # (T-1, N)
    cum = np.vstack([np.zeros((1, N)), np.cumsum(step, axis=0)])
    onroad = np.isin(st, (FAST, SLOW))
    for t in range(0, T - k, 3):
        m = onroad[t] & onroad[t + k] & (cum[t + k] - cum[t] > 1000) & (np.hypot(*(tr[t + k] - tr[t]).T) < 150)
        hit |= m
    return float(hit.mean())


def job(a):
    warnings.filterwarnings("ignore")
    sweep, val, i = a
    from rl_search.rl_eval_set import V2_SPEEDS, diverse_lkp, vehicle_config
    from rl_search.scripts.rl_eval.branch_check import branches
    from rl_search.vehicle import NO_BACKTRACK, simulate_vehicle
    net = _NET[REGION]
    s = 9_000_000 + i
    rng = np.random.default_rng(s)
    lkp = diverse_lkp(net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    over = dict(BASE)
    key = SWEEPS[sweep][1]
    if key == "speed":
        f, sl = val
        over.update(fast_kmh=float(f), slow_kmh=float(sl), speed_cap_kmh=float(f) + 20.0)
    else:
        over[key] = float(val)
    vc = dict(V2_SPEEDS) | dict(p_leave_fast_min=0.0, p_leave_slow_min=0.0, p_return_min=1.0) | NO_BACKTRACK | dict(deadend_offroad=False) \
        | over | (dict(detour_keep_tier=True) if sweep == "detour_tier" else {})
    if key == "speed":
        vc.update(fast_kmh=float(val[0]), slow_kmh=float(val[1]), speed_cap_kmh=float(val[0]) + 20.0)
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=3600.0, exit_straight=True)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    tr, st, dt = res.track.astype(float), res.state, float(cfg.save_dt_s)
    L = np.asarray(res.lkp_road, float)
    half = float(net.terr.axis[-1])
    m = {}
    for mm in (30, 60):
        k = min(int(round(mm*60/dt)), tr.shape[0] - 1)
        P = tr[k]
        b, e = branches(P)
        m[mm] = dict(r95=float(np.percentile(np.hypot(*(P - L).T), 95)/1000), branches=b, eff=e,
                     cells=int(len(np.unique(np.floor(P/1000).astype(int), axis=0))), outside=float((np.abs(P).max(axis=1) > half).mean()))
    out = dict(sweep=sweep, val=val if not isinstance(val, tuple) else list(val), i=i, m={str(k): v for k, v in m.items()},
               circling=circling_share(tr, st, dt), lkp=L.tolist())
    if i < 2:
        out["P60"] = tr[min(360, tr.shape[0] - 1), ::2].round(0).tolist()
    return out


def run(n, workers):
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(sw, v, i) for sw, (_, _, vals) in SWEEPS.items() for v in vals for i in range(n)]
    with get_context("fork").Pool(workers) as pool:
        rows = []
        for j, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            rows.append(r)
            if j % 50 == 0:
                print(f"{j}/{len(jobs)}", flush=True)
    (WORK/"rows.json").write_text(json.dumps(rows))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    rows = json.loads((WORK/"rows.json").read_text())
    metrics = (("r95", "60분 95% 반경 (km)", 1), ("eff", "60분 유효 갈래 수", 1), ("cells", "60분 입자가 있는 1 km 칸", 1),
               ("outside", "60분 지도 밖 (%)", 100), ("circling", "작은 고리·왕복을 한 입자 (%)", 100))
    names = list(SWEEPS)
    fig, axs = plt.subplots(len(names), len(metrics) + 1, figsize=(4.6*(len(metrics) + 1), 3.6*len(names)), facecolor=BG, squeeze=False)
    img, ext = map_image(preload(REGION)["net"], 1600)
    L = ["# 산악 지형 도로형: 변수별 영향 (충주 120 km, 같은 시나리오 20개)", ""]
    for r_, sw in enumerate(names):
        ttl, key, vals = SWEEPS[sw]
        lab = [f"{v[0]}/{v[1]}" if isinstance(v, (list, tuple)) else f"{v:g}" for v in vals]
        L += [f"## {ttl}", "", "| 값 | " + " | ".join(t for _, t, _ in metrics) + " |", "|---|" + "---|"*len(metrics)]
        table = {lb: [] for lb in lab}
        for c_, (mk, mt, sc) in enumerate(metrics):
            ax = axs[r_, c_]
            vv = []
            for v in vals:
                vkey = list(v) if isinstance(v, tuple) else v
                x = [rr for rr in rows if rr["sweep"] == sw and rr["val"] == vkey]
                vv.append([sc*(rr["circling"] if mk == "circling" else rr["m"]["60"][mk]) for rr in x])
            ax.boxplot(vv, positions=range(len(vals)), widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="#2a78d6", alpha=0.3), medianprops=dict(color=INK))
            ax.plot(range(len(vals)), [np.median(v) for v in vv], color="#2a78d6", lw=1.5, marker="o")
            ax.set_xticks(range(len(vals)), lab, fontsize=8)
            ax.set_title(mt if r_ == 0 else "", loc="left", fontsize=10, color=INK)
            if c_ == 0:
                ax.set_ylabel(ttl, fontsize=9, color=INK)
            for lb, v in zip(lab, vv):
                table[lb].append(np.median(v))
        for lb in lab:
            L.append(f"| {lb} | " + " | ".join(f"{x:.2f}" if i_ == 1 else f"{x:.1f}" for i_, x in enumerate(table[lb])) + " |")
        L.append("")
        # example: particles at 60 min, the lowest and highest value
        ax = axs[r_, -1]
        ax.imshow(img, extent=ext, origin="upper", alpha=0.8)
        for v, col in ((vals[0], "#2a78d6"), (vals[-1], "#eb6834")):
            vkey = list(v) if isinstance(v, tuple) else v
            x = [rr for rr in rows if rr["sweep"] == sw and rr["val"] == vkey and rr["i"] == 0]
            if x:
                P = np.array(x[0]["P60"])/1000
                ax.scatter(P[:, 0], P[:, 1], s=1.5, color=col, alpha=0.5, lw=0, label=f"{ttl.split(' (')[0]} {lab[vals.index(v)]}")
        allp = np.vstack([np.array(rr["P60"]) for rr in rows if rr["sweep"] == sw and rr["i"] == 0])/1000
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        cc, h = (lo + hi)/2, max((hi - lo).max()/2 + 2, 8)
        ax.set_xlim(cc[0] - h, cc[0] + h); ax.set_ylim(cc[1] - h, cc[1] + h); ax.set_aspect("equal")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left", markerscale=6)
        ax.set_title("예 1의 60분 입자 (파랑 = 가장 작은 값, 주황 = 가장 큰 값)" if r_ == 0 else "", loc="left", fontsize=10, color=INK)
        ax.tick_params(labelsize=7)
    for a_ in axs.ravel():
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED)
    fig.suptitle("산악 지형 도로형: 변수 하나씩 바꿀 때 입자 분포 (충주 120 km, 같은 시나리오 20개; 기본 = 방향 퍼짐 35°, 우회 0.075, 흔들림 2.5, 목표 구역 20°, "
                 "50/30 km/h, 막다른 길에서 돌아섬)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT/"mountain_sweep.png", dpi=62, facecolor=BG)
    (OUT/"mountain_sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.n, a.workers) if a.what == "run" else figure()


if __name__ == "__main__":
    main()
