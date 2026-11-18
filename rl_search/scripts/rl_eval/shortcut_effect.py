"""What the shortcuts change on Chungju 60 km (user 2026-11-18: mountain terrain, off the road only as shortcuts). The same scenarios
(seed, LKP in the central 20 km, heading, route settings, the redesigned speeds, the agreed rules NO_BACKTRACK) in four versions:
  free   - off the road as before (the agreed rules), leaving the road at 0.05 / min
  sc05   - shortcuts only (shortcut_m 2 km, factor 1.5), trying to leave the road at 0.05 / min
  sc20   - the same at 0.2 / min
  sc50   - the same at 0.5 / min
Per scenario and minute: R50 / R95 around the LKP, branches (500 m pieces with 5 % of the particles or more, branch_check.branches),
1 km cells holding a particle, share off the road, particles stuck (trap_check.stuck_stats). Figure report/shortcut_effect.png.

    python -m rl_search.scripts.rl_eval.shortcut_effect run --n 20 --workers 8
    python -m rl_search.scripts.rl_eval.shortcut_effect figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"shortcut_effect"
VAR = {"free": dict(p=0.05, sc=False), "sc05": dict(p=0.05, sc=True), "sc20": dict(p=0.2, sc=True), "sc50": dict(p=0.5, sc=True)}
VN = {"free": "지름길 없음 (도로 밖 자유, 0.05/분)", "sc05": "지름길, 0.05/분", "sc20": "지름길, 0.2/분", "sc50": "지름길, 0.5/분"}
VC = {"free": "#9a9a95", "sc05": "#2a78d6", "sc20": "#1baf7a", "sc50": "#eb6834"}
MIN = np.arange(0, 61, 5)
_NET = {}


def _init():
    from rl_search.rl_real import preload
    _NET["chungju"] = preload("chungju")["net"]


def job(a):
    warnings.filterwarnings("ignore")
    var, i = a
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, route_vc, vehicle_config, V2_SPEEDS
    from rl_search.scripts.rl_eval.branch_check import branches
    from rl_search.scripts.rl_eval.trap_check import stuck_stats
    from rl_search.vehicle import NO_BACKTRACK, simulate_vehicle
    net = _NET["chungju"]
    s = 8_500_000 + i
    rng = np.random.default_rng(s)
    p = VAR[var]
    vc = multi_vc(rng, "mixed") | V2_SPEEDS | route_vc(s) | NO_BACKTRACK | dict(p_leave_fast_min=p["p"], p_leave_slow_min=p["p"], p_return_min=0.5) \
        | (dict(shortcut_m=2000.0, shortcut_factor=1.5) if p["sc"] else {})
    lkp = diverse_lkp(net, rng, 10000.0)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=3600.0, exit_straight=True)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    tr, st, dt = res.track.astype(float), res.state, float(cfg.save_dt_s)
    L = np.asarray(res.lkp_road, float)
    rows = []
    for m in MIN:
        k = min(int(round(m*60/dt)), tr.shape[0] - 1)
        P = tr[k]
        d = np.hypot(*(P - L).T)/1000
        b, e = branches(P)
        cells = len(np.unique(np.floor(P/1000).astype(int), axis=0))
        rows.append(dict(m=int(m), r50=float(np.percentile(d, 50)), r95=float(np.percentile(d, 95)), branches=b, eff=e, cells=cells,
                         off=float((st[k] == 2).mean())))
    ss = stuck_stats(tr, st, dt)
    out = dict(var=var, i=i, rows=rows, stuck=ss["stuck"], n_shortcut=int(len(res.shortcut_len)), lkp=L.tolist())
    if i < 3:
        out["P30"] = tr[180, ::2].round(0).tolist()
        out["P60"] = tr[min(360, tr.shape[0] - 1), ::2].round(0).tolist()
        out["S60"] = st[min(360, tr.shape[0] - 1), ::2].tolist()
    return out


def run(n, workers):
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(v, i) for v in VAR for i in range(n)]
    with get_context("fork").Pool(workers) as pool:
        rows = list(pool.imap_unordered(job, jobs, chunksize=1))
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
    fig = plt.figure(figsize=(26, 19), facecolor=BG)
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1.15, 1.15])
    for col, (key, ttl, sc) in enumerate((("r95", "LKP에서 95% 반경 (km)", 1), ("r50", "LKP에서 50% 반경 (km)", 1), ("branches", "갈래 수 (입자 5% 넘는 조각)", 1),
                                          ("cells", "입자가 있는 1 km 칸 수", 1), ("off", "도로 밖 입자 (%)", 100))):
        ax = fig.add_subplot(gs[0, col])
        for v in VAR:
            a = np.array([[r[key] for r in x["rows"]] for x in rows if x["var"] == v], float)
            ax.plot(MIN, sc*np.median(a, axis=0), color=VC[v], lw=2, marker="o", ms=3, label=VN[v])
        ax.set_xlabel("LKP 뒤 분", fontsize=9, color=MUTED); ax.set_title(ttl + " (중앙값, 20개)", loc="left", fontsize=10.5, color=INK)
        if col == 0:
            ax.legend(fontsize=8, frameon=False)
    img, ext = map_image(preload("chungju")["net"], 1600)
    for row, k in ((1, 0), (2, 1)):
        for col, v in enumerate(VAR):
            ax = fig.add_subplot(gs[row, col])
            x = [r for r in rows if r["var"] == v and r["i"] == k][0]
            P, S = np.array(x["P60"])/1000, np.array(x["S60"])
            ref = [np.array(r["P60"])/1000 for r in rows if r["i"] == k]
            allp = np.vstack(ref)
            lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
            c, h = (lo + hi)/2, max((hi - lo).max()/2 + 1, 4)
            ax.imshow(img, extent=ext, origin="upper", alpha=0.8)
            ax.scatter(P[S != 2, 0], P[S != 2, 1], s=2, color="#1f3a93", lw=0, alpha=0.6, label="도로 위")
            ax.scatter(P[S == 2, 0], P[S == 2, 1], s=6, color="#eb6834", lw=0, alpha=0.9, label="도로 밖")
            ax.plot(*(np.array(x["lkp"])/1000), marker="*", ms=13, color="#d6336c", mec="white")
            ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_aspect("equal")
            b60 = x["rows"][-1]
            ax.set_title(f"예 {k + 1}, {VN[v]}\n60분: 95% 반경 {b60['r95']:.1f} km, 갈래 {b60['branches']}, 1 km 칸 {b60['cells']}, 갇힘 {100*x['stuck']:.0f}%",
                         loc="left", fontsize=9.5, color=INK)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax = fig.add_subplot(gs[row, 4]); ax.axis("off")
    txt = ["표 (60분, 20개 중앙값 / 평균)"]
    for v in VAR:
        a = [x for x in rows if x["var"] == v]
        r60 = [x["rows"][-1] for x in a]
        txt.append(f"{VN[v]}:\n  95% 반경 {np.median([r['r95'] for r in r60]):.1f} km, 갈래 {np.mean([r['branches'] for r in r60]):.2f}, "
                   f"1 km 칸 {np.median([r['cells'] for r in r60]):.0f}\n  도로 밖 {100*np.mean([r['off'] for r in r60]):.1f}%, 갇힘 {100*np.mean([x['stuck'] for x in a]):.1f}%, "
                   f"지름길 {np.mean([x['n_shortcut'] for x in a]):.0f}번/시나리오")
    fig.axes[-1].text(0, 1, "\n".join(txt), va="top", fontsize=10, color=INK, transform=fig.axes[-1].transAxes, linespacing=1.6)
    for a_ in fig.axes:
        a_.set_facecolor(BG)
        if a_.axison:
            a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("충주 60 km(산악 지형), 같은 시나리오 20개: 지름길이 없을 때와 지름길 빈도별 입자 분포 (아래 = 예시 두 개의 60분 입자, 주황 = 도로 밖)",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"shortcut_effect.png", dpi=72, facecolor=BG)
    print("\n".join(txt))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.n, a.workers) if a.what == "run" else figure()


if __name__ == "__main__":
    main()
