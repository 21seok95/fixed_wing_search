"""Particles that leave the map: straight on vs the mean velocity of the last minutes and a drifting direction (user 2026-11-18,
docs/scenario_redesign.md; vehicle.VehicleConfig.exit_trend_min).

The same scenarios (V2 rules: new speeds, the agreed particle rules, 70 min) are made twice, ``exit_trend_min`` 0 (straight on at the
speed and direction a particle has when it crosses the edge) and 5 (the mean velocity of its last 5 min inside the map, then the
direction drifts by the scenario's escape-heading random walk, kept within 90 deg of the first direction). The drift draws from its own
random stream, so the particles inside the map are the same in both; only what happens outside differs.

Groups: Barstow 60 km road / mixed (desert terrain; many particles leave in 70 min), Chungju 60 km road (mountain terrain, winding roads).

    python -m rl_search.scripts.rl_eval.exit_trend_check --n 8 --workers 6
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
WORK = OUT/"exit_trend_check"
GROUPS = (("mojave", "desert:road"), ("mojave", "desert:mixed"), ("chungju", "mountain:road"))
NAMES = {"desert:road": "도로형", "desert:mixed": "혼합형", "mountain:road": "도로형"}
REGION_NAMES = {"mojave": "바스토 60 km", "chungju": "충주 60 km"}
_NET = {}


def _init():
    from rl_search.vehicle import load_net
    for r in {g[0] for g in GROUPS}:
        if r not in _NET:
            _NET[r] = load_net(r, 25.0)


def job(a):
    warnings.filterwarnings("ignore")
    region, kind, i, trend = a
    from rl_search.rl_eval_set import diverse_lkp, scenario_vc, vehicle_config
    from rl_search.vehicle import simulate_vehicle
    net = _NET[region]
    s = 9_100_000 + 1000*GROUPS.index((region, kind)) + i
    rng = np.random.default_rng(s)
    vc = scenario_vc(rng, kind, s)
    lkp = diverse_lkp(net, rng, 10000.0)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=4200.0, exit_straight=True, exit_trend_min=trend)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    tr = res.track.astype(float)
    half = float(net.terr.axis[-1])
    ex = res.exit_step.astype(int)
    out_ids = np.flatnonzero(ex > 0)
    k60 = int(3600/cfg.save_dt_s)
    dt = float(cfg.save_dt_s)
    rec = dict(region=region, kind=kind, i=i, trend=trend, n=int(tr.shape[1]), out60=float((ex[(ex >= 0)] <= k60).sum()/tr.shape[1]),
               out70=float(len(out_ids)/tr.shape[1]), reorient=float(cfg.reorient_std_deg_min))
    # per exited particle (left before 60 min, so 10 min outside are seen): directions and speeds
    sel = out_ids[(ex[out_ids] >= 31) & (ex[out_ids] <= k60)]
    if len(sel):
        k = ex[sel]
        p_ex = tr[k, sel]
        d_inst = tr[k, sel] - tr[k - 1, sel]                                   # last 10 s
        d_mean = tr[k, sel] - tr[k - 30, sel]                                  # last 5 min
        k_end = np.minimum(k + 60, tr.shape[0] - 1)                            # 10 min later
        d_out = tr[k_end, sel] - p_ex
        a = lambda v: np.arctan2(v[:, 1], v[:, 0])
        wrapd = lambda x: np.degrees(np.abs((x + np.pi) % (2*np.pi) - np.pi))
        rec.update(inst_vs_mean=wrapd(a(d_inst) - a(d_mean)).tolist(), out_vs_mean=wrapd(a(d_out) - a(d_mean)).tolist(),
                   speed_out=(np.hypot(*d_out.T)/((k_end - k)*dt)*3.6).tolist(), speed_inst=(np.hypot(*d_inst.T)/dt*3.6).tolist(),
                   speed_mean=(np.hypot(*d_mean.T)/(30*dt)*3.6).tolist())
    # how far outside the map at 70 min, and the spread at 60 / 70 min
    end = tr[-1]
    beyond = np.maximum(np.abs(end).max(1) - half, 0.0)
    rec["beyond_km70"] = (beyond[out_ids]/1000).tolist()
    lk = np.asarray(res.lkp_road, float)
    rec["r95_60"] = float(np.quantile(np.hypot(*(tr[k60] - lk).T), .95)/1000)
    rec["r95_70"] = float(np.quantile(np.hypot(*(end - lk).T), .95)/1000)
    rec["inside_hash"] = float(np.where(ex < 0, np.hypot(*(end - lk).T), 0).sum())
    if i == 0:                                              # example: up to 60 particles that leave, their whole tracks every 30 s
        ids = out_ids[:: max(len(out_ids)//60, 1)][:60]
        rec["ex_track"] = tr[::3, ids].astype(np.float32).tolist()
        rec["ex_step"] = (ex[ids]//3).tolist()
        rec["lkp"] = lk.tolist()
        rec["half"] = half
    return rec


def run(n, workers):
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(r, k, i, t) for r, k in GROUPS for i in range(n) for t in (0.0, 5.0)]
    with get_context("fork").Pool(workers) as pool:
        rows = []
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec)
            print(f"{rec['region']} {rec['kind']} {rec['i']} trend {rec['trend']:g}: out by 60 min {rec['out60']:.1%}, R95 70 min "
                  f"{rec['r95_70']:.1f} km", flush=True)
    (WORK/"rows.json").write_text(json.dumps(rows))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    rows = json.loads((WORK/"rows.json").read_text())
    C0, C1 = "#8a8f98", "#2a6fdb"
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6)
    # top: examples, one per group, straight vs trend
    for g, (region, kind) in enumerate(GROUPS):
        for j, tv in enumerate((0.0, 5.0)):
            ax = fig.add_subplot(gs[0, 2*g + j])
            r = next(x for x in rows if x["region"] == region and x["kind"] == kind and x["i"] == 0 and x["trend"] == tv)
            T = np.array(r["ex_track"])/1000
            h = r["half"]/1000
            ax.add_patch(plt.Rectangle((-h, -h), 2*h, 2*h, fill=False, ec="#222", lw=1))
            for q in range(T.shape[1]):
                k = r["ex_step"][q]
                ax.plot(T[:k + 1, q, 0], T[:k + 1, q, 1], color="#c9cdd2", lw=.6)
                ax.plot(T[k:, q, 0], T[k:, q, 1], color=C1 if tv else C0, lw=.9)
            ax.plot(*np.array(r["lkp"])/1000, "*", color="#d9480f", ms=10)
            ax.set_aspect("equal")
            ax.set_xlim(-h*1.5, h*1.5), ax.set_ylim(-h*1.5, h*1.5)
            ax.set_title(f"{REGION_NAMES[region]} · {NAMES[kind]}\n{'나가기 전 5분 방향 + 나갈 때 속도 + 표류' if tv else '나갈 때 방향으로 직진 (지금)'}", fontsize=9)
            ax.tick_params(labelsize=7)
    # bottom: distributions over all scenarios
    def col(key, tv, kinds=None):
        v = []
        for x in rows:
            if x["trend"] == tv and key in x and (kinds is None or (x["region"], x["kind"]) in kinds):
                v += x[key]
        return np.array(v)
    ax = fig.add_subplot(gs[1, 0:2])
    bins = np.arange(0, 181, 10)
    for (region, kind), c in zip(GROUPS, ("#2a6fdb", "#e07b39", "#3a9a5b")):
        v = col("inst_vs_mean", 0.0, [(region, kind)])
        ax.hist(v, bins, histtype="step", lw=2, color=c, density=True, label=f"{REGION_NAMES[region]} {NAMES[kind]} (중앙값 {np.median(v):.0f}°)")
    ax.set_xlabel("나가는 순간 방향 vs 나가기 전 5분 평균 방향 (°)")
    ax.set_ylabel("밀도")
    ax.set_title("직진 규칙이 쓰는 '나가는 순간 방향'은 그동안 가던 방향과 얼마나 다른가", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[1, 2:4])
    for tv, c, lab in ((0.0, C0, "직진 (지금)"), (5.0, C1, "5분 방향 + 표류")):
        v = col("out_vs_mean", tv)
        ax.hist(v, bins, histtype="step", lw=2, color=c, density=True, label=f"{lab}: 중앙값 {np.median(v):.0f}°, 90% {np.quantile(v, .9):.0f}°")
    ax.set_xlabel("나간 뒤 10분 이동 방향 vs 나가기 전 5분 평균 방향 (°)")
    ax.set_title("지도 밖 이동이 그동안의 경로와 같은 쪽인가 (세 무리 합)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[1, 4:6])
    sb = np.arange(0, 91, 5)
    for tv, c, lab in ((0.0, C0, "직진 (지금)"), (5.0, C1, "5분 방향 + 표류")):
        v = col("speed_out", tv)
        ax.hist(v, sb, histtype="step", lw=2, color=c, density=True, label=f"{lab}: 중앙값 {np.median(v):.0f} km/h")
    v = col("speed_mean", 0.0)
    ax.hist(v, sb, histtype="step", lw=1.2, ls="--", color="#222", density=True, label=f"나가기 전 5분 순 이동 속도: 중앙값 {np.median(v):.0f} km/h")
    ax.set_xlabel("지도 밖 첫 10분 순 이동 속도 (km/h)")
    ax.set_title("지도 밖에서 얼마나 빨리 멀어지나", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("지도 밖으로 나간 입자: 나갈 때 방향 직진 vs 나가기 전 5분 방향 + 나갈 때 속도 + 표류 (±90°)  ·  회색 = 지도 안 경로", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"exit_trend_check.png", dpi=120)
    # table
    L = ["# 지도 밖 입자: 직진 vs 5분 평균 + 표류", "", "| 무리 | 시나리오 | 60분까지 나간 몫 | 지도 안 입자 같음 | R95 70분 (km) 직진 / 평균 | 순간 방향과 5분 평균 차 (중앙값) | 지도 밖 10분 방향 차 (중앙값, 직진 / 평균) | 지도 밖 속도 중앙값 (km/h) 직진 / 평균 |",
         "|---|---|---|---|---|---|---|---|"]
    for region, kind in GROUPS:
        A = sorted([x for x in rows if x["region"] == region and x["kind"] == kind and x["trend"] == 0.0], key=lambda x: x["i"])
        B = sorted([x for x in rows if x["region"] == region and x["kind"] == kind and x["trend"] == 5.0], key=lambda x: x["i"])
        same = all(abs(a["inside_hash"] - b["inside_hash"]) < 1e-6 and a["out60"] == b["out60"] for a, b in zip(A, B))
        g = [(region, kind)]
        L.append(f"| {REGION_NAMES[region]} {NAMES[kind]} | {len(A)} | {np.mean([a['out60'] for a in A]):.1%} | {'예' if same else '아니오'} | "
                 f"{np.mean([a['r95_70'] for a in A]):.1f} / {np.mean([b['r95_70'] for b in B]):.1f} | {np.median(col('inst_vs_mean', 0.0, g)):.0f}° | "
                 f"{np.median(col('out_vs_mean', 0.0, g)):.0f}° / {np.median(col('out_vs_mean', 5.0, g)):.0f}° | "
                 f"{np.median(col('speed_out', 0.0, g)):.0f} / {np.median(col('speed_out', 5.0, g)):.0f} |")
    (OUT/"exit_trend_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=("all", "run", "figure"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.what in ("all", "run"):
        run(a.n, a.workers)
    if a.what in ("all", "figure"):
        figure()


if __name__ == "__main__":
    main()
