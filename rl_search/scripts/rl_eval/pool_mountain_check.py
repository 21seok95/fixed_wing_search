"""The mountain-terrain road-type training pool (user 2026-11-18: 4,000 scenarios on Chungju 120 km, each used twice in 6M steps): what the
scenarios hold. A random sample of the pool files: LKPs over the map, the drawn settings (heading spread, detour, drift, speed spread),
branches and R95 at 30 / 60 min, share outside the map at 60 / 70 min; and 1,000 launch points drawn as training draws them.

    python -m rl_search.scripts.rl_eval.pool_mountain_check --sample 400
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REGION = "chungju120"
POOL = Path("rl_search/data/real/chungju120/rl_pool_v2")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")


def one(f):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.rl_eval.branch_check import branches
    d = pickle.load(open(f, "rb"))
    sc, vc = d["scen"], d.get("vc", {})
    tr = sc.track
    L = np.asarray(sc.lkp, float)
    half = 60000.0 - 12.5
    out = dict(file=f.name, lkp=L.tolist(), heading=float(sc.heading_deg), vc={k: vc.get(k) for k in ("heading_std_deg", "p_detour", "reorient_std_deg_min",
                                                                                                     "fast_std_kmh", "goal_sector_deg")})
    for m in (30, 60, 70):
        k = min(int(round(m*60/sc.save_dt_s)), tr.shape[0] - 1)
        P = tr[k].astype(float)
        b, e = branches(P)
        out[str(m)] = dict(branches=b, eff=e, r95=float(np.percentile(np.hypot(*(P - L).T), 95)/1000),
                           outside=float((np.abs(P).max(axis=1) > half).mean()))
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(POOL.glob("road_s*.pkl"))
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), min(a.sample, len(files)), replace=False)]
    with get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(one, pick)
    from rl_search.rl_eval_set import V2, _launch_bad, diverse_lkp, start_stage3
    from rl_search.rl_real import preload
    st = preload(REGION)
    t, net = st["terr"], st["net"]
    half = float(t.axis[-1])
    lr = np.random.default_rng(1)
    launches = []
    for k in range(1000):
        L = np.array(rows[k % len(rows)]["lkp"])
        _, s, info = start_stage3(L, lr, half, V2["ring_m"], terr=t)
        launches.append((s[0], s[1], info["launch_dist_m"]))
    launches = np.array(launches)
    figure(rows, launches, len(files), st)
    (OUT/"pool_mountain_check.json").write_text(json.dumps(dict(n_files=len(files), rows=rows)))


def figure(rows, launches, n_files, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    img, ext = map_image(st["net"], 1600)
    fig = plt.figure(figsize=(26, 14), facecolor=BG)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.5, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, extent=ext, origin="upper", alpha=0.85)
    L = np.array([r["lkp"] for r in rows])/1000
    ax.scatter(launches[:, 0]/1000, launches[:, 1]/1000, s=5, color="#2a78d6", alpha=0.5, lw=0, label="이륙점 1,000번 (학습처럼 뽑음)")
    ax.scatter(L[:, 0], L[:, 1], s=14, color="#d6336c", ec="white", lw=0.4, label=f"LKP ({len(rows)}개 표본)")
    ax.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], color=INK, lw=1, ls="--", label="LKP 범위 (중앙 30 km)")
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.set_title("충주 120 km: LKP와 이륙점 (이륙은 산·물·도시 밖, LKP에서 10~40 km)", loc="left", fontsize=11, color=INK)
    hists = [("heading_std_deg", "방향 퍼짐 (°)", np.arange(10, 91, 5)), ("p_detour", "갈림길 우회 확률", np.linspace(0, 0.05, 11)),
             ("reorient_std_deg_min", "방향 흔들림 (°/√분)", np.linspace(0, 3, 13)), ("fast_std_kmh", "포장 속도 퍼짐 (km/h)", np.linspace(5, 10, 11))]
    for i, (k, ttl, bins) in enumerate(hists):
        ax = fig.add_subplot(gs[0, 1 + i])
        ax.hist([r["vc"][k] for r in rows if r["vc"].get(k) is not None], bins=bins, color="#2a78d6", alpha=0.75)
        ax.set_title(f"시나리오 설정: {ttl}", loc="left", fontsize=10.5, color=INK)
    ax = fig.add_subplot(gs[1, 1])
    for m, col in (("30", "#2a78d6"), ("60", "#eb6834")):
        ax.hist([r[m]["eff"] for r in rows], bins=np.arange(0, 80, 4), color=col, alpha=0.55, label=f"{m}분")
    ax.set_title("유효 갈래 수 (500 m 조각, 몫의 엔트로피)", loc="left", fontsize=10.5, color=INK); ax.legend(fontsize=9, frameon=False)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter([r["vc"]["heading_std_deg"] for r in rows], [r["60"]["eff"] for r in rows], s=10, color="#2a78d6", alpha=0.6)
    ax.set_xlabel("방향 퍼짐 (°)", fontsize=9, color=MUTED); ax.set_ylabel("60분 유효 갈래", fontsize=9, color=MUTED)
    ax.set_title("방향 퍼짐과 갈래 수", loc="left", fontsize=10.5, color=INK)
    ax = fig.add_subplot(gs[1, 3])
    for m, col in (("30", "#2a78d6"), ("60", "#eb6834")):
        ax.hist([r[m]["r95"] for r in rows], bins=np.arange(0, 70, 2.5), color=col, alpha=0.55, label=f"{m}분")
    ax.set_title("LKP에서 95% 반경 (km)", loc="left", fontsize=10.5, color=INK); ax.legend(fontsize=9, frameon=False)
    ax = fig.add_subplot(gs[1, 4])
    ax.hist(launches[:, 2]/1000, bins=np.arange(8, 42, 2), color="#2a78d6", alpha=0.75, label="이륙 거리")
    o60 = np.array([r["60"]["outside"] for r in rows]); o70 = np.array([r["70"]["outside"] for r in rows])
    ax.set_title(f"이륙 거리 (km); 지도 밖 입자 60분 평균 {100*o60.mean():.2f}%, 70분 {100*o70.mean():.2f}%", loc="left", fontsize=10, color=INK)
    for a_ in fig.axes:
        a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8); a_.grid(color=GRID, lw=.5)
        for s in a_.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"산악 지형 도로형 학습 풀 (충주 120 km, 파일 {n_files}개 중 {len(rows)}개 표본)", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"pool_mountain_check.png", dpi=72, facecolor=BG)
    e30, e60 = np.array([r["30"]["eff"] for r in rows]), np.array([r["60"]["eff"] for r in rows])
    print(f"files {n_files}; effective branches 30 min median {np.median(e30):.1f}, 60 min {np.median(e60):.1f} (10-90%: {np.percentile(e60, 10):.1f}-"
          f"{np.percentile(e60, 90):.1f}); R95 60 min median {np.median([r['60']['r95'] for r in rows]):.1f} km; outside 60/70 min {100*o60.mean():.2f}/{100*o70.mean():.2f}%")


if __name__ == "__main__":
    main()
