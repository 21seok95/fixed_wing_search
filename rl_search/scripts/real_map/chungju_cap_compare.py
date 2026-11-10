"""Forced end of a task (12 min cap): now / fix A / no cap, under three road and off-road use settings of the vehicles.

Settings: lead + gimbal 8 s, each pass 0.7 x model, pass gap 5 s, planner lane width 190 m, 30 deg bank, lane time flown,
transit limit 30 min, flat ground. Results in rl_search/data/real/chungju/cap_cmp/<env>_<cap>/, figure cap_cmp/cap_compare.png

    python -m rl_search.scripts.real_map.chungju_cap_compare --jobs 5
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.chungju_step3 import run, stats

ROOT = DATA/"chungju"/"cap_cmp"
CAPS = {"now": "12분 상한 (지금)", "A": "수정 A (다듬기도 10분 확인)", "none": "상한 없음"}
ENVS = {"road": ("도로 위주", dict(p_leave_fast_min=.01, p_leave_slow_min=.01)),
        "base": ("지금 설정", {}),
        "field": ("논밭 많이", dict(p_leave_fast_min=.2, p_leave_slow_min=.2, p_return_min=.1))}
COL = {"now": "#7f8c8d", "A": "#2a4d7a", "none": "#c0392b"}
GAP, TRANSIT = 5.0, 1800.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--jobs", type=int, default=5, help="one flight peaks at about 1.5 GB; 10 filled a 16 GB Mac (2026-11-09)")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    jobs = []
    for e, (_, kw) in ENVS.items():
        for c in CAPS:
            out = ROOT/f"{e}_{c}"
            out.mkdir(parents=True, exist_ok=True)
            jobs += [(s, 2000, "flown", out, TRANSIT, GAP, c, kw) for s in args.seeds if not (out/f"run_{s}.npz").exists()]
    if jobs and not args.figures_only:
        print(len(jobs), "runs", flush=True)
        with get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
            pool.starmap(run, jobs, chunksize=1)

    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    grid = np.arange(0, 3601, 60)
    R = {}
    for e in ENVS:
        for c in CAPS:
            rows = []
            for s in args.seeds:
                o = dict(np.load(ROOT/f"{e}_{c}"/f"run_{s}.npz"))
                t = o["tasks"]
                done = np.isfinite(t[:, 2]) & np.isfinite(t[:, 3])
                rows.append(dict(curve=np.interp(grid, o["pos_t"], o["pos"], left=0), pos=float(o["pos"][-1]), off=float(o["off_share"]),
                                 plans=len(o["rects"]), lane_min=float(((t[done, 3] - t[done, 2])/60).mean()) if done.any() else np.nan,
                                 lane_max=float(((t[done, 3] - t[done, 2])/60).max()) if done.any() else np.nan, escaped=float(o["escaped"]), **stats(o)))
            R[e, c] = rows
    n = len(args.seeds)
    fig = plt.figure(figsize=(22, 13), layout="constrained")
    gs = fig.add_gridspec(2, 4)
    for j, (e, (lab, _)) in enumerate(ENVS.items()):
        ax = fig.add_subplot(gs[0, j])
        off = np.mean([r["off"] for r in R[e, "now"]])
        for c, cl in CAPS.items():
            m = np.mean([r["curve"] for r in R[e, c]], 0)
            ax.plot(grid/60, m, color=COL[c], lw=2.4, label=f"{cl}: {m[-1]:.3f}")
        ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel("평균 누적 POS"); ax.legend(fontsize=9)
        ax.set_title(f"{lab} (도로 밖 시간 {off:.0%})", fontsize=11)
    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(len(ENVS)); w = .36
    for k, c in enumerate(("A", "none")):
        d = [np.array([a["pos"] - b["pos"] for a, b in zip(R[e, c], R[e, "now"])]) for e in ENVS]
        ax.bar(x + (k - .5)*w, [v.mean() for v in d], w, yerr=[v.std(ddof=1)/np.sqrt(n) for v in d], color=COL[c], capsize=4, label=CAPS[c])
        for i, v in enumerate(d):
            ax.scatter(np.full(n, i + (k - .5)*w), v, s=12, color="black", alpha=.5, zorder=3)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([v[0] for v in ENVS.values()])
    ax.set_ylabel("최종 POS 차이 (지금 대비)"); ax.legend(fontsize=9); ax.set_title(f"시나리오별 짝지은 차이 (점 = 시나리오, 막대 = 평균 ± 표준오차, {n}개)", fontsize=11)
    metrics = [("capped", "12분 상한에 걸린 작업 비율", (0, 1)), ("lane_min", "끝난 작업의 평균 레인 시간 (분)", None),
               ("plans", "비행당 계획 횟수", None), ("fp_in_rect", "레인 작업 중 발자국 중심이 사각형 안", (0, 1))]
    w = .26
    for j, (key, t, yl) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, j])
        for k, (c, cl) in enumerate(CAPS.items()):
            v = [np.nanmean([r[key] for r in R[e, c]]) for e in ENVS]
            ax.bar(x + (k - 1)*w, v, w, color=COL[c], label=cl)
            for xx, y in zip(x + (k - 1)*w, v):
                ax.text(xx, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([v[0] for v in ENVS.values()]); ax.set_title(t, fontsize=11); ax.legend(fontsize=8)
        if yl:
            ax.set_ylim(*yl)
    fig.suptitle(f"작업 강제 종료 비교: 표적의 도로·논밭 이용률별 (앞섬 + 짐벌 8초, 통과마다 0.7 × 모델, 통과 간격 5초, 폭 190 m, 롤 30°, 이동 제한 30분, 시나리오 {n}개)", fontsize=13)
    fig.savefig(ROOT/"cap_compare.png", dpi=95)
    summary = {e: {c: {k: float(np.nanmean([r[k] for r in R[e, c]])) for k in ("pos", "off", "plans", "lane_min", "lane_max", "capped", "turn_in_lane", "fp_in_rect", "lane", "escaped")}
                   for c in CAPS} for e in ENVS}
    for e in ENVS:
        for c in ("A", "none"):
            d = np.array([a["pos"] - b["pos"] for a, b in zip(R[e, c], R[e, "now"])])
            summary[e][c]["diff"], summary[e][c]["se"], summary[e][c]["wins"] = float(d.mean()), float(d.std(ddof=1)/np.sqrt(n)), int((d > 0).sum())
            summary[e][c]["per_seed"] = np.round(d, 3).tolist()
    (ROOT/"cap_compare.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=0, ensure_ascii=False))
    print("Saved", ROOT/"cap_compare.png")


if __name__ == "__main__":
    main()
