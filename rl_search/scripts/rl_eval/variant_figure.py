"""Comparison of the fine-tuning tests of 2026-11-11 (straight-flight bonus on the window A) on the 30 test cases, from the rows of
rescore_area.py: scores, the paired difference to SAROPS, turning and weaving, and the paths of one case.

    python -m rl_search.scripts.rl_eval.variant_figure --labels rl T0 T1 T2 --case 9

Writes rl_search/data/real/amboy/rl_eval/rescore/variants.png, variants_paths.png and variants.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rl_search.scripts.rl_eval.rescore_area import OUT, TURN_DPS
from rl_search.scripts.rl_eval.rescore_figure import BG, GRID, INK, MUTED, mse, style

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
NAMES = {"sarops": "SAROPS", "rl": "amboy_v1\n(620만 스텝)", "T0": "T0: +100만\n보상 없음", "T1": "T1: +100만\n직진 0.05",
         "T2": "T2: +100만\n직진 0.1"}
COLORS = {"sarops": "#c9772e", "rl": "#2a6db0", "T0": "#7a9cc6", "T1": "#2f9e8f", "T2": "#6a51a3"}
G = 9.80665


def load(labels):
    rows = json.loads((OUT.parent/"rescore_area_test.json").read_text())
    by = {(r["method"], r["k"]): r for r in rows}
    ks = sorted(k for k in range(30) if all((m, k) in by for m in labels))
    return by, ks


def action_stats(m, ks, by):
    a = np.concatenate([np.abs(np.load(OUT/f"{m}_k{k}.npz")["actions"]) for k in ks])
    return float(a.mean()), float((a < .05).mean())


def fig_variants(labels, by, ks):
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.5), layout="constrained", facecolor=BG)
    xs = np.arange(len(labels))
    cols = [COLORS.get(m, "#888") for m in labels]
    names = [NAMES.get(m, m) for m in labels]
    panels = [("total_old", "총점: 이전 A + POS", None), ("pos", "POS", None), ("A_old", "이전 A (x 70)", None), ("A_new", "참고: 새 A (x 80)", None)]
    for ax, (key, title, _) in zip(axs[0], panels):
        style(ax)
        mu = [mse([by[(m, k)][key] for k in ks]) for m in labels]
        ax.bar(xs, [u for u, _ in mu], .62, yerr=[e for _, e in mu], color=cols, capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs, mu):
            ax.text(x, u + .01, f"{u:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        lo = min(u for u, _ in mu)
        ax.set_ylim(max(0, lo - .25), max(u for u, _ in mu) + .08)
        ax.set_xticks(xs, names, fontsize=8.5, color=INK)
        ax.set_title(title + f" (시험 {len(ks)}개 평균 ± 표준오차)", loc="left", color=INK, fontsize=11)
    turn = [("share", "선회 중 시간 (|ω| > 5°/s)", lambda m: [by[(m, k)]["turn"]["share"]*100 for k in ks], "%"),
            ("rev", "분당 좌우 전환", lambda m: [by[(m, k)]["turn"]["reversals_per_min"] for k in ks], ""),
            ("abs", "평균 |행동| (1 = 5초에 49°)", None, ""),
            ("gain", "이전 A 중 선회 때 얻은 몫", lambda m: [by[(m, k)]["gained_turning"]["A_old"]*100 for k in ks], "%")]
    for ax, (key, title, f, unit) in zip(axs[1], turn):
        style(ax)
        if key == "abs":
            vals = [(action_stats(m, ks, by)[0], 0.0) for m in labels]
        else:
            vals = [mse(f(m)) for m in labels]
        ax.bar(xs, [u for u, _ in vals], .62, yerr=[e for _, e in vals], color=cols, capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs, vals):
            ax.text(x, u*1.02 + 1e-3, f"{u:.2f}" if key == "abs" else (f"{u:.1f}" if key == "rev" else f"{u:.0f}{unit}"), ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(xs, names, fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.suptitle("직진 보상 시험: 같은 시험 30개, 평가는 이전 A + POS (직진 보상은 학습에만)", fontsize=14, color=INK, x=.01, ha="left")
    fig.savefig(OUT/"variants.png", dpi=105, facecolor=BG)
    plt.close(fig)


def fig_paths(labels, by, k):
    from rl_search.rl_eval_set import REGION
    g = np.load(f"rl_search/data/real/{REGION}/road_graph_25m.npz")
    pp, oo = g["edge_poly"].astype(float), g["edge_poly_offset"]
    logs = {m: np.load(OUT/f"{m}_k{k}.npz")["log"].astype(float) for m in labels}
    allxy = np.vstack([l[l[:, 0] > by[(m, k)]["t_handover"], 1:3] for m, l in logs.items()])
    lo, hi = np.percentile(allxy, 1, axis=0) - 600, np.percentile(allxy, 99, axis=0) + 600
    c, half = (lo + hi)/2, (hi - lo).max()/2
    roads = [pp[oo[i]:oo[i + 1]] for i in range(len(oo) - 1)]
    roads = [r for r in roads if (np.abs(r - c) < half*1.3).all(1).any()]
    n = len(labels)
    zoom = None
    if "rl" in logs and k == 9:          # the weaving stretch of the first re-scoring figure (example.png): 39.3-41.8 min of amboy_v1
        l = logs["rl"]
        w = (l[:, 0] >= 2340.0) & (l[:, 0] <= 2530.0)
        zoom = (l[w, 1:3].mean(0), np.ptp(l[w, 1:3], axis=0).max()/2 + 250)
    rows = 2 if zoom else 1
    fig, axs = plt.subplots(rows, n, figsize=(4.2*n, 5.0*rows), layout="constrained", facecolor=BG, squeeze=False)
    for i, m in enumerate(labels):
        log = logs[m]
        log = log[log[:, 0] > by[(m, k)]["t_handover"]]
        om = np.abs(np.degrees(G*np.tan(log[:, 4])/33.0))
        segs = np.stack([log[:-1, 1:3], log[1:, 1:3]], 1)
        turning = om[1:] > TURN_DPS
        for row in range(rows):
            ax = axs[row, i]
            cc, hh = (c, half) if row == 0 else zoom
            ax.set_facecolor("#f4f3ef")
            ax.add_collection(LineCollection(roads, colors="#c9c8c1", linewidths=.8 if row == 0 else 1.6, zorder=1))
            ax.add_collection(LineCollection(segs[~turning], colors="#1f4e8c", linewidths=1.1 if row == 0 else 1.8, zorder=2))
            ax.add_collection(LineCollection(segs[turning], colors="#d1495b", linewidths=1.1 if row == 0 else 1.8, zorder=3))
            ax.set_xlim(cc[0] - hh, cc[0] + hh); ax.set_ylim(cc[1] - hh, cc[1] + hh); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
        r = by[(m, k)]
        axs[0, i].set_title(f"{NAMES.get(m, m).replace(chr(10), ' ')}\n총점 {r['total_old']:.3f} · 선회 {r['turn']['share']:.0%} · 전환 {r['turn']['reversals_per_min']:.1f}/분",
                            loc="left", fontsize=9.5, color=INK)
        if zoom:
            axs[1, i].set_title(f"확대: 같은 직선 구간 (한 변 {2*zoom[1]/1000:.1f} km), 이 구간을 지난 모든 비행", loc="left", fontsize=9, color=MUTED)
    fig.suptitle(f"시험 시나리오 {k + 1}의 경로 (인계 뒤): 파랑 = 직진에 가까움, 빨강 = 선회 중 (|ω| > {TURN_DPS:.0f}°/s) · 위 한 변 {2*half/1000:.1f} km",
                 fontsize=12.5, color=INK, x=.01, ha="left")
    fig.savefig(OUT/"variants_paths.png", dpi=105, facecolor=BG)
    plt.close(fig)


def table(labels, by, ks):
    lines = ["| 방법 | 총점 (이전 A + POS) | POS | 이전 A | 새 A (참고) | − SAROPS (총점) | SAROPS보다 높은 사례 | 선회 중 시간 | 분당 좌우 전환 | 평균 행동 크기 | 선회 때 얻은 이전 A |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in labels:
        f = lambda key: mse([by[(m, k)][key] for k in ks])
        d = np.array([by[(m, k)]["total_old"] - by[("sarops", k)]["total_old"] for k in ks]) if m != "sarops" else None
        dd = f"{d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}" if d is not None else "–"
        win = f"{int((d > 0).sum())}/{len(ks)}" if d is not None else "–"
        tu = np.mean([by[(m, k)]["turn"]["share"] for k in ks])
        rv = np.mean([by[(m, k)]["turn"]["reversals_per_min"] for k in ks])
        ga = np.mean([by[(m, k)]["gained_turning"]["A_old"] for k in ks])
        aa = action_stats(m, ks, by)[0]
        cells = " | ".join(f"{u:.3f} ± {e:.3f}" for u, e in (f("total_old"), f("pos"), f("A_old"), f("A_new")))
        lines.append(f"| {NAMES.get(m, m).replace(chr(10), ' ')} | {cells} | {dd} | {win} | {tu:.0%} | {rv:.1f} | {aa:.2f} | {ga:.0%} |")
    (OUT/"variants.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["rl", "T0", "T1", "T2"])
    ap.add_argument("--case", type=int, default=9)
    args = ap.parse_args()
    labels = ["sarops"] + [m for m in args.labels if m != "sarops"]
    by, ks = load(labels)
    labels = [m for m in labels if any((m, k) in by for k in ks)]
    fig_variants(labels, by, ks)
    fig_paths(labels, by, args.case)
    print(table(labels, by, ks))


if __name__ == "__main__":
    main()
