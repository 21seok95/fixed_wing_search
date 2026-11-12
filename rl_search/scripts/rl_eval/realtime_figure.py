"""The 60 re-scored test flights with three A's: the 5 min window A (x 70), the real-time window A (region every 10 s, x 70) and the
seen-particle A (x 80). Scores, the RL lead, the share gained while turning, how each A moves with POS, and one map with the 5 min
window region next to the real-time region at the same moment.

    python -m rl_search.scripts.rl_eval.realtime_figure

Writes rl_search/data/real/amboy/rl_eval/rescore/{realtime.png, realtime_example.png, realtime.md}.
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from rl_search.scripts.rl_eval.rescore_area import OUT, X_OLD
from rl_search.scripts.rl_eval.rescore_figure import BG, COL, GRID, INK, MUTED, NAME, load, mse, step_gains, style

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
DEFS = [("A_old", "5분 창 A"), ("A_rt", "실시간 A"), ("A_new", "본 particle A")]


def step_rt(m, k, by):
    """Per 5 s step after the hand-over: (dPOS, dA_rt)."""
    z = np.load(OUT/f"{m}_k{k}.npz")
    r = np.load(OUT/f"{m}_k{k}_rt.npz")
    t_ho = by[(m, k)]["t_handover"]
    pt, pp = z["pos_t"], z["pos"]
    dpos = np.diff(np.concatenate([[0.0], pp]))
    keep = pt > t_ho + 1e-6
    edges = np.concatenate([[pt[keep][0] - 5.0], pt[keep]])
    return dpos[keep], np.histogram(r["t"], bins=edges, weights=r["gain"])[0]


def fig_main(by, ks):
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.6), layout="constrained", facecolor=BG, gridspec_kw=dict(width_ratios=[1.3, 1, 1.15, 1.15]))
    xs = np.arange(len(DEFS))
    ax = axs[0]
    style(ax)
    for i, m in enumerate(("rl", "sarops")):
        mu = [mse([by[(m, k)][key] for k in ks]) for key, _ in DEFS]
        ax.bar(xs + (i - .5)*.36, [u for u, _ in mu], .34, yerr=[e for _, e in mu], color=COL[m], label=NAME[m], capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs + (i - .5)*.36, mu):
            ax.text(x, u + .02, f"{u:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs, [l for _, l in DEFS], color=INK)
    ax.set_ylabel("시험 30개 평균 (± 표준오차)", color=MUTED)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("A (같은 비행, hold-out particle)", loc="left", color=INK, fontsize=12)
    ax = axs[1]
    style(ax)
    out = {}
    for j, (key, lab) in enumerate(DEFS):
        tot = key.replace("A_old", "total_old").replace("A_rt", "total_rt").replace("A_new", "total_new")
        d = np.array([by[("rl", k)][tot] - by[("sarops", k)][tot] for k in ks])
        u, e = mse(d)
        out[key] = (u, e, int((d > 0).sum()))
        ax.bar(j, u, .6, yerr=e, color="#2a6db0", capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        ax.text(j, u + .012, f"{u:+.3f}\n{int((d > 0).sum())}/{len(ks)}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, [l for _, l in DEFS], color=INK)
    ax.set_ylabel("RL − SAROPS 총점 (A + POS)", color=MUTED)
    ax.set_ylim(0, max(v[0] for v in out.values()) + .1)
    ax.set_title("RL 우위 (숫자 아래 = RL이 높은 사례)", loc="left", color=INK, fontsize=12)
    ax = axs[2]
    style(ax)
    gk = {"A_old": "A_old", "A_rt": "A_rt", "A_new": "A_new"}
    for i, m in enumerate(("rl", "sarops")):
        mu = [mse([by[(m, k)]["gained_turning"][gk[key]]*100 for k in ks]) for key, _ in DEFS]
        ax.bar(xs + (i - .5)*.36, [u for u, _ in mu], .34, yerr=[e for _, e in mu], color=COL[m], label=NAME[m], capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs + (i - .5)*.36, mu):
            ax.text(x, u + 1, f"{u:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, [l for _, l in DEFS], color=INK)
    ax.set_ylabel("선회 중(|ω| > 5°/s)에 얻은 몫 (%)", color=MUTED)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("A를 선회 중에 얼마나 얻었나", loc="left", color=INK, fontsize=12)
    ax = axs[3]
    style(ax)
    for i, m in enumerate(("rl", "sarops")):
        case_r = [float(np.corrcoef([by[(m, k)]["pos"] for k in ks], [by[(m, k)][key] for k in ks])[0, 1]) for key, _ in DEFS]
        steps = {"A_old": [], "A_rt": [], "A_new": []}
        for k in ks:
            dp, ao, an = step_gains(m, k, by)
            _, ar = step_rt(m, k, by)
            for key, v in (("A_old", ao), ("A_rt", ar), ("A_new", an)):
                if dp.std() > 0 and v.std() > 0:
                    steps[key].append(np.corrcoef(dp, v)[0, 1])
        step_r = [float(np.mean(steps[key])) for key, _ in DEFS]
        ax.plot(xs + (i - .5)*.12, case_r, "o-", color=COL[m], label=f"{NAME[m]}: 사례별", lw=1.6)
        ax.plot(xs + (i - .5)*.12, step_r, "s--", color=COL[m], label=f"{NAME[m]}: 5초 스텝별", lw=1.2, mfc="white")
        for x, v in zip(xs + (i - .5)*.12, case_r):
            ax.text(x + .05, v, f"{v:.2f}", fontsize=8.5, color=INK, va="center")
        out[m] = dict(case=case_r, step=step_r)
    ax.set_xticks(xs, [l for _, l in DEFS], color=INK)
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel("POS와의 상관", color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("POS와 얼마나 겹치나 (1 = 같은 것을 잼)", loc="left", color=INK, fontsize=12)
    fig.suptitle("세 가지 A로 같은 시험 비행 60개를 채점: 실시간 A = 5분 창 A의 영역을 10초마다 그 순간의 분포로 (σ 150 m, x 70%)", fontsize=13.5,
                 color=INK, x=.01, ha="left")
    fig.savefig(OUT/"realtime.png", dpi=105, facecolor=BG)
    plt.close(fig)
    return out


def fig_example(by, k: int = 9, t0: float = 2360.0, t1: float = 2510.0):
    from rl_search.area_score import AreaScore, RealtimeArea
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import preload
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, "test", 3, k)
    terr = st["terr"]
    tm = (t0 + t1)/2
    a = AreaScore(terr, 150.0, hold.track, hold.save_dt_s, X_OLD, 0.0)
    ra = RealtimeArea(terr, 150.0, hold.track, hold.exit_step, hold.save_dt_s, X_OLD)
    log = np.load(OUT/f"rl_k{k}.npz")["log"].astype(float)
    seg = (log[:, 0] >= t0 - 20) & (log[:, 0] <= t1 + 20)
    c = log[seg, 1:3].mean(0)
    half = max(np.ptp(log[seg, 1:3], axis=0).max()/2 + 500, 1500)
    g = np.load(f"rl_search/data/real/{REGION}/road_graph_25m.npz")
    pp, oo = g["edge_poly"].astype(float), g["edge_poly_offset"]
    roads = [pp[oo[i]:oo[i + 1]] for i in range(len(oo) - 1)]
    roads = [r for r in roads if (np.abs(r - c) < half*1.5).all(1).any()]
    jf = ra.frame_now(tm)
    act = (hold.exit_step < 0) | (hold.exit_step > jf)
    P = hold.track[jf][act].astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.8), layout="constrained", facecolor=BG)
    j = a.window(tm)
    bits = ((a.B >> (j - 1)) & 1).astype(bool)
    keys = ra.H[jf - 1]
    for ax, title in zip(axs, (f"5분 창 A: 창 {j}({(j - 1)*5}~{j*5}분)의 영역 = {j*5}분의 분포", f"실시간 A: {tm/60:.1f}분의 분포로 만든 영역 (10초마다 갱신)")):
        ax.set_facecolor("#f4f3ef")
        ax.add_collection(LineCollection(roads, colors="#c9c8c1", linewidths=1.2, zorder=1))
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    cx0 = np.searchsorted(a.gx, c[0] - half - 50); cx1 = np.searchsorted(a.gx, c[0] + half + 50)
    cy0 = np.searchsorted(a.gy, c[1] - half - 50); cy1 = np.searchsorted(a.gy, c[1] + half + 50)
    sub = bits[cy0:cy1, cx0:cx1]
    img = np.zeros(sub.shape + (4,))
    img[sub] = (0, 158/255, 115/255, .35)
    if sub.size:
        axs[0].imshow(img, origin="lower", interpolation="nearest", zorder=2,
                      extent=[a.gx[cx0] - 12.5, a.gx[cx1 - 1] + 12.5, a.gy[cy0] - 12.5, a.gy[cy1 - 1] + 12.5])
    Q = ra.centres(keys)
    near = (np.abs(Q[:, 0] - c[0]) < half + 50) & (np.abs(Q[:, 1] - c[1]) < half + 50)
    if near.any():
        lo = Q[near].min(0) - 12.5
        ij = np.floor((Q[near] - lo)/25.0).astype(int)
        img2 = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1, 4))
        img2[ij[:, 1], ij[:, 0]] = (0, 158/255, 115/255, .35)
        axs[1].imshow(img2, origin="lower", interpolation="nearest", zorder=2,
                      extent=[lo[0], lo[0] + 25.0*img2.shape[1], lo[1], lo[1] + 25.0*img2.shape[0]])
    w = (log[:, 0] >= t0) & (log[:, 0] <= t1)
    for ax in axs:
        ax.scatter(P[:, 0], P[:, 1], s=3, color="#7a5a9e", alpha=.6, lw=0, zorder=3)
        ax.plot(log[w, 1], log[w, 2], color="#0b2f5e", lw=2.2, zorder=4)
    fig.suptitle(f"같은 순간의 두 영역 (시험 시나리오 {k + 1}, {tm/60:.1f}분, 진한 선 = RL의 {t0/60:.1f}~{t1/60:.1f}분 경로, 보라 = 그 순간 particle, 한 변 {2*half/1000:.1f} km)",
                 fontsize=12.5, color=INK, x=.01, ha="left")
    fig.savefig(OUT/"realtime_example.png", dpi=105, facecolor=BG)
    plt.close(fig)


def table(by, ks, out):
    lines = ["| 방법 | POS | 5분 창 A | 실시간 A | 본 particle A | 총점 (5분 창) | 총점 (실시간) | 선회 중 얻은 5분 창 A | 선회 중 얻은 실시간 A |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in ("rl", "sarops"):
        f = lambda key: mse([by[(m, k)][key] for k in ks])
        cells = " | ".join(f"{u:.3f} ± {e:.3f}" for u, e in (f("pos"), f("A_old"), f("A_rt"), f("A_new"), f("total_old"), f("total_rt")))
        g = lambda key: np.mean([by[(m, k)]["gained_turning"][key] for k in ks])
        lines.append(f"| {NAME[m]} | {cells} | {g('A_old'):.0%} | {g('A_rt'):.0%} |")
    lines += ["", "RL − SAROPS 총점: " + ", ".join(f"{lab} {out[key][0]:+.3f} ± {out[key][1]:.3f} ({out[key][2]}/{len(ks)})" for key, lab in DEFS)]
    lines += ["POS와의 상관 (사례별 / 5초 스텝별): " + "; ".join(
        f"{NAME[m]} " + ", ".join(f"{lab} {out[m]['case'][i]:.2f}/{out[m]['step'][i]:.2f}" for i, (_, lab) in enumerate(DEFS)) for m in ("rl", "sarops"))]
    (OUT/"realtime.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    by, ks = load()
    ks = [k for k in ks if "A_rt" in by[("rl", k)] and "A_rt" in by[("sarops", k)]]
    out = fig_main(by, ks)
    fig_example(by)
    print(table(by, ks, out))


if __name__ == "__main__":
    main()
