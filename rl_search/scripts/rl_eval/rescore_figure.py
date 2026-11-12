"""Figures and table of the re-scoring (rescore_area.py): the old flights with the window A and the seen-particle A.

    python -m rl_search.scripts.rl_eval.rescore_figure

Writes rl_search/data/real/amboy/rl_eval/rescore/{scores,turning,example}.png and table.md.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from rl_search.scripts.rl_eval.rescore_area import OUT, POD, TURN_DPS, X_NEW, X_OLD

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
COL = {"rl": "#2a6db0", "sarops": "#c9772e"}
NAME = {"rl": "RL (amboy_v1)", "sarops": "SAROPS"}
G = 9.80665


def load():
    rows = json.loads((OUT.parent/"rescore_area_test.json").read_text())
    by = {(r["method"], r["k"]): r for r in rows}
    ks = sorted({k for m, k in by if (("rl", k) in by and ("sarops", k) in by)})
    return by, ks


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)


def mse(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0


def fig_scores(by, ks):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), layout="constrained", facecolor=BG, gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    style(ax)
    items = [("A_old", f"이전 A\n(5분 창, x {X_OLD:.0f})"), ("A_new", f"새 A\n(본 particle, x {X_NEW:.0f})"), ("pos", "POS"),
             ("total_old", "총점\n이전 A + POS"), ("total_new", "총점\n새 A + POS")]
    xs = np.arange(len(items))
    for i, m in enumerate(("rl", "sarops")):
        mu = [mse([by[(m, k)][key] for k in ks]) for key, _ in items]
        b = ax.bar(xs + (i - .5)*.36, [u for u, _ in mu], .34, yerr=[e for _, e in mu], color=COL[m], label=NAME[m], capsize=3,
                   error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs + (i - .5)*.36, mu):
            ax.text(x, u + .03, f"{u:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs, [l for _, l in items], fontsize=9.5, color=INK)
    ax.set_ylabel("시험 30개 평균 (± 표준오차)", color=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("점수: 같은 비행을 두 A로 채점 (hold-out particle)", loc="left", color=INK, fontsize=12)
    # paired difference RL - SAROPS, old vs new
    ax = axs[1]
    style(ax)
    d_old = np.array([by[("rl", k)]["total_old"] - by[("sarops", k)]["total_old"] for k in ks])
    d_new = np.array([by[("rl", k)]["total_new"] - by[("sarops", k)]["total_new"] for k in ks])
    lim = [min(d_old.min(), d_new.min(), 0) - .05, max(d_old.max(), d_new.max()) + .05]
    ax.plot(lim, lim, color=GRID, lw=1.2, zorder=1)
    ax.axhline(0, color=MUTED, lw=.8, ls="--", zorder=1)
    ax.axvline(0, color=MUTED, lw=.8, ls="--", zorder=1)
    ax.scatter(d_old, d_new, s=34, color=COL["rl"], edgecolor="white", lw=.8, zorder=3)
    for k, a, b in zip(ks, d_old, d_new):
        ax.annotate(str(k + 1), (a, b), xytext=(3, 2), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.grid(axis="x", color=GRID, lw=.8)
    mo, eo = mse(d_old)
    mn, en = mse(d_new)
    ax.set_xlabel(f"이전 A 총점 차이 (RL − SAROPS)\n평균 {mo:+.3f} ± {eo:.3f}, RL이 높은 사례 {int((d_old > 0).sum())}/{len(ks)}", color=INK, fontsize=9.5)
    ax.set_ylabel(f"새 A 총점 차이 (RL − SAROPS)\n평균 {mn:+.3f} ± {en:.3f}, RL이 높은 사례 {int((d_new > 0).sum())}/{len(ks)}", color=INK, fontsize=9.5)
    ax.set_title("사례별 RL 우위: 이전 A → 새 A (점 번호 = 시나리오)", loc="left", color=INK, fontsize=12)
    # A old vs new per case
    ax = axs[2]
    style(ax)
    for m in ("sarops", "rl"):
        ao = [by[(m, k)]["A_old"] for k in ks]
        an = [by[(m, k)]["A_new"] for k in ks]
        ax.scatter(ao, an, s=30, color=COL[m], edgecolor="white", lw=.7, label=NAME[m], zorder=3)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_xlabel(f"이전 A (5분 창, x {X_OLD:.0f})", color=INK)
    ax.set_ylabel(f"새 A (본 particle, x {X_NEW:.0f}, POD ≥ {POD})", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_title("사례별 A: 이전 → 새", loc="left", color=INK, fontsize=12)
    fig.suptitle("같은 비행, 다른 A: 지금 RL과 SAROPS의 시험 비행 30개를 새 A로 다시 채점", fontsize=14, color=INK, x=.01, ha="left")
    out = OUT/"scores.png"
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)
    return out, dict(d_old=mse(d_old), d_new=mse(d_new), win_old=int((d_old > 0).sum()), win_new=int((d_new > 0).sum()))


def step_gains(m, k, by):
    """Per 5 s step after the hand-over: (dPOS, dA_old, dA_new) of one flight."""
    z = np.load(OUT/f"{m}_k{k}.npz")
    t_ho = by[(m, k)]["t_handover"]
    pt, pp = z["pos_t"], z["pos"]
    dpos = np.diff(np.concatenate([[0.0], pp]))
    keep = pt > t_ho + 1e-6
    edges = np.concatenate([[pt[keep][0] - 5.0], pt[keep]])
    log, g_old, g_new = z["log"].astype(float), z["g_old"].astype(float), z["g_new"]
    a_old = np.histogram(log[:, 0], bins=edges, weights=g_old)[0]
    a_new = np.histogram(g_new[:, 0], bins=edges, weights=g_new[:, 1])[0] if len(g_new) else np.zeros(len(edges) - 1)
    return dpos[keep], a_old, a_new


def fig_similarity(by, ks):
    """How much each A moves with POS: per case, and step by step within the flights."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained", facecolor=BG, gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    style(ax)
    ax.grid(axis="x", color=GRID, lw=.8)
    out = {}
    for key, mk, lab in (("A_old", "o", "이전 A"), ("A_new", "s", "새 A")):
        for m in ("rl", "sarops"):
            x = np.array([by[(m, k)]["pos"] for k in ks])
            y = np.array([by[(m, k)][key] for k in ks])
            r = float(np.corrcoef(x, y)[0, 1])
            out[(key, m, "case")] = r
            ax.scatter(x, y, s=30, marker=mk, facecolor=COL[m] if key == "A_new" else "white", edgecolor=COL[m], lw=1.2,
                       label=f"{NAME[m]} · {lab}: r = {r:.2f}", zorder=3)
    ax.set_xlabel("POS (사례별)", color=INK)
    ax.set_ylabel("A (사례별)", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("사례별: POS가 높은 비행이 A도 높은가 (r = 상관계수)", loc="left", color=INK, fontsize=12)
    ax = axs[1]
    style(ax)
    xs = np.arange(2)
    for i, m in enumerate(("rl", "sarops")):
        rs = {"A_old": [], "A_new": []}
        for k in ks:
            dp, ao, an = step_gains(m, k, by)
            for key, v in (("A_old", ao), ("A_new", an)):
                if dp.std() > 0 and v.std() > 0:
                    rs[key].append(np.corrcoef(dp, v)[0, 1])
        mu = [mse(rs[key]) for key in ("A_old", "A_new")]
        for key, (u, _) in zip(("A_old", "A_new"), mu):
            out[(key, m, "step")] = u
        ax.bar(xs + (i - .5)*.36, [u for u, _ in mu], .34, yerr=[e for _, e in mu], color=COL[m], label=NAME[m], capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs + (i - .5)*.36, mu):
            ax.text(x, u + .02, f"{u:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs, ["이전 A와 POS", "새 A와 POS"], color=INK, fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("5초 스텝별 증가량의 상관 (비행별 평균 ± 표준오차)", color=MUTED)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("스텝별: 같은 순간에 같이 오르는가 (1이면 같은 보상을 두 번 주는 셈)", loc="left", color=INK, fontsize=12)
    fig.suptitle("A와 POS가 얼마나 겹치는가: 같은 비행 60개(시험 30개 × RL, SAROPS)를 두 A로 채점", fontsize=14, color=INK, x=.01, ha="left")
    fig.savefig(OUT/"similarity.png", dpi=110, facecolor=BG)
    plt.close(fig)
    return out


def turn_rates(m, ks, by):
    out = []
    for k in ks:
        z = np.load(OUT/f"{m}_k{k}.npz")
        log = z["log"].astype(float)
        after = log[:, 0] > by[(m, k)]["t_handover"] + 1e-6
        out.append(np.degrees(G*np.tan(log[after, 4])/33.0))
    return np.concatenate(out)


def fig_turning(by, ks):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained", facecolor=BG, gridspec_kw=dict(width_ratios=[1.25, 1.2, .9]))
    ax = axs[0]
    style(ax)
    bins = np.arange(0, 20.5, .5)
    for m in ("sarops", "rl"):
        om = np.abs(turn_rates(m, ks, by))
        h, _ = np.histogram(np.clip(om, 0, 20), bins=bins)
        ax.stairs(h/len(om)*100, bins, color=COL[m], lw=2, label=f"{NAME[m]}: 선회 중 {np.mean(om > TURN_DPS):.0%}, 평균 {om.mean():.1f}°/s")
    ax.axvline(TURN_DPS, color=MUTED, lw=1, ls="--")
    ax.text(TURN_DPS + .3, ax.get_ylim()[1]*.55, "5°/s: 이보다 빠르면\n영상 흔들림으로 POD 감소", fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel("선회율 |ω| (°/s, 1초 표본, 이동 구간 제외)", color=INK)
    ax.set_ylabel("비행 시간 비율 (%)", color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("선회율 분포", loc="left", color=INK, fontsize=12)
    ax = axs[1]
    style(ax)
    items = [("A_old", "이전 A"), ("A_new", "새 A"), ("pos", "POS")]
    xs = np.arange(len(items))
    for i, m in enumerate(("rl", "sarops")):
        mu = [mse([by[(m, k)]["gained_turning"][key]*100 for k in ks]) for key, _ in items]
        ax.bar(xs + (i - .5)*.36, [u for u, _ in mu], .34, yerr=[e for _, e in mu], color=COL[m], label=NAME[m], capsize=3, error_kw=dict(ecolor=INK, lw=.9))
        for x, (u, _) in zip(xs + (i - .5)*.36, mu):
            ax.text(x, u + 1, f"{u:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, [l for _, l in items], color=INK, fontsize=10.5)
    ax.set_ylabel(f"선회 중(|ω| > {TURN_DPS:.0f}°/s)에 얻은 몫 (%)", color=MUTED)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("점수를 선회 중에 얼마나 얻었나", loc="left", color=INK, fontsize=12)
    ax = axs[2]
    style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate(("rl", "sarops")):
        v = np.array([by[(m, k)]["turn"]["reversals_per_min"] for k in ks])
        ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, s=22, color=COL[m], alpha=.75, edgecolor="white", lw=.5)
        ax.hlines(v.mean(), i - .25, i + .25, color=INK, lw=2)
        ax.text(i + .28, v.mean(), f"{v.mean():.1f}", va="center", fontsize=9.5, color=INK)
    ax.set_xticks([0, 1], [NAME["rl"], NAME["sarops"]], color=INK)
    ax.set_xlim(-.6, 1.7)
    ax.set_ylabel("분당 좌우 전환 (행동 부호가 바뀐 횟수)", color=MUTED)
    ax.set_title("좌우로 흔드는 빈도", loc="left", color=INK, fontsize=12)
    fig.suptitle("선회: 이전 A는 선회하며 화면이 쓸고 지나간 칸도 세지만, 새 A는 POD 0.5 미만으로 스친 particle을 세지 않는다", fontsize=13.5,
                 color=INK, x=.01, ha="left")
    out = OUT/"turning.png"
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def weave_window(log, t_ho, span=150.0):
    """The 150 s of an RL flight with the most back-and-forth heading change and little net turn: (t0, t1, score)."""
    t, yaw = log[:, 0], np.unwrap(log[:, 3])
    best = (None, None, -1.0)
    for t0 in np.arange(t_ho + 60.0, t[-1] - span, 10.0):
        m = (t >= t0) & (t <= t0 + span)
        if m.sum() < span*.8:
            continue
        y = yaw[m]
        net, tot = abs(y[-1] - y[0]), np.abs(np.diff(y)).sum()
        if np.degrees(net) < 30 and tot - net > best[2]:
            best = (t0, t0 + span, tot - net)
    return best


def fig_example(by, ks):
    from rl_search.area_score import AreaScore, SeenArea
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import preload
    cand = []
    for k in ks:
        z = np.load(OUT/f"rl_k{k}.npz")
        t0, t1, sc = weave_window(z["log"].astype(float), by[("rl", k)]["t_handover"])
        if t0 is not None:
            cand.append((sc, k, t0, t1))
    sc, k, t0, t1 = max(cand)
    z = np.load(OUT/f"rl_k{k}.npz")
    log = z["log"].astype(float)
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, "test", 3, k)
    tm = (t0 + t1)/2
    # old A: the window region at tm and the cells swept by then (replay of the sample log)
    terr = st["terr"]
    from rl_search.scripts.real_map.web_optimize import team_env
    from rl_search.scripts.rl_eval.rescore_area import old_settings
    e = team_env(old_settings(), REGION, terr, case.seed)
    cfg = e.detector.cfg
    full = None
    a = AreaScore(terr, e.h, hold.track, hold.save_dt_s, X_OLD, 0.0)
    # the full logs carry the camera frames; rebuild them by replaying the actions on the hold-out particles
    from rl_search.multi import run_team_launch
    e.sample_log = []
    run_team_launch([e], None, case.options(hold), 60.0, actions=[list(z["actions"].astype(float))])
    full = np.array(e.sample_log)
    full = full[np.argsort(full[:, 0], kind="stable")]
    for row in full[full[:, 0] <= tm]:
        a.mark_frame(row[1], row[2], (row[5:8][None], row[8:11][None], row[11:14][None]), row[0], cfg)
    j = a.window(tm)
    sa = SeenArea(hold.track, hold.exit_step, hold.save_dt_s, X_NEW, POD)
    jf = sa.frame_now(tm)
    P = hold.track[jf].astype(float)
    t_seen = z["t_seen"].astype(float)
    seg = (log[:, 0] >= t0 - 20) & (log[:, 0] <= t1 + 20)
    xy = log[seg, 1:3]
    c = xy.mean(0)
    half = max(np.ptp(xy, axis=0).max()/2 + 250, 900)
    g = np.load(f"rl_search/data/real/{REGION}/road_graph_25m.npz")
    pp, oo = g["edge_poly"].astype(float), g["edge_poly_offset"]
    roads = [pp[oo[i]:oo[i + 1]] for i in range(len(oo) - 1)]
    roads = [r for r in roads if (np.abs(r - c) < half*1.5).all(1).any()]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.4), layout="constrained", facecolor=BG)
    for ax in axs:
        ax.set_facecolor("#f4f3ef")
        ax.add_collection(LineCollection(roads, colors="#bdbcb5", linewidths=1.0, zorder=1))
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    # left: window region (band) and swept cells
    ax = axs[0]
    bits = ((a.B >> (j - 1)) & 1).astype(bool)
    cx0 = np.searchsorted(a.gx, c[0] - half - 50); cx1 = np.searchsorted(a.gx, c[0] + half + 50)
    cy0 = np.searchsorted(a.gy, c[1] - half - 50); cy1 = np.searchsorted(a.gy, c[1] + half + 50)
    sub, swp = bits[cy0:cy1, cx0:cx1], a.S[cy0:cy1, cx0:cx1]
    img = np.zeros(sub.shape + (4,))
    img[sub] = (0, 158/255, 115/255, .16)
    img[sub & swp] = (0, 158/255, 115/255, .62)
    if sub.size:
        ax.imshow(img, origin="lower", interpolation="nearest", zorder=2,
                  extent=[a.gx[cx0] - 12.5, a.gx[cx1 - 1] + 12.5, a.gy[cy0] - 12.5, a.gy[cy1 - 1] + 12.5])
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    # right: seen-A region cells (25 m), shaded by the seen share of their particles at tm
    ax = axs[1]
    core = sa.a[jf - 1] > 0
    seen = t_seen <= tm
    ic = np.floor(P[core]/25.0).astype(np.int64)
    keys, inv = np.unique(ic[:, 0]*100000 + ic[:, 1], return_inverse=True)
    n = np.bincount(inv.ravel())
    s_ = np.bincount(inv.ravel(), weights=seen[core].astype(float))
    cx_, cy_ = (keys//100000 + .5)*25.0, (keys - (keys//100000)*100000 + .5)*25.0
    for x_, y_, f in zip(cx_, cy_, s_/n):
        if abs(x_ - c[0]) < half + 50 and abs(y_ - c[1]) < half + 50:
            ax.add_patch(Rectangle((x_ - 12.5, y_ - 12.5), 25, 25, color="#009E73", alpha=.2 + .6*f, lw=0, zorder=2))
    for ax in axs:
        act = (hold.exit_step < 0) | (hold.exit_step > jf)
        Pn = P[act]
        ax.scatter(Pn[:, 0], Pn[:, 1], s=3, color="#7a5a9e", alpha=.55, lw=0, zorder=3)
        pts = log[seg]
        segs = np.stack([pts[:-1, 1:3], pts[1:, 1:3]], 1)
        lc = LineCollection(segs, colors=[COL["rl"]], linewidths=1.8, zorder=4)
        ax.add_collection(lc)
        w = (log[:, 0] >= t0) & (log[:, 0] <= t1)
        ax.plot(log[w, 1], log[w, 2], color="#0b2f5e", lw=2.6, zorder=5)
        i_m = np.argmin(np.abs(log[:, 0] - tm))
        ax.plot(log[i_m, 1], log[i_m, 2], marker=(3, 0, np.degrees(log[i_m, 3]) - 90), ms=12, color="#0b2f5e", zorder=6)
    rk = by[("rl", k)]
    axs[0].set_title(f"이전 A: 5분 창 영역 (σ 150 m로 흐린 x {X_OLD:.0f}% 띠, 창 {j}: {(j - 1)*5}~{j*5}분)\n연한 초록 = 영역, 진한 초록 = {tm/60:.1f}분까지 훑은 칸. "
                     f"이 비행의 이전 A {rk['A_old']:.3f}", loc="left", fontsize=11, color=INK)
    axs[1].set_title(f"새 A: 그 순간 particle 상위 {X_NEW:.0f}%가 있는 25 m 칸 ({tm/60:.1f}분)\n칸 안 particle을 본 비율만큼 진하게 (봤음 = POD ≥ {POD}). "
                     f"이 비행의 새 A {rk['A_new']:.3f}", loc="left", fontsize=11, color=INK)
    fig.suptitle(f"RL이 직선 구간에서 좌우로 흔드는 예 (시험 시나리오 {k + 1}, {t0/60:.1f}~{t1/60:.1f}분, 진한 선) · 보라 점 = 그 시각 particle · "
                 f"한 변 {2*half/1000:.1f} km", fontsize=13, color=INK, x=.01, ha="left")
    out = OUT/"example.png"
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)
    return k, t0, t1


def table(by, ks, extra):
    lines = ["| 방법 | POS | 이전 A (x 70) | 새 A (x 80) | 총점 (이전) | 총점 (새) | 선회 중 시간 | 선회 중 얻은 이전 A | 선회 중 얻은 새 A | 분당 좌우 전환 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("rl", "sarops"):
        f = lambda key: mse([by[(m, k)][key] for k in ks])
        g = lambda a, b: mse([by[(m, k)][a][b] for k in ks])
        cells = [f("pos"), f("A_old"), f("A_new"), f("total_old"), f("total_new")]
        lines.append(f"| {NAME[m]} | " + " | ".join(f"{u:.3f} ± {e:.3f}" for u, e in cells)
                     + f" | {g('turn', 'share')[0]:.0%} | {g('gained_turning', 'A_old')[0]:.0%} | {g('gained_turning', 'A_new')[0]:.0%}"
                     + f" | {g('turn', 'reversals_per_min')[0]:.1f} |")
    (uo, eo), (un, en) = extra["d_old"], extra["d_new"]
    lines += ["", f"RL − SAROPS 총점: 이전 A {uo:+.3f} ± {eo:.3f} (RL이 높은 사례 {extra['win_old']}/{len(ks)}), "
                  f"새 A {un:+.3f} ± {en:.3f} (RL이 높은 사례 {extra['win_new']}/{len(ks)}). 값은 시험 {len(ks)}개 평균 ± 표준오차."]
    (OUT/"table.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    by, ks = load()
    out, extra = fig_scores(by, ks)
    sim = fig_similarity(by, ks)
    print("similarity", {f"{a}/{m}/{w}": round(v, 3) for (a, m, w), v in sim.items()})
    fig_turning(by, ks)
    ex = fig_example(by, ks)
    print(table(by, ks, extra))
    print("example case", ex)


if __name__ == "__main__":
    main()
