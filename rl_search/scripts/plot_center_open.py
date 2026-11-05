"""Figures for the area score runs (compare_center_open), from the saved results: no planner rerun.

summary_*.png: POS and swept share over time, time inside the region, and each scenario against the baseline.
map_*.png: per scenario, at 15, 30, 45 and 60 min: terrain, particles by remaining probability, the moving x% region, what
was swept, rectangles, tracks and close pairs, POS only against A + POS.

    python -m rl_search.scripts.plot_center_open --input rl_search/data/center_open/first_check
    python -m rl_search.scripts.plot_center_open --input rl_search/data/center_open/sweep --n 5 --x 85 --w 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS_C, SCORE_C, CAND_C = "#2a78d6", "#1baf7a", MUTED     # the two compared series validated; candidates only recede
SWEPT, REGION_LINE, CLOSE = "#eb6834", "#1c5cab", "#d03b3b"
LABEL = {"pos": "POS only (baseline)", "score": "A + POS", "cand": "POS only + extra starts"}
COLOR = {"pos": POS_C, "score": SCORE_C, "cand": CAND_C}


def style():
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SEC, "ytick.labelcolor": SEC,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def dumbbell(ax, seeds, a, b, xlabel, title, fmt="{:+.3f}"):
    for y, (p, q) in enumerate(zip(a, b)):
        ax.plot([p, q], [y, y], color=AXIS, lw=1.6, zorder=2)
        ax.scatter(p, y, s=50, color=POS_C, edgecolor=SURFACE, lw=1.4, zorder=3)
        ax.scatter(q, y, s=50, color=SCORE_C, edgecolor=SURFACE, lw=1.4, zorder=3)
    y = int(np.argmax(np.abs(np.array(b) - np.array(a))))
    ax.annotate(fmt.format(b[y] - a[y]), (max(a[y], b[y]), y), xytext=(8, 0), textcoords="offset points", va="center", color=SEC)
    ax.set(ylim=(len(seeds) - .5, -.5), yticks=range(len(seeds)), yticklabels=seeds, xlabel=xlabel, ylabel="Scenario")
    ax.set_title(title)
    clean(ax, "x")


def summary(res, out, n, x, w):
    seeds = sorted({r["seed"] for r in res})
    by = {(r["seed"], r["config"], r["n"]): r for r in res}
    cfgs = [c for c in ("cand", "pos", "score") if (seeds[0], c, n) in by]
    sx = str(int(x))
    mean = lambda c, f: np.mean([f(by[(s, c, n)]) for s in seeds], axis=0)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.8), constrained_layout=True)
    ax = axs[0, 0]
    for c in cfgs:
        ax.plot(np.arange(0, 61, 5), mean(c, lambda r: r["pos_t"]), color=COLOR[c], lw=1.2 if c == "cand" else 1.8)
    ax.set(xlim=(0, 60), ylim=(0, 1), xticks=range(0, 61, 15), xlabel="Minutes after first handover", ylabel="Mean held-out POS")
    ax.set_title(f"POS over time\nEnd: baseline {mean('pos', lambda r: r['pos']):.3f}, A + POS {mean('score', lambda r: r['pos']):.3f}")
    clean(ax)
    ax = axs[0, 1]
    for c in cfgs:
        ax.plot(np.arange(0, 61, 5), np.r_[0.0, mean(c, lambda r: r["scores"][sx]["c"])], color=COLOR[c], lw=1.2 if c == "cand" else 1.8)
    ax.set(xlim=(0, 60), ylim=(0, 1), xticks=range(0, 61, 15), xlabel="Minutes after the LKP report", ylabel=f"Share of the current {x:g}% region swept")
    ax.set_title(f"Swept share c(t), its mean is A\nA: baseline {mean('pos', lambda r: r['scores'][sx]['A']):.3f}, A + POS {mean('score', lambda r: r['scores'][sx]['A']):.3f}")
    clean(ax)
    ax = axs[0, 2]
    for c in cfgs:
        y = [mean(c, lambda r: r["scores"][sx]["inside"][str(lo)]) for lo in (0, 15, 30, 45)]
        ax.plot([7.5, 22.5, 37.5, 52.5], y, color=COLOR[c], lw=1.2 if c == "cand" else 1.8, marker="o", ms=5.5, mec=SURFACE)
    ax.set(xlim=(0, 60), ylim=(0, 1), xticks=range(0, 61, 15), xlabel="Minutes after the LKP report (15 min bins)", ylabel="Share of flight time inside the region")
    ax.set_title(f"Time inside the current {x:g}% region\nLaunch and approach count as outside")
    clean(ax)
    a = [by[(s, "pos", n)] for s in seeds]
    b = [by[(s, "score", n)] for s in seeds]
    dumbbell(axs[1, 0], seeds, [r["pos"] for r in a], [r["pos"] for r in b], "Held-out POS at end of flight", "POS per scenario (label: largest gap)")
    axs[1, 0].set_xlim(0, 1.04)
    dumbbell(axs[1, 1], seeds, [r["scores"][sx]["A"] for r in a], [r["scores"][sx]["A"] for r in b], "Base score A", "A per scenario (label: largest gap)")
    axs[1, 1].set_xlim(left=0)
    dumbbell(axs[1, 2], seeds, [r["below"] for r in a], [r["below"] for r in b], "Seconds with a UAV pair closer than 300 m", "Time close together per scenario (no constraint)", "{:+.0f} s")
    axs[1, 2].set_xlim(left=0)
    axs[1, 2].margins(x=.14)
    keys = [Line2D([], [], color=COLOR[c], lw=1.2 if c == "cand" else 1.8, marker=None if c == "cand" else "o", ms=6, mec=SURFACE, label=LABEL[c]) for c in ("pos", "score", "cand") if c in cfgs]
    fig.legend(handles=keys, loc="outside lower center", ncol=3)
    fig.suptitle(f"SAROPS, {n} UAV{'s' if n > 1 else ''}, scenarios {seeds[0]}-{seeds[-1]}: POS only against A + POS (x {x:g}, w {w:g})", x=.01, ha="left", fontsize=12)
    fig.savefig(out/f"summary_n{n}_x{x:g}_w{w:g}.png", dpi=125)
    plt.close(fig)


def maps(res, out, n, x, w, seeds=None, minutes=(15, 30, 45, 60)):
    from rl_search.area_score import AreaScore, CELL_M
    from rl_search.patterns import _rot
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center_open import make_envs
    from rl_search.scripts.plot_center_terrain import close_with_time, remaining_weights, terrain_rgb
    by = {(r["seed"], r["config"], r["n"]): r for r in res}
    sx = str(int(x))
    all_seeds = sorted({r["seed"] for r in res})
    gain = {s: by[(s, "score", n)]["scores"][sx]["A"] - by[(s, "pos", n)]["scores"][sx]["A"] for s in all_seeds}
    cmap = LinearSegmentedColormap.from_list("violet", ["#e4e0fa", "#a79cec", "#6250d6", "#2d2180"])
    norm = Normalize(0, 1)
    orange = to_rgb(SWEPT)
    for seed in (seeds or (max(gain, key=gain.get), min(gain, key=gain.get))):
        env = make_envs(1, seed)[0]
        terr = env.terr
        scen = open_scenario(env.sampler, seed)
        hold = holdout_scenario(env, scen, seed + 10**6)
        rgb, ext_t = terrain_rgb(terr), [terr.axis[0]/1000, terr.axis[-1]/1000]*2
        cuts = [m*60.0 for m in minutes]
        data = {}
        for cfg in ("pos", "score"):
            r = by[(seed, cfg, n)]
            xy, tt = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]]
            a = AreaScore(terr, env.h, hold.track, hold.save_dt_s, x, env.lrc_R)
            seg = sorted((float(t[k + 1]), u, k) for u, t in enumerate(tt) for k in range(len(t) - 1))
            q, frames = 0, []
            for c, m in zip(cuts, minutes):
                while q < len(seg) and seg[q][0] <= c:
                    _, u, k = seg[q]
                    a.mark(xy[u][k:k + 2], tt[u][k:k + 2])
                    q += 1
                reg = ((a.B >> (m//5 - 1)) & 1).astype(bool)
                frames.append((reg, a.S & reg, a.S & ~reg))
            data[cfg] = dict(r=r, xy=xy, tt=tt, W=remaining_weights(env, hold, xy, tt, cuts), a=a, frames=frames, close=close_with_time(xy, tt))
        views = []
        for c in cuts:
            P = np.asarray(hold.track[min(int(c/hold.save_dt_s), len(hold.track) - 1)], float)
            pts = [P[data[k]["W"][c] > .02] for k in data]
            pts += [p[(t <= c) & (t >= c - 600)] for k in data for p, t in zip(data[k]["xy"], data[k]["tt"])]
            pts = np.concatenate(pts)
            lo, hi = np.quantile(pts, .01, axis=0) - 1500, np.quantile(pts, .99, axis=0) + 1500
            views.append(((lo + hi)/2000, (hi - lo).max()/2000, P))
        fig, axs = plt.subplots(2, len(minutes), figsize=(4.6*len(minutes) + 1, 10.6), constrained_layout=True)
        for i, (cfg, name) in enumerate((("pos", "POS only"), ("score", f"A + {w:g} POS"))):
            d = data[cfg]
            r, a = d["r"], d["a"]
            ext = [a.x0/1000, (a.x0 + a.B.shape[1]*CELL_M)/1000, a.y0/1000, (a.y0 + a.B.shape[0]*CELL_M)/1000]
            for j, (c, m) in enumerate(zip(cuts, minutes)):
                ax = axs[i, j]
                mid, half, P = views[j]
                reg, swept, left = d["frames"][j]
                ax.imshow(rgb, origin="lower", extent=ext_t, interpolation="nearest")
                over = np.zeros(reg.shape + (4,))
                over[left] = (*orange, .22)
                over[swept] = (*orange, .55)
                ax.imshow(over, origin="lower", extent=ext, interpolation="nearest")
                wt = d["W"][c]
                k = wt > 1e-3
                o = np.argsort(wt[k])
                ax.scatter(P[k][o, 0]/1000, P[k][o, 1]/1000, c=wt[k][o], s=3, cmap=cmap, norm=norm, linewidths=0)
                ax.contour(a.gx/1000, a.gy/1000, reg.astype(float), levels=[.5], colors=[REGION_LINE], linewidths=1.0)
                for u, (p, t) in enumerate(zip(d["xy"], d["tt"])):
                    sel = (t <= c) & (t >= c - 600)
                    ax.plot(p[sel, 0]/1000, p[sel, 1]/1000, color=INK, lw=.8)
                    if (t <= c).any():
                        e = p[np.flatnonzero(t <= c)[-1]]
                        ax.plot(e[0]/1000, e[1]/1000, "o", color=INK, ms=6, mec="white", mew=1.1, zorder=5)
                    for tp, cx, cy, ell, wid, th, hs in r["rects"][u]:
                        if tp <= c < tp + hs + 1500:
                            corners = np.array([[sx_*ell/2, sy_*wid/2] for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                            ax.add_patch(Polygon((np.array([cx, cy]) + corners @ _rot(th).T)/1000, fill=False, ec=INK, lw=.7, ls="--"))
                near = d["close"][np.abs(d["close"][:, 0] - c) <= 300] if len(d["close"]) else d["close"]
                if len(near):
                    ax.scatter(near[:, 1]/1000, near[:, 2]/1000, s=14, c=CLOSE, zorder=6, linewidths=0)
                ax.plot(*np.array(r["lkp"])/1000, "*", color=INK, ms=11, mec="white", zorder=6)
                ax.set(xlim=(mid[0] - half, mid[0] + half), ylim=(mid[1] - half, mid[1] + half))
                ax.set_title(f"{name}, {m} min: remaining {wt.mean():.2f}, {r['scores'][sx]['c'][m//5 - 1]:.0%} of region swept")
                ax.tick_params(length=0)
                if j == 0:
                    ax.set_ylabel("km")
                if i == 1:
                    ax.set_xlabel("km")
        fig.colorbar(ScalarMappable(norm, cmap), ax=axs, shrink=.45, label="Remaining probability of a particle")
        keys = [Line2D([], [], color=REGION_LINE, lw=1, label=f"{x:g}% region now (unweighted)"), Patch(color=(*orange, .55), label="Swept, in the region now"),
                Patch(color=(*orange, .22), label="Swept earlier, region moved on"), Line2D([], [], color=INK, lw=.8, label="Tracks, last 10 min"),
                Line2D([], [], color=INK, lw=.7, ls="--", label="Current rectangles"), Line2D([], [], marker="o", ls="", color=INK, mec="white", ms=6, label="UAV now"),
                Line2D([], [], marker="o", ls="", color=CLOSE, ms=4.5, label="Pair < 300 m, within 5 min"), Line2D([], [], marker="*", ls="", color=INK, mec="white", ms=11, label="LKP"),
                Patch(color=(.55, .55, .55), label="No-go"), Patch(color=(.70, .82, .95), label="Water"), Patch(color=(.80, .90, .78), label="Forest")]
        fig.legend(handles=keys, loc="outside lower center", ncol=6)
        pa, pb = data["pos"]["r"], data["score"]["r"]
        fig.suptitle(f"Scenario {seed}, {n} UAV{'s' if n > 1 else ''}, x {x:g}, w {w:g}{', separation constraint on' if pa.get('sep') else ''}{', rectangle rule on' if pa.get('no_overlap') else ''}: A {pa['scores'][sx]['A']:.3f} to {pb['scores'][sx]['A']:.3f}, POS {pa['pos']:.3f} to {pb['pos']:.3f}. "
                     "Each column uses one scale.", x=.01, ha="left", fontsize=12)
        f = out/f"map_s{seed}_n{n}_x{x:g}_w{w:g}.png"
        fig.savefig(f, dpi=95)
        plt.close(fig)
        print("wrote", f, flush=True)


def sweep_tradeoff(res, out):
    """What each (x, w) gains in A and costs in POS against POS only, paired by scenario, and how long pairs stay close."""
    seeds = sorted({r["seed"] for r in res})
    by = {(r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ()): r for r in res}
    xs, ws = (50.0, 70.0, 85.0), (0.5, 1.0, 2.0)
    col = {50.0: "#2a78d6", 70.0: "#eb6834", 85.0: "#1baf7a"}     # first three slots, validated all-pairs
    mark = {0.5: "^", 1.0: "o", 2.0: "s"}
    ns = (1, 3, 5)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.6), height_ratios=[1.25, 1], constrained_layout=True)
    se = lambda d: d.std(ddof=1)/np.sqrt(len(d))
    for j, n in enumerate(ns):
        ax = axs[0, j]
        ax.axhline(0, color=SEC, lw=1)
        ax.axvline(0, color=SEC, lw=1)
        for x in xs:
            sx = str(int(x))
            for w in ws:
                run = [by[(s, "score", n, x, w)] for s in seeds]
                base = [by[(s, "pos", n)] for s in seeds]
                dp = np.array([a["pos"] - b["pos"] for a, b in zip(run, base)])
                da = np.array([a["scores"][sx]["A"] - b["scores"][sx]["A"] for a, b in zip(run, base)])
                ax.errorbar(dp.mean(), da.mean(), xerr=se(dp), yerr=se(da), color=col[x], lw=.8, alpha=.5, zorder=2)
                ax.scatter(dp.mean(), da.mean(), s=62, marker=mark[w], color=col[x], edgecolor=SURFACE, lw=1.2, zorder=3)
                if x == 70.0 and w == 1.0:
                    ax.annotate("x 70, w 1", (dp.mean(), da.mean()), xytext=(8, 7), textcoords="offset points", color=SEC, fontsize=9)
        base_pos = np.mean([by[(s, "pos", n)]["pos"] for s in seeds])
        ax.set(xlim=(-.032, .032), ylim=(-.005, .112), xlabel="POS change from POS only", ylabel="A change from POS only (at the run's own x)")
        ax.set_title(f"{n} UAV{'s' if n > 1 else ''} (POS only: POS {base_pos:.3f}). Upper right is better")
        clean(ax)
        ax.grid(axis="x", color=GRID, lw=.8)
        ax = axs[1, j]
        base = np.mean([by[(s, "pos", n)]["below"] for s in seeds])
        for i, x in enumerate(xs):
            for k, w in enumerate(ws):
                v = np.mean([by[(s, "score", n, x, w)]["below"] for s in seeds])
                ax.scatter(i + (k - 1)*.2, v, s=62, marker=mark[w], color=col[x], edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.axhline(base, color=SEC, lw=1)
        ax.set(xlim=(-.5, 2.5), ylim=(0, 400), xticks=range(3), xticklabels=[f"x {x:.0f}" for x in xs], ylabel="Seconds with a pair closer than 300 m")
        ax.set_title(f"Time close together, no constraint (line: POS only, {base:.0f} s)" if n > 1 else "One UAV has no pairs")
        clean(ax)
    keys = [Line2D([], [], marker="o", ls="", color=col[x], ms=7, label=f"x {x:.0f}") for x in xs]
    keys += [Line2D([], [], marker=mark[w], ls="", color=MUTED, ms=7, label=f"w {w:g}") for w in ws]
    keys.append(Line2D([], [], color=MUTED, lw=.8, label="1 standard error"))
    fig.legend(handles=keys, loc="outside lower center", ncol=7)
    fig.suptitle("SAROPS with A + w POS, scenarios 500-509: x 70 or 85 with w 1 gain A without losing POS; w 0.5 gains a little more A but loses POS with 3 or 5 UAVs", x=.01, ha="left", fontsize=12)
    fig.savefig(out/"tradeoff.png", dpi=125)
    plt.close(fig)


def sep_overview(res, ref, out):
    """Separation constraint off (hollow) and on (filled) for each configuration, and when and how close pairs get."""
    seeds = sorted({r["seed"] for r in res})
    key = lambda r: (r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ())
    on, off = {key(r): r for r in res}, {key(r): r for r in ref}
    cfgs = (("POS only", ("pos",), "#2a78d6"), ("A + POS, x 70", ("score", 70.0, 1.0), "#eb6834"), ("A + POS, x 85", ("score", 85.0, 1.0), "#1baf7a"))     # validated, all pairs
    get = lambda d, s, n, c: d[(s, c[0], n) + tuple(c[1:])]
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.8), height_ratios=[1, 1.1], constrained_layout=True)
    rows = [(3, c) for c in cfgs] + [(5, c) for c in cfgs]
    ypos = [0, 1, 2, 3.6, 4.6, 5.6]
    labels = [f"{n} UAVs, {c[0]}" for n, c in rows]
    panels = ((axs[0, 0], "POS at end of flight", lambda r: r["pos"], (.5, .8), "{:.3f}"),
              (axs[0, 1], "A at x 70 (one yardstick for all rows)", lambda r: r["scores"]["70"]["A"], (.15, .4), "{:.3f}"),
              (axs[0, 2], "Seconds with a pair closer than 300 m", lambda r: r["below"], (0, 300), "{:.0f} s"))
    for ax, title, f, lim, fmt in panels:
        for y, (n, c) in zip(ypos, rows):
            a = np.mean([f(get(off, s, n, c[1])) for s in seeds])
            b = np.mean([f(get(on, s, n, c[1])) for s in seeds])
            ax.plot([a, b], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(a, y, s=58, facecolor=SURFACE, edgecolor=c[2], lw=1.6, zorder=3)
            ax.scatter(b, y, s=58, color=c[2], edgecolor=SURFACE, lw=1.2, zorder=4)
            ax.annotate(fmt.format(b), (max(a, b), y), xytext=(9, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
        ax.set(xlim=lim, ylim=(6.2, -.6), yticks=ypos, yticklabels=labels if ax is axs[0, 0] else [])
        ax.set_title(title)
        clean(ax, "x")
    bins = ("0-15", "15-30", "30-45", "45-60")
    for ax, n in ((axs[1, 0], 3), (axs[1, 1], 5)):
        for name, c, col in cfgs:
            v = np.mean([get(on, s, n, c)["close_by15"][:4] for s in seeds], axis=0)
            ax.plot(range(4), v, color=col, lw=1.8, marker="o", ms=6, mec=SURFACE)
        ax.set(xticks=range(4), xticklabels=bins, xlabel="Minutes after the LKP report", ylabel="Seconds below 300 m, constraint on", ylim=(0, 160))
        ax.set_title(f"{n} UAVs: when pairs get close")
        clean(ax)
    ax = axs[1, 2]
    for k, (name, c, col) in enumerate(cfgs):
        v = [max(get(on, s, 5, c)["dmin"], 1.0) for s in seeds]
        ax.scatter(v, np.arange(len(seeds)) + (k - 1)*.22, s=44, color=col, edgecolor=SURFACE, lw=1, zorder=3)
    ax.axvline(300, color=CLOSE, lw=1.2)
    ax.annotate("300 m", (300, -.45), xytext=(5, 0), textcoords="offset points", color=SEC, fontsize=9, va="center")
    ax.set(xscale="log", xlim=(.8, 1200), ylim=(len(seeds) - .5, -.7), yticks=range(len(seeds)), yticklabels=seeds, xlabel="Closest pair in the flight, m (log scale)", ylabel="Scenario")
    ax.set_title("5 UAVs, constraint on: closest pair per scenario")
    clean(ax, "x")
    keys = [Line2D([], [], marker="o", ls="", color=c[2], ms=7, label=c[0]) for c in cfgs]
    keys += [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, mew=1.4, ms=7, label="Constraint off"), Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="Constraint on")]
    fig.legend(handles=keys, loc="outside lower center", ncol=5)
    fig.suptitle("Separation constraint, SAROPS, scenarios 500-509: costs about 0.02 POS, cuts close time most with A + POS, but pairs still come within a few metres in the first 30 min",
                 x=.01, ha="left", fontsize=11.5)
    fig.savefig(out/"overview.png", dpi=125)
    plt.close(fig)


def rule_overview(base, out):
    """No constraint (hollow), time rule (light), time + rectangle rule (solid), and when and how close pairs get."""
    load = lambda name: {(r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ()): r
                         for r in map(json.loads, (base/name/"results.jsonl").read_text().splitlines())}
    off, tim, rule = load("sweep"), load("sep"), load("no_overlap")
    seeds = sorted({k[0] for k in rule})
    cfgs = (("POS only", ("pos",), "#2a78d6"), ("A + POS, x 70", ("score", 70.0, 1.0), "#eb6834"), ("A + POS, x 85", ("score", 85.0, 1.0), "#1baf7a"))     # validated, all pairs
    get = lambda d, s, n, c: d[(s, c[0], n) + tuple(c[1:])]
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.8), height_ratios=[1, 1.1], constrained_layout=True)
    rows = [(3, c) for c in cfgs] + [(5, c) for c in cfgs]
    ypos = [0, 1, 2, 3.6, 4.6, 5.6]
    panels = ((axs[0, 0], "POS at end of flight", lambda r: r["pos"], (.55, .78), "{:.3f}"),
              (axs[0, 1], "A at x 70 (one yardstick for all rows)", lambda r: r["scores"]["70"]["A"], (.18, .38), "{:.3f}"),
              (axs[0, 2], "Seconds with a pair closer than 300 m", lambda r: r["below"], (0, 300), "{:.0f} s"))
    for ax, title, f, lim, fmt in panels:
        for y, (n, c) in zip(ypos, rows):
            a, b, d = (np.mean([f(get(t, s, n, c[1])) for s in seeds]) for t in (off, tim, rule))
            ax.plot([min(a, b, d), max(a, b, d)], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(a, y, s=58, facecolor=SURFACE, edgecolor=c[2], lw=1.6, zorder=3)
            ax.scatter(b, y, s=58, color=c[2], alpha=.4, linewidths=0, zorder=4)
            ax.scatter(d, y, s=58, color=c[2], edgecolor=SURFACE, lw=1.2, zorder=5)
            ax.annotate(fmt.format(d), (max(a, b, d), y), xytext=(9, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
        ax.set(xlim=lim, ylim=(6.2, -.6), yticks=ypos, yticklabels=[f"{n} UAVs, {c[0]}" for n, c in rows] if ax is axs[0, 0] else [])
        ax.set_title(title)
        clean(ax, "x")
    bins = ("0-15", "15-30", "30-45", "45-60")
    for ax, n in ((axs[1, 0], 3), (axs[1, 1], 5)):
        for name, c, col in cfgs:
            v = np.mean([get(rule, s, n, c)["close_by15"][:4] for s in seeds], axis=0)
            ax.plot(range(4), v, color=col, lw=1.8, marker="o", ms=6, mec=SURFACE)
        ax.set(xticks=range(4), xticklabels=bins, xlabel="Minutes after the LKP report", ylabel="Seconds below 300 m, both rules on", ylim=(0, 160))
        ax.set_title(f"{n} UAVs: when pairs get close")
        clean(ax)
    ax = axs[1, 2]
    for k, (name, c, col) in enumerate(cfgs):
        v = [max(get(rule, s, 5, c)["dmin"], 1.0) for s in seeds]
        ax.scatter(v, np.arange(len(seeds)) + (k - 1)*.22, s=44, color=col, edgecolor=SURFACE, lw=1, zorder=3)
    ax.axvline(300, color=CLOSE, lw=1.2)
    ax.annotate("300 m", (300, -.45), xytext=(5, 0), textcoords="offset points", color=SEC, fontsize=9, va="center")
    ax.set(xscale="log", xlim=(.8, 1200), ylim=(len(seeds) - .5, -.7), yticks=range(len(seeds)), yticklabels=seeds, xlabel="Closest pair in the flight, m (log scale)", ylabel="Scenario")
    ax.set_title("5 UAVs, both rules on: closest pair per scenario")
    clean(ax, "x")
    keys = [Line2D([], [], marker="o", ls="", color=c[2], ms=7, label=c[0]) for c in cfgs]
    keys += [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, mew=1.4, ms=7, label="No constraint"), Line2D([], [], marker="o", ls="", color=MUTED, alpha=.4, ms=7, label="Time rule"),
             Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="Time + rectangle rule")]
    fig.legend(handles=keys, loc="outside lower center", ncol=6)
    fig.suptitle("Adding the rectangle rule, SAROPS, scenarios 500-509: POS holds and close time falls further, but a few scenarios still bring pairs within metres in the first 30 min",
                 x=.01, ha="left", fontsize=11.5)
    fig.savefig(out/"overview.png", dpi=125)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("rl_search/data/center_open/first_check"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--x", type=float, default=70.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="map scenarios; default the largest and smallest gain in A")
    ap.add_argument("--tradeoff", action="store_true", help="only the x and w overview of a sweep")
    ap.add_argument("--sep-overview", action="store_true", help="only the constraint on and off overview (--input is the sep folder)")
    ap.add_argument("--rule-overview", action="store_true", help="only the three rule levels (--input is the no_overlap folder)")
    args = ap.parse_args()
    style()
    if args.rule_overview:
        rule_overview(args.input.parent, args.input)
        return
    res = [json.loads(line) for line in (args.input/"results.jsonl").read_text().splitlines()]
    if args.sep_overview:
        sep_overview(res, [json.loads(line) for line in (args.input.parent/"sweep"/"results.jsonl").read_text().splitlines()], args.input)
        return
    if args.tradeoff:
        sweep_tradeoff(res, args.input)
        return
    res = [r for r in res if r["n"] == args.n and (r["config"] != "score" or (r["x"] == args.x and r["w"] == args.w))]
    summary(res, args.input, args.n, args.x, args.w)
    maps(res, args.input, args.n, args.x, args.w, args.seeds)


if __name__ == "__main__":
    main()
