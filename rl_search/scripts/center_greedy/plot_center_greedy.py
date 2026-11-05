"""Exploratory figures from the saved results (no planning rerun): SAROPS vs Greedy90 on the centre scenarios 400-409."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SAR, GRE, SEP, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"     # validated: scripts/validate_palette.js
SEEDS = list(range(400, 410))
MINUTES = np.arange(0, 61, 5)

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS,
                     "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SEC,
                     "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlesize": 10.5,
                     "legend.frameon": False})

NEW = {(r["seed"], r["method"], r["n"], r["sep"]): r for r in map(json.loads, open(HERE/"center_greedy.jsonl"))}
OLD = {(r["seed"], r["mode"]): r for r in json.load(open("rl_search/data/compare_center/results.json"))}


def get(seed, method, n, sep=False):
    """One run: the existing compare_center result where it exists, else this session's run."""
    if method == "sarops" and n in (1, 5):
        r = OLD[(seed, "1" if n == 1 else ("5 sep" if sep else "5 no sep"))]
        return dict(r, hold=r["hold_pos"])
    return NEW[(seed, method, n, sep)]


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def uav_label(n):
    return f"{n} UAV" + ("s" if n > 1 else "")


def summary():
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), height_ratios=[1, 1.15], constrained_layout=True)
    for j, n in enumerate((1, 3, 5)):
        ax = axs[0, j]
        s = np.mean([get(k, "sarops", n)["hold"] for k in SEEDS])
        g = np.mean([get(k, "greedy", n)["hold"] for k in SEEDS])
        for m, c, lab in (("sarops", SAR, "SAROPS"), ("greedy", GRE, "greedy")):
            ax.plot(MINUTES, np.mean([get(k, m, n)["hold_t"] for k in SEEDS], axis=0), color=c, lw=1.8, label=lab,
                    solid_capstyle="round")
        ax.set(xlim=(0, 60), ylim=(0, 1), xlabel="Minutes after first handover", xticks=range(0, 61, 15))
        ax.set_title(f"{uav_label(n)}\nEnd of flight: SAROPS {s:.3f}, greedy {g:.3f}")
        clean(ax)
        if j == 0:
            ax.set_ylabel("Mean held-out POS")

        ax = axs[1, j]
        rows = [(k, get(k, "sarops", n)["hold"], get(k, "greedy", n)["hold"]) for k in SEEDS]
        for y, (k, a, b) in enumerate(rows):
            ax.plot([a, b], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(a, y, s=52, color=SAR, edgecolor=SURFACE, lw=1.4, zorder=3)
            ax.scatter(b, y, s=52, color=GRE, edgecolor=SURFACE, lw=1.4, zorder=3)
        y, (k, a, b) = max(enumerate(rows), key=lambda x: abs(x[1][2] - x[1][1]))
        ax.annotate(f"{b - a:+.3f}", (max(a, b), y), xytext=(8, 0), textcoords="offset points", va="center", color=SEC)
        ax.set(xlim=(.3, 1.04), ylim=(len(SEEDS) - .5, -.5), yticks=range(len(SEEDS)), yticklabels=SEEDS,
               xlabel="Held-out POS at end of flight")
        ax.set_title("Each scenario (label: largest gap)")
        clean(ax, "x")
        if j == 0:
            ax.set_ylabel("Scenario")
    fig.legend(handles=[Line2D([], [], color=SAR, lw=1.8, marker="o", ms=6, mec=SURFACE, label="SAROPS"),
                        Line2D([], [], color=GRE, lw=1.8, marker="o", ms=6, mec=SURFACE, label="greedy")],
               loc="outside lower center", ncol=2)
    fig.suptitle("Centre scenarios 400-409: greedy finds slightly more than SAROPS, and the number of UAVs matters far more "
                 "(no separation constraint)", x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/"center_pos_summary.png", dpi=130)
    plt.close(fig)


def near_miss():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, constrained_layout=True)
    series = (("sarops", False, SAR, "SAROPS, no separation"), ("greedy", False, GRE, "greedy, no separation"),
              ("sarops", True, SEP, "SAROPS, separation constraint"))
    for ax, n in zip(axs, (3, 5)):
        for y, k in enumerate(SEEDS):
            ax.axhline(y, color=GRID, lw=.8, zorder=1)
        means = []
        for m, sep, c, lab in series:
            v = [get(k, m, n, sep)["below"] for k in SEEDS]
            means.append(np.mean(v))
            ax.scatter(v, range(len(SEEDS)), s=52, color=c, edgecolor=SURFACE, lw=1.4, zorder=3, label=lab)
        ax.set(ylim=(len(SEEDS) - .5, -.5), yticks=range(len(SEEDS)), yticklabels=SEEDS, xlim=(-60, 2450),
               xlabel="Seconds with a UAV pair closer than 300 m (summed over pairs)")
        ax.set_title(f"{uav_label(n)}\nMean: SAROPS {means[0]:.0f} s, greedy {means[1]:.0f} s, SAROPS with constraint {means[2]:.0f} s")
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("Scenario")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("On average, greedy flies UAVs close together far longer than SAROPS (greedy has no separation constraint)", x=.01,
                 ha="left", fontsize=12)
    fig.savefig(HERE/"center_near_miss.png", dpi=130)
    plt.close(fig)


def paths(seed, n=5, minutes=(15, 30, 45)):
    from rl_search.patterns import _rot
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario
    from rl_search.scripts.plot_center_terrain import close_with_time, remaining_weights, terrain_rgb
    env = make_envs(1, seed)[0]
    terr = env.terr
    rgb, ext = terrain_rgb(terr), [terr.axis[0]/1000, terr.axis[-1]/1000]*2
    scen = center_scenario(env.sampler, seed)
    t_h = env.preset.handover.t_plan_s
    cuts = [t_h + m*60 for m in minutes]
    end = 1e9
    cmap = LinearSegmentedColormap.from_list("violet", ["#e4e0fa", "#a79cec", "#6250d6", "#2d2180"])
    norm = Normalize(0, 1)
    data = {}
    for m in ("sarops", "greedy"):
        r = get(seed, m, n)
        xy, tt = [np.array(x, float) for x in r["xy"]], [np.array(t, float) for t in r["tt"]]
        W = remaining_weights(env, scen, xy, tt, cuts + [end])
        data[m] = (r, xy, tt, W, close_with_time(xy, tt))
        print(f"seed {seed} {m} N={n}: replayed on saved paths {1 - W[end].mean():.3f}, recorded on planner's particles "
              f"{r['pos']:.3f}, held-out {r['hold']:.3f}", flush=True)
    views = []
    for c in cuts:
        P = np.asarray(scen.track[min(int(c/scen.save_dt_s), len(scen.track) - 1)], float)
        pts = [P[data[m][3][c] > .02] for m in data]
        pts += [x[(t <= c) & (t >= c - 600)] for m in data for x, t in zip(data[m][1], data[m][2])]
        pts = np.concatenate(pts)
        lo, hi = np.quantile(pts, .01, axis=0) - 1500, np.quantile(pts, .99, axis=0) + 1500
        views.append(((lo + hi)/2, max(hi - lo)/2, P))
    fig, axs = plt.subplots(2, len(cuts), figsize=(5.0*len(cuts) + 1, 10.9), constrained_layout=True)
    for i, (m, name) in enumerate((("sarops", "SAROPS"), ("greedy", "greedy"))):
        r, xy, tt, W, cl = data[m]
        for j, c in enumerate(cuts):
            ax = axs[i, j]
            mid, half, P = views[j]
            ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
            w = W[c]
            k = w > 1e-3
            o = np.argsort(w[k])
            ax.scatter(P[k][o, 0]/1000, P[k][o, 1]/1000, c=w[k][o], s=3, cmap=cmap, norm=norm, linewidths=0)
            for u in range(len(xy)):
                sel = (tt[u] <= c) & (tt[u] >= c - 600)
                ax.plot(xy[u][sel, 0]/1000, xy[u][sel, 1]/1000, color=INK, lw=.9)
                if (tt[u] <= c).any():
                    p = xy[u][np.flatnonzero(tt[u] <= c)[-1]]
                    ax.plot(p[0]/1000, p[1]/1000, "o", color=INK, ms=6.5, mec="white", mew=1.2, zorder=5)
                for tp, cx, cy, ell, wid, th, hs in r.get("rects", [[]]*len(xy))[u]:
                    if tp <= c < tp + hs + 1500:
                        corners = np.array([[sx*ell/2, sy*wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                        ax.add_patch(Polygon((np.array([cx, cy]) + corners @ _rot(th).T)/1000, fill=False, ec=INK, lw=.8,
                                             ls="--"))
            near = cl[np.abs(cl[:, 0] - c) <= 300] if len(cl) else cl
            if len(near):
                ax.scatter(near[:, 1]/1000, near[:, 2]/1000, s=16, c=CRIT, zorder=6, linewidths=0)
            ax.set(xlim=((mid[0] - half)/1000, (mid[0] + half)/1000), ylim=((mid[1] - half)/1000, (mid[1] + half)/1000))
            ax.set_title(f"{name}, {minutes[j]} min: remaining {w.mean():.2f}")
            ax.tick_params(length=0)
            if j == 0:
                ax.set_ylabel("km")
            if i == 1:
                ax.set_xlabel("km")
    fig.colorbar(ScalarMappable(norm, cmap), ax=axs, shrink=.5, label="Remaining probability of a particle")
    keys = [Line2D([], [], color=INK, lw=.9, label="Track, last 10 min"),
            Line2D([], [], marker="o", color=INK, mec="white", ms=6.5, ls="", label="UAV now"),
            Line2D([], [], color=INK, lw=.8, ls="--", label="SAROPS rectangle"),
            Line2D([], [], marker="o", color=CRIT, ms=4.5, ls="", label="Pair < 300 m, within 5 min"),
            Patch(color=(.55, .55, .55), label="No-go"), Patch(color=(.70, .82, .95), label="Water"),
            Patch(color=(.80, .90, .78), label="Forest")]
    fig.legend(handles=keys, loc="outside lower center", ncol=len(keys))
    a, b = data["sarops"][0]["hold"], data["greedy"][0]["hold"]
    fig.suptitle(f"Scenario {seed}, {uav_label(n)}, no separation constraint: held-out POS SAROPS {a:.3f}, greedy {b:.3f}. "
                 "Each column uses one scale.", x=.01, ha="left", fontsize=12)
    f = HERE/f"center_paths_s{seed}_n{n}.png"
    fig.savefig(f, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    summary()
    near_miss()
    paths(408)
    paths(402)
    print("done")
