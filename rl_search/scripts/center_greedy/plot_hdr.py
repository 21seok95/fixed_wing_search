"""Exploratory figures for hdr_coverage.py: share of the 85% region swept, SAROPS vs greedy, centre scenarios 400-409."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parents[3]/"rl_search/data/center_greedy"
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SAR, GRE = "#2a78d6", "#eb6834"                                      # validated: scripts/validate_palette.js
REGION, IN_REGION, OUTSIDE = "#cde2fb", "#1c5cab", "#c3c2b7"         # one sequential hue plus a neutral
SEEDS, NS = list(range(400, 410)), (1, 3, 5)
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS,
                     "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SEC,
                     "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlesize": 10.5,
                     "legend.frameon": False})
ROWS = {(r["seed"], r["method"], r["n"]): r for r in json.load(open(HERE/"hdr_coverage.json"))}


def uav_label(n):
    return f"{n} UAV" + ("s" if n > 1 else "")


def summary():
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), height_ratios=[1, 1.15], constrained_layout=True)
    for j, n in enumerate(NS):
        ax = axs[0, j]
        vals = {}
        for m, c, lab in (("sarops", SAR, "SAROPS"), ("greedy", GRE, "greedy")):
            rs = [ROWS[(s, m, n)] for s in SEEDS]
            vals[m] = np.mean([r["ratio"] for r in rs])
            y = [0.0] + [np.mean([r[f"ratio{k}"] for r in rs]) for k in (15, 30, 45, 60)]
            ax.plot([0, 15, 30, 45, 60], y, color=c, lw=1.8, marker="o", ms=5.5, mec=SURFACE, label=lab)
        ax.set(xlim=(0, 62), ylim=(0, 1), xticks=range(0, 61, 15), xlabel="Minutes after first handover")
        ax.set_title(f"{uav_label(n)}\nEnd of flight: SAROPS {vals['sarops']:.3f}, greedy {vals['greedy']:.3f}")
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        if j == 0:
            ax.set_ylabel("Mean share of the 85% region swept")
        ax = axs[1, j]
        rows = [(s, ROWS[(s, "sarops", n)]["ratio"], ROWS[(s, "greedy", n)]["ratio"]) for s in SEEDS]
        for y, (s, a, b) in enumerate(rows):
            ax.plot([a, b], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(a, y, s=52, color=SAR, edgecolor=SURFACE, lw=1.4, zorder=3)
            ax.scatter(b, y, s=52, color=GRE, edgecolor=SURFACE, lw=1.4, zorder=3)
        y, (s, a, b) = max(enumerate(rows), key=lambda x: abs(x[1][2] - x[1][1]))
        ax.annotate(f"{b - a:+.3f}", (max(a, b), y), xytext=(8, 0), textcoords="offset points", va="center", color=SEC)
        ax.set(xlim=(0, 1.04), ylim=(len(SEEDS) - .5, -.5), yticks=range(len(SEEDS)), yticklabels=SEEDS,
               xlabel="Share of the 85% region swept, end of flight")
        ax.set_title("Each scenario (label: largest gap)")
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        if j == 0:
            ax.set_ylabel("Scenario")
    fig.legend(handles=[Line2D([], [], color=SAR, lw=1.8, marker="o", ms=6, mec=SURFACE, label="SAROPS"),
                        Line2D([], [], color=GRE, lw=1.8, marker="o", ms=6, mec=SURFACE, label="greedy")],
               loc="outside lower center", ncol=2)
    fig.suptitle("Share of the 85% region swept: greedy ahead with 1 UAV, no clear difference with 3 or 5 UAVs "
                 "(centre scenarios 400-409)", x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/"hdr_summary.png", dpi=130)
    plt.close(fig)


def maps(seed):
    d = np.load(HERE/f"hdr_maps_s{seed}.npz")
    region, lo, c = d["region"], float(d["lo"]), float(d["c"])
    n = region.shape[0]
    iy, ix = np.nonzero(region)
    pad = int(1500/c)
    y0, y1, x0, x1 = max(iy.min() - pad, 0), min(iy.max() + pad, n), max(ix.min() - pad, 0), min(ix.max() + pad, n)
    half = max(y1 - y0, x1 - x0)//2
    ym, xm = (y0 + y1)//2, (x0 + x1)//2
    y0, y1, x0, x1 = max(ym - half, 0), min(ym + half, n), max(xm - half, 0), min(xm + half, n)
    ext = [(lo + x0*c)/1000, (lo + x1*c)/1000, (lo + y0*c)/1000, (lo + y1*c)/1000]
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for i, (m, name) in enumerate((("sarops", "SAROPS"), ("greedy", "greedy"))):
        for j, N in enumerate(NS):
            done = d[f"{m}_{N}"][y0:y1, x0:x1]
            reg = region[y0:y1, x0:x1]
            img = np.ones(reg.shape + (3,))*np.array(to_rgb(SURFACE))
            img[reg] = to_rgb(REGION)
            img[done & ~reg] = to_rgb(OUTSIDE)
            img[done & reg] = to_rgb(IN_REGION)
            ax = axs[i, j]
            ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
            r = ROWS[(seed, m, N)]
            ax.set_title(f"{name}, {uav_label(N)}: {r['ratio']:.0%} of region swept, POS {r['hold']:.3f}")
            ax.tick_params(length=0)
            if j == 0:
                ax.set_ylabel("km")
            if i == 1:
                ax.set_xlabel("km")
    keys = [Patch(color=REGION, label="85% region, not swept"), Patch(color=IN_REGION, label="85% region, swept"),
            Patch(color=OUTSIDE, label="Swept outside the region")]
    fig.legend(handles=keys, loc="outside lower center", ncol=3)
    fig.suptitle(f"Scenario {seed}: where the sensors swept during the whole flight, against the 85% region at 60 min "
                 f"({ROWS[(seed, 'greedy', 1)]['region_km2']:.0f} km²)", x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/f"hdr_map_s{seed}.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    summary()
    maps(402)
    maps(404)
