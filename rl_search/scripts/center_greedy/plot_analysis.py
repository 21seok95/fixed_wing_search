"""Figures for the analyses on scenarios 400-409 that had only tables: flights scored against other target models, and
planning POD against true POD. From the saved results, no rerun.

    python rl_search/scripts/center_greedy/plot_analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parents[3]/"rl_search/data/center_greedy"
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SAR, GRE = "#2a78d6", "#eb6834"
SEEDS, NS = list(range(400, 410)), (1, 3, 5)
ORDER = ("nominal", "mostly hiding", "always moving", "speed x0.5", "speed x1.5", "heading +45", "heading -45", "heading +90",
         "heading -90", "heading 180", "heading unknown", "LKP error 1 km")
NAMES = {"nominal": "As planned", "mostly hiding": "Mostly hiding", "always moving": "Always moving", "speed x0.5": "Speed x0.5",
         "speed x1.5": "Speed x1.5", "heading +45": "Heading +45 deg", "heading -45": "Heading -45 deg", "heading +90": "Heading +90 deg",
         "heading -90": "Heading -90 deg", "heading 180": "Heading opposite", "heading unknown": "Heading unknown", "LKP error 1 km": "LKP error 1 km"}
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED,
                     "ytick.color": MUTED, "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})


def keys(loc):
    return [Line2D([], [], color=SAR, lw=1.8, marker="o", ms=6, mec=SURFACE, label="SAROPS"),
            Line2D([], [], color=GRE, lw=1.8, marker="o", ms=6, mec=SURFACE, label="greedy")]


def mismatch():
    rows = json.load(open(HERE/"mismatch_replay.json"))
    fig, axs = plt.subplots(1, 3, figsize=(14, 6.4), sharey=True, constrained_layout=True)
    for ax, n in zip(axs, NS):
        for y, t in enumerate(ORDER):
            a = np.mean([r["pend"] for r in rows if r["truth"] == t and r["method"] == "sarops" and r["n"] == n])
            b = np.mean([r["pend"] for r in rows if r["truth"] == t and r["method"] == "greedy" and r["n"] == n])
            ax.plot([a, b], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(a, y, s=50, color=SAR, edgecolor=SURFACE, lw=1.4, zorder=3)
            ax.scatter(b, y, s=50, color=GRE, edgecolor=SURFACE, lw=1.4, zorder=3)
        ax.axhline(2.5, color=AXIS, lw=.8)
        ax.set(xlim=(0, 1), ylim=(len(ORDER) - .5, -.5), yticks=range(len(ORDER)), yticklabels=[NAMES[t] for t in ORDER], xlabel="Mean POS at end of flight")
        ax.set_title(f"{n} UAV" + ("s" if n > 1 else ""))
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=keys(None), loc="outside lower center", ncol=2)
    fig.suptitle("Flights planned on one target model, scored on others: both methods lose about the same (above the line: as planned and in-motion changes)",
                 x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/"mismatch.png", dpi=120)
    plt.close(fig)


def plan_pod():
    rows = json.load(open(HERE/"plan_pod_scores.json"))
    by = {(r["seed"], r["method"], r["n"], r["plan_pod"]): r for r in rows}
    true = ("0.3", "0.5", "0.7", "0.9")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, n in zip(axs, NS):
        for m, c in (("sarops", SAR), ("greedy", GRE)):
            d = np.array([[by[(s, m, n, .3)]["scored"][t]["end"] - by[(s, m, n, .7)]["scored"][t]["end"] for t in true] for s in SEEDS])
            mean, se = d.mean(0), d.std(0, ddof=1)/np.sqrt(len(SEEDS))
            x = [float(t) for t in true]
            ax.fill_between(x, mean - se, mean + se, color=c, alpha=.1, lw=0)
            ax.plot(x, mean, color=c, lw=1.8, marker="o", ms=6, mec=SURFACE)
        ax.axhline(0, color=SEC, lw=1)
        ax.set(xlim=(.28, .92), xticks=[.3, .5, .7, .9], xlabel="True POD")
        ax.set_title(f"{n} UAV" + ("s" if n > 1 else ""))
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("POS, planned with POD 0.3 minus POD 0.7 (band: 1 se)")
    fig.legend(handles=keys(None), loc="outside lower center", ncol=2)
    fig.suptitle("Planning with a low POD pays only when the true POD is that low (old scenarios 400-409)", x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/"plan_pod.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    mismatch()
    plan_pod()
