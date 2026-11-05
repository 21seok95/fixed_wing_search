import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
rows = json.load(open(sys.argv[1])); ex = json.load(open(sys.argv[2])); out = sys.argv[3]
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
pick = lambda s, n, c="score": [r for r in rows if r["setting"] == s and r["n"] == n and r["config"] == c]
fig = plt.figure(figsize=(16, 8.4), constrained_layout=True)
gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1, 1])
# A: where the double-swept area comes from
ax = fig.add_subplot(gs[:, 0])
PARTS = (("Launch and approach", ("launch",), "#2a78d6"), ("Moving between rectangles", ("move",), "#eb6834"),
         ("Lanes, both rectangles active", ("same_overlap", "same_apart"), "#1baf7a"), ("Lanes, after the other finished", ("later_overlap", "later_apart"), "#eda100"))
y, ticks, labels = 0, [], []
for s in ("no constraint", "time rule", "time + rectangle"):
    for n in (3, 5):
        g = pick(s, n); left = 0.0
        for name, keys, col in PARTS:
            v = float(np.mean([sum(r[k] for k in keys) for r in g]))
            ax.barh(y, v, left=left, height=.62, color=col, edgecolor=SURFACE, lw=1.4)
            if v > .6: ax.text(left + v/2, y, f"{v:.1f}", ha="center", va="center", color=INK if col == "#eda100" else "#ffffff", fontsize=9)
            left += v
        ax.annotate(f"{left:.1f} km²", (left, y), xytext=(6, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
        ticks.append(y); labels.append(f"{s}, {n} UAVs"); y += 1
    y += .6
ax.set(yticks=ticks, yticklabels=labels, ylim=(y - .9, -.7), xlim=(0, 14), xlabel="Area swept by 2+ UAVs (km²)")
ax.set_title("Where the double-swept area comes from (A + POS, x 70)")
ax.legend(handles=[Patch(color=c, label=n) for n, _, c in PARTS], loc="upper center", bbox_to_anchor=(.5, -.07), ncol=2, fontsize=9)
ax.grid(axis="x", color=GRID, lw=.8)
# B: rectangle conflict by order
ax = fig.add_subplot(gs[0, 1])
p = [x for r in pick("time + rectangle", 5) for x in r["plans"]]
cats = (("First", lambda x: x["i"] == 0), ("Second", lambda x: x["i"] == 1), ("Third or later", lambda x: x["i"] >= 2))
vals = [np.mean([x["c300"] for x in p if f(x)]) for _, f in cats]; cnt = [sum(1 for x in p if f(x)) for _, f in cats]; hit = [sum(x["c300"] for x in p if f(x)) for _, f in cats]
ax.barh(range(3), vals, height=.58, color=SEC)
for i, (v, h, c) in enumerate(zip(vals, hit, cnt)):
    ax.annotate(f"{h} of {c}", (v, i), xytext=(6, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
ax.set(yticks=range(3), yticklabels=[c[0] for c in cats], ylim=(2.6, -.6), xlim=(0, .5), xlabel="Plans within 300 m of another current rectangle")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Rectangle conflicts by planning order (5 UAVs)")
ax.grid(axis="x", color=GRID, lw=.8)
# C: close time by round
ax = fig.add_subplot(gs[1:, 1])
ck = (("Before a first rectangle", "before"), ("Both on first rectangle", "both_first"), ("One on first rectangle", "one_first"), ("Both on later rectangles", "later"))
for j, (s, col) in enumerate((("no constraint", AXIS), ("time + rectangle", SEC))):
    c = ex[f"{s}|5|score"]["close"]
    for i, (_, k) in enumerate(ck):
        yy = i + (j - .5)*.36
        ax.barh(yy, c[k], height=.32, color=col)
        ax.annotate(f"{c[k]:.0f} s", (c[k], yy), xytext=(5, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
ax.set(yticks=range(4), yticklabels=[c[0] for c in ck], ylim=(3.6, -.6), xlim=(0, 170), xlabel="Seconds with a pair within 300 m, per flight")
ax.set_title("When close time happens (5 UAVs)")
ax.legend(handles=[Patch(color=AXIS, label="No constraint"), Patch(color=SEC, label="Time + rectangle rule")], loc="lower right", fontsize=9)
ax.grid(axis="x", color=GRID, lw=.8)
# D: rectangle sizes do not follow team size
bins = np.arange(0, 3.01, .1)
for i, n in enumerate((1, 3, 5)):
    ax = fig.add_subplot(gs[i, 2])
    a = np.array(ex[f"no constraint|{n}|score"]["areas"])
    ax.hist(a, bins=bins, weights=np.ones_like(a)/len(a), color=SEC, edgecolor=SURFACE, lw=.8)
    ax.set(xlim=(0, 3), ylim=(0, .4))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(("Area: " if i == 0 else "") + f"{n} UAV{'s' if n > 1 else ''}, {len(a)} rectangles")
    ax.grid(axis="y", color=GRID, lw=.8)
    if i == 2: ax.set_xlabel("Rectangle area (km²), no constraint")
    else: ax.set_xticklabels([])
    if i == 0:
        for xp, t in ((.55, "3 passes"), (1.3, "2 passes"), (2.3, "1 pass")):
            ax.annotate(t, (xp, .37), ha="center", va="top", color=SEC, fontsize=9)
for ax in fig.axes:
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0)
fig.suptitle("Why swept area and rectangles still overlap: saved flights, SAROPS, scenarios 500-509, no planner rerun", x=.01, ha="left", fontsize=12)
fig.savefig(out, dpi=120)
