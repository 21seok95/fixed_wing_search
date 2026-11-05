import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
D = [json.loads(l) for l in open(sys.argv[1])]; out_stats, out_map = sys.argv[2], sys.argv[3]
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
R = [r for d in D if d["mode"] == "free" and d["config"] == "pos" for r in d["rows"]]
G = (("First rectangle", lambda r: r["i"] == 0 and r["u"] > 0), ("Second", lambda r: r["i"] == 1), ("Third or later", lambda r: r["i"] >= 2))
groups = [(n, [r for r in R if f(r)]) for n, f in G]
def clean(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0)
fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
top, bot = fig.subfigures(2, 1, height_ratios=[1.75, 1])
A_ = top.subplots(1, 3)
B_ = bot.subplots(1, 7)
# a: overlap under other beliefs
ax = A_[0]
rows4 = (("No commitment known", "no_commit"), ("As flown", None), ("Seen particles deleted", "seen_removed"), ("Everything through the rectangle deleted", "zone_removed"))
y, ticks, labels = 0, [], []
for name, g in groups:
    c = [r for r in g if "cf" in r]
    ax.text(0, y - .05, name, fontsize=10, color=INK, fontweight="bold", va="center"); y += .85
    for lab, k in rows4:
        v = float(np.mean([r["ov"] for r in c])) if k is None else float(np.mean([r["cf"][k]["ov"] for r in c if r["cf"].get(k)]))
        ax.barh(y, v, height=.68, color=BLUE if k is None else AXIS)
        ax.annotate(f"{v:.0%}", (v, y), xytext=(5, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
        ticks.append(y); labels.append(lab); y += .85
    y += .35
ax.set(yticks=ticks, yticklabels=labels, ylim=(y - .7, -.6), xlim=(0, 1), xlabel="New rectangle on another current rectangle")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Same plan, other beliefs given to the planner"); clean(ax)
# b: what is left on the overlapped ground
ax = A_[1]
parts = (("Seen by the other UAV, weight x 0.3 per pass", "zone_seen", ORANGE), ("Crossed its rectangle during its lanes, not seen", "zone_unseen_in", AQUA), ("Not in its rectangle during its lanes", "zone_unseen_out", YELLOW))
for i, (name, g) in enumerate(groups):
    z = [r for r in g if r["conflict"]]
    vals = [np.mean([r[k] for r in z]) for _, k, _ in parts]; tot = sum(vals); left = 0.0
    for (lab, k, col), v in zip(parts, vals):
        s = v/tot
        ax.barh(i, s, left=left, height=.5, color=col, edgecolor=SURFACE, lw=1.4)
        if s > .06: ax.text(left + s/2, i, f"{s:.0%}", ha="center", va="center", color=INK if col == YELLOW else "#ffffff", fontsize=9.5)
        left += s
ax.set(yticks=range(3), yticklabels=[n for n, _ in groups], ylim=(2.6, -.6), xlim=(0, 1), xlabel="Probability left on the overlapped ground")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("What is left on the overlapped ground"); clean(ax)
ax.legend(handles=[Patch(color=c, label=l) for l, _, c in parts], loc="upper center", bbox_to_anchor=(.45, -.11), fontsize=9, ncol=1)
# c: forced events under the rule
ax = A_[2]
F = [r for d in D if d["mode"] == "rule" and d["config"] == "pos" for r in d["rows"] if r["kind"] == "forced"]
a = sum(r["none"] for r in F)
b = sum((not r["none"]) and any(r["clear_plain"]) for r in F)
c = sum((not r["none"]) and not any(r["clear_plain"]) for r in F)
vals = [a, c, b]; labs = ["Nothing in reach yet,\nkeeps approaching\n(no rectangle flown)", "Rectangle flown: peaks away\nfrom the others are\nout of reach", "Rectangle flown: reachable\nclear peak, search\nblocked at the start"]
ax.barh(range(3), vals, height=.5, color=[AXIS, SEC, SEC])
for i, v in enumerate(vals):
    ax.annotate(str(v), (v, i), xytext=(5, 0), textcoords="offset points", va="center", color=SEC, fontsize=10)
ax.set(yticks=range(3), yticklabels=labs, ylim=(2.6, -.6), xlim=(0, 46), xlabel="Count (POS only, both rules)")
ax.set_title(f"The {len(F)} 'no clear rectangle' events"); clean(ax)
# bottom: first vs later
fr = lambda g, f: [f(r) for r in g]
FLOW = json.load(open(sys.argv[4]))
def flow_ratio(g):
    keys = {(r["seed"], r["u"], r["i"]) for r in g}
    h = [x for x in FLOW if (x["seed"], x["u"], x["i"]) in keys]
    return np.mean([x["in1"] for x in h])/np.mean([x["in0"] for x in h])
panels = (("Plan touches a\ncurrent rectangle", lambda g: np.mean([r["conflict"] for r in g]), "pct"),
          ("Of those, lanes\nat the same time", lambda g: np.mean([r["time_overlap"] for r in g if r["conflict"]]), "pct"),
          ("Particles left at lane\nend vs lane start", flow_ratio, "pct"),
          ("Top peak inside\nanother rectangle", lambda g: np.mean([r["peaks"][0]["inside"] for r in g if r["peaks"]]), "pct"),
          ("Mass within 2 km\nof the top peak", lambda g: np.mean([r["peaks"][0]["mass"]/(r["total_S"]*(1 - r["removed"])) for r in g if r["peaks"]]), "pct"),
          ("Transit to the\nrectangle (min)", lambda g: np.mean([r["tr"] for r in g])/60, "num"),
          ("85% region during\nlanes (km²)", lambda g: np.mean([r["hdr85_S"] for r in g]), "int"))
for j, (t, f, kind) in enumerate(panels):
    ax = B_[j]
    v = [float(f(g)) for _, g in groups]
    ax.bar(range(3), v, width=.56, color=[BLUE, AXIS, AXIS])
    top = max(v)
    for i, x in enumerate(v):
        txt = f"{x:.0%}" if kind == "pct" else f"{x:.1f}" if kind == "num" else f"{x:.0f}"
        ax.annotate(txt, (i, x), xytext=(0, 4), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
    ax.set(xticks=range(3), xticklabels=["First", "Second", "Third+"], ylim=(0, top*1.2), yticks=[])
    ax.set_title(t); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
fig.suptitle("Why first rectangles overlap: reruns identical to the saved flights (POS only, 5 UAVs, no constraint unless stated, scenarios 500-509). First = UAV 1-4", x=.01, ha="left", fontsize=12)
fig.savefig(out_stats, dpi=115)
plt.close(fig)
# map
from pathlib import Path
from rl_search.mission import ParticleBelief
from rl_search.patterns import _rot
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
from rl_search.scripts.plot_center_terrain import terrain_rgb
SEED = 508
r = next(x for x in map(json.loads, open("rl_search/data/center_open/sweep/results.jsonl")) if x["seed"] == SEED and x["n"] == 5 and x["config"] == "pos")
env = make_envs(1, SEED)[0]
scen = open_scenario(env.sampler, SEED)
env.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
b = env.belief; N = b.n
xy, tt = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]]
ra, rb = r["rects"][0][0], r["rects"][1][0]
t_end = r["rects"][0][1][0]
w = np.ones(N); st = (np.zeros(N), np.full(N, -np.inf))
for k in range(len(tt[0]) - 1):
    if tt[0][k + 1] > t_end: break
    env._detect_lrc(xy[0][k:k + 2], tt[0][k:k + 2], w, st)
seen = w < 1
print("map: seen by UAV 0 during its first rectangle", round(float(seen.mean()), 3))
row = next(x for d in D if d["mode"] == "free" and d["config"] == "pos" and d["seed"] == SEED for x in d["rows"] if x["u"] == 1 and x["i"] == 0)
t0a = ra[0] + next(x for d in D if d["mode"] == "free" and d["config"] == "pos" and d["seed"] == SEED for x in d["rows"] if x["u"] == 0 and x["i"] == 0)["tr"]
t0b = rb[0] + row["tr"]
times = (t0a, t0b + 300, t_end)
def in_r(P, q):
    c, s = np.cos(q[5]), np.sin(q[5]); dx, dy = P[:, 0]-q[1], P[:, 1]-q[2]
    return (np.abs(dx*c+dy*s) <= q[3]/2) & (np.abs(-dx*s+dy*c) <= q[4]/2)
def corners(q):
    c = np.array([[sx*q[3]/2, sy*q[4]/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return (np.array([q[1], q[2]]) + c @ _rot(q[5]).T)/1000
rgb, ext = terrain_rgb(env.terr), [env.terr.axis[0]/1000, env.terr.axis[-1]/1000]*2
mid = (np.array([ra[1], ra[2]]) + np.array([rb[1], rb[2]]))/2000
half = 4.2
fig, axs = plt.subplots(1, 3, figsize=(16, 6.9), constrained_layout=True, sharey=True)
fig.get_layout_engine().set(rect=(0, .055, 1, .895))
titles = ("UAV 0 starts its lanes", "UAV 1 halfway through its lanes", "UAV 0 finishes its lanes")
for ax, t, name in zip(axs, times, titles):
    P = b.positions(t); act = b.active(t)
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    ax.scatter(P[act & seen, 0]/1000, P[act & seen, 1]/1000, s=5, color=SEC, linewidths=0, label="_")
    ax.scatter(P[act & ~seen, 0]/1000, P[act & ~seen, 1]/1000, s=5, color=BLUE, linewidths=0, label="_")
    ax.add_patch(Polygon(corners(ra), fill=False, ec=INK, lw=2))
    ax.add_patch(Polygon(corners(rb), fill=False, ec=ORANGE, lw=2))
    for u, col in ((0, INK), (1, ORANGE)):
        sel = tt[u] <= t
        ax.plot(xy[u][sel, 0]/1000, xy[u][sel, 1]/1000, color=col, lw=.9)
        ax.plot(xy[u][sel, 0][-1]/1000, xy[u][sel, 1][-1]/1000, "o", color=col, ms=7, mec="white", mew=1.1, zorder=5)
    ov = in_r(P, ra) & in_r(P, rb) & act
    wt = np.where(seen, .3, 1.0)
    share = wt[ov & ~seen].sum()/max(wt[ov].sum(), 1e-9)
    ax.plot(*np.array(r["lkp"])/1000, "*", color=INK, ms=12, mec="white", zorder=6)
    ax.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half), xlabel="km")
    ax.set_title(f"{t/60:.1f} min: {name}\nOn the overlap, {share:.0%} of what is left is unseen")
    ax.tick_params(length=0)
axs[0].set_ylabel("km")
keys = [Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label="Particle UAV 0's first rectangle never sees"), Line2D([], [], marker="o", ls="", color=SEC, ms=5, label="Particle it sees (weight x 0.3 per pass)"),
        Patch(fill=False, ec=INK, lw=2, label="UAV 0's first rectangle, planned at 2.0 min"), Patch(fill=False, ec=ORANGE, lw=2, label="UAV 1's first rectangle, planned at 3.0 min"),
        Line2D([], [], color=INK, lw=.9, label="Tracks so far"), Line2D([], [], marker="*", ls="", color=INK, mec="white", ms=11, label="LKP")]
fig.legend(handles=keys, loc="lower center", ncol=6, fontsize=9)
fig.text(.01, .965, f"Scenario {SEED}, POS only, 5 UAVs, no constraint: particles keep flowing through a rectangle that is already committed. Same scale in all panels", ha="left", fontsize=12)
fig.savefig(out_map, dpi=105)
