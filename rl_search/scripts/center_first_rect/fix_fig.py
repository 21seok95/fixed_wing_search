import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
from rl_search.mission import ParticleBelief
from rl_search.patterns import _rot
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
from rl_search.scripts.plot_center_terrain import terrain_rgb
rows = json.load(open(sys.argv[1])); out1, out2 = sys.argv[2], sys.argv[3]
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
V = (("baseline", "Baseline", AXIS), ("start peak at intercept", "Start peak at the lanes", "#2a78d6"), ("neighbouring lanes", "Neighbouring lanes", "#eb6834"), ("both fixes", "Both fixes", "#1baf7a"))
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
# empty rectangles: particles inside 10 min after the plan
B = Path("rl_search/data/center_open")
def load(setting, tag):
    p = B/setting/"results.jsonl" if tag is None else B/"first_fix"/tag/setting/"results.jsonl"
    return {r["seed"]: r for r in map(json.loads, p.read_text().splitlines()) if r["n"] == 5 and r["config"] == "score" and r["x"] == 70 and r["w"] == 1}
def inr(P, q):
    c, s = np.cos(q[5]), np.sin(q[5]); dx, dy = P[:, 0]-q[1], P[:, 1]-q[2]
    return (np.abs(dx*c+dy*s) <= q[3]/2) & (np.abs(-dx*s+dy*c) <= q[4]/2)
bel = {}
for s in range(500, 510):
    env = make_envs(1, s)[0]; scen = open_scenario(env.sampler, s); bel[s] = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
# ---------- figure 1: metrics
se = lambda d: float(np.std(d, ddof=1)/np.sqrt(len(d))) if len(d) > 1 else 0.0
fig, axs = plt.subplots(2, 6, figsize=(17, 7.6), constrained_layout=True)
for i, (sname, setting) in enumerate((("No constraint", "no constraint"), ("Both rules", "both rules"))):
    base = {r["seed"]: r for r in rows if r["setting"] == setting and r["variant"] == "baseline"}
    G = [[r for r in rows if r["setting"] == setting and r["variant"] == v] for v, _, _ in V]
    mets = (("First rectangles touching\nanother current one", lambda g: np.mean([p["conflict"] for r in g for p in r["plans"] if p["i"] == 0 and p["u"] > 0]), "pct"),
            ("Share of a first rectangle\non another one", lambda g: np.mean([p["ov"] for r in g for p in r["plans"] if p["i"] == 0 and p["u"] > 0]), "pct"),
            ("Particles kept in the first\nrectangle, lane end vs start", lambda g: np.mean([f["in1"] for r in g for f in r["first"]])/np.mean([f["in0"] for r in g for f in r["first"]]), "pct"),
            ("Pair within 300 m\n(s per flight)", lambda g: np.mean([r["below"] for r in g]), "int"))
    for j, (t, f, kind) in enumerate(mets):
        ax = axs[i, j]
        vals = [float(f(g)) for g in G]
        ax.bar(range(4), vals, width=.62, color=[c for _, _, c in V])
        for k, x in enumerate(vals):
            ax.annotate(f"{x:.0%}" if kind == "pct" else f"{x:.0f}", (k, x), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9)
        ax.set(xticks=[], yticks=[], ylim=(0, max(vals)*1.22 or 1))
        ax.set_title(t if i == 0 else ""); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
        if j == 0: ax.set_ylabel(sname, fontsize=11, color=INK)
    for j, (key, t) in enumerate((("pos", "POS against the baseline\n(paired, ± 1 SE)"), ("A", "A (x 70) against the baseline\n(paired, ± 1 SE)"))):
        ax = axs[i, 4 + j]
        for k, g in enumerate(G[1:], start=1):
            d = [r[key] - base[r["seed"]][key] for r in g]
            ax.errorbar(np.mean(d), k, xerr=se(d), fmt="o", color=V[k][2], ms=8, capsize=3, lw=1.4)
            ax.annotate(f"{np.mean(d):+.3f}", (np.mean(d), k), xytext=(0, 8), textcoords="offset points", ha="center", color=SEC, fontsize=9)
        ax.axvline(0, color=AXIS, lw=1)
        lim = .05
        ax.set(xlim=(-lim, lim), ylim=(3.6, .4), yticks=[]); ax.set_title(t if i == 0 else "")
        ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
fig.get_layout_engine().set(rect=(0, 0, 1, .9))
fig.legend(handles=[Patch(color=c, label=l) for _, l, c in V], loc="center", bbox_to_anchor=(.5, .935), ncol=4, fontsize=10)
fig.text(.01, .975, "First-rectangle fixes: 5 UAVs, A + POS, x 70, w 1, scenarios 500-509, held-out particles. First rectangles = UAV 1-4 (UAV 0 plans with nothing committed)", ha="left", fontsize=12)
fig.savefig(out1, dpi=108); plt.close(fig)
# ---------- figure 2: map, scenario 504, both rules
SEED, T = 504, 1020.0
env = make_envs(1, SEED)[0]; b = bel[SEED]
rgb, ext = terrain_rgb(env.terr), [env.terr.axis[0]/1000, env.terr.axis[-1]/1000]*2
def corners(q):
    c = np.array([[sx*q[3]/2, sy*q[4]/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return (np.array([q[1], q[2]]) + c @ _rot(q[5]).T)/1000
runs = (("Baseline", load("no_overlap", None)[SEED]), ("Both fixes", load("no_overlap", "intercept_adjacent")[SEED]))
P = b.positions(T)
allc = np.concatenate([corners(r["rects"][u][0]) for _, r in runs for u in range(5)])
mid = np.median(P, axis=0)/2000 + allc.mean(0)/2; half = 4.6
fig, axs = plt.subplots(1, 2, figsize=(13.5, 8.2), constrained_layout=True, sharey=True)
fig.get_layout_engine().set(rect=(0, .08, 1, .86))
cols = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
for ax, (name, r) in zip(axs, runs):
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    ax.scatter(P[:, 0]/1000, P[:, 1]/1000, s=4, color=MUTED, linewidths=0)
    for u in range(5):
        q = r["rects"][u][0]
        ax.add_patch(Polygon(corners(q), fill=False, ec=cols[u], lw=2.2))
        tt, xy = np.array(r["tt"][u]), np.array(r["xy"][u])
        sel = tt <= T
        ax.plot(xy[sel, 0]/1000, xy[sel, 1]/1000, color=cols[u], lw=.9)
    ax.plot(*np.array(r["lkp"])/1000, "*", color=INK, ms=12, mec="white", zorder=6)
    ax.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half), xlabel="km")
    ax.set_title(f"{name}: POS {r['pos']:.3f}, pair within 300 m {r['below']:.0f} s"); ax.tick_params(length=0)
axs[0].set_ylabel("km")
fig.legend(handles=[Line2D([], [], color=c, lw=2.2, label=f"UAV {u} first rectangle and track") for u, c in enumerate(cols)] + [Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="Particles"), Line2D([], [], marker="*", ls="", color=INK, mec="white", ms=11, label="LKP")],
           loc="lower center", ncol=4, fontsize=9)
fig.text(.01, .955, f"Scenario {SEED}, both rules, 5 UAVs, A + POS, x 70: first rectangles with particles at {T/60:.0f} min (about the middle of the first lanes)", ha="left", fontsize=12)
fig.savefig(out2, dpi=105)
