import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
from rl_search.mission import ParticleBelief
from rl_search.patterns import _rot
from rl_search.patterns_v2 import find_clusters
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
from rl_search.scripts.plot_center_terrain import terrain_rgb
E = json.load(open(sys.argv[1])); D = [json.loads(l) for l in open(sys.argv[2])]; out1, out2 = sys.argv[3], sys.argv[4]
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
def clean(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0)
def pm(g, k): return float(np.mean([x[k] for x in g if x.get(k) is not None]))
B = E["B"]; groups = (("First", [x for x in B if x["i"] == 0]), ("Second", [x for x in B if x["i"] == 1]), ("Third+", [x for x in B if x["i"] >= 2]))
# ---------- figure 1
fig = plt.figure(figsize=(16.5, 8.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.75, 1.0])
# density
ax = fig.add_subplot(gs[0, 0])
A = E["A"]
vals = [pm(A, "over_before"), pm(A, "over_after"), pm(A, "rest"), pm(A, "region")]
labs = ["Overlap, before\nthe weights", "Overlap, after\nthe weights", "Rest of the\nnew rectangle", "85% region\naverage"]
ax.barh(range(4), vals, height=.56, color=[AXIS, BLUE, SEC, AXIS])
for i, v in enumerate(vals):
    ax.annotate(f"{v*100:.1f}%", (v, i), xytext=(5, 0), textcoords="offset points", va="center", color=SEC, fontsize=9.5)
ax.set(yticks=range(4), yticklabels=labs, ylim=(3.6, -.6), xlim=(0, .08), xlabel="Probability per km² during the new lanes")
ax.xaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
ax.set_title(f"Weights do lower the overlap ({len(A)} first plans)"); clean(ax)
# dwell vs sweep
ax = fig.add_subplot(gs[1, 0])
x = np.arange(3); wd = .36
sw = [pm(g, "dur")/60 for _, g in groups]; dw = [pm(g, "dwell")/60 for _, g in groups]
ax.bar(x - wd/2, sw, wd*.94, color=SEC, label="Time to fly the lanes")
ax.bar(x + wd/2, dw, wd*.94, color=BLUE, label="Time a particle stays inside")
for i in range(3):
    ax.annotate(f"{sw[i]:.1f}", (i - wd/2, sw[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
    ax.annotate(f"{dw[i]:.1f}", (i + wd/2, dw[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
ax.set(xticks=x, xticklabels=[n for n, _ in groups], ylabel="Minutes", ylim=(0, 15.5))
ax.set_title("Sweep time against stay time"); ax.legend(loc="upper right", fontsize=9); clean(ax, "y")
# time-space example
ax = fig.add_subplot(gs[:, 1])
ex = E["example"]; t = np.array(ex["t"])/60; y = np.array(ex["y"]); seen = np.array(ex["seen"]); hw = ex["wid"]/2
Rw = 143.0
su = np.array(ex["su"])
ax.fill_between(t, (su-Rw)/1000, (su+Rw)/1000, color=YELLOW, alpha=.35, lw=0, zorder=1)
for j in np.flatnonzero(~seen):
    ax.plot(t, y[:, j]/1000, color=BLUE, lw=.8, alpha=.75, zorder=2)
for j in np.flatnonzero(seen):
    ax.plot(t, y[:, j]/1000, color=MUTED, lw=.8, alpha=.8, zorder=2)
ax.plot(t, su/1000, color=INK, lw=1.3, zorder=3)
ax.axhline(hw/1000, color=INK, lw=1); ax.axhline(-hw/1000, color=INK, lw=1)
ax.axvline(ex["t0"]/60, color=AXIS, lw=1); ax.axvline(ex["t1"]/60, color=AXIS, lw=1)
ax.set(xlim=(t[0], t[-1]), ylim=(-hw/1000*1.9, hw/1000*1.9), xlabel="Minutes after the LKP report", ylabel="Position across the lanes (km)")
ax.set_title(f"Scenario 508, UAV 0, first rectangle: particles inside at the start of its lanes ({len(seen)} sampled)")
ax.legend(handles=[Line2D([], [], color=INK, lw=1.3, label="UAV across the lanes"), Patch(color=YELLOW, alpha=.35, label="Sensor strip, 143 m each side"), Line2D([], [], color=BLUE, lw=1, label="Particle never seen"),
                   Line2D([], [], color=MUTED, lw=1, label="Particle seen"), Line2D([], [], color=INK, lw=1, label="Rectangle edges (thin vertical: lanes start, end)")], loc="lower left", fontsize=9, ncol=2)
clean(ax, "y")
# frozen vs moving
ax = fig.add_subplot(gs[0, 2])
fz = [pm(g, "frz_A0") for _, g in groups]; mv = [pm(g, "mov_A0") for _, g in groups]
ax.bar(x - wd/2, fz, wd*.94, color=AXIS, label="Same path, particles frozen")
ax.bar(x + wd/2, mv, wd*.94, color=BLUE, label="Same path, particles moving")
for i in range(3):
    ax.annotate(f"{fz[i]:.0%}", (i - wd/2, fz[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
    ax.annotate(f"{mv[i]:.0%}", (i + wd/2, mv[i]), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
ax.set(xticks=x, xticklabels=[n for n, _ in groups], ylim=(0, 1.35), yticks=[])
ax.spines["left"].set_visible(False)
ax.set_title("Particles inside at lane start that reach the sensor"); ax.legend(loc="upper right", fontsize=9); clean(ax, "y")
# what the missed did
ax = fig.add_subplot(gs[1, 2])
parts = (("Stayed ahead of the sweep, left", "ahead", ORANGE), ("Inside the strip, UAV elsewhere on the lane", "band", AQUA), ("Came in behind the sweep", "behind", YELLOW))
for i, (n, g) in enumerate(groups):
    tot = sum(z["miss"] for z in g); left = 0.0
    for lab, k, col in parts:
        s = sum(z[k]*z["miss"] for z in g)/tot
        ax.barh(i, s, left=left, height=.5, color=col, edgecolor=SURFACE, lw=1.4)
        if s > .08: ax.text(left+s/2, i, f"{s:.0%}", ha="center", va="center", color=INK if col == YELLOW else "#ffffff", fontsize=9.5)
        left += s
ax.set(yticks=range(3), yticklabels=[n for n, _ in groups], ylim=(2.6, -.6), xlim=(0, 1), xlabel="Share of the particles missed inside the rectangle")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("What the missed particles did"); clean(ax)
ax.legend(handles=[Patch(color=c, label=l) for l, _, c in parts], loc="upper center", bbox_to_anchor=(.45, -.2), fontsize=9)
fig.suptitle("Why flying the lanes does not see the particles: saved flights, POS only, 5 UAVs, no constraint, scenarios 500-509 (First = all UAVs here). Seen = reached the sensor, before the 0.7 POD", x=.01, ha="left", fontsize=12)
fig.savefig(out1, dpi=112); plt.close(fig)
# ---------- peak reach over all scenarios and the example map
reach = []
ex_map = None
for d in D:
    if d["mode"] != "free" or d["config"] != "pos": continue
    seed = d["seed"]
    env = make_envs(1, seed)[0]; scen = open_scenario(env.sampler, seed)
    env.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True); b = env.belief; N = b.n
    s = next(x for x in map(json.loads, open("rl_search/data/center_open/sweep/results.jsonl")) if x["seed"] == seed and x["n"] == 5 and x["config"] == "pos")
    xy, tt = [np.array(p, float) for p in s["xy"]], [np.array(t, float) for t in s["tt"]]; rects = s["rects"]
    rows = {(r["u"], r["i"]): r for r in d["rows"]}
    for u in range(1, 5):
        q = rects[u][0]; tp = q[0]; r = rows[(u, 0)]
        w = np.ones(N); others = []
        for v in range(5):
            if v == u or rects[v][0][0] > tp or (rects[v][0][0] == tp and v > u): continue
            others.append(rects[v][0]); st = (np.zeros(N), np.full(N, -np.inf)); tend = rects[v][1][0] if len(rects[v]) > 1 else tt[v][-1]
            for k in range(len(tt[v])-1):
                if tt[v][k+1] > tend: break
                env._detect_lrc(xy[v][k:k+2], tt[v][k:k+2], w, st)
        b.w[:] = w
        pk5 = find_clusters(env, tp+300.0, k=3, smooth=250.0, sep=1500.0); pkm = find_clusters(env, tp+q[6]+r["tr"]+300.0, k=3, smooth=250.0, sep=1500.0)
        pose = np.array([np.interp(tp, tt[u], xy[u][:, 0]), np.interp(tp, tt[u], xy[u][:, 1])])
        if pkm:
            reach.append(dict(seed=seed, u=u, d5=float(np.hypot(*(pk5[0].peak-pose))), dm=float(np.hypot(*(pkm[0].peak-pose)))))
        if seed == 508 and u == 1:
            ex_map = dict(env=env, b=b, w=w.copy(), q=q, others=list(others), tp=tp, tm=tp+q[6]+r["tr"]+300.0, p5=pk5[0].peak.copy(), pm=pkm[0].peak.copy(), lkp=np.array(s["lkp"]), pose=pose, xy=xy, tt=tt)
        b.w[:] = 1.0
d5, dm = np.array([x["d5"] for x in reach]), np.array([x["dm"] for x in reach])
print("reach (9 km): plan+5 peak within", round(float((d5 <= 9000).mean()), 2), "mean km", round(float(d5.mean())/1000, 1), "; lane-mid top peak within", round(float((dm <= 9000).mean()), 2), "mean km", round(float(dm.mean())/1000, 1), "n", len(reach))
C = [x for x in E["C"] if x["u"] > 0 and "shift" in x]
fig = plt.figure(figsize=(16.5, 7.2), constrained_layout=True)
fig.get_layout_engine().set(rect=(0, .05, 1, .9))
gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.55, .85, .85])
m = ex_map; env = m["env"]; b = m["b"]
rgb, ext = terrain_rgb(env.terr), [env.terr.axis[0]/1000, env.terr.axis[-1]/1000]*2
def corners(q):
    c = np.array([[sx*q[3]/2, sy*q[4]/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return (np.array([q[1], q[2]]) + c @ _rot(q[5]).T)/1000
mid = np.array([m["q"][1], m["q"][2]])/1000 + np.array([.2, -.8]); half = 4.3
for j, (tv, pk, name) in enumerate(((m["tp"]+300.0, m["p5"], "plan + 5 min: where the rectangle starts"), (m["tm"], m["pm"], "middle of the lanes: where the UAV searches"))):
    ax = fig.add_subplot(gs[:, j])
    P = b.positions(tv); low = m["w"] < 1
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    ax.scatter(P[low, 0]/1000, P[low, 1]/1000, s=5, color=MUTED, linewidths=0)
    ax.scatter(P[~low, 0]/1000, P[~low, 1]/1000, s=5, color=BLUE, linewidths=0)
    for o in m["others"]:
        ax.add_patch(Polygon(corners(o), fill=False, ec=INK, lw=2))
    ax.add_patch(Polygon(corners(m["q"]), fill=False, ec=ORANGE, lw=2))
    ax.plot(pk[0]/1000, pk[1]/1000, "P", color=YELLOW, ms=13, mec=INK, mew=1, zorder=6)
    if j == 1:
        ax.plot(m["p5"][0]/1000, m["p5"][1]/1000, "P", color=SURFACE, ms=11, mec=INK, mew=1, zorder=6)
        ax.annotate("", xy=(pk[0]/1000, pk[1]/1000), xytext=(m["p5"][0]/1000, m["p5"][1]/1000), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3), zorder=5)
    ax.plot(*m["lkp"]/1000, "*", color=INK, ms=12, mec="white", zorder=6)
    ax.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half), xlabel="km", ylabel="km" if j == 0 else None)
    ax.set_title(f"{tv/60:.1f} min, {name}"); ax.tick_params(length=0)
def mini(ax, a, b_, title, fmt):
    ax.bar([0, 1], [a, b_], width=.56, color=[AXIS, BLUE])
    for i, v in enumerate((a, b_)):
        ax.annotate(fmt.format(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
    ax.set(xticks=[0, 1], xticklabels=["Plan + 5 min", "Lane middle"], yticks=[], ylim=(0, max(a, b_)*1.25 or 1)); ax.set_title(title)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
tin5, tinm = np.mean([x["top5_in"] for x in C]), np.mean([x["topm_in"] for x in C])
mini(fig.add_subplot(gs[0, 2]), tin5*100, tinm*100, "Top peak inside\nanother rectangle (%)", "{:.0f}%")
mini(fig.add_subplot(gs[0, 3]), pm(C, "mass5")*100, pm(C, "massm")*100, "Probability within 2 km\nof the top peak (%)", "{:.0f}%")
mini(fig.add_subplot(gs[1, 2]), pm(C, "n5"), pm(C, "nm"), "Peaks found\n(3 at most)", "{:.1f}")
mini(fig.add_subplot(gs[1, 3]), pm(C, "rect_c_to_5")/1000, pm(C, "rect_c_to_m")/1000, "Flown rectangle centre\nto the top peak (km)", "{:.1f}")
keys = [Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label="Particle the earlier lanes do not see"), Line2D([], [], marker="o", ls="", color=MUTED, ms=5, label="Particle they see (weight lowered)"),
        Patch(fill=False, ec=INK, lw=2, label="Earlier rectangle"), Patch(fill=False, ec=ORANGE, lw=2, label="This UAV's rectangle"), Line2D([], [], marker="P", ls="", color=YELLOW, mec=INK, ms=10, label="Top peak at that time"),
        Line2D([], [], marker="P", ls="", color=SURFACE, mec=INK, ms=9, label="Top peak at plan + 5 min"), Line2D([], [], marker="*", ls="", color=INK, mec="white", ms=11, label="LKP")]
fig.legend(handles=keys, loc="lower center", ncol=7, fontsize=9)
fig.text(.01, .955, "When the start peak is read: Scenario 508, UAV 1, first rectangle (left), and all first rectangles of UAV 1-4 (right). Saved flights, POS only, no constraint", ha="left", fontsize=12)
fig.savefig(out2, dpi=108)
