"""Rectangle size and particles inside, baseline against both fixes (both rules), and when the intercept start peak is
read against the actual lane middle (unweighted belief, approximation). Saved flights only.

    python rl_search/scripts/center_first_rect/size_check.py rl_search/data/center_open/first_fix/size_check.png
"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rl_search.mission import ParticleBelief
from rl_search.patterns_v2 import find_clusters
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
rows = json.load(open("rl_search/data/center_open/first_rect/fix_rows.json"))
g = {v: [x for r in rows if r["setting"] == "both rules" and r["variant"] == v for x in r["first"]] for v in ("baseline", "both fixes")}
R = {r["seed"]: r for r in map(json.loads, open("rl_search/data/center_open/first_fix/intercept_adjacent/no_overlap/results.jsonl"))}
gap = []
for s in range(500, 510):
    env = make_envs(1, s)[0]; scen = open_scenario(env.sampler, s); env.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
    r = R[s]
    for u in range(5):
        q, k = r["rects"][u][0], r["tasks"][u][0]
        if not k["lane_t"]: continue
        tt, xy = np.array(r["tt"][u]), np.array(r["xy"][u]); x = np.array([np.interp(q[0], tt, xy[:, 0]), np.interp(q[0], tt, xy[:, 1])])
        tau = 0.0
        for _ in range(5):
            cl = find_clusters(env, q[0] + tau + 300.0, k=5, smooth=250.0, sep=1500.0)
            if not cl: break
            new = min(float(np.hypot(*(cl[0].peak - x)))/15.0, 1200.0); stop = abs(new - tau) < 30.0; tau = new
            if stop: break
        gap.append(((q[0] + tau + 300) - (k["lane_t"] + k["end"])/2)/60)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
cols = {"baseline": AXIS, "both fixes": "#1baf7a"}
ax = axs[0]
for j, (lab, f) in enumerate((("UAV 0", lambda x: x["u"] == 0), ("UAV 1-4", lambda x: x["u"] > 0))):
    for i, v in enumerate(("baseline", "both fixes")):
        a = np.mean([x["area"] for x in g[v] if f(x)])
        ax.bar(j + (i - .5)*.36, a, .34, color=cols[v])
        ax.annotate(f"{a:.2f}", (j + (i - .5)*.36, a), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
ax.axhline(2.37, color=MUTED, lw=1); ax.text(-.45, 2.41, "one pass filling 10 min (2.37 km²)", ha="left", va="bottom", color=SEC, fontsize=9)
ax.set(xticks=[0, 1], xticklabels=["UAV 0", "UAV 1-4"], ylabel="First rectangle area (km²)", ylim=(0, 2.8)); ax.set_title("Size of the first rectangles")
ax = axs[1]
for j, (lab, f) in enumerate((("UAV 0", lambda x: x["u"] == 0), ("UAV 1-4", lambda x: x["u"] > 0))):
    for i, v in enumerate(("baseline", "both fixes")):
        st, en = np.mean([x["in0"] for x in g[v] if f(x)]), np.mean([x["in1"] for x in g[v] if f(x)])
        xx = j*2.4 + i*1.0
        ax.plot([xx, xx + .6], [st, en], color=cols[v], lw=2.4, marker="o", ms=7)
        ax.annotate(f"{st:.0%}", (xx, st), xytext=(-4, 5), textcoords="offset points", ha="right", color=SEC, fontsize=9)
        ax.annotate(f"{en:.0%}", (xx + .6, en), xytext=(4, 5), textcoords="offset points", ha="left", color=SEC, fontsize=9)
ax.set(xticks=[.8, 3.2], xticklabels=["UAV 0", "UAV 1-4 (mean of one)"], ylabel="Particles inside (share of all)", ylim=(0, .34), xlim=(-.6, 4.6))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.set_title("Particles inside, lane start to lane end")
ax = axs[2]
ax.hist(gap, bins=np.arange(-4, 9, 1.0), color=SEC, edgecolor=SURFACE)
ax.axvline(0, color=INK, lw=1); ax.axvline(np.mean(gap), color="#1baf7a", lw=2)
ax.annotate(f"mean {np.mean(gap):+.1f} min", (np.mean(gap), ax.get_ylim()[1]*.92), xytext=(5, 0), textcoords="offset points", color=SEC, fontsize=9.5)
ax.set(xlabel="Start peak read time minus actual lane middle (min)", ylabel="First rectangles"); ax.set_title("Intercept peaks are read later than the lanes run")
for a in axs:
    a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False); a.tick_params(length=0)
fig.legend(handles=[Patch(color=c, label=l.capitalize()) for l, c in cols.items()], loc="outside upper right", ncol=2)
fig.suptitle("First rectangles, both rules, 5 UAVs, A + POS, x 70, scenarios 500-509 (read time: unweighted belief, approximation)", x=.01, ha="left", fontsize=12)
fig.savefig(sys.argv[1], dpi=110)
print("gap mean", round(float(np.mean(gap)), 2), "n", len(gap))
