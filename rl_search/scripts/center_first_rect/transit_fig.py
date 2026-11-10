"""Transit limit lifted: baseline, both fixes with the 10 min limit, and both fixes with a 30 min limit (both rules, 5 UAVs,
A + POS, x 70, w 1, scenarios 500-509). Saved flights only.

    python rl_search/scripts/center_first_rect/transit_fig.py rl_search/data/center_open/first_fix/transit_metrics.png rl_search/data/center_open/first_fix/transit_map_504.png
"""
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

B = Path("rl_search/data/center_open")
RUNS = (("Baseline (10 min limit)", B/"no_overlap", "#c3c2b7"), ("Both fixes, 10 min limit", B/"first_fix/intercept_adjacent/no_overlap", "#1baf7a"),
        ("Both fixes, 30 min limit", B/"first_fix/intercept_adjacent_transit1800/no_overlap", "#6250d6"))
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})


def load(folder):
    return {r["seed"]: r for r in map(json.loads, (folder/"results.jsonl").read_text().splitlines())
            if r["n"] == 5 and r["config"] == "score" and r["x"] == 70 and r["w"] == 1}


def in_rect(q, r, m=0.0):
    c, s = np.cos(r[5]), np.sin(r[5]); dx, dy = q[..., 0]-r[1], q[..., 1]-r[2]
    return (np.abs(dx*c+dy*s) <= r[3]/2+m) & (np.abs(-dx*s+dy*c) <= r[4]/2+m)


def conflict(a, b):
    d = np.array([b[1]-a[1], b[2]-a[2]])
    ca, sa, cb, sb = np.cos(a[5]), np.sin(a[5]), np.cos(b[5]), np.sin(b[5])
    ua, va, ub, vb = np.array([ca, sa]), np.array([-sa, ca]), np.array([cb, sb]), np.array([-sb, cb])
    return all(abs(d @ n) <= a[3]/2*abs(ua @ n)+a[4]/2*abs(va @ n)+b[3]/2*abs(ub @ n)+b[4]/2*abs(vb @ n) for n in (ua, va, ub, vb))


def samples(r, nx=30, ny=12):
    u = (np.arange(nx)+.5)/nx*r[3]-r[3]/2; v = (np.arange(ny)+.5)/ny*r[4]-r[4]/2
    U, V = np.meshgrid(u, v); c, s = np.cos(r[5]), np.sin(r[5])
    return np.stack([r[1]+U*c-V*s, r[2]+U*s+V*c], -1)


def cur(rects, t):
    i = -1
    for k, r in enumerate(rects):
        if r[0] <= t: i = k
    return i


def corners(q):
    c = np.array([[sx*q[3]/2, sy*q[4]/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return (np.array([q[1], q[2]]) + c @ _rot(q[5]).T)/1000


bel = {}
for s in range(500, 510):
    env = make_envs(1, s)[0]; scen = open_scenario(env.sampler, s)
    bel[s] = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
data = [load(f) for _, f, _ in RUNS]
stats = []
for run in data:
    st = dict(conf=[], ov=[], area0=[], area14=[], in0_0=[], in0_14=[], in1_14=[], tr_first=[], tr_later=[], move=[], pos={}, A={}, below=[])
    for s, r in run.items():
        b = bel[s]
        xy, tt, rects = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]], r["rects"]
        st["pos"][s], st["A"][s] = r["pos"], r["scores"]["70"]["A"]; st["below"].append(r["below"])
        mv = tot = 0.0
        for u in range(5):
            q = rects[u][0]
            others = [rects[v][j] for v in range(5) if v != u for j in [cur(rects[v], q[0])] if j >= 0]
            if u > 0:
                st["conf"].append(any(conflict(q, o) for o in others))
                st["ov"].append(float(np.any([in_rect(samples(q), o) for o in others], axis=0).mean()) if others else 0.0)
            inside = np.flatnonzero((tt[u] >= q[0]) & in_rect(xy[u], q, 150.0))
            t0 = tt[u][inside[0]] if len(inside) else q[0]
            t1 = rects[u][1][0] if len(rects[u]) > 1 else tt[u][-1]
            (st["area0"] if u == 0 else st["area14"]).append(q[3]*q[4]/1e6)
            if u == 0:
                st["in0_0"].append(float(in_rect(b.positions(t0), q).mean()))
            else:
                st["in0_14"].append(float(in_rect(b.positions(t0), q).mean())); st["in1_14"].append(float(in_rect(b.positions(t1), q).mean()))
            # transit of each task: plan time to the first sample inside the rectangle
            for i, qq in enumerate(rects[u]):
                ins = np.flatnonzero((tt[u] >= qq[0]) & in_rect(xy[u], qq, 150.0))
                if len(ins):
                    (st["tr_first"] if i == 0 else st["tr_later"]).append((tt[u][ins[0]] - qq[0])/60)
            for k in range(len(tt[u]) - 1):
                tm = (tt[u][k] + tt[u][k+1])/2; i = cur(rects[u], tm); dt = tt[u][k+1] - tt[u][k]
                tot += dt
                if i >= 0 and not (tm >= rects[u][i][0] + rects[u][i][6] and in_rect((xy[u][k] + xy[u][k+1])/2, rects[u][i], 150.0)):
                    mv += dt
        st["move"].append(mv/tot)
    stats.append(st)
for (name, _, _), st in zip(RUNS, stats):
    print(f"{name}: first conflict {np.mean(st['conf']):.0%}, ov {np.mean(st['ov']):.0%}, close {np.mean(st['below']):.0f} s, POS {np.mean(list(st['pos'].values())):.3f}, A {np.mean(list(st['A'].values())):.3f}, "
          f"first transit {np.mean(st['tr_first']):.1f} min (max {np.max(st['tr_first']):.1f}), later {np.mean(st['tr_later']):.1f} (max {np.max(st['tr_later']):.1f}), moving share {np.mean(st['move']):.0%}, "
          f"UAV 1-4 in rect at start {np.mean(st['in0_14']):.1%}, area UAV 0 {np.mean(st['area0']):.2f} UAV 1-4 {np.mean(st['area14']):.2f}")
se = lambda d: float(np.std(d, ddof=1)/np.sqrt(len(d))) if len(d) > 1 else 0.0
fig, axs = plt.subplots(2, 5, figsize=(17, 8.4), constrained_layout=True)
fig.get_layout_engine().set(rect=(0, 0, 1, .9))
cols = [c for _, _, c in RUNS]
bars = (("First rectangles touching\nanother current one", lambda st: np.mean(st["conf"]), "pct"),
        ("Share of a first rectangle\non another one", lambda st: np.mean(st["ov"]), "pct"),
        ("Pair within 300 m\n(s per flight)", lambda st: np.mean(st["below"]), "int"),
        ("Particles in a UAV 1-4 first\nrectangle at lane start", lambda st: np.mean(st["in0_14"]), "pct1"),
        ("UAV 1-4 first rectangle\narea (km²)", lambda st: np.mean(st["area14"]), "f2"),
        ("Transit to the first\nrectangle (min)", lambda st: np.mean(st["tr_first"]), "f1"),
        ("Transit to later\nrectangles (min)", lambda st: np.mean(st["tr_later"]), "f1"),
        ("Share of flight time\nmoving or holding", lambda st: np.mean(st["move"]), "pct"))
fmt = {"pct": "{:.0%}", "pct1": "{:.1%}", "int": "{:.0f}", "f1": "{:.1f}", "f2": "{:.2f}"}
for j, (t, f, kind) in enumerate(bars):
    ax = axs.flat[j if j < 4 else j + 1]
    v = [float(f(st)) for st in stats]
    ax.bar(range(3), v, width=.62, color=cols)
    for k, x in enumerate(v):
        ax.annotate(fmt[kind].format(x), (k, x), xytext=(0, 3), textcoords="offset points", ha="center", color=SEC, fontsize=9.5)
    ax.set(xticks=[], yticks=[], ylim=(0, max(v)*1.25 or 1)); ax.set_title(t)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
ax = axs[0, 4]
base = stats[0]
for k, st in enumerate(stats[1:], start=1):
    for row, key in ((0, "pos"), (1, "A")):
        d = [st[key][s] - base[key][s] for s in st[key]]
        y = k + (row - .5)*.36
        ax.errorbar(np.mean(d), y, xerr=se(d), fmt="o" if row == 0 else "s", color=cols[k], ms=7, capsize=3, lw=1.4)
        ax.annotate(f"{'POS' if row == 0 else 'A'} {np.mean(d):+.3f}", (np.mean(d), y), xytext=(0, -9), textcoords="offset points", ha="center", color=SEC, fontsize=9)
ax.axvline(0, color=AXIS, lw=1)
ax.set(xlim=(-.06, .08), ylim=(2.6, .4), yticks=[]); ax.set_title("POS (circle) and A x 70 (square)\nagainst the baseline, ± 1 SE")
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
ax = axs[1, 4]
mins = np.arange(0, 61, 5)
for k, run in enumerate(data[1:], start=1):
    D = np.array([run[s]["pos_t"] for s in sorted(run)]) - np.array([data[0][s]["pos_t"] for s in sorted(run)])
    m, e = D.mean(0), D.std(0, ddof=1)/np.sqrt(len(D))
    ax.fill_between(mins, m - e, m + e, color=cols[k], alpha=.18, lw=0)
    ax.plot(mins, m, color=cols[k], lw=2)
ax.axhline(0, color=AXIS, lw=1)
ax.set(xlabel="Minutes", xticks=[0, 15, 30, 45, 60]); ax.set_title("POS against the baseline over time\n(paired mean ± 1 SE)")
ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0)
fig.legend(handles=[Patch(color=c, label=n) for n, _, c in RUNS], loc="center", bbox_to_anchor=(.5, .935), ncol=3, fontsize=10)
fig.text(.01, .975, "Transit limit lifted: both rules, 5 UAVs, A + POS, x 70, w 1, scenarios 500-509, held-out particles. First rectangles of UAV 1-4 unless stated", ha="left", fontsize=12)
fig.savefig(sys.argv[1], dpi=108); plt.close(fig)
# map
SEED, T = 504, 1020.0
env = make_envs(1, SEED)[0]; b = bel[SEED]
rgb, ext = terrain_rgb(env.terr), [env.terr.axis[0]/1000, env.terr.axis[-1]/1000]*2
P = b.positions(T)
allc = np.concatenate([corners(run[SEED]["rects"][u][0]) for run in data for u in range(5)])
mid = (allc.min(0) + allc.max(0))/2; half = max(4.6, float((allc.max(0) - allc.min(0)).max())/2 + .8)
fig, axs = plt.subplots(1, 3, figsize=(19, 8.6), constrained_layout=True, sharey=True)
fig.get_layout_engine().set(rect=(0, .07, 1, .85))
ucol = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
for ax, (name, _, _), run in zip(axs, RUNS, data):
    r = run[SEED]
    ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    ax.scatter(P[:, 0]/1000, P[:, 1]/1000, s=4, color=MUTED, linewidths=0)
    for u in range(5):
        ax.add_patch(Polygon(corners(r["rects"][u][0]), fill=False, ec=ucol[u], lw=2.2))
        tt, xy = np.array(r["tt"][u]), np.array(r["xy"][u]); sel = tt <= T
        ax.plot(xy[sel, 0]/1000, xy[sel, 1]/1000, color=ucol[u], lw=.9)
    ax.plot(*np.array(r["lkp"])/1000, "*", color=INK, ms=12, mec="white", zorder=6)
    ax.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half), xlabel="km")
    ax.set_title(f"{name}\nPOS {r['pos']:.3f}, A {r['scores']['70']['A']:.3f}, pair within 300 m {r['below']:.0f} s"); ax.tick_params(length=0)
axs[0].set_ylabel("km")
fig.legend(handles=[Line2D([], [], color=c, lw=2.2, label=f"UAV {u} first rectangle and track") for u, c in enumerate(ucol)] + [Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="Particles"), Line2D([], [], marker="*", ls="", color=INK, mec="white", ms=11, label="LKP")],
           loc="lower center", ncol=7, fontsize=9)
fig.text(.01, .955, f"Scenario {SEED}, both rules, 5 UAVs, A + POS, x 70: first rectangles with particles at {T/60:.0f} min", ha="left", fontsize=12)
fig.savefig(sys.argv[2], dpi=100)
