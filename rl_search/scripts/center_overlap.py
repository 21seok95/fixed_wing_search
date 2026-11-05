"""How much would it cost to keep the UAVs' coverage apart? Estimated from the saved flights, no planner rerun.

For each flight on scenarios 500-509, on held-out particles and the 10 s paths:
  overlap    area swept by two or more UAVs, and how long after the first UAV the second one came
  POS value  POS with every particle examined only by the UAV that saw it first. The difference from the real POS is an
             upper bound on the POS those repeat looks provide, with no credit for what the freed flight time would find
  rectangles share of each new rectangle already inside another UAV's current rectangle
  close time seconds below 300 m split by what the pair was doing: both on lanes with overlapping rectangles, both on
             lanes with separate rectangles, one on lanes and one moving or holding, both moving or holding

    python -m rl_search.scripts.center_overlap --workers 8
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import warnings

import numpy as np

BASE = Path("rl_search/data/center_open")
OUT = BASE/"overlap"
CONFIGS = (("POS only", "pos", None), ("A + POS, x 70", "score", 70.0), ("A + POS, x 85", "score", 85.0))
COLS = ("#2a78d6", "#eb6834", "#1baf7a")      # first three categorical slots, validated all pairs
CLASSES = ("both on lanes, rectangles overlap", "both on lanes, rectangles apart", "one on lanes, one moving or holding", "both moving or holding")
SETTINGS = (("no constraint", "sweep"), ("time rule", "sep"), ("time + rectangle rule", "no_overlap"))
SEEDS = list(range(500, 510))
D_SAFE = 300.0
MARGIN = 150.0


def first_visits(xy, tt, R, c=25.0):
    """Earliest time each 25 m cell comes within R of each UAV's path, on a grid around all the paths. (U, H, W)."""
    pts = np.concatenate(xy)
    x0, y0 = pts[:, 0].min() - R - c, pts[:, 1].min() - R - c
    w, h = int((pts[:, 0].max() + R + c - x0)//c) + 1, int((pts[:, 1].max() + R + c - y0)//c) + 1
    out = np.full((len(xy), h, w), np.inf, np.float32)
    m = int(np.ceil(R/c)) + 1
    for u, (p, t) in enumerate(zip(xy, tt)):
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            ix0, ix1 = max(int((min(a[0], b[0]) - x0)//c) - m, 0), min(int((max(a[0], b[0]) - x0)//c) + m + 1, w)
            iy0, iy1 = max(int((min(a[1], b[1]) - y0)//c) - m, 0), min(int((max(a[1], b[1]) - y0)//c) + m + 1, h)
            X = x0 + (np.arange(ix0, ix1)[None, :] + .5)*c
            Y = y0 + (np.arange(iy0, iy1)[:, None] + .5)*c
            ab = b - a
            den = float(ab @ ab)
            s = np.clip(((X - a[0])*ab[0] + (Y - a[1])*ab[1])/den, 0.0, 1.0) if den > 0 else 0.0
            hit = (X - a[0] - s*ab[0])**2 + (Y - a[1] - s*ab[1])**2 <= R*R
            when = t[k] + s*(t[k + 1] - t[k])
            view = out[u, iy0:iy1, ix0:ix1]
            view[hit] = np.minimum(view[hit], when[hit] if den > 0 else when)
    return out


def in_rect(q, r, margin=0.0):
    """q (..., 2) inside rectangle r = (tp, cx, cy, ell, wid, theta, hold)."""
    c, s = np.cos(r[5]), np.sin(r[5])
    dx, dy = q[..., 0] - r[1], q[..., 1] - r[2]
    return (np.abs(dx*c + dy*s) <= r[3]/2 + margin) & (np.abs(-dx*s + dy*c) <= r[4]/2 + margin)


def rect_samples(r, nx=30, ny=12):
    u = (np.arange(nx) + .5)/nx*r[3] - r[3]/2
    v = (np.arange(ny) + .5)/ny*r[4] - r[4]/2
    U, V = np.meshgrid(u, v)
    c, s = np.cos(r[5]), np.sin(r[5])
    return np.stack([r[1] + U*c - V*s, r[2] + U*s + V*c], -1)


def current(rects, t):
    now = None
    for r in rects:
        if r[0] <= t:
            now = r
    return now


def phase(rects, p, t):
    r = current(rects, t)
    if r is None or t < r[0] + r[6]:
        return "moving", r
    return ("lanes", r) if in_rect(p, r, MARGIN) else ("moving", r)


def analyse(env, hold, r) -> dict:
    from rl_search.mission import ParticleBelief
    xy, tt = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]]
    n = len(xy)
    rects = r["rects"]
    # overlap of the swept area
    F = first_visits(xy, tt, env.lrc_R)
    seen = np.isfinite(F)
    count = seen.sum(0)
    cell = 25.0**2/1e6
    team, total = float((count > 0).sum()*cell), float(seen.sum()*cell)
    multi = count >= 2
    order = np.sort(F[:, multi], axis=0)
    gap = order[1] - order[0] if multi.any() else np.zeros(0)
    # POS with each particle examined only by the first UAV to see it
    env.belief = ParticleBelief(hold.track, hold.save_dt_s, hold.exit_step, True)
    N = env.belief.n
    w = np.ones((n, N))
    first = np.full((n, N), np.inf)
    for u in range(n):
        state = (np.zeros(N), np.full(N, -np.inf))
        for k in range(len(tt[u]) - 1):
            env._detect_lrc(xy[u][k:k + 2], tt[u][k:k + 2], w[u], state)
            new = (w[u] < 1) & ~np.isfinite(first[u])
            first[u][new] = tt[u][k + 1]
    pos_all = float((1 - w.prod(0)).mean())
    who = first.argmin(0)
    pos_first = float((1 - w[who, np.arange(N)]).mean())
    two = float(((w < 1).sum(0) >= 2).mean())
    # rectangles against the other UAVs' current rectangles
    shares = []
    for u in range(n):
        for q in rects[u]:
            pts = rect_samples(q)
            inside = np.zeros(pts.shape[:2], bool)
            for v in range(n):
                other = current(rects[v], q[0]) if v != u else None
                if other is not None:
                    inside |= in_rect(pts, other)
            shares.append(float(inside.mean()))
    area = [q[3]*q[4]/1e6 for rs in rects for q in rs]
    # close time by what the pair was doing
    close = {c: 0.0 for c in CLASSES}
    for a in range(n):
        for b in range(a + 1, n):
            t0, t1 = max(tt[a][0], tt[b][0]), min(tt[a][-1], tt[b][-1])
            if t1 <= t0:
                continue
            g = np.arange(t0, t1, 5.0)
            pa = np.stack([np.interp(g, tt[a], xy[a][:, 0]), np.interp(g, tt[a], xy[a][:, 1])], 1)
            pb = np.stack([np.interp(g, tt[b], xy[b][:, 0]), np.interp(g, tt[b], xy[b][:, 1])], 1)
            for i in np.flatnonzero(np.hypot(*(pa - pb).T) < D_SAFE):
                sa, ra = phase(rects[a], pa[i], g[i])
                sb, rb = phase(rects[b], pb[i], g[i])
                if sa == "lanes" and sb == "lanes":
                    together = in_rect(rect_samples(ra), rb).any() or in_rect(rect_samples(rb), ra).any()
                    close[CLASSES[0] if together else CLASSES[1]] += 5.0
                elif sa == "lanes" or sb == "lanes":
                    close[CLASSES[2]] += 5.0
                else:
                    close[CLASSES[3]] += 5.0
    return dict(seed=r["seed"], config=r["config"], n=n, x=r["x"], sep=bool(r.get("sep")), pos=r["pos"], below=r["below"], team_km2=team, total_km2=total,
                multi_km2=float(multi.sum()*cell), gap_10=float((gap <= 600).mean()) if len(gap) else 0.0, gap_30=float((gap <= 1800).mean()) if len(gap) else 0.0,
                pos_all=pos_all, pos_first=pos_first, seen_two=two, rect_share=float(np.mean(shares)) if shares else 0.0, rect_km2=float(np.mean(area)) if area else 0.0, close=close)


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center_open import make_envs
    env = make_envs(1, seed)[0]
    scen = open_scenario(env.sampler, seed)
    hold = holdout_scenario(env, scen, seed + 10**6)
    out = []
    for setting, folder in SETTINGS:
        for line in (BASE/folder/"results.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["seed"] != seed or r["n"] not in (3, 5):
                continue
            if r["config"] == "score" and not (r["w"] == 1.0 and r["x"] in (70.0, 85.0)):
                continue
            row = analyse(env, hold, r)
            row["setting"] = setting
            row["forced"] = r.get("overlap_forced", 0)
            row["plans"] = r["plans"]
            out.append(row)
    print(f"seed {seed}: {len(out)} flights", flush=True)
    return out


def table(rows) -> str:
    L = ["Saved flights, SAROPS, scenarios 500-509, held-out particles, 10 s paths. Means over scenarios.", "",
         "| constraint | UAVs | config | POS | total swept (sum over UAVs) km2 | team swept km2 | swept by 2+ UAVs km2 (share of team) | second UAV within 10 / 30 min | particles seen by 2+ UAVs | POS, first UAV only (upper bound on the loss) | new rectangle inside another's current rectangle | rectangle km2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for setting, _ in SETTINGS:
        for n in (3, 5):
            for name, cfg, x in CONFIGS:
                g = [r for r in rows if r["setting"] == setting and r["n"] == n and r["config"] == cfg and (x is None or r["x"] == x)]
                m = lambda k: np.mean([r[k] for r in g])
                d = np.array([r["pos_all"] - r["pos_first"] for r in g])
                L.append(f"| {setting} | {n} | {name} | {m('pos_all'):.3f} | {m('total_km2'):.1f} | {m('team_km2'):.1f} | {m('multi_km2'):.1f} ({m('multi_km2')/m('team_km2'):.0%}) | {m('gap_10'):.0%} / {m('gap_30'):.0%} | {m('seen_two'):.0%} | "
                         f"{m('pos_first'):.3f} (-{d.mean():.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f}) | {m('rect_share'):.0%} | {m('rect_km2'):.2f} |")
    L += ["", "| constraint | UAVs | config | close time s | " + " | ".join(CLASSES) + " |", "|---|---|---|---|---|---|---|---|"]
    for setting, _ in SETTINGS:
        for n in (3, 5):
            for name, cfg, x in CONFIGS:
                g = [r for r in rows if r["setting"] == setting and r["n"] == n and r["config"] == cfg and (x is None or r["x"] == x)]
                L.append(f"| {setting} | {n} | {name} | {np.mean([r['below'] for r in g]):.0f} | " + " | ".join(f"{np.mean([r['close'][c] for r in g]):.0f}" for c in CLASSES) + " |")
    return "\n".join(L)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    SURFACE, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 10, "text.color": "#0b0b0b", "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
    cases = [(s, n, c) for s, _ in SETTINGS for n in (3, 5) for c in CONFIGS]
    labels = [f"{s}, {n} UAVs, {c[0]}" for s, n, c in cases]
    ypos = np.arange(len(cases)) + np.repeat([0, .7, 1.9, 2.6, 3.8, 4.5], 3)
    pick = lambda s, n, c: [r for r in rows if r["setting"] == s and r["n"] == n and r["config"] == c[1] and (c[2] is None or r["x"] == c[2])]
    col = {c[0]: k for c, k in zip(CONFIGS, COLS)}
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 10), sharey=True, constrained_layout=True)
    ax = axs[0]
    for y, (sep, n, c) in zip(ypos, cases):
        g = pick(sep, n, c)
        v = np.mean([r["multi_km2"]/r["team_km2"] for r in g])
        ax.barh(y, v, height=.62, color=col[c[0]])
        ax.annotate(f"{v:.0%}", (v, y), xytext=(5, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
    ax.set(xlim=(0, .6), xlabel="Share of team area swept by 2+ UAVs", yticks=ypos, yticklabels=labels, ylim=(ypos[-1] + .7, -.7))
    ax.set_title("How much coverage overlaps")
    ax = axs[1]
    for y, (sep, n, c) in zip(ypos, cases):
        g = pick(sep, n, c)
        a, b = np.mean([r["pos_all"] for r in g]), np.mean([r["pos_first"] for r in g])
        ax.plot([a, b], [y, y], color=AXIS, lw=1.6, zorder=2)
        ax.scatter(a, y, s=58, color=col[c[0]], edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.scatter(b, y, s=58, facecolor=SURFACE, edgecolor=col[c[0]], lw=1.6, zorder=3)
        ax.annotate(f"{b - a:+.3f}", (a, y), xytext=(8, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
    ax.set(xlim=(.3, .85), xlabel="POS: real (filled), first UAV only (hollow)")
    ax.set_title("Upper bound on the POS from repeat looks")
    ax = axs[2]
    shades = ("#d03b3b", "#ec835a", "#b7d3f6", "#c3c2b7")
    for y, (sep, n, c) in zip(ypos, cases):
        g = pick(sep, n, c)
        left = 0.0
        for cl, shade in zip(CLASSES, shades):
            v = np.mean([r["close"][cl] for r in g])
            ax.barh(y, v, left=left, height=.62, color=shade, edgecolor=SURFACE, lw=1.2)
            left += v
    ax.set(xlim=(0, 300), xlabel="Seconds with a pair closer than 300 m")
    ax.set_title("What the pair was doing when close")
    for ax in axs:
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    keys = [Patch(color=k, label=c[0]) for c, k in zip(CONFIGS, COLS)] + [Patch(color=s, label=cl.capitalize()) for cl, s in zip(CLASSES, shades)]
    fig.legend(handles=keys, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Coverage overlap and close time under each rule, from the saved flights (SAROPS, scenarios 500-509)", x=.01, ha="left", fontsize=12)
    fig.savefig(out/"overlap.png", dpi=120)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/"overlap.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        with Pool(args.workers) as pool:
            rows = [r for part in pool.imap_unordered(job, SEEDS, chunksize=1) for r in part]
        path.write_text(json.dumps(rows))
    txt = table(rows)
    (OUT/"table.md").write_text(txt + "\n")
    print(txt)
    plot(rows, OUT)


if __name__ == "__main__":
    main()
