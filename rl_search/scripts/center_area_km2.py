"""The sweep results in km2 instead of the share A (from the saved paths, no planner rerun).

For every run: the swept area inside the current x% region over time (c_j x region area), the cells that scored (swept
while inside the region at that time), and the whole footprint, with or without a region. c_j comes from the saved
5 s paths; the areas that need a replay use the 10 s paths stored in results.jsonl.

    python -m rl_search.scripts.center_area_km2 --workers 8
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

SWEEP = Path("rl_search/data/center_open/sweep")
UAV_KM2_MIN = 15*60*286/1e6
XS = (50.0, 70.0, 85.0)
WS = (0.5, 1.0, 2.0)
NS = (1, 3, 5)
SURFACE, INK, SEC, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS_C, SCORE_C = "#2a78d6", "#1baf7a"


def footprint_km2(n, lo, c, R, xy, tt) -> float:
    """Unique area within R of any path on the 25 m grid of the whole map."""
    seen = np.zeros((n, n), bool)
    m = int(np.ceil(R/c)) + 1
    for p in xy:
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            ix0, ix1 = int((min(a[0], b[0]) - lo)//c) - m, int((max(a[0], b[0]) - lo)//c) + m + 1
            iy0, iy1 = int((min(a[1], b[1]) - lo)//c) - m, int((max(a[1], b[1]) - lo)//c) + m + 1
            ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, n), min(iy1, n)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            X = lo + (np.arange(ix0, ix1)[None, :] + .5)*c
            Y = lo + (np.arange(iy0, iy1)[:, None] + .5)*c
            ab = b - a
            den = float(ab @ ab)
            s = np.clip(((X - a[0])*ab[0] + (Y - a[1])*ab[1])/den, 0.0, 1.0) if den > 0 else 0.0
            seen[iy0:iy1, ix0:ix1] |= (X - a[0] - s*ab[0])**2 + (Y - a[1] - s*ab[1])**2 <= R*R
    return float(seen.sum()*c*c/1e6)


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.area_score import AreaScore, CELL_M
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center_open import make_envs
    rows = [r for r in map(json.loads, (SWEEP/"results.jsonl").read_text().splitlines()) if r["seed"] == seed]
    env = make_envs(1, seed)[0]
    scen = open_scenario(env.sampler, seed)
    hold = holdout_scenario(env, scen, seed + 10**6)
    scores = {x: AreaScore(env.terr, env.h, hold.track, hold.save_dt_s, x, env.lrc_R) for x in XS}
    out = []
    for r in rows:
        xy, tt = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]]
        rec = dict(seed=seed, config=r["config"], n=r["n"], x=r["x"], w=r["w"], pos=r["pos"],
                   footprint=footprint_km2(env.n, env.lo, env.cfg.cell_m, env.lrc_R, xy, tt), by_x={})
        seg = sorted((float(t[k + 1]), u, k) for u, t in enumerate(tt) for k in range(len(t) - 1))
        for x in (XS if r["config"] == "pos" else (r["x"],)):
            a = scores[x]
            a.S[:] = False
            a.log.clear()
            for _, u, k in seg:
                a.mark(xy[u][k:k + 2], tt[u][k:k + 2])
            s = r["scores"][str(int(x))]
            now = (np.array(s["c"])*np.array(s["area_km2"])).tolist()
            rec["by_x"][str(int(x))] = dict(scored=float(a.S.sum()*CELL_M**2/1e6), now=now, region=s["area_km2"], A=s["A"], c=s["c"])
        out.append(rec)
    print(f"seed {seed} done", flush=True)
    return out


def table(rows) -> str:
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ()): r for r in rows}
    se = lambda d: d.std(ddof=1)/np.sqrt(len(d))
    L = ["All areas in km2, on held-out particles, scenarios 500-509. Differences are paired against POS only (se in brackets). "
         f"One UAV sweeps at most {UAV_KM2_MIN*60:.1f} km2 in 60 min.", ""]
    for n in NS:
        L += [f"## {n} UAV" + ("s" if n > 1 else ""), "",
              "| x | w | region at 60 min | swept in the current region at 60 min, base / run | diff | diff in UAV minutes | time average, base / run | diff | scored cells, base / run | diff | whole footprint, base / run | A diff |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in XS:
            sx = str(int(x))
            for w in WS:
                base = [by[(s, "pos", n)] for s in seeds]
                run = [by[(s, "score", n, x, w)] for s in seeds]
                g = lambda r, f: f(r["by_x"][sx])
                b60, r60 = np.array([g(r, lambda v: v["now"][-1]) for r in base]), np.array([g(r, lambda v: v["now"][-1]) for r in run])
                bm, rm = np.array([g(r, lambda v: np.mean(v["now"])) for r in base]), np.array([g(r, lambda v: np.mean(v["now"])) for r in run])
                bs, rs = np.array([g(r, lambda v: v["scored"]) for r in base]), np.array([g(r, lambda v: v["scored"]) for r in run])
                da = np.array([g(a, lambda v: v["A"]) - g(b, lambda v: v["A"]) for a, b in zip(run, base)])
                reg = np.mean([g(r, lambda v: v["region"][-1]) for r in base])
                L.append(f"| {x:.0f} | {w:g} | {reg:.1f} | {b60.mean():.1f} / {r60.mean():.1f} | {(r60 - b60).mean():+.1f} ({se(r60 - b60):.1f}) | {(r60 - b60).mean()/UAV_KM2_MIN:+.0f} | "
                         f"{bm.mean():.1f} / {rm.mean():.1f} | {(rm - bm).mean():+.1f} ({se(rm - bm):.1f}) | {bs.mean():.1f} / {rs.mean():.1f} | {(rs - bs).mean():+.1f} ({se(rs - bs):.1f}) | "
                         f"{np.mean([r['footprint'] for r in base]):.1f} / {np.mean([r['footprint'] for r in run]):.1f} | {da.mean():+.3f} |")
        L.append("")
    return "\n".join(L)


def plot(rows, out, x=70.0, w=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED,
                         "ytick.color": MUTED, "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ()): r for r in rows}
    sx = str(int(x))
    t = np.arange(5, 61, 5)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.8), height_ratios=[1, 1.1], constrained_layout=True)
    for j, n in enumerate(NS):
        base = [by[(s, "pos", n)]["by_x"][sx] for s in seeds]
        run = [by[(s, "score", n, x, w)]["by_x"][sx] for s in seeds]
        ax = axs[0, j]
        region = np.mean([v["region"] for v in base], axis=0)
        ax.fill_between(t, 0, region, color=AXIS, alpha=.35, lw=0)
        ax.plot(t, region, color=MUTED, lw=1.2)
        ax.plot(t, np.mean([v["now"] for v in base], axis=0), color=POS_C, lw=1.8)
        ax.plot(t, np.mean([v["now"] for v in run], axis=0), color=SCORE_C, lw=1.8)
        d = np.mean([a["now"][-1] - b["now"][-1] for a, b in zip(run, base)])
        ax.set(xlim=(5, 60), ylim=(0, 32), xticks=range(15, 61, 15), xlabel="Minutes after the LKP report", ylabel="km²")
        ax.set_title(f"{n} UAV{'s' if n > 1 else ''}: at 60 min {np.mean([v['now'][-1] for v in base]):.1f} to {np.mean([v['now'][-1] for v in run]):.1f} km² ({d:+.1f}) of {region[-1]:.1f} km²")
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        ax = axs[1, j]
        a, b = [v["now"][-1] for v in base], [v["now"][-1] for v in run]
        for y, (p, q) in enumerate(zip(a, b)):
            ax.plot([p, q], [y, y], color=AXIS, lw=1.6, zorder=2)
            ax.scatter(p, y, s=50, color=POS_C, edgecolor=SURFACE, lw=1.4, zorder=3)
            ax.scatter(q, y, s=50, color=SCORE_C, edgecolor=SURFACE, lw=1.4, zorder=3)
        k = int(np.argmax(np.abs(np.array(b) - np.array(a))))
        ax.annotate(f"{b[k] - a[k]:+.1f} km²", (max(a[k], b[k]), k), xytext=(8, 0), textcoords="offset points", va="center", color=SEC)
        ax.set(xlim=(0, 30), ylim=(len(seeds) - .5, -.5), yticks=range(len(seeds)), yticklabels=seeds, xlabel="Swept in the current region at 60 min, km²", ylabel="Scenario")
        ax.set_title("Each scenario (label: largest gap)")
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    keys = [Line2D([], [], color=POS_C, lw=1.8, marker="o", ms=6, mec=SURFACE, label="POS only"), Line2D([], [], color=SCORE_C, lw=1.8, marker="o", ms=6, mec=SURFACE, label=f"A + POS (x {x:g}, w {w:g})"),
            Line2D([], [], color=MUTED, lw=1.2, label=f"Area of the current {x:g}% region")]
    fig.legend(handles=keys, loc="outside lower center", ncol=3)
    fig.suptitle(f"The gain in A as area: swept inside the current {x:g}% region, SAROPS, scenarios {seeds[0]}-{seeds[-1]}", x=.01, ha="left", fontsize=12)
    fig.savefig(out/f"km2_x{x:g}_w{w:g}.png", dpi=125)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cache = SWEEP/"km2.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        with Pool(args.workers) as pool:
            rows = [r for part in pool.imap_unordered(job, range(500, 510), chunksize=1) for r in part]
        cache.write_text(json.dumps(rows))
    txt = table(rows)
    (SWEEP/"km2.md").write_text(txt + "\n")
    print(txt)
    plot(rows, SWEEP)
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["config"], r["n"]) + ((r["x"], r["w"]) if r["config"] == "score" else ()): r for r in rows}
    print("check: mean c at 60 min x 70 from the stored 5 s paths against scored cells still in the region are in km2 above")
    for n in NS:
        d = np.array([by[(s, "score", n, 70.0, 1.0)]["by_x"]["70"]["now"][-1] - by[(s, "pos", n)]["by_x"]["70"]["now"][-1] for s in seeds])
        print(f"N={n} x 70 w 1, swept now at 60 min, diff per scenario:", np.round(d, 1).tolist(), f"higher in {int((d > 0).sum())}/10")


if __name__ == "__main__":
    main()
