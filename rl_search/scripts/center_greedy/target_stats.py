"""Exploratory measure: what the target model actually produces on the centre scenarios 400-409, plus one figure."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"
SEEDS = list(range(400, 410))
SHOW = (406, 400, 402, 408)
SURFACE, INK, SEC, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
TIMES = ((15, "#2a78d6"), (30, "#eb6834"), (60, "#1baf7a"))        # first three slots, validated all-pairs


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario
    env = make_envs(1, seed)[0]
    s = center_scenario(env.sampler, seed)
    tr = np.asarray(s.track, float)
    k60 = int(3600/s.save_dt_s)
    d = np.hypot(*np.moveaxis(np.diff(tr[:k60 + 1], axis=0), 2, 0))     # (T, N) m per 10 s
    v = d/s.save_dt_s
    moving = v > .05
    path = d.sum(0)/1000
    out = dict(seed=seed, nominal=s.speed_mps*3.6, inst=float(v.mean()*3.6),
               moving_speed=float(v[moving].mean()*3.6), stopped=float((~moving).mean()), path=float(path.mean()))
    for m in (15, 30, 60):
        p = tr[int(m*60/s.save_dt_s)]
        r = p - s.lkp
        dist = np.hypot(*r.T)/1000
        ang = np.arctan2(r[:, 1], r[:, 0])[dist > .05]
        rel = np.angle(np.exp(1j*(ang - np.deg2rad(s.heading_deg))))
        mean_dir = np.angle(np.exp(1j*rel).mean())
        R = abs(np.exp(1j*rel).mean())
        out[f"dist{m}"] = float(dist.mean())
        out[f"p95{m}"] = float(np.quantile(dist, .95))
        out[f"dir{m}"] = float(np.degrees(mean_dir))
        out[f"R{m}"] = float(R)
        out[f"within45_{m}"] = float((np.abs(rel) <= np.deg2rad(45)).mean())
        out[f"nominal_reach{m}"] = s.speed_mps*m*60/1000
    still = (d[-60:].sum(0) < 1.0)                                     # no movement in the last 10 min
    out["still_last10"] = float(still.mean())
    if seed in SHOW:
        out["tracks"] = {m: tr[int(m*60/s.save_dt_s)].tolist() for m, _ in TIMES}
        out["heading"], out["lkp"] = s.heading_deg, s.lkp.tolist()
    return out


def main():
    with Pool(3) as pool:
        rows = pool.map(job, SEEDS, chunksize=1)
    print("seed | nominal km/h | mean speed incl. stops | speed while moving | share of time stopped | still in last 10 min | path km | mean dist 15/30/60 km | p95 at 60 | nominal reach 60 | mean direction vs heading 15/30/60 deg | within +-45 deg of heading 15/30/60")
    for r in rows:
        print(r["seed"], f"{r['nominal']:.1f} {r['inst']:.1f} {r['moving_speed']:.1f} {r['stopped']:.2f} {r['still_last10']:.2f} {r['path']:.1f} | "
              f"{r['dist15']:.1f} {r['dist30']:.1f} {r['dist60']:.1f} {r['p9560']:.1f} {r['nominal_reach60']:.1f} | "
              f"{r['dir15']:+.0f} {r['dir30']:+.0f} {r['dir60']:+.0f} | {r['within45_15']:.2f} {r['within45_30']:.2f} {r['within45_60']:.2f}")
    k = ["nominal", "inst", "moving_speed", "stopped", "still_last10", "path", "dist15", "dist30", "dist60", "p9560", "nominal_reach60", "within45_15", "within45_30", "within45_60"]
    print("\nmean: " + ", ".join(f"{x} {np.mean([r[x] for r in rows]):.2f}" for x in k))
    print("|mean direction offset| 15/30/60: " + " ".join(f"{np.mean([abs(r[f'dir{m}']) for r in rows]):.0f}" for m in (15, 30, 60)))
    print("net displacement / path length:", np.round(np.mean([r["dist60"]/r["path"] for r in rows]), 2),
          "; mean dist 60 / nominal reach:", np.round(np.mean([r["dist60"]/r["nominal_reach60"] for r in rows]), 2))
    from scipy.stats import spearmanr
    print("Spearman speed vs mean dist 60:", np.round(spearmanr([r["nominal"] for r in rows], [r["dist60"] for r in rows]).statistic, 2))

    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.plot_center_terrain import terrain_rgb
    terr = make_envs(1, 400)[0].terr
    rgb = terrain_rgb(terr)
    ext = [terr.axis[0]/1000, terr.axis[-1]/1000]*2
    by = {r["seed"]: r for r in rows}
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": SEC, "axes.edgecolor": AXIS, "xtick.color": MUTED,
                         "ytick.color": MUTED, "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.titlelocation": "left", "axes.titlesize": 10.5, "legend.frameon": False})
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 12), constrained_layout=True)
    for ax, s in zip(axs.flat, SHOW):
        r = by[s]
        lkp = np.array(r["lkp"])/1000
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
        for m, c in reversed(TIMES):
            p = np.array(r["tracks"][m])/1000
            ax.scatter(p[:, 0], p[:, 1], s=2.2, c=c, linewidths=0, alpha=.75)
        reach = r["nominal_reach60"]
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(lkp[0] + reach*np.cos(th), lkp[1] + reach*np.sin(th), color=SEC, lw=1)
        h = np.deg2rad(r["heading"])
        ax.annotate("", xy=(lkp[0] + reach*np.cos(h), lkp[1] + reach*np.sin(h)), xytext=lkp,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.plot(*lkp, "*", color=INK, ms=12, mec="white", zorder=6)
        lim = reach*1.1
        ax.set(xlim=(lkp[0] - lim, lkp[0] + lim), ylim=(lkp[1] - lim, lkp[1] + lim), xlabel="km", ylabel="km")
        ax.set_title(f"Scenario {s}: reported {r['nominal']:.0f} km/h, heading {r['heading']:.0f} deg\nMean distance from LKP at 60 min: {r['dist60']:.1f} km")
        ax.tick_params(length=0)
    keys = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=f"Particles at {m} min") for m, c in TIMES]
    keys += [Line2D([], [], color=INK, lw=1.4, marker=">", ms=5, label="Reported heading"),
             Line2D([], [], color=SEC, lw=1, label="Reach at reported speed, 60 min")]
    fig.legend(handles=keys, loc="outside lower center", ncol=5)
    fig.suptitle("The target model spreads far less than the reported speed suggests, and the terrain shapes where it goes",
                 x=.01, ha="left", fontsize=12)
    fig.savefig(HERE/"target_distribution.png", dpi=95)


if __name__ == "__main__":
    main()
