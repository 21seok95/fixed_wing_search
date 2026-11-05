"""Centre scenarios on open ground (2026-11-05): the target starts near the map centre but away from the no-go ridge, so
it meets terrain only after moving for a while, and it stops less (hide share 5% instead of 20%). The default target
model and the older scenario sets are unchanged.

This script checks the scenario set before any planner runs on it: speeds and stops, when particles first come near
no-go terrain, map exits, and how the x% region of the unweighted particles grows, against the old centre set (400-409).

    python -m rl_search.scripts.center_open --n-scen 10 --workers 8
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from dataclasses import replace
import json
from multiprocessing import Pool
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

OUT = Path("rl_search/data/center_open")
P_STOP = 0.0421               # hide share 5% with p_move 0.5 and p_reorient 0.3 (action_probs normalises the three)
MIN_CLEARANCE_M = 3000.0      # LKP: the cell nearest the map centre at least this far from any no-go cell
NEAR_TERRAIN_M = 200.0        # a particle "meets terrain" when it first comes this close to no-go
XS = (50, 70, 85, 95)
SWEEP_KM2_MIN = 15*60*286/1e6  # one UAV: 15 m/s, 286 m swath


def use_open_target(sampler) -> None:
    """Switch this sampler to the open-ground target model. holdout_scenario reads the same config."""
    sampler.cfg = replace(sampler.cfg, target=replace(sampler.cfg.target, p_stop=P_STOP))


def open_lkp(terr) -> np.ndarray:
    ok = np.argwhere(terr.clearance >= MIN_CLEARANCE_M)
    xy = np.stack([terr.axis[ok[:, 1]], terr.axis[ok[:, 0]]], 1)
    return xy[int(np.argmin(np.hypot(*xy.T)))].astype(float)


def open_scenario(sampler, seed, dist=(5000.0, 15000.0)):
    """As compare_center.center_scenario with the open-ground LKP and target model: heading uniform, speed as the
    sampler, launch at a random bearing and distance inside the map, heading toward the LKP."""
    from rl_search.scenario import Scenario, scenario_metrics
    from rl_search.target import simulate
    if abs(sampler.cfg.target.p_stop - P_STOP) > 1e-12:
        use_open_target(sampler)
    t0 = time.perf_counter()
    terr, cfg = sampler.terr, sampler.cfg
    rng = np.random.default_rng(seed)
    lkp = open_lkp(terr)
    heading = float(rng.uniform(0, 360))
    speed = float(rng.uniform(*cfg.speed_kmh))/3.6
    track, exit_step = simulate(terr, cfg.target, lkp, heading, speed, rng)
    lim = terr.axis[-1] - 1500.0
    for _ in range(1000):
        b, d = rng.uniform(-np.pi, np.pi), rng.uniform(*dist)
        start = lkp + d*np.array([np.cos(b), np.sin(b)])
        if np.all(np.abs(start) <= lim):
            break
    sh = float(np.degrees(np.arctan2(lkp[1] - start[1], lkp[0] - start[0])))
    metrics = scenario_metrics(track, exit_step, cfg.target.save_dt_s, start, cfg)
    return Scenario(seed, lkp, heading, speed, start, sh, track, exit_step, cfg.target.save_dt_s, metrics,
                    time.perf_counter() - t0)


def regions(terr, h, pts) -> dict:
    """x% region per x: smoothed density as rl_env._zone, threshold set so that x% of the particles are inside."""
    tc = float(terr.axis[1] - terr.axis[0])
    nt = len(terr.axis)
    ij = np.clip(np.round((pts - terr.axis[0])/tc).astype(int), 0, nt - 1)
    hist = np.zeros((nt, nt))
    np.add.at(hist, (ij[:, 1], ij[:, 0]), 1.0)
    dens = gaussian_filter(hist, h/tc, mode="constant")
    dens[terr.forbidden] = 0
    at = dens[ij[:, 1], ij[:, 0]]
    return {x: (dens >= max(np.quantile(at, 1 - x/100), 1e-12)) & (dens > 0) for x in XS}


def stats(env, s) -> dict:
    terr = env.terr
    tr = np.asarray(s.track, float)
    dt = s.save_dt_s
    k60 = int(3600/dt)
    d = np.hypot(*np.moveaxis(np.diff(tr[:k60 + 1], axis=0), 2, 0))
    v = d/dt
    moving = v > .05
    out = dict(seed=s.seed, heading=s.heading_deg, speed=s.speed_mps*3.6, launch_km=s.metrics["start_lkp_km"],
               escaped=s.metrics["escaped_end"], lkp=s.lkp.tolist(), start=s.start.tolist(),
               mean_speed=float(v.mean()*3.6), moving_speed=float(v[moving].mean()*3.6),
               stopped=float((~moving).mean()), path=float(d.sum(0).mean()/1000),
               edge_margin_km=float((env.half_map - np.abs(tr[:k60 + 1]).max())/1000))
    # first time near no-go terrain, sampled every 30 s
    step = int(30/dt)
    ax0, c, nt = terr.axis[0], terr.cell, len(terr.axis)
    idx = np.clip(np.round((tr[::step][:int(3600/30) + 1] - ax0)/c).astype(int), 0, nt - 1)
    near = terr.clearance[idx[..., 1], idx[..., 0]] < NEAR_TERRAIN_M
    hit = near.any(0)
    first = np.where(hit, near.argmax(0)*30.0/60.0, np.inf)
    out["near_by"] = {m: float((first <= m).mean()) for m in (5, 10, 20, 30, 60)}
    out["near_median_min"] = float(np.median(first))
    for m in (15, 30, 60):
        p = tr[int(m*60/dt)]
        r = p - s.lkp
        dist = np.hypot(*r.T)/1000
        rel = np.angle(np.exp(1j*(np.arctan2(r[:, 1], r[:, 0]) - np.deg2rad(s.heading_deg))))
        out[f"dist{m}"] = float(dist.mean())
        out[f"p95{m}"] = float(np.quantile(dist, .95))
        out[f"within45_{m}"] = float((np.abs(rel) <= np.deg2rad(45)).mean())
    km2 = terr.cell**2/1e6
    reg = {m: regions(terr, env.h, tr[int(m*60/dt)]) for m in (10, 20, 30, 40, 50, 60)}
    out["area"] = {x: {m: float(reg[m][x].sum()*km2) for m in reg} for x in XS}
    out["kept_30"] = {x: float((reg[30][x] & reg[60][x]).sum()/max(reg[30][x].sum(), 1)) for x in XS}
    out["tracks"] = {m: tr[int(m*60/dt)][::2].round(0).tolist() for m in (15, 30, 60)}
    return out


def job(item):
    kind, seed = item
    warnings.filterwarnings("ignore")
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario
    env = make_envs(1, seed)[0]
    return kind, stats(env, open_scenario(env.sampler, seed) if kind == "open" else center_scenario(env.sampler, seed))


def summarize(res) -> str:
    L = ["Scenario check, no planner runs. old: centre scenarios 400-409 (LKP at the centre, hide share 20%). open: LKP "
         f"away from the ridge (clearance at least {MIN_CLEARANCE_M/1000:.0f} km), hide share 5%.", ""]
    L += ["| set | LKP km | reported km/h | mean km/h incl. stops | km/h while moving | stopped | distance 15 / 30 / 60 min km | 95% within at 60 min km | "
          "within 45 deg of heading at 60 min (range) | near terrain by 5 / 10 / 20 / 30 min | escaped | margin to map edge km |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("old", "open"):
        g = res[kind]
        m = lambda k: np.mean([r[k] for r in g])
        w = [r["within45_60"] for r in g]
        L.append(f"| {kind} | ({g[0]['lkp'][0]/1000:.1f}, {g[0]['lkp'][1]/1000:.1f}) | {m('speed'):.1f} | {m('mean_speed'):.1f} | {m('moving_speed'):.1f} | {m('stopped'):.1%} | "
                 f"{m('dist15'):.1f} / {m('dist30'):.1f} / {m('dist60'):.1f} | {m('p9560'):.1f} | {np.mean(w):.0%} ({min(w):.0%}-{max(w):.0%}) | "
                 + " / ".join(f"{np.mean([r['near_by'][t] for r in g]):.0%}" for t in (5, 10, 20, 30))
                 + f" | {m('escaped'):.1%} | {min(r['edge_margin_km'] for r in g):.1f} |")
    L += ["", f"Sweep capacity per UAV: {SWEEP_KM2_MIN:.3f} km2/min, {SWEEP_KM2_MIN*60:.1f} km2 in 60 min.", "",
          "| set | x% | region km2 at 10 / 20 / 30 / 40 / 50 / 60 min | range at 60 min | growth 30-60 min km2/min | above 1 UAV | above 5 UAVs | 30-min region still in the 60-min region |",
          "|---|---|---|---|---|---|---|---|"]
    for kind in ("old", "open"):
        g = res[kind]
        for x in XS:
            a = [np.mean([r["area"][x][t] for r in g]) for t in (10, 20, 30, 40, 50, 60)]
            e = [r["area"][x][60] for r in g]
            gr = [(r["area"][x][60] - r["area"][x][30])/30 for r in g]
            L.append(f"| {kind} | {x} | " + " / ".join(f"{v:.1f}" for v in a) + f" | {min(e):.1f}-{max(e):.1f} | {np.mean(gr):+.2f} | "
                     f"{sum(v > SWEEP_KM2_MIN for v in gr)}/{len(g)} | {sum(v > 5*SWEEP_KM2_MIN for v in gr)}/{len(g)} | {np.mean([r['kept_30'][x] for r in g]):.0%} |")
    L += ["", "| open seed | heading | reported km/h | launch-LKP km | km/h incl. stops | stopped | distance at 60 min km | within 45 deg at 60 min | median first near terrain min |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["open"]:
        L.append(f"| {r['seed']} | {r['heading']:.0f} | {r['speed']:.1f} | {r['launch_km']:.1f} | {r['mean_speed']:.1f} | {r['stopped']:.1%} | {r['dist60']:.1f} | {r['within45_60']:.0%} | {r['near_median_min']:.1f} |")
    return "\n".join(L)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.plot_center_terrain import terrain_rgb
    surface, ink, sec, muted, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    times = ((15, "#2a78d6"), (30, "#eb6834"), (60, "#1baf7a"))   # first three slots, validated all-pairs
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": sec, "axes.edgecolor": axis, "xtick.color": muted,
                         "ytick.color": muted, "xtick.labelcolor": sec, "ytick.labelcolor": sec, "figure.facecolor": surface,
                         "savefig.facecolor": surface, "axes.titlelocation": "left", "axes.titlesize": 9.5, "legend.frameon": False})
    terr = make_envs(1, 400)[0].terr
    rgb = terrain_rgb(terr)
    ext = [terr.axis[0]/1000, terr.axis[-1]/1000]*2
    g = res["open"]
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.6), sharex=True, sharey=True, constrained_layout=True)
    lim = 20.0
    for ax, r in zip(axs.flat, g):
        lkp = np.array(r["lkp"])/1000
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
        for m, c in reversed(times):
            p = np.array(r["tracks"][m])/1000
            ax.scatter(p[:, 0], p[:, 1], s=1.6, c=c, linewidths=0, alpha=.75)
        h = np.deg2rad(r["heading"])
        ax.annotate("", xy=(lkp[0] + 10*np.cos(h), lkp[1] + 10*np.sin(h)), xytext=lkp, arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.3))
        ax.plot(*lkp, "*", color=ink, ms=10, mec="white", zorder=6)
        ax.set(xlim=(lkp[0] - lim, lkp[0] + lim), ylim=(lkp[1] - lim, lkp[1] + lim))
        ax.set_title(f"{r['seed']}: {r['speed']:.0f} km/h, heading {r['heading']:.0f} deg\n{r['within45_60']:.0%} within 45 deg at 60 min")
        ax.tick_params(length=0)
    for ax in axs[-1]:
        ax.set_xlabel("km")
    for ax in axs[:, 0]:
        ax.set_ylabel("km")
    keys = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=f"Particles at {m} min") for m, c in times]
    keys.append(Line2D([], [], color=ink, lw=1.3, marker=">", ms=5, label="Reported heading (arrow 10 km)"))
    fig.legend(handles=keys, loc="outside lower center", ncol=4)
    fig.suptitle("Open-ground centre scenarios: particles at 15, 30 and 60 min (same scale in every panel)", x=.01, ha="left", fontsize=12)
    fig.savefig(out/"scenarios.png", dpi=95)
    plt.close(fig)

    old_c, open_c = muted, "#2a78d6"
    mean = lambda kind, f: np.mean([f(r) for r in res[kind]], axis=0)
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.6), constrained_layout=True)
    tm = (10, 20, 30, 40, 50, 60)
    panels = ((axs[0], "85% region area", "km²", lambda r: [r["area"][85][t] for t in tm], tm, 75, "Minutes after the LKP report"),
              (axs[1], "Particles that have come near no-go terrain", "Share of particles", lambda r: [r["near_by"][t] for t in (5, 10, 20, 30, 60)], (5, 10, 20, 30, 60), 1, "Minutes after the LKP report"),
              (axs[2], "Mean distance from the LKP", "km", lambda r: [r[f"dist{t}"] for t in (15, 30, 60)], (15, 30, 60), 11, "Minutes after the LKP report"),
              (axs[3], "30-min region still inside the 60-min region", "Share of the 30-min region", lambda r: [r["kept_30"][x] for x in XS], XS, 1, "Region x%"))
    for ax, title, yl, f, xs, top, xl in panels:
        for kind, c in (("old", old_c), ("open", open_c)):
            ax.plot(xs, mean(kind, f), color=c, lw=1.8, marker="o", ms=5.5, mec=surface)
        ax.set(ylim=(0, top), xticks=xs, xlabel=xl, ylabel=yl)
        ax.set_title(title)
        ax.grid(axis="y", color="#e1e0d9", lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[Line2D([], [], color=old_c, lw=1.8, marker="o", ms=6, mec=surface, label="Old centre scenarios 400-409"),
                        Line2D([], [], color=open_c, lw=1.8, marker="o", ms=6, mec=surface, label=f"Open-ground scenarios {g[0]['seed']}-{g[-1]['seed']}")],
               loc="outside lower center", ncol=2)
    fig.suptitle("The open-ground scenarios meet terrain later, and the region moves instead of only growing", x=.01, ha="left", fontsize=12)
    fig.savefig(out/"check.png", dpi=115)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [("open", s) for s in range(args.seed0, args.seed0 + args.n_scen)] + [("old", s) for s in range(400, 410)]
    with Pool(args.workers) as pool:
        done = pool.map(job, jobs, chunksize=1)
    res = {k: sorted((r for kk, r in done if kk == k), key=lambda r: r["seed"]) for k in ("old", "open")}
    txt = summarize(res)
    (args.output/"check.md").write_text(txt + "\n")
    plot(res, args.output)
    print(txt)


if __name__ == "__main__":
    main()
