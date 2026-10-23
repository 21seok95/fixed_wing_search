"""Build the fixed-seed evaluation set and check the training scenario distribution.

Evaluation scenarios use seeds ``eval_seed + i`` and are saved (track.npy, exit_step.npy, meta.json).
The distribution check draws ``n_stats`` scenarios with the same seed stream as
``ScenarioGenerator(seed=stats_seed)`` and only keeps their metrics.

Run from the repository root:
    python -m rl_search.scripts.make_scenarios --n-eval 5 --n-stats 100
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rl_search.plotting import draw_terrain
from rl_search.scenario import ScenarioConfig, ScenarioSampler, make_scenario, save_scenario
from rl_search.terrain import load_terrain

DEFAULT_TERRAIN = Path("rl_search/data/terrain_seed7_60km.npz")
_worker: dict = {}


def _init(terrain_path: str, cfg: ScenarioConfig):
    _worker["terrain"] = Path(terrain_path)
    _worker["sampler"] = ScenarioSampler(load_terrain(Path(terrain_path)), cfg)


def _eval_job(args):
    seed, out = args
    sampler = _worker["sampler"]
    scen = make_scenario(sampler, seed)
    save_scenario(Path(out), scen, sampler.cfg, _worker["terrain"])
    k_end = min(int(round(sampler.cfg.flight_s/scen.save_dt_s)), len(scen.track) - 1)
    in_map = (scen.exit_step < 0) | (scen.exit_step > k_end)
    return scen.meta(), np.asarray(scen.track[k_end][in_map][::5])


def _stats_job(seed):
    return make_scenario(_worker["sampler"], seed).meta()


def summary_figure(terr, metas, clouds, path: Path):
    fig, ax = plt.subplots(figsize=(11, 8.6), layout="constrained")
    colors = plt.cm.tab10(np.arange(len(metas)) % 10)
    draw_terrain(ax, terr, legend_extra=(
        Line2D([], [], marker="^", color="k", ls="", markersize=8, label="UAV launch"),
        Line2D([], [], marker="*", color="k", ls="", markersize=11, label="LKP + heading"),
        Line2D([], [], marker="o", color="k", ls="", markersize=3, label="Particles at flight end")))
    for m, pts, col in zip(metas, clouds, colors):
        (sx, sy), (lx, ly) = np.array(m["start"])/1000, np.array(m["lkp"])/1000
        ax.scatter(pts[:, 0]/1000, pts[:, 1]/1000, s=5, color=col, alpha=.8, edgecolors="none", zorder=7)
        ax.plot([sx, lx], [sy, ly], ls="--", lw=1, color=col, zorder=8)
        ax.scatter(sx, sy, marker="^", s=90, color=col, edgecolor="white", zorder=9, clip_on=False)
        ax.scatter(lx, ly, marker="*", s=180, color=col, edgecolor="white", zorder=9)
        h = np.deg2rad(m["heading_deg"])
        ax.annotate("", xy=(lx + 2.5*np.cos(h), ly + 2.5*np.sin(h)), xytext=(lx, ly), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.text(lx + .5, ly - 1.2, f"s{m['seed']}", color=col, fontsize=8, weight="bold", zorder=9)
    ax.set_title(f"Evaluation scenarios ({len(metas)}): launch, LKP and particles at the end of the flight")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram_figure(metas, path: Path):
    def m(k):
        v = [x["metrics"][k] if k in x["metrics"] else x[k] for x in metas]
        return np.array([np.nan if a is None else a for a in v], float)
    panels = [("escaped_end", "Escaped mass at flight end", 1),
              ("intercept_s", "Centroid intercept time (min)", 1/60),
              ("reachable", "Reachable mass", 1),
              ("start_lkp_km", "Launch to LKP (km)", 1),
              ("speed_kmh", "Target mean speed (km/h)", 1),
              ("gen_s", "Generation time (s)", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    for ax, (k, title, scale) in zip(axes.ravel(), panels):
        v = m(k)*scale
        v = v[np.isfinite(v)]
        ax.hist(v, bins=20, color="#3949ab", alpha=.85)
        ax.set_title(f"{title}\nmedian {np.median(v):.3g}, min {v.min():.3g}, max {v.max():.3g}", fontsize=9)
    fig.suptitle(f"Training scenario distribution ({len(metas)} scenarios)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", type=Path, default=DEFAULT_TERRAIN)
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/scenarios"))
    ap.add_argument("--n-eval", type=int, default=5)
    ap.add_argument("--eval-seed", type=int, default=1_000_000, help="Evaluation seeds start here")
    ap.add_argument("--n-stats", type=int, default=100, help="Scenarios for the distribution check (not saved)")
    ap.add_argument("--stats-seed", type=int, default=0, help="Seed of the ScenarioGenerator stream")
    ap.add_argument("--particles", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    cfg = ScenarioConfig()
    if args.particles:
        cfg.target.particles = args.particles
    terr = load_terrain(args.terrain)
    eval_dir = args.output/"eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.terrain), cfg))

    t0 = time.perf_counter()
    jobs = [(args.eval_seed + i, str(eval_dir/f"s{args.eval_seed + i}")) for i in range(args.n_eval)]
    results = list(pool.map(_eval_job, jobs))
    metas = [r[0] for r in results]
    print(f"Saved {len(metas)} evaluation scenarios in {eval_dir} ({time.perf_counter()-t0:.0f} s)")
    for x in metas:
        mt = x["metrics"]
        t_int = "none" if mt["intercept_s"] is None else f"{mt['intercept_s']/60:.1f} min"
        print(f"  s{x['seed']}: LKP {np.round(np.array(x['lkp'])/1000, 1)} km, heading {x['heading_deg']:.0f} deg, "
              f"{x['speed_kmh']:.1f} km/h | launch {np.round(np.array(x['start'])/1000, 1)} km, "
              f"{mt['start_lkp_km']:.1f} km away | escaped {mt['escaped_end']:.1%}, "
              f"intercept {t_int}, reachable {mt['reachable']:.1%} | {x['gen_s']:.1f} s")
    summary_figure(terr, metas, [r[1] for r in results], args.output/"eval_overview.png")
    print(f"Saved {args.output/'eval_overview.png'}")

    if args.n_stats > 0:
        t0 = time.perf_counter()
        rng = np.random.default_rng(args.stats_seed)
        seeds = [int(rng.integers(2**62)) for _ in range(args.n_stats)]
        stats = list(pool.map(_stats_job, seeds))
        wall = time.perf_counter() - t0
        (args.output/"train_stats.json").write_text(json.dumps(stats, indent=1))
        histogram_figure(stats, args.output/"train_stats.png")
        gen = np.array([s["gen_s"] for s in stats])
        print(f"Distribution check: {len(stats)} scenarios in {wall:.0f} s with {args.workers} workers "
              f"(per scenario {gen.mean():.1f} s on one core) | saved {args.output/'train_stats.png'}")
    pool.shutdown()


if __name__ == "__main__":
    main()
