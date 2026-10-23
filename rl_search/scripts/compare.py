"""Stages 3, 4 and 6: compare search methods on the evaluation set under the same energy budget.

Every run is one (scenario, config, method, wind mode). A *config* is a named set of preset
overrides, ``NAME:key=value,key=value`` (empty after the colon = the preset as is), so any setting
can be compared without code changes. Methods: straight, lawnmower, greedy, tree (and any planner
in planners.PLANNERS). All are flown as waypoint missions by the path follower (runner.py).

Outputs in rl_search/data/compare/<tag>/: results.json (all runs), table.csv / table.md (mean and
std per config x method x wind), pos_time.png, pos_energy.png, scenarios.png; with
--save-missions N the first N scenarios also get search_<method>_<config>[_<wind>]/mission.npz in
their export folder for mission_viewer.py.

Run from the repository root:
    python -m rl_search.scripts.compare --n 50                                         # stage 3: POD 1 and 0.3
    python -m rl_search.scripts.compare --methods greedy tree --tag plants \\
        --configs fw:plant.kind=fixed_wing kin:plant.kind=kinematic                     # stage 4
    python -m rl_search.scripts.compare --methods greedy lawnmower --tag wind \\
        --configs pod0.3: --wind-modes unaware aware replan                             # stage 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker; the pool provides the parallelism
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.config import apply_overrides, finalize, preset_from_meta
from rl_search.runner import METHODS, WIND_MODES, run_method
from rl_search.scenario import load_scenario
from rl_search.scripts.export_plan import save_viewer_mission
from rl_search.scripts.export_scenario import export, export_dir
from rl_search.terrain import load_terrain

COLORS = {"straight": "#757575", "lawnmower": "#fb8c00", "greedy": "#43a047", "tree": "#3949ab"}
METRICS = ("final_pos", "pos_ratio", "bound", "energy_used", "pos_per_energy", "searched_km2", "compute_s",
           "planned_pos", "exec_loss", "sched_err_end_s", "sched_err_abs_max_s", "cross_track_mean_m",
           "n_waypoints", "n_replans", "t_end_s", "max_roll_deg")
_w: dict = {}


def parse_config(text: str):
    name, _, rest = text.partition(":")
    return name, [kv for kv in rest.split(",") if kv]


def _init(terrain_path: str):
    warnings.filterwarnings("ignore", message="Roll exceeds the gimbal")
    _w["terr"] = load_terrain(Path(terrain_path))


def _job(args):
    scen_dir, cfg_name, overrides, base_overrides, method, wind, keep = args
    meta = json.loads((Path(scen_dir)/"meta.json").read_text())
    preset = preset_from_meta(meta, base_overrides)
    preset = finalize(apply_overrides(preset, overrides))
    scen = load_scenario(Path(scen_dir), mmap=True)
    t0 = time.perf_counter()
    r = run_method(preset, _w["terr"], scen, method, wind, keep_logs=True)
    log = r.pop("log")
    r.pop("mission"), r.pop("plan_xy"), r.pop("plan_t")
    sub = slice(None, None, 5)
    r |= dict(config=cfg_name, scenario_dir=scen_dir, wall_s=time.perf_counter() - t0,
              curve_t=log["t_s"][sub].tolist() + [float(log["t_s"][-1])],
              curve_pos=log["pos_cum"][sub].tolist() + [float(log["pos_cum"][-1])],
              curve_e=(preset.energy.budget_s - log["energy_left"][sub]).tolist()
              + [float(preset.energy.budget_s - log["energy_left"][-1])] if preset.energy.enabled else [])
    if "planned" in r:
        r["planned_pos"] = r["planned"]["final_pos"]
        r["exec_loss"] = (r["final_pos"] - r["planned_pos"])/max(r["planned_pos"], 1e-12)
    return r, log if keep else None


def table(results, keys):
    rows = []
    for key in keys:
        sel = [r for r in results if (r["config"], r["method"], r["wind_mode"]) == key]
        row = dict(config=key[0], method=key[1], wind=key[2], n=len(sel))
        for m in METRICS:
            v = np.array([r.get(m, np.nan) for r in sel], float)
            v = v[np.isfinite(v)]
            row[m] = (float(v.mean()), float(v.std())) if len(v) else (np.nan, np.nan)
        rows.append(row)
    return rows


def write_tables(rows, out: Path):
    cols = ["config", "method", "wind", "n"] + list(METRICS)
    with open(out/"table.csv", "w") as f:
        f.write(",".join(cols[:4] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]) + "\n")
        for r in rows:
            f.write(",".join([str(r[c]) for c in cols[:4]] + [f"{x:.6g}" for m in METRICS for x in r[m]]) + "\n")
    show = ("final_pos", "pos_ratio", "energy_used", "pos_per_energy", "searched_km2", "compute_s",
            "exec_loss", "sched_err_end_s", "n_waypoints")
    lines = ["| config | method | wind | n | " + " | ".join(show) + " |",
             "|" + "---|"*(4 + len(show))]
    for r in rows:
        cells = []
        for m in show:
            mu, sd = r[m]
            scale = 1e3 if m == "pos_per_energy" else 1.0
            cells.append("-" if not np.isfinite(mu) else f"{mu*scale:.4g} ± {sd*scale:.2g}")
        lines.append(f"| {r['config']} | {r['method']} | {r['wind']} | {r['n']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n\npos_per_energy is POS per 1000 trim-seconds; exec_loss = executed/planned POS - 1.\n"
    (out/"table.md").write_text(text)
    return text


def curve_figures(results, keys, out: Path):
    groups = sorted({(k[0], k[2]) for k in keys})
    for kind, xkey, xlabel, scale in (("pos_time", "curve_t", "time (min)", 1/60),
                                      ("pos_energy", "curve_e", "energy used (trim-min)", 1/60)):
        fig, axes = plt.subplots(1, len(groups), figsize=(5.5*len(groups), 4.5), squeeze=False, layout="constrained")
        for ax, (cfg, wind) in zip(axes[0], groups):
            for key in keys:
                if (key[0], key[2]) != (cfg, wind):
                    continue
                sel = [r for r in results if (r["config"], r["method"], r["wind_mode"]) == key and r[xkey]]
                if not sel:
                    continue
                grid = np.linspace(0, max(max(r[xkey]) for r in sel), 200)
                ys = np.array([np.interp(grid, r[xkey], r["curve_pos"]) for r in sel])
                col = COLORS.get(key[1])
                ax.plot(grid*scale, ys.mean(0), color=col, label=f"{key[1]} ({ys[:, -1].mean():.3f})")
                ax.fill_between(grid*scale, np.percentile(ys, 25, 0), np.percentile(ys, 75, 0), color=col, alpha=.15)
            ax.set_title(f"{cfg}, wind {wind}: mean POS (IQR shaded)", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("POS")
            ax.legend(fontsize=8)
        fig.savefig(out/f"{kind}.png", dpi=130)
        plt.close(fig)


def scenario_figure(results, keys, out: Path):
    scen = sorted({r["scenario"] for r in results})
    fig, ax = plt.subplots(figsize=(max(8, .28*len(scen)*len(keys)/2), 4.5), layout="constrained")
    w = .8/len(keys)
    for i, key in enumerate(keys):
        by = {r["scenario"]: r["final_pos"] for r in results if (r["config"], r["method"], r["wind_mode"]) == key}
        ax.bar(np.arange(len(scen)) + (i - (len(keys) - 1)/2)*w, [by.get(s, np.nan) for s in scen], w,
               color=COLORS.get(key[1]), alpha=.35 + .65*(1 - i/len(keys)) if len(keys) > 4 else .9,
               label="/".join(k for k in key if k != "none"))
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([f"s{s}"[-3:] for s in scen], fontsize=7, rotation=90)
    ax.set_ylabel("final POS (executed)")
    ax.legend(fontsize=7, ncol=4)
    fig.savefig(out/"scenarios.png", dpi=130)
    plt.close(fig)


def save_viewer(log, r):
    scen_dir = Path(r["scenario_dir"])
    meta = json.loads((scen_dir/"meta.json").read_text())
    run_dir = export(load_scenario(scen_dir, mmap=False), _w["terr"], export_dir(scen_dir))
    name = f"{r['method']}_{r['config']}" + ("" if r["wind_mode"] == "none" else f"_{r['wind_mode']}")
    save_viewer_mission(run_dir/f"search_{name}", dict(r, log=log), preset_from_meta(meta), name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=Path("rl_search/data/scenarios/rc_cessna/eval"))
    ap.add_argument("--n", type=int, default=50, help="First n scenarios of the set")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--configs", nargs="+", default=["pod1:search.pod=1", "pod0.3:search.pod=0.3"])
    ap.add_argument("--wind-modes", nargs="+", default=["none"], choices=WIND_MODES)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Overrides for every config")
    ap.add_argument("--tag", default="methods", help="Output folder name under rl_search/data/compare/")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--save-missions", type=int, default=0, help="Viewer missions for the first N scenarios")
    args = ap.parse_args()

    dirs = sorted(d for d in args.scenarios.iterdir() if (d/"meta.json").exists())[:args.n]
    meta0 = json.loads((dirs[0]/"meta.json").read_text())
    configs = [parse_config(c) for c in args.configs]
    keys = [(c, m, w) for c, _ in configs for w in args.wind_modes for m in args.methods]
    save = {str(d) for d in dirs[:args.save_missions]}
    jobs = [(str(d), c, ov, args.set, m, w, str(d) in save) for d in dirs for c, ov in configs
            for w in args.wind_modes for m in args.methods]
    jobs.sort(key=lambda j: j[4] != "tree")
    out = Path("rl_search/data/compare")/args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} runs: {len(dirs)} scenarios x configs {[c for c, _ in configs]} x methods {args.methods} "
          f"x wind {args.wind_modes}, {args.workers} workers -> {out}")
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(meta0["terrain"],)) as pool:
        _init(meta0["terrain"])
        futs = [pool.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r, log = f.result()
            results.append(r)
            if log is not None:
                save_viewer(log, r)
            if i % max(1, len(jobs)//20) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    results.sort(key=lambda r: (r["scenario"], r["config"], r["method"], r["wind_mode"]))
    (out/"results.json").write_text(json.dumps(dict(args=vars(args) | dict(scenarios=str(args.scenarios)),
                                                    results=results), default=str))
    rows = table(results, keys)
    print(write_tables(rows, out))
    curve_figures(results, keys, out)
    scenario_figure(results, keys, out)
    print(f"Saved {out}/table.md, table.csv, results.json, pos_time.png, pos_energy.png, scenarios.png "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
