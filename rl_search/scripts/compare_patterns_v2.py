"""Pattern baseline v2 (cyclic re-planning) vs v1 and greedy variants on the same scenarios.

v2: every cycle (7.5 / 10 / 15 min) the planner finds density peaks of the remaining belief at mid-cycle and
places an IAMSAR pattern by cross-entropy optimisation (rl_search/patterns_v2.py). Per-kind runs keep one
pattern kind; "mixed" lets the planner pick the kind each cycle. Greedy90: 90 s lookahead, re-plan every 30 s.
v1 numbers (single placement for the whole flight) are read from compare_patterns/results.json if present.

    python -m rl_search.scripts.compare_patterns_v2 --n-scen 8 --workers 2
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from rl_search.scripts.compare_patterns import T_GRID, curve, detection_times, truth_tracks


def methods(args):
    from rl_search.patterns import KINDS
    from rl_search.patterns_v2 import CyclicPatternPlanner, Greedy90
    m = {}
    for k in KINDS:
        m[f"v2 {k} T10"] = (lambda k=k, s=0: CyclicPatternPlanner(kinds=(k,), cycle_s=600, seed=s), False)
    for T in (450, 600, 900):
        m[f"v2 mixed T{T/60:g}"] = (lambda T=T, s=0: CyclicPatternPlanner(cycle_s=T, seed=s), T == 600)
    m["greedy90"] = (lambda s=0: Greedy90(), True)
    return m


def run_scenario(job):
    seed, args = job
    warnings.filterwarnings("ignore")
    from rl_search.config import get_preset
    from rl_search.patterns import run_episode
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scenario import make_scenario
    from rl_search.terrain import load_terrain
    t0 = time.perf_counter()
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    preset, preset1 = get_preset("rc_cessna", [f"search.pod={args.pod}"]), get_preset("rc_cessna", ["search.pod=1"])
    env, env1 = RLSearchEnv(terr, preset, RLConfig(stage=3), seed=seed), RLSearchEnv(terr, preset1, RLConfig(stage=3), seed=seed)
    env.fast = env1.fast = True
    scen = make_scenario(env.sampler, seed)
    opts = dict(scenario=scen, stage=3)
    truths = truth_tracks(terr, preset, scen, args.truths, seed + 7919)
    t_h = preset.handover.t_plan_s
    out = dict(seed=int(seed), methods={})
    for name, (make, with_pod1) in methods(args).items():
        if args.only and not any(o in name for o in args.only):
            continue
        tm = time.perf_counter()
        pol = make(s=seed)
        log = run_episode(env, opts, pol)
        sec = time.perf_counter() - tm
        row = dict(pos=round(log.pos[-1], 4), pos_curve=curve(log, t_h), sec=round(sec, 1),
                   cycles=getattr(pol, "log", None), path=np.array(log.xy)[::6].round(0).tolist())
        if with_pod1:
            log1 = run_episode(env1, opts, make(s=seed), keep_cells=True)
            ttd = detection_times(env1, log1.cells, truths, t_h)
            row |= dict(pos_pod1=round(log1.pos[-1], 4), detect_frac=float(np.isfinite(ttd).mean()),
                        ttd_min=[round(float(x), 2) if np.isfinite(x) else None for x in ttd])
        out["methods"][name] = row
    out["sec"] = round(time.perf_counter() - t0, 1)
    print(f"seed {seed} done in {out['sec']} s: " + ", ".join(f"{k} {v['pos']:.3f}" for k, v in out["methods"].items()),
          flush=True)
    return out


def summarize(results, v1, out_dir, n_cycles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = [r["seed"] for r in results]
    rows = {}
    for n in results[0]["methods"]:
        rows[n] = dict(pos=[r["methods"][n]["pos"] for r in results],
                       ttd=[r["methods"][n].get("ttd_min") for r in results],
                       sec=np.mean([r["methods"][n]["sec"] for r in results]),
                       curve=np.mean([r["methods"][n]["pos_curve"] for r in results], axis=0))
    if v1:
        by = {r["seed"]: r for r in v1}
        if all(s in by for s in seeds):
            for n, label in (("spiral", "v1 spiral (best v1 pattern)"), ("greedy", "v1 greedy (15 s)"), ("seek", "v1 seek")):
                rows[label] = dict(pos=[by[s]["methods"][n]["pos"] for s in seeds],
                                   ttd=[by[s]["methods"][n]["ttd_min"] for s in seeds], sec=np.nan,
                                   curve=np.mean([by[s]["methods"][n]["pos_curve"] for s in seeds], axis=0))
    best = np.max([rows[n]["pos"] for n in rows], axis=0)
    lines = ["| method | POS (POD 0.3) mean ± sd | min | best in scenarios | true target found (POD 1) | found ≤ 30 min | sec / episode |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(rows, key=lambda n: -np.mean(rows[n]["pos"]))
    for n in order:
        r = rows[n]
        pos = np.array(r["pos"])
        if r["ttd"][0] is not None:
            ttd = np.concatenate([[np.inf if x is None else x for x in t] for t in r["ttd"]])
            det, d30 = f"{np.isfinite(ttd).mean():.1%}", f"{(ttd <= 30).mean():.1%}"
        else:
            det = d30 = "–"
        lines.append(f"| {n} | {pos.mean():.3f} ± {pos.std():.3f} | {pos.min():.3f} | {int((pos >= best - 1e-12).sum())}/{len(pos)} | "
                     f"{det} | {d30} | {r['sec']:.0f} |")
    lines.append(f"\nPer-scenario best of all methods: mean {best.mean():.3f}. Scenarios: {seeds}.")
    (out_dir/"table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    for n in order:
        ls = "--" if n.startswith("v1") else ("-" if "mixed" in n or "greedy" in n else ":")
        ax[0].plot(T_GRID, rows[n]["curve"], ls, lw=2 if ls != ":" else 1.2, label=n)
        if rows[n]["ttd"][0] is not None:
            ttd = np.concatenate([[np.inf if x is None else x for x in t] for t in rows[n]["ttd"]])
            ax[1].plot(T_GRID, [(ttd <= t).mean() for t in T_GRID], ls, lw=2, label=n)
    ax[0].set(title=f"Mean POS over time (POD 0.3), {len(results)} scenarios", xlabel="minutes after handover", ylabel="POS")
    ax[1].set(title="Single true targets found (POD 1)", xlabel="minutes after handover", ylabel="share found")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir/"curves.png", dpi=100)
    plt.close(fig)
    show = [n for n in ("v2 mixed T10", "v2 spiral T10", "greedy90") if n in results[0]["methods"]]
    fig, axs = plt.subplots(min(2, len(results)), len(show), figsize=(5*len(show), 5*min(2, len(results))), squeeze=False)
    for i, r in enumerate(results[:2]):
        for j, n in enumerate(show):
            xy = np.array(r["methods"][n]["path"])/1e3
            a = axs[i, j]
            a.plot(xy[:, 0], xy[:, 1], lw=.6)
            a.plot(*xy[0], "k*", ms=8)
            cyc = r["methods"][n].get("cycles") or []
            kinds = " → ".join(c["kind"][:4] for c in cyc[:n_cycles])
            a.set_title(f"s{r['seed']} {n}: POS {r['methods'][n]['pos']:.3f}\n{kinds}", fontsize=8)
            a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir/"paths.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--pod", type=float, default=0.3)
    ap.add_argument("--truths", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="Run only methods whose name contains one of these")
    ap.add_argument("--v1", type=Path, default=Path("rl_search/data/compare_patterns/results.json"))
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/compare_patterns_v2"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed0 + i, args) for i in range(args.n_scen)]
    with Pool(args.workers) as pool:
        results = pool.map(run_scenario, jobs)
    (args.output/"results.json").write_text(json.dumps(results))
    v1 = json.loads(args.v1.read_text()) if args.v1.exists() else None
    summarize(results, v1, args.output, 7)


if __name__ == "__main__":
    main()
