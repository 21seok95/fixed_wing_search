"""Trained PPO models against seek and straight on the compare_greedy scenarios, held-out POS.

Run from the repository root:

    python -m rl_search.scripts.compare_rl --models final=rl_search/data/rl/stage3_pos/model_final.zip \
        --set search.fov_cross_m=286 search.fov_along_m=207 search.pod=0.7 \
        --rl detect_model=lrc w_area=0 pos_ref=0.45 --output rl_search/data/rl/stage3_pos/analysis

Each episode flies on the scenario's particles, then the same actions replay on fresh particles (held-out POS,
same as compare_greedy). ``results.json`` also gets the share of the flight inside the particle cloud (radius
holding 80 % of the live particles, around their median) and the POS gained there.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl_search.scripts.compare_greedy import COLORS, GRID, INK, MUTED, make_env
from rl_search.scripts.check_sarops import holdout_scenario
from rl_search.scripts.train_ppo import parse_kv

RL_COLOR = COLORS["greedy90"]


def rollout(path, scen, seed, overrides, rl):
    """Deterministic policy on a scenario. Returns (actions, POS)."""
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_gym import RLSearchGym
    torch.set_num_threads(1)
    gym_env = RLSearchGym(preset="rc_cessna", overrides=overrides, rl=rl)
    model = PPO.load(path, device="cpu")
    obs, _ = gym_env.reset(seed=seed, options=dict(scenario=scen))
    acts, done = [], False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = gym_env.step(a)
        acts.append(float(np.ravel(a)[0]))
        done = term or trunc
    return acts, info["pos_policy"]


def inside_cloud(scen, xy, t):
    """Per decision: is the aircraft inside the radius holding 80 % of the live particles?"""
    out = []
    for p, tt in zip(xy, t):
        k = min(int(round(tt/scen.save_dt_s)), len(scen.track) - 1)
        pts = scen.track[k][(scen.exit_step < 0) | (scen.exit_step > k)]
        c = np.median(pts, axis=0)
        out.append(np.hypot(*(p - c)) <= np.quantile(np.hypot(*(pts - c).T), .8))
    return np.array(out)


def cloud_stats(scen, xy, t, pos):
    """Share of decisions inside the cloud, POS gained per minute inside and outside it."""
    xy, t, pos = np.array(xy), np.array(t), np.array(pos)
    inside = inside_cloud(scen, xy, t)[1:]
    gain, dt = np.diff(pos), np.diff(t)/60
    return dict(inside=float(inside.mean()), inside_min=float(dt[inside].sum()), gain_inside=float(gain[inside].sum()),
                gain_outside=float(gain[~inside].sum()))


def run(job):
    seed, name, path, overrides, rl = job
    warnings.filterwarnings("ignore")
    from rl_search.patterns import run_episode, seek_action
    from rl_search.scenario import make_scenario
    env = make_env(seed)
    scen = make_scenario(env.sampler, seed)
    same, hold = dict(scenario=scen, stage=3), dict(scenario=holdout_scenario(env, scen, seed + 10**6), stage=3)
    if path:
        acts, _ = rollout(path, scen, seed, overrides, rl)
    elif name == "seek":
        acts = []
        run_episode(env, same, lambda e: acts.append(float(seek_action(e))) or acts[-1])
    else:
        acts = [0.0]*2000
    it = iter(acts)
    log = run_episode(env, same, lambda e: next(it))
    it = iter(acts)
    h = run_episode(env, hold, lambda e: next(it))
    print(seed, name, round(log.pos[-1], 3), round(h.pos[-1], 3), flush=True)
    return dict(seed=seed, name=name, pos=log.pos[-1], hold=h.pos[-1], t=log.t, pos_t=log.pos, hold_t=h.pos,
                xy=np.array(log.xy).tolist(), acts=acts[:len(log.t) - 1],
                **cloud_stats(scen, log.xy, log.t, log.pos))


def sarops_stats(x):
    warnings.filterwarnings("ignore")
    from rl_search.scenario import make_scenario
    env = make_env(x["seed"])
    m = x["methods"]["SAROPS"]
    stats = cloud_stats(make_scenario(env.sampler, x["seed"]), m["xy"], m["t"], m["pos_t"])
    return dict(seed=x["seed"], name="SAROPS", hold=m["hold"], pos=m["pos"], **stats)


def plot_summary(rows, greedy, out: Path, rl="final"):
    """Mean held-out POS per method; share of the flight inside the cloud against POS, per scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    by = lambda n: [r for r in rows if r["name"] == n]
    means = {"SAROPS": np.mean([r["hold"] for r in by("SAROPS")]), "greedy90": greedy,
             "seek": np.mean([r["hold"] for r in by("seek")]), "straight": np.mean([r["hold"] for r in by("straight")]),
             "RL": np.mean([r["hold"] for r in by(rl)])}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    names = list(means)[::-1]
    a1.barh(names, [means[n] for n in names], height=.6, color=[RL_COLOR if n == "RL" else "#b9b8b3" for n in names])
    for i, n in enumerate(names):
        a1.annotate(f"{means[n]:.3f}", (means[n], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=10)
    a1.set(xlabel="POS, mean of scenarios (held-out particles)", xlim=(0, .5), title=f"Same {len(by(rl))} scenarios")
    a1.grid(axis="x", color=GRID, lw=.8)
    a1.set_axisbelow(True)
    a1.spines[["top", "right"]].set_visible(False)
    for n, c, lab in (("SAROPS", COLORS["SAROPS"], "SAROPS"), (rl, RL_COLOR, "RL")):
        r = by(n)
        a2.scatter([x["inside"]*100 for x in r], [x["hold"] for x in r], s=70, color=c, edgecolor="white", lw=1.5, label=lab, zorder=3)
    a2.set(xlim=(-3, 103), ylim=(0, None), xlabel="flight time inside the particle cloud (%)",
           ylabel="POS (held-out particles)", title="Each scenario: time in the cloud and POS")
    a2.legend(loc="upper left", frameon=False)
    a2.grid(color=GRID, lw=.8)
    a2.set_axisbelow(True)
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stage 3 PPO against the planners, new sensor + LRC", fontsize=13)
    fig.savefig(out/"summary.png", dpi=130)
    plt.close(fig)


def plot_paths(res_rl, res_plan, seed, out: Path):
    """SAROPS and the RL model on one scenario, three time slices."""
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.plotting import draw_rl_compare
    from rl_search.scenario import make_scenario
    env = make_env(seed)
    opts = dict(scenario=make_scenario(env.sampler, seed), stage=3)
    runs = {}
    for name, x in (("SAROPS", res_plan), ("RL", res_rl)):
        t = np.array(x["t"])
        times = [t[min(int(round(f*(len(t) - 1))), len(t) - 1)] for f in (1/3, 2/3)]
        snaps = []
        env.reset(options=opts)
        for a in x["acts"]:
            if any(abs(env.t - s) < 1e-6 for s in times):
                snaps.append((env.t, env.belief.w.copy(), env.pos_cum))
            env.step([a])
        snaps.append((env.t, env.belief.w.copy(), env.pos_cum))
        runs[name] = (np.array(x["xy"]), t, snaps)
    title = f"Scenario {seed}: held-out POS SAROPS {res_plan['hold']:.3f}, RL {res_rl['hold']:.3f}"
    fig = draw_rl_compare(env.terr, env.belief, runs, title)
    fig.savefig(out/f"paths_s{seed}.png", dpi=85)
    plt.close(fig)


def make_plots(out: Path, planners: Path, seeds, rl="final"):
    plan = json.loads((planners/"results.json").read_text())
    plan = plan if isinstance(plan, list) else plan["results"]
    res = json.loads((out/"results.json").read_text())
    with ProcessPoolExecutor(6) as ex:
        sar = list(ex.map(sarops_stats, plan))
    (out/"sarops_cloud.json").write_text(json.dumps(sar))
    plot_summary(res + sar, np.mean([x["methods"]["greedy90"]["hold"] for x in plan]), out, rl)
    for s in seeds:
        plot_paths(next(r for r in res if r["seed"] == s and r["name"] == rl),
                   next(x["methods"]["SAROPS"] for x in plan if x["seed"] == s), s, out)
    for n, r in (("SAROPS", sar), (rl, [x for x in res if x["name"] == rl])):
        print(n, f"inside {np.mean([x['inside'] for x in r]):.2f}, {np.mean([x['inside_min'] for x in r]):.1f} min, "
              f"POS/min inside {sum(x['gain_inside'] for x in r)/sum(x['inside_min'] for x in r):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[], help="name=path.zip")
    ap.add_argument("--plot-only", action="store_true", help="Figures from results.json and --planners")
    ap.add_argument("--planners", type=Path, default=Path("rl_search/data/compare_greedy"))
    ap.add_argument("--path-seeds", nargs="*", type=int, default=[328, 316])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--rl", nargs="*", default=[])
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--n-scen", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.plot_only:
        make_plots(args.output, args.planners, args.path_seeds)
        return

    rl = parse_kv(args.rl) | {"stage": 3}
    models = dict(m.split("=", 1) for m in args.models)
    if not args.no_baselines:
        models |= dict(seek=None, straight=None)
    seeds = range(args.seed0, args.seed0 + args.n_scen)
    jobs = [(s, n, p, args.set, rl) for s in seeds for n, p in models.items()]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs))
    (args.output/"results.json").write_text(json.dumps(res))
    for n in models:
        r = [x for x in res if x["name"] == n]
        print(f"{n}: held-out POS {np.mean([x['hold'] for x in r]):.3f}, inside cloud {np.mean([x['inside'] for x in r]):.2f}")


if __name__ == "__main__":
    main()
