"""Print the tables of docs/rl_rectangle_policy_discussion.md from the saved results (no simulation).

    PYTHONPATH=. python rl_search/scripts/rect_rl_probe/summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path("rl_search/data")
R = DATA/"rect_rl_probe"


def load(p):
    d = json.loads(Path(p).read_text())
    return d if isinstance(d, list) else d["results"]


def main():
    ladder = load(R/"ladder.json")
    roll = {r["seed"]: r for r in load(R/"rollout.json")}
    team = load(R/"team.json")
    marg = load(R/"marginal.json")
    multi = load(DATA/"compare_multi_sarops/results.json")
    fixed = {x["seed"]: x["methods"]["SAROPS"] for x in load(DATA/"compare_greedy/results.json")}
    flat = {x["seed"]: x for x in load(DATA/"rl/stage3_pos/analysis/results.json") if x["name"] == "final"}
    sar1 = {r["seed"]: r for r in multi if r["n"] == 1}
    seeds = [r["seed"] for r in ladder]

    H = {n: np.array([r["methods"][n]["hold"] for r in ladder]) for n in ladder[0]["methods"]}
    sec = {n: np.mean([r["methods"][n]["sec"] for r in ladder]) for n in ladder[0]["methods"]}
    H["sarops_task"] = np.array([sar1[s]["hold"] for s in seeds])
    sec["sarops_task"] = np.mean([sar1[s]["sec"] for s in seeds])
    H["sarops_fixed"] = np.array([fixed[s]["hold"] for s in seeds])
    sec["sarops_fixed"] = np.mean([fixed[s]["sec"] for s in seeds])
    H["rollout"] = np.array([roll[s]["hold"] for s in seeds])
    sec["rollout"] = np.mean([roll[s]["sec"] for s in seeds])
    H["flat_ppo"] = np.array([flat[s]["hold"] for s in seeds])

    print("1 UAV, held-out POS, scenarios", seeds[0], "-", seeds[-1])
    for n in ("random_turn", "rect_uniform", "flat_ppo", "rect_density", "loc_climb", "peak_greedy", "sarops_task", "rollout", "sarops_fixed"):
        s = f"{sec[n]:.1f} s" if n in sec else "-"
        print(f"  {n:14s} {H[n].mean():.3f}  {s}")

    def diff(a, b):
        d = H[a] - H[b]
        print(f"  {a} - {b}: {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}, higher in {(d > 0).sum()}/{len(d)}")
    diff("rollout", "sarops_task")
    diff("rollout", "peak_greedy")
    diff("sarops_fixed", "sarops_task")
    print(f"  candidates per decision in rollout: {np.mean([roll[s]['cands'] for s in seeds]):.1f}")
    print(f"  decisions per episode, peak_greedy: {np.mean([r['methods']['peak_greedy']['tasks'] for r in ladder]):.1f}")

    print("\n5 UAVs, separation check")
    for n in ("rect_density", "rect_density+check", "loc_climb", "loc_climb+check"):
        r = [x for x in team if x["name"] == n]
        print(f"  {n:20s} POS {np.mean([x['hold'] for x in r]):.3f}, below 300 m {np.mean([x['below'] for x in r]):.0f} s, "
              f"closest {min(x['dmin'] for x in r):.0f} m, no feasible option {sum(x['violations'] for x in r)}, "
              f"{np.mean([x['sec'] for x in r]):.1f} s/episode, {np.mean([x['plans'] for x in r]):.1f} decisions")
    r = [x for x in multi if x["n"] == 5]
    print(f"  {'SAROPS':20s} POS {np.mean([x['hold'] for x in r]):.3f}, below 300 m {np.mean([x['below'] for x in r]):.0f} s, "
          f"closest {min(x['dmin'] for x in r):.0f} m, no feasible option {sum(x['violations'] for x in r)}")

    print("\nSum of per-decision marginal POS against team POS (flown: location from probability + accordion shape)")
    for n in (1, 5):
        r = [x for x in marg if x["n"] == n]
        t, s = np.array([x["team"] for x in r]), np.array([x["sum_marginal"] for x in r])
        print(f"  {n} UAV: team {t.mean():.4f}, sum {s.mean():.4f}, largest difference {np.abs(t - s).max():.4f}, "
              f"approach flight {np.mean([x['approach_s'] for x in r])/n/3600*100:.1f} % of 60 min, {len(r)} scenarios")


if __name__ == "__main__":
    main()
