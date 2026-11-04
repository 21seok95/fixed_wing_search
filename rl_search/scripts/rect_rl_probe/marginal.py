"""Exploratory: does the per-decision marginal POS add up to the team POS?

At each planning event the planner sees weights w = (shared survival so far) x (what the others' committed lanes will still
add). Right after the plan, future_factor gives this UAV's own survival factor f for the task. Marginal gain of the decision
= sum w (1 - f) / N. With independent detections (product rule, SAROPS eq. 2) these should telescope to the team POS, apart
from flight outside committed tasks (approach phases).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def run(job):
    seed, n, name = job
    warnings.filterwarnings("ignore")
    import rl_search.multi as M
    from rl_search.accordion import AccordionPlanner, sweep_width
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_multi import make_envs
    from team import make_policy
    orig = M.future_factor
    gains = []

    def wrapped(env, pol):
        f, trk = orig(env, pol)
        snap = getattr(pol, "_w_at_plan", None)
        if snap is not None:
            gains.append(float((snap*(1 - f)).sum()/env.belief.n))
            pol._w_at_plan = None
        return f, trk
    M.future_factor = wrapped

    probe = make_envs(1, seed)[0]
    scen = make_scenario(probe.sampler, seed)
    opts = dict(scenario=scen, stage=3)
    W = sweep_width(probe)
    pols = [make_policy(name, seed + 13*u, W) for u in range(n)]
    for p in pols:
        base_choose = p.choose

        def choose(env, acc, _b=base_choose, _p=p):
            _p._w_at_plan = env.belief.w.copy()      # at this moment: shared survival x others' committed future
            return _b(env, acc)
        p.choose = choose
    res = M.run_team_launch(make_envs(n, seed), pols, opts, 60.0, deconflict=False, hard=False)
    M.future_factor = orig
    out = dict(seed=seed, n=n, team=float(res["pos"][-1]), sum_marginal=float(sum(gains)), plans=len(gains),
               approach_s=float(sum((b or a) - a for p in pols for a, b in p.approach)))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    jobs = [(s, n, "loc_climb") for s in range(300, 308) for n in (1, 5)]
    with Pool(4) as p:
        res = p.map(run, jobs, chunksize=1)
    Path(sys.argv[1]).write_text(json.dumps(res))
    for n in (1, 5):
        r = [x for x in res if x["n"] == n]
        t, s = np.array([x["team"] for x in r]), np.array([x["sum_marginal"] for x in r])
        print(f"n={n}: team POS {t.mean():.4f}, sum of marginals {s.mean():.4f}, max |diff| {np.abs(t - s).max():.4f}")
