"""How much of the distribution is inside each rectangle while it is flown. Saved rectangles and transit times, no planner."""
import json, sys
import numpy as np
from rl_search.mission import ParticleBelief
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
D = [json.loads(l) for l in open(sys.argv[1])]
out = []
for d in D:
    if d["mode"] != "free" or d["config"] != "pos": continue
    seed = d["seed"]
    env = make_envs(1, seed)[0]
    scen = open_scenario(env.sampler, seed)
    b = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
    saved = next(x for x in map(json.loads, open("rl_search/data/center_open/sweep/results.jsonl")) if x["seed"] == seed and x["n"] == 5 and x["config"] == "pos")
    def inside(q, t):
        P = b.positions(t); c, s = np.cos(q[5]), np.sin(q[5]); dx, dy = P[:, 0]-q[1], P[:, 1]-q[2]
        return float(((np.abs(dx*c+dy*s) <= q[3]/2) & (np.abs(-dx*s+dy*c) <= q[4]/2) & b.active(t)).mean())
    def med(t):
        P = b.positions(t)[b.active(t)]
        return np.median(P, axis=0)
    for r in d["rows"]:
        q = saved["rects"][r["u"]][r["i"]]
        t0, t1 = q[0] + r["tr"], q[0] + r["tr"] + 600.0
        m0 = med(q[0] + 300.0); m1 = med((t0+t1)/2)
        # how far the distribution core has moved past the rectangle at lane end, along the mean motion
        out.append(dict(seed=seed, u=r["u"], i=r["i"], tr=r["tr"], in0=inside(q, t0), inm=inside(q, (t0+t1)/2), in1=inside(q, t1), shift=float(np.hypot(*(m1-m0))),
                        c_to_med=float(np.hypot(q[1]-m1[0], q[2]-m1[1])), area=q[3]*q[4]/1e6))
json.dump(out, open(sys.argv[2], "w"))
G = (("first (all UAVs)", lambda r: r["i"] == 0), ("second", lambda r: r["i"] == 1), ("third+", lambda r: r["i"] >= 2))
for n, f in G:
    g = [r for r in out if f(r)]
    a, m, e = [np.mean([r[k] for r in g]) for k in ("in0", "inm", "in1")]
    print(n, "n", len(g), f"particles in rectangle start/mid/end {a:.3f}/{m:.3f}/{e:.3f}, end over start {e/a:.2f}; core moved plan+5 min to lane mid {np.mean([r['shift'] for r in g]):.0f} m; rectangle centre to median at lane mid {np.mean([r['c_to_med'] for r in g]):.0f} m")
