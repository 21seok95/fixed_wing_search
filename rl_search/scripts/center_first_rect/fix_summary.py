"""Baseline against the first_fix variants (5 UAVs, A + POS, x 70, w 1, scenarios 500-509), from the saved flights."""
import json, sys
from pathlib import Path
import numpy as np
from rl_search.mission import ParticleBelief
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
B = Path("rl_search/data/center_open")
SETTINGS = (("no constraint", "sweep"), ("both rules", "no_overlap"))
VARIANTS = (("baseline", None), ("start peak at intercept", "intercept"), ("neighbouring lanes", "adjacent"), ("both fixes", "intercept_adjacent"))
R, C = 143.0, 25.0

def load(setting, tag):
    p = B/setting/"results.jsonl" if tag is None else B/"first_fix"/tag/setting/"results.jsonl"
    if not p.exists(): return {}
    out = {}
    for l in p.read_text().splitlines():
        r = json.loads(l)
        if r["n"] == 5 and r["config"] == "score" and r["x"] == 70 and r["w"] == 1: out[r["seed"]] = r
    return out

def in_rect(q, r, m=0.0):
    c, s = np.cos(r[5]), np.sin(r[5]); dx, dy = q[..., 0]-r[1], q[..., 1]-r[2]
    return (np.abs(dx*c+dy*s) <= r[3]/2+m) & (np.abs(-dx*s+dy*c) <= r[4]/2+m)

def conflict(a, b, m=0.0):
    d = np.array([b[1]-a[1], b[2]-a[2]])
    ca, sa, cb, sb = np.cos(a[5]), np.sin(a[5]), np.cos(b[5]), np.sin(b[5])
    ua, va, ub, vb = np.array([ca, sa]), np.array([-sa, ca]), np.array([cb, sb]), np.array([-sb, cb])
    for n in (ua, va, ub, vb):
        if abs(d @ n) > a[3]/2*abs(ua @ n)+a[4]/2*abs(va @ n)+(b[3]/2+m)*abs(ub @ n)+(b[4]/2+m)*abs(vb @ n): return False
    return True

def samples(r, nx=30, ny=12):
    u = (np.arange(nx)+.5)/nx*r[3]-r[3]/2; v = (np.arange(ny)+.5)/ny*r[4]-r[4]/2
    U, V = np.meshgrid(u, v); c, s = np.cos(r[5]), np.sin(r[5])
    return np.stack([r[1]+U*c-V*s, r[2]+U*s+V*c], -1)

def cur(rects, t):
    i = -1
    for k, r in enumerate(rects):
        if r[0] <= t: i = k
    return i

def swept(xy, tt, rects):
    """Area swept by 2+ UAVs (km2), split as launch/approach, moving, lanes."""
    n = len(xy); pts = np.concatenate(xy)
    x0, y0 = pts[:, 0].min()-R-C, pts[:, 1].min()-R-C
    w, h = int((pts[:, 0].max()+R+C-x0)//C)+1, int((pts[:, 1].max()+R+C-y0)//C)+1
    S = np.zeros((n, 3, h, w), bool); m = int(np.ceil(R/C))+1
    for u in range(n):
        for k in range(len(xy[u])-1):
            a, b = xy[u][k], xy[u][k+1]; tm = (tt[u][k]+tt[u][k+1])/2; i = cur(rects[u], tm)
            ph = 0 if i < 0 else (2 if tm >= rects[u][i][0]+rects[u][i][6] and in_rect((a+b)/2, rects[u][i], 150.0) else 1)
            ix0, ix1 = max(int((min(a[0], b[0])-x0)//C)-m, 0), min(int((max(a[0], b[0])-x0)//C)+m+1, w)
            iy0, iy1 = max(int((min(a[1], b[1])-y0)//C)-m, 0), min(int((max(a[1], b[1])-y0)//C)+m+1, h)
            X = x0+(np.arange(ix0, ix1)[None, :]+.5)*C; Y = y0+(np.arange(iy0, iy1)[:, None]+.5)*C
            ab = b-a; den = float(ab @ ab)
            s = np.clip(((X-a[0])*ab[0]+(Y-a[1])*ab[1])/den, 0, 1) if den > 0 else np.zeros_like(X*Y)
            S[u, ph, iy0:iy1, ix0:ix1] |= (X-a[0]-s*ab[0])**2+(Y-a[1]-s*ab[1])**2 <= R*R
    cell = C**2/1e6
    any_, nl, la = S.any(1).sum(0), S[:, 1:].any(1).sum(0), S[:, 2].sum(0)
    return dict(team=float((any_ > 0).sum()*cell), multi=float((any_ >= 2).sum()*cell), launch=float(((any_ >= 2) & (nl < 2)).sum()*cell),
                move=float(((nl >= 2) & (la < 2)).sum()*cell), lanes=float((la >= 2).sum()*cell))

rows = []
beliefs = {}
for sname, setting in SETTINGS:
    for vname, tag in VARIANTS:
        for seed, r in sorted(load(setting, tag).items()):
            if seed not in beliefs:
                env = make_envs(1, seed)[0]; scen = open_scenario(env.sampler, seed)
                beliefs[seed] = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
            b = beliefs[seed]
            xy, tt, rects = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]], r["rects"]
            plans = []
            for u in range(5):
                for i, q in enumerate(rects[u]):
                    others = [rects[v][j] for v in range(5) if v != u for j in [cur(rects[v], q[0])] if j >= 0]
                    ov = float(np.any([in_rect(samples(q), o) for o in others], axis=0).mean()) if others else 0.0
                    plans.append(dict(u=u, i=i, tp=q[0], conflict=any(conflict(q, o) for o in others), ov=ov, area=q[3]*q[4]/1e6))
            first = []
            for u in range(5):
                q = rects[u][0]
                inside = np.flatnonzero((tt[u] >= q[0]) & in_rect(xy[u], q, 150.0))
                t0 = tt[u][inside[0]] if len(inside) else q[0]
                t1 = rects[u][1][0] if len(rects[u]) > 1 else tt[u][-1]
                P0, P1, Pm = b.positions(t0), b.positions(t1), b.positions((t0+t1)/2)
                med = np.median(Pm[b.active((t0+t1)/2)], axis=0)
                first.append(dict(u=u, tp=q[0], lane0=float(t0), lane_min=float((t1-t0)/60), in0=float(in_rect(P0, q).mean()), in1=float(in_rect(P1, q).mean()),
                                  c_to_med=float(np.hypot(q[1]-med[0], q[2]-med[1])), area=q[3]*q[4]/1e6))
            rows.append(dict(setting=sname, variant=vname, seed=seed, pos=r["pos"], A=r["scores"]["70"]["A"], below=r["below"], by15=r.get("close_by15"), dmin=r["dmin"],
                             plans=plans, first=first, swept=swept(xy, tt, rects), n_plans=r["plans"]))
json.dump(rows, open(sys.argv[1], "w"))
se = lambda d: float(np.std(d, ddof=1)/np.sqrt(len(d))) if len(d) > 1 else 0.0
for sname, _ in SETTINGS:
    base = {r["seed"]: r for r in rows if r["setting"] == sname and r["variant"] == "baseline"}
    print(f"== {sname}")
    for vname, _ in VARIANTS:
        g = [r for r in rows if r["setting"] == sname and r["variant"] == vname]
        if not g: continue
        k = [r for r in g if r["seed"] in base]
        dp = [r["pos"]-base[r["seed"]]["pos"] for r in k]; da = [r["A"]-base[r["seed"]]["A"] for r in k]
        fp = [p for r in g for p in r["plans"] if p["i"] == 0 and p["u"] > 0]; lp = [p for r in g for p in r["plans"] if p["i"] >= 1]
        fr = [f for r in g for f in r["first"]]
        print(f"{vname:24s} n={len(g):2d} POS {np.mean([r['pos'] for r in g]):.3f} ({np.mean(dp):+.3f} ± {se(dp):.3f}, {sum(d > 0 for d in dp)}/{len(dp)}) "
              f"A {np.mean([r['A'] for r in g]):.3f} ({np.mean(da):+.3f} ± {se(da):.3f}) close {np.mean([r['below'] for r in g]):.0f} s (first 15 min {np.mean([r['by15'][0] for r in g if r['by15']]):.0f}) "
              f"first-rect conflict {np.mean([p['conflict'] for p in fp]):.2f} ov {np.mean([p['ov'] for p in fp]):.2f} | later conflict {np.mean([p['conflict'] for p in lp]):.2f} | "
              f"2+ swept {np.mean([r['swept']['multi'] for r in g]):.1f} km2 (lanes {np.mean([r['swept']['lanes'] for r in g]):.1f}) | first plan {np.mean([f['tp'] for f in fr])/60:.1f} min, lanes {np.mean([f['lane0'] for f in fr])/60:.1f} min, "
              f"lane time {np.mean([f['lane_min'] for f in fr]):.1f} min, kept {np.mean([f['in1'] for f in fr])/np.mean([f['in0'] for f in fr]):.2f}, in rect at start {np.mean([f['in0'] for f in fr]):.3f}, to median {np.mean([f['c_to_med'] for f in fr]):.0f} m, small first rects {sum(f['area'] < .5 for f in fr)}")
