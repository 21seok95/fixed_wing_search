"""Three explanations from the saved flights, no planner rerun:
A why a rectangle is still chosen on ground whose particles were down-weighted (density there before and after the weights)
B why flying the lanes does not see the particles (the same path against frozen and moving particles; what the missed ones did)
C what the start peak looks like at plan + 5 min against the middle of the lanes."""
import json, sys
import numpy as np
from rl_search.mission import ParticleBelief
from rl_search.patterns_v2 import find_clusters
from rl_search.scripts.center_open import open_scenario
from rl_search.scripts.compare_center_open import make_envs
D = [json.loads(l) for l in open(sys.argv[1])]
R = 143.0
out = dict(A=[], B=[], C=[], example=None)

def inr(P, q, m=0.0):
    c, s = np.cos(q[5]), np.sin(q[5]); dx, dy = P[..., 0]-q[1], P[..., 1]-q[2]
    return (np.abs(dx*c+dy*s) <= q[3]/2+m) & (np.abs(-dx*s+dy*c) <= q[4]/2+m)

def across(P, q):
    c, s = np.cos(q[5]), np.sin(q[5])
    return -(P[..., 0]-q[1])*s + (P[..., 1]-q[2])*c

def along(P, q):
    c, s = np.cos(q[5]), np.sin(q[5])
    return (P[..., 0]-q[1])*c + (P[..., 1]-q[2])*s

def seen_geo(pos, xy, tt, t0, t1, sub=4):
    N = pos(t0).shape[0]; m = np.zeros(N, bool)
    for k in range(len(tt)-1):
        if tt[k+1] <= t0 or tt[k] >= t1: continue
        for a in np.arange(sub)/sub:
            t = tt[k]+a*(tt[k+1]-tt[k]); p = xy[k]+a*(xy[k+1]-xy[k]); P = pos(t)
            m |= (P[:, 0]-p[0])**2+(P[:, 1]-p[1])**2 <= R*R
    return m

def kernel(P, w, at, h=250.0):
    d2 = ((P-at)**2).sum(1)
    return float((w*np.exp(-d2/(2*h*h))).sum()/len(w)/(2*np.pi*h*h)*1e6)   # share of total per km2

for d in D:
    if d["mode"] != "free" or d["config"] != "pos": continue
    seed = d["seed"]
    env = make_envs(1, seed)[0]
    scen = open_scenario(env.sampler, seed)
    env.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
    b = env.belief; N = b.n
    saved = next(x for x in map(json.loads, open("rl_search/data/center_open/sweep/results.jsonl")) if x["seed"] == seed and x["n"] == 5 and x["config"] == "pos")
    xy, tt = [np.array(p, float) for p in saved["xy"]], [np.array(t, float) for t in saved["tt"]]
    rects = saved["rects"]
    rows = {(r["u"], r["i"]): r for r in d["rows"]}
    tend = [[(rects[u][i+1][0] if i+1 < len(rects[u]) else tt[u][-1]) for i in range(len(rects[u]))] for u in range(5)]
    # B: each rectangle against its own lanes
    for (u, i), r in rows.items():
        q = rects[u][i]
        t0 = q[0]+q[6]+r["tr"]; t1 = max(min(tend[u][i], t0+720.0), t0+60.0)
        P0 = b.positions(t0); A0 = inr(P0, q) & b.active(t0)
        pos = lambda t: b.positions(t)
        mov = seen_geo(pos, xy[u], tt[u], t0, t1)
        frz = seen_geo(lambda t: P0, xy[u], tt[u], t0, t1)
        ts = np.arange(t0, t1+1, 30.0)
        Ps = np.stack([b.positions(t) for t in ts])
        ins = np.stack([inr(Ps[j], q) for j in range(len(ts))])
        ever = ins.any(0)
        su = np.interp(ts, tt[u], across(xy[u], q)); xu = np.interp(ts, tt[u], along(xy[u], q))
        di = np.sign(su[-1]-su[0]) or 1.0
        yy = across(Ps, q)
        rel = (yy-su[:, None])*di
        miss = ever & ~mov
        band = ((np.abs(rel) <= R) & ins)[:, miss].any(0)
        ahead = ((rel > R) | ~ins)[:, miss].all(0) & ~band
        behind = ((rel < -R) | ~ins)[:, miss].all(0) & ~band
        vy = np.median((yy[-1, A0]-yy[0, A0])/(ts[-1]-ts[0])) if A0.any() else 0.0
        step = np.hypot(*(Ps[1:, A0]-Ps[:-1, A0]).transpose(2, 0, 1))/30.0 if A0.any() else np.zeros((1, 1))
        lanes_w = abs(su[-1]-su[0])
        out["B"].append(dict(seed=seed, u=u, i=i, A0=float(A0.mean()), mov_A0=float(mov[A0].mean()) if A0.any() else None, frz_A0=float(frz[A0].mean()) if A0.any() else None,
                             ever=float(ever.mean()), seen_ever=float(mov[ever].mean()) if ever.any() else None, miss=float(miss.mean()), band=float(band.mean()) if miss.any() else 0.0,
                             ahead=float(ahead.mean()) if miss.any() else 0.0, behind=float(behind.mean()) if miss.any() else 0.0, front=float(lanes_w/(ts[-1]-ts[0])), vy=float(vy*di), speed=float(np.median(step)) if A0.any() else 0.0,
                             dwell=float(np.median(ins[:, ever].sum(0)*30.0)) if ever.any() else 0.0, end_left=float(inr(b.positions(t1), q).mean()/max(A0.mean(), 1e-9)), ell=q[3], wid=q[4], dur=float(t1-t0)))
        if seed == 508 and u == 0 and i == 0:
            k = np.flatnonzero(A0)
            rng = np.random.default_rng(1); k = rng.choice(k, min(90, len(k)), replace=False)
            tfine = np.arange(t0-120, t1+121, 10.0)
            out["example"] = dict(t=tfine.tolist(), su=np.interp(tfine, tt[u], across(xy[u], q)).tolist(), xu=np.interp(tfine, tt[u], along(xy[u], q)).tolist(),
                                  y=[across(b.positions(t)[k], q).tolist() for t in tfine], seen=mov[k].tolist(), wid=q[4], ell=q[3], t0=t0, t1=t1)
    # A and C: conditional weights at each first plan
    for u in range(5):
        q = rects[u][0]; tp = q[0]; r = rows[(u, 0)]
        w = np.ones(N); st = {v: (np.zeros(N), np.full(N, -np.inf)) for v in range(5)}
        others = []
        for v in range(5):
            if v == u or not rects[v] or rects[v][0][0] > tp or (rects[v][0][0] == tp and v > u): continue
            others.append(rects[v][0])
            for k in range(len(tt[v])-1):
                if tt[v][k+1] > tend[v][0]: break
                env._detect_lrc(xy[v][k:k+2], tt[v][k:k+2], w, st[v])
        b.w[:] = 1.0
        S = b.w.copy()
        t_plan5, t_mid = tp+300.0, tp+q[6]+r["tr"]+300.0
        b.w[:] = S
        pk_S = find_clusters(env, t_plan5, k=3, smooth=250.0, sep=1500.0)
        b.w[:] = w
        pk5 = find_clusters(env, t_plan5, k=3, smooth=250.0, sep=1500.0)
        pkm = find_clusters(env, t_mid, k=3, smooth=250.0, sep=1500.0)
        b.w[:] = S
        P5, Pm = b.positions(t_plan5), b.positions(t_mid)
        rec = dict(seed=seed, u=u, tp=tp, removed=1-float(w.mean()), removed_diag=r["removed"], n5=len(pk5), nm=len(pkm), tr=r["tr"])
        if pk5 and pkm:
            rec |= dict(top5_in=bool(any(inr(pk5[0].peak, o) for o in others)), topm_in=bool(any(inr(pkm[0].peak, o) for o in others)),
                        shift=float(np.hypot(*(pk5[0].peak-pkm[0].peak))), mass5=float(pk5[0].mass/w.mean()), massm=float(pkm[0].mass/w.mean()),
                        topS_same=float(np.hypot(*(pk_S[0].peak-pk5[0].peak))), h_S=kernel(P5, S, pk_S[0].peak), h_w_at_S=kernel(P5, w, pk_S[0].peak), h_w=kernel(P5, w, pk5[0].peak),
                        rect_c_to_5=float(np.hypot(q[1]-pk5[0].peak[0], q[2]-pk5[0].peak[1])), rect_c_to_m=float(np.hypot(q[1]-pkm[0].peak[0], q[2]-pkm[0].peak[1])),
                        any_m_clear=bool(any(not any(inr(c.peak, o) for o in others) for c in pkm)))
        out["C"].append(rec)
        if others:
            ts = np.arange(tp+q[6]+r["tr"], tp+q[6]+r["tr"]+601, 60.0)
            over_b = over_a = rest = 0.0
            for t in ts:
                P = b.positions(t); act = b.active(t)
                rr = inr(P, q) & act; oo = np.any([inr(P, o) for o in others], axis=0)
                over_b += float(S[rr & oo].sum()/N); over_a += float(w[rr & oo].sum()/N); rest += float(w[rr & ~oo].sum()/N)
            n = len(ts); area = q[3]*q[4]/1e6; ov = r["ov"]
            if ov > .05:
                out["A"].append(dict(seed=seed, u=u, ov=ov, over_before=over_b/n/(ov*area), over_after=over_a/n/(ov*area), rest=(rest/n/((1-ov)*area)) if ov < .95 else None, region=.85*float(w.mean())/r["hdr85_w"], area=area))
    b.w[:] = 1.0
json.dump(out, open(sys.argv[2], "w"))
def pm(g, k): return float(np.mean([x[k] for x in g if x[k] is not None]))
print("== B: rectangle against its own lanes")
for name, f in (("first", lambda x: x["i"] == 0), ("second", lambda x: x["i"] == 1), ("third+", lambda x: x["i"] >= 2)):
    g = [x for x in out["B"] if f(x)]
    print(name, len(g), f"in rect at lane start {pm(g,'A0'):.3f}; of those seen: frozen {pm(g,'frz_A0'):.2f} moving {pm(g,'mov_A0'):.2f}; ever in rect {pm(g,'ever'):.3f} seen {pm(g,'seen_ever'):.2f}; front {pm(g,'front'):.2f} m/s, target speed {pm(g,'speed'):.2f}, across-lane target {pm(g,'vy'):+.2f} (+ = with the front); dwell {pm(g,'dwell'):.0f} s; lanes {pm(g,'dur'):.0f} s; size {pm(g,'ell'):.0f}x{pm(g,'wid'):.0f}")
    print("   missed share of all particles", round(pm(g,'miss'), 3), "of which ahead", round(np.sum([x['ahead']*x['miss'] for x in g])/max(np.sum([x['miss'] for x in g]), 1e-9), 2), "behind", round(np.sum([x['behind']*x['miss'] for x in g])/max(np.sum([x['miss'] for x in g]), 1e-9), 2), "in the lane band", round(np.sum([x['band']*x['miss'] for x in g])/max(np.sum([x['miss'] for x in g]), 1e-9), 2), "same-direction rects", round(float(np.mean([x['vy'] > 0 for x in g])), 2))
print("== A: overlap density (share of total per km2)")
g = out["A"]
print(len(g), "plans; before", round(pm(g,'over_before'), 3), "after", round(pm(g,'over_after'), 3), "rest of own rectangle", round(pm(g,'rest'), 3), "85% region mean", round(pm(g,'region'), 3), "after/before", round(pm(g,'over_after')/pm(g,'over_before'), 2), "after > rest:", round(float(np.mean([x['over_after'] > x['rest'] for x in g if x['rest'] is not None])), 2), "after > region mean", round(float(np.mean([x['over_after'] > x['region'] for x in g])), 2))
print("== C: peaks")
for name, f in (("UAV 0", lambda x: x["u"] == 0), ("UAV 1-4", lambda x: x["u"] > 0)):
    g = [x for x in out["C"] if f(x) and "shift" in x]
    print(name, len(g), f"peak count plan+5 {pm(g,'n5'):.2f} vs lane mid {pm(g,'nm'):.2f}; top peak moved {pm(g,'shift'):.0f} m; top mass within 2 km {pm(g,'mass5'):.2f} vs {pm(g,'massm'):.2f}; rect centre to peak {pm(g,'rect_c_to_5'):.0f} vs {pm(g,'rect_c_to_m'):.0f} m; removed {pm(g,'removed'):.3f} (diag {pm(g,'removed_diag'):.3f})")
    if name != "UAV 0":
        print("   top inside other rect: plan+5", round(float(np.mean([x['top5_in'] for x in g])), 2), "lane mid", round(float(np.mean([x['topm_in'] for x in g])), 2), "some lane-mid peak clear", round(float(np.mean([x['any_m_clear'] for x in g])), 2), "height at unweighted top peak after/before", round(float(np.mean([x['h_w_at_S']/x['h_S'] for x in g])), 2), "top peak moved by weights", round(pm(g,'topS_same')), "m")
