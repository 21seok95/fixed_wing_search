"""Where the double-swept area comes from and when rectangles conflict. Saved flights only, no planner rerun."""
import json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
BASE = Path("rl_search/data/center_open"); OUT = Path(sys.argv[1])
R, C = 143.0, 25.0
SET = (("no constraint", "sweep"), ("time rule", "sep"), ("time + rectangle", "no_overlap"))
CFG = (("POS only", "pos"), ("A + POS, x 70", "score"))

def in_rect(q, r, m=0.0):
    c, s = np.cos(r[5]), np.sin(r[5]); dx, dy = q[..., 0]-r[1], q[..., 1]-r[2]
    return (np.abs(dx*c+dy*s) <= r[3]/2+m) & (np.abs(-dx*s+dy*c) <= r[4]/2+m)

def conflict(a, b, m=0.0):
    d = np.array([b[1]-a[1], b[2]-a[2]])
    ca, sa, cb, sb = np.cos(a[5]), np.sin(a[5]), np.cos(b[5]), np.sin(b[5])
    ua, va, ub, vb = np.array([ca, sa]), np.array([-sa, ca]), np.array([cb, sb]), np.array([-sb, cb])
    ax, ay, bx, by = a[3]/2, a[4]/2, b[3]/2+m, b[4]/2+m
    for n in (ua, va, ub, vb):
        if abs(d @ n) > ax*abs(ua @ n)+ay*abs(va @ n)+bx*abs(ub @ n)+by*abs(vb @ n):
            return False
    return True

def cur(rects, t):
    i = -1
    for k, r in enumerate(rects):
        if r[0] <= t: i = k
    return i

def span(rects, i):
    return rects[i][0], (rects[i+1][0] if i+1 < len(rects) else np.inf)

def flight(r):
    xy, tt, rects = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]], r["rects"]
    n = len(xy); pts = np.concatenate(xy)
    x0, y0 = pts[:, 0].min()-R-C, pts[:, 1].min()-R-C
    w, h = int((pts[:, 0].max()+R+C-x0)//C)+1, int((pts[:, 1].max()+R+C-y0)//C)+1
    F = np.full((n, 3, h, w), np.inf, np.float32); I = np.full((n, h, w), -1, np.int16)
    m = int(np.ceil(R/C))+1
    for u in range(n):
        for k in range(len(xy[u])-1):
            a, b = xy[u][k], xy[u][k+1]; tm = (tt[u][k]+tt[u][k+1])/2; i = cur(rects[u], tm)
            if i < 0: ph = 0
            else:
                q = rects[u][i]
                ph = 2 if tm >= q[0]+q[6] and in_rect((a+b)/2, q, 150.0) else 1
            ix0, ix1 = max(int((min(a[0], b[0])-x0)//C)-m, 0), min(int((max(a[0], b[0])-x0)//C)+m+1, w)
            iy0, iy1 = max(int((min(a[1], b[1])-y0)//C)-m, 0), min(int((max(a[1], b[1])-y0)//C)+m+1, h)
            X = x0+(np.arange(ix0, ix1)[None, :]+.5)*C; Y = y0+(np.arange(iy0, iy1)[:, None]+.5)*C
            ab = b-a; den = float(ab @ ab)
            s = np.clip(((X-a[0])*ab[0]+(Y-a[1])*ab[1])/den, 0, 1) if den > 0 else np.zeros_like(X*Y)
            hit = (X-a[0]-s*ab[0])**2+(Y-a[1]-s*ab[1])**2 <= R*R
            when = tt[u][k]+s*(tt[u][k+1]-tt[u][k])
            v = F[u, ph, iy0:iy1, ix0:ix1]; better = hit & (when < v); v[better] = when[better]
            if ph == 2: I[u, iy0:iy1, ix0:ix1][better] = i
    cell = C**2/1e6
    seen = np.isfinite(F)
    any_, nl, la = seen.any(1).sum(0), seen[:, 1:].any(1).sum(0), seen[:, 2].sum(0)
    out = dict(team=float((any_ > 0).sum()*cell), total=float(seen.any(1).sum()*cell), multi=float((any_ >= 2).sum()*cell),
               launch=float(((any_ >= 2) & (nl < 2)).sum()*cell), move=float(((nl >= 2) & (la < 2)).sum()*cell), lanes=float((la >= 2).sum()*cell))
    rel = dict(same_overlap=0, same_apart=0, later_overlap=0, later_apart=0, same_first=0)
    if (la >= 2).any():
        T = F[:, 2, la >= 2]; Ix = I[:, la >= 2]
        o = np.argsort(T, axis=0)[:2]; idx = np.arange(T.shape[1])
        key = np.stack([o[0], Ix[o[0], idx], o[1], Ix[o[1], idx]], 1)
        uk, cnt = np.unique(key, axis=0, return_counts=True)
        for (a, ia, b, ib), c in zip(uk.tolist(), cnt.tolist()):
            ra, rb = rects[a][ia], rects[b][ib]
            s0, s1 = span(rects[a], ia), span(rects[b], ib)
            same = s0[0] < s1[1] and s1[0] < s0[1]
            rel[("same_" if same else "later_")+("overlap" if conflict(ra, rb) else "apart")] += c*cell
            if same and conflict(ra, rb) and (ia == 0 or ib == 0): rel["same_first"] += c*cell
    out |= rel
    plans = []
    for u in range(n):
        for i, q in enumerate(rects[u]):
            c300 = c0 = False
            for v in range(n):
                if v == u: continue
                j = cur(rects[v], q[0])
                if j >= 0:
                    c300 |= conflict(q, rects[v][j], 300.0); c0 |= conflict(q, rects[v][j])
            plans.append(dict(u=u, i=i, tp=q[0], c300=bool(c300), c0=bool(c0), area=q[3]*q[4]/1e6, ell=q[3], wid=q[4]))
    out["plans"] = plans
    out["first_plan"] = [rects[u][0][0] if rects[u] else None for u in range(n)]
    return out

rows = []
for setting, folder in SET:
    for line in (BASE/folder/"results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["w"] != 1.0 or r["x"] != 70.0 or r["n"] not in (1, 3, 5): continue
        if r["config"] not in ("pos", "score"): continue
        d = flight(r) if r["n"] > 1 else dict(plans=[dict(u=0, i=i, tp=q[0], c300=False, c0=False, area=q[3]*q[4]/1e6, ell=q[3], wid=q[4]) for i, q in enumerate(r["rects"][0])])
        d |= dict(setting=setting, seed=r["seed"], n=r["n"], config=r["config"], pos=r["pos"], forced=r.get("overlap_forced", 0))
        rows.append(d)
OUT.write_text(json.dumps(rows))
print(len(rows), "flights")
