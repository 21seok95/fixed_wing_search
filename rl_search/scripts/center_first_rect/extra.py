"""Flight time by phase, close time by rectangle round, rectangle size levels. Saved flights only."""
import json, sys
from pathlib import Path
import numpy as np
BASE = Path("rl_search/data/center_open")
def in_rect(p, r, m=0.0):
    c, s = np.cos(r[5]), np.sin(r[5]); dx, dy = p[0]-r[1], p[1]-r[2]
    return abs(dx*c+dy*s) <= r[3]/2+m and abs(-dx*s+dy*c) <= r[4]/2+m
def cur(rects, t):
    i = -1
    for k, r in enumerate(rects):
        if r[0] <= t: i = k
    return i
res = {}
for setting, folder in (("no constraint", "sweep"), ("time rule", "sep"), ("time + rectangle", "no_overlap")):
    for line in (BASE/folder/"results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["w"] != 1.0 or r["x"] != 70.0 or r["n"] not in (1, 3, 5) or r["config"] not in ("pos", "score"): continue
        xy, tt, rects = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]], r["rects"]
        n = r["n"]; T = dict(launch=0.0, move=0.0, lanes=0.0)
        for u in range(n):
            for k in range(len(tt[u])-1):
                dt = tt[u][k+1]-tt[u][k]; tm = (tt[u][k]+tt[u][k+1])/2; i = cur(rects[u], tm)
                if i < 0: T["launch"] += dt
                else:
                    q = rects[u][i]
                    T["lanes" if tm >= q[0]+q[6] and in_rect((xy[u][k]+xy[u][k+1])/2, q, 150.0) else "move"] += dt
        close = dict(both_first=0.0, one_first=0.0, later=0.0, before=0.0)
        for a in range(n):
            for b in range(a+1, n):
                t0, t1 = max(tt[a][0], tt[b][0]), min(tt[a][-1], tt[b][-1])
                if t1 <= t0: continue
                g = np.arange(t0, t1, 5.0)
                pa = np.stack([np.interp(g, tt[a], xy[a][:, 0]), np.interp(g, tt[a], xy[a][:, 1])], 1)
                pb = np.stack([np.interp(g, tt[b], xy[b][:, 0]), np.interp(g, tt[b], xy[b][:, 1])], 1)
                for t in g[np.hypot(*(pa-pb).T) < 300.0]:
                    ia, ib = cur(rects[a], t), cur(rects[b], t)
                    key = "before" if ia < 0 or ib < 0 else "both_first" if ia == 0 and ib == 0 else "one_first" if ia == 0 or ib == 0 else "later"
                    close[key] += 5.0
        areas = [q[3]*q[4]/1e6 for rs in rects for q in rs]
        res.setdefault((setting, n, r["config"]), []).append(dict(T=T, close=close, areas=areas, first=[q[3]*q[4]/1e6 for rs in rects for q in rs[:1]]))
out = {}
for k, g in sorted(res.items(), key=str):
    tot = {p: np.mean([x["T"][p] for x in g]) for p in ("launch", "move", "lanes")}; s = sum(tot.values())
    cl = {p: np.mean([x["close"][p] for x in g]) for p in ("before", "both_first", "one_first", "later")}
    a = np.array([v for x in g for v in x["areas"]])
    print(k, "time share", {p: round(v/s, 3) for p, v in tot.items()}, "close", {p: round(float(v), 1) for p, v in cl.items()})
    out["|".join(map(str, k))] = dict(share={p: v/s for p, v in tot.items()}, close=cl, areas=a.tolist())
json.dump(out, open(sys.argv[1], "w"))
a = np.array(out["time + rectangle|5|score"]["areas"])
print("area levels n=5 rule:", dict(zip(*[x.tolist() for x in np.unique(np.round(a, 1), return_counts=True)])))
a = np.array(out["no constraint|1|score"]["areas"])
print("area levels n=1:", dict(zip(*[x.tolist() for x in np.unique(np.round(a, 1), return_counts=True)])))
