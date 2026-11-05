"""Why rectangles overlap although the earlier UAV's lanes are committed. Planner code is untouched: subclasses only log
and try other beliefs at each plan, and the flights must match the saved ones exactly.

free: no constraint. At each plan: what the commitment removed, what is left on the overlapped ground, per-start rates,
      and the rectangle chosen under other beliefs (seen particles removed, everything through the rectangle removed, no commitment).
rule: time + rectangle rule. Why each "no clear rectangle" event happened.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

SAVED = {"free": "sweep", "rule": "no_overlap"}


def hdr(pts, wt, fracs=(.5, .85), cell=100.0, smooth=250.0):
    keep = wt > 0
    pts, wt = pts[keep], wt[keep]
    lo, hi = pts.min(0) - 1500, pts.max(0) + 1500
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[np.arange(lo[0], hi[0] + cell, cell), np.arange(lo[1], hi[1] + cell, cell)], weights=wt)
    v = np.sort(gaussian_filter(H, smooth/cell).ravel())[::-1]
    c = np.cumsum(v)/v.sum()
    return [float((np.searchsorted(c, f) + 1)*cell**2/1e6) for f in fracs]


def run(job):
    seed, mode, config = job
    warnings.filterwarnings("ignore")
    import rl_search.sarops as S_mod
    from rl_search import multi
    from rl_search.accordion import Accordion, Rect, simulate, start_transit, sweep_width
    from rl_search.area_score import AreaSaropsPlanner, AreaScore, NoOverlapAreaSarops, NoOverlapSarops, inside_rect, rects_conflict
    from rl_search.sarops import SaropsPlanner
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.compare_center_open import make_envs
    probe = make_envs(1, seed)[0]
    scen = open_scenario(probe.sampler, seed)
    W = sweep_width(probe)
    n = 5
    sep = mode == "rule"
    kw = dict(W=W, n_starts=5 if sep else 3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover")
    calls, rows = [], []
    orig = S_mod.refine

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append((out[0], float(out[1])))
        return out
    S_mod.refine = spy

    def share(r, others, nx=40, ny=20):
        if not others:
            return 0.0
        u = (np.arange(nx) + .5)/nx*r.ell - r.ell/2
        v = (np.arange(ny) + .5)/ny*r.wid - r.wid/2
        U, V = np.meshgrid(u, v)
        c, s = np.cos(r.theta), np.sin(r.theta)
        q = np.stack([r.cx + U*c - V*s, r.cy + U*s + V*c], -1)
        return float(np.any([inside_rect(q, o) for o in others], axis=0).mean())

    class Free(AreaSaropsPlanner if config == "score" else SaropsPlanner):
        envs = None; pending = None

        def current(self):
            return [p for p in self.team if p is not self and p.rects and p.tasks and p.tasks[-1]["end"] is None]

        def make_acc(self, env):
            acc = Accordion(env, self.variant, self.cycle_s, self.cell, self.W, search_s=self.search_s, max_transit_s=self.max_transit_s, **self.acc_kw)
            if self.hook is not None:
                self.hook(env, acc)
            return acc

        def choose(self, env, acc):
            calls.clear()
            r0 = super().choose(env, acc)
            starts = list(calls)
            if r0 is None:
                return r0
            i = len(self.rects)
            cur = self.current()
            orects = [p.rects[-1][1] for p in cur]
            b, N = env.belief, env.belief.n
            w = b.w.copy()
            Sv = np.prod([e.surv for e in self.envs], axis=0)
            others = np.minimum(np.where(Sv > 0, w/np.maximum(Sv, 1e-300), 1.0), 1.0)
            seen = others < 1 - 1e-9
            tr = float(start_transit(env, r0, self.cycle_s, self.search_s))
            lane0, lane1 = env.t + r0.hold_s + tr, env.t + r0.hold_s + tr + self.search_s
            ever = np.zeros(N, bool)
            windows = []
            for p in cur:
                k = p.tasks[-1]
                ls = k["t0"] + k["hold"] + k["transit"]
                le = ls + self.search_s
                windows.append((ls, le))
                for tau in np.arange(max(env.t, ls), le + 1, 60.0):
                    ever |= inside_rect(b.positions(tau), p.rects[-1][1]) & b.active(tau)
            ts = np.arange(lane0, lane1 + 1, 60.0)
            mass_r = zs = zu_in = zu_out = zone_S = 0.0
            for tau in ts:
                P, act = b.positions(tau), b.active(tau)
                in_r = inside_rect(P, r0) & act
                in_o = np.any([inside_rect(P, o) for o in orects], axis=0) if orects else np.zeros(N, bool)
                z = in_r & in_o
                mass_r += w[in_r].sum()
                zs += w[z & seen].sum(); zu_in += w[z & ~seen & ever].sum(); zu_out += w[z & ~seen & ~ever].sum(); zone_S += Sv[z].sum()
            d = len(ts)*N
            tm = env.t + tr + 300.0
            Pm = b.positions(tm)
            act = b.active(tm)
            hS, hw = hdr(Pm[act], Sv[act]), hdr(Pm[act], w[act])
            ov = share(r0, orects)
            time_overlap = 0.0
            for o, (ls, le) in zip(orects, windows):
                if rects_conflict(r0, o):
                    time_overlap = max(time_overlap, max(0.0, min(le, lane1) - max(ls, lane0))/self.search_s)
            row = dict(seed=seed, config=config, u=self.uav, i=i, t=float(env.t), tr=tr, n_others=len(cur), ov=ov, conflict=any(rects_conflict(r0, o) for o in orects),
                       conflict300=any(rects_conflict(r0, o, 300.0) for o in orects), area=r0.ell*r0.wid/1e6, removed=1 - float(w.sum()/max(Sv.sum(), 1e-300)), total_S=float(Sv.sum()/N),
                       mass_r=mass_r/d, zone_seen=zs/d, zone_unseen_in=zu_in/d, zone_unseen_out=zu_out/d, zone_S=zone_S/d, hdr50_S=hS[0], hdr85_S=hS[1], hdr50_w=hw[0], hdr85_w=hw[1],
                       ever_mass=float(Sv[ever].sum()/N), ever_seen_mass=float(Sv[ever & seen].sum()/N), seen_mass=float(Sv[seen].sum()/N),
                       time_overlap=time_overlap, rate=float(simulate(env, r0, self.cycle_s, self.task)),
                       starts=[dict(rate=v, ov=share(q, orects), conflict=any(rects_conflict(q, o) for o in orects), peak_d=float(np.hypot(q.cx - env.plant.pose[0], q.cy - env.plant.pose[1]))) for q, v in starts])
            pk = acc.peaks() or []
            row["peaks"] = [dict(mass=float(c.mass), inside=bool(any(inside_rect(c.peak, o) for o in orects)), inside300=bool(any(inside_rect(c.peak, o, 300.0) for o in orects)), d=float(np.hypot(*(c.peak - env.plant.pose[:2])))) for c in pk]
            if orects and (config == "pos" or i == 0):
                cf = {}
                for name, wv in (("seen_removed", Sv*(~seen)), ("zone_removed", Sv*(~seen & ~ever)), ("no_commit", Sv)):
                    b.w[:] = np.where(w > 0, wv, 0.0)
                    q = super().choose(env, self.make_acc(env))
                    b.w[:] = w
                    cf[name] = None if q is None else dict(ov=share(q, orects), conflict=any(rects_conflict(q, o) for o in orects), area=q.ell*q.wid/1e6, rate=float(simulate(env, q, self.cycle_s, self.task)))
                row["cf"] = cf
            self.pending = (row, seen, w)
            return r0

        def _plan(self, env):
            ok = super()._plan(env)
            if ok and self.pending is not None:
                row, seen, w = self.pending
                f, _ = multi.future_factor(env, self)
                g = w*(1 - f)
                row["gain"] = float(g.sum()/env.belief.n)
                row["gain_seen_share"] = float(g[seen].sum()/max(g.sum(), 1e-300))
                rows.append(row)
            self.pending = None
            return ok

    class Rule(NoOverlapAreaSarops if config == "score" else NoOverlapSarops):
        def choose(self, env, acc):
            f0 = self.overlap_forced
            base_approx, base_peaks = acc.approx, acc.peaks
            r = super().choose(env, acc)
            i = len(self.rects)
            others = self._others()
            ov = bool(r is not None and any(rects_conflict(r, o) for o in others))
            rows.append(dict(kind="plan", seed=seed, config=config, u=self.uav, i=i, t=float(env.t), n_others=len(others), forced=self.overlap_forced > f0, none=r is None, conflict=ov,
                             conflict300=bool(r is not None and any(rects_conflict(r, o, self.d_safe) for o in others)), ov=share(r, others) if r is not None else 0.0))
            if self.overlap_forced > f0:
                m = self.d_safe
                b = env.belief
                raw = base_peaks() or []
                pts = b.positions(env.t + acc.T/2)
                keep = b.w.copy()
                total = keep.sum()
                for o in others:
                    b.w[inside_rect(pts, o, m)] = 0.0
                left = float(b.w.sum()/max(total, 1e-300))
                masked = base_peaks() or []
                b.w[:] = keep
                clear_pk = [c for c in masked if not any(inside_rect(c.peak, o, m) for o in others)]

                def approx(G, i0, i1, j0, j1):
                    mid = np.array([(G["ex"][i0] + G["ex"][i1])/2, (G["ey"][j0] + G["ey"][j1])/2]) @ G["R"].T
                    q = Rect(float(mid[0]), float(mid[1]), float((i1 - i0)*acc.g), float((j1 - j0)*acc.gy), float(G["theta"]), 0.0)
                    return base_approx(G, i0, i1, j0, j1) if not any(rects_conflict(q, o, m) for o in others) else -1.0

                def climb(c, f):
                    G = acc._grid(c.theta)
                    q = c.peak @ G["R"]
                    ci, cj = int(np.clip((q[0] - G["ex"][0])//acc.g, 0, len(G["ex"]) - 2)), int(np.clip((q[1] - G["ey"][0])//acc.gy, 0, len(G["ey"]) - 2))
                    acc.approx = f
                    try:
                        return acc._climb(G, ci, cj)
                    finally:
                        acc.approx = base_approx
                pose = env.plant.pose[:2]
                rows.append(dict(kind="forced", seed=seed, config=config, u=self.uav, i=i, t=float(env.t), none=r is None, conflict=ov, n_others=len(others), left=left,
                                 n_raw=len(raw), n_masked=len(masked), n_clear=len(clear_pk),
                                 raw_d=[float(np.hypot(*(c.peak - pose))) for c in raw], clear_d=[float(np.hypot(*(c.peak - pose))) for c in clear_pk],
                                 raw_plain=[climb(c, base_approx) is not None for c in raw], clear_plain=[climb(c, base_approx) is not None for c in clear_pk],
                                 clear_clear=[climb(c, approx) is not None for c in clear_pk]))
            return r

    if config == "pos":
        envs = make_envs(n, seed)
    else:
        envs = make_envs(n, seed, area_env=True)
        area = AreaScore(probe.terr, probe.h, scen.track, scen.save_dt_s, 70.0, probe.lrc_R)
        for e in envs:
            e.area, e.pos_w, e.area_w = area, 1.0, 1.0
    pols = [(Rule if sep else Free)(**kw) for _ in range(n)]
    for u, p in enumerate(pols):
        p.team, p.envs, p.uav = pols, envs, u
    multi.run_team_launch(envs, pols, dict(scenario=scen, stage=3), 60.0, deconflict=sep, hard=sep)
    saved = None
    for line in Path(f"rl_search/data/center_open/{SAVED[mode]}/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["seed"] == seed and r["n"] == n and r["config"] == config and r["x"] == 70.0 and r["w"] == 1.0:
            saved = r
    same = saved is not None and all(len(p.rects) == len(s) and all(abs(q[1].cx - z[1]) < 1e-6 and abs(q[1].cy - z[2]) < 1e-6 for q, z in zip(p.rects, s)) for p, s in zip(pols, saved["rects"]))
    print(f"seed {seed} {mode} {config}: rows {len(rows)}, matches saved flight: {same}", flush=True)
    return dict(seed=seed, mode=mode, config=config, same=same, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=9); ap.add_argument("--seeds", type=int, nargs="*", default=list(range(500, 510)))
    ap.add_argument("--modes", nargs="*", default=["free:pos", "free:score", "rule:pos", "rule:score"])
    a = ap.parse_args()
    jobs = [(s, *m.split(":")) for m in a.modes for s in a.seeds]
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        for out in pool.imap_unordered(run, jobs, chunksize=1):
            f.write(json.dumps(out) + "\n"); f.flush()
    print("done")
