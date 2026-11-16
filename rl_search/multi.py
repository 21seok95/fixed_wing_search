"""Several UAVs on one shared belief (docs/sarops_approach_todo.md, 2026-11-04 multi-UAV).

One single-UAV ``RLSearchEnv`` per UAV (own plant, energy and CPA pass state). Each env records the survival factor of
its own detections per particle (``surv``). Detections are independent, so the shared belief is the product
(Kratzke et al. 2010, eq. 2): found probability of particle i = 1 - prod_u surv_u[i]. The envs step together (5 s);
the belief is shared only when a UAV plans (an event): its env's weights are set to the shared survival so far times
the survival the other UAVs' committed lanes will still add (simulated on their own envs until their tasks end),
then restored. UAVs due at the same step plan in turn, each given the plans made before it (SAROPS Step 1).

Planners: one per UAV, task mode (``search_s`` set) or fixed cycle; both are supported through ``due``.
"""
from __future__ import annotations

import numpy as np

from rl_search.accordion import HOLD_ACTION
from rl_search.rl_env import RLSearchEnv


class TrackedEnv(RLSearchEnv):
    """RLSearchEnv that also keeps ``surv``: the product of its own detection factors per particle."""

    def reset(self, seed=None, options=None):
        out = super().reset(seed, options)
        self.surv = np.ones(self.belief.n)
        return out

    def _detect_lrc(self, xy, ts, w=None, state=None):
        if w is not None or not hasattr(self, "surv"):
            return super()._detect_lrc(xy, ts, w, state)
        before = self.belief.w.copy()
        gain = super()._detect_lrc(xy, ts)
        m = before > 0
        self.surv[m] *= self.belief.w[m]/before[m]
        return gain

    def _detect_oblique(self, samples, ts):
        if not hasattr(self, "surv"):
            return super()._detect_oblique(samples, ts)
        before = self.belief.w.copy()
        gain = super()._detect_oblique(samples, ts)
        m = before > 0
        self.surv[m] *= self.belief.w[m]/before[m]
        return gain

    def idle(self):
        """Wait one decision interval on the ground (staggered launch): time and target exits advance, the UAV
        does not move, sense or spend energy."""
        self.t += self.cfg.decision_dt_s
        self.belief.update_exits(self.t)

    def snapshot(self):
        s = super().snapshot()
        s["surv"] = self.surv.copy()
        return s

    def restore(self, s):
        super().restore(s)
        self.surv[:] = s["surv"]


def due(pol, env) -> bool:
    """Will ``pol`` plan at this step (same conditions as AccordionPlanner.__call__)?"""
    if pol.search_s is None:
        return env.t >= pol.t_next or pol.follower is None
    cur = pol.tasks[-1] if pol.tasks and pol.tasks[-1]["end"] is None else None
    if cur is not None:
        return pol.follower.laps >= 1 or env.t >= cur["t_cap"]
    return env.t >= pol.t_next


def future_factor(env, pol):
    """Survival factor per particle that the UAV's committed lanes still add (until the lanes are done or the task
    cap / end of cycle), and the track (t, xy) flown meanwhile; env and follower state restored. (ones, None) if
    nothing is committed."""
    f = pol.follower
    if f is None:
        return np.ones(env.belief.n), None
    if pol.search_s is not None:
        cur = pol.tasks[-1] if pol.tasks and pol.tasks[-1]["end"] is None else None
        if cur is None:
            return np.ones(env.belief.n), None
        t_end = cur["t_cap"]
    else:
        t_end = pol.t_next
    snap, fs = env.snapshot(), (f.s, f.started, f.laps)
    s0 = env.surv.copy()
    skip, env.skip_cells = env.skip_cells, True
    laps0 = f.laps
    ts, xs = [env.t], [env.plant.pose[:2].copy()]
    hold_until = getattr(pol, "hold_until", -np.inf)
    while env.t < t_end - 1e-6 and f.laps == laps0:
        a = HOLD_ACTION if env.t < hold_until - 1e-6 else f.action()
        _, _, term, _, _ = env.step([a])
        ts.append(env.t)
        xs.append(env.plant.pose[:2].copy())
        if term:
            break
    out = np.where(s0 > 0, env.surv/np.maximum(s0, 1e-300), 1.0)
    env.skip_cells = skip
    env.restore(snap)
    f.s, f.started, f.laps = fs
    return out, (np.array(ts), np.array(xs))


def approach_track(env, horizon_s=300.0):
    """Predicted track of a UAV with no committed lanes: lead pursuit of the remaining mass (state restored)."""
    from rl_search.patterns import lead_action
    snap = env.snapshot()
    skip, env.skip_cells = env.skip_cells, True
    ts, xs = [env.t], [env.plant.pose[:2].copy()]
    t_end = env.t + horizon_s
    while env.t < t_end - 1e-6:
        _, _, term, _, _ = env.step([lead_action(env)])
        ts.append(env.t)
        xs.append(env.plant.pose[:2].copy())
        if term:
            break
    env.skip_cells = skip
    env.restore(snap)
    return np.array(ts), np.array(xs)


def run_team_launch(envs, planners, options, launch_gap_s=60.0, actions=None, log_plans=None, deconflict=False,
                    hard=False):
    """Fly N UAVs launched ``launch_gap_s`` apart from the common launch point (each env flies its own pre-handover
    transit from its launch time, ``t_launch_s``), stepping each env when it is the furthest behind in time.
    ``deconflict``: the planner gets the traffic: the other UAVs' committed lanes (until their tasks end), their
    predicted approach (lead pursuit, 5 min) when nothing is committed, and their launch transits. ``hard``:
    separation as a constraint (holds tried, ``pol.violations`` counts plans with no feasible candidate)."""
    n = len(envs)
    for u, e in enumerate(envs):
        e.reset(options=dict(options, t_launch_s=float(options.get("t_launch_s", 0.0)) + u*launch_gap_s))   # base: a late launch
    if planners is not None:
        for p in planners:
            p.hard_sep = hard
    s0 = [e.surv.copy() for e in envs]
    replay = actions is not None
    acts = actions if replay else [[] for _ in range(n)]
    it = [iter(a) for a in acts] if replay else None
    alive = [True]*n
    xy = [[np.array(p[1:]) for p in e.launch_track] for e in envs]
    tt = [[p[0] for p in e.launch_track] for e in envs]
    launch = [np.array([p[0] for p in e.launch_track]) for e in envs]
    launch_xy = [np.array([p[1:] for p in e.launch_track]) for e in envs]
    t_log, pos_log = [], []
    while any(alive):
        T = min(envs[u].t for u in range(n) if alive[u])
        todo = [u for u in range(n) if alive[u] and envs[u].t <= T + 1e-6]
        planning = [] if replay else [u for u in todo if due(planners[u], envs[u])]
        fut, trk = {}, {}
        if planning:
            for v in range(n):
                if not alive[v] or v in planning:
                    continue
                if envs[v].t <= T + 1e-6:
                    fut[v], trk[v] = future_factor(envs[v], planners[v])
                    if trk[v] is None and deconflict:
                        trk[v] = approach_track(envs[v])
        for u in todo:
            e = envs[u]
            if replay:
                a = next(it[u], None)
                if a is None:
                    alive[u] = False
                    continue
            elif u in planning:
                S = np.prod([x.surv for x in envs], axis=0)
                others = np.prod([f for v, f in fut.items() if v != u], axis=0) if fut else 1.0
                own = e.belief.w.copy()
                e.belief.w[:] = np.where(own > 0, S*others, 0.0)
                if deconflict:
                    traffic = [x for v, x in trk.items() if v != u and x is not None]
                    traffic += [(launch[v], launch_xy[v]) for v in range(n) if v != u and launch[v][-1] > e.t]
                    planners[u].traffic = traffic
                a = float(planners[u](e))
                planners[u].traffic = None
                e.belief.w[:] = own
                fut[u], trk[u] = future_factor(e, planners[u])
                if log_plans is not None:
                    log_plans.append((e.t, u))
            else:
                a = float(planners[u](e))
            if not replay:
                acts[u].append(a)
            _, _, term, _, _ = e.step([a])
            xy[u].append(e.plant.pose[:2].copy())
            tt[u].append(e.t)
            if term:
                alive[u] = False
        S = np.prod([x.surv/s for x, s in zip(envs, s0)], axis=0)
        t_log.append(T + 5.0)
        pos_log.append(float((1 - S).sum()/envs[0].belief.n))
    return dict(t=np.array(t_log), pos=np.array(pos_log), actions=acts, xy=[np.array(x) for x in xy],
                tt=[np.array(x) for x in tt])


def separation_pairs(res, d_safe=300.0, early_s=900.0):
    """Like ``separation`` but each pair over the time both are airborne (from the later launch, launch transits
    included). Returns (min distance, seconds below d_safe, encounters, seconds below d_safe within early_s of
    the later launch)."""
    xy, tt = res["xy"], res["tt"]
    n = len(xy)
    if n < 2:
        return np.inf, 0.0, 0, 0.0
    dmin, below, enc, early = np.inf, 0.0, 0, 0.0
    for a in range(n):
        for b in range(a + 1, n):
            t0, t1 = max(tt[a][0], tt[b][0]), min(tt[a][-1], tt[b][-1])
            if t1 <= t0:
                continue
            g = np.arange(t0, t1, 5.0)
            pa = np.stack([np.interp(g, tt[a], xy[a][:, 0]), np.interp(g, tt[a], xy[a][:, 1])], 1)
            pb = np.stack([np.interp(g, tt[b], xy[b][:, 0]), np.interp(g, tt[b], xy[b][:, 1])], 1)
            d = np.hypot(*(pa - pb).T)
            dmin = min(dmin, d.min())
            close = d < d_safe
            below += close.sum()*5.0
            early += close[g < t0 + early_s].sum()*5.0
            enc += int(np.sum(np.diff(close.astype(int)) == 1) + close[0])
    return dmin, below, enc, early


def run_team(envs, planners, options, actions=None, log_plans=None, stagger_s=0.0, deconflict=False):
    """Fly N UAVs together. ``planners``: one per env; or None to replay ``actions`` (list per UAV). Returns dict(t,
    pos: shared POS over time (policy part, after the handover), actions, xy per UAV, times per UAV).
    ``log_plans``: list to append (t, uav, n_others_committed) per planning event. ``stagger_s``: UAV u starts its
    mission u * stagger_s later (approximation: it waits after the common launch transit, rounded to 5 s steps).
    ``deconflict``: planning-time separation, the planner gets the other UAVs' committed tracks (``pol.traffic``)."""
    n = len(envs)
    for e in envs:
        e.reset(options=options)
    s0 = [e.surv.copy() for e in envs]       # detections during the pre-handover transit are not counted
    replay = actions is not None
    acts = actions if replay else [[] for _ in range(n)]
    it = [iter(a) for a in acts] if replay else None
    alive = [True]*n
    t_log, pos_log = [envs[0].t], [0.0]
    xy = [[e.plant.pose[:2].copy()] for e in envs]
    tt = [[e.t] for e in envs]
    wait = [int(round(u*stagger_s/envs[0].cfg.decision_dt_s)) for u in range(n)]
    while any(alive):
        idling = []
        for u in range(n):
            if wait[u] > 0 and alive[u]:
                envs[u].idle()
                wait[u] -= 1
                idling.append(u)
        todo = [u for u in range(n) if alive[u] and u not in idling]
        planning = [] if replay else [u for u in todo if due(planners[u], envs[u])]
        fut, trk = {}, {}
        if planning:
            for v in todo:
                if v not in planning:
                    fut[v], trk[v] = future_factor(envs[v], planners[v])
        for u in todo:
            e = envs[u]
            if replay:
                a = next(it[u], None)
                if a is None:
                    alive[u] = False
                    continue
            elif u in planning:
                S = np.prod([x.surv for x in envs], axis=0)
                others = np.prod([f for v, f in fut.items() if v != u], axis=0) if fut else 1.0
                own = e.belief.w.copy()
                e.belief.w[:] = np.where(own > 0, S*others, 0.0)
                if deconflict:
                    planners[u].traffic = [x for v, x in trk.items() if v != u and x is not None]
                a = float(planners[u](e))
                planners[u].traffic = None
                e.belief.w[:] = own
                fut[u], trk[u] = future_factor(e, planners[u])
                if log_plans is not None:
                    log_plans.append((e.t, u, len([v for v in fut if v != u])))
            else:
                a = float(planners[u](e))
            if not replay:
                acts[u].append(a)
            _, _, term, _, _ = e.step([a])
            xy[u].append(e.plant.pose[:2].copy())
            tt[u].append(e.t)
            if term:
                alive[u] = False
        S = np.prod([x.surv/s for x, s in zip(envs, s0)], axis=0)
        t_log.append(max(x.t for x in envs))
        pos_log.append(float((1 - S).sum()/envs[0].belief.n))
    return dict(t=np.array(t_log), pos=np.array(pos_log), actions=acts, xy=[np.array(x) for x in xy],
                tt=[np.array(x) for x in tt])


def separation(res, d_safe=300.0, early_s=900.0):
    """Closest horizontal distance between any two UAVs at the same time, and the time any pair spends closer than
    ``d_safe`` (common 5 s grid, from the last UAV's launch: before that the waiting UAVs sit at the handover point).
    Returns (min distance m, seconds below d_safe, number of close encounters, seconds below d_safe in the first
    ``early_s`` after the last launch)."""
    xy, tt = res["xy"], res["tt"]
    n = len(xy)
    if n < 2:
        return np.inf, 0.0, 0, 0.0
    t0, t1 = max(t[1] - 5.0 if len(t) > 1 else t[0] for t in tt), min(t[-1] for t in tt)   # from the last launch
    grid = np.arange(t0, t1, 5.0)
    P = [np.stack([np.interp(grid, tt[u], xy[u][:, 0]), np.interp(grid, tt[u], xy[u][:, 1])], 1) for u in range(n)]
    dmin, below, enc, early = np.inf, 0.0, 0, 0.0
    for a in range(n):
        for b in range(a + 1, n):
            d = np.hypot(*(P[a] - P[b]).T)
            dmin = min(dmin, d.min())
            close = d < d_safe
            below += close.sum()*5.0
            early += close[grid < t0 + early_s].sum()*5.0
            enc += int(np.sum(np.diff(close.astype(int)) == 1) + close[0])
    return dmin, below, enc, early
