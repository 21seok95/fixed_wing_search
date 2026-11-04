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


def future_factor(env, pol) -> np.ndarray:
    """Survival factor per particle that the UAV's committed lanes still add (until the lanes are done or the task
    cap / end of cycle); env and follower state restored. Ones if nothing is committed."""
    f = pol.follower
    if f is None:
        return np.ones(env.belief.n)
    if pol.search_s is not None:
        cur = pol.tasks[-1] if pol.tasks and pol.tasks[-1]["end"] is None else None
        if cur is None:
            return np.ones(env.belief.n)
        t_end = cur["t_cap"]
    else:
        t_end = pol.t_next
    snap, fs = env.snapshot(), (f.s, f.started, f.laps)
    s0 = env.surv.copy()
    skip, env.skip_cells = env.skip_cells, True
    laps0 = f.laps
    while env.t < t_end - 1e-6 and f.laps == laps0:
        _, _, term, _, _ = env.step([f.action()])
        if term:
            break
    out = np.where(s0 > 0, env.surv/np.maximum(s0, 1e-300), 1.0)
    env.skip_cells = skip
    env.restore(snap)
    f.s, f.started, f.laps = fs
    return out


def run_team(envs, planners, options, actions=None, log_plans=None):
    """Fly N UAVs together. ``planners``: one per env; or None to replay ``actions`` (list per UAV). Returns dict(t,
    pos: shared POS over time (policy part, after the handover), actions). ``log_plans``: list to append
    (t, uav, n_others_committed) per planning event."""
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
    while any(alive):
        todo = [u for u in range(n) if alive[u]]
        planning = [] if replay else [u for u in todo if due(planners[u], envs[u])]
        fut = {}
        if planning:
            fut = {v: future_factor(envs[v], planners[v]) for v in todo if v not in planning}
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
                a = float(planners[u](e))
                e.belief.w[:] = own
                fut[u] = future_factor(e, planners[u])
                if log_plans is not None:
                    log_plans.append((e.t, u, len([v for v in fut if v != u])))
            else:
                a = float(planners[u](e))
            if not replay:
                acts[u].append(a)
            _, _, term, _, _ = e.step([a])
            xy[u].append(e.plant.pose[:2].copy())
            if term:
                alive[u] = False
        S = np.prod([x.surv/s for x, s in zip(envs, s0)], axis=0)
        t_log.append(max(x.t for x in envs))
        pos_log.append(float((1 - S).sum()/envs[0].belief.n))
    return dict(t=np.array(t_log), pos=np.array(pos_log), actions=acts, xy=[np.array(x) for x in xy])
