"""SAROPS-style local search on the accordion rectangle (docs/sarops_approach_todo.md S.10, S.9).

The accordion places a rectangle with a fast approximation; the local search then improves it on the real
objective: the probability found by flying its lanes for one cycle (``accordion.simulate``, all particles,
same plant, follower, geofence and CPA + LRC as the flight, deterministic). Kratzke et al. (2010) use 12 moves
without listing them; here only the ones the accordion does not search (S.10 2026-11-03):
  rotate   theta +- 15 deg (the accordion keeps the main-axis angle)
  corner   the other 3 start corners (the accordion takes the one nearest the UAV; includes the first turn)
  spacing  track spacing +- 25 %, same lanes (the accordion's spacing comes from the exponential detection formula)
  shift    centre +- 200 m along and across the lanes (optional: the accordion's edge moves already place it)
Each round evaluates all neighbours and moves to the best; with no improvement the steps are halved once.

Big jump: with one rectangle per cycle there is no other rectangle to move to the best cell, so the jump is a
restart on another density peak: the local search runs from each peak's accordion rectangle and the best flown
result wins, instead of trusting the approximation's choice of peak.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import time
import numpy as np

from rl_search.accordion import AccordionPlanner, Rect, lane_count, lane_plan, simulate

MOVES = ("rotate", "corner", "spacing")
CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def neighbours(env, r: Rect, moves, scale: float, gamma: float, cycle_s: float, d_theta=np.deg2rad(15.0),
               d_spacing=0.25, d_shift=200.0, search_s=None) -> list[tuple[str, Rect]]:
    """Admissible neighbours of ``r`` (corner set) as (move, rectangle)."""
    out = []
    if "rotate" in moves:
        out += [("rotate", replace(r, theta=r.theta + d*scale*d_theta)) for d in (-1, 1)]
    if "corner" in moves:
        out += [("corner", replace(r, corner=c)) for c in CORNERS if c != r.corner]
    if "spacing" in moves and r.spacing is not None and r.k_max > 1:    # lane_mode "cover": passes k +- 1
        cw = 2*env.lrc_R if getattr(env, "lrc", False) else env.fp_cross
        k = int(round(cw/r.spacing))
        out += [("spacing", replace(r, spacing=cw/kk)) for kk in (k - 1, k + 1) if 1 <= kk <= r.k_max]
    if "spacing" in moves and r.spacing is None:      # fixed spacing (lane_mode "fixed"): no spacing move
        for d in (-1, 1):
            q = replace(r, wid=r.wid*(1 + d*scale*d_spacing))
            if q.wid/lane_count(env, q, cycle_s, search_s=search_s) >= gamma:
                out.append(("spacing", q))
    if "shift" in moves:
        c, s = np.cos(r.theta), np.sin(r.theta)
        for ux, uy in ((c, s), (-s, c)):
            out += [("shift", replace(r, cx=r.cx + d*scale*d_shift*ux, cy=r.cy + d*scale*d_shift*uy)) for d in (-1, 1)]
    return out


def refine(env, r0: Rect, moves=MOVES, gamma: float = 0.0, cycle_s=600.0, time_s: float | None = None,
           task: dict | None = None):
    """Local search from ``r0``. Returns (best rectangle, its gain, log). ``time_s``: stop after this long.
    ``task``: task mode (gain = POS per minute of transit + lanes, see ``accordion.simulate``)."""
    t0 = time.perf_counter()
    ss = task["search_s"] if task else None
    cur = replace(r0, corner=lane_plan(env, r0, cycle_s, search_s=ss)[1])
    gain0 = val = simulate(env, cur, cycle_s, task)
    evals, rounds, used, scale = 1, 0, Counter(), 1.0
    stop = False
    while not stop:
        best, best_v, kind = None, val, None
        for k, q in neighbours(env, cur, moves, scale, gamma, cycle_s, search_s=ss):
            if time_s is not None and time.perf_counter() - t0 > time_s:
                stop = True
                break
            v = simulate(env, q, cycle_s, task)
            evals += 1
            if v > best_v + 1e-9:
                best, best_v, kind = q, v, k
        if best is not None:
            cur, val = best, best_v
            used[kind] += 1
            rounds += 1
        elif scale == 1.0 and not stop:
            scale = 0.5
        else:
            stop = True
    return cur, val, dict(gain0=gain0, gain=val, evals=evals, rounds=rounds, used=dict(used),
                          sec=time.perf_counter() - t0)


class SaropsPlanner(AccordionPlanner):
    """Accordion placement + local search (``big_jump``: from every peak) each cycle; ``search_log`` holds one
    log per cycle: the winner's refine log, totals over the starts, ``picked`` (rank of the winning peak by the
    approximation) and ``gain_first`` (result of the approximation's own choice)."""

    def __init__(self, variant="tavg_pca", cycle_s=600.0, cell=200.0, W=None, moves=MOVES, time_s=None,
                 big_jump=False, **acc_kw):
        super().__init__(variant, cycle_s, cell, W, **acc_kw)
        self.moves, self.time_s, self.big_jump, self.search_log = moves, time_s, big_jump, []

    def choose(self, env, acc) -> Rect | None:
        rects = acc.peak_rects() if self.big_jump else None
        if rects is None:
            r = acc.plan()
            rects = [] if r is None else [r]
        if not rects:
            return None
        res = [refine(env, r, self.moves, acc.gamma, self.cycle_s, self.time_s, self.task) for r in rects]
        picked = int(np.argmax([x[1] for x in res]))
        self.search_log.append(res[picked][2] | dict(evals=sum(x[2]["evals"] for x in res), sec=sum(x[2]["sec"] for x in res),
                                                     starts=len(res), picked=picked, gain_first=res[0][1]))
        return res[picked][0]
