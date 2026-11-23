"""Several UAVs on one evader (docs/multi_agent_design.md, agreed with the user item by item). Two designs (``design``):

- "m1" (section "구현 준비 (2026-11-19)", the M1 run): team reward, penalty from 200 m / never within 150 m, launch gaps 5 / 10 / 15 s,
  a teammate tail channel on every map and a vector of ``EXTRA_NAMES``
- "m2" (sections "M1 점검과 다음 학습(M2) 설계" and "다음 학습 보상·관측 최종안", user 2026-11-20 ~ 11-23): reward (team found + 0.25 x what the
  UAV found) / the probability left before the step, penalty from 250 m (the distance term only while the pair closes) / never within
  100 m, launch gaps 5-10 s drawn per UAV, teammate channels of ``rl_search.mate_maps`` (small maps ③ claim + danger zone, large maps
  plan B) and one vector entry 1 / team size (pending UAVs included)

One single-UAV env (``rl_real.RealRLEnv``: its flight, camera, detections, observation) per UAV, all on the same scenario:

- belief: every UAV keeps its own belief and every detection (whoever made it) lowers every belief (same observations). Stage 1: all
  beliefs are the same weights ``W``. After each 5 s step the factor by which each UAV's detections lowered each particle
  (s_u = w_after / w_before) is multiplied: W_new = W_old x prod_u s_u (detections are independent).
  Team found = (sum W_old - sum W_new) / N; own share of UAV i = what the team would not have found this step without it,
  sum W_old x prod_{u != i} s_u x (1 - s_i) / N; seen by UAV i = sum W_old (1 - s_i) / N.
- reward of UAV i per step: reward_scale x (pos_w x F_i + look_w x its own look reward A') - mask_cost x (its filter cut + edge cut)
  - its safety penalty - P_HARD x its violations, with F_i = team found (m1, "team"), its own share ("share"), or (team found + own_w x
  seen by i) / the belief left before the step (m2, "team_rel": the same find counts more late in a flight; with gamma 0.999, user 11-23).
- launches: one base (the scenario's launch point), one after another, each UAV flies its policy from its own launch (user 2026-11-19).
  m2: a launch between two 5 s decisions joins at the next one, placed where it would be after flying straight on along the launch
  heading for the seconds in between (spacing = gap x speed). A UAV not yet joined, or whose flight has ended, is inactive: its steps
  are not training data (``active`` per UAV, the trainer masks them; rl_multi_vec ends its slot's episode at its own flight end).
- safety: penalty from ``d_soft``: predicted closest approach within 20 s if both fly straight below d_soft (c x (1 - t/20) x
  (1 - d/d_soft)) and the distance now below d_soft (c x (1 - d/d_soft); m2: only while the two close in); never within ``d_hard``:
  before the step a filter flies every UAV's command for the 5 s step and 15 s more (the plant model, the command held) and moves a
  command that brings a pair within d_hard to the nearest one that does not (no such command: the one with the largest separation); the
  cut is charged like the edge mask (``mask_cost``). Within d_hard all the same: penalty ``P_HARD`` and a violation count (0 expected).

- sequential decisions (``sequential=True``, user 2026-11-23 "순차 결정"): within each 5 s step the active UAVs decide one after another in a
  random order; each flies its 5 s at once, its detections lower the shared belief, and the next UAV's observation shows that (the moved
  UAVs where they are after the step, the belief after their detections) plus one more vector entry, its turn (0 first .. 1 last).
  The filter of a sequential step keeps each UAV d_hard from the UAVs that already moved (their flown step and their command held 15 s
  more) and from those still to move (flying straight on); the reward, penalty and episode records are the same as ``step``.
  ``begin_step`` -> (``turn_observation(m)``, ``turn_act(m, a)``) for m in the order -> ``end_step``; ``seq_step(act_fn)`` runs one.

    env = MultiRealEnv(3, cfg, seed=0, design="m2"); obs = env.reset(options=case.options(scen)); obs, rewards, done, info = env.step(actions)
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from rl_search.plants import G

GAPS_S = (5.0, 10.0, 15.0)        # m1 launch gaps (user 2026-11-19: 5-15 s; on the 5 s decision clock)
D_SOFT = 200.0                    # m1 penalty distance (m)
D_HARD = 150.0                    # m1 filter distance (m)
LOOK_S = 20.0                     # closest-approach look-ahead (s) of the penalty and of the filter (5 s step + 15 s held)
C_CPA, C_DIST, P_HARD = 1.0, 1.0, 10.0   # penalty sizes
N_CAND = 21                       # commands tried by the filter in [-1, 1]
EXTRA_NAMES = ("mates_airborne", "mates_pending", "next_launch", "launch_rank", "nearest_m", "cpa_m", "cpa_t")   # m1 vector
SMALL_SIDE = 6400.0               # m2: maps up to this side get ③ + danger, larger maps plan B
# m3 (user 2026-11-24, docs/multi_agent_design.md "2대 단계: 이동 중과 탐색 중 충돌 규칙"): a UAV is in transit until the first step at which
# at least SEARCH_FRAC of the belief left lies inside its 6.4 km map (it sees the particles there), then it searches (for good). A pair keeps
# m2's rules (never within 100 m, penalty from 250 m while closing, 20 s look-ahead) while both are in transit, and the search rules once
# either searches: never within 500 m (head-on and following alike), 35 s look-ahead; a pair that is closer than 500 m when the rules change
# may not get closer than it has been (the limit rises with the separation up to 500 m). Each UAV's penalty follows its own mode: searching,
# C_DIST x g(distance) while the pair is not moving apart faster than sep_mps, plus C_CPA x (1 - t / look) x g(closest approach) with g =
# ``danger_level`` (the same g draws its danger channel); in transit m2's penalty.
SEARCH_FRAC, SEARCH_SIDE = 0.01, 6400.0
STEP_IN, STEP_EDGE = 0.7, 0.3     # m3 "step" level just outside d_hard and at d_soft
DESIGNS = {
    "m1": dict(d_soft=D_SOFT, d_hard=D_HARD, gaps=GAPS_S, gap_range=None, closing_only=False, reward="team", mates="tail"),
    "m2": dict(d_soft=250.0, d_hard=100.0, gaps=None, gap_range=(5.0, 10.0), closing_only=True, reward="team_rel", own_w=0.25,
               mates="zones", vec_k=True),
    "m3": dict(d_soft=250.0, d_hard=100.0, gaps=None, gap_range=(5.0, 10.0), closing_only=True, reward="team_rel", own_w=0.25,
               mates="zones", vec_k=True, search=dict(d_hard=500.0, d_soft=1000.0, look=35.0, sep_mps=3.0, shape="step")),
}


def danger_level(d, shape: str = "step", d_hard: float = 500.0, d_soft: float = 1000.0) -> np.ndarray:
    """m3 search rules: the penalty level per step (x C_DIST) and the danger channel's value at distance d (m) from a teammate. "step": 1
    within d_hard (never allowed), STEP_IN just outside it falling linearly to STEP_EDGE at d_soft, 0 beyond (both lines show); "linear":
    1 - d / d_soft."""
    d = np.asarray(d, float)
    if shape == "linear":
        return np.clip(1.0 - d/d_soft, 0.0, 1.0)
    out = np.where(d <= d_hard, 1.0, STEP_IN + (STEP_EDGE - STEP_IN)*(d - d_hard)/(d_soft - d_hard))
    return np.where(d > d_soft, 0.0, out)
REM_FLOOR = 0.02                  # "team_rel": the belief left is taken as at least this (a flight never gets near it; 11-23 check 0.45-0.66 at 45 min)
FOLLOW_M, FOLLOW_DEG = 1500.0, 20.0     # "following" (logged, user 11-23): a teammate within 1.5 km ahead, within 20 deg of my heading, heading within 20 deg


def predict_paths(plant, chi_cmd, seconds: float, every_s: float = 1.0) -> np.ndarray:
    """Positions (M, S, 2), every ``every_s`` over ``seconds``, of the plant holding each course command in ``chi_cmd`` (M,) from its
    present state (the plant is not changed): the same equations as ``plants.FixedWing._integrate``, vectorised over the commands."""
    chi = np.atleast_1d(np.asarray(chi_cmd, float))
    m = len(chi)
    h, v = plant.h, plant.v
    n, k = int(round(seconds/h)), max(1, int(round(every_s/h)))
    x, y, yaw = (np.full(m, float(c)) for c in plant.pose)
    phi, r = np.full(m, float(plant.phi)), np.full(m, float(plant.r))
    w2, c2 = plant.omega**2, 2*plant.zeta*plant.omega
    out = []
    for i in range(n):
        err = (chi - yaw + np.pi) % (2*np.pi) - np.pi
        r = r + (w2*err - c2*r)*h
        phi_cmd = np.clip(np.arctan(r*v/G), -plant.phi_max, plant.phi_max)
        phi = phi + np.clip((phi_cmd - phi)/plant.tau, -plant.p_max, plant.p_max)*h
        yaw = yaw + G*np.tan(phi)/v*h
        x, y = x + v*np.cos(yaw)*h, y + v*np.sin(yaw)*h
        if (i + 1) % k == 0:
            out.append(np.stack([x, y], 1))
    return np.stack(out, 1)


def predict_paths_multi(plants, chis, seconds: float, every_s: float = 1.0) -> list:
    """``predict_paths`` for several plants at once (one integration loop; 11-24: a call costs about 4 ms whatever the number of commands,
    so the filter and the allowed range of a world's UAVs share one): ``chis`` one list of course commands per plant; returns one (M_u, S, 2)
    array per plant, element for element the same numbers as ``predict_paths`` (the plants share their flight model)."""
    p0 = plants[0]
    for p in plants[1:]:
        if (p.h, p.omega, p.zeta, p.phi_max, p.tau, p.p_max) != (p0.h, p0.omega, p0.zeta, p0.phi_max, p0.tau, p0.p_max):
            raise ValueError("predict_paths_multi: the plants must share their flight model")
    sizes = [len(np.atleast_1d(c)) for c in chis]
    chi = np.concatenate([np.atleast_1d(np.asarray(c, float)) for c in chis])
    rep = lambda f: np.concatenate([np.full(m, float(f(p))) for p, m in zip(plants, sizes)])
    x, y, yaw = rep(lambda p: p.pose[0]), rep(lambda p: p.pose[1]), rep(lambda p: p.pose[2])
    phi, r, v = rep(lambda p: p.phi), rep(lambda p: p.r), rep(lambda p: p.v)
    h = p0.h
    n, k = int(round(seconds/h)), max(1, int(round(every_s/h)))
    w2, c2 = p0.omega**2, 2*p0.zeta*p0.omega
    out = []
    for i in range(n):
        err = (chi - yaw + np.pi) % (2*np.pi) - np.pi
        r = r + (w2*err - c2*r)*h
        phi_cmd = np.clip(np.arctan(r*v/G), -p0.phi_max, p0.phi_max)
        phi = phi + np.clip((phi_cmd - phi)/p0.tau, -p0.p_max, p0.p_max)*h
        yaw = yaw + G*np.tan(phi)/v*h
        x, y = x + v*np.cos(yaw)*h, y + v*np.sin(yaw)*h
        if (i + 1) % k == 0:
            out.append(np.stack([x, y], 1))
    allp = np.stack(out, 1)
    return np.split(allp, np.cumsum(sizes)[:-1])


def cpa(p_i, h_i, p_j, h_j, v: float) -> tuple:
    """Time (s) and distance (m) of the closest approach if both fly straight on at speed v from positions p and headings h (rad);
    (inf, distance now) when they do not come closer."""
    r = np.asarray(p_j, float) - np.asarray(p_i, float)
    w = v*np.array([np.cos(h_j) - np.cos(h_i), np.sin(h_j) - np.sin(h_i)])
    ww = float(w @ w)
    if ww < 1e-9:
        return np.inf, float(np.hypot(*r))
    t = -float(r @ w)/ww
    if t <= 0:
        return np.inf, float(np.hypot(*r))
    return t, float(np.hypot(*(r + w*t)))


class MultiRealEnv:
    """k UAVs (``RealRLEnv`` each, same settings) on one scenario with a shared belief, staggered launches and the safety filter."""

    def __init__(self, k: int, cfg, seed: int = 0, safety: bool = True, gaps_s=None, design: str = "m1", sequential: bool = False):
        from rl_search.rl_real import RealRLEnv
        if design not in DESIGNS:
            raise ValueError(f"design {design!r}: one of {sorted(DESIGNS)}")
        self.k, self.safety, self.design = int(k), bool(safety), design
        self.sequential = bool(sequential)
        if self.sequential and DESIGNS[design]["mates"] != "zones":
            raise ValueError("sequential decisions need the m2 teammate channels (design m2)")
        self._seq = None                  # the sequential step under way (begin_step .. end_step)
        self._turn = np.zeros(self.k, np.float32)      # sequential: each UAV's turn in the current step (0 first .. 1 last)
        d = DESIGNS[design]
        self.d_soft, self.d_hard, self.closing_only = float(d["d_soft"]), float(d["d_hard"]), bool(d["closing_only"])
        self.reward_mode, self.mates_mode = d["reward"], d["mates"]
        self.own_w, self.vec_k = float(d.get("own_w", 0.0)), bool(d.get("vec_k", False))
        self.search_rules = dict(d["search"]) if d.get("search") else None          # m3: the rules once a UAV searches
        self.look_max = max(LOOK_S, self.search_rules["look"]) if self.search_rules else LOOK_S
        self.danger_shape = self.search_rules["shape"] if self.search_rules else None
        self.solo_view = False            # True: every UAV's teammate inputs read "no teammate" (a one-UAV model flown in a team, e.g. R16)
        self.gaps_s = tuple(gaps_s) if gaps_s is not None else d["gaps"]
        self.gap_range = None if gaps_s is not None else d["gap_range"]
        self.uavs = [RealRLEnv(replace(cfg, settings=dict(cfg.settings)), seed=seed + 1000*i) for i in range(self.k)]
        self.rng = np.random.default_rng(seed)
        self.maps = self.uavs[0].cfg.maps
        self.dt = float(self.uavs[0].env.cfg.decision_dt_s)
        self.obs_u8 = bool(self.uavs[0].s.get("obs_u8"))
        self.timing = None                # a dict: seconds spent per part (filter, fly_detect, merge, safety_check, obs_*), for compute checks
        self.return_obs = True            # False: step() returns no observations (a replay of recorded commands, e.g. hold-out scoring)

    # ------------------------------------------------------------------ per-step cache (11-20: computed once per step and shared by all
    # suits; before, every suit predicted every teammate's path and turned every teammate's whole flight record into an array, k^2)
    def _step_cache(self) -> dict:
        if getattr(self, "_cache_T", None) != self.T:
            self._cache_T, self._cache = self.T, {}
        return self._cache

    def _prepare_paths(self, i: int):
        """Fill the step cache with ("cand", u) (N_CAND candidate commands, (N_CAND, S, 2)) and ("straight", u) (course target held, (S, 2))
        of UAV i, and of every other active UAV still missing them, in one ``predict_paths_multi`` call (11-24; the same numbers as one call
        per UAV and kind)."""
        c = self._step_cache()
        todo = [int(u) for u in np.flatnonzero(self.active) if ("cand", int(u)) not in c or ("straight", int(u)) not in c]
        if i not in todo:
            todo.append(int(i))
        cand = np.linspace(-1.0, 1.0, N_CAND)
        res = predict_paths_multi([self.uavs[u].env.plant for u in todo],
                                  [[self._chi(u, x) for x in cand] + [self.uavs[u].env.chi_tgt] for u in todo], self.look_max)
        for u, P in zip(todo, res):
            c[("cand", u)], c[("straight", u)] = P[:N_CAND], P[N_CAND]

    def _cand_paths(self, i: int) -> np.ndarray:
        """UAV i's N_CAND candidate commands held for LOOK_S s from its present state: (N_CAND, S, 2)."""
        c = self._step_cache()
        if ("cand", i) not in c:
            self._prepare_paths(i)
        return c[("cand", i)]

    def _straight_path(self, i: int) -> np.ndarray:
        """UAV i flying on (its course target held) for LOOK_S s: (S, 2)."""
        c = self._step_cache()
        if ("straight", i) not in c:
            self._prepare_paths(i)
        return c[("straight", i)]

    def _recent(self, j: int) -> np.ndarray:
        """UAV j's 1 s positions of the last 600 s (the longest tail) as an array (t, x, y)."""
        c = self._step_cache()
        if ("hist", j) not in c:
            h = np.asarray(self.hist[j][-610:], float)
            c[("hist", j)] = h[h[:, 0] >= self.T - 600.0 - 1e-9] if len(h) else h.reshape(0, 3)
        return c[("hist", j)]

    def _fans(self) -> dict:
        """m2: every active UAV's reach fan for the longest lead of the small maps, once per step (``mate_maps.reach_fans``)."""
        c = self._step_cache()
        if "fans" not in c:
            from rl_search.mate_maps import reach_fans
            act = [int(j) for j in np.flatnonzero(self.active)]
            T = max([float(lead) for _, side, lead in self.maps if side <= SMALL_SIDE] + [0.0])
            es = [self.uavs[j].env for j in act]
            fans = reach_fans([e.plant for e in es], [float(e.chi_tgt) for e in es], [float(e.turn) for e in es], self.dt, T) if act and T > 0 else []
            c["fans"] = dict(zip(act, fans))
        elif self._seq is not None:                             # sequential: the fans of the UAVs that moved since, from where they are now
            from rl_search.mate_maps import reach_fans
            stale = [int(j) for j in np.flatnonzero(self.active) if int(j) not in c["fans"]]
            T = max([float(lead) for _, side, lead in self.maps if side <= SMALL_SIDE] + [0.0])
            if stale and T > 0:
                es = [self.uavs[j].env for j in stale]
                c["fans"].update(zip(stale, reach_fans([e.plant for e in es], [float(e.chi_tgt) for e in es], [float(e.turn) for e in es],
                                                       self.dt, T)))
        return c["fans"]

    def _mate_path(self, j: int) -> np.ndarray:
        """UAV j's 1 s positions over LOOK_S s from the step's start as the filter and the safety range see it: flying straight on (its
        course target held) from where it is; in a sequential step, once it has moved: its flown 1 s samples of the step, then its new
        course target held from where it is now (S, 2)."""
        c = self._step_cache()
        if ("mate", j) not in c:
            if self._seq is not None and self._seq["moved"][j]:
                h = [p for p in self.hist[j] if p[0] > self.T + 1e-9]
                done = np.array([p[1:] for p in h], float).reshape(-1, 2)
                more = predict_paths(self.uavs[j].env.plant, [self.uavs[j].env.chi_tgt], self.look_max - len(done))[0] if len(done) < self.look_max else np.zeros((0, 2))
                c[("mate", j)] = np.concatenate([done, more])[:int(round(self.look_max))]
            else:
                c[("mate", j)] = self._straight_path(j)
        return c[("mate", j)]

    def _clock(self, key: str, t0: float) -> float:
        """Add the time since t0 to ``timing[key]`` (when timing is on); returns now."""
        now = time.perf_counter()
        if self.timing is not None:
            self.timing[key] = self.timing.get(key, 0.0) + now - t0
        return now

    # ------------------------------------------------------------------ episode
    def reset(self, options: dict | None = None, gaps=None) -> list:
        """``options``: a case's options (scenario, stage 3) or None (the first UAV's training draw: pool scenario and launch).
        ``gaps``: launch gaps (s) after the first UAV, else drawn (m1: from GAPS_S; m2: uniform in the design's range, per UAV)."""
        u0 = self.uavs[0]
        if options is None:
            if u0._scen is None or u0._uses >= u0.cfg.reuse:
                u0._scen, u0._uses = u0._new_scenario(), 0
            u0._uses += 1
            options = u0.curriculum_options(u0._scen, 3)
        if gaps is None:
            if self.gap_range is not None:
                gaps = [float(self.rng.uniform(*self.gap_range)) for _ in range(self.k - 1)]
            else:
                gaps = [float(self.rng.choice(self.gaps_s)) for _ in range(self.k - 1)]
        t0 = float(options.get("t_launch_s", 0.0))
        self.launch_t = t0 + np.concatenate([[0.0], np.cumsum(gaps)]) if self.k > 1 else np.array([t0])
        # joining on the decision clock: the next decision at or after the launch; the seconds in between flown straight on
        self.join_t = t0 + self.dt*np.ceil((self.launch_t - t0)/self.dt - 1e-9)
        scen = options["scenario"]
        base = np.asarray(scen.start[:2], float)
        hd = np.radians(float(getattr(scen, "start_heading_deg", 0.0)))
        self.hist = []
        for u, tl, tj in zip(self.uavs, self.launch_t, self.join_t):
            extra = float(tj - tl)
            opt = dict(options, t_launch_s=float(tj))
            if extra > 1e-6:
                v = float(u.env.v)
                p = base + extra*v*np.array([np.cos(hd), np.sin(hd)])
                opt["scenario"] = replace(scen, start=np.array([p[0], p[1]], float))
            u.reset(options=opt)
            if extra > 1e-6:
                u.env.energy -= extra*float(u.env.power.power(0.0, u.env.v))
            self.hist.append(([(float(tl), float(base[0]), float(base[1]))] if extra > 1e-6 else []) + [(float(tj), *map(float, u.env.plant.pose[:2]))])
        self.N = u0.env.belief.n
        self.W = u0.env.belief.w.copy()
        for u in self.uavs:
            u.env.belief.w[:] = self.W
        self.T = t0
        self._cache_T = None                                    # the per-step cache of the last episode is not valid any more
        self._seq = None
        self._turn[:] = 0.0
        self.launched = self.join_t <= self.T + 1e-9
        self.searching = np.zeros(self.k, bool)                 # m3: in search mode (for good) since search_t
        self.search_t = np.full(self.k, np.nan)
        self.lim = np.full((self.k, self.k), np.nan)            # m3: the hard limit of a pair under the search rules (<= 500 m)
        self.ended = np.zeros(self.k, bool)
        self.found = 0.0
        self.log = dict(t=[], found=[], dist=[], cut=[], viol=[], share=[], seen=[])
        if self.search_rules:
            self._update_search()
        return [self.observation(i) for i in range(self.k)]

    # ------------------------------------------------------------------ trainer hooks (rl_multi_vec forwards env_method to the worlds)
    def set_ring(self, lo_m: float, hi_m: float):
        """Launch ring of this world's next episodes (the first UAV draws the scenario and the launch)."""
        self.uavs[0].ring_m = (float(lo_m), float(hi_m))

    def pool_cursor(self) -> dict:
        return dict(self.uavs[0]._pool_next)

    def set_pool_cursor(self, cursor: dict):
        self.uavs[0]._pool_next = {str(k): int(v) for k, v in (cursor or {}).items()}

    @property
    def active(self) -> np.ndarray:
        return self.launched & ~self.ended

    @property
    def done(self) -> bool:
        return bool(self.launched.all() and self.ended.all())

    # ------------------------------------------------------------------ safety filter
    def _chi(self, i: int, a: float) -> float:
        e = self.uavs[i].env
        return float(e.chi_tgt + float(np.clip(a, -1.0, 1.0))*e.turn)

    def safe_actions(self, actions) -> tuple:
        """(applied actions, cut per UAV): each active UAV's command is kept unless, held for LOOK_S s, it brings it within d_hard of
        another active UAV's held command; then the nearest command that does not (none: the largest separation). Two passes in order."""
        a = np.array([float(np.clip(np.ravel(x)[0], -1.0, 1.0)) for x in actions])
        act = np.flatnonzero(self.active)
        if not self.safety or len(act) < 2:
            return a, np.zeros(self.k)
        paths = dict(zip(act, (P[0] for P in predict_paths_multi([self.uavs[i].env.plant for i in act], [[self._chi(i, a[i])] for i in act],
                                                                   self.look_max))))
        cand = np.linspace(-1.0, 1.0, N_CAND)
        for _ in range(2):
            for i in act:
                others = [(paths[j], *self._pair_rule(i, j)) for j in act if j != i]
                if self._margins(paths[i][None], others)[0] >= 0.0:
                    continue
                P = self._cand_paths(i)                                                                     # (M, S, 2)
                marg = self._margins(P, others)
                ok = np.flatnonzero(marg >= 0.0)
                j = ok[np.argmin(np.abs(cand[ok] - a[i]))] if len(ok) else int(np.argmax(marg))
                a[i], paths[i] = float(cand[j]), P[j]
        prop = np.array([float(np.clip(np.ravel(x)[0], -1.0, 1.0)) for x in actions])
        return a, np.abs(a - prop)*self.active

    def _pair_rule(self, i: int, j: int) -> tuple:
        """(hard limit m, look-ahead s) of the pair now: the design's (m1 / m2, and m3 while both are in transit), or m3's search rules once
        either searches (the pair's limit, at most 500 m)."""
        s = self.search_rules
        if s is None or not (self.searching[i] or self.searching[j]):
            return self.d_hard, LOOK_S
        lim = self.lim[min(i, j), max(i, j)]
        return (float(lim) if np.isfinite(lim) else s["d_hard"]), s["look"]

    @staticmethod
    def _margins(P: np.ndarray, others) -> np.ndarray:
        """Per candidate path in P (M, S, 2): the smallest (separation - hard limit) to the ``others`` [(path (S', 2), limit, look s)] over
        each pair's look-ahead (1 s samples); >= 0 keeps every limit."""
        out = []
        for path, dh, look in others:
            n = int(round(look))
            out.append(np.hypot(*(P[:, :n] - path[None, :n]).transpose(2, 0, 1)).min(1) - dh)
        return np.min(out, axis=0)

    def safe_range(self, i: int) -> tuple:
        """Commands of UAV i that keep d_hard from every other active UAV flying straight on (its course target held), as (low, high);
        (-1, 1) alone or without the filter. Shown in the observation (the filter itself uses the teammates' actual commands)."""
        act = [j for j in np.flatnonzero(self.active) if j != i]
        if not self.safety or not act or not self.active[i]:
            return -1.0, 1.0
        cand = np.linspace(-1.0, 1.0, N_CAND)
        P = self._cand_paths(i)
        marg = self._margins(P, [(self._mate_path(j), *self._pair_rule(i, j)) for j in act])
        ok = np.flatnonzero(marg >= 0.0)
        return (float(cand[ok[0]]), float(cand[ok[-1]])) if len(ok) else (float(cand[np.argmax(marg)]),)*2

    # ------------------------------------------------------------------ step
    def step(self, actions) -> tuple:
        """One 5 s decision of every active UAV. Returns (observations, rewards, done, info); rewards and observations of inactive UAVs
        are 0 / their last state (the trainer masks them, ``info["active"]`` of the step's start; ``info["ended_now"]`` = flights that
        ended in this step)."""
        if self._seq is not None:
            raise RuntimeError("a sequential step is under way (begin_step .. end_step)")
        act0 = self.active.copy()
        ended0 = self.ended.copy()
        t0 = time.perf_counter()
        a, cut = self.safe_actions(actions)
        t0 = self._clock("filter", t0)
        W0 = self.W.copy()
        s = np.ones((self.k, self.N))
        infos, look = [None]*self.k, np.zeros(self.k)
        for i in np.flatnonzero(act0):
            t0 = time.perf_counter()
            u = self.uavs[i]
            u.env.belief.w[:] = W0
            lc = u.look_cum
            _, term, _, inf = u._advance(np.array([a[i]]))
            m = W0 > 0
            s[i, m] = u.env.belief.w[m]/W0[m]
            look[i] = u.look_cum - lc
            infos[i] = inf
            smp = getattr(u.env, "last_samples", None)
            t_s = self.T + np.arange(1, 6)*self.dt/5
            if smp is not None and len(smp):
                self.hist[i] += [(float(t), float(p[0]), float(p[1])) for t, p in zip(t_s, smp)]
            self.ended[i] |= bool(term)
            self._clock("fly_detect", t0)
        t0 = time.perf_counter()
        W1 = W0*np.prod(s, axis=0)
        rewards, info = self._finish(act0, ended0, W0, W1, s, a, cut, look, infos, t0)
        obs = [self.observation(i) for i in range(self.k)] if self.return_obs else [None]*self.k
        return obs, rewards, self.done, info

    def _finish(self, act0, ended0, W0, W1, s, a, cut, look, infos, t0) -> tuple:
        """The end of a step (simultaneous or sequential): the shared belief W1, team / own finds, the clock, the safety penalty over the
        step's 1 s samples, the rewards, the launches due, the log. Returns (rewards, info)."""
        rewards = np.zeros(self.k)
        team = float((W0.sum() - W1.sum())/self.N)
        share = np.array([float((W0*np.prod(np.delete(s, i, 0), axis=0)*(1 - s[i])).sum()/self.N) for i in range(self.k)])
        seen = np.array([float((W0*(1 - s[i])).sum()/self.N) for i in range(self.k)])
        self.W = W1
        for u in self.uavs:
            u.env.belief.w[:] = W1
        self.found += team
        self.T += self.dt
        t0 = self._clock("merge", t0)
        # safety: distances over the step's 1 s samples, the penalty from d_soft, violations within d_hard
        pen, viol, dmin = np.zeros(self.k), np.zeros(self.k), np.inf
        pairs = []
        act_ids = np.flatnonzero(act0)
        for x, i in enumerate(act_ids):
            for j in act_ids[x + 1:]:
                pi, pj = (np.array([h[1:] for h in self.hist[q][-5:]]) for q in (i, j))
                n = min(len(pi), len(pj))
                d = np.hypot(*(pi[-n:] - pj[-n:]).T) if n else np.array([np.inf])
                dmin = min(dmin, float(d.min()))
                ei, ej = self.uavs[i].env.plant, self.uavs[j].env.plant
                t_c, d_c = cpa(ei.pose[:2], ei.pose[2], ej.pose[:2], ej.pose[2], ei.v)
                if self.search_rules is not None:               # m3: each UAV's penalty by its own mode, the pair's limit by the pair's rule
                    pen_ij, v, row = self._pair_penalty_m3(i, j, d, t_c, d_c)
                    pen[[i, j]] += pen_ij
                    viol[[i, j]] += v
                    pairs.append(row)
                    continue
                closing = True
                if self.closing_only:                           # m2: the distance term only while the pair closes in
                    rel = np.asarray(ej.pose[:2], float) - np.asarray(ei.pose[:2], float)
                    vrel = ej.v*np.array([np.cos(ej.pose[2]), np.sin(ej.pose[2])]) - ei.v*np.array([np.cos(ei.pose[2]), np.sin(ei.pose[2])])
                    closing = float(rel @ vrel) < 0.0
                p = (C_CPA*(1 - t_c/LOOK_S)*(1 - d_c/self.d_soft) if (t_c <= LOOK_S and d_c < self.d_soft) else 0.0) \
                    + (C_DIST*(1 - d[-1]/self.d_soft) if (d[-1] < self.d_soft and closing) else 0.0)
                v = float((d < self.d_hard).any())
                pen[[i, j]] += p
                viol[[i, j]] += v
        self._clock("safety_check", t0)
        cfg = self.uavs[0].cfg
        pos_w = float(self.uavs[0].env.pos_w)
        rem = float(W0.sum()/self.N)
        if self.reward_mode == "share":
            found_i = share
        elif self.reward_mode == "team_rel":
            found_i = (team + self.own_w*seen)/max(rem, REM_FLOOR)
        else:
            found_i = np.full(self.k, team)
        parts = np.zeros((4, self.k))                           # per UAV: search, A', cut cost, safety (penalty + violations), as rewarded
        for i in np.flatnonzero(act0):
            u = self.uavs[i]
            edge_cut = abs(a[i] - float(getattr(u.env, "a_applied", a[i])))
            parts[:, i] = (cfg.reward_scale*pos_w*found_i[i], cfg.reward_scale*u.look_w*look[i], cfg.mask_cost*(cut[i] + edge_cut),
                           pen[i] + P_HARD*viol[i])
            rewards[i] = parts[0, i] + parts[1, i] - parts[2, i] - parts[3, i]
        follow = self.following() & act0
        newly = (~self.launched) & (self.join_t <= self.T + 1e-9)
        self.launched |= newly
        if self.search_rules is not None:
            self._update_search()
        self.log["t"].append(self.T)
        self.log["found"].append(self.found)
        self.log["dist"].append(dmin)
        self.log["cut"].append(cut.copy())
        self.log["viol"].append(viol.copy())
        self.log["share"].append(share)
        self.log["seen"].append(seen)
        info = dict(active=act0, team=team, found=self.found, share=share, seen=seen, cut=cut, penalty=pen, violation=viol, dmin=dmin,
                    applied=a, uav=infos, look=look, ended_now=self.ended & ~ended0, rem=rem, parts=parts, follow=follow)
        if self.search_rules is not None:
            info.update(searching=self.searching.copy(), pairs=pairs)
        return rewards, info

    # ------------------------------------------------------------------ m3: transit / search rules (user 2026-11-24, see DESIGNS)
    def _pair_penalty_m3(self, i: int, j: int, d: np.ndarray, t_c: float, d_c: float) -> tuple:
        """m3: (penalty of i and j, violation 0/1, record) of a pair over the step: ``d`` the distances at the step's 1 s samples, (t_c, d_c)
        the closest approach if both fly straight on from the end of the step. The limit is the pair's rule at the step's start."""
        s = self.search_rules
        ei, ej = self.uavs[i].env.plant, self.uavs[j].env.plant
        rel = np.asarray(ej.pose[:2], float) - np.asarray(ei.pose[:2], float)
        vrel = ej.v*np.array([np.cos(ej.pose[2]), np.sin(ej.pose[2])]) - ei.v*np.array([np.cos(ei.pose[2]), np.sin(ei.pose[2])])
        rdot = float(rel @ vrel)/max(float(np.hypot(*rel)), 1e-9)        # m/s, > 0 moving apart
        dh, _ = self._pair_rule(i, j)
        tol = 1.0 if dh < s["d_hard"] else 0.0               # a limit below 500 m (rules changed with the pair closer): 1 m slack
        v = float((d < dh - tol).any())
        out = np.zeros(2)
        for q, u in enumerate((i, j)):
            if self.searching[u]:
                g = lambda x: float(danger_level(x, self.danger_shape, s["d_hard"], s["d_soft"]))
                out[q] = (C_DIST*g(d[-1]) if rdot <= s["sep_mps"] else 0.0) + \
                    (C_CPA*(1 - t_c/s["look"])*g(d_c) if t_c <= s["look"] else 0.0)
            else:                                            # transit: m2's penalty
                out[q] = (C_CPA*(1 - t_c/LOOK_S)*(1 - d_c/self.d_soft) if (t_c <= LOOK_S and d_c < self.d_soft) else 0.0) \
                    + (C_DIST*(1 - d[-1]/self.d_soft) if (d[-1] < self.d_soft and rdot < 0.0) else 0.0)
        return out, v, (int(i), int(j), float(d[-1]), float(d.min()), rdot, float(t_c), float(d_c), float(dh), bool(self.searching[i]),
                        bool(self.searching[j]))

    def _update_search(self):
        """m3: a UAV in transit starts searching (for good) once at least SEARCH_FRAC of the belief left lies inside its SEARCH_SIDE map
        (heading up, the UAV in the centre); a pair that comes under the search rules gets the limit min(500 m, its distance now), and a
        limit below 500 m rises with the pair's separation."""
        s = self.search_rules
        was = self.searching.copy()
        todo = [int(i) for i in np.flatnonzero(self.active) if not self.searching[i]]
        tot = float(self.W.sum())
        if todo and tot > 0:
            b = self.uavs[0].env.belief
            live = (self.W > 0) & b.active(self.T)
            P, w = b.positions(self.T)[live], self.W[live]
            half = SEARCH_SIDE/2
            for i in todo:
                x, y, yaw = self.uavs[i].env.plant.pose
                dx, dy = P[:, 0] - x, P[:, 1] - y
                c, sn = np.cos(yaw), np.sin(yaw)
                inside = (np.abs(dx*c + dy*sn) <= half) & (np.abs(dx*sn - dy*c) <= half)
                if float(w[inside].sum())/tot >= SEARCH_FRAC:
                    self.searching[i] = True
                    self.search_t[i] = self.T
        on = np.flatnonzero(self.launched & ~self.ended)
        for x, i in enumerate(on):
            for j in on[x + 1:]:
                if not (self.searching[i] or self.searching[j]):
                    continue
                dnow = float(np.hypot(*(self.uavs[i].env.plant.pose[:2] - self.uavs[j].env.plant.pose[:2])))
                if not (was[i] or was[j]) or not np.isfinite(self.lim[i, j]):
                    self.lim[i, j] = min(s["d_hard"], dnow)
                elif self.lim[i, j] < s["d_hard"]:
                    self.lim[i, j] = min(s["d_hard"], max(self.lim[i, j], dnow))

    # ------------------------------------------------------------------ sequential decisions (user 2026-11-23, see the module doc)
    def begin_step(self) -> list:
        """Start a sequential 5 s step: the active UAVs in a random order (returned); every UAV's turn entry is set (0 first .. 1 last;
        a UAV not flying this step: 0)."""
        if not self.sequential:
            raise RuntimeError("begin_step needs sequential=True")
        if self._seq is not None:
            raise RuntimeError("the last sequential step was not ended")
        act0 = self.active.copy()
        order = [int(i) for i in self.rng.permutation(np.flatnonzero(act0))]
        self._turn[:] = 0.0
        for m, i in enumerate(order):
            self._turn[i] = m/max(len(order) - 1, 1)
        self._seq = dict(act0=act0, ended0=self.ended.copy(), W0=self.W.copy(), s=np.ones((self.k, self.N)), order=order,
                         a=np.zeros(self.k), cut=np.zeros(self.k), look=np.zeros(self.k), infos=[None]*self.k, moved=np.zeros(self.k, bool))
        return order

    def turn_observation(self, m: int) -> dict:
        """The observation of the m-th UAV of the order: the belief after the detections of the UAVs before it, those UAVs where they are
        after the step, its turn in the vector."""
        return self.observation(self._seq["order"][m])

    def _seq_safe(self, i: int, prop: float) -> float:
        """The filter of a sequential step for UAV i's command ``prop``: kept unless, held for LOOK_S s, it brings UAV i within d_hard of
        another active UAV's ``_mate_path``; then the nearest command that does not (none: the largest separation)."""
        mates = [int(j) for j in np.flatnonzero(self.active) if j != i]
        if not self.safety or not mates:
            return prop
        others = [(self._mate_path(j), *self._pair_rule(i, j)) for j in mates]
        own = predict_paths(self.uavs[i].env.plant, [self._chi(i, prop)], self.look_max)[0]
        if self._margins(own[None], others)[0] >= 0.0:
            return prop
        P = self._cand_paths(i)
        marg = self._margins(P, others)
        ok = np.flatnonzero(marg >= 0.0)
        cand = np.linspace(-1.0, 1.0, N_CAND)
        return float(cand[ok[np.argmin(np.abs(cand[ok] - prop))]] if len(ok) else cand[int(np.argmax(marg))])

    def turn_act(self, m: int, action) -> None:
        """The m-th UAV of the order flies its 5 s with ``action`` (after the filter); its detections lower the shared belief at once."""
        q = self._seq
        i = q["order"][m]
        prop = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        t0 = time.perf_counter()
        a = self._seq_safe(i, prop)
        t0 = self._clock("filter", t0)
        Wb = self.W.copy()
        u = self.uavs[i]
        u.env.belief.w[:] = Wb
        lc = u.look_cum
        _, term, _, inf = u._advance(np.array([a]))
        mk = Wb > 0
        q["s"][i, mk] = u.env.belief.w[mk]/Wb[mk]
        q["look"][i] = u.look_cum - lc
        q["infos"][i] = inf
        q["a"][i], q["cut"][i] = a, abs(a - prop)
        smp = getattr(u.env, "last_samples", None)
        t_s = self.T + np.arange(1, 6)*self.dt/5
        if smp is not None and len(smp):
            self.hist[i] += [(float(t), float(p[0]), float(p[1])) for t, p in zip(t_s, smp)]
        self.ended[i] |= bool(term)
        self.W = u.env.belief.w.copy()
        for v in self.uavs:
            v.env.belief.w[:] = self.W
        q["moved"][i] = True
        c = self._step_cache()                                  # what was computed from UAV i's state before it moved
        for key in (("cand", i), ("straight", i), ("mate", i), ("hist", i)):
            c.pop(key, None)
        c.get("fans", {}).pop(i, None)
        self._clock("fly_detect", t0)

    def end_step(self) -> tuple:
        """End the sequential step: (rewards, done, info) as ``step`` (info also has the order); the observations come with the next
        step's turns."""
        q = self._seq
        unmoved = [i for i in q["order"] if not q["moved"][i]]
        if unmoved:
            raise RuntimeError(f"UAVs {unmoved} have not moved in this sequential step")
        rewards, info = self._finish(q["act0"], q["ended0"], q["W0"], self.W.copy(), q["s"], q["a"], q["cut"], q["look"], q["infos"],
                                     time.perf_counter())
        info["order"] = np.array(q["order"], int)
        self._seq = None
        self._turn[:] = 0.0
        return rewards, self.done, info

    def seq_step(self, act_fn) -> tuple:
        """One sequential step with ``act_fn(i, obs) -> action`` for the UAV i whose turn it is (evaluation flights): (rewards, done, info)."""
        order = self.begin_step()
        for m, i in enumerate(order):
            self.turn_act(m, act_fn(i, self.turn_observation(m)))
        return self.end_step()

    def following(self) -> np.ndarray:
        """Per UAV: a teammate (active) within FOLLOW_M ahead, within FOLLOW_DEG of its heading, flying within FOLLOW_DEG of the same
        heading (the column flights of M1; logged, user 11-23)."""
        out = np.zeros(self.k, bool)
        act = np.flatnonzero(self.active)
        lim = np.radians(FOLLOW_DEG)
        for i in act:
            xi, yi, hi = self.uavs[i].env.plant.pose
            for j in act:
                if j == i:
                    continue
                xj, yj, hj = self.uavs[j].env.plant.pose
                dx, dy = xj - xi, yj - yi
                d = float(np.hypot(dx, dy))
                if d > FOLLOW_M or d < 1e-6:
                    continue
                bearing = abs((np.arctan2(dy, dx) - hi + np.pi) % (2*np.pi) - np.pi)
                if bearing <= lim and abs((hj - hi + np.pi) % (2*np.pi) - np.pi) <= lim:
                    out[i] = True
                    break
        return out

    # ------------------------------------------------------------------ observation
    def _tail(self, i: int, k: int, x: float, y: float, yaw: float) -> np.ndarray:
        """m1: teammate tails on map k of UAV i (heading up, UAV i in the centre): the 1 s positions of the last ``lead`` s of every
        launched teammate, value 1 - age / lead, the largest per cell."""
        _, side, lead = self.maps[k]
        px = self.uavs[i].cfg.map_px
        g = np.zeros((px, px), np.float32)
        win = max(float(lead), 5.0)
        pts, vals = [], []
        for j in range(self.k):
            if j == i or not self.launched[j]:
                continue
            h = self._recent(j)
            h = h[h[:, 0] >= self.T - win - 1e-9]
            if len(h):
                pts.append(h[:, 1:])
                vals.append(1.0 - (self.T - h[:, 0])/win)
        if not pts:
            return g
        P, V = np.concatenate(pts), np.concatenate(vals)
        d = P - np.array([x, y])
        c, s = np.cos(yaw), np.sin(yaw)
        fwd, right = d[:, 0]*c + d[:, 1]*s, d[:, 0]*s - d[:, 1]*c
        cell = side/px
        row = np.floor((side/2 - fwd)/cell).astype(int)
        col = np.floor((right + side/2)/cell).astype(int)
        ok = (row >= 0) & (row < px) & (col >= 0) & (col < px)
        np.maximum.at(g, (row[ok], col[ok]), V[ok].astype(np.float32))
        return g

    def extras(self, i: int) -> np.ndarray:
        """m1 EXTRA_NAMES: teammates airborne and pending (/ 9), time to the next launch (/ 60 s), launch rank (0 first .. 1 last), and the
        most threatening airborne teammate: distance now and closest approach if both fly straight (/ 2000 m, 1 = none) and its time (/ LOOK_S)."""
        mates = [j for j in range(self.k) if j != i]
        air = [j for j in mates if self.active[j]]
        pend = [j for j in mates if not self.launched[j]]
        nxt = min((self.join_t[j] - self.T for j in pend), default=0.0)
        rank = i/(self.k - 1) if self.k > 1 else 0.0
        near, c_d, c_t = 2000.0, 2000.0, LOOK_S
        if self.active[i]:
            pi = self.uavs[i].env.plant
            for j in air:
                pj = self.uavs[j].env.plant
                near = min(near, float(np.hypot(*(pj.pose[:2] - pi.pose[:2]))))
                t, d = cpa(pi.pose[:2], pi.pose[2], pj.pose[:2], pj.pose[2], pi.v)
                if t <= LOOK_S and d < c_d:
                    c_d, c_t = d, t
        return np.array([len(air)/9, len(pend)/9, min(nxt/60.0, 1.0), rank, min(near/2000.0, 1.0), min(c_d/2000.0, 1.0),
                         min(c_t/LOOK_S, 1.0)], np.float32)

    def base_observation(self, i: int) -> dict:
        """UAV i's single-UAV observation (R15's inputs, float32) on the shared belief, the allowed range narrowed by the safety range."""
        t0 = time.perf_counter()
        u = self.uavs[i]
        obs = u.observation_raw() if hasattr(u, "observation_raw") else u.observation()
        t0 = self._clock("obs_maps", t0)
        lo, hi = self.safe_range(i)
        self._clock("obs_safe_range", t0)
        v = obs["vec"].copy()
        v[6], v[7] = max(v[6], lo), min(v[7], hi)
        if v[6] > v[7]:
            v[6] = v[7] = lo
        obs["vec"] = v
        return obs

    def mate_channels(self, i: int) -> dict:
        """m2: the teammate channels of UAV i per map name: small maps [③ claim, danger], large maps [plan B] (``rl_search.mate_maps``);
        teammates = the other active UAVs (one not yet joined is not shown)."""
        from rl_search.mate_maps import claim_map, danger_map, danger_map_graded, time_adv_map
        x, y, yaw = self.uavs[i].env.plant.pose
        px = self.uavs[i].cfg.map_px
        mates = [] if self.solo_view else [j for j in np.flatnonzero(self.active) if j != i]
        graded = self.search_rules is not None and bool(self.searching[i])     # m3: searching, the danger channel draws danger_level
        mates_xy = [tuple(self.uavs[j].env.plant.pose[:2]) for j in mates]
        speed = float(self.uavs[i].env.v)
        out = {}
        fans = None
        for name, side, lead in self.maps:
            if side <= SMALL_SIDE:
                if fans is None:
                    allf = self._fans()
                    fans = [allf[j] for j in mates if j in allf]
                if graded:
                    s = self.search_rules
                    dz = danger_map_graded(x, y, yaw, fans, side, float(lead), lambda dd: danger_level(dd, self.danger_shape, s["d_hard"], s["d_soft"]),
                                           s["d_soft"], px)
                else:
                    dz = danger_map(x, y, yaw, fans, side, float(lead), self.d_soft, px)
                out[name] = [claim_map(x, y, yaw, mates_xy, side, px), dz]
            else:
                out[name] = [time_adv_map(x, y, yaw, mates_xy, side, float(lead), speed, px)]
        return out

    def mate_ranges(self) -> dict:
        """Value range of every channel per map name (base channels (0, 1), then the teammate channels), for ``rl_real.maps_to_u8``."""
        out = {}
        for name, side, _ in self.maps:
            n_base = self._n_base.get(name, 0)
            if self.mates_mode == "zones":
                extra = [(-1.0, 1.0), (0.0, 1.0)] if side <= SMALL_SIDE else [(-1.0, 1.0)]
            else:
                extra = [(0.0, 1.0)]
            out[name] = [(0.0, 1.0)]*n_base + extra
        return out

    def observation(self, i: int) -> dict:
        """base_observation + the teammate inputs of the design (m1: a tail channel on every map + EXTRA_NAMES in the vector; m2: the
        ``mate_channels`` and 1 / team size in the vector), the maps as uint8 with the setting ``obs_u8``."""
        obs = self.base_observation(i)
        self._n_base = {name: obs[name].shape[0] for name, _, _ in self.maps}
        t0 = time.perf_counter()
        if self.mates_mode == "zones":
            ch = self.mate_channels(i)
            for name, _, _ in self.maps:
                obs[name] = np.concatenate([obs[name], np.stack(ch[name]).astype(np.float32)], 0)
            if self.vec_k:                                      # 1 / team size, pending UAVs included (user 11-23)
                obs["vec"] = np.concatenate([obs["vec"], np.array([1.0 if self.solo_view else 1.0/self.k], np.float32)])
            if self.sequential:                                 # its turn in the step (0 first .. 1 last; user 11-23 "순차 결정")
                obs["vec"] = np.concatenate([obs["vec"], self._turn[i:i + 1]])
            self._clock("obs_mates", t0)
        else:
            x, y, yaw = self.uavs[i].env.plant.pose
            for k, (name, _, _) in enumerate(self.maps):
                obs[name] = np.concatenate([obs[name], self._tail(i, k, x, y, yaw)[None]], 0)
            t0 = self._clock("obs_tails", t0)
            obs["vec"] = np.concatenate([obs["vec"], self.extras(i)])
            self._clock("obs_extras", t0)
        if self.obs_u8:
            from rl_search.rl_real import maps_to_u8
            obs = maps_to_u8(obs, self.mate_ranges())
        return obs
