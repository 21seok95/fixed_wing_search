"""Target motion prediction: Monte Carlo particles from the LKP over the fixed terrain.

Fork of ``create_guidelines`` and ``simulate`` from battlefield_infiltrator_demo.py with
these changes for the RL search scenarios:

- LKP, heading and speed are arguments (sampled per scenario) instead of fixed settings.
- The instantaneous speed is capped (fast-move mode included) at ``speed_cap_mps``.
- Guidelines reach ``speed_cap * duration`` and continue past the map edge.
- A particle that leaves the map is frozen at its first outside position and its exit
  time is recorded; the mission layer stops considering it from then on.
- One movement step is a named move model (only ``infiltrator`` for now) so a vehicle
  model on paved / unpaved roads can be added later. Speed is scaled per ``surface`` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from rl_search.terrain import Terrain

MODES = ("CM", "FM", "HD", "RO", "SM")
CM, FM, HD, RO, SM = range(len(MODES))
MOVE_MODES = np.array([CM, FM, SM])
OFFSETS = np.deg2rad(np.array([-100, -75, -50, -30, -15, 0, 15, 30, 50, 75, 100], float))


@dataclass
class TargetConfig:
    particles: int = 5000
    duration_s: float = 4200.0         # flight time + handover margin
    dt_s: float = 5.0                  # simulation step
    save_dt_s: float = 10.0            # stored track interval (multiple of dt_s)
    speed_cap_mps: float = 30/3.6      # instantaneous speed limit, also sets the guideline reach
    speed_rel_std: float = 0.1         # speed_std = speed_rel_std * speed_mean
    heading_std_deg: float = 25.0
    reorient_std_deg: float = 20.0
    p_move: float = .5
    p_reorient: float = .3
    p_stop: float = .2
    move_split: tuple = (.6, .2, .2)   # normal / fast / slow share of moving
    fast_factor: float = 1.6
    slow_factor: float = .5
    action_persistence_min: float = 10.0
    surface_speed: tuple = (1.0, 1.0, 1.0)  # speed factor for off-road / unpaved / paved
    guide_stride_m: float = 60.0
    move_model: str = "infiltrator"

    @property
    def save_every(self) -> int:
        k = self.save_dt_s/self.dt_s
        if abs(k - round(k)) > 1e-9 or k < 1:
            raise ValueError("save_dt_s must be a positive multiple of dt_s")
        return int(round(k))

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s/self.dt_s))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def sample_categorical_rows(rng, scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    prob = np.exp(scores)
    prob /= prob.sum(axis=1, keepdims=True)
    u = rng.random((len(prob), 1))
    return (np.cumsum(prob, axis=1) < u).sum(axis=1)


def _unit_sum(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v/v.sum()


def action_probs(cfg: TargetConfig) -> np.ndarray:
    top = _unit_sum([cfg.p_move, cfg.p_reorient, cfg.p_stop])
    p = np.zeros(len(MODES))
    p[MOVE_MODES] = top[0]*_unit_sum(cfg.move_split)
    p[RO], p[HD] = top[1], top[2]
    return p


def create_guidelines(terr: Terrain, cfg: TargetConfig, lkp: np.ndarray):
    """360 APF-inspired reference lines from the LKP; outside the map they continue straight."""
    stride = cfg.guide_stride_m
    steps = int(np.ceil(cfg.speed_cap_mps*cfg.duration_s/stride))
    theta = np.deg2rad(np.arange(360.0))
    head = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pos = np.tile(lkp, (360, 1)).astype(float)
    path = np.repeat(pos[:, None, :], steps + 1, axis=1)
    active = np.ones(360, bool)
    guide_len = np.full(360, steps)
    heading_vec = head.copy()
    for k in range(1, steps+1):
        px, py = pos[:, 0], pos[:, 1]
        d = np.nan_to_num(terr.sample(terr.clearance, px, py), nan=1e9)
        repel_g = np.stack([np.nan_to_num(terr.sample(terr.clearance_gx, px, py)),
                            np.nan_to_num(terr.sample(terr.clearance_gy, px, py))], axis=1)
        downhill = np.stack([-np.nan_to_num(terr.sample(terr.z_gx, px, py)),
                             -np.nan_to_num(terr.sample(terr.z_gy, px, py))], axis=1)
        repel = normalize(repel_g) * (2.8 * np.clip((175-d)/175, 0, 1))[:, None]
        force = head + repel + normalize(downhill)*.16
        heading_vec = normalize(.60*heading_vec + .40*normalize(force))
        cand = pos + stride*heading_vec
        blocked = np.nan_to_num(terr.sample(terr.forbidden, cand[:, 0], cand[:, 1]), nan=0.0) > .40
        guide_len[active & blocked] = k-1
        active &= ~blocked
        pos[active] = cand[active]
        path[:, k, :] = pos
    return path, guide_len


def init_infiltrator(terr: Terrain, cfg: TargetConfig, guide, lkp, heading_deg, speed_mps, rng):
    n = cfg.particles
    heading = heading_deg + rng.normal(0, cfg.heading_std_deg, n)
    return SimpleNamespace(
        pos=np.tile(lkp, (n, 1)).astype(float), psi=np.deg2rad(heading),
        guide_index=np.rint(heading).astype(int) % 360, gk=np.zeros(n, int),
        turn_dir=rng.choice([-1, 1], n), actions=rng.choice(len(MODES), size=n, p=action_probs(cfg)),
        speed_mean=speed_mps, speed_std=cfg.speed_rel_std*speed_mps)


def step_infiltrator(terr: Terrain, cfg: TargetConfig, guide, guide_len, st, rng, first: bool):
    """One simulation step of the infiltrator model; returns the proposed new positions (N, 2)."""
    probs = action_probs(cfg)
    split = _unit_sum(cfg.move_split)
    pos, psi, n = st.pos, st.psi, len(st.pos)
    rows = np.arange(n)
    if not first:
        switch = rng.random(n) < min(1.0, cfg.dt_s/(cfg.action_persistence_min*60))
        st.actions[switch] = rng.choice(len(MODES), size=switch.sum(), p=probs)
    ro = st.actions == RO
    if ro.any():
        shift = np.rint(rng.normal(0, cfg.reorient_std_deg, ro.sum())).astype(int)
        st.guide_index[ro] = (st.guide_index[ro] + shift) % 360
        d = np.linalg.norm(guide[st.guide_index[ro]] - pos[ro, None, :], axis=2)
        st.gk[ro] = d.argmin(axis=1)
        st.actions[ro] = rng.choice(MOVE_MODES, size=ro.sum(), p=split)

    gi, gk = st.guide_index, st.gk
    end = guide_len[gi]
    for _ in range(20):
        tgt = guide[gi, np.minimum(gk+3, end)]
        adv = (np.linalg.norm(tgt-pos, axis=1) < 90) & (gk+3 < end)
        if not adv.any():
            break
        gk[adv] += 1
    tgt = guide[gi, np.minimum(gk+3, end)]
    ended = (gk+3 >= end) & (np.linalg.norm(tgt-pos, axis=1) < 150)
    if ended.any():
        gi[ended] = (gi[ended] + st.turn_dir[ended]) % 360
        d = np.linalg.norm(guide[gi[ended]] - pos[ended, None, :], axis=2)
        gk[ended] = d.argmin(axis=1)
    end = guide_len[gi]
    lookahead = guide[gi, np.minimum(gk+3, end)]
    to_guide = np.arctan2(lookahead[:, 1]-pos[:, 1], lookahead[:, 0]-pos[:, 0])
    desired = np.arctan2(.4*np.sin(psi) + .6*np.sin(to_guide), .4*np.cos(psi) + .6*np.cos(to_guide))

    cover_here = terr.sample(terr.cover, pos[:, 0], pos[:, 1])
    hd = st.actions == HD
    hidden = hd & (cover_here >= .5)
    seek = hd & ~hidden
    to_cover = np.arctan2(np.nan_to_num(terr.sample(terr.cover_gy, pos[:, 0], pos[:, 1])),
                          np.nan_to_num(terr.sample(terr.cover_gx, pos[:, 0], pos[:, 1])))
    desired[seek] = to_cover[seek]

    fm, sm = st.actions == FM, st.actions == SM
    factor = np.select([fm, sm], [cfg.fast_factor, cfg.slow_factor], 1.0)
    r, c = np.rint((pos[:, 1] - terr.axis[0])/terr.cell).astype(int), \
        np.rint((pos[:, 0] - terr.axis[0])/terr.cell).astype(int)
    last = len(terr.axis) - 1
    surf = terr.surface[np.clip(r, 0, last), np.clip(c, 0, last)]
    factor = factor*np.asarray(cfg.surface_speed, float)[surf]
    speed = np.clip(rng.normal(st.speed_mean*factor, st.speed_std*factor), 0, cfg.speed_cap_mps)
    speed[hidden] = 0

    angles = psi[:, None] + OFFSETS[None, :]
    dist = speed[:, None]*cfg.dt_s
    nx = pos[:, 0, None] + dist*np.cos(angles)
    ny = pos[:, 1, None] + dist*np.sin(angles)
    inside = terr.inside(nx, ny)
    e0 = terr.sample(terr.z, pos[:, 0], pos[:, 1])[:, None]
    ec = np.where(inside, terr.sample(terr.z, nx, ny), e0)
    slope_rad = np.arctan2(ec-e0, np.maximum(dist, 1e-4))
    slope_abs = np.abs(np.degrees(slope_rad))
    cov_c = np.nan_to_num(terr.sample(terr.cover, nx, ny))
    crest = np.clip(np.nan_to_num(terr.sample(terr.tpi, nx, ny))/25, 0, 2)
    w_open = np.select([fm, seek, sm], [.4, 3.0, 3.0], 2.2)[:, None]
    w_crest = np.select([fm, seek, sm], [.3, .8, 1.8], 1.2)[:, None]
    conceal_cost = w_open*(1-cov_c) + w_crest*crest
    slope_cost = 1.2*np.clip(slope_abs/terr.cfg.max_slope_deg, 0, 2)**2
    w_head = np.select([fm, seek, sm], [5.0, 4.0, 3.5], 4.0)[:, None]
    heading_reward = w_head*np.cos(angles-desired[:, None])
    turn_cost = .6*(OFFSETS[None, :]/np.pi)**2
    scores = heading_reward - turn_cost - conceal_cost - slope_cost
    forbidden = np.nan_to_num(terr.sample(terr.forbidden, nx, ny), nan=0.0) > .40
    invalid = forbidden | (slope_abs > terr.cfg.max_slope_deg) | (~np.isfinite(scores))
    scores[invalid] = -1e7
    no_feasible = np.all(invalid, axis=1)
    scores[no_feasible, 5] = 0.0
    choices = sample_categorical_rows(rng, scores)
    chosen_theta = angles[rows, choices]
    angle_deg = np.rad2deg(slope_rad[rows, choices])
    max_slope = terr.cfg.max_slope_deg
    q = np.where(angle_deg >= 0, 1-angle_deg/max_slope, 1-angle_deg/(-(max_slope + 4)))
    q = np.clip(q, 0, 1)
    move = speed*cfg.dt_s*q*(1 - .35*cov_c[rows, choices])
    move[no_feasible] = 0
    new_pos = pos + move[:, None]*np.stack([np.cos(chosen_theta), np.sin(chosen_theta)], axis=1)
    blocked = np.nan_to_num(terr.sample(terr.forbidden, new_pos[:, 0], new_pos[:, 1]), nan=0.0) >= .4
    new_pos[blocked] = pos[blocked]
    move[blocked] = 0
    psi[move > 0] = chosen_theta[move > 0]
    return new_pos


MOVE_MODELS = {"infiltrator": (init_infiltrator, step_infiltrator)}


def simulate(terr: Terrain, cfg: TargetConfig, lkp, heading_deg: float, speed_mps: float, rng):
    """Particle tracks from the LKP.

    Returns ``track`` (T, N, 2) float32 every ``save_dt_s`` and ``exit_step`` (N,) int16: the first
    stored index at which the particle is outside the map (-1 = never leaves). Exited particles stay
    at their first outside position.
    """
    if cfg.move_model not in MOVE_MODELS:
        raise ValueError(f"Unknown move model {cfg.move_model!r}; choose from {sorted(MOVE_MODELS)}")
    init, step = MOVE_MODELS[cfg.move_model]
    lkp = np.asarray(lkp, float)
    guide, guide_len = create_guidelines(terr, cfg, lkp)
    st = init(terr, cfg, guide, lkp, heading_deg, speed_mps, rng)
    every, n = cfg.save_every, cfg.particles
    track = np.empty((cfg.n_steps//every + 1, n, 2), np.float32)
    track[0] = lkp
    exit_step = np.full(n, -1, np.int16)
    alive = np.ones(n, bool)
    for t in range(cfg.n_steps):
        new_pos = step(terr, cfg, guide, guide_len, st, rng, first=t == 0)
        new_pos[~alive] = st.pos[~alive]
        out = alive & ~terr.inside(new_pos[:, 0], new_pos[:, 1])
        if out.any():
            exit_step[out] = -(-(t+1)//every)
            alive &= ~out
        st.pos = new_pos
        if (t+1) % every == 0:
            track[(t+1)//every] = new_pos
    return track, exit_step
