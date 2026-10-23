"""Planners that fly through SearchEnv: straight, greedy and tree, forked from uav_search.py.

Planners still read the belief directly and plan with the kinematic step model (yaw += turn,
fixed step); ``act`` returns only the heading change of the first planned step, which the env
passes to its plant. With ``KinematicPlant`` this reproduces uav_search.run_mission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
import numpy as np

from rl_search.mission import Coverage, ParticleBelief, SearchConfig, footprint_probs, poa


@dataclass
class Items:
    """Probability carriers near the UAV: moving particles first (``m`` of them), then static unseen cells.

    P: (H, M, 2) positions at each planned step; w: mass in POS units; pod, h: per-item detection
    probability and kernel width. Far particles (and coarse unseen cells) only enter the terminal term.
    """
    P: np.ndarray
    w: np.ndarray
    pod: np.ndarray
    h: np.ndarray
    m: int
    pts_far: np.ndarray
    w_far: np.ndarray
    bin_cell: float
    blocks: tuple | None = None

    def probs(self, j: int, nodes: np.ndarray, cfg: SearchConfig) -> np.ndarray:
        """Footprint probabilities (len(nodes), M) for poses ``nodes`` at planned step j."""
        f = lambda pts, h: footprint_probs(pts[:, 0], pts[:, 1], nodes[:, 0], nodes[:, 1], nodes[:, 2],
                                           cfg.fov_along_m, cfg.fov_cross_m, h)
        if self.m == len(self.w):
            return f(self.P[j], cfg.bandwidth_m)
        return np.hstack([f(self.P[j, :self.m], cfg.bandwidth_m), f(self.P[j, self.m:], self.h[self.m])])

    def far(self, j: int):
        xy, w = bin_weights(self.pts_far[j], self.w_far, self.bin_cell)
        if self.blocks is not None:
            xy, w = np.vstack([xy, self.blocks[0]]), np.r_[w, self.blocks[1]]
        return xy, w


def bin_weights(xy, w, cell):
    """Aggregate weighted points into square cells: (centroids (B, 2), weights (B,))."""
    if len(w) == 0:
        return xy, w
    _, inv = np.unique(np.floor(xy/cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    wsum = np.bincount(inv, w)
    cx = np.bincount(inv, w*xy[:, 0])/wsum
    cy = np.bincount(inv, w*xy[:, 1])/wsum
    return np.stack([cx, cy], axis=1), wsum


class Planner:
    """Chooses future poses; ``plan`` returns (U, H, 3) poses for t + dt ... t + H*dt."""
    name = "base"
    horizon = 1

    def __init__(self, cfg: SearchConfig, n_headings: int = 5):
        self.cfg = cfg
        self.turns = np.deg2rad(np.linspace(-cfg.max_turn_deg, cfg.max_turn_deg, n_headings))
        self.fallback = False
        self.coverage: Coverage | None = None

    @property
    def exploring(self) -> bool:
        return self.coverage is not None and (self.cfg.explore_weight > 0 or self.cfg.explore_density > 0)

    def mass_scale(self, belief: ParticleBelief) -> float:
        if self.cfg.particle_weights == "conditional":
            return max(float(belief.w.sum()), 1e-300)
        return float(belief.n)

    def gather(self, belief: ParticleBelief, times: np.ndarray, pose: np.ndarray, min_weight: float,
               near_sigmas: float, bin_cell: float) -> Items:
        """Items scored exactly for paths from ``pose`` over ``times``, plus far mass for the terminal term."""
        cfg = self.cfg
        path = len(times)*cfg.step_m + .5*np.hypot(cfg.fov_along_m, cfg.fov_cross_m)
        pts_t = np.stack([belief.positions(t) for t in times])
        live = belief.w > min_weight
        d = np.minimum(np.hypot(*(pts_t[0] - pose[:2]).T), np.hypot(*(pts_t[-1] - pose[:2]).T))
        near = live & (d < path + near_sigmas*cfg.bandwidth_m)
        far = live & ~near
        m = int(near.sum())
        scale = self.mass_scale(belief)
        it = Items(P=pts_t[:, near], w=belief.w[near]/scale, pod=np.full(m, cfg.pod),
                   h=np.full(m, cfg.bandwidth_m), m=m, pts_far=pts_t[:, far], w_far=belief.w[far]/scale,
                   bin_cell=bin_cell)
        if self.exploring:
            cov = self.coverage
            cxy, cw = cov.near_cells(pose[0], pose[1], path + 3*cov.c, times[0])
            it.P = np.concatenate([it.P, np.broadcast_to(cxy, (len(times),) + cxy.shape)], axis=1)
            it.w = np.r_[it.w, cw]
            it.pod = np.r_[it.pod, np.ones(len(cw))]
            it.h = np.r_[it.h, np.full(len(cw), cov.c)]
            it.blocks = cov.far_blocks(times[0])
        return it

    def advance(self, poses: np.ndarray, turn) -> np.ndarray:
        yaw = poses[..., 2] + turn
        step = self.cfg.step_m
        return np.stack([poses[..., 0] + step*np.cos(yaw), poses[..., 1] + step*np.sin(yaw), yaw], axis=-1)

    def plan(self, belief: ParticleBelief, poses: np.ndarray, t_s: float) -> np.ndarray:
        raise NotImplementedError

    def act(self, belief: ParticleBelief, pose: np.ndarray, t_s: float):
        """Heading change for this step and the planned path (H, 3)."""
        path = self.plan(belief, pose[None, :], t_s)[0]
        return float(path[0, 2] - pose[2]), path

    def toward_centroid(self, belief: ParticleBelief, cand: np.ndarray, t_s: float) -> int:
        """Index of the candidate pose heading most directly at the weighted particle centroid."""
        self.fallback = True
        pts = belief.positions(t_s)
        target = (pts*belief.w[:, None]).sum(0)/max(belief.w.sum(), 1e-300)
        bearing = np.arctan2(target[1]-cand[:, 1], target[0]-cand[:, 0])
        return int(np.argmin(np.abs(np.angle(np.exp(1j*(bearing - cand[:, 2]))))))


class StraightPlanner(Planner):
    """Baseline: keep the launch heading."""
    name = "straight"

    def plan(self, belief, poses, t_s):
        return self.advance(poses, 0.0)[:, None, :]


class GreedyPlanner(Planner):
    """One-step lookahead: the heading whose next footprint holds the most remaining probability.

    If every candidate sees (numerically) nothing, steer towards the weighted centroid instead.
    """
    name = "greedy"

    def __init__(self, cfg, n_headings=5, min_gain=1e-9, **_):
        super().__init__(cfg, n_headings)
        self.min_gain = min_gain

    def plan(self, belief, poses, t_s):
        out = np.empty((len(poses), 1, 3))
        self.fallback = False
        t_next = t_s + self.cfg.dt_s
        for u, pose in enumerate(poses):
            cand = self.advance(pose[None, :], self.turns)
            gain = poa(belief, t_next, cand[:, 0], cand[:, 1], cand[:, 2], self.cfg)
            if self.exploring:
                cov = self.coverage
                reach = self.cfg.step_m + .5*np.hypot(self.cfg.fov_along_m, self.cfg.fov_cross_m) + 3*cov.c
                cxy, cw = cov.near_cells(pose[0], pose[1], reach, t_next)
                p = footprint_probs(cxy[:, 0], cxy[:, 1], cand[:, 0], cand[:, 1], cand[:, 2],
                                    self.cfg.fov_along_m, self.cfg.fov_cross_m, cov.c)
                gain = self.cfg.pod*gain*belief.n/self.mass_scale(belief) + p @ cw
            best = self.toward_centroid(belief, cand, t_next) if gain.max() < self.min_gain else int(np.argmax(gain))
            out[u, 0] = cand[best]
        return out


class TreePlanner(Planner):
    """Receding-horizon beam search over the heading lattice with a wide-kernel terminal term.

    See uav_search.TreePlanner for the objective.
    """
    name = "tree"

    def __init__(self, cfg, n_headings=5, horizon=6, terminal_weight=1.0, terminal_bw_m=600.0,
                 beam=50, min_weight=1e-6, near_sigmas=3.5, chunk=2500, **_):
        super().__init__(cfg, n_headings)
        self.horizon = horizon
        self.terminal_weight = terminal_weight
        self.terminal_bw_m = terminal_bw_m
        self.beam = beam
        self.min_weight = min_weight
        self.near_sigmas = near_sigmas
        self.chunk = chunk

    def _terminal(self, nodes, pts, w_surv, far_xy, far_w):
        """Wide-kernel value of one extra footprint at each node: (M,)."""
        R = self.terminal_bw_m
        k = self.cfg.fov_along_m*self.cfg.fov_cross_m/(2*np.pi*R*R)
        d2 = (nodes[:, 0:1]-pts[None, :, 0])**2 + (nodes[:, 1:2]-pts[None, :, 1])**2
        val = (w_surv*np.exp(-d2/(2*R*R))).sum(1)
        if len(far_w):
            d2 = (nodes[:, 0:1]-far_xy[None, :, 0])**2 + (nodes[:, 1:2]-far_xy[None, :, 1])**2
            val += np.exp(-d2/(2*R*R)) @ far_w
        return k*val

    def plan(self, belief, poses, t_s):
        cfg, H, nh = self.cfg, self.horizon, len(self.turns)
        out = np.empty((len(poses), H, 3))
        self.fallback = False
        times = t_s + cfg.dt_s*np.arange(1, H+1)
        for u, pose in enumerate(poses):
            it = self.gather(belief, times, pose, self.min_weight, self.near_sigmas, self.terminal_bw_m/4)
            w = it.w

            nodes, surv = pose[None, :], np.ones((1, len(w)))
            history = []
            for j in range(H):
                parent = np.repeat(np.arange(len(nodes)), nh)
                child = self.advance(nodes[parent], np.tile(self.turns, len(nodes)))
                last = j == H-1
                prune = self.beam > 0 and len(child) > self.beam
                score_needed = last or prune
                p_near = it.P[j]
                if score_needed and self.terminal_weight > 0:
                    far_xy, far_w = it.far(j)
                s_all = None if last else np.empty((len(child), len(w)))
                score = np.empty(len(child)) if score_needed else None
                for a in range(0, len(child), self.chunk):
                    b = min(a + self.chunk, len(child))
                    s = surv[parent[a:b]]*(1 - it.pod*it.probs(j, child[a:b], cfg))
                    if s_all is not None:
                        s_all[a:b] = s
                    if score_needed:
                        score[a:b] = (w*(1 - s)).sum(1)
                        if self.terminal_weight > 0:
                            score[a:b] += self.terminal_weight*self._terminal(child[a:b], p_near, w*s, far_xy, far_w)
                if prune and not last:
                    keep = np.argpartition(-score, self.beam)[:self.beam]
                    child, parent, s_all, score = child[keep], parent[keep], s_all[keep], score[keep]
                history.append((child, parent))
                nodes, surv = child, s_all

            if score.max() < 1e-12:
                first = self.advance(pose[None, :], self.turns)
                path = [first[self.toward_centroid(belief, first, times[0])]]
                for _ in range(H-1):
                    path.append(self.advance(path[-1], 0.0))
                out[u] = np.array(path)
                continue
            idx = int(np.argmax(score))
            for j in range(H-1, -1, -1):
                child, parent = history[j]
                out[u, j] = child[idx]
                idx = parent[idx]
        return out


PLANNERS = {p.name: p for p in (StraightPlanner, GreedyPlanner, TreePlanner)}


def run_episode(env, planner: Planner, options: dict | None = None, seed: int | None = None) -> dict:
    """Fly one episode with a planner; returns the env log (if recording) plus plan and fallback."""
    obs, info = env.reset(seed=seed, options=options)
    planner.coverage = env.coverage
    plans, fallback = [], []
    done = False
    t0 = time.perf_counter()
    while not done:
        turn, path = planner.act(env.belief, env.pose, env.t_s)
        plans.append(path)
        fallback.append(planner.fallback)
        obs, reward, terminated, truncated, info = env.step([turn])
        done = terminated or truncated
    elapsed = time.perf_counter() - t0
    log = env.log() if env.record else {}
    H = planner.horizon
    plan = np.full((len(plans) + 1, 1, H, 3), np.nan)
    plan[:-1, 0] = np.array(plans)
    log |= dict(plan=plan, fallback=np.r_[fallback, False])
    log["info"] = info | dict(run_time_s=elapsed, steps=len(plans), terminated=bool(terminated))
    return log


def save_mission(out_dir: Path, log: dict, cfg: SearchConfig, planner: Planner, extra: dict | None = None):
    """Write mission.npz in the layout read by mission_viewer.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, yaw = log["poses"][0, 0]
    meta = dict(asdict(cfg), planner=planner.name, horizon=planner.horizon, n_headings=len(planner.turns),
                start_x=float(x), start_y=float(y), start_heading_deg=float(np.degrees(yaw)),
                run_time_s=round(log["info"]["run_time_s"], 1)) | (extra or {})
    arrays = {k: v for k, v in log.items() if k != "info"}
    np.savez_compressed(out_dir/"mission.npz", meta=json.dumps(meta), **arrays)
    (out_dir/"config.json").write_text(json.dumps(meta, indent=2))
