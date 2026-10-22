"""Infiltrator movement prediction on a synthetic battlefield (Monte Carlo + KDE).

Variant of ge_probability_demo.py. The target is an infiltrator whose goal is
unknown, but whose last known position (LKP) and heading are known. The map
has only natural terrain: wide rivers with fords, no-go mountains (absolute
elevation), rolling plains with gullies and low hills, and forest. The
infiltrator does not use roads or trails: it keeps a general heading, prefers
forest and low ground (concealment), avoids open ground and crests, lays up in
cover, and detours around obstacles using the paper's guideline and
route-transition mechanism.

Scale and speed follow Ge et al. (2026) Sec. IV-A: a 65 km x 65 km area, walking
speed N(0.75, 0.25^2) m/s, and 2.5 h of motion (UAVs launch 30 min after the
LKP and search for 2 h). The hand-placed 14 km battlefield sits around the LKP;
the rest of the area is filled procedurally with rivers, fords and massifs.

Run: python battlefield_infiltrator_demo.py --output results_battlefield_paper
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from ge_probability_demo import kde_map, normalize, sample_categorical_rows

# Top level: move / reorient / stop. Moving has three sub-modes:
# CM: covert move (normal speed), FM: fast move (faster, less concealment), SM: slow move (stalking, most concealment).
# HD: stop = go to cover and lay up. RO: reorient = shift the intended heading, then move in a sampled sub-mode.
MODES = ("CM", "FM", "HD", "RO", "SM")
CM, FM, HD, RO, SM = range(len(MODES))
MOVE_MODES = np.array([CM, FM, SM])

PEAKS = [  # (x, y, sigma_x, sigma_y, height) in metres
    (  200,  4000, 1700, 1100, 650),
    (-4900,  2600, 1300, 1500, 600),
    ( 4600, -3200, 1500, 1300, 620),
    (-1900, -1500,  850,  750, 520),
    ( 3800,  2300,  900,  800, 500),
]
FORDS_X = (-2500.0, 700.0, 4800.0)
CORE_HALF_M = 8000.0  # hand-placed battlefield core around the LKP; procedural terrain outside

RIVERS = (  # centre lines y = f(x) in metres; RIVERS[0] is the river near the LKP
    lambda xx: 450*np.sin((xx+1500)/2200) + .12*xx + 200,
    lambda xx: 17000 + 700*np.sin((xx-4000)/2600) - .08*xx,
    lambda xx: -19000 + 600*np.sin((xx+2000)/3000) + .05*xx,
)


@dataclass
class Settings:
    seed: int = 7
    particles: int = 1440
    hours: float = 2.5
    dt_s: float = 30.0
    bandwidth_m: float = 150.0
    map_m: float = 65000.0
    cell_m: float = 50.0
    speed_mean_mps: float = .75
    speed_std_mps: float = .25
    p_move: float = .5                 # top-level behaviour probabilities (normalised)
    p_reorient: float = .3
    p_stop: float = .2
    move_split: tuple = (.6, .2, .2)   # normal / fast / slow share of moving
    fast_factor: float = 1.6           # speed multipliers; with the split above moving averages ~1.02 x mean speed
    slow_factor: float = .5
    max_slope_deg: float = 32.0
    action_persistence_min: float = 10.0
    mountain_z_m: float = 460.0
    river_half_width_m: float = 100.0
    ford_half_width_m: float = 150.0
    forest_threshold: float = 1.0
    lkp_x: float = -4500.0
    lkp_y: float = -3000.0
    heading_deg: float = 40.0
    heading_std_deg: float = 25.0
    reorient_std_deg: float = 20.0


@dataclass
class Terrain:
    axis: np.ndarray
    z: np.ndarray
    water: np.ndarray
    fords: np.ndarray
    mountain: np.ndarray
    forest: np.ndarray
    cover: np.ndarray
    cover_gx: np.ndarray
    cover_gy: np.ndarray
    tpi: np.ndarray
    forbidden: np.ndarray
    river_y: np.ndarray
    slope_deg: np.ndarray
    clearance: np.ndarray
    clearance_gx: np.ndarray
    clearance_gy: np.ndarray
    z_gx: np.ndarray
    z_gy: np.ndarray
    cell: float
    _float_cache: dict = field(default_factory=dict, repr=False)

    def sample(self, fld: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear map lookup; scalar or vector locations, metric coordinates."""
        if fld.dtype != np.float64:
            if id(fld) not in self._float_cache:
                self._float_cache[id(fld)] = fld.astype(float)
            fld = self._float_cache[id(fld)]
        xx, yy = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        rows = (yy - self.axis[0]) / self.cell
        cols = (xx - self.axis[0]) / self.cell
        vals = map_coordinates(fld, [rows.ravel(), cols.ravel()], order=1,
                               mode="constant", cval=np.nan)
        return vals.reshape(xx.shape)

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lo, hi = self.axis[0], self.axis[-1]
        return (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)


def river_center(xx):
    return RIVERS[0](xx)


def outer_massifs(cfg: Settings, rng) -> list:
    """Random no-go massifs outside the hand-placed core, away from rivers and each other."""
    half = cfg.map_m/2
    if half <= CORE_HALF_M:
        return []
    target = int((cfg.map_m/1000)**2 / 90)
    peaks = []
    for _ in range(50*target):
        if len(peaks) >= target:
            break
        px, py = rng.uniform(-half, half, 2)
        if abs(px) < CORE_HALF_M and abs(py) < CORE_HALF_M:
            continue
        if min(abs(py - r(px)) for r in RIVERS) < 2500:
            continue
        if any(np.hypot(px-q[0], py-q[1]) < 4000 for q in peaks + PEAKS):
            continue
        peaks.append((px, py, rng.uniform(800, 2200), rng.uniform(800, 2200), rng.uniform(420, 700)))
    return peaks


def ford_positions(cfg: Settings, rng) -> list:
    """Ford x-positions per river: the core fords plus one roughly every 6 km outside the core."""
    half = cfg.map_m/2
    out = []
    for k in range(len(RIVERS)):
        xs = list(FORDS_X) if k == 0 else []
        for gx in np.arange(-half + 3000, half - 2000, 6000.0):
            if k == 0 and abs(gx) < CORE_HALF_M:
                continue
            xs.append(gx + rng.uniform(-1500, 1500))
        out.append(xs)
    return out


def make_terrain(cfg: Settings) -> Terrain:
    """River valleys across rolling plains, no-go massifs and forest patches.

    A synthetic surface, not a measured DEM.
    """
    rng = np.random.default_rng(cfg.seed)
    layout_rng = np.random.default_rng(cfg.seed + 7)
    n = int(round(cfg.map_m/cfg.cell_m)) + 1
    axis = np.linspace(-cfg.map_m/2, cfg.map_m/2, n)
    cell = axis[1] - axis[0]
    x, y = np.meshgrid(axis, axis)
    d_center = np.min([np.abs(y - r(x)) for r in RIVERS], axis=0)
    z = 170 + 14*(np.minimum(d_center, 9000)/1000)**1.1
    for s_m, a in ((640, 55), (240, 18), (72, 3)):
        t = gaussian_filter(rng.normal(size=z.shape), s_m/cell)
        z += a*t/(t.std()+1e-9)
    for px, py, sx, sy, h in PEAKS + outer_massifs(cfg, layout_rng):
        z += h*np.exp(-((x-px)/sx)**2 - ((y-py)/sy)**2)

    water = d_center <= cfg.river_half_width_m
    z = np.where(water, np.minimum(z, 150), z)
    z = gaussian_filter(z, 40/cell)
    fords = np.zeros_like(water)
    for r, xs in zip(RIVERS, ford_positions(cfg, layout_rng)):
        near = np.abs(y - r(x)) <= cfg.river_half_width_m
        for fx in xs:
            fords |= (np.abs(x-fx) <= cfg.ford_half_width_m) & near & water
    water &= ~fords
    mountain = z >= cfg.mountain_z_m

    # Topographic position index: negative in gullies (dead ground), positive on crests
    tpi = z - gaussian_filter(z, 500/cell)
    d_river = distance_transform_edt(~(water | fords))*cell
    f = gaussian_filter(rng.normal(size=z.shape), 200/cell)
    f /= f.std()
    forest_score = f + 1.3*np.exp(-(d_river/450)**2) - .035*tpi
    forest = (forest_score > cfg.forest_threshold) & ~water & ~fords & ~mountain
    cover = gaussian_filter(forest.astype(float), 40/cell)
    # Wide-scale cover field so agents far from forest still sense where it is
    cover_far = gaussian_filter(forest.astype(float), 320/cell)
    cgy_c, cgx_c = np.gradient(cover_far, cell, cell)

    dzdy, dzdx = np.gradient(z, cell, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    forbidden = water | mountain | (slope_deg >= cfg.max_slope_deg)
    clearance = distance_transform_edt(~forbidden)*cell
    cgy, cgx = np.gradient(clearance, cell, cell)
    return Terrain(axis, z, water, fords, mountain, forest, cover, cgx_c, cgy_c, tpi,
                   forbidden, river_center(axis), slope_deg, clearance, cgx, cgy,
                   dzdx, dzdy, cell)


def create_guidelines(terr: Terrain, cfg: Settings, lkp: np.ndarray, stride_m: float = 30.0):
    """360 APF-inspired reference lines from the LKP (same scheme as the hiker demo).

    Lines are only generated as far as the fastest agent could plausibly walk.
    """
    reach_m = 1.3*(cfg.speed_mean_mps + 3*cfg.speed_std_mps)*max(cfg.fast_factor, 1.0)*cfg.hours*3600
    steps = int(min(cfg.map_m, reach_m) / stride_m)
    theta = np.deg2rad(np.arange(360.0))
    head = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    pos = np.tile(lkp, (360, 1)).astype(float)
    path = np.repeat(pos[:, None, :], steps + 1, axis=1)
    active = np.ones(360, bool)
    guide_len = np.full(360, steps)
    heading_vec = head.copy()
    for k in range(1, steps+1):
        d = np.nan_to_num(terr.sample(terr.clearance, pos[:, 0], pos[:, 1]), nan=0.0)
        repel_g = np.stack([np.nan_to_num(terr.sample(terr.clearance_gx, pos[:, 0], pos[:, 1])),
                            np.nan_to_num(terr.sample(terr.clearance_gy, pos[:, 0], pos[:, 1]))], axis=1)
        downhill = np.stack([-np.nan_to_num(terr.sample(terr.z_gx, pos[:, 0], pos[:, 1])),
                             -np.nan_to_num(terr.sample(terr.z_gy, pos[:, 0], pos[:, 1]))], axis=1)
        repel = normalize(repel_g) * (2.8 * np.clip((175-d)/175, 0, 1))[:, None]
        downhill = normalize(downhill) * .16
        force = head + repel + downhill
        heading_vec = normalize(.60*heading_vec + .40*normalize(force))
        cand = pos + stride_m*heading_vec
        ok = terr.inside(cand[:, 0], cand[:, 1])
        candidate_forbidden = np.ones(360, bool)
        candidate_forbidden[ok] = terr.sample(terr.forbidden, cand[ok, 0], cand[ok, 1]) > .40
        step_ok = ok & ~candidate_forbidden
        guide_len[active & ~step_ok] = k-1
        active &= step_ok
        pos[active] = cand[active]
        path[:, k, :] = pos
    return path, guide_len


def _unit_sum(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v/v.sum()


def move_split(cfg: Settings) -> np.ndarray:
    return _unit_sum(cfg.move_split)


def action_probs(cfg: Settings) -> np.ndarray:
    """Per-mode probabilities in MODES order from the top-level and moving-split settings."""
    top = _unit_sum([cfg.p_move, cfg.p_reorient, cfg.p_stop])
    split = move_split(cfg)
    p = np.zeros(len(MODES))
    p[MOVE_MODES] = top[0]*split
    p[RO], p[HD] = top[1], top[2]
    return p


def simulate(terr: Terrain, guide: np.ndarray, guide_len: np.ndarray, lkp: np.ndarray,
             cfg: Settings, probs: np.ndarray | None = None):
    probs = action_probs(cfg) if probs is None else probs
    split = move_split(cfg)
    rng = np.random.default_rng(cfg.seed + 101)
    n = cfg.particles
    n_steps = int(round(cfg.hours * 3600/cfg.dt_s))
    track = np.empty((n_steps+1, n, 2), float)
    track[0] = lkp
    pos = np.tile(lkp, (n, 1)).astype(float)
    rows = np.arange(n)
    # Known last heading with uncertainty; each particle commits to one intended heading
    heading = cfg.heading_deg + rng.normal(0, cfg.heading_std_deg, n)
    psi = np.deg2rad(heading)
    guide_index = np.rint(heading).astype(int) % 360
    gk = np.zeros(n, int)
    turn_dir = rng.choice([-1, 1], n)
    offsets = np.deg2rad(np.array([-100, -75, -50, -30, -15, 0, 15, 30, 50, 75, 100], float))
    action_counts = np.zeros(len(MODES), dtype=int)
    transfers = 0
    actions = rng.choice(len(MODES), size=n, p=probs)
    switch_chance = min(1.0, cfg.dt_s/(cfg.action_persistence_min*60))
    for t in range(n_steps):
        if t > 0:
            switch = rng.random(n) < switch_chance
            actions[switch] = rng.choice(len(MODES), size=switch.sum(), p=probs)
        # RO is instantaneous: shift the intended heading, then move in a sampled sub-mode
        ro = actions == RO
        action_counts[RO] += int(ro.sum())
        if ro.any():
            shift = np.rint(rng.normal(0, cfg.reorient_std_deg, ro.sum())).astype(int)
            guide_index[ro] = (guide_index[ro] + shift) % 360
            d = np.linalg.norm(guide[guide_index[ro]] - pos[ro, None, :], axis=2)
            gk[ro] = d.argmin(axis=1)
            actions[ro] = rng.choice(MOVE_MODES, size=ro.sum(), p=split)
        counts = np.bincount(actions, minlength=len(MODES))
        counts[RO] = 0
        action_counts += counts

        # Virtual target point on the current guideline; route transition where it ends
        end = guide_len[guide_index]
        for _ in range(20):
            tgt = guide[guide_index, np.minimum(gk+3, end)]
            adv = (np.linalg.norm(tgt-pos, axis=1) < 90) & (gk+3 < end)
            if not adv.any():
                break
            gk[adv] += 1
        tgt = guide[guide_index, np.minimum(gk+3, end)]
        ended = (gk+3 >= end) & (np.linalg.norm(tgt-pos, axis=1) < 150)
        if ended.any():
            transfers += int(ended.sum())
            guide_index[ended] = (guide_index[ended] + turn_dir[ended]) % 360
            d = np.linalg.norm(guide[guide_index[ended]] - pos[ended, None, :], axis=2)
            gk[ended] = d.argmin(axis=1)
        end = guide_len[guide_index]
        lookahead = guide[guide_index, np.minimum(gk+3, end)]
        to_guide = np.arctan2(lookahead[:, 1]-pos[:, 1], lookahead[:, 0]-pos[:, 0])
        desired = np.arctan2(.4*np.sin(psi) + .6*np.sin(to_guide),
                             .4*np.cos(psi) + .6*np.cos(to_guide))

        cover_here = terr.sample(terr.cover, pos[:, 0], pos[:, 1])
        hd = actions == HD
        hidden = hd & (cover_here >= .5)
        seek = hd & ~hidden
        to_cover = np.arctan2(np.nan_to_num(terr.sample(terr.cover_gy, pos[:, 0], pos[:, 1])),
                              np.nan_to_num(terr.sample(terr.cover_gx, pos[:, 0], pos[:, 1])))
        desired[seek] = to_cover[seek]

        fm = actions == FM
        sm = actions == SM
        # Paper Eq. (3): v_t ~ N(v_m, sigma^2), redrawn every step; v_m and sigma scaled per moving sub-mode
        factor = np.select([fm, sm], [cfg.fast_factor, cfg.slow_factor], 1.0)
        speed = np.clip(rng.normal(cfg.speed_mean_mps*factor, cfg.speed_std_mps*factor), 0, None)
        speed[hidden] = 0

        angles = psi[:, None] + offsets[None, :]
        dist = speed[:, None]*cfg.dt_s
        nx = pos[:, 0, None]+dist*np.cos(angles)
        ny = pos[:, 1, None]+dist*np.sin(angles)
        inside = terr.inside(nx, ny)
        ec = terr.sample(terr.z, nx, ny)
        e0 = terr.sample(terr.z, pos[:, 0], pos[:, 1])[:, None]
        slope_rad = np.arctan2(ec-e0, np.maximum(dist, 1e-4))
        slope_abs = np.abs(np.degrees(slope_rad))
        cov_c = np.nan_to_num(terr.sample(terr.cover, nx, ny))
        crest = np.clip(np.nan_to_num(terr.sample(terr.tpi, nx, ny))/25, 0, 2)
        # Concealment: covert movers weigh open ground and crests heavily, fast movers barely, stalkers most
        w_open = np.select([fm, seek, sm], [.4, 3.0, 3.0], 2.2)[:, None]
        w_crest = np.select([fm, seek, sm], [.3, .8, 1.8], 1.2)[:, None]
        conceal_cost = w_open*(1-cov_c) + w_crest*crest
        slope_cost = 1.2*np.clip(slope_abs/cfg.max_slope_deg, 0, 2)**2
        w_head = np.select([fm, seek, sm], [5.0, 4.0, 3.5], 4.0)[:, None]
        heading_reward = w_head*np.cos(angles-desired[:, None])
        turn_cost = .6*(offsets[None, :]/np.pi)**2
        scores = heading_reward - turn_cost - conceal_cost - slope_cost
        forbidden = np.nan_to_num(terr.sample(terr.forbidden, nx, ny), nan=1.0) > .40
        invalid = (~inside) | forbidden | (slope_abs > cfg.max_slope_deg) | (~np.isfinite(scores))
        scores[invalid] = -1e7
        no_feasible = np.all(invalid, axis=1)
        scores[no_feasible, 5] = 0.0
        choices = sample_categorical_rows(rng, scores)
        chosen_theta = angles[rows, choices]
        angle_deg = np.rad2deg(slope_rad[rows, choices])
        downhill_limit = cfg.max_slope_deg + 4
        q = np.where(angle_deg >= 0, 1-angle_deg/cfg.max_slope_deg,
                     1-angle_deg/(-downhill_limit))
        q = np.clip(q, 0, 1)
        forest_factor = 1 - .35*cov_c[rows, choices]
        move = speed*cfg.dt_s*q*forest_factor
        move[no_feasible] = 0
        new_pos = pos + move[:, None]*np.stack([np.cos(chosen_theta), np.sin(chosen_theta)], axis=1)
        still_ok = terr.inside(new_pos[:, 0], new_pos[:, 1]) & \
            (terr.sample(terr.forbidden, new_pos[:, 0], new_pos[:, 1]) < .4)
        new_pos[~still_ok] = pos[~still_ok]
        move[~still_ok] = 0
        psi[move > 0] = chosen_theta[move > 0]
        pos = new_pos
        track[t+1] = pos
    return track, action_counts, transfers


def zoom_window(terr: Terrain, track: np.ndarray, margin_m: float = 2000.0):
    """Square window (x0, x1, y0, y1) in metres around every particle position, clipped to the map."""
    lo = track.reshape(-1, 2).min(axis=0) - margin_m
    hi = track.reshape(-1, 2).max(axis=0) + margin_m
    centre, half = (lo + hi)/2, (hi - lo).max()/2
    a0, a1 = terr.axis[0], terr.axis[-1]
    half = min(half, (a1 - a0)/2)
    centre = np.clip(centre, a0 + half, a1 - half)
    return (centre[0]-half, centre[0]+half, centre[1]-half, centre[1]+half)


def draw_background(ax, terr: Terrain, cfg: Settings, alpha=.8, legend_extra=(), window=None):
    """Draw terrain layers; only the grid inside ``window`` (metres) is rendered."""
    if window is None:
        window = (terr.axis[0], terr.axis[-1], terr.axis[0], terr.axis[-1])
    x0, x1, y0, y1 = window
    ci = np.flatnonzero((terr.axis >= x0 - 2*terr.cell) & (terr.axis <= x1 + 2*terr.cell))
    ri = np.flatnonzero((terr.axis >= y0 - 2*terr.cell) & (terr.axis <= y1 + 2*terr.cell))
    rs, cs = slice(ri[0], ri[-1]+1), slice(ci[0], ci[-1]+1)
    ax_, ay_ = terr.axis[cs]/1000, terr.axis[rs]/1000
    extent = (ax_[0], ax_[-1], ay_[0], ay_[-1])
    crop = lambda f: f[rs, cs]
    smooth = .8*40/terr.cell
    ax.imshow(crop(terr.z), origin="lower", extent=extent, cmap="terrain", alpha=alpha, vmin=60, vmax=950)
    ax.contour(ax_, ay_, crop(terr.z), levels=[200, 250, 300, 350, 400], linewidths=.45,
               colors="black", alpha=.18)
    ax.contourf(ax_, ay_, crop(terr.cover), levels=[.5, 2], colors=["#1b5e20"], alpha=.45)
    ax.contourf(ax_, ay_, crop(terr.mountain).astype(float), levels=[.5, 1.5], colors="none", hatches=["///"])
    ax.contour(ax_, ay_, crop(terr.mountain).astype(float), levels=[.5], colors="#5b3a1a", linewidths=1.3)
    ax.contourf(ax_, ay_, gaussian_filter(crop(terr.water).astype(float), smooth), levels=[.5, 2],
                colors=["#1f78c8"], alpha=.95)
    ax.contour(ax_, ay_, gaussian_filter(crop(terr.fords).astype(float), smooth), levels=[.4],
               colors="#ff8f00", linewidths=1.6)
    handles = [Patch(facecolor="none", hatch="///", edgecolor="#5b3a1a",
                     label=f"Mountain (≥{cfg.mountain_z_m:.0f} m, no-go)"),
               Patch(color="#1f78c8", label="River"),
               Patch(facecolor="none", edgecolor="#ff8f00", lw=1.6, label="Ford"),
               Patch(color="#1b5e20", alpha=.45, label="Forest (cover)"),
               *legend_extra]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=.9)
    ax.set(xlabel="x (km)", ylabel="y (km)", xlim=(x0/1000, x1/1000), ylim=(y0/1000, y1/1000))
    ax.set_aspect("equal")
    return extent


def draw_lkp(ax, lkp, cfg: Settings):
    lx, ly = lkp/1000
    ax.scatter(lx, ly, color="red", marker="*", s=130, edgecolor="white", linewidth=.8, zorder=9)
    for dh, ls, lw in ((0, "-", 2.0), (-cfg.heading_std_deg, "--", 1.0), (cfg.heading_std_deg, "--", 1.0)):
        h = np.deg2rad(cfg.heading_deg + dh)
        ax.annotate("", xy=(lx+1.6*np.cos(h), ly+1.6*np.sin(h)), xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="-|>" if dh == 0 else "-", color="red", lw=lw, ls=ls),
                    zorder=9)


def save_outputs(out: Path, cfg: Settings, terr: Terrain, guides, guide_len, lkp, track,
                 action_counts, transfers):
    out.mkdir(parents=True, exist_ok=True)
    lkp_handle = Line2D([], [], marker="*", color="red", markeredgecolor="white", ls="",
                        markersize=12, label="LKP + known heading (±1σ)")

    win = zoom_window(terr, track)
    full = (terr.axis[0], terr.axis[-1], terr.axis[0], terr.axis[-1])

    fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
    draw_background(ax, terr, cfg, window=full, legend_extra=(
        Patch(facecolor="none", edgecolor="red", lw=1.4, label="Particle extent (zoom window)"),
        lkp_handle))
    ax.add_patch(plt.Rectangle((win[0]/1000, win[2]/1000), (win[1]-win[0])/1000, (win[3]-win[2])/1000,
                               fill=False, ec="red", lw=1.4, zorder=8))
    ax.scatter(*(lkp/1000), color="red", marker="*", s=90, edgecolor="white", linewidth=.6, zorder=9)
    ax.set_title(f"Search area {cfg.map_m/1000:.0f} km × {cfg.map_m/1000:.0f} km (battlefield terrain)")
    fig.savefig(out/"00_overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
    draw_background(ax, terr, cfg, window=win, legend_extra=(
        Line2D([], [], color="#343f60", lw=.8, label="Guidelines (every 5°)"), lkp_handle))
    for idx in range(0, 360, 5):
        g = guides[idx, :guide_len[idx]+1]
        ax.plot(g[:, 0]/1000, g[:, 1]/1000, color="#343f60", alpha=.4, lw=.6)
    draw_lkp(ax, lkp, cfg)
    ax.set_title("Battlefield terrain near the LKP and APF-inspired guidelines")
    fig.savefig(out/"01_terrain_guidelines.png", dpi=160)
    plt.close(fig)

    nsteps = track.shape[0]-1
    for fraction in (.2, .6, 1.0):
        idx = int(round(nsteps*fraction))
        minutes = idx*cfg.dt_s/60
        label = f"{int(round(minutes)):03d}min"
        pts = track[idx]
        fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
        draw_background(ax, terr, cfg, alpha=.7, window=win, legend_extra=(
            Line2D([], [], marker="o", color="#0d2c8c", ls="", markersize=4,
                   label=f"Particles ({cfg.particles})"), lkp_handle))
        for p in range(0, cfg.particles, 48):
            ax.plot(track[:idx+1, p, 0]/1000, track[:idx+1, p, 1]/1000, lw=.8, alpha=.35, color="#0d2c8c")
        ax.scatter(pts[:, 0]/1000, pts[:, 1]/1000, s=6, alpha=.7, color="#0d2c8c", edgecolors="none")
        draw_lkp(ax, lkp, cfg)
        ax.set_title(f"Infiltrator Monte Carlo trajectories: t = {minutes:.0f} min")
        fig.savefig(out/f"02_particles_{label}.png", dpi=160)
        plt.close(fig)

        density = kde_map(terr, pts, cfg.bandwidth_m)
        fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
        draw_background(ax, terr, cfg, alpha=.5, window=win, legend_extra=(lkp_handle,))
        strength = density / (float(density.max()) + 1e-15)
        heat_alpha = np.clip(1.65*np.sqrt(strength), 0, .86)
        pc = ax.imshow(density*1e6, origin="lower",
                       extent=(terr.axis[0]/1000, terr.axis[-1]/1000, terr.axis[0]/1000, terr.axis[-1]/1000),
                       cmap="magma", alpha=heat_alpha, interpolation="bilinear")
        cb = fig.colorbar(pc, ax=ax, shrink=.78, pad=.02)
        cb.set_label("Probability density (per km²)")
        draw_lkp(ax, lkp, cfg)
        ax.set_title(f"Infiltrator KDE: t = {minutes:.0f} min | h = {cfg.bandwidth_m:g} m")
        fig.savefig(out/f"03_kde_{label}.png", dpi=160)
        plt.close(fig)
        np.save(out/f"kde_{label}.npy", density)

    np.savez_compressed(out/"simulation_data.npz", terrain=terr.z, axis=terr.axis, track=track,
                        guidelines=guides, guide_len=guide_len, lkp=lkp, forbidden=terr.forbidden,
                        mountain=terr.mountain, water=terr.water, fords=terr.fords,
                        forest=terr.forest, river_y=terr.river_y, dt_s=cfg.dt_s,
                        heading_deg=cfg.heading_deg, map_m=cfg.map_m,
                        speed_mean_mps=cfg.speed_mean_mps, speed_std_mps=cfg.speed_std_mps,
                        modes=np.array(MODES), action_probs=action_probs(cfg),
                        fast_factor=cfg.fast_factor, slow_factor=cfg.slow_factor)

    flat_x, flat_y = track[..., 0].ravel(), track[..., 1].ravel()
    in_mountain = terr.sample(terr.mountain, flat_x, flat_y) > .5
    in_water = terr.sample(terr.water, flat_x, flat_y) > .5
    final = track[-1]
    final_cover = terr.sample(terr.cover, final[:, 0], final[:, 1]) >= .5
    free = ~terr.forbidden
    crossed = (final[:, 1] - river_center(final[:, 0])) * (lkp[1] - river_center(lkp[0])) < 0
    disp = final - lkp
    h0 = np.deg2rad(cfg.heading_deg)
    along = disp @ np.array([np.cos(h0), np.sin(h0)])
    print(f"Done: {out.resolve()}")
    print(f"Particle trajectories: {track.shape} | Mode probabilities: "
          f"{ {m: round(float(p), 3) for m, p in zip(MODES, action_probs(cfg))} }")
    print(f"Particle-steps per mode (RO = reorient events): {dict(zip(MODES, action_counts.tolist()))}")
    print(f"Route transfers: {transfers}")
    print(f"Elevation range: {terr.z.min():.0f}–{terr.z.max():.0f} m; mountain: {terr.mountain.mean():.1%}; "
          f"forest share of passable land: {terr.forest[free].mean():.1%}")
    print(f"Particle-steps inside mountain: {in_mountain.mean():.3%} | inside river: {in_water.mean():.3%}")
    print(f"Final particles in forest cover: {final_cover.mean():.1%} | crossed the river: {crossed.mean():.1%}")
    print(f"Final distance from LKP: median {np.median(np.linalg.norm(disp, axis=1))/1000:.2f} km | "
          f"along known heading: median {np.median(along)/1000:.2f} km")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results_battlefield_paper"))
    ap.add_argument("--particles", type=int, default=1440)
    ap.add_argument("--hours", type=float, default=2.5,
                    help="Simulated time; 2.5 h = 30 min launch delay + 2 h UAV search (paper)")
    ap.add_argument("--map-km", type=float, default=65.0, help="Square map side length (paper: 65 km)")
    ap.add_argument("--cell", type=float, default=50.0, help="Grid cell size in metres")
    ap.add_argument("--speed-mean", type=float, default=.75, help="Mean walking speed m/s (paper: 0.75)")
    ap.add_argument("--speed-std", type=float, default=.25, help="Walking speed std m/s (paper: 0.25)")
    d = Settings()
    ap.add_argument("--behavior", type=float, nargs=3, default=(d.p_move, d.p_reorient, d.p_stop),
                    metavar=("MOVE", "REORIENT", "STOP"), help="Top-level behaviour probabilities")
    ap.add_argument("--move-split", type=float, nargs=3, default=d.move_split,
                    metavar=("NORMAL", "FAST", "SLOW"), help="Share of normal / fast / slow moving")
    ap.add_argument("--fast-factor", type=float, default=d.fast_factor,
                    help="Speed multiplier in the fast-move mode")
    ap.add_argument("--slow-factor", type=float, default=d.slow_factor,
                    help="Speed multiplier in the slow-move mode")
    ap.add_argument("--bandwidth", type=float, default=150.0)
    ap.add_argument("--lkp", type=float, nargs=2, default=(-4500.0, -3000.0), metavar=("X", "Y"),
                    help="Last known position in metres")
    ap.add_argument("--heading", type=float, default=40.0,
                    help="Last known heading in degrees (0 = east, counter-clockwise)")
    ap.add_argument("--heading-std", type=float, default=25.0,
                    help="Uncertainty of the known heading (degrees, 1σ)")
    ap.add_argument("--mountain-z", type=float, default=460.0,
                    help="Absolute elevation (m) at and above which terrain is a no-go mountain")
    ap.add_argument("--action-persistence", type=float, default=10.0,
                    help="Expected behavior-mode duration in minutes")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.particles < 12 or args.hours <= 0 or args.bandwidth <= 0 or args.heading_std < 0 \
            or args.action_persistence <= 0 or args.map_km < 10 or args.cell <= 0 \
            or args.speed_mean <= 0 or args.speed_std < 0 or args.fast_factor <= 0 or args.slow_factor <= 0 \
            or min(args.behavior) < 0 or sum(args.behavior) <= 0 or min(args.move_split) < 0 \
            or sum(args.move_split) <= 0:
        ap.error("Need particles>=12, hours>0, bandwidth>0, heading-std>=0, action-persistence>0, "
                 "map-km>=10, cell>0, speed-mean>0, speed-std>=0, fast/slow-factor>0, "
                 "non-negative behavior and move-split with positive sums")
    cfg = Settings(seed=args.seed, particles=args.particles, hours=args.hours,
                   bandwidth_m=args.bandwidth, mountain_z_m=args.mountain_z,
                   map_m=args.map_km*1000, cell_m=args.cell, speed_mean_mps=args.speed_mean,
                   speed_std_mps=args.speed_std, fast_factor=args.fast_factor, slow_factor=args.slow_factor,
                   p_move=args.behavior[0], p_reorient=args.behavior[1], p_stop=args.behavior[2],
                   move_split=tuple(args.move_split),
                   lkp_x=args.lkp[0], lkp_y=args.lkp[1], heading_deg=args.heading,
                   heading_std_deg=args.heading_std, action_persistence_min=args.action_persistence)
    terr = make_terrain(cfg)
    lkp = np.array([cfg.lkp_x, cfg.lkp_y])
    if terr.sample(terr.forbidden, *lkp) > .4:
        ap.error("LKP lies inside a river, mountain or impassable slope")
    guides, guide_len = create_guidelines(terr, cfg, lkp)
    track, actions, transfers = simulate(terr, guides, guide_len, lkp, cfg)
    save_outputs(args.output, cfg, terr, guides, guide_len, lkp, track, actions, transfers)


if __name__ == "__main__":
    main()
