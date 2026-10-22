"""Ge-inspired terrain-aware Monte Carlo + KDE demo (NOT the authors' code).

Generates a synthetic valley map with a wide river, several mountain massifs
and a trail network (valley trails, trails around each mountain, connectors and
bridges). Mountains above an absolute elevation are hard no-go areas for the
target; ground-moving agents follow the paper's behavior profile and route
transition, and an evolving spatial probability density is estimated via KDE.

Run: python ge_probability_demo.py --output results
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

MODES = ("RM", "DT", "RT", "SP", "VE", "BT")
# Ge et al. 2026 experienced-hiker profile (RM, DT, RT, SP, VE, BT) with VE
# removed: the target never climbs mountains in this model.
GE_ACTION_PROBS = np.array([.055, .559, .377, .003, 0.0, 0.0])
GE_ACTION_PROBS = GE_ACTION_PROBS / GE_ACTION_PROBS.sum()

PEAKS = [  # (x, y, sigma_x, sigma_y, height) in metres
    (-3800,  2900, 1700, 1300, 620),
    ( 1200,  3400, 2300, 1100, 700),
    ( 4300,  1500, 1300, 1500, 560),
    (-2600, -3900, 1900, 1200, 640),
    ( 2900, -3600, 1500, 1300, 600),
]
BRIDGES_X = (-1500.0, 3200.0)
VALLEY_TRAIL_OFFSET_M = 170.0


@dataclass
class Settings:
    seed: int = 26
    particles: int = 1440
    hours: float = 2.0
    dt_s: float = 30.0
    bandwidth_m: float = 150.0
    beta_terrain: float = 1.5
    map_m: float = 14000.0
    grid_size: int = 351
    mean_speed_mps: float = .75
    speed_std_mps: float = .25
    max_slope_deg: float = 32.0
    action_persistence_min: float = 8.0
    mountain_z_m: float = 460.0
    river_half_width_m: float = 90.0
    trail_offset_m: float = 220.0


@dataclass
class Terrain:
    axis: np.ndarray
    z: np.ndarray
    water: np.ndarray
    mountain: np.ndarray
    forbidden: np.ndarray
    trail: np.ndarray
    trail_dist: np.ndarray
    trail_gx: np.ndarray
    trail_gy: np.ndarray
    trail_tx: np.ndarray
    trail_ty: np.ndarray
    river_y: np.ndarray
    slope_deg: np.ndarray
    clearance: np.ndarray
    clearance_gx: np.ndarray
    clearance_gy: np.ndarray
    z_gx: np.ndarray
    z_gy: np.ndarray
    cell: float

    def sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear map lookup; scalar or vector locations, metric coordinates."""
        xx, yy = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        rows = (yy - self.axis[0]) / self.cell
        cols = (xx - self.axis[0]) / self.cell
        vals = map_coordinates(field.astype(float), [rows.ravel(), cols.ravel()], order=1,
                               mode="constant", cval=np.nan)
        return vals.reshape(xx.shape)

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lo, hi = self.axis[0], self.axis[-1]
        return (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)


def river_center(xx):
    return 350*np.sin((xx+600)/1500) + 250*np.sin((xx-900)/3100) - 500


def paint_line(mask, axis, cell, p0, p1, stop=None):
    """Rasterize the segment p0->p1 into mask, halting where stop(row, col) is True."""
    n = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]) / (cell*.5)) + 2
    for s in np.linspace(0, 1, n):
        px, py = p0[0] + s*(p1[0]-p0[0]), p0[1] + s*(p1[1]-p0[1])
        r = int(round((py-axis[0])/cell))
        c = int(round((px-axis[0])/cell))
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]):
            return
        if stop is not None and stop(r, c):
            return
        mask[r, c] = True


def paint_curve(mask, axis, cell, ys):
    """Rasterize y = ys[col] (one value per grid column), filling vertical gaps."""
    rows = np.rint((ys-axis[0])/cell).astype(int)
    for c in range(len(axis)):
        r0 = rows[c]
        r1 = rows[c+1] if c+1 < len(axis) else r0
        lo, hi = sorted((r0, r1))
        lo, hi = max(lo, 0), min(hi, mask.shape[0]-1)
        if lo <= hi:
            mask[lo:hi+1, c] = True


def make_terrain(cfg: Settings) -> Terrain:
    """Wide river valley, five mountain massifs and a connected trail network.

    A physically-plausible *synthetic* surface, not a measured DEM or the
    paper's MATLAB terrain-generator implementation.
    """
    rng = np.random.default_rng(cfg.seed)
    axis = np.linspace(-cfg.map_m/2, cfg.map_m/2, cfg.grid_size)
    cell = axis[1] - axis[0]
    x, y = np.meshgrid(axis, axis)
    ry = river_center(x)
    z = 160 + 22*np.abs((y-ry)/1000)**1.2
    for px, py, sx, sy, h in PEAKS:
        z += h*np.exp(-((x-px)/sx)**2 - ((y-py)/sy)**2)
    for s, a in ((14, 22), (5, 9), (1.6, 2.5)):
        t = gaussian_filter(rng.normal(size=z.shape), s)
        z += a*t/(t.std()+1e-9)

    water = np.abs(y-ry) <= cfg.river_half_width_m
    z = np.where(water, np.minimum(z, 150), z)
    z = gaussian_filter(z, 1.0)
    bridge = np.zeros_like(water)
    for bx in BRIDGES_X:
        bridge |= (np.abs(x-bx) <= 1.5*cell) & water
    water &= ~bridge

    mountain = z >= cfg.mountain_z_m
    d_m = distance_transform_edt(~mountain)*cell

    trail = np.abs(gaussian_filter(d_m, 1.5)-cfg.trail_offset_m) < .6*cell
    north_y = river_center(axis) + cfg.river_half_width_m + VALLEY_TRAIL_OFFSET_M
    south_y = river_center(axis) - cfg.river_half_width_m - VALLEY_TRAIL_OFFSET_M
    paint_curve(trail, axis, cell, north_y)
    paint_curve(trail, axis, cell, south_y)
    for bx in BRIDGES_X:
        paint_line(trail, axis, cell, (bx, float(np.interp(bx, axis, south_y))),
                   (bx, float(np.interp(bx, axis, north_y))))
    for px, py, *_ in PEAKS:
        vx = float(np.clip(px, axis[0], axis[-1]))
        vy = float(np.interp(vx, axis, north_y if py > 0 else south_y))
        paint_line(trail, axis, cell, (vx, vy), (px, py),
                   stop=lambda r, c: d_m[r, c] < cfg.trail_offset_m)
    trail &= ~mountain & ~water

    trail_dist = distance_transform_edt(~trail)*cell
    tgy, tgx = np.gradient(trail_dist, cell, cell)
    # Trail tangent from the structure tensor of the smoothed trail mask;
    # well defined on the trail centreline where the plain gradient vanishes.
    sm = gaussian_filter(trail.astype(float), 1.5)
    sgy, sgx = np.gradient(sm, cell, cell)
    jxx = gaussian_filter(sgx*sgx, 2.5)
    jxy = gaussian_filter(sgx*sgy, 2.5)
    jyy = gaussian_filter(sgy*sgy, 2.5)
    normal = .5*np.arctan2(2*jxy, jxx-jyy)
    tx, ty = -np.sin(normal), np.cos(normal)

    dzdy, dzdx = np.gradient(z, cell, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    forbidden = water | mountain | ((slope_deg >= cfg.max_slope_deg) & ~trail)
    clearance = distance_transform_edt(~forbidden)*cell
    cgy, cgx = np.gradient(clearance, cell, cell)
    return Terrain(axis, z, water, mountain, forbidden, trail, trail_dist, tgx, tgy,
                   tx, ty, river_center(axis), slope_deg, clearance, cgx, cgy,
                   dzdx, dzdy, cell)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def create_guidelines(terr: Terrain, cfg: Settings, stride_m: float = 30.0):
    """Generate 360 directional APF-*inspired* reference lines from the LKP.

    Attraction toward distant heading goals; obstacle (river, mountain, steep
    slope) repulsion uses clearance gradients. Returns the index of the last
    valid point of each line so agents can transfer routes where it ends.
    """
    steps = int(cfg.map_m / stride_m)
    theta = np.deg2rad(np.arange(360.0))
    head = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    lkp_x = -1500.0
    lkp_y = float(river_center(lkp_x) + cfg.river_half_width_m + 220.0)
    pos = np.tile(np.array([lkp_x, lkp_y]), (360, 1))
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
    return path, guide_len, np.array([lkp_x, lkp_y])


def sample_categorical_rows(rng, scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    prob = np.exp(scores)
    prob /= prob.sum(axis=1, keepdims=True)
    u = rng.random((len(prob), 1))
    return (np.cumsum(prob, axis=1) < u).sum(axis=1)


def simulate(terr: Terrain, guide: np.ndarray, guide_len: np.ndarray, lkp: np.ndarray,
             cfg: Settings):
    rng = np.random.default_rng(cfg.seed + 101)
    n = cfg.particles
    n_steps = int(round(cfg.hours * 3600/cfg.dt_s))
    track = np.empty((n_steps+1, n, 2), float)
    track[0, :, :] = lkp
    pos = np.tile(lkp, (n, 1)).astype(float)
    rows = np.arange(n)
    psi_initial = (np.arange(n) + rng.uniform(-.35, .35, n))*2*np.pi/n
    psi = psi_initial.copy()
    guide_index = np.rint(np.rad2deg(psi_initial)).astype(int) % 360
    gk = np.zeros(n, int)
    # Route transition (Sec. II-C3): left/right is drawn once and then kept.
    turn_dir = rng.choice([-1, 1], n)
    offsets = np.deg2rad(np.array([-100, -75, -50, -30, -15, 0, 15, 30, 50, 75, 100], float))
    z0, z9 = np.percentile(terr.z, [10, 90])
    action_counts = np.zeros(6, dtype=int)
    avoid_count = 0
    transfers = 0
    # Practical extension: human behavior lasts for several minutes. The paper
    # allows Selectaction at each step; it does not specify dwell-time dynamics.
    actions = rng.choice(len(MODES), size=n, p=GE_ACTION_PROBS)
    switch_chance = min(1.0, cfg.dt_s/(cfg.action_persistence_min*60))
    for t in range(n_steps):
        if t > 0:
            switch = rng.random(n) < switch_chance
            actions[switch] = rng.choice(len(MODES), size=switch.sum(), p=GE_ACTION_PROBS)
        action_counts += np.bincount(actions, minlength=6)
        speed = np.clip(rng.normal(cfg.mean_speed_mps, cfg.speed_std_mps, n), 0, 1.65)
        speed[actions == 3] = 0

        # Virtual target point on the current guideline, advanced as the agent nears it
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
        desired_guide = np.arctan2(lookahead[:, 1]-pos[:, 1], lookahead[:, 0]-pos[:, 0])
        guide_near = np.linalg.norm(lookahead-pos, axis=1) < 480

        # Default DT: continuity of travel direction and attraction to guideline
        blend = .40 * guide_near.astype(float)
        cpx = (1-blend)*np.cos(psi)+blend*np.cos(desired_guide)
        cpy = (1-blend)*np.sin(psi)+blend*np.sin(desired_guide)
        desired = np.arctan2(cpy, cpx)
        # RT: approach the nearest trail; on it, walk along the tangent closest to psi
        td = terr.sample(terr.trail_dist, pos[:, 0], pos[:, 1])
        ax_ = -np.nan_to_num(terr.sample(terr.trail_gx, pos[:, 0], pos[:, 1]))
        ay_ = -np.nan_to_num(terr.sample(terr.trail_gy, pos[:, 0], pos[:, 1]))
        tx_ = np.nan_to_num(terr.sample(terr.trail_tx, pos[:, 0], pos[:, 1]))
        ty_ = np.nan_to_num(terr.sample(terr.trail_ty, pos[:, 0], pos[:, 1]))
        flip = (tx_*np.cos(psi) + ty_*np.sin(psi)) < 0
        tx_[flip], ty_[flip] = -tx_[flip], -ty_[flip]
        on_trail = td < 1.5*terr.cell
        route_dir = np.where(on_trail, np.arctan2(ty_, tx_), np.arctan2(ay_, ax_))
        trail_near = td < 600.0
        rt = (actions == 2) & trail_near
        desired[rt] = route_dir[rt]
        # RM: persistent movement with broadly stochastic angular perturbation
        rand_head = psi + rng.normal(0, np.deg2rad(52), n)
        desired[actions == 0] = rand_head[actions == 0]
        # BT: head toward a point visited 10 movement steps earlier; cited profile gives BT=0
        history_pos = track[max(0, t-10), :, :]
        back = np.arctan2(history_pos[:, 1]-pos[:, 1], history_pos[:, 0]-pos[:, 0])
        desired[actions == 5] = back[actions == 5]

        angles = psi[:, None] + offsets[None, :]
        dist = speed[:, None]*cfg.dt_s
        nx = pos[:, 0, None]+dist*np.cos(angles)
        ny = pos[:, 1, None]+dist*np.sin(angles)
        inside = terr.inside(nx, ny)
        ec = terr.sample(terr.z, nx, ny)
        e0 = terr.sample(terr.z, pos[:, 0], pos[:, 1])[:, None]
        slope_rad = np.arctan2(ec-e0, np.maximum(dist, 1e-4))
        slope_abs = np.abs(np.degrees(slope_rad))
        altcost = np.clip((ec-z0)/(z9-z0), 0, 1.6)
        cand_trail_dist = terr.sample(terr.trail_dist, nx, ny)
        terrain_cost = .90*altcost + 2.5*np.clip(slope_abs/cfg.max_slope_deg, 0, 2)**2
        terrain_cost += .10*np.clip(cand_trail_dist/650, 0, 3)
        desired_weight = np.where(actions == 0, 2.5, np.where(actions == 2, 5.3, 4.6))
        heading_reward = desired_weight[:, None]*np.cos(angles-desired[:, None])
        turn_cost = .6*(offsets[None, :]/np.pi)**2
        scores = heading_reward - turn_cost - cfg.beta_terrain*terrain_cost
        scores -= rt[:, None]*.55*np.clip(cand_trail_dist/500, 0, 4)
        forbidden = np.nan_to_num(terr.sample(terr.forbidden, nx, ny), nan=1.0) > .40
        invalid = (~inside) | forbidden | (slope_abs > cfg.max_slope_deg) | (~np.isfinite(scores))
        scores[invalid] = -1e7
        no_feasible = np.all(invalid, axis=1)
        avoid_count += int(no_feasible.sum())
        scores[no_feasible, 5] = 0.0
        choices = sample_categorical_rows(rng, scores)
        chosen_theta = angles[rows, choices]
        chosen_slope = slope_rad[rows, choices]
        angle_deg = np.rad2deg(chosen_slope)
        downhill_limit = cfg.max_slope_deg + 4
        q = np.where(angle_deg >= 0, 1-angle_deg/cfg.max_slope_deg,
                     1-angle_deg/(-downhill_limit))
        q = np.clip(q, 0, 1)
        move = speed*cfg.dt_s*q
        move[no_feasible] = 0
        new_pos = pos + move[:, None]*np.stack([np.cos(chosen_theta), np.sin(chosen_theta)], axis=1)
        still_ok = terr.inside(new_pos[:, 0], new_pos[:, 1]) & \
            (terr.sample(terr.forbidden, new_pos[:, 0], new_pos[:, 1]) < .4)
        new_pos[~still_ok] = pos[~still_ok]
        move[~still_ok] = 0
        psi[move > 0] = chosen_theta[move > 0]
        pos = new_pos
        track[t+1] = pos
    return track, action_counts, avoid_count, transfers


def kde_analytic(points: np.ndarray, queries: np.ndarray, h: float, chunks: int = 2000):
    """Exact continuous isotropic Gaussian KDE, evaluated at arbitrary (x,y)."""
    results = []
    for i in range(0, len(queries), chunks):
        q = queries[i:i+chunks]
        d2 = ((q[:, None, :]-points[None, :, :])**2).sum(axis=2)
        vals = np.exp(-d2/(2*h*h)).mean(axis=1)/(2*np.pi*h*h)
        results.append(vals)
    return np.concatenate(results)


def kde_map(terr: Terrain, points: np.ndarray, bandwidth_m: float):
    """Fast grid approximation to Gaussian KDE via smoothing binned samples."""
    bins = np.linspace(terr.axis[0]-terr.cell/2, terr.axis[-1]+terr.cell/2, len(terr.axis)+1)
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(bins, bins))
    density = gaussian_filter(counts, sigma=bandwidth_m/terr.cell, mode="constant")
    density /= len(points)*terr.cell**2
    # No probability mass inside river, mountain or impassable steep cells.
    density[terr.forbidden] = 0.0
    integral = density.sum()*terr.cell**2
    if integral > 0:
        density /= integral
    return density


def draw_background(ax, terr: Terrain, cfg: Settings, alpha=.83, legend_extra=()):
    a = terr.axis/1000
    extent = (a[0], a[-1], a[0], a[-1])
    ax.imshow(terr.z, origin="lower", extent=extent, cmap="terrain", alpha=alpha, vmin=80, vmax=1000)
    ax.contour(a, a, terr.z, levels=[250, 350, 600, 750, 900],
               linewidths=.5, colors="black", alpha=.2)
    ax.contourf(a, a, terr.mountain.astype(float), levels=[.5, 1.5],
                colors="none", hatches=["///"])
    ax.contour(a, a, terr.mountain.astype(float), levels=[.5], colors="#5b3a1a", linewidths=1.3)
    ax.contourf(a, a, gaussian_filter(terr.water.astype(float), .8), levels=[.5, 2],
                colors=["#1f78c8"], alpha=.95)
    ax.imshow(np.ma.masked_where(~terr.trail, terr.trail.astype(float)), origin="lower",
              extent=extent, cmap=ListedColormap(["#e8b923"]), interpolation="nearest")
    handles = [Patch(facecolor="none", hatch="///", edgecolor="#5b3a1a",
                     label=f"Mountain (≥{cfg.mountain_z_m:.0f} m, no-go)"),
               Patch(color="#1f78c8", label="River"),
               Line2D([], [], color="#e8b923", lw=1.6, label="Trails"),
               *legend_extra]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=.9)
    ax.set(xlabel="x (km)", ylabel="y (km)", xlim=extent[:2], ylim=extent[2:])
    ax.set_aspect("equal")
    return extent


def save_outputs(out: Path, cfg: Settings, terr: Terrain, guides, guide_len, lkp, track,
                 action_counts, failed, transfers):
    out.mkdir(parents=True, exist_ok=True)
    lkp_handle = Line2D([], [], marker="*", color="red", markeredgecolor="white", ls="",
                        markersize=12, label="LKP")
    fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
    draw_background(ax, terr, cfg, legend_extra=(
        Line2D([], [], color="#343f60", lw=.8, label="Guidelines (every 10°)"), lkp_handle))
    for idx in range(0, 360, 10):
        g = guides[idx, :guide_len[idx]+1]
        ax.plot(g[:, 0]/1000, g[:, 1]/1000, color="#343f60", alpha=.45, lw=.65)
    ax.scatter(*(lkp/1000), color="red", s=110, marker="*", edgecolor="white", linewidth=.8, zorder=9)
    ax.set_title("Synthetic terrain, trail network and APF-inspired guidelines")
    fig.savefig(out/"01_terrain_guidelines.png", dpi=160)
    plt.close(fig)

    nsteps = track.shape[0]-1
    for fraction in (.25, .5, 1.0):
        idx = int(round(nsteps*fraction))
        minutes = idx*cfg.dt_s/60
        label = f"{int(round(minutes)):03d}min"
        pts = track[idx]
        fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
        draw_background(ax, terr, cfg, alpha=.7, legend_extra=(
            Line2D([], [], marker="o", color="#164db8", ls="", markersize=4,
                   label=f"Particles ({cfg.particles})"), lkp_handle))
        for p in range(0, cfg.particles, 48):
            ax.plot(track[:idx+1, p, 0]/1000, track[:idx+1, p, 1]/1000,
                    lw=.8, alpha=.3, color="#203b76")
        ax.scatter(pts[:, 0]/1000, pts[:, 1]/1000, s=6, alpha=.65, color="#164db8", edgecolors="none")
        ax.scatter(*(lkp/1000), color="red", marker="*", s=100, edgecolor="white", linewidth=.65, zorder=9)
        ax.set_title(f"Monte Carlo trajectories: t = {minutes:.0f} min")
        fig.savefig(out/f"02_particles_{label}.png", dpi=160)
        plt.close(fig)

        density = kde_map(terr, pts, cfg.bandwidth_m)
        fig, ax = plt.subplots(figsize=(8.6, 8.0), layout="constrained")
        draw_background(ax, terr, cfg, alpha=.5, legend_extra=(lkp_handle,))
        strength = density / (float(density.max()) + 1e-15)
        heat_alpha = np.clip(1.65*np.sqrt(strength), 0, .86)
        pc = ax.imshow(density*1e6, origin="lower",
                       extent=(terr.axis[0]/1000, terr.axis[-1]/1000, terr.axis[0]/1000, terr.axis[-1]/1000),
                       cmap="magma", alpha=heat_alpha, interpolation="bilinear")
        cb = fig.colorbar(pc, ax=ax, shrink=.78, pad=.02)
        cb.set_label("Probability density (per km²)")
        ax.scatter(*(lkp/1000), color="red", marker="*", s=100, edgecolor="white", linewidth=.65, zorder=9)
        ax.set_title(f"Terrain-aware KDE: t = {minutes:.0f} min | h = {cfg.bandwidth_m:g} m")
        fig.savefig(out/f"03_kde_{label}.png", dpi=160)
        plt.close(fig)
        np.save(out/f"kde_{label}.npy", density)

    np.savez_compressed(out/"simulation_data.npz", terrain=terr.z, axis=terr.axis,
                        track=track, guidelines=guides, guide_len=guide_len, lkp=lkp,
                        forbidden=terr.forbidden, mountain=terr.mountain, water=terr.water,
                        trail=terr.trail, river_y=terr.river_y, dt_s=cfg.dt_s)
    flat_x, flat_y = track[..., 0].ravel(), track[..., 1].ravel()
    in_mountain = terr.sample(terr.mountain, flat_x, flat_y) > .5
    in_water = terr.sample(terr.water, flat_x, flat_y) > .5
    final_td = terr.sample(terr.trail_dist, track[-1, :, 0], track[-1, :, 1])
    edge = (np.abs(track[-1]) > terr.axis[-1]-50).any(axis=1)
    print(f"Done: {out.resolve()}")
    print(f"Particle trajectories: {track.shape} | Action counts: {dict(zip(MODES, action_counts))}")
    print(f"No-valid-step fallback occurrences: {failed} | Route transfers: {transfers}")
    print(f"Elevation range: {terr.z.min():.0f}–{terr.z.max():.0f} m; mountain cells: "
          f"{terr.mountain.mean():.1%}; forbidden cells: {terr.forbidden.mean():.1%}")
    print(f"Particle-steps inside mountain: {in_mountain.mean():.3%} | inside river: {in_water.mean():.3%}")
    print(f"Final particles within 100 m of a trail: {(final_td < 100).mean():.1%} | at map edge: {edge.mean():.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results"))
    ap.add_argument("--particles", type=int, default=1440)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--bandwidth", type=float, default=150.0)
    ap.add_argument("--beta", type=float, default=1.5,
                    help="Soft preference for low, gentle terrain outside the no-go mountains")
    ap.add_argument("--mountain-z", type=float, default=460.0,
                    help="Absolute elevation (m) at and above which terrain is a no-go mountain")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--action-persistence", type=float, default=8.0,
                    help="Expected behavior-mode duration in minutes")
    args = ap.parse_args()
    if args.particles < 12 or args.hours <= 0 or args.bandwidth <= 0 or args.beta < 0 or args.action_persistence <= 0:
        ap.error("Need particles>=12, hours>0, bandwidth>0, beta>=0, action-persistence>0")
    cfg = Settings(seed=args.seed, particles=args.particles, hours=args.hours,
                   bandwidth_m=args.bandwidth, beta_terrain=args.beta,
                   action_persistence_min=args.action_persistence, mountain_z_m=args.mountain_z)
    terr = make_terrain(cfg)
    guidelines, guide_len, lkp = create_guidelines(terr, cfg)
    track, actions, failed, transfers = simulate(terr, guidelines, guide_len, lkp, cfg)
    save_outputs(args.output, cfg, terr, guidelines, guide_len, lkp, track, actions, failed, transfers)


if __name__ == "__main__":
    main()
