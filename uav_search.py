"""Single-UAV search on a saved Monte Carlo run (Ge et al. 2026, Sec. III, one UAV).

Step 1 (this file so far): search configuration, particle belief with time
interpolation and weights, rectangular camera footprint, and the probability
of area (POA) inside the footprint computed by integrating the Gaussian KDE
analytically over the rotated rectangle.

Check the footprint POA against numerical integration and particle counting:
    python uav_search.py results_battlefield_paper --check-fov

Step 2: fly a mission (detections reduce particle weights, w <- w(1 - POD p))
and replay it over the density with mission_viewer.py:
    python uav_search.py results_battlefield_paper --plan greedy
    python mission_viewer.py results_battlefield_paper/search_greedy
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr


@dataclass
class SearchConfig:
    launch_delay_min: float = 30.0     # UAV launch time after the LKP (paper: 30 min)
    flight_min: float = 120.0          # flight-time limit (paper: 2 h)
    uav_speed_mps: float = 5.0         # paper: 5 m/s
    dt_s: float = 5.0                  # planning interval; one step = speed * dt (paper: 25 m)
    fov_along_m: float = 83.0          # footprint length along the heading (L)
    fov_cross_m: float = 115.0         # footprint width across the heading (W, swath)
    max_turn_deg: float = 20.0         # heading change limit per step
    bandwidth_m: float = 150.0         # KDE spatial bandwidth h_s
    pod: float = 1.0                   # probability of detection inside the footprint
    start_x: float | None = None       # launch point; None = LKP + start offset
    start_y: float | None = None
    start_offset_x: float = 0.0        # launch point relative to the LKP (m), used when start_x/y is None
    start_offset_y: float = 0.0
    start_heading_deg: float | None = None  # None = last known heading if stored, else 0
    particle_weights: str = "remaining"  # planner particle mass: "remaining" w/N (sums to 1-POS) or "conditional" w/sum(w)
    explore_weight: float = 0.0        # exploration bonus lambda: total mass spread uniformly over the zone (0 = off)
    explore_density: float = 0.0       # exploration bonus b per km^2 of unseen zone ground, fixed in time (overrides lambda)
    explore_zone: str = "disc"         # "disc": LKP + explore_speed * t; "prior": prior-density HDR at time t
    explore_speed_mps: float = 0.75    # disc radius speed (0 = whole passable map)
    explore_hdr: float = 0.95          # prior zone: highest-density region holding this share of the prior
    explore_cell_m: float = 25.0       # coverage grid cell

    @property
    def step_m(self) -> float:
        return self.uav_speed_mps*self.dt_s

    @property
    def n_steps(self) -> int:
        return int(round(self.flight_min*60/self.dt_s))


class ParticleBelief:
    """Monte Carlo particles over time with per-particle weights.

    ``track`` has shape (T, N, 2) sampled every ``track_dt_s`` from the LKP
    (t = 0). Weights start at 1 and are reduced as the UAV searches; total
    probability is ``w.sum() / N`` and is never renormalised, so it equals
    1 - cumulative POS.
    """

    def __init__(self, track: np.ndarray, track_dt_s: float):
        self.track = track
        self.track_dt_s = float(track_dt_s)
        self.n = track.shape[1]
        self.w = np.ones(self.n)

    @property
    def duration_s(self) -> float:
        return (self.track.shape[0]-1)*self.track_dt_s

    @property
    def mass(self) -> float:
        return float(self.w.sum()/self.n)

    def positions(self, t_s: float) -> np.ndarray:
        """Particle positions (N, 2) at time t_s after the LKP, linearly interpolated."""
        f = np.clip(t_s/self.track_dt_s, 0, self.track.shape[0]-1)
        i = min(int(np.floor(f)), self.track.shape[0]-2)
        a = f - i
        return (1-a)*self.track[i] + a*self.track[i+1]


def footprint_probs(px, py, cx, cy, yaw, along_m, cross_m, h_m):
    """Fraction of each particle's Gaussian kernel inside each rotated footprint.

    px, py: (N,) particle positions; cx, cy, yaw: (M,) footprint centres and
    headings (rad). Returns (M, N). The footprint is a rectangle of length
    ``along_m`` along the heading and width ``cross_m`` across it; with an
    isotropic Gaussian kernel the integral separates into two erf factors in
    the footprint frame. h_m = 0 gives the hard inside/outside indicator.
    """
    cx, cy, yaw = (np.atleast_1d(np.asarray(v, float))[:, None] for v in (cx, cy, yaw))
    dx, dy = px[None, :] - cx, py[None, :] - cy
    c, s = np.cos(yaw), np.sin(yaw)
    u = dx*c + dy*s
    v = -dx*s + dy*c
    if h_m <= 0:
        return ((np.abs(u) <= along_m/2) & (np.abs(v) <= cross_m/2)).astype(float)
    pu = ndtr((along_m/2 - u)/h_m) - ndtr((-along_m/2 - u)/h_m)
    pv = ndtr((cross_m/2 - v)/h_m) - ndtr((-cross_m/2 - v)/h_m)
    return pu*pv


def poa(belief: ParticleBelief, t_s, cx, cy, yaw, cfg: SearchConfig):
    """POA of each footprint at time t_s: share of the total (initial) probability inside it."""
    pts = belief.positions(t_s)
    p = footprint_probs(pts[:, 0], pts[:, 1], cx, cy, yaw, cfg.fov_along_m, cfg.fov_cross_m, cfg.bandwidth_m)
    return (p*belief.w[None, :]).sum(axis=1)/belief.n


def footprint_corners(cx, cy, yaw, along_m, cross_m) -> np.ndarray:
    """(5, 2) closed polygon of a footprint, for plotting."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])*[along_m/2, cross_m/2]
    return np.stack([cx + local[:, 0]*c - local[:, 1]*s, cy + local[:, 0]*s + local[:, 1]*c], axis=1)


def load_run(run: Path) -> dict:
    f = run/"simulation_data.npz" if run.is_dir() else run
    if not f.exists():
        raise SystemExit(f"{f} not found. Simulate the target first, e.g.\n"
                         f"  python battlefield_infiltrator_demo.py --output {run}")
    d = np.load(f)
    data = {k: d[k] for k in d.files}
    data["dt_s"] = float(data.get("dt_s", 30.0))
    data["cell"] = float(data["axis"][1] - data["axis"][0])
    data["dir"] = f.parent
    return data


def validate(cfg: SearchConfig, belief: ParticleBelief) -> SearchConfig:
    """Check timing and footprint coverage; shorten the flight if the track is too short."""
    if not 0 < cfg.pod <= 1:
        raise ValueError(f"pod must be in (0, 1], got {cfg.pod}")
    if cfg.explore_weight < 0 or cfg.explore_density < 0:
        raise ValueError("explore_weight and explore_density must be >= 0")
    if cfg.particle_weights not in ("remaining", "conditional"):
        raise ValueError(f"particle_weights must be 'remaining' or 'conditional', got {cfg.particle_weights}")
    if cfg.explore_zone not in ("disc", "prior"):
        raise ValueError(f"explore_zone must be 'disc' or 'prior', got {cfg.explore_zone}")
    if not 0 < cfg.explore_hdr <= 1:
        raise ValueError(f"explore_hdr must be in (0, 1], got {cfg.explore_hdr}")
    if cfg.explore_weight > 0 and cfg.explore_density > 0:
        warnings.warn("Both explore_weight and explore_density set; the fixed density is used.")
    end_s = (cfg.launch_delay_min + cfg.flight_min)*60
    if end_s > belief.duration_s + 1e-6:
        new_flight = max(0.0, belief.duration_s/60 - cfg.launch_delay_min)
        warnings.warn(f"Track covers {belief.duration_s/60:.0f} min but launch + flight needs "
                      f"{end_s/60:.0f} min; flight limited to {new_flight:.0f} min. "
                      "Regenerate the run with a longer --hours to fly the full time.")
        cfg.flight_min = new_flight
    if cfg.step_m > cfg.fov_along_m:
        warnings.warn(f"Step {cfg.step_m:.0f} m exceeds footprint length {cfg.fov_along_m:.0f} m: "
                      "gaps between consecutive footprints are never searched.")
    return cfg


def setup(run: Path, cfg: SearchConfig):
    data = load_run(run)
    belief = ParticleBelief(data["track"], data["dt_s"])
    cfg = validate(cfg, belief)
    if cfg.start_x is None or cfg.start_y is None:
        cfg.start_x = float(data["lkp"][0]) + cfg.start_offset_x
        cfg.start_y = float(data["lkp"][1]) + cfg.start_offset_y
    if cfg.start_heading_deg is None:
        cfg.start_heading_deg = float(data.get("heading_deg", 0.0))
    return data, belief, cfg


class Coverage:
    """Unsearched passable ground on a fine grid, for statistics and the exploration bonus.

    ``u`` is the unseen fraction of each cell (1 = never inside a footprint, 0 = searched); impassable
    cells (rivers, mountains, steep slopes) start at 0 and never score. For the bonus, unseen cells inside
    the zone act as static extra particles detected with POD 1. The zone is either the reachable disc
    (radius explore_speed * t around the LKP) or the prior-density HDR at time t (where the target model
    says it could be, before any search). Each unseen cell holds either a fixed ``explore_density`` per
    km^2 or a total ``explore_weight`` spread uniformly over the disc.
    """

    def __init__(self, data: dict, cfg: SearchConfig, block_m: float = 300.0):
        c = cfg.explore_cell_m
        axis, cell = data["axis"], data["cell"]
        lo = axis[0] - cell/2
        n = int(np.ceil((axis[-1] + cell/2 - lo)/c))
        self.cfg, self.c, self.lo = cfg, c, lo
        self.centres = lo + (np.arange(n) + .5)*c
        idx = np.clip(np.round((self.centres - axis[0])/cell).astype(int), 0, len(axis)-1)
        self.u = (~data["forbidden"][np.ix_(idx, idx)]).astype(float)
        self.passable_km2 = self.u.sum()*c*c/1e6
        self.lkp = data["lkp"].astype(float)
        self.axis, self.grid_cell, self.forbidden = axis, cell, data["forbidden"]
        self.track, self.track_dt = data["track"], data["dt_s"]
        self._zone_k = None
        self.b = max(1, int(round(block_m/c)))
        nb = -(-n//self.b)
        self.coarse = np.zeros((nb, nb))
        self._refresh_coarse(0, n, 0, n)
        bc = lo + (np.arange(nb) + .5)*self.b*c
        self.block_x, self.block_y = np.meshgrid(bc, bc)

    def _refresh_coarse(self, iy0, iy1, ix0, ix1):
        b = self.b
        by0, by1, bx0, bx1 = iy0//b, -(-iy1//b), ix0//b, -(-ix1//b)
        sub = self.u[by0*b:by1*b, bx0*b:bx1*b]
        pad = np.zeros(((by1-by0)*b, (bx1-bx0)*b))
        pad[:sub.shape[0], :sub.shape[1]] = sub
        self.coarse[by0:by1, bx0:bx1] = pad.reshape(by1-by0, b, bx1-bx0, b).sum((1, 3))

    def _window(self, x, y, r):
        n = len(self.centres)
        lo = lambda v: int(np.clip(np.floor((v - r - self.lo)/self.c), 0, n))
        hi = lambda v: int(np.clip(np.ceil((v + r - self.lo)/self.c), 0, n))
        return lo(y), hi(y), lo(x), hi(x)

    def radius(self, t_s: float) -> float:
        return self.cfg.explore_speed_mps*t_s if self.cfg.explore_speed_mps > 0 else np.inf

    def cell_mass(self, t_s: float) -> float:
        """Bonus mass of one fully unseen cell at time t_s."""
        if self.cfg.explore_density > 0:
            return self.cfg.explore_density*self.c**2/1e6
        r = self.radius(t_s)
        area = np.pi*r*r if np.isfinite(r) else self.passable_km2*1e6
        return self.cfg.explore_weight*self.c**2/max(area, self.c**2)

    def _prior_zone(self, t_s: float):
        """HDR mask of the unsearched prior KDE at the nearest track snapshot: (mask, (row0, col0))."""
        k = int(round(np.clip(t_s/self.track_dt, 0, len(self.track)-1)))
        if k != self._zone_k:
            cell, axis, h = self.grid_cell, self.axis, self.cfg.bandwidth_m
            ij = np.clip(np.round((self.track[k] - axis[0])/cell).astype(int), 0, len(axis)-1)
            m = int(np.ceil(4*h/cell))
            c0, c1 = max(ij[:, 0].min() - m, 0), min(ij[:, 0].max() + m + 1, len(axis))
            r0, r1 = max(ij[:, 1].min() - m, 0), min(ij[:, 1].max() + m + 1, len(axis))
            hist = np.zeros((r1 - r0, c1 - c0))
            np.add.at(hist, (ij[:, 1] - r0, ij[:, 0] - c0), 1.0)
            dens = gaussian_filter(hist, h/cell, mode="constant")
            dens[self.forbidden[r0:r1, c0:c1]] = 0
            v = np.sort(dens.ravel())[::-1]
            cum = np.cumsum(v)
            thr = v[min(np.searchsorted(cum, self.cfg.explore_hdr*cum[-1]), len(v)-1)]
            self._zone, self._zone_off, self._zone_k = (dens >= thr) & (dens > 0), (r0, c0), k
        return self._zone, self._zone_off

    def in_zone(self, x, y, t_s: float) -> np.ndarray:
        if self.cfg.explore_zone == "disc":
            return np.hypot(x - self.lkp[0], y - self.lkp[1]) < self.radius(t_s)
        zone, (r0, c0) = self._prior_zone(t_s)
        col = np.round((np.asarray(x) - self.axis[0])/self.grid_cell).astype(int) - c0
        row = np.round((np.asarray(y) - self.axis[0])/self.grid_cell).astype(int) - r0
        ok = (row >= 0) & (row < zone.shape[0]) & (col >= 0) & (col < zone.shape[1])
        out = np.zeros(np.shape(x), bool)
        out[ok] = zone[row[ok], col[ok]]
        return out

    def near_cells(self, x, y, reach, t_s):
        """Unseen cells within ``reach`` of (x, y) inside the zone: (xy (K, 2), mass (K,))."""
        iy0, iy1, ix0, ix1 = self._window(x, y, reach)
        gx, gy = np.meshgrid(self.centres[ix0:ix1], self.centres[iy0:iy1])
        u = self.u[iy0:iy1, ix0:ix1]
        ok = (u > 1e-3) & (np.hypot(gx - x, gy - y) < reach)
        ok[ok] = self.in_zone(gx[ok], gy[ok], t_s)
        return np.stack([gx[ok], gy[ok]], axis=1), u[ok]*self.cell_mass(t_s)

    def far_blocks(self, t_s):
        """Coarse unseen mass inside the zone (tested at block centres), for the terminal term."""
        ok = self.coarse > 1e-3
        ok[ok] = self.in_zone(self.block_x[ok], self.block_y[ok], t_s)
        return np.stack([self.block_x[ok], self.block_y[ok]], axis=1), self.coarse[ok]*self.cell_mass(t_s)

    def update(self, poses: np.ndarray):
        cfg = self.cfg
        r = .5*np.hypot(cfg.fov_along_m, cfg.fov_cross_m) + 3*self.c
        for x, y, yaw in poses:
            iy0, iy1, ix0, ix1 = self._window(x, y, r)
            gx, gy = np.meshgrid(self.centres[ix0:ix1], self.centres[iy0:iy1])
            p = footprint_probs(gx.ravel(), gy.ravel(), x, y, yaw, cfg.fov_along_m, cfg.fov_cross_m, self.c)
            self.u[iy0:iy1, ix0:ix1] *= (1 - p[0]).reshape(gx.shape)
            self._refresh_coarse(iy0, iy1, ix0, ix1)

    @property
    def searched_km2(self) -> float:
        return self.passable_km2 - self.u.sum()*self.c**2/1e6


def exact_kde_integral(pts, w, n, cx, cy, yaw, cfg: SearchConfig, spacing_m=2.0):
    """Midpoint-rule integral of the exact (unmasked) KDE over one footprint."""
    nu, nv = int(np.ceil(cfg.fov_along_m/spacing_m)), int(np.ceil(cfg.fov_cross_m/spacing_m))
    du, dv = cfg.fov_along_m/nu, cfg.fov_cross_m/nv
    uu, vv = np.meshgrid((np.arange(nu)+.5)*du - cfg.fov_along_m/2, (np.arange(nv)+.5)*dv - cfg.fov_cross_m/2)
    c, s = np.cos(yaw), np.sin(yaw)
    qx = (cx + uu*c - vv*s).ravel()
    qy = (cy + uu*s + vv*c).ravel()
    h = cfg.bandwidth_m
    total = 0.0
    for k in range(0, len(qx), 4000):
        d2 = (qx[k:k+4000, None]-pts[None, :, 0])**2 + (qy[k:k+4000, None]-pts[None, :, 1])**2
        total += (np.exp(-d2/(2*h*h)) @ w).sum()
    return total/(n*2*np.pi*h*h)*du*dv


def masked_grid_poa(data, pts, w, cx, cy, yaw, cfg: SearchConfig):
    """POA from the viewer's grid KDE (impassable cells zeroed, renormalised) for comparison."""
    from scipy.ndimage import gaussian_filter
    axis, cell = data["axis"], data["cell"]
    bins = np.linspace(axis[0]-cell/2, axis[-1]+cell/2, len(axis)+1)
    counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(bins, bins), weights=w)
    dens = gaussian_filter(counts, cfg.bandwidth_m/cell, mode="constant")
    dens[data["forbidden"]] = 0
    dens *= (w.sum()/len(w))/max(dens.sum(), 1e-300)
    xx, yy = np.meshgrid(axis, axis)
    inside = footprint_probs(xx.ravel(), yy.ravel(), [cx], [cy], [yaw],
                             cfg.fov_along_m, cfg.fov_cross_m, 0)[0].reshape(xx.shape)
    return float((dens*inside).sum())


def check_fov(run: Path, cfg: SearchConfig, out_dir: Path | None = None):
    """Compare the analytic footprint POA with numerical integration and particle counting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    data, belief, cfg = setup(run, cfg)
    t_s = cfg.launch_delay_min*60
    pts = belief.positions(t_s)
    w = belief.w
    axis, cell = data["axis"], data["cell"]
    bins = np.linspace(axis[0]-cell/2, axis[-1]+cell/2, len(axis)+1)
    counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(bins, bins))
    dens = gaussian_filter(counts, cfg.bandwidth_m/cell, mode="constant")
    r, c = np.unravel_index(np.argmax(dens), dens.shape)
    peak = np.array([axis[c], axis[r]])
    lkp = data["lkp"].astype(float)
    hdg = np.deg2rad(cfg.start_heading_deg)
    cases = [
        ("KDE peak, heading 0°", peak, 0.0),
        ("KDE peak, heading 45°", peak, np.pi/4),
        ("KDE peak, heading 90°", peak, np.pi/2),
        ("LKP, known heading", lkp, hdg),
        ("300 m ahead of peak", peak + 300*np.array([np.cos(hdg), np.sin(hdg)]), hdg),
        ("2 km from peak (sparse)", peak + 2000*np.array([np.cos(hdg+np.pi/2), np.sin(hdg+np.pi/2)]), 0.0),
    ]
    rows = []
    for name, (cx, cy), yaw in cases:
        analytic = float(poa(belief, t_s, [cx], [cy], [yaw], cfg)[0])
        numeric = exact_kde_integral(pts, w, belief.n, cx, cy, yaw, cfg)
        count = float(footprint_probs(pts[:, 0], pts[:, 1], [cx], [cy], [yaw],
                                      cfg.fov_along_m, cfg.fov_cross_m, 0)[0] @ w)/belief.n
        masked = masked_grid_poa(data, pts, w, cx, cy, yaw, cfg)
        rows.append((name, cx, cy, yaw, analytic, numeric, count, masked))

    print(f"Run: {data['dir']} | t = {t_s/60:.0f} min after LKP | h_s = {cfg.bandwidth_m:g} m | "
          f"footprint {cfg.fov_along_m:g} m (along) x {cfg.fov_cross_m:g} m (across)")
    print(f"{'case':26s} {'analytic':>10s} {'numeric':>10s} {'rel.err':>8s} {'count':>8s} {'masked grid':>12s}")
    for name, *_, a, nu, cnt, m in rows:
        rel = abs(a-nu)/max(nu, 1e-300)
        print(f"{name:26s} {a:10.3e} {nu:10.3e} {rel:8.1e} {cnt:8.3e} {m:12.3e}")

    out_dir = out_dir or data["dir"]/"search_step1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pts_all = np.concatenate([pts, np.array([[x, y] for _, x, y, *_ in rows])])
    lo, hi = pts_all.min(axis=0) - 800, pts_all.max(axis=0) + 800
    half = (hi-lo).max()/2
    ctr = (lo+hi)/2
    sel = (axis >= ctr[0]-half) & (axis <= ctr[0]+half)
    sely = (axis >= ctr[1]-half) & (axis <= ctr[1]+half)
    fig, ax = plt.subplots(figsize=(9, 8.4), layout="constrained")
    dsub = dens[np.ix_(sely, sel)]/(belief.n*cell**2)*1e6
    ax.imshow(np.ma.masked_where(dsub < dsub.max()*2e-3, dsub), origin="lower", cmap="turbo",
              extent=(axis[sel][0]/1000, axis[sel][-1]/1000, axis[sely][0]/1000, axis[sely][-1]/1000),
              alpha=.85, interpolation="bilinear")
    wat = data["water"][np.ix_(sely, sel)]
    ax.contourf(axis[sel]/1000, axis[sely]/1000, wat.astype(float), levels=[.5, 2], colors=["#1565c0"])
    mtn = data["mountain"][np.ix_(sely, sel)]
    ax.contour(axis[sel]/1000, axis[sely]/1000, mtn.astype(float), levels=[.5], colors="#4e342e")
    ax.scatter(pts[:, 0]/1000, pts[:, 1]/1000, s=3, c="k", alpha=.35, edgecolors="none")
    for k, (name, cx, cy, yaw, a, *_rest) in enumerate(rows):
        poly = footprint_corners(cx, cy, yaw, cfg.fov_along_m, cfg.fov_cross_m)/1000
        ax.plot(poly[:, 0], poly[:, 1], color="white", lw=2.6)
        ax.plot(poly[:, 0], poly[:, 1], color="magenta", lw=1.3)
        ax.annotate("", xy=(cx/1000 + .12*np.cos(yaw), cy/1000 + .12*np.sin(yaw)), xytext=(cx/1000, cy/1000),
                    arrowprops=dict(arrowstyle="-|>", color="magenta", lw=1.2))
        ax.text(cx/1000 + .08, cy/1000 - .12*(k % 3), f"{k+1}: {a:.2e}", fontsize=8, color="black",
                bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax.scatter(*(lkp/1000), marker="*", s=150, color="red", edgecolor="white", zorder=9)
    ax.set(xlim=((ctr[0]-half)/1000, (ctr[0]+half)/1000), ylim=((ctr[1]-half)/1000, (ctr[1]+half)/1000),
           xlabel="x (km)", ylabel="y (km)", aspect="equal",
           title=f"Footprint POA check at UAV launch (t = {t_s/60:.0f} min), "
                 f"{cfg.fov_along_m:g}×{cfg.fov_cross_m:g} m, h = {cfg.bandwidth_m:g} m")
    fig.savefig(out_dir/"fov_check.png", dpi=150)
    plt.close(fig)
    (out_dir/"config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"Saved {out_dir/'fov_check.png'}")
    return rows


# ---------------------------------------------------------------- planners and mission loop

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


class Planner:
    """Chooses future poses for every UAV.

    ``plan`` receives the belief, current poses (U, 3) = x, y, yaw and the
    current time, and returns planned poses (U, H, 3) for t + dt ... t + H*dt.
    The mission loop executes only the first planned pose (receding horizon).
    """
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
        """Divisor turning particle weights into planner mass (N, or the remaining sum for conditional)."""
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

    If every candidate sees (numerically) nothing, steer towards the weighted
    centroid of the remaining particles instead of flying straight away.
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


class TreePlanner(Planner):
    """Receding-horizon search over the heading lattice (n_headings ** horizon paths).

    With ``beam > 0`` only the best ``beam`` partial paths (value so far plus
    terminal term) are expanded at each depth.

    Path value counts every particle once (single counting):
        J = sum_i w_i [1 - prod_j (1 - POD p_ij)] / N
    with p_ij evaluated at t + j*dt. An optional terminal term adds the value of
    one more footprint at the path end under a wide kernel (bandwidth R), using
    the weights the path would leave behind; it pulls the UAV towards remaining
    probability that lies beyond the horizon. Particles whose footprint
    probabilities are negligible for every path are skipped for the exact term.
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


class OptPlanner(Planner):
    """Receding-horizon trajectory optimisation over continuous turn angles (Ge et al. Sec. III, CasADi-free).

    Decision variables are the turns d_1..d_H in [-max_turn, +max_turn]; yaw_k = yaw_0 + sum_{m<=k} d_m and
    each step has the fixed length v*dt, so the turn and spacing constraints become simple bounds. The
    objective is the tree planner's single-counting value plus the same wide-kernel terminal term, maximised
    with L-BFGS-B from several starts (previous solution shifted by one step, straight, gentle arcs) using an
    analytic gradient.
    """
    name = "opt"

    def __init__(self, cfg, n_headings=5, horizon=20, terminal_weight=1.0, terminal_bw_m=600.0,
                 max_iter=40, min_weight=1e-6, near_sigmas=3.5, **_):
        super().__init__(cfg, n_headings)
        self.horizon = horizon
        self.terminal_weight = terminal_weight
        self.terminal_bw_m = terminal_bw_m
        self.max_iter = max_iter
        self.min_weight = min_weight
        self.near_sigmas = near_sigmas
        self.max_turn = np.deg2rad(cfg.max_turn_deg)
        self.prev: dict[int, np.ndarray] = {}

    def rollout(self, pose, turns):
        yaw = pose[2] + np.cumsum(turns)
        x = pose[0] + self.cfg.step_m*np.cumsum(np.cos(yaw))
        y = pose[1] + self.cfg.step_m*np.cumsum(np.sin(yaw))
        return x, y, yaw

    def objective(self, turns, pose, P, w, pod, h, far_xy, far_w):
        """Negative path value and its gradient w.r.t. the turns. P: (H, M, 2) near items per step."""
        cfg = self.cfg
        L, W, step = cfg.fov_along_m, cfg.fov_cross_m, cfg.step_m
        x, y, yaw = self.rollout(pose, turns)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        dx, dy = P[..., 0] - x[:, None], P[..., 1] - y[:, None]
        u, v = dx*c + dy*s, -dx*s + dy*c
        a1, a2, b1, b2 = (L/2 - u)/h, (-L/2 - u)/h, (W/2 - v)/h, (-W/2 - v)/h
        pu, pv = ndtr(a1) - ndtr(a2), ndtr(b1) - ndtr(b2)
        q = 1 - pod*pu*pv
        ones = np.ones((1, q.shape[1]))
        pre = np.cumprod(np.vstack([ones, q[:-1]]), axis=0)
        suf = np.cumprod(np.vstack([ones, q[:0:-1]]), axis=0)[::-1]
        surv = pre[-1]*q[-1]
        excl = pre*suf

        R = self.terminal_bw_m
        kt = self.terminal_weight*L*W/(2*np.pi*R*R)
        ex, ey = P[-1, :, 0] - x[-1], P[-1, :, 1] - y[-1]
        K = np.exp(-(ex*ex + ey*ey)/(2*R*R))
        value = (w*(1 - surv)).sum() + kt*(w*surv*K).sum()
        gx_end = kt*(w*surv*K*ex).sum()/(R*R)
        gy_end = kt*(w*surv*K*ey).sum()/(R*R)
        if len(far_w):
            fx, fy = far_xy[:, 0] - x[-1], far_xy[:, 1] - y[-1]
            Kf = np.exp(-(fx*fx + fy*fy)/(2*R*R))
            value += kt*(far_w*Kf).sum()
            gx_end += kt*(far_w*Kf*fx).sum()/(R*R)
            gy_end += kt*(far_w*Kf*fy).sum()/(R*R)

        dval_dp = pod*excl*(w*(1 - kt*K))[None, :]
        inv = 1/(np.sqrt(2*np.pi)*h)
        dpu_du = (np.exp(-a2*a2/2) - np.exp(-a1*a1/2))*inv
        dpv_dv = (np.exp(-b2*b2/2) - np.exp(-b1*b1/2))*inv
        A, B = dval_dp*pv*dpu_du, dval_dp*pu*dpv_dv
        gx = (-A*c + B*s).sum(1)
        gy = (-A*s - B*c).sum(1)
        gyaw = (A*v - B*u).sum(1)
        gx[-1] += gx_end
        gy[-1] += gy_end
        rgx, rgy = np.cumsum(gx[::-1])[::-1], np.cumsum(gy[::-1])[::-1]
        g_yaw_total = gyaw + step*(-np.sin(yaw)*rgx + np.cos(yaw)*rgy)
        grad = np.cumsum(g_yaw_total[::-1])[::-1]
        return -value, -grad

    def plan(self, belief, poses, t_s):
        from scipy.optimize import minimize
        cfg, H = self.cfg, self.horizon
        out = np.empty((len(poses), H, 3))
        self.fallback = False
        times = t_s + cfg.dt_s*np.arange(1, H+1)
        bounds = [(-self.max_turn, self.max_turn)]*H
        for u, pose in enumerate(poses):
            it = self.gather(belief, times, pose, self.min_weight, self.near_sigmas, self.terminal_bw_m/4)
            args = (pose, it.P, it.w, it.pod, it.h, *it.far(H-1))
            starts = [np.zeros(H), np.full(H, .5*self.max_turn), np.full(H, -.5*self.max_turn)]
            if u in self.prev:
                starts.insert(0, np.r_[self.prev[u][1:], 0.0])
            best = None
            for d0 in starts:
                res = minimize(self.objective, d0, args=args, jac=True, method="L-BFGS-B", bounds=bounds,
                               options=dict(maxiter=self.max_iter))
                if best is None or res.fun < best.fun:
                    best = res
            if -best.fun < 1e-12:
                first = self.advance(pose[None, :], self.turns)
                k = self.toward_centroid(belief, first, times[0])
                turns = np.r_[self.turns[k], np.zeros(H-1)]
                self.prev.pop(u, None)
            else:
                turns = best.x
                self.prev[u] = turns
            out[u] = np.stack(self.rollout(pose, turns), axis=1)
        return out


PLANNERS = {p.name: p for p in (StraightPlanner, GreedyPlanner, TreePlanner, OptPlanner)}


def detect(belief: ParticleBelief, t_s: float, poses: np.ndarray, cfg: SearchConfig) -> float:
    """Apply w <- w * prod_u (1 - POD * p_u) for footprints at time t_s; return the probability found."""
    before = belief.mass
    pts = belief.positions(t_s)
    p = footprint_probs(pts[:, 0], pts[:, 1], poses[:, 0], poses[:, 1], poses[:, 2],
                        cfg.fov_along_m, cfg.fov_cross_m, cfg.bandwidth_m)
    belief.w *= np.prod(1 - cfg.pod*p, axis=0)
    return before - belief.mass


def run_mission(belief: ParticleBelief, cfg: SearchConfig, planner: Planner, start_poses: np.ndarray,
                coverage: Coverage | None = None) -> dict:
    """Fly the mission and log every step in the format read by ``sim_viewer``."""
    n_steps = cfg.n_steps
    n_uav = len(start_poses)
    t = cfg.launch_delay_min*60 + np.arange(n_steps+1)*cfg.dt_s
    poses = np.empty((n_steps+1, n_uav, 3))
    plan = np.full((n_steps+1, n_uav, planner.horizon, 3), np.nan)
    weights = np.empty((n_steps+1, belief.n), np.float32)
    gain = np.zeros(n_steps+1)
    fallback = np.zeros(n_steps+1, bool)
    searched = np.zeros(n_steps+1)
    planner.coverage = coverage

    def observe(k):
        gain[k] = detect(belief, t[k], poses[k], cfg)
        weights[k] = belief.w
        if coverage is not None:
            coverage.update(poses[k])
            searched[k] = coverage.searched_km2

    poses[0] = start_poses
    observe(0)
    report = max(1, n_steps//10)
    for k in range(n_steps):
        if k and k % report == 0 and planner.horizon > 1:
            print(f"  step {k}/{n_steps} | POS {1 - belief.mass:.4f}", flush=True)
        p = planner.plan(belief, poses[k], t[k])
        plan[k] = p
        fallback[k] = planner.fallback
        poses[k+1] = p[:, 0]
        observe(k+1)
    log = dict(t_s=t, poses=poses, plan=plan, weights=weights, gain=gain, pos_cum=np.cumsum(gain),
               fallback=fallback)
    if coverage is not None:
        log["searched_km2"] = searched
    return log


def fly(run: Path, cfg: SearchConfig, planner_name: str, name: str | None = None, n_headings: int = 5,
        **planner_opts):
    data, belief, cfg = setup(run, cfg)
    planner = PLANNERS[planner_name](cfg, n_headings, **planner_opts)
    start = np.array([[cfg.start_x, cfg.start_y, np.deg2rad(cfg.start_heading_deg)]])
    t0 = time.perf_counter()
    log = run_mission(belief, cfg, planner, start, Coverage(data, cfg))
    elapsed = time.perf_counter() - t0
    out_dir = data["dir"]/f"search_{name or planner_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(asdict(cfg), planner=planner_name, n_headings=n_headings, horizon=planner.horizon,
                planner_opts={k: getattr(planner, k) for k in planner_opts if hasattr(planner, k)},
                run_time_s=round(elapsed, 1), run=str(data["dir"].resolve()))
    np.savez_compressed(out_dir/"mission.npz", meta=json.dumps(meta), **log)
    (out_dir/"config.json").write_text(json.dumps(meta, indent=2))
    pos = log["pos_cum"]
    print(f"Planner {planner_name}: {cfg.n_steps} steps of {cfg.step_m:g} m, "
          f"launch {cfg.launch_delay_min:g} min, flight {cfg.flight_min:g} min")
    for m in (15, 30, 60, 90, 120):
        k = int(round(m*60/cfg.dt_s))
        if k < len(pos):
            print(f"  POS after {m:3d} min of flight: {pos[k]:.4f}")
    print(f"  final POS {pos[-1]:.4f} | searched ground {log['searched_km2'][-1]:.2f} km2 | "
          f"fallback steps {int(log['fallback'].sum())} | "
          f"{elapsed:.1f} s ({1000*elapsed/max(cfg.n_steps, 1):.0f} ms/step)")
    print(f"Saved {out_dir/'mission.npz'}\nView:  python mission_viewer.py {out_dir}")
    return log


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="Result folder (or simulation_data.npz) of a Monte Carlo run")
    d = SearchConfig()
    ap.add_argument("--launch-delay", type=float, default=d.launch_delay_min, help="Minutes after the LKP")
    ap.add_argument("--flight", type=float, default=d.flight_min, help="Flight-time limit in minutes")
    ap.add_argument("--speed", type=float, default=d.uav_speed_mps, help="UAV speed m/s")
    ap.add_argument("--dt", type=float, default=d.dt_s, help="Planning interval s")
    ap.add_argument("--fov", type=float, nargs=2, default=(d.fov_along_m, d.fov_cross_m),
                    metavar=("ALONG", "ACROSS"), help="Footprint size in metres")
    ap.add_argument("--max-turn", type=float, default=d.max_turn_deg, help="Heading change limit per step (deg)")
    ap.add_argument("--bandwidth", type=float, default=d.bandwidth_m, help="KDE bandwidth h_s (m)")
    ap.add_argument("--pod", type=float, default=d.pod,
                    help="Probability of detection for a target inside the footprint (0-1]")
    ap.add_argument("--start", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="Absolute launch point in map metres (default: LKP)")
    ap.add_argument("--start-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="Launch point relative to the LKP in metres, e.g. --start-offset -1000 500")
    ap.add_argument("--start-heading", type=float, default=None, help="Initial heading deg (default: known heading)")
    ap.add_argument("--explore", type=float, default=d.explore_weight,
                    help="Exploration bonus: total mass spread uniformly over unseen passable ground inside "
                         "the reachable disc (0 = off; e.g. 0.3)")
    ap.add_argument("--explore-density", type=float, default=d.explore_density,
                    help="Exploration bonus b per km^2 of unseen zone ground, fixed in time (overrides --explore)")
    ap.add_argument("--explore-zone", choices=("disc", "prior"), default=d.explore_zone,
                    help="Where unseen ground scores: 'disc' (LKP + speed * t) or 'prior' (prior-density HDR)")
    ap.add_argument("--explore-hdr", type=float, default=d.explore_hdr,
                    help="Prior zone: HDR level, e.g. 0.95 = smallest area holding 95%% of the prior")
    ap.add_argument("--weights", choices=("remaining", "conditional"), default=d.particle_weights,
                    help="Particle mass in the planner objective: remaining (sums to 1-POS) or conditional (sums to 1)")
    ap.add_argument("--explore-speed", type=float, default=d.explore_speed_mps,
                    help="Reachable disc radius = speed * t after the LKP (m/s, 0 = whole passable map)")
    ap.add_argument("--explore-cell", type=float, default=d.explore_cell_m, help="Coverage grid cell (m)")
    ap.add_argument("--check-fov", action="store_true", help="Verify the footprint POA and save a figure")
    ap.add_argument("--plan", choices=sorted(PLANNERS), default=None,
                    help="Fly a mission with this planner and save search_<name>/mission.npz")
    ap.add_argument("--name", default=None, help="Output folder suffix (default: planner name)")
    ap.add_argument("--headings", type=int, default=5, help="Candidate headings per step")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Tree/opt planner: lookahead steps n_l (default 6 for tree, 20 for opt)")
    ap.add_argument("--terminal-weight", type=float, default=1.0,
                    help="Tree/opt planner: terminal term weight (0 = off)")
    ap.add_argument("--terminal-bw", type=float, default=600.0,
                    help="Tree/opt planner: terminal kernel width R (m)")
    ap.add_argument("--beam", type=int, default=50,
                    help="Tree planner: partial paths kept per depth (0 = full tree, much slower)")
    ap.add_argument("--max-iter", type=int, default=40, help="Opt planner: L-BFGS-B iterations per start")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = SearchConfig(launch_delay_min=args.launch_delay, flight_min=args.flight,
                       uav_speed_mps=args.speed, dt_s=args.dt, fov_along_m=args.fov[0],
                       fov_cross_m=args.fov[1], max_turn_deg=args.max_turn, bandwidth_m=args.bandwidth,
                       pod=args.pod,
                       start_x=None if args.start is None else args.start[0],
                       start_y=None if args.start is None else args.start[1],
                       start_offset_x=args.start_offset[0], start_offset_y=args.start_offset[1],
                       start_heading_deg=args.start_heading, particle_weights=args.weights,
                       explore_weight=args.explore, explore_density=args.explore_density,
                       explore_zone=args.explore_zone, explore_hdr=args.explore_hdr,
                       explore_speed_mps=args.explore_speed, explore_cell_m=args.explore_cell)
    if args.check_fov:
        check_fov(args.run, cfg)
    elif args.plan:
        opts = {}
        if args.plan == "tree":
            opts = dict(horizon=args.horizon or 6, terminal_weight=args.terminal_weight,
                        terminal_bw_m=args.terminal_bw, beam=args.beam)
        elif args.plan == "opt":
            opts = dict(horizon=args.horizon or 20, terminal_weight=args.terminal_weight,
                        terminal_bw_m=args.terminal_bw, max_iter=args.max_iter)
        fly(args.run, cfg, args.plan, args.name, args.headings, **opts)
    else:
        data, belief, cfg = setup(args.run, cfg)
        print(json.dumps(asdict(cfg), indent=2))
        print(f"Particles: {belief.n} | track: {belief.duration_s/60:.0f} min every {belief.track_dt_s:g} s | "
              f"steps: {cfg.n_steps} of {cfg.step_m:g} m")


if __name__ == "__main__":
    main()
