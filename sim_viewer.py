"""Time-slider viewer for Monte Carlo runs and UAV search missions.

Shared by ``kde_viewer.py`` (probability density only) and ``mission_viewer.py``
(density + UAV trajectories, footprints, plans and cumulative POS).

A mission is a ``mission.npz`` written by ``uav_search.py`` into
``<run>/search_<name>/``. The mission viewer lists every mission of a run and
switches between them inside the window. Mission arrays carry a UAV axis (U)
so multi-UAV missions and longer planning horizons display without changes:

    t_s      (S+1,)          time after the LKP of every mission step
    poses    (S+1, U, 3)     x, y, yaw (rad) of each UAV
    plan     (S+1, U, H, 3)  poses planned at that step (NaN if unused)
    weights  (S+1, N)        particle weights after the detection at that step
    pos_cum  (S+1,)          cumulative probability of success
    meta     JSON string     config, planner name, footprint size
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from scipy.ndimage import gaussian_filter

HDR_LEVELS = (.50, .80, .95)
HDR_STYLES = {.50: ("white", 2.0, "-"), .80: ("white", 1.3, "--"), .95: ("white", .9, ":")}
UAV_COLORS = ("#e040fb", "#00e5ff", "#ffea00", "#ff3d00")
DENSITY_MODES = ("Prior", "Remaining", "Conditional")


# ---------------------------------------------------------------- data

def load_run(path: Path, dt_override=None) -> dict:
    f = path/"simulation_data.npz" if path.is_dir() else path
    d = np.load(f)
    run = {k: d[k] for k in d.files}
    run["dt_s"] = float(dt_override if dt_override is not None else run.get("dt_s", 30.0))
    run["cell"] = float(run["axis"][1] - run["axis"][0])
    run["name"] = f.parent.name
    run["dir"] = f.parent
    run["map_km"] = float(run["axis"][-1] - run["axis"][0])/1000
    run["axis_x"] = run["axis_y"] = run["axis"]
    return run


def load_mission(path: Path) -> dict:
    f = path/"mission.npz" if path.is_dir() else path
    d = np.load(f)
    m = {k: d[k] for k in d.files}
    m["meta"] = json.loads(str(m["meta"]))
    m["dir"] = f.parent
    m["name"] = f.parent.name.removeprefix("search_")
    return m


def find_missions(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("search_*/mission.npz"))


def crop_run(run: dict, points_xy: np.ndarray, margin_m: float):
    """Crop every grid layer to a square window around the given points (metres)."""
    axis = run["axis"]
    lo, hi = points_xy.min(axis=0) - margin_m, points_xy.max(axis=0) + margin_m
    centre, half = (lo + hi)/2, (hi - lo).max()/2
    half = min(half, (axis[-1] - axis[0])/2)
    centre = np.clip(centre, axis[0] + half, axis[-1] - half)
    sel_x = (axis >= centre[0] - half) & (axis <= centre[0] + half)
    sel_y = (axis >= centre[1] - half) & (axis <= centre[1] + half)
    n = min(sel_x.sum(), sel_y.sum())
    cx, cy = np.flatnonzero(sel_x)[:n], np.flatnonzero(sel_y)[:n]
    n_grid = len(axis)
    for k, v in list(run.items()):
        if isinstance(v, np.ndarray) and v.shape == (n_grid, n_grid):
            run[k] = v[np.ix_(cy, cx)]
    run["axis_x"], run["axis_y"] = axis[cx], axis[cy]


def interp_positions(track: np.ndarray, dt_s: float, t_s: float) -> np.ndarray:
    f = np.clip(t_s/dt_s, 0, track.shape[0]-1)
    i = min(int(np.floor(f)), track.shape[0]-2)
    a = f - i
    return (1-a)*track[i] + a*track[i+1]


def kde_map(run, points, bandwidth_m, weights=None, total_mass=1.0):
    """Weighted grid KDE (1/m^2): impassable cells zeroed, then scaled to integrate to total_mass."""
    xa, ya, cell = run["axis_x"], run["axis_y"], run["cell"]
    bx = np.linspace(xa[0]-cell/2, xa[-1]+cell/2, len(xa)+1)
    by = np.linspace(ya[0]-cell/2, ya[-1]+cell/2, len(ya)+1)
    counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(by, bx), weights=weights)
    density = gaussian_filter(counts, sigma=bandwidth_m/cell, mode="constant")
    density[run["forbidden"]] = 0.0
    s = density.sum()*cell**2
    return density*(total_mass/s) if s > 0 else density


def hdr_thresholds(density, cell):
    """Density levels enclosing 50/80/95 % of the mass shown (highest-density regions)."""
    v = np.sort(density.ravel())[::-1]
    mass = np.cumsum(v)*cell**2
    mass /= max(mass[-1], 1e-300)
    return {p: v[min(np.searchsorted(mass, p), len(v)-1)] for p in HDR_LEVELS}


def footprint_corners(cx, cy, yaw, along_m, cross_m) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])*[along_m/2, cross_m/2]
    return np.stack([cx + local[:, 0]*c - local[:, 1]*s, cy + local[:, 0]*s + local[:, 1]*c], axis=1)


# ---------------------------------------------------------------- drawing

def draw_terrain(ax, run, extent):
    xa, ya = run["axis_x"]/1000, run["axis_y"]/1000
    smooth = .8*40/run["cell"]
    shade = LightSource(azdeg=315, altdeg=40).hillshade(run["terrain"], vert_exag=4,
                                                         dx=run["cell"], dy=run["cell"])
    ax.imshow(shade, origin="lower", extent=extent, cmap="gray", vmin=-.2, vmax=1.25, zorder=0)
    ax.contour(xa, ya, run["terrain"], levels=np.arange(200, 1000, 100), colors="0.35",
               linewidths=.35, alpha=.5, zorder=1)
    handles = []
    if "forest" in run:
        ax.contourf(xa, ya, gaussian_filter(run["forest"].astype(float), smooth), levels=[.5, 2],
                    colors=["#2e7d32"], alpha=.18, zorder=1)
        handles.append(Patch(color="#2e7d32", alpha=.35, label="Forest"))
    mtn = run["mountain"].astype(float)
    if mtn.any():
        ax.contourf(xa, ya, mtn, levels=[.5, 1.5], colors=["#6d4c41"], alpha=.25, zorder=2)
        ax.contourf(xa, ya, mtn, levels=[.5, 1.5], colors="none", hatches=["//"], zorder=2)
        ax.contour(xa, ya, mtn, levels=[.5], colors="#4e342e", linewidths=1.2, zorder=2)
    handles.append(Patch(facecolor="#6d4c41", alpha=.4, hatch="//", label="Mountain (no-go)"))
    if run["water"].any():
        ax.contourf(xa, ya, gaussian_filter(run["water"].astype(float), smooth), levels=[.5, 2],
                    colors=["#1565c0"], zorder=3)
    handles.append(Patch(color="#1565c0", label="River"))
    if "fords" in run and run["fords"].any():
        ax.contour(xa, ya, gaussian_filter(run["fords"].astype(float), smooth), levels=[.4],
                   colors="#ff6f00", linewidths=1.8, zorder=3)
        handles.append(Line2D([], [], color="#ff6f00", lw=1.8, label="Ford"))
    if "trail" in run:
        ax.imshow(np.ma.masked_where(~run["trail"], run["trail"].astype(float)), origin="lower",
                  extent=extent, cmap=ListedColormap(["#fdd835"]), interpolation="nearest", zorder=3)
        handles.append(Line2D([], [], color="#fdd835", lw=2, label="Trail"))
    return handles


def heat_cmap():
    return ListedColormap(plt.get_cmap("turbo")(np.linspace(.12, 1.0, 256)))


# ---------------------------------------------------------------- viewer

class SimViewer:
    """Density over time; with missions, UAV paths, footprints, plans and POS of the selected one."""

    def __init__(self, run: dict, missions: list[Path] | None = None, current: int = 0, t_min=None,
                 h0=150.0, hmin=25.0, hmax=800.0):
        self.run = run
        self.mission_paths = list(missions or [])
        self.wide = bool(self.mission_paths)
        self._cache: dict[int, dict] = {}
        self.mission, self.cur = None, None
        self.track = run["track"]
        self.n_particles = self.track.shape[1]
        self.t_end = (self.track.shape[0]-1)*run["dt_s"]
        self.slider_dt = run["dt_s"]
        self.play_s = max(self.slider_dt, 30.0)   # simulated seconds per playback frame (frames every 0.1 s)
        self.h_fixed = h0
        self.t_launch = 0.0
        self.xa, self.ya = run["axis_x"]/1000, run["axis_y"]/1000
        self.extent = (self.xa[0], self.xa[-1], self.ya[0], self.ya[-1])
        self.locked_vmax = None
        self.contours = []
        self.playing = False
        self.ready = False
        self.mode = "Remaining" if self.wide else "Prior"

        self.fig = plt.figure(figsize=(14.5, 9.6) if self.wide else (10.5, 9.6))
        if self.wide:
            self.ax = self.fig.add_axes([.045, .20, .56, .75])
            self.cax = self.fig.add_axes([.61, .30, .012, .55])
            self.ax_pos = self.fig.add_axes([.715, .64, .265, .29])
        else:
            self.ax = self.fig.add_axes([.07, .20, .76, .74])
            self.cax = self.fig.add_axes([.85, .30, .022, .54])
        handles = draw_terrain(self.ax, run, self.extent)

        self.im = self.ax.imshow(np.zeros_like(run["terrain"]), origin="lower", extent=self.extent,
                                 cmap=heat_cmap(), interpolation="bilinear", zorder=4)
        self.cb = self.fig.colorbar(self.im, cax=self.cax)
        self.cb.set_label("Probability density (per km²)")
        self.scat = self.ax.scatter([], [], s=3, c="black", alpha=.45, edgecolors="none", zorder=6)
        lx, ly = run["lkp"]/1000
        self.ax.scatter(lx, ly, color="red", marker="*", s=160, edgecolor="white", linewidth=.9, zorder=12)
        if "heading_deg" in run:
            h = np.deg2rad(float(run["heading_deg"]))
            self.ax.annotate("", xy=(lx+1.2*np.cos(h), ly+1.2*np.sin(h)), xytext=(lx, ly),
                             arrowprops=dict(arrowstyle="-|>", color="red", lw=1.6), zorder=12)
        handles.append(Line2D([], [], marker="*", color="red", markeredgecolor="white", ls="",
                              markersize=12, label="LKP"))
        handles += [Line2D([], [], color="black", lw=w, ls=s, label=f"HDR {int(p*100)}%")
                    for p, (_, w, s) in HDR_STYLES.items()]
        if self.wide:
            handles += self._init_mission_artists(current)
        self.ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=.92).set_zorder(20)
        self.ax.set(xlabel="x (km)", ylabel="y (km)", xlim=self.extent[:2], ylim=self.extent[2:])
        self.ax.set_aspect("equal")
        self.info = self.ax.text(.99, .01, "", transform=self.ax.transAxes, ha="right", va="bottom",
                                 fontsize=9, zorder=21,
                                 bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=.9))
        self._init_controls(t_min, h0, hmin, hmax)
        if self.wide:
            self.select_mission(current)
        self.ready = True
        self.update()

    # ---- mission artists
    def _init_mission_artists(self, current):
        ax = self.ax
        self.names = [p.parent.name.removeprefix("search_") for p in self.mission_paths]
        self.trails, self.fov_now, self.uav_dot = [], [], []
        self.n_uav = 0
        self.swath = PolyCollection([], facecolors="white", edgecolors="none", alpha=.22, zorder=5)
        ax.add_collection(self.swath)
        self.plan_coll = PolyCollection([], facecolors="none", edgecolors="white", linestyles="--",
                                        linewidths=.8, zorder=10)
        ax.add_collection(self.plan_coll)

        cmap = plt.get_cmap("tab10")
        self.compare_lines = []
        t_last = 0.0
        for i, p in enumerate(self.mission_paths):
            d = np.load(p)
            t_last = max(t_last, float(d["t_s"][-1]))
            line, = self.ax_pos.plot(d["t_s"]/60, d["pos_cum"], lw=1.2, alpha=.9, visible=False,
                                     color=cmap(i % 10), label=self.names[i])
            self.compare_lines.append(line)
        self.own_line, = self.ax_pos.plot([], [], color="#283593", lw=2.2, zorder=5)
        self.own_fill = self.launch_span = self.launch_txt = None
        self.pos_cursor = self.ax_pos.axvline(0, color="red", lw=1, zorder=6)
        self.pos_dot, = self.ax_pos.plot([], [], "o", color="red", ms=5, zorder=7)
        self.ax_pos.set(xlabel="time after LKP (min)", ylabel="cumulative POS",
                        xlim=(0, min(self.t_end, t_last)/60), ylim=(0, 1.02))
        self.ax_pos.grid(alpha=.3)
        self.cfg_text = self.fig.text(.715, .57, "", fontsize=8, va="top", family="monospace")

        n = len(self.names)
        height = min(.18, .022*n + .03)
        y0 = .41 - height
        ax_sel = self.fig.add_axes([.715, y0, .125, height])
        ax_sel.set_title("View mission", fontsize=8, loc="left")
        self.radio_mis = RadioButtons(ax_sel, self.names, active=current)
        ax_cmp = self.fig.add_axes([.855, y0, .125, height])
        ax_cmp.set_title("Compare POS with", fontsize=8, loc="left")
        self.chk_cmp = CheckButtons(ax_cmp, self.names, [False]*n)
        for lbl, ln in zip(self.chk_cmp.labels, self.compare_lines):
            lbl.set_fontsize(7.5)
            lbl.set_color(ln.get_color())
        for lbl in self.radio_mis.labels:
            lbl.set_fontsize(7.5)
        self.radio_mis.on_clicked(lambda label: self.select_mission(self.names.index(label)))
        self.chk_cmp.on_clicked(self._on_compare)

        return [Line2D([], [], color=UAV_COLORS[0], lw=1.6, marker="o", mec="black", label="UAV path / current FOV"),
                Patch(facecolor="white", alpha=.4, edgecolor="0.5", label="Searched swath"),
                Patch(facecolor="none", edgecolor="0.3", ls="--", label="Planned footprints")]

    def _ensure_uav_artists(self, n):
        while len(self.trails) < n:
            col = UAV_COLORS[len(self.trails) % len(UAV_COLORS)]
            self.trails.append(self.ax.plot([], [], color=col, lw=1.4, zorder=9)[0])
            self.fov_now.append(self.ax.fill([], [], facecolor=col, edgecolor="black", alpha=.55,
                                             lw=1.0, zorder=11)[0])
            self.uav_dot.append(self.ax.plot([], [], marker="o", color=col, mec="black", ms=6, zorder=12)[0])

    def select_mission(self, i: int):
        if i == self.cur:
            return
        if i not in self._cache:
            self._cache[i] = load_mission(self.mission_paths[i])
        m = self.mission = self._cache[i]
        self.cur = i
        meta = m["meta"]
        self.t_launch = float(m["t_s"][0])
        self.dt_m = float(m["t_s"][1] - m["t_s"][0]) if len(m["t_s"]) > 1 else self.run["dt_s"]
        self.slider_dt = self.dt_m
        self.s_time.valstep = self.dt_m/60
        self.fov = (meta["fov_along_m"], meta["fov_cross_m"])
        self.h_fixed = meta.get("bandwidth_m", self.h_fixed)
        self.n_uav = m["poses"].shape[1]
        self._ensure_uav_artists(self.n_uav)

        t = m["t_s"]/60
        self.own_line.set_data(t, m["pos_cum"])
        self.own_line.set_label(f"{m['name']} (shown)")
        for a in (self.own_fill, self.launch_span, self.launch_txt):
            if a is not None:
                a.remove()
        self.own_fill = self.ax_pos.fill_between(t, 0, m["pos_cum"], color="#3949ab", alpha=.15, zorder=4)
        self.launch_span = self.ax_pos.axvspan(0, self.t_launch/60, color="0.85", zorder=0)
        self.launch_txt = self.ax_pos.text(self.t_launch/120, .02, "before\nlaunch", ha="center", va="bottom",
                                           fontsize=7, transform=self.ax_pos.get_xaxis_transform())
        self.ax_pos.set_title(f"{m['name']}: planner {meta.get('planner', '?')} | UAVs {self.n_uav} | "
                              f"final POS {m['pos_cum'][-1]:.3f}", fontsize=9)
        opts = meta.get("planner_opts") or {}
        cfg_lines = [f"FOV {self.fov[0]:g}×{self.fov[1]:g} m (along×across)",
                     f"speed {meta.get('uav_speed_mps', '?')} m/s, Δt {meta.get('dt_s', '?')} s",
                     f"turn ±{meta.get('max_turn_deg', '?')}°/step, {meta.get('n_headings', '?')} headings, "
                     f"horizon {m['plan'].shape[2]}",
                     f"launch +{self.t_launch/60:.0f} min, flight {meta.get('flight_min', '?')} min",
                     f"start ({meta.get('start_x', 0)/1000:.2f}, {meta.get('start_y', 0)/1000:.2f}) km",
                     f"h_s {meta.get('bandwidth_m', '?')} m, POD {meta.get('pod', '?')}"]
        if opts:
            cfg_lines += textwrap.wrap(", ".join(f"{k} {v}" for k, v in opts.items()), 46)
        self.cfg_text.set_text("\n".join(cfg_lines))
        self._on_compare(None)
        if self.ready:
            self.update()

    def _on_compare(self, _label):
        for i, (ln, on) in enumerate(zip(self.compare_lines, self.chk_cmp.get_status())):
            ln.set_visible(on and i != self.cur)
        shown = [ln for ln in self.compare_lines if ln.get_visible()] + [self.own_line]
        leg = self.ax_pos.get_legend()
        if leg is not None:
            leg.remove()
        if len(shown) > 1:
            self.ax_pos.legend(handles=shown, fontsize=7, loc="lower right", framealpha=.9)
        self.fig.canvas.draw_idle()

    # ---- controls
    def _init_controls(self, t_min, h0, hmin, hmax):
        wide = self.wide
        t0 = self.t_end if t_min is None else float(np.clip(t_min*60, 0, self.t_end))
        sw = .44 if wide else .50
        ax_t = self.fig.add_axes([.10, .08 if wide else .10, sw, .03])
        self.s_time = Slider(ax_t, "Time (min)", 0, self.t_end/60, valinit=t0/60,
                             valstep=self.slider_dt/60, valfmt="%.1f")
        # A mission's weights were computed with one bandwidth; viewing it with another would misrepresent POS.
        self.s_bw = None
        if not wide:
            ax_h = self.fig.add_axes([.10, .05, sw, .03])
            self.s_bw = Slider(ax_h, "h_s (m)", hmin, hmax, valinit=h0, valstep=5, valfmt="%.0f")
        labels = ["Particles", "HDR 50/80/95%", "Log color", "Lock color scale"]
        status = [False, True, False, False]
        if wide:
            labels += ["Searched swath", "Planned path"]
            status += [True, True]
        ax_chk = self.fig.add_axes([.68 if wide else .74, .025, .14 if wide else .16, .17 if wide else .13])
        self.chk = CheckButtons(ax_chk, labels, status)
        for lbl in self.chk.labels:
            lbl.set_fontsize(8)
        if wide:
            ax_rad = self.fig.add_axes([.83, .025, .10, .17])
            ax_rad.set_title("Density", fontsize=8)
            self.radio = RadioButtons(ax_rad, DENSITY_MODES, active=DENSITY_MODES.index(self.mode))
            for lbl in self.radio.labels:
                lbl.set_fontsize(8)
            self.radio.on_clicked(self._on_mode)
        ax_play = self.fig.add_axes([.935 if wide else .91, .10, .055 if wide else .07, .04])
        self.b_play = Button(ax_play, "Play")
        self.s_time.on_changed(lambda _: self.update())
        if self.s_bw is not None:
            self.s_bw.on_changed(lambda _: self.update())
        self.chk.on_clicked(self._on_check)
        self.b_play.on_clicked(lambda _: self.toggle_play())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self._tick)

    def _on_mode(self, label):
        self.mode = label
        self.update()

    def _on_check(self, label):
        if label == "Lock color scale":
            self.locked_vmax = self.vmax if self.chk.get_status()[3] else None
        self.update()

    def _on_key(self, ev):
        step = self.slider_dt/60
        if ev.key in ("right", "left"):
            s = self.s_time
            s.set_val(float(np.clip(s.val + (step if ev.key == "right" else -step), s.valmin, s.valmax)))
        elif ev.key in ("up", "down"):
            if self.s_bw is not None:
                s = self.s_bw
                s.set_val(float(np.clip(s.val + (25 if ev.key == "up" else -25), s.valmin, s.valmax)))
            else:
                n = len(self.names)
                self.radio_mis.set_active((self.cur + (-1 if ev.key == "up" else 1)) % n)
        elif ev.key == " ":
            self.toggle_play()

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")
        (self.timer.start if self.playing else self.timer.stop)()

    def _tick(self):
        jump = self.play_s/60
        nxt = self.s_time.val + jump
        self.s_time.set_val(self.s_time.valmin if nxt > self.s_time.valmax + 1e-9 else nxt)

    # ---- state at time t
    def _mission_step(self, t_s):
        if self.mission is None or t_s < self.t_launch - 1e-9:
            return None
        return int(np.clip(np.floor((t_s - self.t_launch)/self.dt_m + 1e-9), 0, len(self.mission["t_s"])-1))

    def update(self):
        if not self.ready:
            return
        status = self.chk.get_status()
        show_pts, show_hdr, log_c = status[:3]
        t_s = float(self.s_time.val)*60
        h = float(self.s_bw.val) if self.s_bw is not None else self.h_fixed
        pts = interp_positions(self.track, self.run["dt_s"], t_s)
        k = self._mission_step(t_s)
        w = None
        mass_shown = 1.0
        remaining = 1.0
        if k is not None:
            w_k = self.mission["weights"][k].astype(float)
            remaining = float(w_k.sum()/self.n_particles)
            if self.mode != "Prior":
                w = w_k
                mass_shown = remaining if self.mode == "Remaining" else 1.0
        density = kde_map(self.run, pts, h, weights=w, total_mass=mass_shown)*1e6
        self.vmax = float(density.max()) or 1.0
        vmax = self.locked_vmax or self.vmax
        strength = np.clip(density/vmax, 0, 1)
        if log_c:
            norm = LogNorm(vmin=vmax*1e-3, vmax=vmax)
            alpha = np.clip((np.log10(np.maximum(strength, 1e-6))+3)/3, 0, 1)
        else:
            norm = Normalize(vmin=0, vmax=vmax)
            alpha = np.sqrt(strength)
        alpha = np.where(strength < 2e-3, 0, np.clip(.25 + .7*alpha, 0, .92))
        self.im.set_data(np.ma.masked_where(density <= 0, density))
        self.im.set_norm(norm)
        self.im.set_alpha(alpha)
        self.cb.update_normal(self.im)

        for c in self.contours:
            c.remove()
        self.contours = []
        hdr_txt = ""
        if show_hdr and density.max() > 0:
            th = hdr_thresholds(density, self.run["cell"])
            areas = []
            for p in HDR_LEVELS:
                col, lw, ls = HDR_STYLES[p]
                self.contours.append(self.ax.contour(self.xa, self.ya, density, levels=[th[p]], colors="black",
                                                     linewidths=lw+1.4, alpha=.55, zorder=7))
                self.contours.append(self.ax.contour(self.xa, self.ya, density, levels=[th[p]], colors=col,
                                                     linewidths=lw, linestyles=ls, zorder=8))
                areas.append((density >= th[p]).sum()*(self.run["cell"]/1000)**2)
            hdr_txt = "\nHDR area 50/80/95%: " + " / ".join(f"{a:.1f}" for a in areas) + " km²"

        self.scat.set_offsets(pts/1000 if show_pts else np.empty((0, 2)))
        mission_txt = ""
        if self.mission is not None:
            mission_txt = self._update_mission(k, t_s, status, remaining)
        fixed = " (planner)" if self.s_bw is None else ""
        self.info.set_text(f"t = {t_s/60:.1f} min | h_s = {h:.0f} m{fixed} | peak = {self.vmax:.2f} /km²"
                           + hdr_txt + mission_txt)
        title = f"{self.run['name']} (map {self.run['map_km']:.0f} km)"
        if self.mission is not None:
            title += f" — {self.mission['name']} — {self.mode}"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def _update_mission(self, k, t_s, status, remaining):
        m = self.mission
        show_swath, show_plan = status[4], status[5]
        self.pos_cursor.set_xdata([t_s/60])
        for u in range(0 if k is None else self.n_uav, len(self.trails)):
            self.trails[u].set_data([], [])
            self.fov_now[u].set_xy(np.empty((0, 2)))
            self.uav_dot[u].set_data([], [])
        if k is None:
            self.swath.set_verts([])
            self.plan_coll.set_verts([])
            self.pos_dot.set_data([], [])
            return f"\nUAV launches at {self.t_launch/60:.0f} min"
        poses = m["poses"]
        L, W = self.fov
        swath, plans = [], []
        for u in range(self.n_uav):
            p = poses[:k+1, u]
            self.trails[u].set_data(p[:, 0]/1000, p[:, 1]/1000)
            x, y, yaw = poses[k, u]
            self.fov_now[u].set_xy(footprint_corners(x, y, yaw, L, W)/1000)
            self.uav_dot[u].set_data([x/1000], [y/1000])
            if show_swath:
                swath += [footprint_corners(px, py, pw, L, W)/1000 for px, py, pw in p[::max(1, k//600)]]
            if show_plan:
                for px, py, pw in m["plan"][k, u]:
                    if np.isfinite(px):
                        plans.append(footprint_corners(px, py, pw, L, W)/1000)
        self.swath.set_verts(swath)
        self.plan_coll.set_verts(plans)
        pos = float(m["pos_cum"][k])
        self.pos_dot.set_data([m["t_s"][k]/60], [pos])
        return f"\nstep {k} | POS = {pos:.3f} | remaining mass = {remaining:.3f}"


# ---------------------------------------------------------------- CLI

def parse_args(mission_mode: bool):
    ap = argparse.ArgumentParser(description=(__doc__ if mission_mode else "Probability density viewer."),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if mission_mode:
        ap.add_argument("path", type=Path,
                        help="Run folder (lists all its search_* missions) or one search folder")
        ap.add_argument("--select", default=None, metavar="NAME", help="Mission shown first, e.g. tree")
        ap.add_argument("--compare", nargs="+", default=[], metavar="NAME",
                        help="Missions also drawn in the POS panel, e.g. --compare greedy straight")
    else:
        ap.add_argument("run", type=Path, help="Result folder or simulation_data.npz")
        ap.add_argument("--bandwidth", type=float, default=150.0, help="Initial bandwidth h_s in metres")
        ap.add_argument("--hmin", type=float, default=25.0)
        ap.add_argument("--hmax", type=float, default=800.0)
    ap.add_argument("--time", type=float, default=None, help="Initial time in minutes (default: last)")
    ap.add_argument("--dt", type=float, default=None, help="Seconds per track step if not stored in the file")
    if mission_mode:
        ap.add_argument("--mode", choices=DENSITY_MODES, default=None, help="Initial density mode")
    ap.add_argument("--snapshot", type=Path, default=None, help="Save a PNG and exit (no window)")
    ap.add_argument("--full", action="store_true", help="Show the whole map instead of cropping")
    ap.add_argument("--margin", type=float, default=2000.0, help="Crop margin (m)")
    return ap.parse_args()


def _resolve_missions(args):
    """Run folder, mission list and the index shown first."""
    path = args.path.parent if args.path.name == "mission.npz" else args.path
    selected = None
    if (path/"mission.npz").exists():
        run_dir, selected = path.parent, path.name.removeprefix("search_")
    else:
        run_dir = path
    missions = find_missions(run_dir)
    if not missions:
        raise SystemExit(f"No search_*/mission.npz in {run_dir}. Fly one first, e.g.\n"
                         f"  python uav_search.py {run_dir} --plan greedy")
    names = [p.parent.name.removeprefix("search_") for p in missions]
    selected = (args.select or selected or names[0]).removeprefix("search_")
    if selected not in names:
        raise SystemExit(f"No mission '{selected}'; available: {', '.join(names)}")
    return run_dir, missions, names.index(selected)


def main(mission_mode: bool):
    args = parse_args(mission_mode)
    if mission_mode:
        run_dir, missions, current = _resolve_missions(args)
        run = load_run(run_dir, args.dt)
        # Particles and launch points only: a baseline flying straight for 2 h would blow up the window.
        pts = [run["track"].reshape(-1, 2)] + [np.load(p)["poses"][0, :, :2] for p in missions]
        pts = np.concatenate(pts)
    else:
        run = load_run(args.run, args.dt)
        pts = run["track"].reshape(-1, 2)
    if not args.full:
        crop_run(run, pts, args.margin)
    if mission_mode:
        v = SimViewer(run, missions, current, args.time)
        if args.mode:
            v.radio.set_active(DENSITY_MODES.index(args.mode))
        for name in args.compare:
            name = name.removeprefix("search_")
            if name in v.names:
                v.chk_cmp.set_active(v.names.index(name))
            else:
                print(f"No mission '{name}' to compare; available: {', '.join(v.names)}")
    else:
        v = SimViewer(run, None, 0, args.time, args.bandwidth, args.hmin, args.hmax)
    if args.snapshot is not None:
        v.fig.savefig(args.snapshot, dpi=140)
        print(f"Saved {args.snapshot.resolve()}")
    else:
        plt.show()
    return v
