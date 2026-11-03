"""Terrain background plots: shaded relief with no-go areas, rivers, fords and forest."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, center_of_mass, gaussian_filter, label

from rl_search.terrain import Terrain

RELIEF = LinearSegmentedColormap.from_list(
    "relief", ["#e4ecd2", "#cfdcab", "#e8dcae", "#d4b483", "#ab865e", "#8c6b52"])
NO_GO = "#b71c1c"
RIVER = "#1565c0"
STREAM = "#5aa0e6"
FORD = "#ff6f00"
FOREST = "#1b5e20"


def draw_terrain(ax, terr: Terrain, legend_extra=(), window=None, legend=True):
    """Draw terrain layers; only the grid inside ``window`` (x0, x1, y0, y1 in metres) is rendered."""
    if window is None:
        window = (terr.axis[0], terr.axis[-1], terr.axis[0], terr.axis[-1])
    x0, x1, y0, y1 = window
    ci = np.flatnonzero((terr.axis >= x0 - 2*terr.cell) & (terr.axis <= x1 + 2*terr.cell))
    ri = np.flatnonzero((terr.axis >= y0 - 2*terr.cell) & (terr.axis <= y1 + 2*terr.cell))
    rs, cs = slice(ri[0], ri[-1]+1), slice(ci[0], ci[-1]+1)
    ax_, ay_ = terr.axis[cs]/1000, terr.axis[rs]/1000
    extent = (ax_[0], ax_[-1], ay_[0], ay_[-1])
    crop = lambda f: f[rs, cs]
    z = crop(terr.z)

    rgb = LightSource(azdeg=315, altdeg=45).shade(
        z, cmap=RELIEF, blend_mode="soft", vert_exag=2, dx=terr.cell, dy=terr.cell,
        vmin=float(np.quantile(terr.z, .01)), vmax=1.15*terr.mountain_z_m)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="bilinear", zorder=0)
    ax.contour(ax_, ay_, z, levels=np.arange(100, 2000, 100), colors="#5d4037", linewidths=.25,
               alpha=.35, zorder=1)
    ax.contourf(ax_, ay_, gaussian_filter(crop(terr.forest).astype(float), 1.0), levels=[.5, 2],
                colors=[FOREST], alpha=.33, zorder=2)

    mtn = crop(terr.mountain).astype(float)
    with plt.rc_context({"hatch.color": NO_GO, "hatch.linewidth": .6}):
        ax.contourf(ax_, ay_, mtn, levels=[.5, 1.5], colors=[NO_GO], alpha=.12, zorder=3)
        ax.contourf(ax_, ay_, mtn, levels=[.5, 1.5], colors="none", hatches=["////"], zorder=3)
    ax.contour(ax_, ay_, mtn, levels=[.5], colors=NO_GO, linewidths=.9, zorder=3)

    stream = gaussian_filter(binary_dilation(crop(terr.stream)).astype(float), 1.0)
    ax.contourf(ax_, ay_, stream, levels=[.35, 2], colors=[STREAM], alpha=.9, zorder=4)
    ax.contourf(ax_, ay_, gaussian_filter(crop(terr.water).astype(float), .8), levels=[.5, 2],
                colors=[RIVER], zorder=5)
    lab, n_fords = label(crop(terr.fords))
    if n_fords:
        fy, fx = np.array(center_of_mass(lab > 0, lab, range(1, n_fords+1))).T
        ax.scatter(ax_[0] + fx*terr.cell/1000, ay_[0] + fy*terr.cell/1000, marker="D", s=26,
                   color=FORD, edgecolor="white", linewidth=.7, zorder=6)

    if legend:
        handles = [Patch(facecolor=NO_GO, alpha=.35, hatch="////", edgecolor=NO_GO,
                         label=f"Mountain no-go (≥{terr.mountain_z_m:.0f} m)"),
                   Patch(color=RIVER, label="River (impassable)"),
                   Line2D([], [], color=STREAM, lw=1.5, label="Stream"),
                   Line2D([], [], marker="D", color=FORD, markeredgecolor="white", ls="", markersize=6,
                          label="Ford"),
                   Patch(color=FOREST, alpha=.4, label="Forest (cover)"),
                   *legend_extra]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=.95)
    ax.set(xlabel="x (km)", ylabel="y (km)", xlim=(x0/1000, x1/1000), ylim=(y0/1000, y1/1000))
    ax.set_aspect("equal")
    return extent


def _rl_window(belief, paths, times, margin=1500.0):
    """Square window (x0, x1, y0, y1) around the paths and the 1-99 % particle range at ``times``."""
    pts = list(paths)
    for t in times:
        p = belief.positions(t)[belief.active(t)]
        if len(p):
            pts.append(np.quantile(p, [.01, .99], axis=0))
    pts = np.vstack(pts)
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    half, (cx, cy) = max(hi - lo)/2, (lo + hi)/2
    return cx - half, cx + half, cy - half, cy + half


def _rl_panel(ax, terr, belief, path_xy, path_t, t, w, window, tnorm, seen_cmap, vmax=.5):
    """One time slice: faded terrain, particles at ``t`` (grey: unseen, red: searched share), path up to ``t``."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    draw_terrain(ax, terr, window=window, legend=False)
    ax.add_patch(Rectangle((window[0]/1000, window[2]/1000), (window[1] - window[0])/1000, (window[3] - window[2])/1000,
                           color="white", alpha=.55, zorder=6.5, lw=0))
    act = belief.active(t)
    p, ww = belief.positions(t)[act]/1000, w[act]
    unseen = ww >= .98
    ax.scatter(p[unseen, 0], p[unseen, 1], s=3, color="#6b6b6b", alpha=.45, lw=0, zorder=7)
    o = np.flatnonzero(~unseen)
    o = o[np.argsort(-ww[o])]
    sc = ax.scatter(p[o, 0], p[o, 1], s=7, c=1 - ww[o], cmap=seen_cmap, vmin=0, vmax=vmax, lw=0, zorder=8)
    k = int(np.searchsorted(path_t, t, side="right"))
    xy = path_xy[:k]/1000
    if len(xy) > 1:
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        ax.plot(xy[:, 0], xy[:, 1], color="black", lw=3.6, solid_capstyle="round", zorder=9)
        lc = LineCollection(seg, cmap="cool", norm=tnorm, lw=2.2, capstyle="round", zorder=10)
        lc.set_array(path_t[1:k]/60)
        ax.add_collection(lc)
    ax.plot(*xy[0], "o", ms=8, color="white", mec="black", mew=1.5, zorder=11)
    ax.plot(*xy[-1], "*", ms=17, color="yellow", mec="black", mew=1.2, zorder=11)
    return sc


def draw_rl_episode(terr: Terrain, belief, path_xy: np.ndarray, path_t: np.ndarray, snaps, title: str = ""):
    """RL episode in time slices: one panel per ``snaps`` entry (t_s, particle weights at t_s, POS, area km²).
    Each panel shows the particles where they are at t_s (grey: unseen, red: share already searched
    away) and the path flown up to t_s, coloured by time, on faded terrain."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    t0, t1 = float(path_t[0]), float(path_t[-1])
    seen_cmap = LinearSegmentedColormap.from_list("seen", plt.cm.Reds(np.linspace(.35, 1, 256)))
    tnorm = Normalize(t0/60, t1/60)
    window = _rl_window(belief, [path_xy], [s[0] for s in snaps])

    fig, axes = plt.subplots(1, len(snaps), figsize=(5.2*len(snaps) + 1.2, 5.6), squeeze=False,
                             constrained_layout=True)
    for ax, (t, w, pos, area) in zip(axes[0], snaps):
        sc = _rl_panel(ax, terr, belief, path_xy, path_t, t, w, window, tnorm, seen_cmap)
        ax.set_title(f"t = {t/60:.0f} min:  POS {pos:.3f},  area {area:.2f} km²", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(tnorm, "cool"), ax=axes[0, -1], fraction=.05, pad=.02,
                 label="flight time (min)")
    fig.colorbar(sc, ax=axes[0, -1], fraction=.05, pad=.08, extend="max",
                 label="searched share of particle (1 − weight)")
    handles = [Line2D([], [], marker="o", ls="", color="white", mec="black", label="start"),
               Line2D([], [], marker="*", ls="", ms=12, color="yellow", mec="black", label="UAV now"),
               Line2D([], [], marker="o", ls="", color="#6b6b6b", alpha=.6, label="unseen particle")]
    axes[0, 0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=.9)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def draw_rl_compare(terr: Terrain, belief, runs: dict, title: str = ""):
    """Methods on one scenario: a row per method, a column per time slice, same window. ``runs``: name ->
    (path_xy, path_t, snaps) with snaps as (t_s, particle weights at t_s, POS)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    seen_cmap = LinearSegmentedColormap.from_list("seen", plt.cm.Reds(np.linspace(.35, 1, 256)))
    paths = [r[0] for r in runs.values()]
    ts = [r[1] for r in runs.values()]
    tnorm = Normalize(min(t[0] for t in ts)/60, max(t[-1] for t in ts)/60)
    window = _rl_window(belief, paths, [s[0] for r in runs.values() for s in r[2]])
    ncol = max(len(r[2]) for r in runs.values())
    fig, axes = plt.subplots(len(runs), ncol, figsize=(4.3*ncol + 1.3, 4.3*len(runs)), squeeze=False,
                             constrained_layout=True)
    for row, (name, (path_xy, path_t, snaps)) in zip(axes, runs.items()):
        for ax, (t, w, pos) in zip(row, snaps):
            sc = _rl_panel(ax, terr, belief, path_xy, path_t, t, w, window, tnorm, seen_cmap, vmax=1.0)   # one pass: POD 0.7
            ax.set_title(f"{name},  t = {t/60:.0f} min:  POS {pos:.3f}", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(tnorm, "cool"), ax=axes[:, -1], fraction=.05, pad=.02, label="flight time (min)")
    fig.colorbar(sc, ax=axes[:, -1], fraction=.05, pad=.06, label="searched share of particle (1 − weight)")
    if title:
        fig.suptitle(title, fontsize=13)
    return fig
