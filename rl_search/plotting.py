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
