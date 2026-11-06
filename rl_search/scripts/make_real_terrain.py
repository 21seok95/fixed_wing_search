"""Build a real-region terrain file (same format as the synthetic one) plus its road graph and figures.

Run from the repository root:
    python -m rl_search.scripts.make_real_terrain --region chungju

Raw downloads are cached in rl_search/data/real/<region>/raw (use --refresh to fetch again).
Outputs: rl_search/data/terrain_real_<region>_60km[_<cell>m].npz (no suffix for 50 m),
rl_search/data/real/<region>/road_graph.npz and figures / stats in rl_search/data/real/<region>/
(file names end in _<cell>m for cells other than 50 m).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import label

from rl_search.plotting import draw_terrain
from rl_search.real_terrain import DATA, HIGHWAY_CODES, REGIONS, build_real_terrain, fetch_raw
from rl_search.terrain import BASE_LAYERS, EXTRA_LAYERS, load_terrain, save_terrain

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FAST_C, SLOW_C, CITY_C = "#111111", "#b5703a", "#d1495b"


def edge_lines(g, mask):
    off, poly = g["edge_poly_offset"], g["edge_poly"]
    return [poly[off[i]:off[i + 1]]/1000 for i in np.flatnonzero(mask)]


def draw_roads(ax, g, lw=(.5, .9)):
    for tier, col, w in ((1, SLOW_C, lw[0]), (2, FAST_C, lw[1])):
        for xy in edge_lines(g, g["edges"]["tier"] == tier):
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=w, solid_capstyle="round", zorder=7)


def draw_city(ax, terr):
    a = terr.axis/1000
    with plt.rc_context({"hatch.color": "#8b1e2d", "hatch.linewidth": .7}):
        ax.contourf(a, a, terr.city.astype(float), levels=[.5, 2], colors=[CITY_C], alpha=.45, zorder=8)
        ax.contourf(a, a, terr.city.astype(float), levels=[.5, 2], colors="none", hatches=["////"], zorder=8)
    ax.contour(a, a, terr.city.astype(float), levels=[.5], colors="#8b1e2d", linewidths=1.1, zorder=8)


def legend_extra():
    return [Line2D([], [], color=FAST_C, lw=2, label="빠른 길 (고속·국도·주요·지방·군도)"),
            Line2D([], [], color=SLOW_C, lw=2, label="느린 길 (마을 간 도로·임도·농로)"),
            Patch(fc=CITY_C, alpha=.45, hatch="////", ec="#8b1e2d", label="큰 도시 (지상 인계)"),
            Patch(color="#c9a0a6", label="그 밖의 시가지 (가림)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="chungju", choices=sorted(REGIONS))
    ap.add_argument("--cell", type=float, default=50.0, help="Grid cell size in metres")
    ap.add_argument("--refresh", action="store_true", help="Download the raw data again")
    args = ap.parse_args()
    region = REGIONS[args.region]
    out_dir = DATA/region.key
    sfx = "" if args.cell == 50 else f"_{args.cell:g}m"
    raw = fetch_raw(region, refresh=args.refresh)
    terr, g, lc = build_real_terrain(region, raw, cell_m=args.cell)

    path = Path("rl_search/data")/f"terrain_real_{region.key}_60km{sfx}.npz"
    save_terrain(path, terr)
    np.savez_compressed(out_dir/f"road_graph{sfx}.npz", **g)
    loaded = load_terrain(path)
    same = all(np.array_equal(getattr(terr, k), getattr(loaded, k))
               for k in BASE_LAYERS + EXTRA_LAYERS + ("forbidden", "cover", "slope_deg"))
    if not same:
        raise SystemExit("Loaded terrain differs from the built one")

    e = g["edges"]
    land = ~(terr.water | terr.fords)
    # River check: with bridges closed the big rivers must split the land; with them open it is one piece
    eight = np.ones((3, 3), bool)
    parts = {}
    for nm, blk in (("bridges_open", terr.water), ("bridges_closed", terr.water | terr.fords)):
        lab, _ = label(~blk, eight)
        a = np.sort(np.bincount(lab.ravel())[1:])[::-1]*terr.cell**2/1e6
        parts[nm] = [round(float(v), 1) for v in a[:3]]
    stats = {
        "region": region.key, "name": region.name_ko, "centre": [region.lat, region.lon],
        "elevation_m_p1_p99": [float(np.percentile(terr.z, 1)), float(np.percentile(terr.z, 99))],
        "forest": float(terr.forest.mean()), "built_outside_city": float(terr.built.mean()),
        "city": float(terr.city.mean()), "water": float(terr.water.mean()),
        "steep_nogo": float(((terr.slope_deg >= terr.cfg.max_slope_deg) & land).mean()),
        "forbidden": float(terr.forbidden.mean()),
        "city_zones": int(label(terr.city)[1]), "bridge_cells": int(terr.fords.sum()),
        "bridge_groups": int(label(terr.fords)[1]),
        "surface_fast": float((terr.surface == 2).mean()), "surface_slow": float((terr.surface == 1).mean()),
        "cell_m": float(terr.cell), "land_parts_km2": parts,
        "graph_nodes": int(len(g["node_xy"])), "graph_edges": int(len(e)),
        "road_km_fast": float(e["length_m"][e["tier"] == 2].sum()/1000),
        "road_km_slow": float(e["length_m"][e["tier"] == 1].sum()/1000),
        "road_km_by_highway": {h: float(e["length_m"][e["highway"] == i].sum()/1000)
                               for i, h in enumerate(HIGHWAY_CODES)},
    }
    (out_dir/f"terrain_stats{sfx}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"Saved {path} ({path.stat().st_size/1e6:.1f} MB), road graph {len(g['node_xy'])} nodes / {len(e)} edges")
    for k, v in stats.items():
        if k != "road_km_by_highway":
            print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    # --------------------------------------------------------- figure 1: overview in the usual terrain style
    fig, ax = plt.subplots(figsize=(12.5, 9.6), layout="constrained")
    draw_terrain(ax, terr, legend_extra=legend_extra())
    a = terr.axis/1000
    ax.contourf(a, a, terr.built.astype(float), levels=[.5, 2], colors=["#c9a0a6"], alpha=.8, zorder=6)
    draw_roads(ax, g)
    draw_city(ax, terr)
    ax.set_title(f"실제 지형 {region.name_ko}: 60 km × 60 km, 중심 {region.lat:.2f}°N {region.lon:.2f}°E, "
                 f"{terr.cell:g} m 격자\n"
                 f"숲 {stats['forest']:.0%}, 물 {stats['water']:.1%}, 급경사 통과 불가 {stats['steep_nogo']:.1%}, "
                 f"큰 도시 {stats['city']:.1%}, 다리 {stats['bridge_groups']}곳")
    fig.savefig(out_dir/f"terrain_overview{sfx}.png", dpi=130)
    plt.close(fig)

    # --------------------------------------------------------- figure 2: layer panels
    hs = LightSource(azdeg=315, altdeg=45).hillshade(terr.z, vert_exag=2, dx=terr.cell, dy=terr.cell)
    ext = (a[0], a[-1], a[0], a[-1])
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), layout="constrained")
    panels = [
        ("고도 (m)", terr.z, "terrain", None),
        ("경사 (°), 검정 = 32° 이상 통과 불가", np.minimum(terr.slope_deg, 45), "Greens", terr.slope_deg >= terr.cfg.max_slope_deg),
        ("가림 정도 cover (숲 + 시가지, 0–1)", terr.cover, "YlGn", None),
        ("통과 불가 forbidden (물 + 급경사)", terr.forbidden.astype(float), "Greys", None),
        ("surface: 0 길 없음, 1 느린 길, 2 빠른 길", terr.surface.astype(float), None, None),
        ("도로 그래프 + 큰 도시 + 다리", None, None, None),
    ]
    for ax, (ttl, fld, cm, over) in zip(axs.ravel(), panels):
        ax.imshow(hs, origin="lower", extent=ext, cmap="gray", vmin=-.3, vmax=1.2)
        if fld is not None and cm is not None:
            im = ax.imshow(fld, origin="lower", extent=ext, cmap=cm, alpha=.75, interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=.75)
        elif fld is not None:
            from matplotlib.colors import ListedColormap, BoundaryNorm
            m = np.ma.masked_where(terr.surface == 0, terr.surface)
            ax.imshow(m, origin="lower", extent=ext, cmap=ListedColormap([SLOW_C, FAST_C]),
                      norm=BoundaryNorm([.5, 1.5, 2.5], 2), interpolation="nearest")
        else:
            ax.contourf(a, a, terr.water.astype(float), levels=[.5, 2], colors=["#2f80ed"], zorder=5)
            draw_roads(ax, g, lw=(.35, .7))
            draw_city(ax, terr)
            lab, nb = label(terr.fords)
            if nb:
                from scipy.ndimage import center_of_mass
                cy, cx = np.array(center_of_mass(lab > 0, lab, range(1, nb + 1))).T
                ax.plot(a[0] + cx*terr.cell/1000, a[0] + cy*terr.cell/1000, "D", ms=4, mfc="#ff6f00", mec="white",
                        mew=.5, ls="", zorder=9)
        if over is not None:
            ax.contourf(a, a, over.astype(float), levels=[.5, 2], colors=["black"], zorder=4)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal"); ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    fig.suptitle(f"{region.name_ko} 지형 레이어 ({path.name})", fontsize=14)
    fig.savefig(out_dir/f"terrain_layers{sfx}.png", dpi=110)
    plt.close(fig)

    # --------------------------------------------------------- figure 3: river crossings close-up
    wins = ([(-20, -4, -14, 2, "충주 시내 남한강"), (0, 16, -16, 0, "충주호")] if region.key == "chungju"
            else [(-16, 0, -8, 8, "중심 서쪽"), (0, 16, -8, 8, "중심 동쪽")])
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.4), layout="constrained")
    for ax, (x0, x1, y0, y1, nm) in zip(axs, wins):
        draw_terrain(ax, terr, window=(x0*1000, x1*1000, y0*1000, y1*1000), legend=False)
        ax.contourf(a, a, terr.built.astype(float), levels=[.5, 2], colors=["#c9a0a6"], alpha=.8, zorder=6)
        draw_roads(ax, g, lw=(.8, 1.3))
        draw_city(ax, terr)
        ax.set_title(f"{nm}: 주황 마름모 = 다리 칸 (물이지만 건널 수 있음)")
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
    fig.savefig(out_dir/f"terrain_crossings{sfx}.png", dpi=120)
    plt.close(fig)
    print(f"Figures in {out_dir}")


if __name__ == "__main__":
    main()
