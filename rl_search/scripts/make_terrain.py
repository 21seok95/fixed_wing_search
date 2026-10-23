"""Generate the fixed 60 km search terrain once and save it with an overview figure.

Run from the repository root:
    python -m rl_search.scripts.make_terrain
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label

from rl_search.plotting import draw_terrain
from rl_search.terrain import BASE_LAYERS, TerrainConfig, load_terrain, make_terrain, save_terrain


def main():
    d = TerrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--map-km", type=float, default=d.map_m/1000)
    ap.add_argument("--cell", type=float, default=d.cell_m, help="Grid cell size in metres")
    ap.add_argument("--output", type=Path, default=None,
                    help="Default: rl_search/data/terrain_seed<seed>_<map>km.npz")
    args = ap.parse_args()
    cfg = TerrainConfig(seed=args.seed, map_m=args.map_km*1000, cell_m=args.cell)
    out = args.output or Path("rl_search/data")/f"terrain_seed{cfg.seed}_{args.map_km:g}km.npz"

    t0 = time.perf_counter()
    terr = make_terrain(cfg)
    save_terrain(out, terr)
    t_make = time.perf_counter() - t0
    t0 = time.perf_counter()
    loaded = load_terrain(out)
    t_load = time.perf_counter() - t0
    same = all(np.array_equal(getattr(terr, k), getattr(loaded, k))
               for k in BASE_LAYERS + ("forbidden", "clearance", "cover", "slope_deg"))
    if not same:
        raise SystemExit("Loaded terrain differs from the generated one")

    free = ~terr.forbidden
    print(f"Saved {out} ({out.stat().st_size/1e6:.1f} MB) | grid {terr.z.shape} at {terr.cell:g} m")
    print(f"Generate {t_make:.1f} s | load + derive {t_load:.1f} s | round trip identical")
    print(f"Elevation {terr.z.min():.0f}-{terr.z.max():.0f} m | no-go elevation {terr.mountain_z_m:.0f} m")
    print(f"Mountain {terr.mountain.mean():.1%} | river {terr.water.mean():.1%} | "
          f"steep {((terr.slope_deg >= cfg.max_slope_deg) & ~terr.mountain & ~terr.water).mean():.1%} | "
          f"no-go total {terr.forbidden.mean():.1%}")
    print(f"Fords {label(terr.fords)[1]} | forest share of passable land {terr.forest[free].mean():.1%}")

    fig, ax = plt.subplots(figsize=(11, 8.6), layout="constrained")
    draw_terrain(ax, terr)
    ax.set_title(f"Search terrain {args.map_km:g} km x {args.map_km:g} km (seed {cfg.seed})")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
