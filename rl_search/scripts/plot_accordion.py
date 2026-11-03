"""Figures of the accordion planner (docs/sarops_approach_todo.md S.10): for each 10-min cycle, the remaining
probability at mid-cycle, the rectangle chosen by the accordion search, its planned lanes and the path flown.

    python -m rl_search.scripts.plot_accordion --seeds 301 302 --sensor new --detect lrc
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import numpy as np


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.accordion import AccordionPlanner
    from rl_search.config import get_preset
    from rl_search.patterns import _rot, run_episode
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scenario import make_scenario
    from rl_search.scripts.check_accordion import SENSORS
    from rl_search.terrain import load_terrain

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302])
    ap.add_argument("--sensor", choices=tuple(SENSORS), default="new")
    ap.add_argument("--detect", choices=("kernel", "lrc"), default="lrc")
    ap.add_argument("--variant", default="tavg_pca")
    ap.add_argument("--cell", type=float, default=200.0)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--n-starts", type=int, default=1, help="accordion starts over this many density peaks")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/accordion_plots"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    for seed in args.seeds:
        env = RLSearchEnv(terr, get_preset("rc_cessna", SENSORS[args.sensor]),
                          RLConfig(stage=3, detect_model=args.detect), seed=seed)
        env.fast = True
        snaps = []

        def hook(env, _):
            b = env.belief
            snaps.append((env.t, b.positions(env.t + 300.0).copy(), b.w.copy()/b.n, env.pos_cum))

        pol = AccordionPlanner(args.variant, cell=args.cell, hook=hook, n_starts=args.n_starts)
        log = run_episode(env, dict(scenario=make_scenario(env.sampler, seed), stage=3), pol)
        t, xy, pos = np.array(log.t), np.array(log.xy), np.array(log.pos)
        n = min(args.cycles, len(pol.rects))
        fig, axs = plt.subplots(2, (n + 2)//2, figsize=(5.2*((n + 2)//2), 10.4), squeeze=False)
        axs = axs.ravel()
        # overview
        ax = axs[0]
        P0, W0 = snaps[0][1], snaps[0][2]
        ax.plot(xy[:, 0]/1000, xy[:, 1]/1000, "k-", lw=.8)
        for (tp, r, L), c in zip(pol.rects, plt.cm.viridis(np.linspace(0, 1, len(pol.rects)))):
            corners = np.array([[sx*r.ell/2, sy*r.wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            ax.add_patch(Polygon((np.array([r.cx, r.cy]) + corners @ _rot(r.theta).T)/1000, fill=False, ec=c, lw=1.5))
        ax.plot(*np.array(env.scen.lkp)/1000, "r*", ms=12, label="LKP")
        ax.plot(*xy[0]/1000, "ko", ms=5, label="launch")
        ax.set(title=f"seed {seed}: whole flight, POS {pos[-1]:.3f}", xlabel="km", ylabel="km", aspect="equal")
        ax.legend(loc="best", fontsize=8)
        for k in range(n):
            ax = axs[k + 1]
            tp, r, L = pol.rects[k]
            _, P, Wt, pos0 = snaps[k]
            t_end = pol.rects[k + 1][0] if k + 1 < len(pol.rects) else t[-1]
            m = (t >= tp) & (t <= t_end)
            seg = xy[m]
            gain = np.interp(t_end, t, pos) - np.interp(tp, t, pos)
            c = np.array([r.cx, r.cy])
            half = max(r.ell, r.wid)/2 + 2500
            lo, hi = c - half, c + half
            ok = Wt > 0
            H, ex, ey = np.histogram2d(P[ok, 0], P[ok, 1], bins=80, range=[[lo[0], hi[0]], [lo[1], hi[1]]], weights=Wt[ok])
            cell_km2 = (ex[1] - ex[0])*(ey[1] - ey[0])/1e6
            im = ax.imshow(H.T/cell_km2, origin="lower", extent=[lo[0]/1000, hi[0]/1000, lo[1]/1000, hi[1]/1000],
                           cmap="Blues", aspect="equal")
            fig.colorbar(im, ax=ax, fraction=.04, label="remaining probability / km²")
            corners = np.array([[sx*r.ell/2, sy*r.wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            ax.add_patch(Polygon((c + corners @ _rot(r.theta).T)/1000, fill=False, ec="red", lw=2, label="accordion rectangle"))
            ax.plot(L[:, 0]/1000, L[:, 1]/1000, color="orange", lw=.8, ls="--", label="planned lanes")
            ax.plot(seg[:, 0]/1000, seg[:, 1]/1000, "k-", lw=1.2, label="flown")
            ax.plot(*seg[0]/1000, "go", ms=6, label="cycle start")
            ax.set(xlim=(lo[0]/1000, hi[0]/1000), ylim=(lo[1]/1000, hi[1]/1000), xlabel="km",
                   title=f"cycle {k + 1}: t {tp/60:.0f}-{t_end/60:.0f} min, {r.ell/1000:.1f} x {r.wid/1000:.1f} km\n"
                         f"approx POS {r.approx:.3f}, simulated gain {gain:.3f}")
            if k == 0:
                ax.legend(loc="lower left", fontsize=7)
        for a in axs[n + 1:]:
            a.axis("off")
        fig.suptitle(f"Accordion planner ({args.variant}, {args.cell:.0f} m grid, starts {args.n_starts}), sensor {args.sensor}, detection {args.detect}; "
                     "probability shown at mid-cycle", fontsize=12)
        fig.tight_layout()
        f = args.out/(f"accordion_s{seed}_{args.sensor}_{args.detect}" + (f"_starts{args.n_starts}" if args.n_starts > 1 else "") + ".png")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        print(f)


if __name__ == "__main__":
    main()
