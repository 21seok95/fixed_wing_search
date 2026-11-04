"""Figures of the clean centre scenarios (compare_center) over terrain and particles, from the saved results (no
planning rerun). Rows: 5 UAVs with / without separation; columns: snapshots (minutes after handover). Each panel:
terrain (no-go dark, water blue, forest green), particles at that time coloured by the remaining probability after
all UAVs' searching so far (CPA + LRC replayed on the saved paths, 10 s), UAV tracks of the last 10 min, UAV
positions, and pairs closer than 300 m within +-5 min (red).

    python -m rl_search.scripts.plot_center_terrain --seeds 405 408 409 402
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
import numpy as np

COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c0392b"]


def terrain_rgb(terr):
    img = np.ones(terr.forbidden.shape + (3,))
    img[terr.forest.astype(bool)] = (0.80, 0.90, 0.78)
    img[terr.water.astype(bool)] = (0.70, 0.82, 0.95)
    img[terr.forbidden.astype(bool) & ~terr.water.astype(bool)] = (0.55, 0.55, 0.55)
    return img


def remaining_weights(env, scen, paths, times, cut_s):
    """Shared remaining weight per particle at each cut time: product over UAVs of the CPA + LRC survival along the
    saved paths (each UAV its own pass state), exits zeroed."""
    from rl_search.mission import ParticleBelief
    out = {}
    survs = []
    for xy, tt in zip(paths, times):
        env.belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
        env._pass_p, env._last_in = np.zeros(env.belief.n), np.full(env.belief.n, -np.inf)
        seq = []
        k = 0
        for c in sorted(cut_s):
            while k + 1 < len(tt) and tt[k + 1] <= c:
                env._detect_lrc(xy[k:k + 2], tt[k:k + 2])
                k += 1
            seq.append(env.belief.w.copy())
        survs.append(seq)
    exit_t = np.where(scen.exit_step >= 0, scen.exit_step*scen.save_dt_s, np.inf)
    for i, c in enumerate(sorted(cut_s)):
        w = np.prod([s[i] for s in survs], axis=0)
        out[c] = np.where(exit_t > c, w, 0.0)
    return out


def close_with_time(xy, tt, d_safe=300.0):
    pts = []
    n = len(xy)
    for a in range(n):
        for b in range(a + 1, n):
            t0, t1 = max(tt[a][0], tt[b][0]), min(tt[a][-1], tt[b][-1])
            if t1 <= t0:
                continue
            g = np.arange(t0, t1, 5.0)
            pa = np.stack([np.interp(g, tt[a], xy[a][:, 0]), np.interp(g, tt[a], xy[a][:, 1])], 1)
            pb = np.stack([np.interp(g, tt[b], xy[b][:, 0]), np.interp(g, tt[b], xy[b][:, 1])], 1)
            m = np.hypot(*(pa - pb).T) < d_safe
            pts += [(t, *((p + q)/2)) for t, p, q in zip(g[m], pa[m], pb[m])]
    return np.array(pts).reshape(-1, 3)


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.patterns import _rot
    from rl_search.scripts.check_multi import make_envs
    from rl_search.scripts.compare_center import center_scenario

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("rl_search/data/compare_center"))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--minutes", type=float, nargs="+", default=[15, 30, 45])
    args = ap.parse_args()
    res = json.loads((args.input/"results.json").read_text())
    by = {(r["seed"], r["mode"]): r for r in res}
    seeds = args.seeds or sorted({r["seed"] for r in res})
    env = make_envs(1, seeds[0])[0]
    terr = env.terr
    rgb = terrain_rgb(terr)
    ext = [terr.axis[0]/1000, terr.axis[-1]/1000, terr.axis[0]/1000, terr.axis[-1]/1000]
    t_h = env.preset.handover.t_plan_s
    for s in seeds:
        modes = [m for m in ("5 sep", "5 no sep") if (s, m) in by]
        if not modes:
            continue
        scen = center_scenario(env.sampler, s)
        cuts = [t_h + m*60 for m in args.minutes]
        fig, axs = plt.subplots(len(modes), len(cuts), figsize=(5.6*len(cuts), 5.6*len(modes)), squeeze=False)
        for i, m in enumerate(modes):
            r = by[(s, m)]
            xy = [np.array(x, float) for x in r["xy"]]
            tt = [np.array(t, float) for t in r["tt"]]
            W = remaining_weights(env, scen, xy, tt, cuts)
            cl = close_with_time(xy, tt)
            for j, c in enumerate(cuts):
                ax = axs[i, j]
                ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
                P = np.asarray(scen.track[min(int(c/scen.save_dt_s), len(scen.track) - 1)], float)
                w = W[c]
                k = w > 1e-3
                order = np.argsort(w[k])
                ax.scatter(P[k][order, 0]/1000, P[k][order, 1]/1000, c=w[k][order], s=2, cmap="magma_r", vmin=0, vmax=1,
                           alpha=.8, linewidths=0)
                for u in range(len(xy)):
                    sel = (tt[u] <= c) & (tt[u] >= c - 600)
                    ax.plot(xy[u][sel, 0]/1000, xy[u][sel, 1]/1000, color=COLS[u % 5], lw=1.2)
                    if (tt[u] <= c).any():
                        p = xy[u][np.flatnonzero(tt[u] <= c)[-1]]
                        ax.plot(p[0]/1000, p[1]/1000, "o", color=COLS[u % 5], ms=7, mec="k")
                    for tp, cx, cy, ell, wid, th, hs in r["rects"][u]:
                        if tp <= c < tp + hs + 1500:
                            corners = np.array([[sx*ell/2, sy*wid/2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                            ax.add_patch(Polygon((np.array([cx, cy]) + corners @ _rot(th).T)/1000, fill=False,
                                                 ec=COLS[u % 5], lw=1, ls="--"))
                near = cl[np.abs(cl[:, 0] - c) <= 300] if len(cl) else cl
                if len(near):
                    ax.scatter(near[:, 1]/1000, near[:, 2]/1000, s=25, c="red", zorder=6)
                live = P[(w > 0.02)]
                pts = np.concatenate([live, *[x[(t <= c) & (t >= c - 600)] for x, t in zip(xy, tt)]]) if len(live) else P
                lo, hi = np.quantile(pts, .01, axis=0) - 1500, np.quantile(pts, .99, axis=0) + 1500
                half = max(hi - lo)/2
                mid = (lo + hi)/2
                ax.set(xlim=((mid[0] - half)/1000, (mid[0] + half)/1000), ylim=((mid[1] - half)/1000, (mid[1] + half)/1000),
                       title=f"{m}, {(c - t_h)/60:.0f} min: remaining {w.sum()/len(w):.2f}, close +-5 min: {len(near)}",
                       xlabel="km", ylabel="km")
        fig.suptitle(f"scenario {s}: terrain (grey no-go, blue water, green forest), particles by remaining probability "
                     "(dark = high), last 10 min of tracks, dashed = current rectangles, red = pair < 300 m within +-5 min",
                     fontsize=10)
        fig.tight_layout()
        f = args.input/f"terrain_s{s}.png"
        fig.savefig(f, dpi=100)
        plt.close(fig)
        print(f, flush=True)


if __name__ == "__main__":
    main()
