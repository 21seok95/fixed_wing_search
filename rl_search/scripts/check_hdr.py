"""How far the target prior spreads, independent of the UAV: HDR areas and the expanded reward zone over time.

For each scenario the unweighted particle prediction (RL TODO 1.1 v3: no observation weights, exited
particles dropped) is turned into a terrain-masked KDE on the terrain grid. The figure shows, at a few
times, the density with the 80/90/95 % HDR contours and the expanded zone (HDR grown by a margin that
grows with time and is capped), and the zone areas over time against what the UAV can still cover.

    python -m rl_search.scripts.check_hdr                       # 3 new scenarios, defaults below
    python -m rl_search.scripts.check_hdr --seeds 1 2 3 4 --hdr 0.9 --margin-rate 0.1 --margin-max 1000
    python -m rl_search.scripts.check_hdr --scenario rl_search/data/scenarios/rc_cessna/eval/s1000000

Expanded zone margin: ``min(margin_rate * target_speed * t, margin_max)`` metres (t after the LKP).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import distance_transform_edt, gaussian_filter

from rl_search.config import add_preset_args, get_preset
from rl_search.scenario import ScenarioSampler, load_scenario, make_scenario
from rl_search.terrain import load_terrain

LEVELS = (0.80, 0.90, 0.95)
LEVEL_STYLE = {0.80: ("#ffffff", "-"), 0.90: ("#ffd166", "-"), 0.95: ("#ef476f", "-")}


def density(terr, scen, t_s, h_m):
    """Terrain-masked KDE (cells, normalised to the in-map active mass) of the unweighted prediction at t_s."""
    k = int(round(np.clip(t_s/scen.save_dt_s, 0, len(scen.track)-1)))
    pts = np.asarray(scen.track[k], float)
    active = (scen.exit_step < 0) | (scen.exit_step > k)
    cell = float(terr.axis[1] - terr.axis[0])
    n = len(terr.axis)
    ij = np.clip(np.round((pts[active] - terr.axis[0])/cell).astype(int), 0, n-1)
    hist = np.zeros((n, n))
    np.add.at(hist, (ij[:, 1], ij[:, 0]), 1.0)
    dens = gaussian_filter(hist, h_m/cell, mode="constant")
    dens[terr.forbidden] = 0
    s = dens.sum()
    return (dens/s if s > 0 else dens), active.mean()


def hdr_mask(dens, p):
    v = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(v)
    thr = v[min(np.searchsorted(cum, p*cum[-1]), len(v)-1)]
    return (dens >= thr) & (dens > 0)


def expand(mask, margin_m, cell, passable):
    if margin_m <= 0:
        return mask
    return (distance_transform_edt(~mask)*cell <= margin_m) & passable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_preset_args(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13], help="New scenarios from these seeds")
    ap.add_argument("--scenario", type=Path, nargs="*", default=[], help="Saved scenario directories instead")
    ap.add_argument("--hdr", type=float, default=0.95, help="HDR level of the expanded zone")
    ap.add_argument("--margin-rate", type=float, default=0.1, help="Margin growth as a share of target speed")
    ap.add_argument("--margin-max", type=float, default=1000.0, help="Margin cap (m)")
    ap.add_argument("--times", type=float, nargs="+", default=[120, 600, 1200, 2400, 3600])
    ap.add_argument("--fp-width", type=float, default=115.0, help="Footprint cross-track width (m)")
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/hdr_check"))
    args = ap.parse_args()

    args.preset = args.preset or "rc_cessna"
    preset = get_preset(args.preset, args.set)
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    cell = float(terr.axis[1] - terr.axis[0])
    passable = ~terr.forbidden
    h_m = preset.search.bandwidth_m
    v_uav = preset.search.uav_speed_mps
    t_end = preset.energy.budget_s  # energy runs out at about this time in straight trim flight
    scens = [load_scenario(p) for p in args.scenario]
    if not scens:
        sampler = ScenarioSampler(terr, preset.scenario)
        scens = [make_scenario(sampler, s) for s in args.seeds]

    args.output.mkdir(parents=True, exist_ok=True)
    t_curve = np.arange(preset.handover.t_plan_s, t_end + 1, 60.0)
    summary = {}
    fig, axes = plt.subplots(len(scens), len(args.times) + 1, figsize=(4.0*(len(args.times)+1), 4.0*len(scens)),
                             squeeze=False)
    for r, scen in enumerate(scens):
        margin = lambda t: min(args.margin_rate*scen.speed_mps*t, args.margin_max)
        for c, t in enumerate(args.times):
            ax = axes[r, c]
            dens, in_map = density(terr, scen, t, h_m)
            masks = {p: hdr_mask(dens, p) for p in LEVELS}
            zone = expand(hdr_mask(dens, args.hdr), margin(t), cell, passable)
            ys, xs = np.nonzero(zone | (dens > 0))
            pad = 20
            r0, r1 = max(ys.min()-pad, 0), min(ys.max()+pad, len(terr.axis)-1)
            c0, c1 = max(xs.min()-pad, 0), min(xs.max()+pad, len(terr.axis)-1)
            ext = [terr.axis[c0]/1e3, terr.axis[c1]/1e3, terr.axis[r0]/1e3, terr.axis[r1]/1e3]
            sub = dens[r0:r1+1, c0:c1+1]
            pos = sub[sub > 0]
            ax.imshow(np.where(terr.forbidden[r0:r1+1, c0:c1+1], 1, np.nan), origin="lower", extent=ext,
                      cmap="Greys", vmin=0, vmax=3, alpha=.6)
            if pos.size:
                ax.imshow(np.ma.masked_less_equal(sub, 0), origin="lower", extent=ext, cmap="viridis",
                          norm=LogNorm(vmin=pos.max()*1e-3, vmax=pos.max()))
            X, Y = np.meshgrid(terr.axis[c0:c1+1]/1e3, terr.axis[r0:r1+1]/1e3)
            for p, m in masks.items():
                col, ls = LEVEL_STYLE[p]
                ax.contour(X, Y, m[r0:r1+1, c0:c1+1].astype(float), levels=[.5], colors=col, linestyles=ls,
                           linewidths=1.0)
            ax.contour(X, Y, zone[r0:r1+1, c0:c1+1].astype(float), levels=[.5], colors="#06d6a0",
                       linestyles="--", linewidths=1.2)
            a = {p: m.sum()*cell**2/1e6 for p, m in masks.items()}
            ax.set_title(f"t={t/60:.0f} min  in-map {in_map:.0%}\n"
                         + "  ".join(f"{int(p*100)}%:{a[p]:.0f}" for p in LEVELS)
                         + f"  zone:{zone.sum()*cell**2/1e6:.0f} km²", fontsize=8)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        # area curves
        ax = axes[r, -1]
        curves = {p: [] for p in LEVELS}
        zone_curve = []
        for t in t_curve:
            dens, _ = density(terr, scen, t, h_m)
            for p in LEVELS:
                curves[p].append(hdr_mask(dens, p).sum()*cell**2/1e6)
            zone_curve.append(expand(hdr_mask(dens, args.hdr), margin(t), cell, passable).sum()*cell**2/1e6)
        tm = t_curve/60
        for p in LEVELS:
            ax.plot(tm, curves[p], color=LEVEL_STYLE[p][0] if p != .8 else "#555555", label=f"HDR {int(p*100)}%")
        ax.plot(tm, zone_curve, "--", color="#06d6a0", label=f"zone ({int(args.hdr*100)}% + margin)")
        cover = v_uav*args.fp_width*(t_end - t_curve)/1e6
        ax.plot(tm, cover, ":", color="k", label="UAV can still cover (no overlap)")
        ax.set_yscale("log")
        ax.set_xlabel("time after LKP (min)", fontsize=8)
        ax.set_ylabel("km²", fontsize=8)
        ax.set_title(f"seed {scen.seed}: {scen.speed_mps*3.6:.0f} km/h, margin max {args.margin_max:.0f} m",
                     fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=.3, which="both")
        summary[int(scen.seed)] = dict(speed_kmh=round(scen.speed_mps*3.6, 1),
                                       **{f"hdr{int(p*100)}_km2": [round(v, 1) for v in curves[p][::10]]
                                          for p in LEVELS},
                                       zone_km2=[round(v, 1) for v in zone_curve[::10]],
                                       t_min=[round(v) for v in tm[::10]])
    fig.suptitle(f"Unweighted prior spread ({args.preset}, KDE h={h_m:.0f} m). White/yellow/red = HDR 80/90/95 %, "
                 f"green dashed = expanded zone (HDR {int(args.hdr*100)}% + min({args.margin_rate}·v·t, "
                 f"{args.margin_max:.0f} m)). Grey = no-go.", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output/"hdr_check.png", dpi=100)
    (args.output/"hdr_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("saved", args.output/"hdr_check.png")


if __name__ == "__main__":
    main()
