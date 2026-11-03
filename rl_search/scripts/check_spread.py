"""How the target belief spreads, and what each spread mechanism contributes (docs/sarops_approach_todo.md S.7).

Runs the infiltrator model with mechanisms switched off one at a time, and with SAROPS-style mechanisms added
(initial position uncertainty, per-particle persistent speed, long action persistence), on a few scenarios.
Reports along-track / cross-track spread (relative to the reported heading), 90% HDR area and the share of mass
near the densest peak over time.

    python -m rl_search.scripts.check_spread --seeds 300 301 302 303 --particles 2000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from rl_search.config import get_preset
from rl_search.scenario import ScenarioSampler
from rl_search.target import simulate
from rl_search.terrain import load_terrain

TIMES_MIN = (5, 10, 20, 30, 45, 60)
VARIANTS = {
    "baseline (current)": {},
    "no heading spread (σ 0°)": dict(heading_std_deg=0.0),
    "no mode switching (always normal move)": dict(p_move=1.0, p_reorient=0.0, p_stop=0.0, move_split=(1, 0, 0)),
    "no reorientation": dict(p_reorient=0.0),
    "SAROPS: LKP position σ 300 m": dict(lkp_pos_std_m=300.0),
    "SAROPS: persistent speed ±25%": dict(speed_persist_rel_std=0.25),
    "SAROPS-like: action persistence 60 min": dict(action_persistence_min=60.0),
}


def metrics(track, exit_step, save_dt, heading_deg, terr):
    u = np.deg2rad(heading_deg)
    ax, cx = np.array([np.cos(u), np.sin(u)]), np.array([-np.sin(u), np.cos(u)])
    out = {}
    for tm in TIMES_MIN:
        k = int(round(tm*60/save_dt))
        act = (exit_step < 0) | (exit_step > k)
        p = np.asarray(track[k], float)[act]
        d = p - p.mean(0)
        along, cross = d @ ax, d @ cx
        H, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=[np.arange(p[:, 0].min()-600, p[:, 0].max()+700, 100),
                                                            np.arange(p[:, 1].min()-600, p[:, 1].max()+700, 100)])
        Hs = gaussian_filter(H, 1.5)
        v = np.sort(Hs.ravel())[::-1]
        hdr90 = (np.searchsorted(np.cumsum(v), .9*v.sum()) + 1)*0.01      # km² (100 m cells)
        i, j = np.unravel_index(Hs.argmax(), Hs.shape)
        pk = np.array([xe[i] + 50, ye[j] + 50])
        near = (np.hypot(*(p - pk).T) < 1000).mean()
        out[tm] = dict(along=float(along.std()), cross=float(cross.std()), hdr90=float(hdr90), peak1km=float(near),
                       mean_adv=float((p.mean(0) - track[0].mean(0)) @ ax))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303])
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/spread_check"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preset = get_preset("rc_cessna")
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    sampler = ScenarioSampler(terr, preset.scenario)
    base = replace(preset.scenario.target, particles=args.particles, duration_s=3700.0)
    res = {v: [] for v in VARIANTS}
    snaps = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        lkp, heading, speed = sampler.sample_target(rng)
        for name, kw in VARIANTS.items():
            cfg = replace(base, **kw)
            track, exit_step = simulate(terr, cfg, lkp, heading, speed, np.random.default_rng(seed + 1))
            res[name].append(dict(seed=seed, speed_kmh=speed*3.6, m=metrics(track, exit_step, cfg.save_dt_s, heading, terr)))
            if seed == args.seeds[0]:
                snaps[name] = (np.asarray(track[int(30*60/cfg.save_dt_s)]), np.asarray(track[int(60*60/cfg.save_dt_s)]), lkp)
            print(seed, name, {t: round(res[name][-1]["m"][t]["along"]) for t in (10, 30, 60)}, flush=True)
    (args.output/"spread.json").write_text(json.dumps(res))

    def mean(name, key):
        return np.array([[r["m"][t][key] for t in TIMES_MIN] for r in res[name]]).mean(0)
    lines = ["| variant | along σ 10/30/60 min (m) | cross σ 10/30/60 min (m) | HDR90 30/60 min (km²) | mass ≤1 km of peak 30/60 min | growth exponent along / cross (20→60 min) |",
             "|---|---|---|---|---|---|"]
    t_idx = {t: i for i, t in enumerate(TIMES_MIN)}
    for name in VARIANTS:
        a, c, h, pk = mean(name, "along"), mean(name, "cross"), mean(name, "hdr90"), mean(name, "peak1km")
        ea = np.log(a[t_idx[60]]/a[t_idx[20]])/np.log(3)
        ec = np.log(c[t_idx[60]]/c[t_idx[20]])/np.log(3)
        lines.append(f"| {name} | {a[t_idx[10]]:.0f} / {a[t_idx[30]]:.0f} / {a[t_idx[60]]:.0f} | "
                     f"{c[t_idx[10]]:.0f} / {c[t_idx[30]]:.0f} / {c[t_idx[60]]:.0f} | {h[t_idx[30]]:.1f} / {h[t_idx[60]]:.1f} | "
                     f"{pk[t_idx[30]]:.0%} / {pk[t_idx[60]]:.0%} | {ea:.2f} / {ec:.2f} |")
    lines.append(f"\nScenarios {args.seeds}, {args.particles} particles. Growth exponent β in σ ∝ t^β: 1 = ballistic "
                 f"(persistent differences), 0.5 = diffusive (random switching).")
    (args.output/"table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for name in VARIANTS:
        ls = "--" if name.startswith("SAROPS") else "-"
        axs[0].plot(TIMES_MIN, mean(name, "along"), ls, marker="o", label=name)
        axs[1].plot(TIMES_MIN, mean(name, "cross"), ls, marker="o", label=name)
        axs[2].plot(TIMES_MIN, mean(name, "peak1km"), ls, marker="o", label=name)
    axs[0].set(title="along-track spread σ (m)", xlabel="minutes after LKP")
    axs[1].set(title="cross-track spread σ (m)", xlabel="minutes after LKP")
    axs[2].set(title="mass within 1 km of the densest peak", xlabel="minutes after LKP")
    for a in axs:
        a.grid(alpha=.3)
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output/"spread_curves.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(2, len(VARIANTS), figsize=(3.3*len(VARIANTS), 6.8), squeeze=False)
    allp = np.concatenate([np.concatenate([s[0], s[1]]) for s in snaps.values()])
    lo, hi = np.quantile(allp, .005, axis=0)/1e3, np.quantile(allp, .995, axis=0)/1e3
    for j, (name, (p30, p60, lkp)) in enumerate(snaps.items()):
        for i, (p, lab) in enumerate(((p30, "30 min"), (p60, "60 min"))):
            a = axs[i, j]
            a.scatter(p[:, 0]/1e3, p[:, 1]/1e3, s=.4, alpha=.5)
            a.plot(lkp[0]/1e3, lkp[1]/1e3, "r*", ms=8)
            a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_aspect("equal")
            a.set_title(f"{name}\n{lab}" if i == 0 else lab, fontsize=7)
            a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(args.output/"spread_snapshots.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
