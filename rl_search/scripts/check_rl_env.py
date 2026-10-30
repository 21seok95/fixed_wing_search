"""Step 9 checks of RLSearchEnv with fixed command sequences (no learning needed).

Runs straight / max circle / left-right alternation / "seek" (steer to the remaining-mass centroid) from
the same start on one scenario per curriculum stage, checks probability mass conservation, area
accounting and termination, and draws paths, cumulative reward terms, the four observation maps and
the course-target gap. Optionally adds a trained policy (--model, needs stable-baselines3).

    python -m rl_search.scripts.check_rl_env
    python -m rl_search.scripts.check_rl_env --stages 1 2 3 --seed 5 --model rl_search/data/rl/smoke/model_final.zip
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.config import get_preset
from rl_search.plants import wrap
from rl_search.rl_env import MAP_NAMES, RLConfig, RLSearchEnv
from rl_search.terrain import load_terrain

COLORS = {"straight": "#577590", "circle": "#f94144", "alternate": "#f8961e", "seek": "#43aa8b", "policy": "#7b2cbf"}


def seek_action(env) -> float:
    b = env.belief
    act = b.w > 0
    if not act.any():
        return 0.0
    pts = b.positions(env.t)[act]
    c = np.average(pts, axis=0, weights=b.w[act])
    bearing = np.arctan2(c[1] - env.plant.pose[1], c[0] - env.plant.pose[0])
    return float(np.clip(wrap(bearing - env.chi_tgt)/env.turn, -1, 1))


def run(env, options, policy, name):
    obs, info = env.reset(options=options)
    pos0 = env.belief.pos
    log = dict(xy=[env.plant.pose[:2].copy()], t=[env.t], dpos=[], darea=[], r=[], gap=[], roll=[], mass_err=[])
    snap = None
    k, done, t_wall = 0, False, time.perf_counter()
    while not done:
        if name == "straight":
            a = 0.0
        elif name == "circle":
            a = 1.0
        elif name == "alternate":
            a = 1.0 if k % 2 == 0 else -1.0
        elif name == "seek":
            a = seek_action(env)
        else:
            a = float(policy.predict(obs, deterministic=True)[0][0])
        obs, r, term, trunc, info = env.step([a])
        b = env.belief
        log["xy"].append(env.plant.pose[:2].copy())
        log["t"].append(env.t)
        log["dpos"].append(info["dpos"])
        log["darea"].append(info["darea_m2"])
        log["r"].append(r)
        log["gap"].append(info["gap_deg"])
        log.setdefault("shield", []).append(info["shield"])
        log["roll"].append(info["roll_deg"])
        log["mass_err"].append(abs(pos0 + np.sum(log["dpos"]) - b.pos) + max(0.0, -float(b.w.min())))
        assert info["darea_m2"] <= info["n_cells"]*env.cfg.cell_m**2 + 1e-6
        if k == 40 and name == "seek":
            snap = {n: obs[n].copy() for n in MAP_NAMES} | {"pose": env.plant.pose.copy(), "t": env.t}
        k += 1
        done = term or trunc
    wall = time.perf_counter() - t_wall
    out = {k_: np.array(v) for k_, v in log.items()}
    c = env.cfg
    out["summary"] = dict(
        steps=k, shield_steps=int(np.sum(log.get("shield", [0]))), flight_min=round((env.t - log["t"][0])/60, 1), exited=bool(info["exited"]),
        energy_end=round(float(info["energy_left"]), 1), pos=round(float(np.sum(log["dpos"])), 4),
        area_km2=round(float(np.sum(log["darea"]))/1e6, 3),
        R_pos=round(float(c.w_pos*np.sum(log["dpos"])/c.pos_ref), 3),
        R_area=round(float(c.w_area*np.sum(log["darea"])/env.area_ref), 3), R=round(float(np.sum(log["r"])), 3),
        gap_abs_p99_deg=round(float(np.percentile(np.abs(log["gap"]), 99)), 2),
        gap_abs_max_deg=round(float(np.max(np.abs(log["gap"]))), 2),
        roll_abs_max_deg=round(float(np.max(np.abs(log["roll"]))), 1),
        pos_consistency_err_max=float(np.max(log["mass_err"])), ms_per_step=round(1000*wall/max(k, 1), 1))
    return out, snap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="rc_cessna")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("rl_search/data/rl_check"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Roll exceeds")
    preset = get_preset(args.preset)
    terr = load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    policy = None
    names = ["straight", "circle", "alternate", "seek"]
    if args.model:
        from stable_baselines3 import PPO
        policy = PPO.load(args.model, device="cpu")
        names.append("policy")
    args.output.mkdir(parents=True, exist_ok=True)
    env = RLSearchEnv(terr, preset, RLConfig(), seed=args.seed)
    report = {}
    for stage in args.stages:
        env.reset(options=dict(stage=stage))
        base = dict(stage=stage, scenario=env.scen)
        if stage in (1, 2):
            base |= dict(t0_s=env.t, start_pose=env.plant.pose.copy())
        logs, snap = {}, None
        for name in names:
            logs[name], s = run(env, base, policy, name)
            snap = snap or s
        report[f"stage{stage}"] = {n: l["summary"] for n, l in logs.items()}
        report[f"stage{stage}"]["scenario"] = dict(seed=int(env.scen.seed), speed_kmh=round(env.scen.speed_mps*3.6, 1),
                                                   t_start_s=round(float(logs["seek"]["t"][0]), 1))
        # ---- figure 1: paths and cumulative terms
        fig, ax = plt.subplots(1, 3, figsize=(19, 6))
        t_mid = logs["seek"]["t"][0]
        b0 = env.belief
        pts = b0.positions(t_mid)
        ax[0].scatter(pts[:, 0]/1e3, pts[:, 1]/1e3, s=1, c="#bbbbbb", label=f"prior particles at start ({t_mid/60:.0f} min)")
        pe = b0.positions(t_mid + 1800)
        ax[0].scatter(pe[:, 0]/1e3, pe[:, 1]/1e3, s=1, c="#ffd6a5", label="prior particles at +30 min")
        for n, l in logs.items():
            ax[0].plot(l["xy"][:, 0]/1e3, l["xy"][:, 1]/1e3, color=COLORS[n], lw=1.0,
                       label=f"{n}: R={l['summary']['R']} (POS {l['summary']['R_pos']} + area {l['summary']['R_area']})")
        ax[0].plot(*logs["seek"]["xy"][0]/1e3, "k*", ms=12)
        ax[0].set_aspect("equal")
        ax[0].legend(fontsize=7, loc="best", markerscale=6)
        ax[0].set_title(f"stage {stage}: paths (km)")
        for n, l in logs.items():
            tm = (l["t"][1:] - l["t"][0])/60
            ax[1].plot(tm, np.cumsum(l["dpos"])*preset.search.pod/preset.search.pod, color=COLORS[n], label=f"{n} POS")
            ax[2].plot(tm, np.cumsum(l["darea"])/1e6, color=COLORS[n], label=f"{n}")
        ax[1].set_title("cumulative POS (policy part)")
        ax[2].set_title("cumulative new area in reward zone (km²)")
        for a_ in ax[1:]:
            a_.set_xlabel("min since start")
            a_.grid(alpha=.3)
            a_.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output/f"stage{stage}_paths.png", dpi=100)
        plt.close(fig)
        # ---- figure 2: observation maps at step 40 of seek
        if snap:
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            for j, n in enumerate(MAP_NAMES):
                for ch, lab in enumerate(("unseen reward-area share", "remaining prob. density (/km²)")):
                    im = axs[ch, j].imshow(snap[n][ch], origin="lower", cmap="viridis" if ch else "Greens")
                    axs[ch, j].set_title(f"{n}: {lab}", fontsize=8)
                    fig.colorbar(im, ax=axs[ch, j], fraction=.046)
                    axs[ch, j].plot(31.5, 31.5, "r+", ms=10) if n != "global" else None
            fig.suptitle(f"stage {stage}: observation maps at step 40 of 'seek' (t={snap['t']/60:.1f} min), red + = UAV")
            fig.tight_layout()
            fig.savefig(args.output/f"stage{stage}_maps.png", dpi=90)
            plt.close(fig)
    (args.output/"report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
