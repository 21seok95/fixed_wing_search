"""Stage 4 check: FixedWingPlant step responses, steady turn radius, Kinematic vs FixedWing, max_turn.

1. Course steps of 30/90/180 deg from straight flight (setpoint held): yaw(t), roll(t); rise time
   (10-90 %), overshoot, max roll.
2. Steady turn at the largest command: measured radius vs V^2/(g tan FW_R_LIM), must agree within 1 %.
3. KinematicPlant vs FixedWingPlant under the same per-step commands (straight, max turn, alternating
   left/right): trajectories and energy.
4. max_turn_deg: heading change in one decision step from straight flight at the largest command.

Writes step_response.png, steady_turn.png, kin_vs_fw.png and plant_check.json to
rl_search/data/plant_check/<preset>/. Run from the repository root:
    python -m rl_search.scripts.check_plant
    python -m rl_search.scripts.check_plant --set plant.FW_R_LIM=45 plant.NPFG_PERIOD=10
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.config import add_preset_args, get_preset
from rl_search.energy import PowerModel
from rl_search.plants import G, FixedWingPlant, KinematicPlant, measure_max_turn


def step_metrics(t, yaw, phi, step):
    y = yaw/step
    i10 = np.argmax(y >= .1) if (y >= .1).any() else None
    i90 = np.argmax(y >= .9) if (y >= .9).any() else None
    rise = float(t[i90] - t[i10]) if i10 is not None and i90 is not None else float("nan")
    settle_idx = np.nonzero(np.abs(y - 1) > .02)[0]
    settle = float(t[settle_idx[-1]]) if len(settle_idx) else 0.0
    return dict(rise_s=rise, overshoot_pct=float(max(y.max() - 1, 0)*100), max_roll_deg=float(np.degrees(np.abs(phi).max())),
                settle_2pct_s=settle)


def fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x*x + y*y
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = c[0]/2, c[1]/2
    return float(np.sqrt(c[2] + xc*xc + yc*yc)), (xc, yc)


def sequences(max_turn, n):
    return {"straight": np.zeros(n), "max turn": np.full(n, max_turn),
            "alternating": max_turn*np.where(np.arange(n) % 2 == 0, 1.0, -1.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_preset_args(ap)
    ap.add_argument("--steps", type=int, default=15, help="Decision steps for the Kinematic vs FixedWing runs")
    args = ap.parse_args()
    preset = get_preset(args.preset or "rc_cessna", args.set)
    pc, sc = preset.plant, preset.search
    out = Path("rl_search/data/plant_check")/preset.name
    out.mkdir(parents=True, exist_ok=True)
    V, dt = pc.FW_AIRSPD_TRIM, sc.dt_s
    res = dict(preset=preset.name, plant=vars(pc).copy(), dt_s=dt)

    # 1. step responses
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, layout="constrained")
    res["steps"] = {}
    for deg, col in ((30, "#1e88e5"), (90, "#43a047"), (179, "#e53935")):
        p = FixedWingPlant(pc, dt)
        p.reset((0, 0, 0))
        h = p.fly(np.deg2rad(deg), 30.0)
        t, yaw, phi = h[:, 0], h[:, 3], h[:, 4]
        m = step_metrics(t, yaw, phi, np.deg2rad(deg))
        res["steps"][deg] = m
        a1.plot(t, np.degrees(yaw), color=col, label=f"{deg} deg: rise {m['rise_s']:.1f} s, overshoot "
                f"{m['overshoot_pct']:.1f} %, max roll {m['max_roll_deg']:.0f} deg")
        a1.axhline(deg, color=col, ls=":", lw=.8)
        a2.plot(t, np.degrees(phi), color=col)
    a2.axhline(pc.FW_R_LIM, color="k", ls="--", lw=.8, label="FW_R_LIM")
    a2.axhline(pc.gimbal_roll_deg, color="0.5", ls=":", lw=.8, label="gimbal limit")
    a1.set_ylabel("yaw (deg)")
    a2.set_ylabel("roll (deg)")
    a2.set_xlabel("time (s)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    a1.set_title(f"Course step responses (V {V:g} m/s, NPFG_PERIOD {pc.NPFG_PERIOD:g} s, "
                 f"FW_R_TC {pc.FW_R_TC:g} s)", fontsize=10)
    fig.savefig(out/"step_response.png", dpi=140)
    plt.close(fig)

    # 2. steady turn
    p = FixedWingPlant(pc, dt)
    p.reset((0, 0, 0))
    h = p.fly(np.deg2rad(179.0), 40.0, relative=True)
    steady = h[:, 0] > 15.0
    r_meas, (xc, yc) = fit_circle(h[steady, 1], h[steady, 2])
    r_theory = V*V/(G*np.tan(np.deg2rad(pc.FW_R_LIM)))
    err = abs(r_meas - r_theory)/r_theory
    res["steady_turn"] = dict(radius_m=r_meas, theory_m=r_theory, rel_error=err, ok=bool(err < .01),
                              roll_deg=float(np.degrees(h[steady, 4].mean())))
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.plot(h[:, 1], h[:, 2], color="#3949ab", label="FixedWingPlant")
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(xc + r_theory*np.cos(th), yc + r_theory*np.sin(th), "k--", lw=.8, label=f"V²/(g tan φmax) = {r_theory:.1f} m")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Steady turn: radius {r_meas:.2f} m vs {r_theory:.2f} m ({err:.2%})", fontsize=10)
    fig.savefig(out/"steady_turn.png", dpi=140)
    plt.close(fig)

    # 4. max turn
    max_turn = measure_max_turn(pc, dt)
    res["max_turn_deg"] = max_turn
    res["max_turn_rate_deg_s"] = float(np.degrees(G*np.tan(np.deg2rad(pc.FW_R_LIM))/V))

    # 3. Kinematic vs FixedWing
    power = PowerModel(preset.energy, V)
    mt = np.deg2rad(max_turn)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    res["kin_vs_fw"] = {}
    for j, (name, cmds) in enumerate(sequences(mt, args.steps).items()):
        tracks, energy = {}, {}
        for plant in (KinematicPlant(V, dt, max_turn), FixedWingPlant(pc, dt, max_turn)):
            plant.reset((0, 0, 0))
            xy, e = [plant.pose[:2].copy()], [0.0]
            for c in cmds:
                plant.step(c)
                xy.append(plant.pose[:2].copy())
                e.append(e[-1] + power.step_energy(plant.trace))
            key = type(plant).__name__.replace("Plant", "")
            tracks[key], energy[key] = np.array(xy), np.array(e)
        d = np.hypot(*(tracks["Kinematic"] - tracks["FixedWing"]).T)
        res["kin_vs_fw"][name] = dict(pos_diff_mean_m=float(d.mean()), pos_diff_max_m=float(d.max()),
                                      energy_kin=float(energy["Kinematic"][-1]), energy_fw=float(energy["FixedWing"][-1]))
        ax = axes[0, j]
        for key, col in (("Kinematic", "#fb8c00"), ("FixedWing", "#3949ab")):
            ax.plot(*tracks[key].T, marker="o", ms=3, color=col, label=key)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{name}: mean |Δxy| {d.mean():.0f} m", fontsize=10)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        t = np.arange(args.steps + 1)*dt
        for key, col in (("Kinematic", "#fb8c00"), ("FixedWing", "#3949ab")):
            ax.plot(t, energy[key], color=col, label=f"{key}: {energy[key][-1]:.1f}")
        ax.plot(t, t, "k:", lw=.8, label="trim (straight)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("energy (trim-seconds)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Same per-step commands (±{max_turn:.1f} deg/step), {args.steps} steps of {dt:g} s", fontsize=11)
    fig.savefig(out/"kin_vs_fw.png", dpi=140)
    plt.close(fig)

    (out/"plant_check.json").write_text(json.dumps(res, indent=1, default=float))
    for deg, m in res["steps"].items():
        print(f"Step {deg:>3} deg: rise {m['rise_s']:.2f} s, overshoot {m['overshoot_pct']:.1f} %, "
              f"max roll {m['max_roll_deg']:.1f} deg, settle(2%) {m['settle_2pct_s']:.1f} s")
    st = res["steady_turn"]
    print(f"Steady turn: radius {st['radius_m']:.2f} m vs theory {st['theory_m']:.2f} m "
          f"({st['rel_error']:.3%}, {'OK' if st['ok'] else 'FAIL'}), roll {st['roll_deg']:.1f} deg")
    for name, m in res["kin_vs_fw"].items():
        print(f"{name:>11}: |Δxy| mean {m['pos_diff_mean_m']:.0f} m, max {m['pos_diff_max_m']:.0f} m | energy "
              f"Kinematic {m['energy_kin']:.1f}, FixedWing {m['energy_fw']:.1f} trim-s")
    print(f"max_turn_deg = {max_turn:.2f} (per {dt:g} s step; max turn rate {res['max_turn_rate_deg_s']:.1f} deg/s)")
    print(f"Saved {out}/step_response.png, steady_turn.png, kin_vs_fw.png, plant_check.json")


if __name__ == "__main__":
    main()
