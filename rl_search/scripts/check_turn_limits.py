"""Run from the repository root: python -m rl_search.scripts.check_turn_limits

Compare +-30 vs +-45 deg per 5 s heading-change limits and heading reference A (actual) / B (previous target)."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rl_search.config import get_preset
from rl_search.plants import FixedWingPlant
from rl_search.energy import make_power
P = get_preset("rc_cessna"); pc = P.plant; power = make_power(P)
DT, CELL, W, L = 5.0, 5.0, 115.0, 83.0
plt.rcParams["font.family"] = "DejaVu Sans"

def fly(actions_deg, lim, ref):
    p = FixedWingPlant(pc, DT); p.reset((0, 0, np.pi/2)); tgt = np.pi/2
    hist = [(0, 0, np.pi/2, 0.0)]; energy = 0.0
    for a in actions_deg:
        d = np.deg2rad(np.clip(a, -1, 1)*lim)
        if ref == "A": cmd = p.pose[2] + d
        else: tgt = tgt + d; cmd = tgt
        out = []
        p._integrate(cmd, p.n_inner, p.v_trim, out=out)
        p.trace = (np.array([o[3] for o in out]), np.full(p.n_inner, p.v_trim), p.h)
        energy += power.step_energy(p.trace)
        hist += [(o[0], o[1], o[2], o[3]) for o in out]
    return np.array(hist), energy

def coverage(h):
    """Binary cell-centre coverage of the nadir footprint, sampled every inner step (0.1 s)."""
    x, y, yaw = h[:, 0], h[:, 1], h[:, 2]
    lo = np.array([x.min(), y.min()]) - 120; hi = np.array([x.max(), y.max()]) + 120
    gx = np.arange(lo[0], hi[0], CELL) + CELL/2; gy = np.arange(lo[1], hi[1], CELL) + CELL/2
    X, Y = np.meshgrid(gx, gy); seen = np.zeros(X.shape, bool); swept = 0.0
    for i in range(0, len(h), 5):  # 0.5 s samples
        c, s = np.cos(yaw[i]), np.sin(yaw[i]); dx, dy = X - x[i], Y - y[i]
        u, v = dx*c + dy*s, -dx*s + dy*c
        seen |= (np.abs(u) <= L/2) & (np.abs(v) <= W/2)
    path = np.hypot(np.diff(x), np.diff(y)).sum()
    return seen.sum()*CELL**2, path*W, (X, Y, seen)

T = 24  # 2 minutes
rng = np.random.default_rng(3)
cases = {
    "Continuous max turn": [1]*T,
    "Alternating ±max": [1, -1]*(T//2),
    "U-turn (180°) then straight": None,
    "Random uniform": list(rng.uniform(-1, 1, T)),
}
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
res = {}
ref_area = coverage(fly([0]*T, 30, "A")[0])[0]
print("straight reference area", ref_area)
for j, (name, acts0) in enumerate(cases.items()):
    for i, lim in enumerate((30, 45)):
        acts = acts0 if acts0 is not None else [1]*(180//lim) + [0]*(T-180//lim)
        ax = axes[i, j]
        for ref, col, ls in (("A", "#d1495b", "-"), ("B", "#2e86ab", "--")):
            h, e = fly(acts, lim, ref)
            area, swept, (X, Y, seen) = coverage(h)
            if ref == "B":
                ax.contourf(X, Y, seen, levels=[0.5, 1.5], colors=["#2e86ab"], alpha=0.15)
            ax.plot(h[:, 0], h[:, 1], ls, color=col, lw=1.6,
                    label=f"{ref}: new area {area/ref_area:.0%} of straight, energy {e:.1f}")
            res[f"{name}|{lim}|{ref}"] = dict(area_m2=round(area), area_vs_straight=round(area/ref_area, 3),
                energy_trim_s=round(e, 1), max_roll_deg=round(float(np.degrees(np.abs(h[:, 3]).max())), 1),
                end_heading_change_deg=round(float(np.degrees(h[-1, 2]-np.pi/2)), 1))
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{name}  (±{lim}° / 5 s)", fontsize=10)
fig.suptitle("FixedWingPlant 15 m/s, 2 min (24 decisions). Red = A (actual heading + Δ), blue dashed = B (previous target + Δ); "
             "shaded = footprint coverage (B)", fontsize=11)
out = Path("rl_search/data/turn_limits"); out.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out/"turn_limits.png", dpi=110)
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in res.items()}, open(out/"turn_limits.json", "w"), indent=1)
for k, v in res.items(): print(k, v)
