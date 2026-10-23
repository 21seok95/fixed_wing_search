"""Stage 5: plan one scenario, thin it into waypoints with ETAs, write CSV + QGC .plan, replay it.

1. Handover: transit to the LKP until t_plan; the planner's compute time for the rest of the flight
   is compared against t_plan (it has to be ready by then), and the handover pose predicted at
   launch against the flown one.
2. Export: Douglas-Peucker with ``export.tolerance_m``, ETA per waypoint, mission item count against
   ``export.mission_limit``; mission.csv (x, y, eta, lat, lon) and mission.plan (QGC, PX4 home).
3. Replay: PathFollower flies the waypoints in a fresh env; planned vs executed POS, schedule error.

Outputs go to rl_search/data/plans/<preset>/<scenario>/<method>/ (plus mission.npz for mission_viewer).
Run from the repository root:
    python -m rl_search.scripts.export_plan rl_search/data/scenarios/rc_cessna/eval/s1000000 --method greedy
    python -m rl_search.scripts.export_plan <scenario> --method lawnmower --max-minutes 10   # short SITL test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.config import add_preset_args, preset_from_meta
from rl_search.planners import PLANNERS
from rl_search.runner import METHODS, WIND_MODES, run_method
from rl_search.scenario import load_scenario
from rl_search.terrain import load_terrain
from rl_search.waypoints import write_plan
from rl_search.scripts.export_scenario import export, export_dir


def save_viewer_mission(out_dir: Path, r: dict, preset, name: str):
    log = r["log"]
    n = len(log["t_s"])
    arrays = {k: v for k, v in log.items()} | dict(plan=np.full((n, 1, 1, 3), np.nan), fallback=np.zeros(n, bool))
    meta = json.loads(json.dumps(vars(preset.search) | dict(planner=name, horizon=1, n_headings=5,
                                 start_x=float(log["poses"][0, 0, 0]), start_y=float(log["poses"][0, 0, 1]),
                                 run_time_s=round(r["compute_s"], 1)), default=float))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir/"mission.npz", meta=json.dumps(meta), **arrays)


def plot_plan(path: Path, r: dict, terr_half: float, title: str):
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    if r["plan_xy"] is not None:
        ax.plot(*(r["plan_xy"].T/1000), color="0.6", lw=.8, label="planned (decision poses)")
    ax.plot(*(r["mission"].xy.T/1000), "o", ms=2, color="#e53935", label=f"waypoints ({len(r['mission'])})")
    ax.plot(*(r["log"]["poses"][:, 0, :2].T/1000), color="#3949ab", lw=.8, label="executed (path following)")
    if "lawnmower" in r:
        c = np.array(r["lawnmower"]["rect_corners"])/1000
        ax.plot(*np.vstack([c, c[:1]]).T, "k--", lw=1, label="search rectangle")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--method", default="greedy", choices=sorted(set(METHODS) | set(PLANNERS)))
    ap.add_argument("--wind", default="none", choices=WIND_MODES)
    ap.add_argument("--max-minutes", type=float, default=None, help="Export only the first minutes (SITL tests)")
    ap.add_argument("--output", type=Path, default=None)
    add_preset_args(ap)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Roll exceeds the gimbal")

    meta = json.loads((args.scenario/"meta.json").read_text())
    preset = preset_from_meta(meta, args.set)
    terr = load_terrain(Path(meta["terrain"]))
    scen = load_scenario(args.scenario, mmap=False)
    out = args.output or Path("rl_search/data/plans")/preset.name/args.scenario.name/args.method
    r = run_method(preset, terr, scen, args.method, args.wind, keep_logs=True)

    wp = r["mission"]
    if args.max_minutes is not None:
        wp = wp.head(args.max_minutes*60)
    summ = write_plan(out, wp, preset.export, scen.start, preset.search.uav_speed_mps)
    h = r["handover"]
    t_plan = r["t_plan_s"]
    lines = [f"Scenario {args.scenario.name}, preset {preset.name}, method {args.method}, wind {args.wind}"]
    if h is not None:
        lines.append(f"Handover at {h['t_s']:.0f} s: pose error vs launch prediction {h['error_m']:.1f} m, "
                     f"transit found {h['found']:.4f}, energy {h['energy_used']:.0f} trim-s")
    ok_time = r["compute_s"] <= t_plan if t_plan > 0 else True
    lines.append(f"Compute time {r['compute_s']:.1f} s vs t_plan {t_plan:.0f} s: {'OK' if ok_time else 'TOO SLOW'}")
    if r["plan_xy"] is not None:
        lines.append(f"Thinning: {len(r['plan_xy'])} poses -> {len(r['mission'])} waypoints, max deviation "
                     f"{r['thin_max_dev_m']:.1f} m (tolerance {preset.export.tolerance_m:g} m)")
    lines.append(f"Exported {summ['n_waypoints']} waypoints, {summ['n_items']} mission items (limit "
                 f"{preset.export.mission_limit}{', OVER LIMIT' if summ['over_limit'] else ''}) -> {summ['plan']}")
    if "planned" in r:
        pl = r["planned"]
        lines.append(f"POS planned {pl['final_pos']:.4f} vs executed {r['final_pos']:.4f} "
                     f"({(r['final_pos'] - pl['final_pos'])/max(pl['final_pos'], 1e-12):+.1%}); energy planned "
                     f"{pl['energy_used']:.0f} vs executed {r['energy_used']:.0f} trim-s")
    else:
        lines.append(f"POS executed {r['final_pos']:.4f}, energy {r['energy_used']:.0f} trim-s")
    lines.append(f"Schedule error: mean {r['sched_err_mean_s']:+.0f} s, max |.| {r['sched_err_abs_max_s']:.0f} s, "
                 f"end {r['sched_err_end_s']:+.0f} s | cross-track mean {r['cross_track_mean_m']:.1f} m, "
                 f"max {r['cross_track_max_m']:.0f} m | max roll {r['max_roll_deg']:.0f} deg")
    print("\n".join(lines))
    keep = {k: v for k, v in r.items() if k not in ("log", "mission", "plan_xy", "plan_t")}
    (out/"summary.json").write_text(json.dumps(keep | dict(export=summ), indent=1, default=float))
    (out/"summary.txt").write_text("\n".join(lines) + "\n")
    plot_plan(out/"plan.png", r, terr.axis[-1], f"{args.scenario.name} {args.method}: POS {r['final_pos']:.3f}")
    run_dir = export(scen, terr, export_dir(args.scenario))
    save_viewer_mission(run_dir/f"search_{args.method}_exec", r, preset, f"{args.method} (executed)")
    print(f"Saved {out}/ (summary, plan.png) | viewer: python mission_viewer.py {run_dir/f'search_{args.method}_exec'}")


if __name__ == "__main__":
    main()
