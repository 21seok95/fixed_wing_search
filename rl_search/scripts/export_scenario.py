"""Export one scenario to the simulation_data.npz format of the existing uav_search.py and viewers.

uav_search.py does not know about exited particles; they stay at their first outside position,
which a footprint rarely covers. Also reports the particle density check: the best footprint POA
at the centroid intercept time for KDE bandwidths 150 m and 300 m.

Run from the repository root:
    python -m rl_search.scripts.export_scenario rl_search/data/scenarios/eval/s1000000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from rl_search.scenario import ScenarioConfig, load_scenario
from rl_search.terrain import load_terrain
from uav_search import footprint_probs

MAX_TURN_DEG = 23.5   # fixed-wing limit at 35 deg bank, 33.3 m/s, 2 s step


def best_poa(track, exit_step, save_dt_s, t_s, heading_deg, h_m, n_centres=2000, seed=0):
    """Largest share of the initial probability inside one footprint at t_s (centres on particles)."""
    k = int(round(t_s/save_dt_s))
    pts = np.asarray(track[k], float)
    in_map = (exit_step < 0) | (exit_step > k)
    rng = np.random.default_rng(seed)
    centres = pts[in_map][rng.choice(in_map.sum(), min(n_centres, in_map.sum()), replace=False)]
    p = in_map[None, :]*footprint_probs(pts[:, 0], pts[:, 1], centres[:, 0], centres[:, 1],
                                        np.full(len(centres), np.deg2rad(heading_deg)), 83.0, 115.0, h_m)
    return float(p.sum(axis=1).max()/len(pts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path, help="Scenario folder with track.npy, exit_step.npy, meta.json")
    ap.add_argument("--output", type=Path, default=None, help="Default: rl_search/data/export/<scenario name>")
    args = ap.parse_args()

    meta = json.loads((args.scenario/"meta.json").read_text())
    scen = load_scenario(args.scenario, mmap=False)
    cfg = ScenarioConfig.from_dict(meta["config"])
    terr = load_terrain(Path(meta["terrain"]))
    out = args.output or Path("rl_search/data/export")/args.scenario.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"simulation_data.npz", terrain=terr.z, axis=terr.axis,
                        track=scen.track.astype(float), lkp=scen.lkp, forbidden=terr.forbidden,
                        mountain=terr.mountain, water=terr.water, fords=terr.fords, forest=terr.forest,
                        dt_s=scen.save_dt_s, heading_deg=scen.heading_deg, map_m=float(terr.axis[-1] - terr.axis[0]),
                        speed_mean_mps=scen.speed_mps, exit_step=scen.exit_step)
    print(f"Saved {out/'simulation_data.npz'}")

    t_int = scen.metrics["intercept_s"]
    if t_int is not None:
        for h in (150.0, 300.0):
            v = best_poa(scen.track, scen.exit_step, scen.save_dt_s, t_int, scen.heading_deg, h)
            flag = "  (below 1e-3: consider a wider bandwidth)" if v < 1e-3 else ""
            print(f"Best footprint POA at intercept t* = {t_int/60:.1f} min, h = {h:g} m: {v:.2e}{flag}")
    sx, sy = scen.start
    print("Fly the existing greedy planner:")
    print(f"  python uav_search.py {out} --plan greedy --launch-delay 0 --flight {cfg.flight_s/60:g} "
          f"--speed {cfg.uav_speed_mps:.2f} --dt 2 --max-turn {MAX_TURN_DEG} "
          f"--start {sx:.0f} {sy:.0f} --start-heading {scen.start_heading_deg:.1f}")
    print(f"  python mission_viewer.py {out}/search_greedy")


if __name__ == "__main__":
    main()
