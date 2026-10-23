"""Plot a PX4 SITL flight (sitl_fly_plan.py output) over the exported waypoints.

    python -m rl_search.scripts.plot_sitl rl_search/data/plans/rc_cessna/s1000000/greedy_14min

Reads mission.csv, mission.plan, sitl_track.csv and sitl_report.json in the plan folder and
writes sitl_track.png there: map view (planned waypoints, flown track, landing pattern) and the
altitude profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.waypoints import R_EARTH


def from_latlon(lat, lon, lat0, lon0):
    return (np.deg2rad(np.asarray(lon) - lon0)*R_EARTH*np.cos(np.deg2rad(lat0)),
            np.deg2rad(np.asarray(lat) - lat0)*R_EARTH)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_dir", type=Path)
    args = ap.parse_args()
    d = args.plan_dir

    wp = np.loadtxt(d/"mission.csv", delimiter=",", skiprows=1, ndmin=2)
    tr = np.loadtxt(d/"sitl_track.csv", delimiter=",", skiprows=1, ndmin=2)
    rep = json.loads((d/"sitl_report.json").read_text())
    plan = json.loads((d/"mission.plan").read_text())
    lat0, lon0 = plan["mission"]["plannedHomePosition"][:2]
    wxy = np.column_stack(from_latlon(wp[:, 3], wp[:, 4], lat0, lon0))
    txy = np.column_stack(from_latlon(tr[:, 1], tr[:, 2], lat0, lon0))
    land = [it for it in plan["mission"]["items"] if it["command"] in (16, 21)][len(wp):]
    lxy = np.array([from_latlon(it["params"][4], it["params"][5], lat0, lon0) for it in land]).reshape(-1, 2)

    fig, (ax, az) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    km = 1e-3
    ax.plot(wxy[:, 0]*km, wxy[:, 1]*km, "o-", ms=2.5, lw=1, color="tab:blue", label=f"planned waypoints ({len(wp)})")
    ax.plot(txy[:, 0]*km, txy[:, 1]*km, "-", lw=1, color="tab:red", alpha=.8, label="PX4 SITL track")
    if len(lxy):
        ax.plot(lxy[:, 0]*km, lxy[:, 1]*km, "s--", color="tab:green", ms=5, label="landing pattern")
    ax.plot(0, 0, "k^", ms=9, label="home / launch")
    ax.set_aspect("equal")
    ax.set_xlabel("east of home (km)")
    ax.set_ylabel("north of home (km)")
    ax.grid(alpha=.3)
    ax.legend(loc="best", fontsize=8)
    ct = rep.get("cross_track_mean_m")
    ax.set_title(f"{d.parent.name}/{d.name}: reached {rep['reached']}/{rep['waypoints']}, "
                 f"cross-track mean {ct:.0f} m, p95 {rep['cross_track_p95_m']:.0f} m" if ct is not None else d.name, fontsize=10)

    az.plot(tr[:, 0], tr[:, 3], color="tab:red", lw=1)
    az.set_xlabel("time since mission start (s, wall clock)")
    az.set_ylabel("altitude above home (m)")
    az.grid(alpha=.3)
    status = "landed and disarmed" if rep.get("disarmed") else ("landed" if rep.get("landed") else "not landed")
    az.set_title(f"altitude: {status}" + (f", touchdown {rep['touchdown_from_home_m']:.0f} m from home"
                                          if rep.get("touchdown_from_home_m") is not None else ""), fontsize=10)
    fig.tight_layout()
    out = d/"sitl_track.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
