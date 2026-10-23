"""Final check in PX4 SITL + Gazebo: upload an exported QGC .plan over MAVLink, fly it, compare the track.

Runs in the fw-gz env (numpy + pymavlink only; no rl_search imports). Needs a running
``make px4_sitl gz_rc_cessna`` (GUI or headless); talks to the PX4 offboard-API port 14540, so
QGroundControl on 14550 can watch at the same time.

1. Parse the .plan (QGC mission v2) into MISSION_ITEM_INT and upload it (MAVLink mission protocol).
2. Arm, start the mission (AUTO.MISSION) and log GLOBAL_POSITION_INT through the search waypoints,
   then (unless ``--no-land``) keep watching the mission's landing pattern until PX4 reports
   landed and disarmed, or ``--timeout`` passes.
3. Compare with the planned waypoints (mission.csv next to the .plan): cross-track distance of the
   flown track from the waypoint polyline between the first and last search waypoint, waypoints
   reached, time offset against the planned ETA, landing result. Writes sitl_track.csv and
   sitl_report.json next to the .plan.

A waypoint counts as reached at MISSION_ITEM_REACHED or when MISSION_CURRENT moves past it (PX4
streams MISSION_ITEM_REACHED once per item and some get dropped when waypoints are seconds apart).
The script sends GCS heartbeats (1 Hz) and, for this instance only, sets NAV_DLL_ACT=0 so its
exit does not trigger the data-link-loss RTL, and FW_LND_ABORT=0 / FW_LND_USETER=0 because
gz_rc_cessna has no rangefinder (the default abort bitmask 3 would hold above the field).

    source envs/activate_gz.sh && cd ~/fixed_wing_search
    python rl_search/scripts/sitl_fly_plan.py rl_search/data/plans/rc_cessna/s1000000/greedy/mission.plan
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import time
import numpy as np
from pymavlink import mavutil

R_EARTH = 6378137.0


def plan_items(plan: dict):
    out = []
    for it in plan["mission"]["items"]:
        p = [float("nan") if v is None else float(v) for v in it["params"]]
        out.append(dict(command=int(it["command"]), frame=int(it["frame"]), params=p,
                        autocontinue=int(it.get("autoContinue", True))))
    return out


def upload(m, items, timeout=30.0):
    m.mav.mission_clear_all_send(m.target_system, m.target_component, 0)
    m.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), 0)
    t_end = time.time() + timeout
    sent = set()
    while time.time() < t_end:
        msg = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ACK":
            if msg.type != 0:
                raise RuntimeError(f"Mission upload rejected: MISSION_ACK type {msg.type}")
            return len(sent)
        s = msg.seq
        it = items[s]
        p = it["params"]
        m.mav.mission_item_int_send(m.target_system, m.target_component, s, it["frame"], it["command"],
                                    0, it["autocontinue"], p[0], p[1], p[2], p[3],
                                    int(round(p[4]*1e7)) if math.isfinite(p[4]) else 0,
                                    int(round(p[5]*1e7)) if math.isfinite(p[5]) else 0,
                                    p[6] if math.isfinite(p[6]) else 0.0, 0)
        sent.add(s)
    raise TimeoutError("Mission upload timed out")


def command(m, cmd, *params, timeout=5.0):
    """COMMAND_LONG; returns the MAV_RESULT of this command's COMMAND_ACK (None on timeout)."""
    p = list(params) + [0]*(7 - len(params))
    m.mav.command_long_send(m.target_system, m.target_component, cmd, 0, *p)
    t_end = time.time() + timeout
    while time.time() < t_end:
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if ack is not None and ack.command == cmd:
            return ack.result
    return None


PX4_AUTO, PX4_AUTO_MISSION = 4, 4


def print_status(m, seconds: float):
    t_end = time.time() + seconds
    while time.time() < t_end:
        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=.5)
        if msg is not None:
            print(f"    PX4: {msg.text}")


def wait_autopilot(m, timeout=30.0):
    """First heartbeat from a PX4 autopilot (skips ground stations); sets the target ids."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb is not None and hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4:
            m.target_system, m.target_component = hb.get_srcSystem(), hb.get_srcComponent()
            return hb
    return None


def heartbeat(m):
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def set_param(m, name: str, value: float, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode("ascii"), float(value), ptype)
    t_end = time.time() + 3
    while time.time() < t_end:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is not None and msg.param_id.replace("\x00", "") == name:
            return msg.param_value
    return None


def start_mission(m, tries=40):
    """Switch to AUTO.MISSION (PX4 custom mode) and arm, retrying while the preflight checks settle."""
    for i in range(tries):
        mode = command(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                       PX4_AUTO, PX4_AUTO_MISSION)
        arm = command(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        print(f"  try {i + 1}: set AUTO.MISSION {mode}, arm {arm}")
        if arm == 0 and mode == 0:
            return True
        print_status(m, 3)
    return False


def local_xy(lat, lon, lat0, lon0):
    return np.column_stack([np.deg2rad(np.asarray(lon) - lon0)*R_EARTH*math.cos(math.radians(lat0)),
                            np.deg2rad(np.asarray(lat) - lat0)*R_EARTH])


def seg_dist(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.hypot(*(p - a).T)
    u = np.clip(((p - a) @ ab)/L2, 0, 1)
    return np.hypot(*(p - (a + u[:, None]*ab)).T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path)
    ap.add_argument("--url", default="udpin:0.0.0.0:14540")
    ap.add_argument("--timeout", type=float, default=2400.0, help="Max flight time (s, wall clock)")
    ap.add_argument("--rtl", action="store_true", help="Send RTL after the last search waypoint (default: the mission's own landing)")
    ap.add_argument("--no-land", action="store_true", help="Stop after the last search waypoint instead of watching the landing")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    items = plan_items(plan)
    wp = np.loadtxt(args.plan.with_suffix(".csv"), delimiter=",", skiprows=1, ndmin=2)
    eta, wlat, wlon = wp[:, 2], wp[:, 3], wp[:, 4]
    nav = [i for i, it in enumerate(items) if it["command"] == 16][:len(wp)]
    print(f"{args.plan}: {len(items)} items ({len(nav)} waypoints)")

    m = mavutil.mavlink_connection(args.url, source_system=245)
    hb = wait_autopilot(m)
    if hb is None:
        raise SystemExit(f"No PX4 heartbeat on {args.url}: start PX4 SITL first")
    print(f"Heartbeat from system {m.target_system} (type {hb.type}, autopilot {hb.autopilot})")
    heartbeat(m)
    for name, val, typ in (("NAV_DLL_ACT", 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
                           ("FW_LND_ABORT", 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32),
                           ("FW_LND_USETER", 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)):
        got = set_param(m, name, val, typ)
        print(f"  {name} -> {got}")
    m.mav.request_data_stream_send(m.target_system, m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
    home = m.recv_match(type="HOME_POSITION", blocking=True, timeout=10)
    if home is None:
        command(m, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 242)
        home = m.recv_match(type="HOME_POSITION", blocking=True, timeout=10)
    lat0, lon0 = (home.latitude/1e7, home.longitude/1e7) if home else (wlat[0], wlon[0])
    print(f"Home {lat0:.6f}, {lon0:.6f}; first waypoint {np.hypot(*local_xy(wlat[:1], wlon[:1], lat0, lon0)[0]):.0f} m away")

    n = upload(m, items)
    print(f"Uploaded {n} items")
    if not start_mission(m):
        raise SystemExit("Could not arm / start the mission (see PX4 messages)")
    print("Mission started")

    t0 = time.time()
    track, reached, current = [], {}, -1
    t_search_end = t_landed = t_disarmed = None
    armed, landed_state = True, None
    last_print = last_hb = 0.0
    while time.time() - t0 < args.timeout:
        if time.time() - last_hb > 1.0:
            heartbeat(m)
            last_hb = time.time()
        msg = m.recv_match(type=["GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED", "MISSION_CURRENT", "STATUSTEXT",
                                 "HEARTBEAT", "EXTENDED_SYS_STATE"], blocking=True, timeout=2)
        if msg is None:
            continue
        t = time.time() - t0
        typ = msg.get_type()
        if typ == "GLOBAL_POSITION_INT":
            track.append((t, msg.lat/1e7, msg.lon/1e7, msg.relative_alt/1e3))
        elif typ == "MISSION_ITEM_REACHED":
            reached.setdefault(msg.seq, t)
        elif typ == "MISSION_CURRENT":
            for s in range(max(current, 0), msg.seq):
                reached.setdefault(s, t)
            current = max(current, msg.seq)
        elif typ == "STATUSTEXT":
            print(f"  [{t:5.0f} s] PX4: {msg.text}")
        elif typ == "HEARTBEAT" and msg.get_srcSystem() == m.target_system and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif typ == "EXTENDED_SYS_STATE":
            landed_state = msg.landed_state
        if t - last_print > 30 and track:
            print(f"  [{t:5.0f} s] item {current}/{len(items) - 1}, alt {track[-1][3]:.0f} m, reached {len(reached)}")
            last_print = t
        if t_search_end is None and nav and nav[-1] in reached:
            t_search_end = reached[nav[-1]]
            print(f"  [{t:5.0f} s] last search waypoint reached")
            if args.rtl:
                command(m, mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
            if args.no_land or args.rtl:
                break
        if t_search_end is not None:
            if t_landed is None and landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND:
                t_landed = t
                print(f"  [{t:5.0f} s] landed")
            if not armed:
                t_disarmed = t
                print(f"  [{t:5.0f} s] disarmed")
                break

    tr = np.array(track)
    xy = local_xy(tr[:, 1], tr[:, 2], lat0, lon0)
    wxy = local_xy(wlat, wlon, lat0, lon0)
    first_nav = min((reached[s] for s in nav if s in reached), default=None)
    end = t_search_end if t_search_end is not None else np.inf
    during = (tr[:, 0] >= (first_nav if first_nav is not None else np.inf)) & (tr[:, 0] <= end)
    d = np.full(len(xy), np.inf)
    for a, b in zip(wxy[:-1], wxy[1:]):
        d = np.minimum(d, seg_dist(xy, a, b))
    hit = [s for s in nav if s in reached]
    k = [nav.index(s) for s in hit]
    offs = np.array([reached[s] - reached[hit[0]] - (eta[j] - eta[k[0]]) for s, j in zip(hit, k)]) if hit else np.zeros(0)
    final = tr[-1] if len(tr) else None
    rep = dict(plan=str(args.plan), items=len(items), waypoints=len(nav), reached=len(hit),
               flight_s=float(tr[-1, 0]) if len(tr) else 0.0,
               search_s=float(end - first_nav) if first_nav is not None and np.isfinite(end) else None,
               planned_search_s=float(eta[-1] - eta[k[0]]) if hit else None,
               cross_track_mean_m=float(d[during].mean()) if during.any() else None,
               cross_track_p95_m=float(np.percentile(d[during], 95)) if during.any() else None,
               cross_track_max_m=float(d[during].max()) if during.any() else None,
               eta_offset_end_s=float(offs[-1]) if len(offs) else None,
               eta_offset_max_abs_s=float(np.abs(offs).max()) if len(offs) else None,
               landed=t_landed is not None, disarmed=t_disarmed is not None,
               landing_s=float(t_landed - t_search_end) if t_landed is not None else None,
               touchdown_from_home_m=float(np.hypot(*xy[-1])) if t_landed is not None else None,
               final_rel_alt_m=float(final[3]) if final is not None else None)
    out = args.plan.parent
    np.savetxt(out/"sitl_track.csv", np.column_stack([tr, xy]), delimiter=",",
               header="t_s,lat_deg,lon_deg,rel_alt_m,x_home_m,y_home_m", comments="", fmt="%.7f")
    (out/"sitl_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"Saved {out/'sitl_track.csv'}, {out/'sitl_report.json'}")


if __name__ == "__main__":
    main()
