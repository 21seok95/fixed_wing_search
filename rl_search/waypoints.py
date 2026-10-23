"""Waypoint layer (stage 5): thin a planned path, give each point an ETA, write CSV and a QGC .plan.

Local map coordinates are metres east (x) and north (y). ``ExportConfig.origin`` puts either the
launch point or the map centre at the PX4 home position; lat/lon use a local flat-earth projection
(error well below the 15 m tolerance over 60 km).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np

R_EARTH = 6378137.0
MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LOITER_TO_ALT, MAV_CMD_NAV_TAKEOFF = 16, 31, 22
MAV_CMD_NAV_LAND, MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_CMD_DO_LAND_START = 21, 20, 189
MAV_FRAME_GLOBAL_RELATIVE_ALT, MAV_FRAME_MISSION = 3, 2


@dataclass
class Waypoints:
    xy: np.ndarray      # (K, 2) m
    eta_s: np.ndarray   # (K,) s after the LKP observation

    def __len__(self) -> int:
        return len(self.xy)

    @property
    def seg_len(self) -> np.ndarray:
        return np.hypot(*np.diff(self.xy, axis=0).T)

    @property
    def s(self) -> np.ndarray:
        """Distance along the polyline at each waypoint."""
        return np.r_[0.0, np.cumsum(self.seg_len)]

    def after(self, t_s: float) -> "Waypoints":
        """Waypoints from time t_s on, starting with the interpolated point at t_s."""
        k = int(np.searchsorted(self.eta_s, t_s, side="right"))
        if k == 0:
            return self
        if k >= len(self):
            return Waypoints(self.xy[-1:].copy(), self.eta_s[-1:].copy())
        a = (t_s - self.eta_s[k-1])/max(self.eta_s[k] - self.eta_s[k-1], 1e-9)
        p = (1 - a)*self.xy[k-1] + a*self.xy[k]
        return Waypoints(np.vstack([p, self.xy[k:]]), np.r_[t_s, self.eta_s[k:]])

    def head(self, t_end_s: float) -> "Waypoints":
        k = int(np.searchsorted(self.eta_s, t_end_s, side="right"))
        return Waypoints(self.xy[:max(k, 2)].copy(), self.eta_s[:max(k, 2)].copy())

    def to_csv(self, path: Path, latlon: np.ndarray | None = None) -> None:
        cols = [self.xy[:, 0], self.xy[:, 1], self.eta_s]
        header = "x_m,y_m,eta_s"
        if latlon is not None:
            cols += [latlon[:, 0], latlon[:, 1]]
            header += ",lat_deg,lon_deg"
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="", fmt="%.7f")

    @classmethod
    def from_csv(cls, path: Path) -> "Waypoints":
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return cls(a[:, :2].copy(), a[:, 2].copy())


def point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of points p (N, 2) to segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.hypot(*(p - a).T)
    u = np.clip(((p - a) @ ab)/L2, 0, 1)
    return np.hypot(*(p - (a + u[:, None]*ab)).T)


def thin_path(xy: np.ndarray, t_s: np.ndarray, tol_m: float) -> Waypoints:
    """Douglas-Peucker: keep the fewest points so every planned pose is within tol_m of the polyline."""
    xy, t_s = np.asarray(xy, float), np.asarray(t_s, float)
    n = len(xy)
    keep = np.zeros(n, bool)
    keep[[0, n-1]] = True
    stack = [(0, n-1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = point_segment_dist(xy[a+1:b], xy[a], xy[b])
        i = int(np.argmax(d))
        if d[i] > tol_m:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return Waypoints(xy[keep].copy(), t_s[keep].copy())


def max_deviation(xy: np.ndarray, wp: Waypoints) -> float:
    """Largest distance of the dense path from the waypoint polyline."""
    d = np.full(len(xy), np.inf)
    for a, b in zip(wp.xy[:-1], wp.xy[1:]):
        d = np.minimum(d, point_segment_dist(xy, a, b))
    return float(d.max()) if len(wp) > 1 else float(np.hypot(*(xy - wp.xy[0]).T).max())


def origin_xy(cfg, launch_xy, map_centre=(0.0, 0.0)) -> np.ndarray:
    if cfg.origin == "launch":
        return np.asarray(launch_xy, float)
    if cfg.origin == "map":
        return np.asarray(map_centre, float)
    raise ValueError(f"Unknown export origin {cfg.origin!r}")


def to_latlon(xy: np.ndarray, cfg, origin: np.ndarray) -> np.ndarray:
    d = np.asarray(xy, float) - origin
    lat0 = np.deg2rad(cfg.home_lat)
    lat = cfg.home_lat + np.degrees(d[:, 1]/R_EARTH)
    lon = cfg.home_lon + np.degrees(d[:, 0]/(R_EARTH*np.cos(lat0)))
    return np.column_stack([lat, lon])


def from_latlon(latlon: np.ndarray, cfg, origin: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(cfg.home_lat)
    y = np.deg2rad(latlon[:, 0] - cfg.home_lat)*R_EARTH
    x = np.deg2rad(latlon[:, 1] - cfg.home_lon)*R_EARTH*np.cos(lat0)
    return np.column_stack([x, y]) + origin


def _item(seq, cmd, frame, params):
    return dict(type="SimpleItem", autoContinue=True, command=cmd, doJumpId=seq, frame=frame,
                params=params, AMSLAltAboveTerrain=None, Altitude=params[6], AltitudeMode=1)


def qgc_plan(wp: Waypoints, cfg, origin: np.ndarray, cruise_mps: float) -> dict:
    """QGroundControl .plan (mission version 2) with NAV_WAYPOINT items, optional takeoff and RTL."""
    ll = to_latlon(wp.xy, cfg, origin)
    alt = float(cfg.altitude_m)
    items = []
    if cfg.takeoff:
        items.append(_item(1, MAV_CMD_NAV_TAKEOFF, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           [15.0, 0, 0, None, float(ll[0, 0]), float(ll[0, 1]), alt]))
    for lat, lon in ll:
        items.append(_item(len(items) + 1, MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           [0, float(cfg.acceptance_m), 0, None, float(lat), float(lon), alt]))
    if cfg.finish == "land":
        h = np.deg2rad(cfg.land_heading_deg)
        home = np.array([[0.0, 0.0]])
        approach = home - cfg.approach_m*np.array([[np.cos(h), np.sin(h)]])
        (alat, alon), (hlat, hlon) = to_latlon(np.vstack([approach, home]), cfg, np.zeros(2))
        items.append(_item(len(items) + 1, MAV_CMD_DO_LAND_START, MAV_FRAME_MISSION, [0, 0, 0, 0, 0, 0, 0]))
        items.append(_item(len(items) + 1, MAV_CMD_NAV_LOITER_TO_ALT, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           [0, 80.0, 0, None, float(alat), float(alon), float(cfg.approach_alt_m)]))
        items.append(_item(len(items) + 1, MAV_CMD_NAV_LAND, MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           [0, 0, 0, None, float(hlat), float(hlon), 0.0]))
    elif cfg.finish == "rtl":
        items.append(_item(len(items) + 1, MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_FRAME_MISSION, [0, 0, 0, 0, 0, 0, 0]))
    elif cfg.finish != "none":
        raise ValueError(f"Unknown export.finish {cfg.finish!r}")
    return dict(fileType="Plan", version=1, groundStation="QGroundControl",
                geoFence=dict(circles=[], polygons=[], version=2), rallyPoints=dict(points=[], version=2),
                mission=dict(version=2, firmwareType=12, vehicleType=1, cruiseSpeed=float(cruise_mps), hoverSpeed=5,
                             plannedHomePosition=[cfg.home_lat, cfg.home_lon, cfg.home_alt_m], items=items))


def write_plan(out_dir: Path, wp: Waypoints, cfg, launch_xy, cruise_mps: float, name: str = "mission") -> dict:
    """Write <name>.csv and <name>.plan; returns a summary (counts, limit check, file paths)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = origin_xy(cfg, launch_xy)
    plan = qgc_plan(wp, cfg, origin, cruise_mps)
    n_items = len(plan["mission"]["items"])
    if n_items > cfg.mission_limit:
        warnings.warn(f"{n_items} mission items exceed the limit {cfg.mission_limit}; "
                      "raise export.tolerance_m or split the mission")
    wp.to_csv(out_dir/f"{name}.csv", to_latlon(wp.xy, cfg, origin))
    (out_dir/f"{name}.plan").write_text(json.dumps(plan, indent=1))
    return dict(n_waypoints=len(wp), n_items=n_items, over_limit=n_items > cfg.mission_limit,
                csv=str(out_dir/f"{name}.csv"), plan=str(out_dir/f"{name}.plan"), origin_xy=origin.tolist())
