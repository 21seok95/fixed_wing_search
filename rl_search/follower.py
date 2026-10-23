"""Path following (stages 3, 5, 6): fly a waypoint polyline the way the PX4 mission mode would.

Each decision step the follower projects the UAV onto the polyline (progress never goes back),
aims at the point ``lookahead_m`` further along it and returns the heading change that points the
ground course there (with a crab angle when a wind estimate is given). With ``schedule`` it also
returns an airspeed command that closes the delay against the planned ETA, inside the plant's
airspeed limits; otherwise the airspeed is NaN (trim).
"""
from __future__ import annotations

import numpy as np

from rl_search.plants import wrap
from rl_search.waypoints import Waypoints


class PathFollower:
    """``dt_s`` bounds the progress per step (1.5 V_max dt plus the wind), so a path that crosses itself
    is not short-cut to its later pass."""

    def __init__(self, wp: Waypoints, cfg, v_trim: float, v_min: float, v_max: float, wind_est=None,
                 dt_s: float = 4.0):
        if len(wp) < 2:
            wp = Waypoints(np.vstack([wp.xy, wp.xy + 1.0]), np.r_[wp.eta_s, wp.eta_s + 1.0])
        self.wp, self.cfg = wp, cfg
        self.v_trim, self.v_min, self.v_max = v_trim, v_min, v_max
        w = 0.0 if wind_est is None else float(np.hypot(*wind_est))
        self.max_advance = 1.5*(v_max + w)*dt_s + cfg.lookahead_m
        self.wind = np.zeros(2) if wind_est is None else np.asarray(wind_est, float)
        self.s_wp = wp.s
        self.total = float(self.s_wp[-1])
        self.a, self.d = wp.xy[:-1], np.diff(wp.xy, axis=0)
        self.L2 = np.maximum((self.d**2).sum(1), 1e-12)
        self.seg = None
        self.progress = 0.0
        self.schedule_error_s = 0.0
        self.cross_track_m = 0.0

    @property
    def done(self) -> bool:
        return self.progress >= self.total - 1.0

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, self.total))
        i = int(np.clip(np.searchsorted(self.s_wp, s, side="right") - 1, 0, len(self.a) - 1))
        u = (s - self.s_wp[i])/np.sqrt(self.L2[i])
        return self.a[i] + min(u, 1.0)*self.d[i]

    def eta_at(self, s: float) -> float:
        return float(np.interp(s, self.s_wp, self.wp.eta_s))

    def _project(self, p: np.ndarray):
        if self.seg is None:
            lo, hi, s_cap = 0, len(self.a), self.total
        else:
            lo = self.seg
            s_cap = self.progress + min(self.max_advance, self.cfg.search_ahead_m)
            hi = int(np.searchsorted(self.s_wp, s_cap, side="right"))
            hi = min(max(hi, lo + 1), len(self.a))
        a, d, L2 = self.a[lo:hi], self.d[lo:hi], self.L2[lo:hi]
        L = np.sqrt(L2)
        s0 = self.s_wp[lo:hi]
        u_cap = np.clip((s_cap - s0)/L, 0, 1)
        u = np.minimum(np.clip(((p - a)*d).sum(1)/L2, 0, 1), u_cap)
        q = a + u[:, None]*d
        dist = np.hypot(*(p - q).T)
        j = int(np.argmin(dist))
        return lo + j, s0[j] + u[j]*L[j], float(dist[j])

    def _pursuit_point(self, p: np.ndarray) -> np.ndarray:
        """First path point past the progress at least ``lookahead_m`` from the UAV: loops tighter than
        that are cut instead of orbited."""
        L = self.cfg.lookahead_m
        s = np.arange(self.progress + L, min(self.progress + self.max_advance + L, self.total) + 5.0, 5.0)
        if not len(s):
            return self.point_at(self.total)
        pts = np.array([self.point_at(v) for v in s])
        far = np.hypot(*(pts - p).T) >= L
        return pts[int(np.argmax(far))] if far.any() else pts[-1]

    def act(self, pose: np.ndarray, t_s: float):
        """(heading change rad, airspeed m/s or NaN) for this step."""
        p = np.asarray(pose[:2], float)
        seg, s, dist = self._project(p)
        if s >= self.progress:
            self.seg, self.progress = seg, s
        elif self.seg is None:
            self.seg = seg
        self.cross_track_m = dist
        target = self._pursuit_point(p)
        chi = np.arctan2(target[1] - p[1], target[0] - p[0])
        v = self.v_trim
        self.schedule_error_s = t_s - self.eta_at(self.progress)
        if self.cfg.schedule:
            v = float(np.clip(self.v_trim + self.cfg.schedule_gain*self.schedule_error_s, self.v_min, self.v_max))
        heading = chi
        if self.wind.any():
            w_perp = -self.wind[0]*np.sin(chi) + self.wind[1]*np.cos(chi)
            heading = chi - np.arcsin(np.clip(w_perp/v, -1, 1))
        return float(wrap(heading - pose[2])), (v if self.cfg.schedule else np.nan)
