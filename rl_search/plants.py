"""Vehicle layer: turns a per-step heading change command into the next pose.

Common interface:

- ``reset(pose) -> pose``; ``step(turn_cmd, airspeed=None, wind=None) -> pose``
- ``max_turn`` (rad per step, env clip and planner lattice), ``attitude`` (roll in rad, for the sensor)
- ``trace``: (roll samples, airspeed samples, sample dt) of the last step, for the energy model
- ``get_state()`` / ``set_state()`` for snapshots, ``obs_extra()`` for the observation vector

A pose is (x, y, yaw) with yaw in rad, 0 = east, counter-clockwise; yaw is not wrapped. ``wind`` is
(east, north) m/s and adds to the ground velocity; yaw is the air-relative heading.
New plants register in ``PLANTS`` and are picked by ``PlantConfig.kind``.
"""
from __future__ import annotations

import numpy as np

G = 9.80665


def wrap(a):
    return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi


class KinematicPlant:
    """Instant response: yaw += turn, then a straight step of ``speed * dt`` (uav_search Planner.advance).

    ``attitude`` is the coordinated-turn roll of the mean turn rate of the step (energy and logs only).
    """

    def __init__(self, speed_mps: float, dt_s: float, max_turn_deg: float,
                 v_min: float | None = None, v_max: float | None = None):
        self.speed, self.dt = float(speed_mps), float(dt_s)
        self.v_min = self.speed if v_min is None else float(v_min)
        self.v_max = self.speed if v_max is None else float(v_max)
        self.max_turn = np.deg2rad(max_turn_deg)
        self.pose = np.zeros(3)
        self.v, self.phi = self.speed, 0.0
        self.trace = (np.zeros(1), np.full(1, self.speed), self.dt)

    @property
    def attitude(self) -> float:
        return self.phi

    def reset(self, pose) -> np.ndarray:
        self.pose = np.asarray(pose, float).copy()
        self.v, self.phi = self.speed, 0.0
        return self.pose.copy()

    def step(self, turn_cmd: float, airspeed: float | None = None, wind=None) -> np.ndarray:
        x, y, yaw = self.pose
        turn = float(np.clip(turn_cmd, -self.max_turn, self.max_turn))
        yaw = yaw + turn
        v = self.speed if airspeed is None else float(np.clip(airspeed, self.v_min, self.v_max))
        step = v*self.dt
        nx, ny = x + step*np.cos(yaw), y + step*np.sin(yaw)
        if wind is not None:
            nx, ny = nx + wind[0]*self.dt, ny + wind[1]*self.dt
        self.pose = np.array([nx, ny, yaw])
        self.v, self.phi = v, float(np.arctan(v*turn/self.dt/G))
        self.trace = (np.array([self.phi]), np.array([v]), self.dt)
        return self.pose.copy()

    def get_state(self) -> dict:
        return dict(pose=self.pose.copy(), v=self.v, phi=self.phi)

    def set_state(self, s: dict) -> None:
        self.pose, self.v, self.phi = s["pose"].copy(), s["v"], s["phi"]

    def obs_extra(self) -> np.ndarray:
        return np.zeros(0, np.float32)


class FixedWingPlant:
    """Course command -> NPFG-like guidance -> roll loop -> coordinated turn, PX4 parameter names.

    Per decision step the course setpoint is chi_cmd = yaw + turn (held for the step) and, every
    ``inner_dt_s``::

        dr/dt   = w^2 wrap(chi_cmd - yaw) - 2 zeta w r          w = 2 pi / NPFG_PERIOD, zeta = NPFG_DAMPING
        phi_cmd = clip(atan(r V / g), +-FW_R_LIM)
        dphi/dt = clip((phi_cmd - phi) / FW_R_TC, +-FW_R_RMAX)
        dyaw/dt = g tan(phi) / V,   dx/dt = V cos(yaw) + wind_e,   dy/dt = V sin(yaw) + wind_n
    """

    def __init__(self, pc, dt_s: float, max_turn_deg: float | None = None):
        self.pc, self.dt = pc, float(dt_s)
        self.v_trim, self.v_min, self.v_max = pc.FW_AIRSPD_TRIM, pc.FW_AIRSPD_MIN, pc.FW_AIRSPD_MAX
        self.phi_max = np.deg2rad(pc.FW_R_LIM)
        self.tau = pc.FW_R_TC
        self.p_max = np.deg2rad(pc.FW_R_RMAX)
        self.omega = 2*np.pi/pc.NPFG_PERIOD
        self.zeta = pc.NPFG_DAMPING
        self.n_inner = max(1, int(round(self.dt/pc.inner_dt_s)))
        self.h = self.dt/self.n_inner
        self.max_turn = np.deg2rad(180.0 if max_turn_deg is None else max_turn_deg)
        self.pose = np.zeros(3)
        self.phi = self.r = 0.0
        self.v = self.v_trim
        self.trace = (np.zeros(1), np.full(1, self.v), self.dt)

    @property
    def attitude(self) -> float:
        return self.phi

    @property
    def r_max(self) -> float:
        return G*np.tan(self.phi_max)/self.v_trim

    def reset(self, pose, phi: float = 0.0, r: float = 0.0) -> np.ndarray:
        self.pose = np.asarray(pose, float).copy()
        self.phi, self.r, self.v = float(phi), float(r), self.v_trim
        return self.pose.copy()

    def _integrate(self, chi_cmd, n: int, v: float, wind=None, relative: bool = False, out=None):
        """n inner steps towards course chi_cmd (absolute, or yaw + chi_cmd when ``relative``)."""
        x, y, yaw = self.pose
        phi, r, h = self.phi, self.r, self.h
        w2, c2 = self.omega**2, 2*self.zeta*self.omega
        we, wn = (0.0, 0.0) if wind is None else (float(wind[0]), float(wind[1]))
        phis = np.empty(n)
        for i in range(n):
            target = yaw + chi_cmd if relative else chi_cmd
            err = (target - yaw + np.pi) % (2*np.pi) - np.pi
            r += (w2*err - c2*r)*h
            phi_cmd = min(max(np.arctan(r*v/G), -self.phi_max), self.phi_max)
            dphi = min(max((phi_cmd - phi)/self.tau, -self.p_max), self.p_max)
            phi += dphi*h
            yaw += G*np.tan(phi)/v*h
            x += (v*np.cos(yaw) + we)*h
            y += (v*np.sin(yaw) + wn)*h
            phis[i] = phi
            if out is not None:
                out.append((x, y, yaw, phi, r))
        self.pose = np.array([x, y, yaw])
        self.phi, self.r, self.v = phi, r, v
        return phis

    def step(self, turn_cmd: float, airspeed: float | None = None, wind=None) -> np.ndarray:
        turn = float(np.clip(turn_cmd, -self.max_turn, self.max_turn))
        v = self.v_trim if airspeed is None else float(np.clip(airspeed, self.v_min, self.v_max))
        phis = self._integrate(self.pose[2] + turn, self.n_inner, v, wind)
        self.trace = (phis, np.full(self.n_inner, v), self.h)
        return self.pose.copy()

    def fly(self, chi_cmd: float, duration_s: float, airspeed: float | None = None, relative: bool = False):
        """Hold course ``chi_cmd``; returns the inner-step history (K, 6): t, x, y, yaw, phi, r."""
        v = self.v_trim if airspeed is None else float(airspeed)
        hist = [(*self.pose, self.phi, self.r)]
        self._integrate(chi_cmd, int(round(duration_s/self.h)), v, relative=relative, out=hist)
        a = np.array(hist)
        return np.column_stack([np.arange(len(a))*self.h, a])

    def get_state(self) -> dict:
        return dict(pose=self.pose.copy(), phi=self.phi, r=self.r, v=self.v)

    def set_state(self, s: dict) -> None:
        self.pose, self.phi, self.r, self.v = s["pose"].copy(), s["phi"], s["r"], s["v"]

    def obs_extra(self) -> np.ndarray:
        return np.array([self.phi/self.phi_max, self.r/self.r_max], np.float32)


def measure_max_turn(pc, dt_s: float) -> float:
    """Heading change (deg) in one decision step from straight, level flight at the largest command."""
    p = FixedWingPlant(pc, dt_s)
    p.reset((0.0, 0.0, 0.0))
    p.step(np.deg2rad(179.0))
    return float(np.degrees(p.pose[2]))


def _kinematic(preset):
    pc, sc = preset.plant, preset.search
    return KinematicPlant(sc.uav_speed_mps, sc.dt_s, sc.max_turn_deg, pc.FW_AIRSPD_MIN, pc.FW_AIRSPD_MAX)


def _fixed_wing(preset):
    return FixedWingPlant(preset.plant, preset.search.dt_s, preset.search.max_turn_deg)


PLANTS = {"kinematic": _kinematic, "fixed_wing": _fixed_wing}


def make_plant(preset, kind: str | None = None):
    return PLANTS[kind or preset.plant.kind](preset)
