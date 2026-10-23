"""Vehicle layer: turns a per-step heading change command into the next pose.

Common interface: ``reset(pose)``, ``step(turn_cmd) -> pose``, ``max_turn`` (rad per step),
``attitude`` (roll in rad, passed to the sensor model). A pose is (x, y, yaw) with yaw in rad,
0 = east, counter-clockwise.
"""
from __future__ import annotations

import numpy as np


class KinematicPlant:
    """Instant response: yaw += turn, then a straight step of ``speed * dt`` (uav_search Planner.advance)."""

    def __init__(self, speed_mps: float, dt_s: float, max_turn_deg: float):
        self.speed, self.dt = float(speed_mps), float(dt_s)
        self.max_turn = np.deg2rad(max_turn_deg)
        self.pose = np.zeros(3)

    @property
    def attitude(self) -> float:
        return 0.0

    def reset(self, pose) -> np.ndarray:
        self.pose = np.asarray(pose, float).copy()
        return self.pose.copy()

    def step(self, turn_cmd: float) -> np.ndarray:
        x, y, yaw = self.pose
        yaw = yaw + float(np.clip(turn_cmd, -self.max_turn, self.max_turn))
        step = self.speed*self.dt
        self.pose = np.array([x + step*np.cos(yaw), y + step*np.sin(yaw), yaw])
        return self.pose.copy()
