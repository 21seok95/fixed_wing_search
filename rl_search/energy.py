"""Energy layer: power of a fixed-wing as a function of roll and airspeed, normalised to P_trim = 1.

    P(phi, V) = (1 - k) (V / V_trim)^3 + k (V_trim / V) / cos^2(phi)

Parasitic power grows with V^3; induced power falls with V and grows with the load factor
1/cos(phi) squared (k = induced share at trim: 1.25x at 35 deg bank, 1.71x at 50 deg for k = 0.5).
The budget is ``EnergyConfig.budget_s`` trim-flight seconds; the env subtracts the power integrated
over the plant's inner steps (``plant.trace``), transit included.
"""
from __future__ import annotations

import numpy as np


class PowerModel:
    def __init__(self, cfg, v_trim: float):
        self.cfg, self.v_trim = cfg, float(v_trim)
        self.k = cfg.induced_share

    @property
    def budget(self) -> float:
        return float(self.cfg.budget_s)

    def power(self, phi, v):
        r = np.asarray(v, float)/self.v_trim
        return (1 - self.k)*r**3 + self.k/r/np.cos(phi)**2

    def step_energy(self, trace) -> float:
        phi, v, h = trace
        return float(np.sum(self.power(phi, v))*h)

    def endurance_s(self, energy_left: float, v: float | None = None) -> float:
        """Straight-flight time the remaining energy lasts at airspeed v (default trim)."""
        return energy_left/float(self.power(0.0, self.v_trim if v is None else v))


def make_power(preset) -> PowerModel | None:
    return PowerModel(preset.energy, preset.search.uav_speed_mps) if preset.energy.enabled else None
