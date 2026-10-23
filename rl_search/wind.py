"""Wind layer (stage 6): a constant wind drawn per episode plus an Ornstein-Uhlenbeck gust.

``sample()`` is called once per decision step and returns (east, north) m/s; the plant adds it to
the ground velocity. ``estimate`` is what a wind-aware planner or follower is told (the constant
part, no gust). ``ConstantWind`` is the planning model with a known wind and no gust.
"""
from __future__ import annotations

import numpy as np


class WindModel:
    def __init__(self, cfg, dt_s: float):
        self.cfg, self.dt = cfg, float(dt_s)
        self.mean = np.zeros(2)
        self.gust = np.zeros(2)
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        speed = rng.uniform(*self.cfg.speed_mps)
        ang = rng.uniform(0, 2*np.pi)
        self.mean = speed*np.array([np.cos(ang), np.sin(ang)])
        self.gust = np.zeros(2)

    @property
    def estimate(self) -> np.ndarray:
        return self.mean.copy()

    def sample(self) -> np.ndarray:
        tau, s = self.cfg.gust_tau_s, self.cfg.gust_std_mps
        if s > 0:
            a = np.exp(-self.dt/tau)
            self.gust = a*self.gust + s*np.sqrt(1 - a*a)*self.rng.standard_normal(2)
        return self.mean + self.gust

    def get_state(self) -> dict:
        return dict(mean=self.mean.copy(), gust=self.gust.copy(), rng=self.rng.bit_generator.state)

    def set_state(self, s: dict) -> None:
        self.mean, self.gust = s["mean"].copy(), s["gust"].copy()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = s["rng"]


class ConstantWind:
    def __init__(self, vec):
        self.mean = np.asarray(vec, float).copy()

    def reset(self, rng=None) -> None:
        pass

    @property
    def estimate(self) -> np.ndarray:
        return self.mean.copy()

    def sample(self) -> np.ndarray:
        return self.mean.copy()

    def get_state(self) -> dict:
        return dict(mean=self.mean.copy())

    def set_state(self, s: dict) -> None:
        pass


def make_wind(preset) -> WindModel | None:
    return WindModel(preset.wind, preset.search.dt_s) if preset.wind.enabled else None
