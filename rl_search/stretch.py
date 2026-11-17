"""Stretched maps (user 2026-11-17, docs/scenario_redesign.md): a 60 km real map read with every length doubled.

The redesigned scenarios need a large map (the new target speeds spread the particles ~45 km in an hour), and the 120 km maps of
Amboy and Barstow are 16 % and 32 % military land. The user's choice: train on the 60 km maps with every speed halved (targets and
UAV), every length halved, time, angles and probabilities kept. That world, measured with a ruler on which 1 m reads 2 m, is the
original problem itself: the UAV at 33 m/s, the camera footprint, the turn radius, the observation maps, the edge band and the hand-over
distance all keep their original values. So the RL env and the UAV, camera and reward code are not changed at all; only the map is:

- ``amboy_x2`` / ``mojave_x2``: the terrain of ``amboy`` / ``mojave`` with ``axis`` and ``cell`` doubled (25 m cells read as 50 m). The
  slope keeps the real value (``StretchedTerrain``: heights are not doubled, so a slope from the stretched cells would be half);
  distances derived from the terrain (clearance, forest edge depth) are in the stretched metres.
- the targets move on the real 60 km terrain and roads at half the redesigned speeds, with the LKP in the central 15 km; the particle
  tracks, LKP and launch point are then doubled (``stretched_training_scenario``). Their slope limits and road choices are the real ones.

What stays different from a real 120 km map is the terrain only: roads twice as far apart, terrain features twice as wide, the same
heights. The road network ``net`` of a stretched region is the base region's (real metres), used only to make the targets.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from rl_search.terrain import Terrain

STRETCH = {"amboy_x2": ("amboy", 2.0), "mojave_x2": ("mojave", 2.0)}     # stretched region: (base region, length factor)


def base_of(region: str) -> tuple:
    """(base region, length factor) of a region: itself and 1 unless it is stretched."""
    return STRETCH.get(region, (region, 1.0))


@dataclass(eq=False)
class StretchedTerrain(Terrain):
    """A terrain whose horizontal lengths are ``k`` times the real ones; slopes and height gradients stay the real ones."""
    k: float = 2.0

    def _slope(self) -> np.ndarray:
        dzdy, dzdx = np.gradient(self.z, self.cell/self.k, self.cell/self.k)
        return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    def _z_grad(self):
        dzdy, dzdx = np.gradient(self.z, self.cell/self.k, self.cell/self.k)
        self.__dict__["z_gx"], self.__dict__["z_gy"] = dzdx, dzdy

    @property
    def z_gx(self) -> np.ndarray:
        if "z_gx" not in self.__dict__:
            self._z_grad()
        return self.__dict__["z_gx"]

    @property
    def z_gy(self) -> np.ndarray:
        if "z_gy" not in self.__dict__:
            self._z_grad()
        return self.__dict__["z_gy"]


def stretch_terrain(t: Terrain, k: float) -> StretchedTerrain:
    """The base layers of ``t`` (shared, not copied) on an axis ``k`` times longer."""
    kw = {f.name: getattr(t, f.name) for f in fields(Terrain) if f.name != "_float_cache"}
    kw.update(cfg=replace(t.cfg, map_m=t.cfg.map_m*k, cell_m=t.cfg.cell_m*k), axis=t.axis*k, cell=t.cell*k)
    return StretchedTerrain(**kw, k=k)


def scale_scenario(scen, k: float):
    """A scenario with every length times ``k`` (particle tracks, LKP, launch point, target speed); times unchanged."""
    return replace(scen, lkp=np.asarray(scen.lkp, float)*k, speed_mps=float(scen.speed_mps)*k, start=np.asarray(scen.start, float)*k,
                   track=(np.asarray(scen.track, np.float32)*np.float32(k)).astype(np.float32),
                   metrics=dict(scen.metrics, length_factor=k))


SPEED_KEYS = ("fast_kmh", "fast_std_kmh", "slow_kmh", "slow_std_kmh", "off_kmh", "off_std_kmh", "speed_cap_kmh")


def stretched_vc(vc: dict, k: float) -> dict:
    """Target model changes for the real map: every speed divided by ``k``."""
    return {key: (float(v)/k if key in SPEED_KEYS else v) for key, v in vc.items()}


def stretched_training_scenario(vs_base, seed: int, typ: str, k: float):
    """A redesigned training scenario (rl_eval_set.V2) for a stretched map: the targets move on the base map (``vs_base``, real metres)
    at the V2 speeds / k with the LKP in the central 30 km / k, then every length is multiplied by k. Returns (scenario in the stretched
    metres, the target model changes used on the base map)."""
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import DIVERSE, diverse_lkp, v2_vc
    rng = np.random.default_rng(seed)
    vc = stretched_vc(v2_vc(rng, typ, seed), k)
    lkp = diverse_lkp(vs_base.net, rng, DIVERSE["lkp_half_m"]/k)
    heading = float(rng.uniform(0.0, 360.0))
    vs2 = VehicleScenarios(vs_base.sampler, replace(vs_base.vcfg, **vc), region=vs_base.region, net=vs_base.net)
    return scale_scenario(vs2.make_at(seed, lkp, heading, start=(0.0, 0.0)), k), vc
