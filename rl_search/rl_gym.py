"""Gymnasium wrapper of ``RLSearchEnv`` and the SB3 feature extractor (8.1 v3, 8.7).

Observation: Dict with four maps ``near``/``mid``/``wide``/``global`` of shape (2, 64, 64) float32
(channel 0 unseen reward area share, channel 1 remaining probability density) and ``vec`` (8,).
Maps are float probabilities, not images: SB3 does not apply /255 to float Boxes.
Action: Box(-1, 1, (1,)) = share of the turn limit added to the course target.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_search.config import get_preset
from rl_search.rl_env import MAP_NAMES, RLConfig, RLSearchEnv
from rl_search.terrain import load_terrain

DEFAULT_TERRAIN = Path("rl_search/data/terrain_seed7_60km.npz")


class RLSearchGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, preset: str = "rc_cessna", overrides: list[str] | None = None, rl: dict | None = None,
                 terrain_path: str | Path = DEFAULT_TERRAIN, seed: int = 0):
        super().__init__()
        terrain_path = Path(terrain_path)
        if not terrain_path.exists():
            raise FileNotFoundError(f"{terrain_path} missing: run `python -m rl_search.scripts.make_terrain` first")
        self.env = RLSearchEnv(load_terrain(terrain_path), get_preset(preset, overrides), RLConfig(**(rl or {})),
                               seed=seed)
        px = self.env.cfg.map_px
        maps = {n: spaces.Box(0.0, np.inf, (2, px, px), np.float32) for n in MAP_NAMES}
        self.observation_space = spaces.Dict({**maps, "vec": spaces.Box(-2.0, 2.0, (8,), np.float32)})
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)


import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiMapExtractor(BaseFeaturesExtractor):
    """Independent CNN per map (no weight sharing, 8.2), vector MLP, joint layer shared by actor/critic."""

    def __init__(self, observation_space: spaces.Dict, channels=(16, 32, 32), map_features: int = 128,
                 vec_features: int = 32, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnns = nn.ModuleDict()
        for name in MAP_NAMES:
            c, h, w = observation_space[name].shape
            layers, cin = [], c
            for cout in channels:
                layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU()]
                cin = cout
            cnn = nn.Sequential(*layers, nn.Flatten())
            with torch.no_grad():
                n = cnn(torch.zeros(1, c, h, w)).shape[1]
            self.cnns[name] = nn.Sequential(cnn, nn.Linear(n, map_features), nn.ReLU())
        nv = observation_space["vec"].shape[0]
        self.vec = nn.Sequential(nn.Linear(nv, vec_features), nn.ReLU(), nn.Linear(vec_features, vec_features),
                                 nn.ReLU())
        self.joint = nn.Sequential(nn.Linear(len(MAP_NAMES)*map_features + vec_features, features_dim), nn.ReLU())

    def forward(self, obs):
        feats = [self.cnns[n](obs[n]) for n in MAP_NAMES] + [self.vec(obs["vec"])]
        return self.joint(torch.cat(feats, dim=1))
