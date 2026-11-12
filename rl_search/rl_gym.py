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


class RealMapExtractor(BaseFeaturesExtractor):
    """Feature extractor of the real-region RL env (rl_search.rl_real, docs/rl_mdp_design.md section 7): an independent CNN per map
    (3x3 kernels, 16 -> 32 -> 32 channels, stride 2, flatten, linear ``map_features``; no pooling, the cell position is the reward),
    the maps in ``fine_maps`` start with an extra stride-1 layer (64 -> 64 -> 32 -> 16 -> 8) to keep the image centre / edge detail of
    the 1.6 km map, a vector MLP, and a joint layer shared by the policy and the value function."""

    def __init__(self, observation_space: spaces.Dict, channels=(16, 32, 32), map_features: int = 128, vec_features: int = 32,
                 features_dim: int = 256, fine_maps=("m1600",)):
        super().__init__(observation_space, features_dim)
        self.names = [k for k in observation_space.spaces if k != "vec"]
        self.cnns = nn.ModuleDict()
        for name in self.names:
            c, h, w = observation_space[name].shape
            layers, cin = [], c
            if name in fine_maps:
                layers += [nn.Conv2d(cin, channels[0], 3, stride=1, padding=1), nn.ReLU()]
                cin = channels[0]
            for cout in channels:
                layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU()]
                cin = cout
            cnn = nn.Sequential(*layers, nn.Flatten())
            with torch.no_grad():
                n = cnn(torch.zeros(1, c, h, w)).shape[1]
            self.cnns[name] = nn.Sequential(cnn, nn.Linear(n, map_features), nn.ReLU())
        nv = observation_space["vec"].shape[0]
        self.vec = nn.Sequential(nn.Linear(nv, vec_features), nn.ReLU(), nn.Linear(vec_features, vec_features), nn.ReLU())
        self.joint = nn.Sequential(nn.Linear(len(self.names)*map_features + vec_features, features_dim), nn.ReLU())

    def forward(self, obs):
        feats = [self.cnns[n](obs[n]) for n in self.names] + [self.vec(obs["vec"])]
        return self.joint(torch.cat(feats, dim=1))


def load_expanded(path, env, device: str = "cpu", **ppo_kw):
    """A new PPO on ``env`` with the weights of the model at ``path`` when ``env``'s maps have more channels (user 2026-11-12: the
    relative-POS channel, appended last). Every weight is copied; the first convolution of each map gets zero weights for the new input
    channels, so the new model acts and values exactly like the old one until training uses the new channels. ``ppo_kw``: PPO arguments
    (learning rate, n_steps, ...); the policy architecture is the old model's."""
    import torch
    from stable_baselines3 import PPO
    old = PPO.load(path, device="cpu")
    model = PPO("MultiInputPolicy", env, policy_kwargs=old.policy_kwargs, device=device, **ppo_kw)
    sd_old, sd_new = old.policy.state_dict(), model.policy.state_dict()
    grown = []
    for k, v in sd_new.items():
        o = sd_old[k]
        if o.shape == v.shape:
            sd_new[k] = o.clone()
        elif o.dim() == 4 and o.shape[0] == v.shape[0] and o.shape[2:] == v.shape[2:] and o.shape[1] < v.shape[1]:
            w = torch.zeros_like(v)
            w[:, :o.shape[1]] = o
            sd_new[k] = w
            grown.append(f"{k} {o.shape[1]} -> {v.shape[1]}")
        else:
            raise ValueError(f"cannot carry {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
    model.policy.load_state_dict(sd_new)
    return model, grown
