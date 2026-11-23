"""Left-right symmetric (mirror-equivariant) actor-critic policy (user 2026-11-23; why and the checks: docs/rl_experiments.md 11-23 16:50,
scripts/multi_design/mirror_check.py, sym_policy_check.py).

The search problem is symmetric under a reflection of the world across the UAV's heading line: every map (heading up, UAV in the centre,
columns = right) flips left-right, the roll, turn-rate state and course-minus-heading entries of the vector change sign, the allowed command
range (low, high) becomes (-high, -low), and the right command is the opposite of the left one. The trained policies (R15, M1, M2, M2b) do
not keep this: their turns prefer one side (R15: 77 % of the hard turns to the left; M2b 0.8M: 98 %), which grew into circling and M2's
collapse. This policy builds the symmetry in: with the network's mean m(o) and value v(o) (the usual MultiInputActorCriticPolicy network,
same parameters),

    mean(o)  = (m(o) - m(mirror(o))) / 2        so mean(mirror(o)) = -mean(o) exactly
    value(o) = (v(o) + v(mirror(o))) / 2        so value(mirror(o)) = value(o)

the Gaussian keeps its state-independent std. Every training sample updates the network for the situation and its mirror image at once;
the one-sided part (m(o) + m(mirror(o))) / 2, which nothing in the task asks for, is gone. A model trained without it loads with the same
weights (``PPO.load(path, custom_objects={"policy_class": SymmetricMultiInputPolicy})``).

Vector layout (``rl_real.VEC_NAMES`` + one log scale per map + optionally 1 / team size and the turn of a sequential step, both unchanged by
the mirror): the constructor refuses any other length, so a new vector entry needs its mirror rule here first.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from rl_search.rl_real import VEC_NAMES

FLIP = tuple(VEC_NAMES.index(n) for n in ("roll", "turn_rate_state", "course_minus_heading"))    # change sign
LOW, HIGH = VEC_NAMES.index("allowed_low"), VEC_NAMES.index("allowed_high")                       # (low, high) -> (-high, -low)
KEEP = tuple(VEC_NAMES.index(n) for n in ("energy_left", "window_time_left", "prob_left"))       # unchanged
EXTRA_KEEP = 2                     # after the map scales (unchanged): at most this many more unchanged entries (1 / team size, design m2;
                                   # the turn in a sequential step, rl_multi sequential=True)
assert sorted(FLIP + KEEP + (LOW, HIGH)) == list(range(len(VEC_NAMES))), "every VEC_NAMES entry needs a mirror rule"


def check_layout(n_vec: int, n_maps: int) -> None:
    """The vector entries this module knows: VEC_NAMES, one scale per map, and up to EXTRA_KEEP unchanged entries."""
    base = len(VEC_NAMES) + n_maps
    if not base <= n_vec <= base + EXTRA_KEEP:
        raise ValueError(f"vector of {n_vec} entries with {n_maps} maps: the mirror rule knows {base} to {base + EXTRA_KEEP} "
                         f"(VEC_NAMES {len(VEC_NAMES)} + map scales {n_maps} + up to {EXTRA_KEEP} unchanged); add the new entries' rule")


def mirror_np(obs: dict) -> dict:
    """The mirrored observation (numpy, single or batched, float or uint8 maps)."""
    out = {}
    for key, v in obs.items():
        if key == "vec":
            m = np.array(v, copy=True)
            m[..., list(FLIP)] = -m[..., list(FLIP)]
            m[..., LOW], m[..., HIGH] = -v[..., HIGH], -v[..., LOW]
            out[key] = m
        else:
            out[key] = np.ascontiguousarray(v[..., ::-1])
    return out


def mirror_th(obs: dict) -> dict:
    """The mirrored observation (torch, batched)."""
    out = {}
    for key, v in obs.items():
        if key == "vec":
            m = v.clone()
            m[..., list(FLIP)] = -v[..., list(FLIP)]
            m[..., LOW] = -v[..., HIGH]
            m[..., HIGH] = -v[..., LOW]
            out[key] = m
        else:
            out[key] = th.flip(v, dims=[-1])
    return out


class SymmetricMultiInputPolicy(MultiInputActorCriticPolicy):
    """MultiInputActorCriticPolicy whose mean action is odd and whose value is even under ``mirror`` (see the module doc)."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if kwargs.get("use_sde"):
            raise ValueError("SymmetricMultiInputPolicy: gSDE is not supported (state-dependent noise is not mirrored)")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        maps = [k for k in observation_space.spaces if k != "vec"]
        check_layout(int(observation_space.spaces["vec"].shape[0]), len(maps))

    def _sym(self, obs: dict) -> tuple[th.Tensor, th.Tensor]:
        """(symmetric mean action, symmetric value) for a batch: one network pass over the batch and its mirror."""
        m = mirror_th(obs)
        both = {k: th.cat([obs[k], m[k]], 0) for k in obs}
        features = self.extract_features(both)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        mean, value = self.action_net(latent_pi), self.value_net(latent_vf)
        n = mean.shape[0]//2
        return (mean[:n] - mean[n:])/2, (value[:n] + value[n:])/2

    def _dist(self, mean: th.Tensor):
        return self.action_dist.proba_distribution(mean, self.log_std)

    def forward(self, obs, deterministic: bool = False):
        mean, value = self._sym(obs)
        dist = self._dist(mean)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), value, log_prob

    def evaluate_actions(self, obs, actions):
        mean, value = self._sym(obs)
        dist = self._dist(mean)
        return value, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self._dist(self._sym(obs)[0])

    def predict_values(self, obs):
        return self._sym(obs)[1]
