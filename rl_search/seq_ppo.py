"""PPO that collects its rollouts with sequential decisions (user 2026-11-23 "순차 결정": within a 5 s step the UAVs of a world decide one
after another in a random order, each seeing what the ones before it did; rl_search.rl_multi ``sequential=True``, driven through
rl_search.rl_multi_vec.MultiWorldVecEnv's ``seq_*`` methods).

Per step: every world draws its order; for turn m = 0, 1, ... the m-th UAV of every world gets its observation (the belief after the
detections of the UAVs before it, those UAVs moved), one policy pass over all of them gives the actions, the worlds fly them; then the
step ends (rewards, done per slot). The rollout buffer gets per slot the observation it decided on, its action, value and log
probability, exactly as SB3's ``collect_rollouts`` stores them, so PPO's update (and the trainer's callbacks: ``infos`` and ``values``
in ``locals``) are unchanged. Slots not flying this step (before the launch, after the flight) are stored with their observation at
the step's start, action 0 and its value / log probability: the trainer masks them (``mask_inactive``). The bootstrap value at the end
of a rollout is that of every slot's observation as the first of the next step would see it (``seq_preview``).

    model = SeqPPO(policy, venv, ...)     or     SeqPPO.load(path, env=venv)
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


def _stack(obs_list) -> dict:
    return {key: np.stack([o[key] for o in obs_list]) for key in obs_list[0]}


class SeqPPO(PPO):
    """PPO whose rollouts are collected turn by turn (see the module doc); the env must offer ``seq_begin`` / ``seq_turn`` / ``seq_end`` /
    ``seq_preview`` (MultiWorldVecEnv over worlds made with ``sequential=True``)."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        if not hasattr(env, "seq_begin"):
            raise TypeError("SeqPPO needs rl_multi_vec.MultiWorldVecEnv (worlds with sequential=True)")
        if self.use_sde:
            raise ValueError("SeqPPO: gSDE is not supported")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        n = env.num_envs
        dones = np.zeros(n, bool)
        while n_steps < n_rollout_steps:
            orders, idle, cur = env.seq_begin()
            obs_of = dict(idle)
            actions = np.zeros((n, *self.action_space.shape), np.float32)
            values = th.zeros((n, 1), device=self.device)
            log_probs = th.zeros(n, device=self.device)
            m = 0
            while cur:                                          # turn m of every world in one policy pass
                slots = sorted(cur)
                with th.no_grad():
                    a, v, lp = self.policy(obs_as_tensor(_stack([cur[s] for s in slots]), self.device))
                a = a.cpu().numpy()
                clipped = np.clip(a, self.action_space.low, self.action_space.high)
                for x, s in enumerate(slots):
                    obs_of[s] = cur[s]
                    actions[s] = a[x]
                values[slots] = v
                log_probs[slots] = lp
                cur = env.seq_turn(m, {s: clipped[x] for x, s in enumerate(slots)})
                m += 1
            if idle:                                            # slots not flying: value and log probability of action 0 (masked later)
                slots = sorted(idle)
                with th.no_grad():
                    v, lp, _ = self.policy.evaluate_actions(obs_as_tensor(_stack([idle[s] for s in slots]), self.device),
                                                            th.zeros((len(slots), *self.action_space.shape), device=self.device))
                values[slots] = v
                log_probs[slots] = lp
            rewards, dones, infos = env.seq_end()
            self.num_timesteps += n
            new_obs = None                                      # (the next observations come with the next step's turns)
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            rollout_buffer.add(_stack([obs_of[s] for s in range(n)]), actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(env.seq_preview(), self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
