"""SB3 VecEnv with several environments in each worker process (user 2026-11-12: more scenarios in each PPO update without more busy cores).

``SubprocVecEnv`` runs one environment per process, so 10 environments keep 10 cores busy. Here ``envs_per_proc`` environments share
one process and step one after another: 10 environments in 5 processes keep 5 cores busy (on the M2 Pro: inside the 8 performance cores
together with the trainer and an evaluation worker, so no environment waits on an efficiency core). Each update still holds the
trajectories of all 10 environments. The steps per second follow the process count, not the environment count.

The worker protocol is SubprocVecEnv's (step with auto-reset, reset with seed and options, env_method, get_attr, set_attr, is_wrapped).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs


def _worker(remote, parent_remote, fns: CloudpickleWrapper) -> None:
    from stable_baselines3.common.env_util import is_wrapped
    parent_remote.close()
    envs = [_patch_env(fn()) for fn in fns.var]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    reset_info: dict = {}
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                    out.append((obs, reward, done, info, reset_info))
                remote.send(out)
            elif cmd == "reset":
                out = []
                for env, (seed, options) in zip(envs, data):
                    out.append(env.reset(seed=seed, **({"options": options} if options else {})))
                remote.send(out)
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "env_method":
                name, args, kwargs, i = data
                remote.send(envs[i].get_wrapper_attr(name)(*args, **kwargs))
            elif cmd == "get_attr":
                name, i = data
                remote.send(envs[i].get_wrapper_attr(name))
            elif cmd == "has_attr":
                name, i = data
                try:
                    envs[i].get_wrapper_attr(name)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                name, value, i = data
                remote.send(setattr(envs[i], name, value))
            elif cmd == "is_wrapped":
                cls, i = data
                remote.send(is_wrapped(envs[i], cls))
            elif cmd == "render":
                remote.send([env.render() for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the grouped worker")
        except (EOFError, KeyboardInterrupt):
            break


class GroupedSubprocVecEnv(VecEnv):
    """``n`` environments in ceil(n / envs_per_proc) processes; environment i lives in process i // envs_per_proc."""

    def __init__(self, env_fns: list, envs_per_proc: int = 2, start_method: str = "fork"):
        self.waiting, self.closed = False, False
        n, k = len(env_fns), max(1, int(envs_per_proc))
        self.groups = [list(range(i, min(i + k, n))) for i in range(0, n, k)]
        self.where = {i: (g, j) for g, idx in enumerate(self.groups) for j, i in enumerate(idx)}
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in self.groups])
        self.processes = []
        for work_remote, remote, idx in zip(self.work_remotes, self.remotes, self.groups):
            p = ctx.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper([env_fns[i] for i in idx])), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(n, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, idx in zip(self.remotes, self.groups):
            remote.send(("step", [actions[i] for i in idx]))
        self.waiting = True

    def step_wait(self):
        results = [r for remote in self.remotes for r in remote.recv()]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote, idx in zip(self.remotes, self.groups):
            remote.send(("reset", [(self._seeds[i], self._options[i]) for i in idx]))
        results = [r for remote in self.remotes for r in remote.recv()]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        for remote in self.remotes:
            remote.close()
        self.closed = True

    def _call(self, cmd: str, payload, indices) -> list[Any]:
        """One request per environment, in index order (these calls are rare: stage switches, checks)."""
        out = []
        for i in self._get_indices(indices):
            g, j = self.where[i]
            self.remotes[g].send((cmd, payload(j)))
            out.append(self.remotes[g].recv())
        return out

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return self._call("get_attr", lambda j: (attr_name, j), indices)

    def has_attr(self, attr_name: str) -> bool:
        return all(self._call("has_attr", lambda j: (attr_name, j), None))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        self._call("set_attr", lambda j: (attr_name, value, j), indices)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        return self._call("env_method", lambda j: (method_name, method_args, method_kwargs, j), indices)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return self._call("is_wrapped", lambda j: (wrapper_class, j), indices)

    def get_images(self):
        for remote in self.remotes:
            remote.send(("render", None))
        return [img for remote in self.remotes for img in remote.recv()]
