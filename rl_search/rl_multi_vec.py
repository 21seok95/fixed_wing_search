"""Multi-UAV worlds as one SB3 VecEnv (docs/multi_agent_design.md "구현 준비 (2026-11-19)"): every UAV of every world is one slot of the
vector env, so one PPO with one shared policy learns from all of them (parameter-shared IPPO, user 2026-11-19/20). Each world
(``rl_multi.MultiRealEnv``, k UAVs) runs in its own process; team sizes can differ between worlds (user: 2, 3 and 4 mixed).

- a world steps all its UAVs together; a slot's episode ends when its UAV's flight ends (``done`` for that slot only, the world flies on;
  11-20 fix), the world is done when every UAV has flown its battery out; then it is reset in place and its slots get the next episode's
  first observations (``info["terminal_observation"]`` keeps the last ones)
- ``info["active"]`` per slot and step: False before the UAV's launch and after its battery ended; the trainer zeroes those steps'
  advantages and value targets (``mask_inactive``), so they are not training data
- ``info["episode"]`` at the end of a world's episode, per slot: the UAV's return and active steps, and the team's: found probability
  (POS = team found in the shared belief), team size, filter cuts, violations, time with the nearest pair within 1 km
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _stack(obs_list) -> dict:
    return {key: np.stack([o[key] for o in obs_list]) for key in obs_list[0]}


def _worker(remote, parent_remote, fn):
    parent_remote.close()
    world = fn()
    ep_r = np.zeros(world.k)
    ep_l = np.zeros(world.k, int)
    ep_look, ep_pen, ep_cut = np.zeros(world.k), np.zeros(world.k), np.zeros(world.k)    # per UAV: A' (look), safety penalty, filter cut
    ep_parts = np.zeros((4, world.k))       # per UAV, as rewarded: search, A', cut cost, safety (penalty + violations) (rl_multi info["parts"])
    ep_seen, ep_follow = np.zeros(world.k), np.zeros(world.k)               # per UAV: probability it found itself, steps following a teammate
    stats = dict(cuts=0, viol=0.0, close1=0, steps=0)

    def fresh():
        ep_r[:] = 0.0
        ep_l[:] = 0
        ep_look[:] = ep_pen[:] = ep_cut[:] = 0.0
        ep_parts[:] = ep_seen[:] = ep_follow[:] = 0.0
        stats.update(cuts=0, viol=0.0, close1=0, steps=0)
        return world.reset()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(_stack(fresh()))
            elif cmd == "step":
                obs, rew, done, info = world.step([np.atleast_1d(a) for a in data])
                act = info["active"]
                ep_r[:] += rew
                ep_l[:] += act.astype(int)
                ep_look[:] += np.asarray(info.get("look", np.zeros(world.k)))
                ep_pen[:] += info["penalty"]
                ep_cut[:] += info["cut"]
                if "parts" in info:
                    ep_parts[:] += info["parts"]
                ep_seen[:] += np.asarray(info["seen"])*act
                ep_follow[:] += np.asarray(info.get("follow", np.zeros(world.k)), float)
                stats["cuts"] += int((info["cut"] > 0).sum())
                stats["viol"] += float(info["violation"].sum())/2
                stats["close1"] += int(np.isfinite(info["dmin"]) and info["dmin"] < 1000.0)
                stats["steps"] += 1
                infos = [dict(active=bool(act[i]), team=float(info["team"]), share=float(info["share"][i]), seen=float(info["seen"][i]),
                              cut=float(info["cut"][i]), viol=float(info["violation"][i]), slot_k=world.k) for i in range(world.k)]
                # a UAV whose flight ended in this step ends its slot's episode here (user 2026-11-20, bug fix): without it the slot ran on
                # (masked) to the world's end and the advantage of its last flying step took the value of the state after its end
                slot_done = np.array(info.get("ended_now", np.zeros(world.k, bool)), bool) | bool(done)
                for i in np.flatnonzero(slot_done):
                    infos[i]["terminal_observation"] = obs[i]
                if done:
                    u0 = world.uavs[0]
                    pos_parts = ep_parts[0].sum() + ep_parts[1].sum()
                    # team: the share of the UAV that found least (an idle UAV), time following a teammate, what the penalties and cuts
                    # took as a share of the positive reward (user 11-23)
                    common = dict(POS=float(world.found), total=float(world.found), team_k=world.k, cuts=stats["cuts"], viol=stats["viol"],
                                  close1km=stats["close1"]/max(stats["steps"], 1), start="launch", typ=getattr(u0, "_ep_typ", ""),
                                  region=u0.cfg.region, own_min=float(ep_seen.min()/max(ep_seen.sum(), 1e-12)),
                                  follow=float(ep_follow.sum()/max(ep_l.sum(), 1)), pen_share=float(ep_parts[3].sum()/max(pos_parts, 1e-9)),
                                  cut_share=float(ep_parts[2].sum()/max(pos_parts, 1e-9)), look_share=float(ep_parts[1].sum()/max(pos_parts, 1e-9)))
                    for i in range(world.k):
                        # look (A') and mask_cut as in the single-UAV runs' episode records (episode/look, episode/mask_cut), plus the safety penalty
                        infos[i]["episode"] = dict(r=float(ep_r[i]), l=int(ep_l[i]), look=float(ep_look[i]), mask_cut=float(ep_cut[i]),
                                                   safety_pen=float(ep_pen[i]), **common)
                    obs = fresh()
                remote.send((_stack(obs), rew.astype(np.float32), slot_done, infos))
            elif cmd == "call":
                name, args, kwargs = data
                remote.send(getattr(world, name)(*args, **kwargs))
            elif cmd == "get":
                remote.send(getattr(world, data, None))           # SB3 asks every env for e.g. render_mode
            elif cmd == "set":
                setattr(world, data[0], data[1])
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class MultiWorldVecEnv(VecEnv):
    """``world_fns``: callables making a ``MultiRealEnv`` each (forked: the parent preloads the terrain); slot i of the vector env is UAV
    i - offset of the world that holds it. ``env_method`` / ``get_attr`` / ``set_attr`` act on the worlds of the given slots."""

    def __init__(self, world_fns, start_method: str = "fork"):
        ctx = mp.get_context(start_method)
        self.remotes, work = zip(*[ctx.Pipe() for _ in world_fns])
        self.processes = []
        for wr, r, fn in zip(work, self.remotes, world_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            self.processes.append(p)
            wr.close()
        for r in self.remotes:
            r.send(("get", "k"))
        self.sizes = [int(r.recv()) for r in self.remotes]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.remotes[0].send(("reset", None))
        o = self.remotes[0].recv()
        # each key's space follows the dtype the world sends: uint8 maps (setting obs_u8, rl_real.maps_to_u8) as an image space SB3
        # divides by 255, float32 otherwise (user 2026-11-23)
        obs_space = spaces.Dict({key: (spaces.Box(0, 255, v.shape[1:], np.uint8) if v.dtype == np.uint8 else
                                       spaces.Box(-2.0 if key == "vec" else -1.0, 2.0 if key == "vec" else np.inf, v.shape[1:], np.float32))
                                 for key, v in o.items()})
        super().__init__(int(sum(self.sizes)), obs_space, spaces.Box(-1.0, 1.0, (1,), np.float32))
        self._actions = None

    def _world_of(self, slot: int) -> int:
        return int(np.searchsorted(self.offsets, slot, side="right") - 1)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs = [r.recv() for r in self.remotes]
        return {key: np.concatenate([o[key] for o in obs]) for key in obs[0]}

    def step_async(self, actions):
        a = np.asarray(actions, np.float32).reshape(self.num_envs, -1)
        for w, r in enumerate(self.remotes):
            r.send(("step", a[self.offsets[w]:self.offsets[w + 1]]))

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        obs = {key: np.concatenate([x[0][key] for x in res]) for key in res[0][0]}
        rew = np.concatenate([x[1] for x in res])
        dones = np.concatenate([np.asarray(x[2], bool) for x in res])      # per slot: its own flight ended, or its world's episode
        infos = [i for x in res for i in x[3]]
        return obs, rew, dones, infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)

    def _worlds(self, indices):
        idx = range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else indices)
        return [(i, self._world_of(i)) for i in idx]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        out, done = {}, {}
        for i, w in self._worlds(indices):
            if w not in done:
                self.remotes[w].send(("call", (method_name, method_args, method_kwargs)))
                done[w] = self.remotes[w].recv()
            out[i] = done[w]
        return [out[i] for i, _ in self._worlds(indices)]

    def get_attr(self, attr_name, indices=None):
        got = {}
        res = []
        for i, w in self._worlds(indices):
            if w not in got:
                self.remotes[w].send(("get", attr_name))
                got[w] = self.remotes[w].recv()
            res.append(got[w])
        return res

    def set_attr(self, attr_name, value, indices=None):
        for w in sorted({w for _, w in self._worlds(indices)}):
            self.remotes[w].send(("set", (attr_name, value)))
            self.remotes[w].recv()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*len(list(self._worlds(indices)))


def mask_inactive(rollout_buffer, active: np.ndarray) -> float:
    """Zero the advantages and set the value targets to the values of the steps whose UAV was not flying (``active`` (n_steps, n_envs),
    False = before its launch or after its battery ended): no policy or value learning from them. Returns the share of masked steps."""
    m = ~np.asarray(active, bool)
    if m.any():
        rollout_buffer.advantages[m] = 0.0
        rollout_buffer.returns[m] = rollout_buffer.values[m]
    return float(m.mean())
