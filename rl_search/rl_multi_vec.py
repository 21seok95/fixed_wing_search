"""Multi-UAV worlds as one SB3 VecEnv (docs/multi_agent_design.md "구현 준비 (2026-11-19)"): every UAV of every world is one slot of the
vector env, so one PPO with one shared policy learns from all of them (parameter-shared IPPO, user 2026-11-19/20). Each world
(``rl_multi.MultiRealEnv``, k UAVs) runs in a worker process, one or several per process (``worlds_per_proc``, user 2026-11-23: more
worlds on the Mac; the terrain is held once per process); team sizes can differ between worlds (user: 2, 3 and 4 mixed).

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


class _World:
    """One world in a worker process with its episode records (per UAV: return, active steps, A', penalty, cut, reward parts, own finds,
    following; team: cuts, violations, close steps)."""

    def __init__(self, fn):
        self.world = fn()
        k = self.world.k
        self.ep_r, self.ep_l = np.zeros(k), np.zeros(k, int)
        self.ep_look, self.ep_pen, self.ep_cut = np.zeros(k), np.zeros(k), np.zeros(k)  # per UAV: A' (look), safety penalty, filter cut
        self.ep_parts = np.zeros((4, k))    # per UAV, as rewarded: search, A', cut cost, safety (penalty + violations) (rl_multi info["parts"])
        self.ep_seen, self.ep_follow = np.zeros(k), np.zeros(k)               # per UAV: probability it found itself, steps following a teammate
        self.stats = dict(cuts=0, viol=0.0, close1=0, steps=0)

    def fresh(self):
        self.ep_r[:] = 0.0
        self.ep_l[:] = 0
        self.ep_look[:] = self.ep_pen[:] = self.ep_cut[:] = 0.0
        self.ep_parts[:] = self.ep_seen[:] = self.ep_follow[:] = 0.0
        self.stats.update(cuts=0, viol=0.0, close1=0, steps=0)
        return _stack(self.world.reset())

    def step(self, data):
        obs, rew, done, info = self.world.step([np.atleast_1d(a) for a in data])
        slot_done, infos = self._record(rew, done, info, obs)
        if done:
            return self.fresh(), rew.astype(np.float32), slot_done, infos
        return _stack(obs), rew.astype(np.float32), slot_done, infos

    def _record(self, rew, done, info, obs):
        """The step's episode records; (per-slot done, per-slot infos), with the episode summary when the world is done. ``obs``: the
        observations after the step (kept as ``terminal_observation`` of the slots that end), None for a sequential step."""
        world, stats = self.world, self.stats
        act = info["active"]
        self.ep_r[:] += rew
        self.ep_l[:] += act.astype(int)
        self.ep_look[:] += np.asarray(info.get("look", np.zeros(world.k)))
        self.ep_pen[:] += info["penalty"]
        self.ep_cut[:] += info["cut"]
        if "parts" in info:
            self.ep_parts[:] += info["parts"]
        self.ep_seen[:] += np.asarray(info["seen"])*act
        self.ep_follow[:] += np.asarray(info.get("follow", np.zeros(world.k)), float)
        stats["cuts"] += int((info["cut"] > 0).sum())
        stats["viol"] += float(info["violation"].sum())/2
        stats["close1"] += int(np.isfinite(info["dmin"]) and info["dmin"] < 1000.0)
        stats["steps"] += 1
        infos = [dict(active=bool(act[i]), team=float(info["team"]), share=float(info["share"][i]), seen=float(info["seen"][i]),
                      cut=float(info["cut"][i]), viol=float(info["violation"][i]), slot_k=world.k) for i in range(world.k)]
        # a UAV whose flight ended in this step ends its slot's episode here (user 2026-11-20, bug fix): without it the slot ran on
        # (masked) to the world's end and the advantage of its last flying step took the value of the state after its end
        slot_done = np.array(info.get("ended_now", np.zeros(world.k, bool)), bool) | bool(done)
        if obs is not None:
            for i in np.flatnonzero(slot_done):
                infos[i]["terminal_observation"] = obs[i]
        if done:
            u0 = world.uavs[0]
            ep_parts = self.ep_parts
            pos_parts = ep_parts[0].sum() + ep_parts[1].sum()
            # team: the share of the UAV that found least (an idle UAV), time following a teammate, what the penalties and cuts
            # took as a share of the positive reward (user 11-23)
            common = dict(POS=float(world.found), total=float(world.found), team_k=world.k, cuts=stats["cuts"], viol=stats["viol"],
                          close1km=stats["close1"]/max(stats["steps"], 1), start="launch", typ=getattr(u0, "_ep_typ", ""),
                          launch_km=float(getattr(u0, "start_info", {}).get("launch_dist_m", np.nan))/1000.0,   # (R16) as rl_real._info
                          region=u0.cfg.region, own_min=float(self.ep_seen.min()/max(self.ep_seen.sum(), 1e-12)),
                          follow=float(self.ep_follow.sum()/max(self.ep_l.sum(), 1)), pen_share=float(ep_parts[3].sum()/max(pos_parts, 1.0)),
                          cut_share=float(ep_parts[2].sum()/max(pos_parts, 1.0)), look_share=float(ep_parts[1].sum()/max(pos_parts, 1.0)))
            # (11-23) shares over at least 1 reward unit: an episode that found almost nothing gave 1e9 in M2's first log
            for i in range(world.k):
                # look (A') and mask_cut as in the single-UAV runs' episode records (episode/look, episode/mask_cut), plus the safety penalty
                infos[i]["episode"] = dict(r=float(self.ep_r[i]), l=int(self.ep_l[i]), look=float(self.ep_look[i]), mask_cut=float(self.ep_cut[i]),
                                           safety_pen=float(self.ep_pen[i]), **common)
        return slot_done, infos

    # sequential decisions (rl_multi sequential=True, user 2026-11-23; collected by rl_search.seq_ppo.SeqPPO)
    def seq_begin(self):
        """Start a step: (order of the local UAV ids, {UAV: observation} of the UAVs not flying this step, observation of the first in the order or None)."""
        w = self.world
        order = w.begin_step()
        idle = {i: w.observation(i) for i in range(w.k) if i not in order}
        return order, idle, (w.turn_observation(0) if order else None)

    def seq_turn(self, m, a):
        """The m-th UAV of the order flies ``a``; the observation of the next one, or None."""
        w = self.world
        w.turn_act(m, np.atleast_1d(a))
        return w.turn_observation(m + 1) if m + 1 < len(w._seq["order"]) else None

    def seq_end(self):
        """End the step: (rewards, per-slot done, infos); a world that is done is reset (its first observations come with the next begin)."""
        rew, done, info = self.world.end_step()
        slot_done, infos = self._record(rew, done, info, None)
        for i, x in enumerate(infos):
            x["turn"] = int(np.flatnonzero(info["order"] == i)[0]) if i in info["order"] else -1
        if done:
            self.fresh()
        return rew.astype(np.float32), slot_done, infos

    def seq_preview(self):
        """Every UAV's observation now as the first of a step would see it (the bootstrap value at the end of a rollout)."""
        return _stack([self.world.observation(i) for i in range(self.world.k)])


def _worker(remote, parent_remote, fns):
    """A worker process holding one or more worlds (``worlds_per_proc``, user 2026-11-23: more worlds on the Mac without a process each;
    the terrain is held once per process). The worlds step one after another; commands carry the world's local index where needed."""
    parent_remote.close()
    worlds = [_World(fn) for fn in fns]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send([w.fresh() for w in worlds])
            elif cmd == "reset_one":                                   # one world only (the space check at the start: as one world per process)
                remote.send(worlds[data].fresh())
            elif cmd == "step":
                remote.send([w.step(a) for w, a in zip(worlds, data)])
            elif cmd == "seq_begin":
                remote.send([w.seq_begin() for w in worlds])
            elif cmd == "seq_turn":
                m, acts = data
                remote.send([w.seq_turn(m, a) if a is not None else None for w, a in zip(worlds, acts)])
            elif cmd == "seq_end":
                remote.send([w.seq_end() for w in worlds])
            elif cmd == "seq_preview":
                remote.send([w.seq_preview() for w in worlds])
            elif cmd == "call":
                j, name, args, kwargs = data
                remote.send(getattr(worlds[j].world, name)(*args, **kwargs))
            elif cmd == "get":
                j, name = data
                remote.send(getattr(worlds[j].world, name, None))       # SB3 asks every env for e.g. render_mode
            elif cmd == "set":
                j, name, value = data
                setattr(worlds[j].world, name, value)
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class MultiWorldVecEnv(VecEnv):
    """``world_fns``: callables making a ``MultiRealEnv`` each (forked: the parent preloads the terrain); slot i of the vector env is UAV
    i - offset of the world that holds it. ``worlds_per_proc`` worlds share a worker process (consecutive worlds, so the slot order is the
    world order as with one world per process). ``env_method`` / ``get_attr`` / ``set_attr`` act on the worlds of the given slots."""

    def __init__(self, world_fns, start_method: str = "fork", worlds_per_proc: int = 1):
        ctx = mp.get_context(start_method)
        wpp = max(int(worlds_per_proc), 1)
        groups = [list(world_fns[g:g + wpp]) for g in range(0, len(world_fns), wpp)]
        self.loc = [(p, j) for p, grp in enumerate(groups) for j in range(len(grp))]     # world -> (process, index in it)
        self.remotes, work = zip(*[ctx.Pipe() for _ in groups])
        self.processes = []
        for wr, r, grp in zip(work, self.remotes, groups):
            p = ctx.Process(target=_worker, args=(wr, r, grp), daemon=True)
            p.start()
            self.processes.append(p)
            wr.close()
        self.sizes = [int(self._get(w, "k")) for w in range(len(self.loc))]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        p0, j0 = self.loc[0]
        self.remotes[p0].send(("reset_one", j0))
        o = self.remotes[p0].recv()
        # each key's space follows the dtype the world sends: uint8 maps (setting obs_u8, rl_real.maps_to_u8) as an image space SB3
        # divides by 255, float32 otherwise (user 2026-11-23)
        obs_space = spaces.Dict({key: (spaces.Box(0, 255, v.shape[1:], np.uint8) if v.dtype == np.uint8 else
                                       spaces.Box(-2.0 if key == "vec" else -1.0, 2.0 if key == "vec" else np.inf, v.shape[1:], np.float32))
                                 for key, v in o.items()})
        super().__init__(int(sum(self.sizes)), obs_space, spaces.Box(-1.0, 1.0, (1,), np.float32))
        self._actions = None

    def _get(self, w: int, name: str):
        p, j = self.loc[w]
        self.remotes[p].send(("get", (j, name)))
        return self.remotes[p].recv()

    def _world_of(self, slot: int) -> int:
        return int(np.searchsorted(self.offsets, slot, side="right") - 1)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs = [o for r in self.remotes for o in r.recv()]            # world order
        return {key: np.concatenate([o[key] for o in obs]) for key in obs[0]}

    def step_async(self, actions):
        a = np.asarray(actions, np.float32).reshape(self.num_envs, -1)
        per = [[] for _ in self.remotes]
        for w, (p, _) in enumerate(self.loc):
            per[p].append(a[self.offsets[w]:self.offsets[w + 1]])
        for r, acts in zip(self.remotes, per):
            r.send(("step", acts))

    def step_wait(self):
        res = [x for r in self.remotes for x in r.recv()]           # world order
        obs = {key: np.concatenate([x[0][key] for x in res]) for key in res[0][0]}
        rew = np.concatenate([x[1] for x in res])
        dones = np.concatenate([np.asarray(x[2], bool) for x in res])      # per slot: its own flight ended, or its world's episode
        infos = [i for x in res for i in x[3]]
        return obs, rew, dones, infos

    # ------------------------------------------------------------------ sequential decisions (rl_search.seq_ppo.SeqPPO drives these)
    def seq_begin(self):
        """Start a sequential step in every world: (per world the acting slots in their order, {slot: observation} of the slots not flying
        this step, {slot: observation} of the first slot of each world's order)."""
        for r in self.remotes:
            r.send(("seq_begin", None))
        res = [x for r in self.remotes for x in r.recv()]           # world order
        self._order = [[int(self.offsets[w] + i) for i in order] for w, (order, _, _) in enumerate(res)]
        idle = {int(self.offsets[w] + i): o for w, (_, idl, _) in enumerate(res) for i, o in idl.items()}
        first = {self._order[w][0]: f for w, (_, _, f) in enumerate(res) if f is not None}
        return self._order, idle, first

    def seq_turn(self, m: int, actions: dict) -> dict:
        """Turn m: ``actions`` {slot: action} of the m-th slot of each world's order; returns {slot: observation} of the next ones."""
        per = [[] for _ in self.remotes]
        for w, (p, _) in enumerate(self.loc):
            o = self._order[w]
            per[p].append(float(np.ravel(actions[o[m]])[0]) if m < len(o) else None)
        for r, acts in zip(self.remotes, per):
            r.send(("seq_turn", (m, acts)))
        res = [x for r in self.remotes for x in r.recv()]
        return {self._order[w][m + 1]: o for w, o in enumerate(res) if o is not None}

    def seq_end(self):
        """End the step in every world: (rewards, dones, infos) per slot, as ``step_wait`` without the observations."""
        for r in self.remotes:
            r.send(("seq_end", None))
        res = [x for r in self.remotes for x in r.recv()]
        return (np.concatenate([x[0] for x in res]), np.concatenate([np.asarray(x[1], bool) for x in res]), [i for x in res for i in x[2]])

    def seq_preview(self) -> dict:
        """Every slot's observation as the first of a step would see it (the bootstrap value at the end of a rollout)."""
        for r in self.remotes:
            r.send(("seq_preview", None))
        obs = [o for r in self.remotes for o in r.recv()]
        return {key: np.concatenate([o[key] for o in obs]) for key in obs[0]}

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
                p, j = self.loc[w]
                self.remotes[p].send(("call", (j, method_name, method_args, method_kwargs)))
                done[w] = self.remotes[p].recv()
            out[i] = done[w]
        return [out[i] for i, _ in self._worlds(indices)]

    def get_attr(self, attr_name, indices=None):
        got = {}
        res = []
        for i, w in self._worlds(indices):
            if w not in got:
                got[w] = self._get(w, attr_name)
            res.append(got[w])
        return res

    def set_attr(self, attr_name, value, indices=None):
        for w in sorted({w for _, w in self._worlds(indices)}):
            p, j = self.loc[w]
            self.remotes[p].send(("set", (j, attr_name, value)))
            self.remotes[p].recv()

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
