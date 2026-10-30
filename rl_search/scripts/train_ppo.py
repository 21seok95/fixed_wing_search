"""PPO training on RLSearchEnv (baseline B of docs/rl_model_recommendation.md).

Run from the repository root (terrain first: python -m rl_search.scripts.make_terrain):

    # Mac / CPU smoke test: 1 env, a few thousand steps
    python -m rl_search.scripts.train_ppo --tag smoke --n-envs 1 --n-steps 256 --total-steps 2048
    # Ubuntu + RTX 2070: parallel envs on the CPU cores, model on the GPU
    python -m rl_search.scripts.train_ppo --tag stage1_g1 --n-envs 8 --total-steps 2000000 --device cuda
    # resume, next curriculum stage
    python -m rl_search.scripts.train_ppo --tag stage2 --stage 2 --resume rl_search/data/rl/stage1_g1/model_final.zip

Outputs go to rl_search/data/rl/<tag>/: config.json, checkpoints, model_final.zip, TensorBoard logs (tb/).
The environment is the bottleneck: set --n-envs to about the number of physical CPU cores.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per env process
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def eval_episode(model, gym_env, scen, seed, r_pos_k, r_area_k, record=False, snap_every=12):
    """One deterministic episode on a fixed scenario. Returns (eval row, trace); with ``record`` the trace
    holds the path, its times and particle weights every ``snap_every`` decisions (for episode_figure)."""
    import numpy as np
    obs, info = gym_env.reset(seed=seed, options=dict(scenario=scen))
    env = gym_env.env
    snap = lambda: (env.t, env.belief.w.copy(), env.pos_cum, env.area_cum/1e6)
    trace = dict(path=[env.plant.pose[:2].copy()], t=[env.t], snaps=[snap()] if record else [])
    R, acts, shields, done = 0.0, [], 0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = gym_env.step(a)
        trace["path"].append(env.plant.pose[:2].copy())
        trace["t"].append(env.t)
        if record and len(acts) % snap_every == 0:
            trace["snaps"].append(snap())
        R += r
        acts.append(abs(float(np.ravel(a)[0])))
        shields += bool(info["shield"])
        done = term or trunc
    if record:
        trace["snaps"].append(snap())
    acts = np.minimum(acts, 1.0)
    row = dict(pos=info["pos_policy"], area_km2=info["area_policy_km2"], R=R,
               R_pos=r_pos_k*info["pos_policy"], R_area=r_area_k*info["area_policy_km2"],
               exited=float(info["exited"]), flight_min=info["t_s"]/60, steps=len(acts),
               action_sat=float((acts >= 0.99).mean()), shield=shields/len(acts))
    return row, trace


def episode_figure(env, trace, title=""):
    """Three time slices (1/3, 2/3, end of the flight) of a recorded eval episode."""
    import numpy as np
    from rl_search.plotting import draw_rl_episode
    snaps = trace["snaps"]
    pick = [snaps[min(int(round(f*(len(snaps) - 1))), len(snaps) - 1)] for f in (1/3, 2/3, 1)]
    return draw_rl_episode(env.terr, env.belief, np.array(trace["path"]), np.array(trace["t"]), pick, title)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preset", default="rc_cessna")
    ap.add_argument("--set", nargs="*", default=[], help="Preset overrides, e.g. search.pod=1")
    ap.add_argument("--rl", nargs="*", default=[], help="RLConfig overrides, e.g. w_area=0.5 hdr_level=0.95")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=1024, help="Rollout steps per env per update")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=1.0, help="1.7 v2: compare 1.0 and 0.999")
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cpu, cuda, mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=50_000, help="Total env steps between checkpoints")
    ap.add_argument("--eval-every", type=int, default=50_000,
                    help="Total env steps between deterministic evaluations on fixed scenarios (0 disables)")
    ap.add_argument("--eval-scenarios", type=int, default=3, help="First N scenarios of --eval-dir")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Default rl_search/data/scenarios/<preset>/eval (not used for training)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/rl"))
    args = ap.parse_args()

    import torch
    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.logger import Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from rl_search.rl_gym import MultiMapExtractor, RLSearchGym
    from rl_search.scenario import load_scenario

    out = args.out/args.tag
    out.mkdir(parents=True, exist_ok=True)
    rl = parse_kv(args.rl) | {"stage": args.stage}
    env_kwargs = dict(preset=args.preset, overrides=args.set, rl=rl)
    (out/"config.json").write_text(json.dumps(vars(args) | {"rl_cfg": rl}, indent=1, default=str))

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    info_keys = ("pos_policy", "area_policy_km2", "exited", "t_s", "energy_left", "escaped")
    venv = make_vec_env(RLSearchGym, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs,
                        vec_env_cls=vec_cls, monitor_kwargs=dict(info_keywords=info_keys))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device={device} n_envs={args.n_envs} torch={torch.__version__}")

    eval_env = RLSearchGym(**env_kwargs)        # main process: reward scales, evaluation
    rcfg = eval_env.env.cfg
    r_pos_k, r_area_k = rcfg.w_pos/rcfg.pos_ref, rcfg.w_area*1e6/eval_env.env.area_ref

    class EpisodeStats(BaseCallback):
        """Episode POS, area, reward terms, map-exit rate, course-target gap and action saturation to
        TensorBoard (9.9, 1.6 v8, 11.4)."""

        def _on_step(self) -> bool:
            acts = np.abs(np.ravel(self.locals["clipped_actions"]))
            self.logger.record_mean("step/action_abs", float(acts.mean()))
            self.logger.record_mean("step/action_sat", float((acts >= 0.99).mean()))
            for info in self.locals["infos"]:
                self.logger.record_mean("step/gap_abs_deg", abs(info.get("gap_deg", 0.0)))
                self.logger.record_mean("step/shield", float(info.get("shield", False)))
                self.logger.record_mean("step/roll_abs_deg", abs(info.get("roll_deg", 0.0)))
                ep = info.get("episode")
                if ep:
                    for k in info_keys:
                        if k in ep:
                            self.logger.record_mean(f"episode/{k}", float(ep[k]))
                    self.logger.record_mean("episode/R_pos", r_pos_k*float(ep["pos_policy"]))
                    self.logger.record_mean("episode/R_area", r_area_k*float(ep["area_policy_km2"]))
            return True

    class FixedEval(BaseCallback):
        """Deterministic policy on fixed held-out scenarios every ``--eval-every`` steps (11.3): eval/*
        scalars (same scenarios and start poses each time, so the curve shows the policy only) and the
        path of the first scenario over the remaining probability as an image."""

        def __init__(self, scens, every):
            super().__init__()
            self.scens, self.every, self.next, self.last = scens, every, every, -1

        def _on_training_start(self):
            self.next = (self.num_timesteps//self.every + 1)*self.every     # --resume starts mid-way

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next:
                self.next += self.every
                self.evaluate()
            return True

        def _on_training_end(self):
            if self.last != self.num_timesteps:
                self.evaluate()

        def evaluate(self):
            self.last = self.num_timesteps
            rows = []
            for i, scen in enumerate(self.scens):
                rows.append(self.episode(scen, seed=1000 + i, draw=(i == 0)))
            for k in rows[0]:
                self.logger.record(f"eval/{k}", float(np.mean([r[k] for r in rows])))
            self.logger.dump(self.num_timesteps)

        def episode(self, scen, seed, draw):
            row, trace = eval_episode(self.model, eval_env, scen, seed, r_pos_k, r_area_k, record=draw)
            if draw:
                fig = episode_figure(eval_env.env, trace, f"step {self.num_timesteps:,}: R {row['R']:.3f} "
                                     f"(R_pos {row['R_pos']:.3f}, R_area {row['R_area']:.3f})")
                self.logger.record("eval/path", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            return row

    callbacks = [EpisodeStats()]
    if args.eval_every > 0:
        eval_dir = args.eval_dir or Path(f"rl_search/data/scenarios/{args.preset}/eval")
        scens = [load_scenario(p) for p in sorted(eval_dir.iterdir())[:args.eval_scenarios]]
        callbacks.append(FixedEval(scens, args.eval_every))
        print(f"eval: {len(scens)} scenarios from {eval_dir} every {args.eval_every:,} steps")

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=device)
        model.tensorboard_log = str(out/"tb")
    else:
        policy_kwargs = dict(features_extractor_class=MultiMapExtractor, share_features_extractor=True,
                             net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=nn.Tanh,
                             log_std_init=args.log_std_init)
        model = PPO("MultiInputPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma,
                    gae_lambda=args.gae_lambda, clip_range=0.2, ent_coef=args.ent_coef, vf_coef=0.5,
                    max_grad_norm=0.5, target_kl=args.target_kl, policy_kwargs=policy_kwargs,
                    tensorboard_log=str(out/"tb"), seed=args.seed, device=device, verbose=1)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"policy parameters: {n_params:,}")
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every//args.n_envs, 1), save_path=str(out/"ckpt"),
                              name_prefix="model")
    model.learn(total_timesteps=args.total_steps, callback=[ckpt, *callbacks],
                reset_num_timesteps=args.resume is None, tb_log_name="ppo")
    model.save(out/"model_final")
    venv.close()
    print("saved", out/"model_final.zip")


if __name__ == "__main__":
    main()
