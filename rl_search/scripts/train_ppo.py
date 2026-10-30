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
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/rl"))
    args = ap.parse_args()

    import torch
    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from rl_search.rl_gym import MultiMapExtractor, RLSearchGym

    out = args.out/args.tag
    out.mkdir(parents=True, exist_ok=True)
    rl = parse_kv(args.rl) | {"stage": args.stage}
    env_kwargs = dict(preset=args.preset, overrides=args.set, rl=rl)
    (out/"config.json").write_text(json.dumps(vars(args) | {"rl_cfg": rl}, indent=1, default=str))

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    info_keys = ("pos_policy", "area_policy_km2", "exited", "t_s", "energy_left")
    venv = make_vec_env(RLSearchGym, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs,
                        vec_env_cls=vec_cls, monitor_kwargs=dict(info_keywords=info_keys))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device={device} n_envs={args.n_envs} torch={torch.__version__}")

    class EpisodeStats(BaseCallback):
        """Episode POS, area, map-exit rate and course-target gap to TensorBoard (9.9, 1.6 v8)."""

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                self.logger.record_mean("step/gap_abs_deg", abs(info.get("gap_deg", 0.0)))
                self.logger.record_mean("step/shield", float(info.get("shield", False)))
                ep = info.get("episode")
                if ep:
                    for k in info_keys:
                        if k in ep:
                            self.logger.record_mean(f"episode/{k}", float(ep[k]))
            return True

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
    model.learn(total_timesteps=args.total_steps, callback=[ckpt, EpisodeStats()],
                reset_num_timesteps=args.resume is None, tb_log_name="ppo")
    model.save(out/"model_final")
    venv.close()
    print("saved", out/"model_final.zip")


if __name__ == "__main__":
    main()
