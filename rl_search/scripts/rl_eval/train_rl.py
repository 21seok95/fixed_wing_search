"""PPO training of the 5 s heading policy on the Amboy map (docs/rl_mdp_design.md, sections 6-8, decisions of 2026-11-10).

- Env: rl_search.rl_real (SAROPS reference flight, camera and score; heading-up maps; reward (dA + w dPOS) x 100), training scenarios from
  the pre-built pool (scripts/rl_eval/make_pool.py), curriculum starts. The terrain is loaded once and the env workers are forked.
- Network: rl_gym.RealMapExtractor (stride-1 first layer on the 1.6 km map), shared by the policy and the value function; value output bias
  set to the mean discounted return of a smooth random policy (``--value-init auto``).
- Every ``--eval-every`` steps: checkpoint, and the 5 validation cases of the current stage flown deterministically in ``--eval-workers``
  forked processes while training goes on; scored on hold-out particles like the SAROPS references and saved as web runs
  (``eval-val-s<stage>-k<k>-rl-<step>``). Stage change when the first of: the mean of the last 3 evaluations reaches ``--pass-ratio`` of
  SAROPS, it rose less than 1 % over the last 5 evaluations, or the stage budget is spent. The best checkpoint of each stage is kept.
- TensorBoard: <out>/<tag>/tb (SB3 + episode A / POS / total) and <out>/<tag>/tb_eval (evaluation, curriculum).

    python -m rl_search.scripts.rl_eval.train_rl --tag amboy_s1 --stop-at 1000000
    python -m rl_search.scripts.rl_eval.train_rl --tag amboy_s1 --resume latest --stop-at 3000000
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per env process
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

STAGE_BUDGET = (3_000_000, 3_000_000, 6_000_000)


def make_env(stage: int, seed: int):
    def f():
        from stable_baselines3.common.monitor import Monitor
        from rl_search.rl_real import RealRLConfig, make_gym
        return Monitor(make_gym(RealRLConfig(stage=stage), seed=seed), info_keywords=("A", "POS", "total", "stage"))
    return f


def random_value_init(venv, gamma: float, steps: int, rng) -> float:
    """Mean discounted return over the states of finished episodes of a smooth random policy (scaled reward)."""
    n = venv.num_envs
    venv.reset()
    a = np.zeros((n, 1), np.float32)
    rews = [[] for _ in range(n)]
    rets = []
    for _ in range(steps):
        a = np.clip(0.8*a + 0.5*rng.normal(size=a.shape), -1, 1).astype(np.float32)
        _, r, done, _ = venv.step(a)
        for i in range(n):
            rews[i].append(float(r[i]))
            if done[i]:
                g, out = 0.0, []
                for x in reversed(rews[i]):
                    g = x + gamma*g
                    out.append(g)
                rets += out
                rews[i] = []
    return float(np.mean(rets)) if rets else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=512, help="per env per update (8 x 512 = 4096)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.25)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--value-init", default="auto", help="'auto' (random-policy returns), a number, or 'none'")
    ap.add_argument("--stage", type=int, default=1, help="curriculum stage to start in (new runs)")
    ap.add_argument("--stage-budget", type=int, nargs=3, default=list(STAGE_BUDGET))
    ap.add_argument("--pass-ratio", type=float, default=0.9)
    ap.add_argument("--stop-at", type=int, default=sum(STAGE_BUDGET), help="stop at this many total steps (checkpoint kept)")
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'latest' for this tag's last checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from multiprocessing import get_context
    from rl_search.rl_real import preload
    out = args.out/args.tag
    (out/"ckpt").mkdir(parents=True, exist_ok=True)
    state_path = out/"state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else \
        dict(stage=args.stage, stage_start=0, history={}, best={}, switches=[])
    preload("amboy")                                     # terrain and road network, shared by every forked process

    from rl_search.rl_real_eval import eval_job
    pool = get_context("fork").Pool(args.eval_workers)          # forked before the env workers and before CUDA is touched
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([make_env(state["stage"], args.seed*1000 + i) for i in range(args.n_envs)], start_method="fork")

    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from rl_search.rl_gym import RealMapExtractor
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    resume = None
    if args.resume:
        resume = Path(args.resume) if args.resume != "latest" else max((out/"ckpt").glob("model_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if resume:
        model = PPO.load(resume, env=venv, device=device)
        model.tensorboard_log = str(out/"tb")
        print("resumed", resume, "stage", state["stage"], flush=True)
    else:
        pk = dict(features_extractor_class=RealMapExtractor, share_features_extractor=True, net_arch=dict(pi=[128, 128], vf=[128, 128]),
                  activation_fn=nn.Tanh, log_std_init=args.log_std_init)
        model = PPO("MultiInputPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                    gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm, target_kl=args.target_kl, policy_kwargs=pk, tensorboard_log=str(out/"tb"),
                    seed=args.seed, device=device, verbose=1)
        if args.value_init != "none":
            v0 = random_value_init(venv, args.gamma, 800, np.random.default_rng(args.seed)) if args.value_init == "auto" else float(args.value_init)
            with torch.no_grad():
                model.policy.value_net.bias.fill_(v0)
            state["value_init"] = v0
            print(f"value output bias set to {v0:.2f} (mean discounted return of a smooth random policy)", flush=True)
    (out/"config.json").write_text(json.dumps(vars(args) | {"parameters": sum(p.numel() for p in model.policy.parameters())}, indent=1, default=str))
    tbe = SummaryWriter(str(out/"tb_eval"))

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = None
            self.pending = []

        def _on_training_start(self):
            self.next_eval = (self.num_timesteps//args.eval_every + 1)*args.eval_every

        def _on_rollout_end(self):
            eps = list(self.model.ep_info_buffer)
            for key in ("A", "POS", "total"):
                vals = [e[key] for e in eps if key in e]
                if vals:
                    self.logger.record(f"episode/{key}", float(np.mean(vals)))
            self.logger.record("curriculum/stage", state["stage"])

        def _on_step(self) -> bool:
            self._collect()
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                self._submit()
            if self.num_timesteps >= args.stop_at:
                return False
            return True

        def _submit(self):
            step = int(self.num_timesteps)
            path = out/"ckpt"/f"model_{step}.zip"
            self.model.save(path)
            st = state["stage"]
            jobs = [(str(path), "val", st, k, step) for k in range(5)]
            self.pending.append((step, st, path, pool.map_async(eval_job, jobs)))
            state_path.write_text(json.dumps(state, indent=1))

        def _collect(self, wait: bool = False):
            keep = []
            for step, st, path, res in self.pending:
                if not (wait or res.ready()):
                    keep.append((step, st, path, res))
                    continue
                try:
                    rows = res.get()
                except Exception as exc:                 # a failed evaluation must not stop the training
                    print(f"evaluation at {step} failed: {exc!r}", flush=True)
                    continue
                self._record(step, st, path, rows)
            self.pending = keep

        def _record(self, step, st, path, rows):
            rl = np.array([r["holdout"]["total"] for r in rows])
            sar = np.array([r["sarops"]["total"] for r in rows])
            ratio = float(rl.mean()/sar.mean())
            hist = state["history"].setdefault(str(st), [])
            hist.append(dict(step=step, ratio=ratio, rl=rl.tolist(), sarops=sar.tolist(), pos=[r["holdout"]["pos"] for r in rows],
                             A=[r["holdout"]["A"] for r in rows], belief=[r["belief"]["total"] for r in rows]))
            tbe.add_scalar(f"eval_s{st}/ratio_to_sarops", ratio, step)
            tbe.add_scalar(f"eval_s{st}/rl_total", float(rl.mean()), step)
            tbe.add_scalar(f"eval_s{st}/rl_pos", float(np.mean(hist[-1]["pos"])), step)
            tbe.add_scalar(f"eval_s{st}/rl_A", float(np.mean(hist[-1]["A"])), step)
            tbe.add_scalar(f"eval_s{st}/paired_diff", float((rl - sar).mean()), step)
            for k in range(len(rl)):
                tbe.add_scalar(f"eval_s{st}/case{k + 1}", float(rl[k]), step)
            tbe.add_scalar("curriculum/stage", st, step)
            tbe.flush()
            recent = [h["ratio"] for h in hist[-3:]]
            score = float(np.mean(recent))
            best = state["best"].get(str(st))
            if best is None or score > best["score"]:
                state["best"][str(st)] = dict(step=step, score=score)
                shutil.copy(path, out/f"best_stage{st}.zip")
            print(f"eval stage {st} at {step:,}: RL {rl.mean():.3f} vs SAROPS {sar.mean():.3f} (ratio {ratio:.3f}, last-3 mean {score:.3f}), "
                  f"per case {np.round(rl, 3).tolist()}, {max(r['sec'] for r in rows):.0f} s", flush=True)
            if st == state["stage"]:
                self._maybe_advance(hist)
            state_path.write_text(json.dumps(state, indent=1))

        def _maybe_advance(self, hist):
            st = state["stage"]
            if st >= 3:
                return
            used = self.num_timesteps - state["stage_start"]
            means = [float(np.mean([h["ratio"] for h in hist[max(0, i - 2):i + 1]])) for i in range(len(hist))]
            why = None
            if len(hist) >= 3 and means[-1] >= args.pass_ratio:
                why = f"last-3 mean {means[-1]:.3f} >= {args.pass_ratio}"
            elif len(hist) >= 8 and means[-1] <= means[-6]*1.01:
                why = f"plateau: last-3 mean {means[-1]:.3f}, 5 evaluations ago {means[-6]:.3f}"
            elif used >= args.stage_budget[st - 1]:
                why = f"stage budget {args.stage_budget[st - 1]:,} spent"
            if why:
                state["switches"].append(dict(step=int(self.num_timesteps), from_stage=st, to_stage=st + 1, why=why))
                state["stage"], state["stage_start"] = st + 1, int(self.num_timesteps)
                self.training_env.env_method("set_stage", st + 1)
                print(f"curriculum: stage {st} -> {st + 1} at {self.num_timesteps:,} ({why})", flush=True)

        def _on_training_end(self):
            self._collect(wait=True)

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.stop_at, callback=Curriculum(), reset_num_timesteps=resume is None, tb_log_name="ppo")
    final = out/"ckpt"/f"model_{model.num_timesteps}.zip"
    model.save(final)
    state_path.write_text(json.dumps(state, indent=1))
    print(f"stopped at {model.num_timesteps:,} steps after {(time.perf_counter() - t0)/3600:.2f} h, stage {state['stage']}, saved {final}", flush=True)
    pool.close()
    venv.close()


if __name__ == "__main__":
    main()
