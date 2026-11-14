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

Fine-tuning (2026-11-11, the seen-particle A): a new run from another model's weights, here amboy_v1 in stage 3 on this Mac:

    python -m rl_search.scripts.rl_eval.train_rl --tag amboy_seen_ft --init rl_search/data/rl_real/amboy_v1/best_stage3.zip --stage 3 \
        --n-envs 6 --eval-workers 2 --device cpu --stop-at 1000000
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


def make_env(stage: int, seed: int, settings: dict | None = None, straight_bonus: float = 0.0, roll_cost: float = 0.0, roll_deg: float = 10.0,
             diverse: bool = False, mask_cost: float = 0.0, straight_raw: bool = False, multi: bool = False, type_weights: str | None = None):
    def f():
        from stable_baselines3.common.monitor import Monitor
        from rl_search.rl_real import POOL_DIR, POOL_DIR_DIVERSE, RealRLConfig, make_gym, pool_dir_multi
        pool = pool_dir_multi(settings) if multi else POOL_DIR_DIVERSE if diverse else POOL_DIR
        cfg = RealRLConfig(stage=stage, settings=dict(settings or {}), straight_bonus=straight_bonus, roll_reversal_cost=roll_cost,
                           roll_reversal_deg=roll_deg, diverse=diverse, pool=pool, mask_cost=mask_cost, straight_on_raw=straight_raw,
                           multi=multi, type_weights=type_weights)
        return Monitor(make_gym(cfg, seed=seed), info_keywords=("A", "POS", "total", "bonus", "roll_rev", "mask_cut", "stage", "typ", "look"))
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
    ap.add_argument("--envs-per-proc", type=int, default=1,
                    help="environments per worker process (rl_search.vec_group): 10 envs with 2 per process keep 5 cores busy")
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
    ap.add_argument("--save-every", type=int, default=0, help="keep checkpoints and evaluation web runs only every this many steps (0 = every evaluation)")
    ap.add_argument("--eval-workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'latest' for this tag's last checkpoint")
    ap.add_argument("--init", default=None, help="new run from this model's weights (fine-tuning); the settings above apply, steps start at 0")
    ap.add_argument("--area-def", default="window", choices=("window", "realtime", "realtime_q", "realtime_t", "seen"),
                    help="A of the reward and the evaluation (realtime: the window A every 10 s; seen: x 80, POD 0.5)")
    ap.add_argument("--straight-bonus", type=float, default=0.0, help="reward per step x (1 - |a|) for flying straight (training only)")
    ap.add_argument("--blur-tau", type=float, default=0.0, help="turn-rate factor of the POD (and A4 quality): 0 = the linear one (D1), > 0 = exp(-w/tau) (D2)")
    ap.add_argument("--blur-dead-roll", type=float, default=0.0, help="with --blur-tau: no turn-rate loss up to this roll's turn rate (D3)")
    ap.add_argument("--blur-tau-low", type=float, default=0.0, help="with --blur-dead-roll: exp(-w/value) below that rate too (D4 = D2 x D3)")
    ap.add_argument("--area-sigma", type=float, default=0.0, help="blur width (m) of the real-time A region (0 = 150 m)")
    ap.add_argument("--area-sigma-off", type=float, default=0.0,
                    help="blur width (m) of the off-road particles in the A region (0 = same as --area-sigma); with --multi the pool is "
                         "rl_pool_multi_o<value> (make_pool --multi --area-sigma-off)")
    ap.add_argument("--handover-km", type=float, default=0.0,
                    help="stage 3: fly by rule toward the nearest remaining belief until 10 %% of it is within this distance, then the policy "
                         "takes over (0 = hand over 2 min after the launch)")
    ap.add_argument("--no-edge", action="store_true", help="no edge mask and no end of the flight off the map; the maps show 0 off the map")
    ap.add_argument("--area-img-min", type=float, default=0.0, help="a cell is searched only where the POD's image-position factor is >= this")
    ap.add_argument("--big-map-km", type=float, default=0.0, help="side (km) of the largest observation map (0 = 51.2), same 64 x 64 cells")
    ap.add_argument("--rel-maps", action="store_true",
                    help="every map gets a last channel: the remaining POS now as a share of the remaining belief (with --init from a model "
                         "without it, the new channel's first-layer weights start at zero)")
    ap.add_argument("--a-weight", type=float, default=1.0,
                    help="reward and score a_weight x A + w x POS; 0 = A off (reward POS only, no A channel or A state; A still reported)")
    ap.add_argument("--pos-norm", action="store_true",
                    help="each map's POS channels divided by its largest cell (log, floor 3 particles at the mean weight), scales in the vector")
    ap.add_argument("--look-w", type=float, default=0.0,
                    help="look reward (A'): this x particles seen per pass / n, every pass again (image centre, gentle turn); training only")
    ap.add_argument("--type-fixed", default="", help="fixed shares of typed training, e.g. desert=0.5; the other types adapt in the rest")
    ap.add_argument("--use-sde", action="store_true", help="state-dependent exploration (gSDE): smooth noise kept for --sde-freq decisions")
    ap.add_argument("--sde-freq", type=int, default=8, help="decisions between new gSDE noise draws (8 = 40 s)")
    ap.add_argument("--fp-rate", action="store_true",
                    help="the 1.6 km map's camera channel is multiplied by the detection's turn-rate factor at the current roll")
    ap.add_argument("--area-roll-max", type=float, default=20.0, help="realtime_t (A4t): roll limit (deg) for a searched cell")
    ap.add_argument("--pos-w", type=float, default=1.0, help="score and reward A + w POS (the SAROPS references must use the same w)")
    ap.add_argument("--mask-cost", type=float, default=0.0, help="reward taken per unit the edge mask cuts off the action (training only)")
    ap.add_argument("--straight-raw", action="store_true", help="straight bonus from the policy's own action, not the masked one")
    ap.add_argument("--diverse", action="store_true", help="diverse training scenarios and stage 3 starts (rl_eval_set.DIVERSE)")
    ap.add_argument("--eval-split", default="val", choices=("val", "dval", "mval", "nval"),
                    help="validation set of the periodic evaluation (nval: the mval cases with targets that never stop, for --no-stop)")
    ap.add_argument("--no-stop", action="store_true",
                    help="R10 (user 2026-11-14): typed targets that never stop and keep a steady speed of their own (rl_eval_set.NOSTOP); "
                         "pool rl_pool_multi_ns (make_pool --multi --no-stop --no-area), validate on nval")
    ap.add_argument("--no-area", action="store_true", help="no A at all in the env, not even for the report (needs --a-weight 0)")
    ap.add_argument("--obs-r10", action="store_true",
                    help="R10 map channels: particle positions now and after the lead (unweighted), POS now normalised per map and absolute, "
                         "visibility, camera on the 1.6 km map (rl_real.RealRLEnv.observation, needs --a-weight 0)")
    ap.add_argument("--multi", action="store_true",
                    help="typed training scenarios (road / mixed / desert, rl_eval_set.MULTI, R5): equal shares first; after each evaluation on "
                         "mval the type furthest behind SAROPS gets 0.05 more share (0.2-0.5), written to <run>/type_weights.json")
    ap.add_argument("--roll-reversal-cost", type=float, default=0.0, help="reward taken per roll reversal (B2, training only)")
    ap.add_argument("--roll-reversal-deg", type=float, default=10.0, help="|roll| that sets a side for the reversal count")
    ap.add_argument("--device", default="cuda", help="cuda, mps (Apple GPU) or cpu; falls back to cpu when the device is missing")
    args = ap.parse_args()

    from multiprocessing import get_context
    from rl_search.rl_real import preload
    out = args.out/args.tag
    (out/"ckpt").mkdir(parents=True, exist_ok=True)
    state_path = out/"state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else \
        dict(stage=args.stage, stage_start=0, history={}, best={}, switches=[])
    preload("amboy")                                     # terrain and road network, shared by every forked process
    from rl_search.scripts.rl_eval.sarops_ref import DEFS
    area_over = ({} if args.area_def == "window" else dict(DEFS[args.area_def])) | ({"blur_tau": args.blur_tau} if args.blur_tau else {}) \
        | ({"area_roll_max": args.area_roll_max} if args.area_def == "realtime_t" else {}) | ({"area_w": args.pos_w} if args.pos_w != 1 else {}) \
        | ({"blur_dead_roll": args.blur_dead_roll} if args.blur_tau and args.blur_dead_roll else {}) \
        | ({"blur_tau_low": args.blur_tau_low} if args.blur_tau and args.blur_dead_roll and args.blur_tau_low else {}) \
        | ({"area_sigma": args.area_sigma} if args.area_sigma and args.area_sigma != 150 else {}) \
        | ({"area_sigma_off": args.area_sigma_off} if args.area_sigma_off else {}) \
        | ({"handover_km": args.handover_km} if args.handover_km else {}) | ({"edge_mode": "none", "off_map_zero": True} if args.no_edge else {}) \
        | ({"area_img_min": args.area_img_min} if args.area_img_min else {}) | ({"big_map_m": args.big_map_km*1000.0} if args.big_map_km else {}) \
        | ({"rel_maps": True} if args.rel_maps else {}) | ({"fp_rate": True} if args.fp_rate else {}) \
        | ({"a_weight": args.a_weight} if args.a_weight != 1.0 else {}) | ({"pos_norm": True} if args.pos_norm else {}) \
        | ({"look_w": args.look_w} if args.look_w else {}) | ({"no_stop": True} if args.no_stop else {}) \
        | ({"no_area": True} if args.no_area else {}) | ({"obs_r10": True} if args.obs_r10 else {})
    if args.multi:
        from pathlib import Path as _P
        from rl_search.rl_real import pool_dir_multi
        if not any(_P(pool_dir_multi(area_over)).glob("*_s*.pkl")):
            raise SystemExit(f"no pool at {pool_dir_multi(area_over)}: build it with make_pool --multi"
                             + (" --no-stop --no-area" if args.no_stop else f" --area-sigma-off {args.area_sigma_off:g}" if args.area_sigma_off else ""))
    if args.no_stop and args.eval_split != "nval":
        raise SystemExit("--no-stop trains on targets that never stop: validate on them too (--eval-split nval)")

    from rl_search.rl_real_eval import eval_job
    pool = get_context("fork").Pool(args.eval_workers)          # forked before the env workers and before CUDA is touched
    from stable_baselines3.common.vec_env import SubprocVecEnv
    tw_path = out/"type_weights.json"
    if args.multi:
        from rl_search.rl_eval_set import MULTI
        fixed = {k: float(v) for k, v in (kv.split("=") for kv in args.type_fixed.split(",") if kv)}     # e.g. desert=0.5 (user 2026-11-13)
        rest = [t for t in MULTI["types"] if t not in fixed]
        state.setdefault("type_weights", {t: fixed.get(t, (1 - sum(fixed.values()))/max(len(rest), 1)) for t in MULTI["types"]})
        state["type_fixed"] = fixed
        tw_path.write_text(json.dumps(state["type_weights"]))
    env_fns = [make_env(state["stage"], args.seed*1000 + i, area_over, args.straight_bonus, args.roll_reversal_cost,
                        args.roll_reversal_deg, args.diverse, args.mask_cost, args.straight_raw, args.multi,
                        str(tw_path) if args.multi else None) for i in range(args.n_envs)]
    if args.envs_per_proc > 1:                       # several envs per worker process (rl_search.vec_group, user 2026-11-12)
        from rl_search.vec_group import GroupedSubprocVecEnv
        venv = GroupedSubprocVecEnv(env_fns, args.envs_per_proc, start_method="fork")
    else:
        venv = SubprocVecEnv(env_fns, start_method="fork")

    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from rl_search.rl_gym import RealMapExtractor
    ok = dict(cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available(), cpu=True)
    device = args.device if ok.get(args.device, False) else "cpu"
    resume = None
    if args.resume:
        resume = Path(args.resume) if args.resume != "latest" else max((out/"ckpt").glob("model_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if resume:
        model = PPO.load(resume, env=venv, device=device)
        model.tensorboard_log = str(out/"tb")
        print("resumed", resume, "stage", state["stage"], flush=True)
    elif args.init:
        from stable_baselines3.common.utils import ConstantSchedule
        over = dict(learning_rate=args.lr, lr_schedule=ConstantSchedule(args.lr), clip_range=ConstantSchedule(args.clip), n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl, tensorboard_log=str(out/"tb"), seed=args.seed)
        old_space = PPO.load(args.init, device="cpu").observation_space
        if old_space == venv.observation_space:
            model = PPO.load(args.init, env=venv, device=device, custom_objects=over)
        else:                                      # more map channels (--rel-maps): copy the weights, new input channels start at zero
            from rl_search.rl_gym import load_expanded
            model, grown = load_expanded(args.init, venv, device, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                                         n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip,
                                         ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                                         tensorboard_log=str(out/"tb"), seed=args.seed, verbose=1)
            print("grown input channels:", "; ".join(grown), flush=True)
        state["init"] = str(args.init)
        print("initialised from", args.init, "stage", state["stage"], flush=True)
    else:
        pk = dict(features_extractor_class=RealMapExtractor, share_features_extractor=True, net_arch=dict(pi=[128, 128], vf=[128, 128]),
                  activation_fn=nn.Tanh, log_std_init=args.log_std_init)
        model = PPO("MultiInputPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                    gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm, target_kl=args.target_kl, policy_kwargs=pk, tensorboard_log=str(out/"tb"),
                    seed=args.seed, device=device, verbose=1, use_sde=args.use_sde, sde_sample_freq=args.sde_freq if args.use_sde else -1)
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
            for key in ("A", "POS", "total", "bonus", "roll_rev", "mask_cut", "look"):
                vals = [e[key] for e in eps if key in e]
                if vals:
                    self.logger.record(f"episode/{key}", float(np.mean(vals)))
            if args.multi:                                       # training episodes per target type
                for t in MULTI["types"]:
                    vals = [e["total"] for e in eps if e.get("typ") == t]
                    if vals:
                        self.logger.record(f"episode_type/{t}", float(np.mean(vals)))
                for t, v in state["type_weights"].items():
                    self.logger.record(f"type_share/{t}", float(v))
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
            label = f"{args.tag}-{step}" if (args.init or state.get("init") or args.multi) else step   # web run names stay apart between runs
            from rl_search.rl_eval_set import MULTI_N
            # --save-every (user 2026-11-12, disk): the checkpoint and the web runs are kept only when a multiple of it was passed;
            # the evaluation itself still runs every --eval-every (it also drives the curriculum and the type shares)
            keep = not args.save_every or step//args.save_every > (step - args.eval_every)//args.save_every
            jobs = [(str(path), args.eval_split, st, k, label, bool(keep), area_over) for k in range(MULTI_N.get(args.eval_split, 5))]
            self.pending.append((step, st, path, pool.map_async(eval_job, jobs), keep))
            state_path.write_text(json.dumps(state, indent=1))

        def _collect(self, wait: bool = False):
            pending = []
            for step, st, path, res, keep in self.pending:
                if not (wait or res.ready()):
                    pending.append((step, st, path, res, keep))
                    continue
                try:
                    rows = res.get()
                    self._record(step, st, path, rows)
                except Exception as exc:                 # a failed evaluation must not stop the training
                    print(f"evaluation at {step} failed: {exc!r}", flush=True)
                if not keep:
                    Path(path).unlink(missing_ok=True)   # best_stage<k>.zip was copied in _record if this one was the best
            self.pending = pending

        def _record(self, step, st, path, rows):
            rl = np.array([r["holdout"]["total"] for r in rows])
            if any(r["sarops"] is None for r in rows):            # SAROPS references of this split / A not made: keep the scores only
                print(f"eval stage {st} at {step:,}: RL {rl.mean():.3f} (no SAROPS reference), per case {np.round(rl, 3).tolist()}", flush=True)
                return
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
            if args.multi:
                self._type_shares(step, rows, rl, sar, hist)
            if st == state["stage"]:
                self._maybe_advance(hist)
            state_path.write_text(json.dumps(state, indent=1))

        def _type_shares(self, step, rows, rl, sar, hist):
            """Typed training (user 2026-11-12: equal shares, then a little more of the type that does worst): per-type ratio to SAROPS on
            the validation cases (case k has type k % 3), smoothed over the last two evaluations; from the third evaluation on, the type
            furthest behind gets +0.05 share, taken from the type furthest ahead among those still above the 0.2 floor (shares 0.2-0.5)."""
            from rl_search.rl_eval_set import case_type
            types = MULTI["types"]
            ks = [r["k"] for r in rows]
            per = {}
            for t in types:
                idx = [i for i, k in enumerate(ks) if case_type(args.eval_split, k) == t]
                if idx:
                    per[t] = float(rl[idx].mean()/sar[idx].mean())
            hist[-1]["type_ratio"] = per
            recent = [h["type_ratio"] for h in hist[-2:] if "type_ratio" in h]
            sm = {t: float(np.mean([r[t] for r in recent if t in r])) for t in per}
            w = dict(state["type_weights"])
            fixed = state.get("type_fixed", {})                 # --type-fixed: those shares stay, the others adapt in the rest
            free = [t for t in types if t not in fixed]
            cap = min(0.5, 1 - sum(fixed.values()) - 0.2*(len(free) - 1))
            if len(hist) >= 3 and len(sm) == len(types) and len(free) > 1:
                worst = min(free, key=sm.get)
                donors = [t for t in free if t != worst and w[t] > 0.2 + 1e-9]   # (fix 11-12: a type at the 0.2 floor no longer blocks the move)
                best = max(donors, key=sm.get) if donors else None
                if best is not None and sm[best] - sm[worst] > 0.02:
                    up = min(0.05, cap - w[worst], w[best] - 0.2)
                    if up > 0:
                        w[worst] += up; w[best] -= up
            state["type_weights"] = w
            tw_path.write_text(json.dumps(w))
            for t in per:
                tbe.add_scalar(f"eval_type/{t}_ratio", per[t], step)
                tbe.add_scalar(f"type_share/{t}", w[t], step)
            tbe.flush()
            print("  by type (ratio to SAROPS, smoothed -> share): " +
                  ", ".join(f"{t} {per.get(t, float('nan')):.3f} ({sm.get(t, float('nan')):.3f}) -> {w[t]:.2f}" for t in types), flush=True)

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
    model.learn(total_timesteps=args.stop_at, callback=Curriculum(), reset_num_timesteps=resume is None, tb_log_name="ppo")   # --init: from 0
    final = out/"ckpt"/f"model_{model.num_timesteps}.zip"
    model.save(final)
    state_path.write_text(json.dumps(state, indent=1))
    print(f"stopped at {model.num_timesteps:,} steps after {(time.perf_counter() - t0)/3600:.2f} h, stage {state['stage']}, saved {final}", flush=True)
    pool.close()
    venv.close()


if __name__ == "__main__":
    main()
