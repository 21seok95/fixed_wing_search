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
             diverse: bool = False, mask_cost: float = 0.0, straight_raw: bool = False, multi: bool = False, type_weights: str | None = None,
             region: str = "amboy", reuse: int = 4, pool_slot: tuple | None = None):
    def f():
        from stable_baselines3.common.monitor import Monitor
        from rl_search.rl_real import POOL_DIR, POOL_DIR_DIVERSE, RealRLConfig, make_gym, pool_dir_multi
        pool = pool_dir_multi(settings, region) if multi else POOL_DIR_DIVERSE if diverse else POOL_DIR
        cfg = RealRLConfig(region=region, stage=stage, settings=dict(settings or {}), straight_bonus=straight_bonus, roll_reversal_cost=roll_cost,
                           roll_reversal_deg=roll_deg, diverse=diverse, pool=pool, mask_cost=mask_cost, straight_on_raw=straight_raw,
                           multi=multi, type_weights=type_weights, reuse=reuse, pool_slot=pool_slot)
        return Monitor(make_gym(cfg, seed=seed), info_keywords=("A", "POS", "total", "bonus", "roll_rev", "mask_cut", "stage", "typ", "look",
                                                                 "region", "start", "launch_km"))
    return f


def parse_ring(spec: str) -> list:
    """Launch-distance stages '10-10,10-20,10-40' (km from the LKP) -> [(10000.0, 10000.0), (10000.0, 20000.0), (10000.0, 40000.0)] in m."""
    out = []
    for part in spec.split(","):
        lo, hi = (float(x)*1000.0 for x in part.strip().split("-"))
        if not 0 < lo <= hi:
            raise ValueError(part)
        out.append((lo, hi))
    return out


def ring_next(blocks: list, used: int, rise: float, min_steps: int, max_steps: int) -> str | None:
    """Why the launch-distance stage moves on now, or None (user 2026-11-18: when the launch flights' POS stops rising). ``blocks``: the
    stage's mean launch POS per block of steps, oldest first; ``used``: steps run in the stage. On after ``min_steps`` when the last block
    is less than ``rise`` (relative) above the block before it; after ``max_steps`` in any case."""
    if used >= max_steps:
        return f"stage budget {max_steps:,} steps spent"
    if used >= min_steps and len(blocks) >= 2 and blocks[-1] < blocks[-2]*(1.0 + rise):
        return f"launch POS {blocks[-2]:.4f} -> {blocks[-1]:.4f}, less than {rise:.0%} higher"
    return None


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
    ap.add_argument("--extra-map-km", type=float, default=0.0,
                    help="R13 (user 2026-11-16): a fifth observation map of this side (km, e.g. 102.4), same channels and 64 x 64 cells")
    ap.add_argument("--extra-map-lead", type=float, default=600.0, help="prediction lead (s) of the fifth map's lead channel")
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
    ap.add_argument("--eval-split", nargs="+", default=["val"], choices=("val", "dval", "mval", "nval", "bval", "v2a", "v2b", "v2j"),
                    help="validation sets of the periodic evaluation (nval: the mval cases with targets that never stop, for --no-stop; "
                         "bval: Barstow cases, never stop, varied routes); with several sets the ratio to SAROPS is over all their cases")
    ap.add_argument("--eval-types", nargs="+", default=None, choices=("road", "mixed", "desert"),
                    help="validate on the cases of these target types only (case k has type k %% 3; R14 type specialists, user 2026-11-16: "
                         "a third of the evaluation time)")
    ap.add_argument("--teams", type=int, nargs="+", default=None,
                    help="multi-UAV (docs/multi_agent_design.md, user 2026-11-19/20): one world per entry with that many UAVs (e.g. 2 3 4 2 3 4), every UAV "
                         "is one slot of the vector env (shared policy, reward and teammate inputs by --ma-design, safety filter, rl_search.rl_multi); --n-envs is ignored, the "
                         "periodic single-UAV evaluation is off (checkpoints only); steps before a launch / after the battery are masked")
    ap.add_argument("--worlds-per-proc", type=int, default=1,
                    help="multi-UAV: worlds per worker process (user 2026-11-23: 3-UAV worlds x 20 on the Mac; the terrain is held once per process, "
                         "the worlds of a process step one after another)")
    ap.add_argument("--ma-design", choices=("m1", "m2"), default="m2",
                    help="multi-UAV design (rl_multi.DESIGNS): m1 = the M1 run (team reward, 150 / 200 m, gaps 5 / 10 / 15 s, tail channel + 7-entry "
                         "vector); m2 = user 2026-11-20 ~ 11-23 (own-share reward, 100 / 250 m with the distance penalty only while closing, gaps "
                         "5-10 s per UAV, small maps claim + danger zone, large maps plan B, no teammate vector)")
    ap.add_argument("--sym-policy", action="store_true",
                    help="left-right symmetric policy (rl_search/sym_policy.py, user 2026-11-23): mean(mirror(o)) = -mean(o), value(mirror(o)) = "
                         "value(o); with --init the old weights are kept; a resumed run keeps its saved policy class")
    ap.add_argument("--sequential", action="store_true",
                    help="multi-UAV: sequential decisions (user 2026-11-23 \"순차 결정\", rl_multi sequential=True, rl_search/seq_ppo.py): within a 5 s "
                         "step the UAVs of a world decide one after another in a random order, each seeing the belief after the ones before it and "
                         "where they flew; one more vector entry (the turn, 0 first .. 1 last); design m2 only")
    ap.add_argument("--sat-stop", type=float, default=0.0,
                    help="multi-UAV: save and stop when, two rollouts in a row, more than this share of the policy's mean actions lies beyond "
                         "the action range (user 2026-11-23, after M2's collapse; 0 = off, the share is logged anyway as multi/act_sat)")
    ap.add_argument("--value-warmup", type=int, default=0,
                    help="the first N PPO updates train only the value head (policy and the shared CNN frozen): after a change of reward "
                         "form or gamma (user 2026-11-23: 5 for the next multi-UAV run)")
    ap.add_argument("--init-std", type=float, default=None,
                    help="with --init: set the policy's action noise std to this value after loading (user 2026-11-20: R15 ended at 0.083 and "
                         "M1 kept it; 0.3 lets the multi-UAV run try other directions)")
    ap.add_argument("--regions", nargs="+", default=["amboy"],
                    help="maps of the training envs (R10 on two maps: amboy mojave); env i is on regions[i %% len], each with its own pool")
    ap.add_argument("--route-var", action="store_true",
                    help="varied route settings of the targets (rl_eval_set.ROUTE, user 2026-11-14); pool <region>/rl_pool_multi_ns_rt")
    ap.add_argument("--no-stop", action="store_true",
                    help="R10 (user 2026-11-14): typed targets that never stop and keep a steady speed of their own (rl_eval_set.NOSTOP); "
                         "pool rl_pool_multi_ns (make_pool --multi --no-stop --no-area), validate on nval")
    ap.add_argument("--no-area", action="store_true", help="no A at all in the env, not even for the report (needs --a-weight 0)")
    ap.add_argument("--maps7", action="store_true",
                    help="seven observation maps 1.6 / 3.2 / 6.4 / 12.8 / 25.6 / 51.2 / 102.4 km (rl_real.MAPS7, user 2026-11-18)")
    ap.add_argument("--no-vis-obs", action="store_true",
                    help="no ground visibility in the observation: no visibility channel, POS channels unweighted (user 2026-11-18)")
    ap.add_argument("--obs-u8", action="store_true",
                    help="observation maps as uint8 0..255 (rl_real.maps_to_u8; rollout memory and transfer 1/4, user 2026-11-23: the Mac trains "
                         "with it, the local PC in float32); needs the R15 maps (--obs-r10, --a-weight 0, --no-vis-obs or --no-edge)")
    ap.add_argument("--launch-policy", action="store_true",
                    help="the policy flies from the launch: no rule transit (no 120 s toward the LKP, no --handover-km; user 2026-11-18)")
    ap.add_argument("--scen-v2", action="store_true",
                    help="the redesigned scenarios (rl_eval_set.V2, docs/scenario_redesign.md): typed pools <region>/rl_pool_v2, launch at the "
                         "report time 10-40 km from the LKP, particles leaving the map kept; implies --multi, needs --no-area and --stage 3")
    ap.add_argument("--types", default="",
                    help="target types of the training, e.g. road (user 2026-11-18: Chungju 120 km road type only); default with --scen-v2 every "
                         "type of the map's terrain (desert: road mixed desert; mountain: road mountain)")
    ap.add_argument("--midflight", type=float, default=None,
                    help="with --scen-v2: share of mid-flight starts (default rl_eval_set.V2, 0.3; user 2026-11-18: 0 = launches only for now)")
    ap.add_argument("--reuse", type=int, default=4, help="episodes per scenario drawn from the pool, each with a new start (user 2026-11-18: 2)")
    ap.add_argument("--pool-order", default=None, choices=("random", "slot"),
                    help="slot (default with --scen-v2, user 2026-11-18): env j of a map's n envs takes the pool files j, j + n, ... in seed order, "
                         "then from the start again, so no scenario is drawn twice before the pool is used up; random: drawn with repeats")
    ap.add_argument("--ring-stages", default="",
                    help="launch-distance curriculum (user 2026-11-18), km from the LKP per stage, e.g. 10-10,10-20,10-40; with --ring-eval the next "
                         "stage when the policy reaches --ring-pass x SAROPS on the stage's validation cases, else when the launch flights' POS stops "
                         "rising (--ring-block, --ring-rise, --ring-min); after --ring-max steps in any case")
    ap.add_argument("--ring-eval", default="",
                    help="validation split per launch-distance stage but the last, e.g. v2j1,v2j2 (the v2j cases launched 10 km / 10-20 km away, user "
                         "2026-11-18): flown every --ring-eval-every steps; the stage moves on when the mean of its last 3 ratios to SAROPS reaches "
                         "--ring-pass (the SAROPS references must be made first)")
    ap.add_argument("--ring-eval-every", type=int, default=100_000, help="steps between the launch-distance stage's evaluations")
    ap.add_argument("--ring-pass", type=float, default=0.9, help="ratio to SAROPS (last-3 mean) that moves the stage on (user 2026-11-18: 0.9)")
    ap.add_argument("--ring-block", type=int, default=500_000, help="steps per block of the launch POS (recorded; the stage rule without --ring-eval)")
    ap.add_argument("--ring-rise", type=float, default=0.02,
                    help="without --ring-eval: next stage when the last block's mean launch POS is less than this much (relative) above the block before")
    ap.add_argument("--ring-min", type=int, default=1_000_000, help="without --ring-eval: steps a launch-distance stage lasts at least")
    ap.add_argument("--ring-max", type=int, default=1_500_000,
                    help="steps a launch-distance stage lasts at most, then the next stage anyway (user 2026-11-18: 1.5M)")
    ap.add_argument("--ring-stage", type=int, default=None, help="put the run in this launch-distance stage (0 = first) now, new run or --resume")
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
    if args.scen_v2:
        args.multi = True                                   # the V2 pools are typed: <type>_s<seed>.pkl
        if not args.no_area:
            ap.error("--scen-v2: the scenario pools have no A; add --no-area --a-weight 0")
        if args.stage < 3 and not args.resume:
            ap.error("--scen-v2 flights start at the launch (or mid-flight): add --stage 3")
    if args.ring_stages:
        try:
            parse_ring(args.ring_stages)
        except ValueError:
            ap.error("--ring-stages: km ranges lo-hi separated by commas, e.g. 10-10,10-20,10-40")
    ring_evals = [x.strip() for x in args.ring_eval.split(",") if x.strip()]
    if ring_evals:
        from rl_search.rl_eval_set import SEED_BASE
        if not args.ring_stages or len(ring_evals) != len(parse_ring(args.ring_stages)) - 1:
            ap.error("--ring-eval: one validation split per launch-distance stage but the last (--ring-stages)")
        if any(x not in SEED_BASE for x in ring_evals):
            ap.error(f"--ring-eval: unknown split in {ring_evals}")
    # R16 (user 2026-11-23): one UAV in the multi-UAV env (--teams 1 1 ..., design m2), validated as R15 was: the periodic and launch-stage
    # evaluations fly the model alone in RealRLEnv with the no-teammate channels (rl_real_eval.SoloMates; same flight as MultiRealEnv k=1)
    solo = bool(args.teams) and set(args.teams) == {1}
    if args.teams and not solo and ring_evals:
        ap.error("--ring-eval flies the model alone: only with one-UAV worlds (--teams 1 1 ...)")
    if solo and (args.sequential or args.ma_design != "m2"):
        ap.error("one-UAV worlds are validated alone with the m2 no-teammate inputs: design m2, no --sequential")
    if args.pool_order is None:
        args.pool_order = "slot" if args.scen_v2 else "random"

    from multiprocessing import get_context
    from rl_search.rl_real import preload
    out = args.out/args.tag
    (out/"ckpt").mkdir(parents=True, exist_ok=True)
    state_path = out/"state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else \
        dict(stage=args.stage, stage_start=0, history={}, best={}, switches=[])
    for region in args.regions:                          # terrain and road network, shared by every forked process
        preload(region)
    from rl_search.scripts.rl_eval.sarops_ref import DEFS
    area_over = ({} if args.area_def == "window" else dict(DEFS[args.area_def])) | ({"blur_tau": args.blur_tau} if args.blur_tau else {}) \
        | ({"area_roll_max": args.area_roll_max} if args.area_def == "realtime_t" else {}) | ({"area_w": args.pos_w} if args.pos_w != 1 else {}) \
        | ({"blur_dead_roll": args.blur_dead_roll} if args.blur_tau and args.blur_dead_roll else {}) \
        | ({"blur_tau_low": args.blur_tau_low} if args.blur_tau and args.blur_dead_roll and args.blur_tau_low else {}) \
        | ({"area_sigma": args.area_sigma} if args.area_sigma and args.area_sigma != 150 else {}) \
        | ({"area_sigma_off": args.area_sigma_off} if args.area_sigma_off else {}) \
        | ({"handover_km": args.handover_km} if args.handover_km else {}) | ({"edge_mode": "none", "off_map_zero": True} if args.no_edge else {}) \
        | ({"area_img_min": args.area_img_min} if args.area_img_min else {}) | ({"big_map_m": args.big_map_km*1000.0} if args.big_map_km else {}) \
        | ({"extra_map_m": args.extra_map_km*1000.0, "extra_map_lead_s": args.extra_map_lead} if args.extra_map_km else {}) \
        | ({"rel_maps": True} if args.rel_maps else {}) | ({"fp_rate": True} if args.fp_rate else {}) \
        | ({"a_weight": args.a_weight} if args.a_weight != 1.0 else {}) | ({"pos_norm": True} if args.pos_norm else {}) \
        | ({"look_w": args.look_w} if args.look_w else {}) | ({"no_stop": True} if args.no_stop else {}) \
        | ({"no_area": True} if args.no_area else {}) | ({"obs_r10": True} if args.obs_r10 else {}) \
        | ({"maps7": True} if args.maps7 else {}) | ({"no_vis_obs": True} if args.no_vis_obs else {}) \
        | ({"launch_policy": True} if args.launch_policy else {}) | ({"obs_u8": True} if args.obs_u8 else {}) \
        | ({"scen_v2": True} if args.scen_v2 else {}) | ({"types": args.types} if args.types else {}) \
        | ({"midflight": args.midflight} if args.midflight is not None else {}) \
        | ({"route_var": True} if args.route_var else {})
    from rl_search.rl_real import run_types, terrain_kind
    all_types = list(dict.fromkeys(t for region in args.regions for t in run_types(area_over, region)))   # target types of the run's maps
    if ring_evals or args.eval_split:                       # SAROPS references of the evaluations (the reference set of these settings)
        from rl_search.scripts.rl_eval.sarops_ref import ref_key_of, ref_path, settings as ref_settings
        key = ref_key_of(ref_settings() | area_over)
        for sp in ring_evals:
            if not ref_path(sp, key).exists():
                raise SystemExit(f"no SAROPS reference {ref_path(sp, key)} for the launch-distance stage switch: make it first with "
                                 f"python -m rl_search.scripts.rl_eval.sarops_ref --split {sp} --stages 3 and this run's A / POD settings")
        for sp in args.eval_split:
            if not ref_path(sp, key).exists():
                print(f"warning: no SAROPS reference {ref_path(sp, key)}: the evaluations of {sp} keep RL scores only (no best checkpoint)",
                      flush=True)
    if args.multi:
        from pathlib import Path as _P
        from rl_search.rl_real import pool_dir_multi
        for region in args.regions:
            d = _P(pool_dir_multi(area_over, region))
            if args.scen_v2:                                  # every target type of this map needs its scenarios (<type>_s<seed>.pkl)
                for t in run_types(area_over, region):
                    if not any(d.glob(f"{t}_s*.pkl")):
                        kind = f" --kind mountain:{t}" if terrain_kind(region) == "mountain" else ""
                        raise SystemExit(f"no {t} scenarios at {d}: build them with make_pool --v2{kind} --region {region} --n <count>")
                continue
            if not any(d.glob("*_s*.pkl")):
                raise SystemExit(f"no pool at {pool_dir_multi(area_over, region)}: build it with make_pool --multi --region {region}"
                                 + (" --no-stop --no-area" + (" --route-var" if args.route_var else "") if args.no_stop else
                                    f" --area-sigma-off {args.area_sigma_off:g}" if args.area_sigma_off else ""))
    from rl_search.rl_eval_set import NOSTOP_SPLITS
    if args.no_stop and any(sp not in NOSTOP_SPLITS for sp in args.eval_split):
        raise SystemExit("--no-stop trains on targets that never stop: validate on them too (--eval-split nval [bval])")

    from rl_search.rl_real_eval import eval_job
    pool = get_context("fork").Pool(args.eval_workers)          # forked before the env workers and before CUDA is touched
    from stable_baselines3.common.vec_env import SubprocVecEnv
    tw_path = out/"type_weights.json"
    if args.multi:
        fixed = {k: float(v) for k, v in (kv.split("=") for kv in args.type_fixed.split(",") if kv)}     # e.g. desert=0.5 (user 2026-11-13)
        rest = [t for t in all_types if t not in fixed]
        state.setdefault("type_weights", {t: fixed.get(t, (1 - sum(fixed.values()))/max(len(rest), 1)) for t in all_types})
        state["type_fixed"] = fixed
        tw_path.write_text(json.dumps(state["type_weights"]))
    nreg = len(args.regions)                         # env i is on regions[i % nreg]; pool slot (j, n) among that map's envs

    def slot(i: int):
        return (i//nreg, len(range(i % nreg, args.n_envs, nreg))) if args.pool_order == "slot" else None
    env_fns = [make_env(state["stage"], args.seed*1000 + i, area_over, args.straight_bonus, args.roll_reversal_cost,
                        args.roll_reversal_deg, args.diverse, args.mask_cost, args.straight_raw, args.multi,
                        str(tw_path) if args.multi else None, args.regions[i % nreg], args.reuse, slot(i)) for i in range(args.n_envs)]
    if args.teams and args.obs_u8:
        import inspect
        from rl_search import rl_multi
        if "maps_to_u8" not in inspect.getsource(rl_multi):    # (2026-11-23) the teammate channels must be converted with the maps
            raise SystemExit("--obs-u8 with --teams: rl_multi.MultiRealEnv.observation must build on observation_raw() and convert the "
                             "maps once at the end with rl_real.maps_to_u8 (not yet)")
    if args.teams:                                   # multi-UAV: worlds of k UAVs, every UAV one slot (rl_search.rl_multi_vec)
        from rl_search.rl_multi_vec import MultiWorldVecEnv

        def world_fn(j: int, k: int):
            def f():
                from rl_search.rl_multi import MultiRealEnv
                from rl_search.rl_real import RealRLConfig, pool_dir_multi
                region = args.regions[j % nreg]
                cfg = RealRLConfig(region=region, stage=3, settings=dict(area_over), multi=args.multi,
                                   pool=pool_dir_multi(area_over, region) if args.multi else None, reuse=args.reuse,
                                   type_weights=str(tw_path) if args.multi else None, mask_cost=args.mask_cost,
                                   pool_slot=(j//nreg, len(range(j % nreg, len(args.teams), nreg))) if args.pool_order == "slot" else None)
                return MultiRealEnv(k, cfg, seed=args.seed*1000 + j, design=args.ma_design, sequential=args.sequential)
            return f
        venv = MultiWorldVecEnv([world_fn(j, k) for j, k in enumerate(args.teams)], worlds_per_proc=args.worlds_per_proc)
        args.n_envs = venv.num_envs
        print(f"multi-UAV: worlds {args.teams}, {venv.num_envs} UAV slots", flush=True)
    elif args.envs_per_proc > 1:                     # several envs per worker process (rl_search.vec_group, user 2026-11-12)
        from rl_search.vec_group import GroupedSubprocVecEnv
        venv = GroupedSubprocVecEnv(env_fns, args.envs_per_proc, start_method="fork")
    else:
        venv = SubprocVecEnv(env_fns, start_method="fork")
    if args.pool_order == "slot":                    # a resumed run goes on where each env stopped in its share of the pool
        cur = state.get("pool_cursor") or []
        if len(cur) == args.n_envs:
            for i, c in enumerate(cur):
                venv.env_method("set_pool_cursor", c, indices=[i])
            print(f"pool order: slot, going on from {sum(sum(c.values()) for c in cur)} scenarios drawn", flush=True)
        else:
            print("pool order: slot (env j of n takes the pool files j, j + n, ... in seed order)", flush=True)
    # launch-distance curriculum (user 2026-11-18): stage k launches rings[k] m from the LKP; the stage and its POS blocks live in state.json
    rings = parse_ring(args.ring_stages) if args.ring_stages else []
    if rings:
        R = state.setdefault("ring", dict(stage=0, start=0, blocks=[], switches=[]))
        if args.ring_stage is not None:
            R["stage"], R["start"] = int(args.ring_stage), None          # the start step is set when training starts
        R["stage"] = min(max(int(R["stage"]), 0), len(rings) - 1)
        R["stages_km"] = args.ring_stages
        venv.env_method("set_ring", *rings[R["stage"]])
        print(f"launch ring stage {R['stage']}: {rings[R['stage']][0]/1000:g}-{rings[R['stage']][1]/1000:g} km from the LKP", flush=True)

    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from rl_search.rl_gym import RealMapExtractor
    if args.sequential and not args.teams:
        raise SystemExit("--sequential needs --teams (multi-UAV worlds)")
    if args.sequential:                                   # sequential decisions (user 11-23): rollouts collected turn by turn
        from rl_search.seq_ppo import SeqPPO as PPO       # noqa: N814 (the same PPO update; collect_rollouts per turn)
    policy_cls = "MultiInputPolicy"
    if args.sym_policy:                                   # left-right symmetric policy (user 11-23, rl_search/sym_policy.py)
        from rl_search.sym_policy import SymmetricMultiInputPolicy
        policy_cls = SymmetricMultiInputPolicy
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
        if args.sym_policy:
            over["policy_class"] = policy_cls                # the same weights in the symmetric policy
        old_space = PPO.load(args.init, device="cpu").observation_space
        if old_space == venv.observation_space:
            model = PPO.load(args.init, env=venv, device=device, custom_objects=over)
        else:                                      # more map channels (--rel-maps): copy the weights, new input channels start at zero
            from rl_search.rl_gym import load_expanded
            model, grown = load_expanded(args.init, venv, device, policy=policy_cls, algo=PPO, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                                         n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip,
                                         ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                                         tensorboard_log=str(out/"tb"), seed=args.seed, verbose=1)
            print("grown input channels:", "; ".join(grown), flush=True)
        state["init"] = str(args.init)
        print("initialised from", args.init, "stage", state["stage"], flush=True)
        if args.init_std:
            with torch.no_grad():
                old_std = float(model.policy.log_std.exp().mean())
                model.policy.log_std.fill_(float(np.log(args.init_std)))
            state["init_std"] = [old_std, float(args.init_std)]
            print(f"action noise std {old_std:.3f} -> {args.init_std:g} (--init-std)", flush=True)
    else:
        pk = dict(features_extractor_class=RealMapExtractor, share_features_extractor=True, net_arch=dict(pi=[128, 128], vf=[128, 128]),
                  activation_fn=nn.Tanh, log_std_init=args.log_std_init)
        model = PPO(policy_cls, venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
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
            self.ring_pos = []                                   # launch POS of the current block (launch-distance curriculum)
            self.ring_block_start = 0
            self.pending_ring = []                               # evaluations of the launch-distance stage (--ring-eval)
            self.next_ring_eval = None
            self.ma_active, self.ma_values = [], []              # multi-UAV: per step, which slots flew and their values

        def _on_training_start(self):
            self.next_eval = (self.num_timesteps//args.eval_every + 1)*args.eval_every
            if rings:
                if state["ring"].get("start") is None:
                    state["ring"]["start"] = int(self.num_timesteps)
                self.ring_block_start = int(self.num_timesteps)
                self.next_ring_eval = (self.num_timesteps//args.ring_eval_every + 1)*args.ring_eval_every

        def _ring_collect(self):
            """Finished launch flights of the current launch-distance stage (launched inside its ring) go into the stage's POS block."""
            lo, hi = rings[state["ring"]["stage"]]
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep and ep.get("start") == "launch" and lo - 1.0 <= 1000.0*float(ep.get("launch_km", np.nan)) <= hi + 1.0:
                    self.ring_pos.append(float(ep["POS"]))
            if self.num_timesteps - self.ring_block_start >= args.ring_block:
                self._ring_block()

        def _ring_block(self, check: bool = True):
            """Close a block: its mean launch POS (recorded); without --ring-eval, move to the next stage when ring_next says so."""
            R = state["ring"]
            st, step = int(R["stage"]), int(self.num_timesteps)
            mean = float(np.mean(self.ring_pos)) if self.ring_pos else float("nan")
            R["blocks"].append(dict(step=step, stage=st, pos=mean, n=len(self.ring_pos)))
            tbe.add_scalar("ring/block_pos", mean, step)
            tbe.add_scalar("ring/block_n", len(self.ring_pos), step)
            print(f"launch ring stage {st} block at {step:,}: launch POS {mean:.4f} over {len(self.ring_pos)} flights", flush=True)
            self.ring_pos, self.ring_block_start = [], step
            if check and not ring_evals and st + 1 < len(rings):
                means = [b["pos"] for b in R["blocks"] if b["stage"] == st and np.isfinite(b["pos"])]
                why = ring_next(means, step - int(R["start"]), args.ring_rise, args.ring_min, args.ring_max)
                if why:
                    self._ring_switch(why)
            tbe.flush()
            state_path.write_text(json.dumps(state, indent=1))

        def _ring_switch(self, why: str):
            """Next launch-distance stage now: every env launches in its ring from its next episode (episodes under way keep their launch)."""
            R = state["ring"]
            st, step = int(R["stage"]), int(self.num_timesteps)
            if self.ring_pos:                                    # the block so far belongs to the stage it was flown in
                self._ring_block(check=False)
            R["switches"].append(dict(step=step, from_stage=st, to_stage=st + 1, why=why))
            R["stage"], R["start"] = st + 1, step
            self.training_env.env_method("set_ring", *rings[st + 1])
            lo, hi = rings[st + 1]
            print(f"launch ring: stage {st} -> {st + 1} ({lo/1000:g}-{hi/1000:g} km) at {step:,} ({why})", flush=True)
            tbe.add_scalar("ring/stage", R["stage"], step)
            tbe.flush()
            state_path.write_text(json.dumps(state, indent=1))

        def _submit_ring(self):
            """Fly the current launch-distance stage's validation cases (--ring-eval) with the policy now, for the stage switch."""
            from rl_search.rl_eval_set import MULTI_N
            st, step = int(state["ring"]["stage"]), int(self.num_timesteps)
            split = ring_evals[st]
            path = out/"ckpt"/f"ring_{step}.zip"
            self.model.save(path)
            jobs = [(str(path), split, 3, k, f"{args.tag}-ring-{step}", False, area_over) for k in range(MULTI_N.get(split, 3))]
            self.pending_ring.append((step, st, split, path, pool.map_async(eval_job, jobs)))

        def _record_ring(self, step, st, split, rows):
            """A launch-distance stage evaluation: ratio to SAROPS; the stage moves on when its last 3 ratios average --ring-pass."""
            R = state["ring"]
            rl = np.array([r["holdout"]["total"] for r in rows])
            if any(r["sarops"] is None for r in rows):
                print(f"launch ring stage {st} eval at {step:,} ({split}): RL {rl.mean():.3f}, no SAROPS reference", flush=True)
                return
            sar = np.array([r["sarops"]["total"] for r in rows])
            ratio = float(rl.mean()/sar.mean())
            R.setdefault("evals", []).append(dict(step=step, stage=st, split=split, ratio=ratio, rl=rl.tolist(), sarops=sar.tolist()))
            rs = [e["ratio"] for e in R["evals"] if e["stage"] == st]
            m3 = float(np.mean(rs[-3:]))
            tbe.add_scalar("ring/ratio_to_sarops", ratio, step)
            tbe.add_scalar("ring/ratio_last3", m3, step)
            tbe.flush()
            print(f"launch ring stage {st} eval at {step:,} ({split}): RL {rl.mean():.3f} vs SAROPS {sar.mean():.3f} (ratio {ratio:.3f}, "
                  f"last-3 mean {m3:.3f}), per case {np.round(rl, 3).tolist()}", flush=True)
            if st == int(R["stage"]) and st + 1 < len(rings) and len(rs) >= 3 and m3 >= args.ring_pass:
                self._ring_switch(f"last-3 mean {m3:.3f} >= {args.ring_pass:g} x SAROPS on {split}")
            state_path.write_text(json.dumps(state, indent=1))

        def _multi_rollout_end(self):
            """Multi-UAV: mask the steps of UAVs not flying (rl_multi_vec.mask_inactive), and record how the shared critic does: the spread of
            V(o_i) among the flying UAVs of a world at the same step (relative to their mean |V|), the explained variance per team size
            (single-UAV R15 about 0.97), team POS / filter cuts / violations / time within 1 km per team size."""
            from rl_search.rl_multi_vec import mask_inactive
            buf = self.model.rollout_buffer
            act = np.array(self.ma_active[-buf.buffer_size:], bool)
            vals = np.array(self.ma_values[-buf.buffer_size:])
            self.ma_active, self.ma_values = [], []
            if act.shape != buf.advantages.shape:
                return
            masked = mask_inactive(buf, act)
            self.logger.record("multi/masked_share", masked)
            self._act_watch(buf, act)
            off = np.concatenate([[0], np.cumsum(args.teams)])
            spread = []
            for w in range(len(args.teams)):
                v, a = vals[:, off[w]:off[w + 1]], act[:, off[w]:off[w + 1]]
                for t in range(len(v)):
                    if a[t].sum() >= 2:
                        x = v[t][a[t]]
                        spread.append(float(np.std(x)/max(np.mean(np.abs(x)), 1e-6)))
            if spread:
                self.logger.record("multi/v_spread", float(np.mean(spread)))
            ret, val = buf.returns, buf.values
            for k in sorted(set(args.teams)):
                cols = np.concatenate([np.arange(off[w], off[w + 1]) for w in range(len(args.teams)) if args.teams[w] == k])
                r, v, a = ret[:, cols].ravel(), val[:, cols].ravel(), act[:, cols].ravel()
                if a.sum() > 10 and np.var(r[a]) > 0:
                    self.logger.record(f"multi/explained_var_k{k}", float(1 - np.var(r[a] - v[a])/np.var(r[a])))
            eps = list(self.model.ep_info_buffer)
            for k in sorted(set(args.teams)):
                e = [x for x in eps if x.get("team_k") == k]
                if e:
                    for key in ("POS", "cuts", "viol", "close1km", "follow", "own_min", "pen_share", "cut_share", "look_share"):
                        vals = [x[key] for x in e if key in x]
                        if vals:
                            self.logger.record(f"team_k{k}/{key}", float(np.mean(vals)))

        def _act_watch(self, buf, act):
            """(user 11-23, after M2 collapsed into a saturated hard turn) the policy's mean action over this rollout's flying steps:
            multi/act_mean, act_absmean, act_plus (share > 0), act_sat (share with |mean| > 1, beyond the action range: the samples are all
            clipped there, so nothing in the env depends on the mean and nothing pulls it back). --sat-stop x: save and stop after two
            rollouts in a row with act_sat > x."""
            import torch
            a = act.reshape(-1)
            if not a.any():
                return
            obs = {k: v.reshape(-1, *v.shape[2:])[a] for k, v in buf.observations.items()}
            n = len(obs["vec"])
            mus = []
            with torch.no_grad():
                for s in range(0, n, 512):
                    ot, _ = self.model.policy.obs_to_tensor({k: v[s:s + 512] for k, v in obs.items()})
                    mus.append(self.model.policy.get_distribution(ot).distribution.mean.cpu().numpy().ravel())
            mu = np.concatenate(mus)
            sat = float((np.abs(mu) > 1.0).mean())
            for key, v in (("act_mean", float(mu.mean())), ("act_absmean", float(np.abs(mu).mean())), ("act_plus", float((mu > 0).mean())),
                           ("act_sat", sat)):
                self.logger.record(f"multi/{key}", v)
            if args.sat_stop:
                self.sat_streak = getattr(self, "sat_streak", 0) + 1 if sat > args.sat_stop else 0
                if self.sat_streak >= 2:
                    path = out/"ckpt"/f"model_{int(self.num_timesteps)}_saturated.zip"
                    self.model.save(path)
                    print(f"STOP (--sat-stop {args.sat_stop:g}): {sat*100:.0f}% of the policy means beyond the action range two rollouts in a "
                          f"row (mean {mu.mean():+.2f}); saved {path.name}", flush=True)
                    self.sat_stop_now = True

        def _value_warmup(self):
            """--value-warmup N (user 11-23): the first N updates train the value head only (the CNN is shared with the policy, so it is
            frozen too): the reward form and gamma changed from the run the weights come from. Counted over the whole run (state.json)."""
            n = int(state.get("value_warmup_done", 0))
            frozen = n < args.value_warmup
            for name, prm in self.model.policy.named_parameters():
                if not (name.startswith("mlp_extractor.value_net") or name.startswith("value_net")):
                    prm.requires_grad_(not frozen)
            if frozen:
                state["value_warmup_done"] = n + 1
                self.logger.record("multi/value_warmup", 1.0)
                print(f"value warm-up update {n + 1}/{args.value_warmup}: policy and CNN frozen", flush=True)
            elif args.value_warmup and n == args.value_warmup and not state.get("value_warmup_over"):
                state["value_warmup_over"] = int(self.num_timesteps)
                print(f"value warm-up over at {self.num_timesteps:,} steps: policy trains", flush=True)

        def _on_rollout_end(self):
            if args.value_warmup:
                self._value_warmup()
            if args.teams:
                self._multi_rollout_end()
            eps = list(self.model.ep_info_buffer)
            for key in ("A", "POS", "total", "bonus", "roll_rev", "mask_cut", "look"):
                vals = [e[key] for e in eps if key in e]
                if vals:
                    self.logger.record(f"episode/{key}", float(np.mean(vals)))
            km = [e["launch_km"] for e in eps if np.isfinite(e.get("launch_km", np.nan))]
            if km:
                self.logger.record("episode/launch_km", float(np.mean(km)))
            if rings:
                lo, hi = rings[state["ring"]["stage"]]
                self.logger.record("ring/stage", state["ring"]["stage"])
                self.logger.record("ring/lo_km", lo/1000.0)
                self.logger.record("ring/hi_km", hi/1000.0)
            if args.multi:                                       # training episodes per target type
                for t in all_types:
                    vals = [e["total"] for e in eps if e.get("typ") == t]
                    if vals:
                        self.logger.record(f"episode_type/{t}", float(np.mean(vals)))
            for s in sorted({e.get("start") for e in eps if e.get("start")}):     # launches and each kind of mid-flight start (stage 4)
                vals = [e["POS"] for e in eps if e.get("start") == s]
                self.logger.record(f"episode_start/{s}", float(np.mean(vals)))
                self.logger.record(f"episode_start_n/{s}", len(vals))
            if len(args.regions) > 1:                            # training episodes per map
                for region in args.regions:
                    vals = [e["total"] for e in eps if e.get("region") == region]
                    if vals:
                        self.logger.record(f"episode_region/{region}", float(np.mean(vals)))
                for t, v in state["type_weights"].items():
                    self.logger.record(f"type_share/{t}", float(v))
            self.logger.record("curriculum/stage", state["stage"])

        def _on_step(self) -> bool:
            if args.teams:                                       # which UAVs were flying this step, and their values V(o_i)
                self.ma_active.append([bool(i.get("active", True)) for i in self.locals.get("infos", [])])
                v = self.locals.get("values")
                self.ma_values.append(np.ravel(v.detach().cpu().numpy()) if v is not None else np.zeros(len(self.ma_active[-1])))
            if rings:
                self._ring_collect()
                R = state["ring"]
                if R["stage"] + 1 < len(rings) and self.num_timesteps - int(R["start"]) >= args.ring_max:
                    self._ring_switch(f"stage budget {args.ring_max:,} steps spent")
                if ring_evals and R["stage"] + 1 < len(rings) and self.num_timesteps >= self.next_ring_eval:
                    self._submit_ring()
                if self.next_ring_eval is not None and self.num_timesteps >= self.next_ring_eval:
                    self.next_ring_eval += args.ring_eval_every
            self._collect()
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                self._submit()
            if self.num_timesteps >= args.stop_at:
                return False
            if getattr(self, "sat_stop_now", False):            # --sat-stop: the policy mean left the action range (see _act_watch)
                return False
            return True

        def _submit(self):
            step = int(self.num_timesteps)
            path = out/"ckpt"/f"model_{step}.zip"
            self.model.save(path)
            if args.pool_order == "slot":                        # with the checkpoint: how far each env is in its share of the pool
                state["pool_cursor"] = self.training_env.env_method("pool_cursor")
            st = state["stage"]
            if args.teams and not solo:                          # multi-UAV: checkpoint only (scripts/multi_design/ma_eval_watch flies it)
                state_path.write_text(json.dumps(state, indent=1))
                print(f"checkpoint {path.name} (multi-UAV: no periodic evaluation yet)", flush=True)
                return
            label = f"{args.tag}-{step}" if (args.init or state.get("init") or args.multi) else step   # web run names stay apart between runs
            from rl_search.rl_eval_set import MULTI_N, case_type
            # --save-every (user 2026-11-12, disk): the checkpoint and the web runs are kept only when a multiple of it was passed;
            # the evaluation itself still runs every --eval-every (it also drives the curriculum and the type shares)
            keep = not args.save_every or step//args.save_every > (step - args.eval_every)//args.save_every
            jobs = [(str(path), sp, st, k, label, bool(keep), area_over) for sp in args.eval_split for k in range(MULTI_N.get(sp, 5))
                    if not args.eval_types or case_type(sp, k) in args.eval_types]
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
            pending_ring = []
            for step, st, split, path, res in self.pending_ring:
                if not (wait or res.ready()):
                    pending_ring.append((step, st, split, path, res))
                    continue
                try:
                    self._record_ring(step, st, split, res.get())
                except Exception as exc:                 # a failed evaluation must not stop the training
                    print(f"launch ring evaluation at {step} failed: {exc!r}", flush=True)
                Path(path).unlink(missing_ok=True)
            self.pending_ring = pending_ring

        def _record(self, step, st, path, rows):
            rl = np.array([r["holdout"]["total"] for r in rows])
            if any(r["sarops"] is None for r in rows):            # SAROPS references of this split / A not made: keep the scores only
                print(f"eval stage {st} at {step:,}: RL {rl.mean():.3f} (no SAROPS reference), per case {np.round(rl, 3).tolist()}", flush=True)
                return
            sar = np.array([r["sarops"]["total"] for r in rows])
            ratio = float(rl.mean()/sar.mean())
            hist = state["history"].setdefault(str(st), [])
            splits = [r["split"] for r in rows]
            by_split = {sp: float(rl[[i for i, x in enumerate(splits) if x == sp]].mean()/sar[[i for i, x in enumerate(splits) if x == sp]].mean())
                        for sp in args.eval_split}
            hist.append(dict(step=step, ratio=ratio, rl=rl.tolist(), sarops=sar.tolist(), pos=[r["holdout"]["pos"] for r in rows],
                             A=[r["holdout"]["A"] for r in rows], belief=[r["belief"]["total"] for r in rows], split=splits, k=[r["k"] for r in rows],
                             by_split=by_split))
            tbe.add_scalar(f"eval_s{st}/ratio_to_sarops", ratio, step)
            tbe.add_scalar(f"eval_s{st}/rl_total", float(rl.mean()), step)
            tbe.add_scalar(f"eval_s{st}/rl_pos", float(np.mean(hist[-1]["pos"])), step)
            tbe.add_scalar(f"eval_s{st}/rl_A", float(np.mean(hist[-1]["A"])), step)
            tbe.add_scalar(f"eval_s{st}/paired_diff", float((rl - sar).mean()), step)
            for i, r in enumerate(rows):
                tbe.add_scalar(f"eval_s{st}/{r['split']}_case{r['k'] + 1}", float(rl[i]), step)
            if len(by_split) > 1:
                for sp, v in by_split.items():
                    tbe.add_scalar(f"eval_s{st}/ratio_{sp}", v, step)
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
            if len(by_split) > 1:
                print("  by split (ratio to SAROPS): " + ", ".join(f"{sp} {v:.3f}" for sp, v in by_split.items()), flush=True)
            if args.multi and len(all_types) > 1:                # one type (user 2026-11-18: road only): no shares to move
                self._type_shares(step, rows, rl, sar, hist)
            if st == state["stage"]:
                self._maybe_advance(hist)
            state_path.write_text(json.dumps(state, indent=1))

        def _type_shares(self, step, rows, rl, sar, hist):
            """Typed training (user 2026-11-12: equal shares, then a little more of the type that does worst): per-type ratio to SAROPS on
            the validation cases (case k has type k % 3), smoothed over the last two evaluations; from the third evaluation on, the type
            furthest behind gets +0.05 share, taken from the type furthest ahead among those still above the 0.2 floor (shares 0.2-0.5)."""
            from rl_search.rl_eval_set import case_type
            types = all_types
            per = {}
            for t in types:
                idx = [i for i, r in enumerate(rows) if case_type(r["split"], r["k"]) == t]
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
