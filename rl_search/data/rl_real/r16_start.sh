#!/bin/bash
# R16 (user 2026-11-23: "처음부터 다시" -- the multi-UAV env (design m2) with one UAV, from scratch, then 2 -> 3 UAVs): R15's training A
# (Chungju 120 km road type, 4,000 scenarios in seed order, reused 2x, launches only, launch-distance stages 10 -> 10-20 -> 10-40 km passed at
# 0.9 x SAROPS on v2j1 / v2j2 every 100k, at most 1.5M per stage, 7 maps, no visibility in the observation, policy from the launch, Jirisan
# v2j every 400k, R13's detection / reward / PPO settings, gamma 0.995) in rl_multi.MultiRealEnv with k = 1 (reward "team_rel" = (found +
# 0.25 x found) / belief left before the step, the teammate channels at their "no teammate" values, 1 / k = 1 in the vector, 8-bit maps) and
# the left-right symmetric policy (--sym-policy, rl_search/sym_policy.py). 20 one-UAV worlds, 2 per process (as R15's 20 envs).
#
# Mac (user 11-23: run here, then the Mac sleeps): MPS, caffeinate while training, `pmset sleepnow` at the end (NOSLEEP=1: stay awake).
# Ubuntu PC (docs/r10_ubuntu.md section 11): CUDA, systemd-inhibit while training, no suspend unless SUSPEND=1. Its run is named r16u_chungju
# so the two PCs' runs never share a name.
# SYM=0: without the symmetric policy (run name + _nosym; e.g. a comparison run on the other PC).
# Check after the first checkpoint (all 7 lines PASS, only with the symmetric policy):
#   python -m rl_search.scripts.multi_design.sym_policy_check exact --model rl_search/data/rl_real/<TAG>/ckpt/model_<step>.zip
#   start:  nohup bash rl_search/data/rl_real/r16_start.sh > /dev/null 2>&1 &      resume: RESUME=1 ...      longer: STOP=...
#   EXTRA="...": more train_rl options (later ones win), e.g. a smoke test:
#   TAG=r16u_smoke STOP=4096 NOSLEEP=1 EXTRA="--eval-every 2048 --save-every 2048 --ring-eval-every 2048" bash rl_search/data/rl_real/r16_start.sh
cd "$(dirname "$0")/../../.."
OS=$(uname)
if [ "$OS" = "Darwin" ]; then TAG=${TAG:-r16_chungju}; DEV=mps; KEEP="caffeinate -i"; else TAG=${TAG:-r16u_chungju}; DEV=cuda
  KEEP="systemd-inhibit --what=idle:sleep --who=$TAG --why=training"; fi
# (Ubuntu PC 2026-11-24: polkit refuses systemd-inhibit there, "Access denied", and auto-suspend is off on AC: then run without it)
if [ "$OS" != "Darwin" ] && ! $KEEP true 2>/dev/null; then KEEP=""; fi
SYMF="--sym-policy"
if [ "$SYM" = "0" ]; then SYMF=""; TAG=${TAG}_nosym; fi
LOG=rl_search/data/rl_real/${TAG}_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 2 --pool-order slot --maps7 --no-vis-obs --launch-policy \
   --ring-stages 10-10,10-20,10-40 --ring-eval v2j1,v2j2 --ring-eval-every 100000 --ring-pass 0.9 --ring-max 1500000"
M="--teams 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --worlds-per-proc 2 --ma-design m2 $SYMF --gamma 0.995"
echo "$(date '+%m-%d %H:%M') $TAG training ${RESUME:+resume}start ($OS, $DEV)" >> $LOG
PYTHONPATH=$PWD nice -n 10 $KEEP .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag $TAG --stage 3 \
  --n-steps 154 --batch-size 256 --target-kl 0.02 \
  --eval-split v2j --eval-every 400000 --save-every 400000 --eval-workers 2 --stop-at ${STOP:-6000000} --device $DEV \
  ${RESUME:+--resume latest} $M $A $DEF $R13 --obs-u8 $EXTRA >> $LOG 2>&1
code=$?
echo "$(date '+%m-%d %H:%M') $TAG training ended (exit $code)" >> $LOG
if [ "$OS" = "Darwin" ] && [ -z "$NOSLEEP" ]; then
  echo "$(date '+%m-%d %H:%M') putting the Mac to sleep (pmset sleepnow; user 2026-11-23)" >> $LOG
  pmset sleepnow >> $LOG 2>&1
elif [ "$OS" != "Darwin" ] && [ -n "$SUSPEND" ]; then
  echo "$(date '+%m-%d %H:%M') suspending (systemctl suspend)" >> $LOG
  systemctl suspend >> $LOG 2>&1
fi
