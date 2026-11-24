#!/bin/bash
# R17 (user 2026-11-24, the second step of "처음부터 다시"): two UAVs per world, shared weights, from scratch, R16's training otherwise.
# - as R15 / R16: Chungju 120 km road type (4,000 scenarios in seed order, reused 2x), launches only, launch-distance stages 10 -> 10-20 ->
#   10-40 km, 7 maps, no visibility in the observation, policy from the launch, R13's detection / reward / PPO settings, gamma 0.995, 8-bit
#   maps, the left-right symmetric policy (--sym-policy), 20 scenarios at once and 3,080 decisions per update, 6M steps
# - new: 20 two-UAV worlds (40 UAV slots x 77 steps = 3,080 per update), 2 per process; rl_multi.MultiRealEnv k = 2, design m3 (reward
#   "team_rel" = (team found + 0.25 x found by the UAV) / belief left before the step; launches 5-10 s apart from one base; in transit m2's
#   rules, searching (its 6.4 km map holds >= 1 % of the belief) never within 500 m, penalty from 1 km with the "step" level x 1, 35 s
#   look-ahead, the danger channel drawn with the same level; teammate channels live, 1 / k = 0.5)
# - the launch stages move on at R16's steps (--ring-max 310000: 10-20 km at about 310k, 10-40 km at about 620k; R16 passed its 0.9 x SAROPS
#   rule at 320k / 619k); no validation flown alone (user 11-24: a team model's team-size input would be never seen or untrue); checkpoints
#   every 400k, the team flights scored by
#     nohup .venv/bin/python -m rl_search.scripts.multi_design.ma_eval_watch --run rl_search/data/rl_real/r17_chungju --prefix r17 --teams 2 \
#       --every 400000 > rl_search/data/rl_real/r17_eval_watch.log 2>&1 &
#   (baseline: R16 600만 two UAVs on their own inputs under m3, ma_eval label r16_6M_m3solo: 0.530 / 0.544 / 0.323)
# Check after the first checkpoint (all 7 lines PASS):
#   python -m rl_search.scripts.multi_design.sym_policy_check exact --model rl_search/data/rl_real/r17_chungju/ckpt/model_<step>.zip
#
# Mac: MPS, caffeinate while training, `pmset sleepnow` at the end (NOSLEEP=1: stay awake). Ubuntu PC: CUDA, systemd-inhibit, run name
# r17u_chungju, no suspend unless SUSPEND=1. SYM=0: without the symmetric policy (run name + _nosym).
#   start:  nohup bash rl_search/data/rl_real/r17_start.sh > /dev/null 2>&1 &      resume: RESUME=1 ...      longer: STOP=...
#   EXTRA="...": more train_rl options (later ones win), e.g. a smoke test: TAG=r17_smoke STOP=6160 NOSLEEP=1 bash rl_search/data/rl_real/r17_start.sh
cd "$(dirname "$0")/../../.."
OS=$(uname)
if [ "$OS" = "Darwin" ]; then TAG=${TAG:-r17_chungju}; DEV=mps; KEEP="caffeinate -i"; else TAG=${TAG:-r17u_chungju}; DEV=cuda
  KEEP="systemd-inhibit --what=idle:sleep --who=$TAG --why=training"; fi
# (Ubuntu PC 2026-11-24: polkit refuses systemd-inhibit there, "Access denied", and auto-suspend is off on AC: then run without it)
if [ "$OS" != "Darwin" ] && ! $KEEP true 2>/dev/null; then KEEP=""; fi
SYMF="--sym-policy"
if [ "$SYM" = "0" ]; then SYMF=""; TAG=${TAG}_nosym; fi
LOG=rl_search/data/rl_real/${TAG}_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 2 --pool-order slot --maps7 --no-vis-obs --launch-policy \
   --ring-stages 10-10,10-20,10-40 --ring-max 310000"
M="--teams 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --worlds-per-proc 2 --ma-design m3 $SYMF --gamma 0.995"
echo "$(date '+%m-%d %H:%M') $TAG training ${RESUME:+resume}start ($OS, $DEV)" >> $LOG
PYTHONPATH=$PWD nice -n 10 $KEEP .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag $TAG --stage 3 \
  --n-steps 77 --batch-size 256 --target-kl 0.02 \
  --eval-split v2j --eval-every 400000 --save-every 400000 --eval-workers 1 --stop-at ${STOP:-6000000} --device $DEV \
  ${RESUME:+--resume latest} $M $A $DEF $R13 --obs-u8 $EXTRA >> $LOG 2>&1
code=$?
echo "$(date '+%m-%d %H:%M') $TAG training ended (exit $code)" >> $LOG
# the team evaluations of the last checkpoints finish before the Mac sleeps (ma_eval_watch stops once every checkpoint is done; at most 40 min)
for i in $(seq 1 80); do pgrep -f "ma_eval_watch --run rl_search/data/rl_real/$TAG" > /dev/null || break; sleep 30; done
if [ "$OS" = "Darwin" ] && [ -z "$NOSLEEP" ]; then
  echo "$(date '+%m-%d %H:%M') putting the Mac to sleep (pmset sleepnow; user 2026-11-23)" >> $LOG
  pmset sleepnow >> $LOG 2>&1
elif [ "$OS" != "Darwin" ] && [ -n "$SUSPEND" ]; then
  echo "$(date '+%m-%d %H:%M') suspending (systemctl suspend)" >> $LOG
  systemctl suspend >> $LOG 2>&1
fi
