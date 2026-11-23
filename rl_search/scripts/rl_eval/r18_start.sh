#!/bin/bash
# R18 (user 2026-11-24, local PC: 384 cores, about 230 GB RAM, two L40S): one shared policy for teams of 1-4 UAVs, from scratch.
# - worlds of a fixed team size, mixed 8 : 6 : 5 : 4 (1 / 2 / 3 / 4 UAVs; samples about 15 / 22 / 27 / 30 %; user: teams of every size will
#   be used): SCALE x (8, 6, 5, 4) worlds (SCALE 10 = 230 worlds, 510 UAV slots); M1..M4 set the counts directly
# - every team size walks the whole scenario pool (--pool-by-team, --reuse 1: a scenario is flown once by each team size per pass, the
#   reuse across team sizes saves pool; user 11-24), pool 15,000 Chungju 120 km road-type scenarios (seeds 14,000,000 + i)
# - R16's training otherwise: launches only, launch-distance stages 10 -> 10-20 -> 10-40 km passed at 0.9 x SAROPS on v2j1 / v2j2 (the policy
#   flown alone: one-UAV worlds are among the worlds, so that is a situation it trains on), Jirisan v2j validation, 7 maps, no visibility in
#   the observation, policy from the launch, R13's detection / reward / PPO settings, gamma 0.995, 8-bit maps, the symmetric policy
# - design m3 (rl_multi.DESIGNS; transit / search collision rules, the "step" danger channel, penalty size 1), launches 5-10 s apart
# - update: 154 steps per slot as R16 (n_steps x slots decisions, e.g. 510 x 154 = 78,540), minibatch 4,096 (about 19 per epoch, 5 epochs:
#   as many gradient steps per update as R16's 12 x 5), learning rate 1e-4 and KL limit 0.02 as R16; intervals in updates (--in-updates):
#   launch-stage evaluation every 32 (R16 100k / 3,080), Jirisan validation and checkpoint every 130 (R16 400k), stage budget 487 (1.5M),
#   2,000 updates (R16's 6M = about 1,950)
# Team flights (1-4 UAVs, every checkpoint):
#   nohup python -m rl_search.scripts.multi_design.ma_eval_watch --run rl_search/data/rl_real/r18_mix --prefix r18 > rl_search/data/rl_real/r18_eval_watch.log 2>&1 &
# Check after the first checkpoint (all 7 lines PASS):
#   python -m rl_search.scripts.multi_design.sym_policy_check exact --model rl_search/data/rl_real/r18_mix/ckpt/model_<step>.zip
#   start:  nohup bash rl_search/scripts/rl_eval/r18_start.sh > /dev/null 2>&1 &       resume: RESUME=1 ...      longer: STOP=<updates>
#   SCALE / M1..M4 / W (worlds per process, default 1) / GPU (default 0) / EXTRA (more train_rl options, later ones win) / TAG
cd "$(dirname "$0")/../../.."
OS=$(uname)
TAG=${TAG:-r18_mix}
SCALE=${SCALE:-10}
M1=${M1:-$((8*SCALE))}; M2=${M2:-$((6*SCALE))}; M3=${M3:-$((5*SCALE))}; M4=${M4:-$((4*SCALE))}
W=${W:-1}
TEAMS=""
for k in 1 2 3 4; do n=$(eval echo \$M$k); for ((i = 0; i < n; i++)); do TEAMS="$TEAMS $k"; done; done
if [ "$OS" = "Darwin" ]; then DEV=mps; KEEP="caffeinate -i"; else DEV=cuda; KEEP="systemd-inhibit --what=idle:sleep --who=$TAG --why=training"
  export CUDA_VISIBLE_DEVICES=${GPU:-0}; fi
LOG=rl_search/data/rl_real/${TAG}_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 1 --pool-order slot --pool-by-team --maps7 --no-vis-obs --launch-policy \
   --ring-stages 10-10,10-20,10-40 --ring-eval v2j1,v2j2 --ring-pass 0.9"
U="--in-updates --ring-eval-every 32 --eval-every 130 --save-every 130 --ring-block 162 --ring-min 325 --ring-max 487 --stop-at ${STOP:-2000}"
M="--teams$TEAMS --worlds-per-proc $W --balance-worlds --ma-design m3 --sym-policy --gamma 0.995"
echo "$(date '+%m-%d %H:%M') $TAG training ${RESUME:+resume}start ($OS, $DEV; worlds 1/2/3/4 UAVs: $M1 / $M2 / $M3 / $M4, $W per process)" >> $LOG
PYTHONPATH=$PWD nice -n 10 $KEEP .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag $TAG --stage 3 \
  --n-steps 154 --batch-size 4096 --target-kl 0.02 --eval-split v2j --eval-workers 4 --device $DEV \
  ${RESUME:+--resume latest} $M $A $U $DEF $R13 --obs-u8 $EXTRA >> $LOG 2>&1
echo "$(date '+%m-%d %H:%M') $TAG training ended (exit $?)" >> $LOG
