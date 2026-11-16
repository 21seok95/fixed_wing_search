#!/bin/bash
# R14 (user 2026-11-16 evening): type specialists on the Ubuntu PC. Changed the same evening (user): R13's settings, the run on the local PC
# (R10 two maps + a fifth 102.4 km map, lead 600 s + A' 0.1; minibatch 256, target_kl 0.02, 20 envs x 154), not R10's, and both types
# within about 10 h on this PC: 3.2M steps each instead of 4M (R12 took 1.33 h per million with 4 maps) and the evaluation every 400k
# flies only the 10 validation cases of the trained type (--eval-types; all 30 took ~20 of every ~30 min of CPU). Only the type shares
# differ from R13: road only (r14_road), then desert only (r14_desert); compared with R13 (local PC, export zip) at the same steps on the
# cases of that type.
# Question: R12 tells the types apart (probe on unseen flights 0.94, type_views.py), yet on road cases it circles in 41 % of the minutes after
# 45 min with 1/7 of the POS per minute while 14 % of the belief is left 3-15 km away. Is that one model learning three types (then a
# road-only model circles less) or the late phase itself (then it circles too)? Desert: does a desert-only model raise desert POS?
# Usage: bash rl_search/data/rl_real/r14_start.sh [road|desert|both]   (default both: road first, then desert)
#        resume a stopped run:  EXTRA="--resume latest" bash rl_search/data/rl_real/r14_start.sh road
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
BASE="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-stop --no-area --obs-r10 --extra-map-km 102.4 --extra-map-lead 600"   # R13 without its type shares
N=${N_ENVS:-20}; P=${ENVS_PER_PROC:-2}; S=$(( (3070 + N - 1)/N ))
run() {
  local typ=$1 shares=$2 log=rl_search/data/rl_real/r14_${1}_train.log
  echo "start $(date) r14_$typ envs $N per proc $P n_steps $S batch ${BATCH:-256} target_kl ${KL:-0.02} type shares $shares" >> $log
  python -m rl_search.scripts.rl_eval.train_rl --tag r14_$typ --stage 3 --n-envs $N --envs-per-proc $P --n-steps $S \
    --batch-size ${BATCH:-256} --target-kl ${KL:-0.02} \
    --eval-workers ${EVAL_WORKERS:-2} --eval-every 400000 --save-every 500000 --stop-at ${STOP:-3200000} --device cuda --mask-cost 0.1 --multi \
    --eval-split nval bval --regions amboy mojave --route-var $DEF $BASE --type-fixed $shares --eval-types $typ $EXTRA >> $log 2>&1
  echo "end $(date) r14_$typ exit $?" >> $log
}
case "${1:-both}" in
  road) run road road=1,mixed=0,desert=0 ;;
  desert) run desert desert=1,road=0,mixed=0 ;;
  both) run road road=1,mixed=0,desert=0; run desert desert=1,road=0,mixed=0 ;;
  *) echo "usage: $0 [road|desert|both]"; exit 1 ;;
esac
