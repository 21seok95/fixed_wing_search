#!/bin/bash
# R10 preparation queue (2026-11-14): after the nval SAROPS references, the Barstow references and the R9 / R8 baselines.
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
OLD="--handover-km 5 --no-edge --a-weight 0 --pos-norm --fp-rate"
LOG=rl_search/data/rl_real/r10_prep.log
while systemctl --user is-active --quiet sarops-nval; do sleep 30; done
echo "bval SAROPS $(date)" >> $LOG
python -m rl_search.scripts.rl_eval.sarops_ref --split bval --stages 3 --workers 3 $DEF --a-weight 0 >> $LOG 2>&1
for M in "amboy_r9 4000000" "amboy_r8 3000000"; do
  set -- $M
  echo "$1 nval $(date)" >> $LOG
  python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/$1/ckpt/model_$2.zip --label $1-$2 --split nval --stages 3 $DEF $OLD --workers 3 >> $LOG 2>&1
done
echo "amboy_r9 bval $(date)" >> $LOG
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r9/ckpt/model_4000000.zip --label amboy_r9-4000000 --split bval --stages 3 $DEF $OLD --workers 3 >> $LOG 2>&1
echo "done $(date)" >> $LOG
