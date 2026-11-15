#!/bin/bash
# R10 comparison baselines (docs/r10_ubuntu.md step 4, deferred while r10_2map trained): R9 400만 and R8 300만 on nval, R9 400만 on bval
cd /home/lee/fixed_wing_search
source envs/activate_train.sh
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
OLD="--handover-km 5 --no-edge --a-weight 0 --pos-norm --fp-rate --workers ${WORKERS:-3}"
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r9/ckpt/model_4000000.zip --label amboy_r9-4000000 --split nval --stages 3 $DEF $OLD
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r9/ckpt/model_4000000.zip --label amboy_r9-4000000 --split bval --stages 3 $DEF $OLD
python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_r8/ckpt/model_3000000.zip --label amboy_r8-3000000 --split nval --stages 3 $DEF $OLD
echo BASELINES DONE
