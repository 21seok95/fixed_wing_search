#!/bin/bash
# M2c (user 2026-11-23, after M2 / M2b: a left-right drift of the mean action, fewer scenarios per update than R15): 3-UAV worlds only,
# 20 of them at once (R15 learned from 20 scenarios at once; M2 / M2b from 9), 4 worlds per worker process (5 processes; possible since
# the visibility-grid memory fix in rl_real, 2026-11-23: a 3-UAV world 2.4 -> 0.8 GB), 57 steps per slot per update (60 slots x 57 =
# 3,420 steps per update, as M2's 3,078). From R15 6.8M (clean start) with the action noise std 0.3, the first 5 updates value head only.
# Reward as M2 / M2b: design m2 ("team_rel": (team found + 0.25 x found by the UAV) / belief left before the step), gamma 0.995 (as M2b),
# 8-bit maps, the vector with 1 / team size; the mean action watched with an automatic stop above 20 % saturated (--sat-stop 0.2).
#   start:  nohup bash rl_search/data/rl_real/m2c_start.sh &     resume: RESUME=1 nohup bash ... &     longer: STOP=...
cd /Users/lee/PycharmProjects/fixed_wing_search
TAG=m2c_multi     # fixed: a test run never takes another run's name (2026-11-23 incident)
LOG=rl_search/data/rl_real/${TAG}_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 2 --pool-order slot --maps7 --no-vis-obs --launch-policy"
M="--teams 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --worlds-per-proc 4 --ma-design m2 --gamma 0.995 --value-warmup 5 --sat-stop 0.2"
INIT=rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip
if [ -n "$RESUME" ]; then START="--resume latest"; else START="--init $INIT --init-std 0.3"; fi
echo "$(date '+%m-%d %H:%M') $TAG training ${RESUME:+resume}start" >> $LOG
PYTHONPATH=$PWD exec nice -n 10 caffeinate -i .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag $TAG --stage 3 \
  --n-steps 57 --batch-size 256 --target-kl 0.02 \
  --eval-split v2j --eval-every 200000 --save-every 200000 --eval-workers 1 --stop-at ${STOP:-3000000} --device mps \
  $START $M $A $DEF $R13 --obs-u8 >> $LOG 2>&1
