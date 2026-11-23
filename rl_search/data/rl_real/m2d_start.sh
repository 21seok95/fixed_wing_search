#!/bin/bash
# M2d (user 2026-11-23, prepared; the start model is the user's choice): M2c's setting (3-UAV worlds x 20, 4 per process, design m2, gamma 0.995,
# 8-bit maps, --sat-stop 0.2) plus
#   - sequential decisions (--sequential, rl_multi sequential=True, rl_search/seq_ppo.py): within a 5 s step the UAVs of a world decide one
#     after another in a random order; each sees the belief after the detections of the ones before it and where they flew; one more vector
#     entry (its turn, 0 first .. 1 last; its input weights start at 0)
#   - the left-right symmetric policy (--sym-policy, rl_search/sym_policy.py): mean(mirror(o)) = -mean(o), value(mirror(o)) = value(o)
# The first 5 updates value head only (the symmetric value and the new input differ from what the weights were trained with).
#   start:  INIT=<checkpoint> nohup bash rl_search/data/rl_real/m2d_start.sh &     (STD=0.3 sets the action noise; STD= keeps the model's)
#   resume: RESUME=1 nohup bash ... &     longer: STOP=...
cd /Users/lee/PycharmProjects/fixed_wing_search
TAG=m2d_multi     # fixed: a test run never takes another run's name (2026-11-23 incident)
LOG=rl_search/data/rl_real/${TAG}_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 2 --pool-order slot --maps7 --no-vis-obs --launch-policy"
M="--teams 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --worlds-per-proc 4 --ma-design m2 --gamma 0.995 --value-warmup 5 --sat-stop 0.2 --sequential --sym-policy"
if [ -n "$RESUME" ]; then
  START="--resume latest"
else
  [ -z "$INIT" ] && { echo "INIT=<checkpoint> is required (the start model is the user's choice)"; exit 1; }
  STD=${STD-0.3}
  START="--init $INIT ${STD:+--init-std $STD}"
fi
echo "$(date '+%m-%d %H:%M') $TAG training ${RESUME:+resume}start ${INIT:+from $INIT}" >> $LOG
PYTHONPATH=$PWD exec nice -n 10 caffeinate -i .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag $TAG --stage 3 \
  --n-steps 57 --batch-size 256 --target-kl 0.02 \
  --eval-split v2j --eval-every 200000 --save-every 200000 --eval-workers 1 --stop-at ${STOP:-3000000} --device mps \
  $START $M $A $DEF $R13 --obs-u8 >> $LOG 2>&1
