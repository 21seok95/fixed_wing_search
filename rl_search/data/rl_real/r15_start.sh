#!/bin/bash
# R15 = 학습 A (user 2026-11-18, docs/scenario_redesign.md "학습 A: 충주 120 km 도로형, 이륙만, 이륙 거리 단계"):
# Chungju 120 km road type (4,000 scenarios, each env its own share in seed order, reused 2x), launches only (--midflight 0),
# launch-distance curriculum 10 -> 10-20 -> 10-40 km (next stage at 0.9 x SAROPS, last-3 mean, on the Jirisan stage cases v2j1 / v2j2
# flown every 100k steps; at most 1.5M steps per stage), 7 observation maps, no visibility in the observation, policy from the launch,
# Jirisan validation v2j every 400k. R13's detection, reward and PPO settings (minus --handover-km, --extra-map-km, the old pool flags).
# Mac (MPS), 20 envs (2 per process). Resume: RESUME=1 nohup bash rl_search/data/rl_real/r15_start.sh &
# Train on past 6M (user 11-19: more training after 6M is possible): STOP=8000000 RESUME=1 nohup bash rl_search/data/rl_real/r15_start.sh &
# (the step count, the launch-distance stage, the pool cursor and the best model carry on; past ~6M each env starts its share of the pool again)
cd /Users/lee/PycharmProjects/fixed_wing_search
LOG=rl_search/data/rl_real/r15_train.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--no-edge --a-weight 0 --fp-rate --look-w 0.1 --ent-coef 0.01 --no-area --obs-r10 --mask-cost 0.1"
A="--regions chungju120 --scen-v2 --types road --midflight 0 --reuse 2 --pool-order slot --maps7 --no-vis-obs --launch-policy \
   --ring-stages 10-10,10-20,10-40 --ring-eval v2j1,v2j2 --ring-eval-every 100000 --ring-pass 0.9 --ring-max 1500000"
echo "$(date '+%m-%d %H:%M') R15 training ${RESUME:+resume}start" >> $LOG
PYTHONPATH=$PWD exec nice -n 10 caffeinate -i .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag r15_chungju --stage 3 \
  --n-envs 20 --envs-per-proc 2 --n-steps 154 --batch-size 256 --target-kl 0.02 \
  --eval-split v2j --eval-every 400000 --save-every 400000 --eval-workers 2 --stop-at ${STOP:-6000000} --device mps \
  ${RESUME:+--resume latest} $A $DEF $R13 >> $LOG 2>&1
