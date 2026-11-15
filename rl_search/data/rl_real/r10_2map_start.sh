#!/bin/bash
# R10 on two maps (2026-11-14): starts once the Barstow SAROPS references are made; the pools keep growing while it trains.
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
while systemctl --user is-active --quiet sarops-bval; do sleep 15; done
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R10="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.01 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10"
N=${N_ENVS:-20}; P=${ENVS_PER_PROC:-2}; S=$(( (3070 + N - 1)/N ))
echo "start $(date) envs $N per proc $P n_steps $S" >> rl_search/data/rl_real/r10_2map_train.log
exec python -m rl_search.scripts.rl_eval.train_rl --tag r10_2map --stage 3 --n-envs $N --envs-per-proc $P --n-steps $S \
  --eval-workers 2 --eval-every 400000 --save-every 500000 --stop-at 4000000 --device cuda --mask-cost 0.1 --multi \
  --eval-split nval bval --regions amboy mojave --route-var $DEF $R10 $EXTRA >> rl_search/data/rl_real/r10_2map_train.log 2>&1
