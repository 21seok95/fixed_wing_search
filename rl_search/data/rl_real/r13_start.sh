#!/bin/bash
# R13 (user 2026-11-16): R10 two maps (targets that never stop, R10 channels, no A; PPO as R10: minibatch 256, target_kl 0.02) plus
#  - a fifth observation map of 102.4 km (--extra-map-km 102.4, lead 600 s, the same channels, 64 x 64 cells of 1.6 km): in the R12
#    validation flights up to 29 % of the remaining belief was outside the 51.2 km map when the UAV was near the terrain edge (road and
#    mixed cases, second half of the flight: 45-60 min mean 1.8 %, a quarter of the minutes over 1 %);
#  - the look reward A' at 0.1 (was 0.01): R12 circles near the end of flights (33 % of the minutes after 45 min) where little is left
#    nearby; A' does not shrink as POS is found and pays only for gentle passes (circling faster than 1.49 deg/s earns none).
# From scratch (the network has a fifth map branch), 6M steps, compared with R12 at the same steps. Run with the web server off (R12's
# first start was killed by systemd-oomd at an evaluation with the web on).
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.1 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10 --extra-map-km 102.4 --extra-map-lead 600"
N=${N_ENVS:-20}; P=${ENVS_PER_PROC:-2}; S=$(( (3070 + N - 1)/N ))
echo "start $(date) envs $N per proc $P n_steps $S batch ${BATCH:-256} target_kl ${KL:-0.02}" >> rl_search/data/rl_real/r13_train.log
exec python -m rl_search.scripts.rl_eval.train_rl --tag r13_2map --stage 3 --n-envs $N --envs-per-proc $P --n-steps $S \
  --batch-size ${BATCH:-256} --target-kl ${KL:-0.02} \
  --eval-workers ${EVAL_WORKERS:-2} --eval-every 400000 --save-every 500000 --stop-at ${STOP:-6000000} --device cuda --mask-cost 0.1 --multi \
  --eval-split nval bval --regions amboy mojave --route-var $DEF $R13 $EXTRA >> rl_search/data/rl_real/r13_train.log 2>&1
