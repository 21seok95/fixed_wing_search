#!/bin/bash
# R11 (user 2026-11-14): R10 two maps with less throttled PPO updates — minibatch 256 -> 770 (3,080 decisions = 4 equal minibatches; 256
# left an 8-sample minibatch every epoch) and target_kl 0.02 -> 0.03 (stop at 0.045). Same 4M R10 model as the start (--init), 2M steps, so it
# is compared with the R10 run continued from 4M to 6M with the old settings. Starts when that run (unit rl-r10) has ended.
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
while systemctl --user is-active --quiet rl-r10; do sleep 30; done
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R10="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.01 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10"
N=${N_ENVS:-20}; P=${ENVS_PER_PROC:-2}; S=$(( (3070 + N - 1)/N ))
echo "start $(date) envs $N per proc $P n_steps $S batch ${BATCH:-770} target_kl ${KL:-0.03}" >> rl_search/data/rl_real/r11_train.log
exec python -m rl_search.scripts.rl_eval.train_rl --tag r11_2map --stage 3 --n-envs $N --envs-per-proc $P --n-steps $S \
  --batch-size ${BATCH:-770} --target-kl ${KL:-0.03} --init rl_search/data/rl_real/r10_2map/ckpt/model_4000000.zip \
  --eval-workers 2 --eval-every 400000 --save-every 500000 --stop-at ${STOP:-2000000} --device cuda --mask-cost 0.1 --multi \
  --eval-split nval bval --regions amboy mojave --route-var $DEF $R10 >> rl_search/data/rl_real/r11_train.log 2>&1
