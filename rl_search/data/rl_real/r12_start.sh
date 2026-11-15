#!/bin/bash
# R12 (user 2026-11-14 allowed it if R11 helped): R10 two maps from scratch with the R11 PPO settings — minibatch 770 (3,080 decisions = 4
# equal minibatches) and target_kl 0.03. R11 (same settings from the R10 4M model) beat the R10 continuation at the same steps
# (+0.009 +- 0.003 POS per case over 5 evaluations, 21/30; 1.279 vs 1.236 at 6M). 6M steps, compared with R10 two maps step by step.
# Run with the web server off: the first start was killed by systemd-oomd at the 400k evaluation (web on, 02:58). One eval worker was
# tried next (03:01) but an evaluation then took ~43 min, longer than 400k steps (~28 min), so evaluations piled up: back to two.
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R10="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.01 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10"
N=${N_ENVS:-20}; P=${ENVS_PER_PROC:-2}; S=$(( (3070 + N - 1)/N ))
echo "start $(date) envs $N per proc $P n_steps $S batch ${BATCH:-770} target_kl ${KL:-0.03}" >> rl_search/data/rl_real/r12_train.log
exec python -m rl_search.scripts.rl_eval.train_rl --tag r12_2map --stage 3 --n-envs $N --envs-per-proc $P --n-steps $S \
  --batch-size ${BATCH:-770} --target-kl ${KL:-0.03} \
  --eval-workers ${EVAL_WORKERS:-2} --eval-every 400000 --save-every 500000 --stop-at ${STOP:-6000000} --device cuda --mask-cost 0.1 --multi \
  --eval-split nval bval --regions amboy mojave --route-var $DEF $R10 $EXTRA >> rl_search/data/rl_real/r12_train.log 2>&1
