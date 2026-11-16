#!/bin/bash
# R13 on the Mac (user 2026-11-16: "여기서 돌려보자. 학습 파라미터는 r10으로"): R10 two maps (PPO minibatch 256, target_kl 0.02, the rest
# as R10-R12) + a fifth 102.4 km map + A' 0.1, from scratch, 6M steps. 10 envs x 307 steps = 3,070 decisions per update, the same update
# size as R10's 20 x 154 on the Ubuntu PC (the Mac has 16 GB). The two-map pools are built first (1000 per map, 3 workers each); the
# training starts once each map has 90 scenarios and re-reads the pool folders every 10 min (as R10 did).
cd /Users/lee/PycharmProjects/fixed_wing_search
LOG=rl_search/data/rl_real/r13_mac.log
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
R13="--handover-km 5 --no-edge --a-weight 0 --fp-rate --look-w 0.1 --type-fixed desert=0.5 --ent-coef 0.01 --no-stop --no-area --obs-r10 --extra-map-km 102.4 --extra-map-lead 600"
echo "$(date '+%m-%d %H:%M') pools start" >> $LOG
for r in amboy mojave; do
  PYTHONPATH=$PWD nohup nice -n 5 caffeinate -i .venv/bin/python -m rl_search.scripts.rl_eval.make_pool --multi --no-stop --no-area --route-var \
    --n 1000 --workers 3 --region $r >> rl_search/data/rl_real/r13_pool_$r.log 2>&1 &
done
until [ "$(ls rl_search/data/real/amboy/rl_pool_multi_ns_rt 2>/dev/null | grep -c pkl)" -ge 90 ] && \
      [ "$(ls rl_search/data/real/mojave/rl_pool_multi_ns_rt 2>/dev/null | grep -c pkl)" -ge 90 ]; do sleep 30; done
echo "$(date '+%m-%d %H:%M') training start" >> $LOG
PYTHONPATH=$PWD exec nice -n 10 caffeinate -i .venv/bin/python -m rl_search.scripts.rl_eval.train_rl --tag r13_2map --stage 3 --n-envs 10 --envs-per-proc 2 \
  --n-steps 307 --batch-size 256 --target-kl 0.02 --eval-workers 1 --eval-every 400000 --save-every 500000 --stop-at 6000000 --device mps \
  --mask-cost 0.1 --multi --eval-split nval bval --regions amboy mojave --route-var $DEF $R13 >> $LOG 2>&1
