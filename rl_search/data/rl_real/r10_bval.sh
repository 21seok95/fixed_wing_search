#!/bin/bash
# Barstow SAROPS references right after the Amboy ones (needed by the in-training validation); the R9 / R8 baselines come later.
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
DEF="--area-def realtime_t --area-roll-max 5 --blur-tau 3 --blur-dead-roll 5 --blur-tau-low 15 --area-sigma 75 --area-sigma-off 150 --area-img-min 0.75 --pos-w 1.5"
while systemctl --user is-active --quiet sarops-nval; do sleep 15; done
python -m rl_search.scripts.rl_eval.sarops_ref --split bval --stages 3 --workers 4 $DEF --a-weight 0 > rl_search/data/real/mojave/rl_eval_bval.log 2>&1
