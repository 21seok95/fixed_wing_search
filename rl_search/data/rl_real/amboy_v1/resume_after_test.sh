#!/bin/bash
# Resume amboy_v1 after the SAROPS test references and the RL test evaluation finish (they share the CPU and memory).
cd /home/lee/fixed_wing_search
source envs/activate_train.sh >/dev/null 2>&1
while systemctl --user is-active --quiet sarops-test || systemctl --user is-active --quiet rl-test-eval; do sleep 30; done
echo "resume $(date)" >> rl_search/data/rl_real/amboy_v1/train.log
python -m rl_search.scripts.rl_eval.train_rl --tag amboy_v1 --resume latest --stop-at 6308456 >> rl_search/data/rl_real/amboy_v1/train.log 2>&1
echo "train exit $? $(date)" >> rl_search/data/rl_real/amboy_v1/train.log
