#!/bin/bash
# R14 watchdog (2026-11-16, overnight): if the rl-r14 unit dies before both runs reach their stop step (systemd-oomd killed R12's first
# start), restart it from the last kept checkpoint (--resume latest) and go on with the desert run; at most 3 restarts.
cd /home/lee/fixed_wing_search
D=rl_search/data/rl_real; n=0
done_() { grep -q "^stopped at [0-9,]* steps" $D/r14_$1_train.log 2>/dev/null; }   # printed only when a run reaches its stop step
while :; do
  sleep 60
  systemctl --user is-active -q rl-r14 && continue
  done_ road && done_ desert && { echo "$(date +%H:%M) both done"; exit 0; }
  [ $n -ge 3 ] && { echo "$(date +%H:%M) rl-r14 down, 3 restarts used: stop"; exit 1; }
  n=$((n + 1)); systemctl --user reset-failed rl-r14 2>/dev/null
  road=""; done_ road || road='EXTRA="--resume latest" bash rl_search/data/rl_real/r14_start.sh road;'
  if ls $D/r14_desert/ckpt/model_*.zip >/dev/null 2>&1; then des='EXTRA="--resume latest" bash rl_search/data/rl_real/r14_start.sh desert'
  else des='bash rl_search/data/rl_real/r14_start.sh desert'; fi
  echo "$(date +%H:%M) rl-r14 down (restart $n): $road $des"
  systemd-run --user --unit=rl-r14 --working-directory=$PWD bash -c "$road $des"
done
