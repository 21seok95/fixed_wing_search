#!/bin/bash
# After R16 without the symmetric policy ends on the Ubuntu PC: commit its weights, TensorBoard, state, config and log, and push to
# feat/rl-env for the Mac (user 2026-11-24: "학습이 모두 완료되면 모델 가중치, 학습 결과(TENSORBOARD)를 깃헙에 자동 푸쉬").
# If the push cannot be rebased onto the remote (someone edited the same lines), the commit goes to branch r16u-nosym-results instead.
cd /home/lee/fixed_wing_search
T=r16u_chungju_nosym; D=rl_search/data/rl_real; L=$D/${T}_train.log; P=$D/r16u_nosym_push.log
until systemctl --user is-active -q rl-r16nosym; do sleep 60; done          # wait for the start (after the prep)
while systemctl --user is-active -q rl-r16nosym; do sleep 120; done          # then for the end
how="finished"; grep -q "^stopped at" $L || how="stopped early"
echo "$(date '+%m-%d %H:%M') training $how; committing" >> $P
source envs/activate_train.sh > /dev/null
summary=$(python - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("rl_search/data/rl_real/r16u_chungju_nosym/state.json").read_text())["history"]["3"]
print(" · ".join(f"{e['step']/1e4:.0f}만 {e['ratio']:.2f}" for e in h))
PY
)
last=$(ls -t $D/$T/ckpt/model_*.zip 2>/dev/null | head -1)
git add -f $D/$T/config.json $D/$T/state.json $D/$T/tb $D/$T/tb_eval $last $D/$T/best_stage3.zip $L $D/r16u_nosym_push.sh $D/r16_start.sh $D/r17_start.sh \
  rl_search/data/real/amboy/rl_eval/report/r16_sym_removed.png docs/rl_experiments.md 2>> $P
git commit -q -m "R16 without the symmetric policy (Ubuntu, r16u_chungju_nosym): $how, weights, TensorBoard, state, log

Jirisan validation (ratio to SAROPS): $summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> $P 2>&1
if git pull -q --rebase --autostash origin feat/rl-env >> $P 2>&1 && git push -q origin HEAD:feat/rl-env >> $P 2>&1; then
  echo "$(date '+%m-%d %H:%M') pushed to feat/rl-env: $(git log --oneline -1)" >> $P
else
  git rebase --abort 2>/dev/null
  git push -q origin HEAD:refs/heads/r16u-nosym-results >> $P 2>&1 && echo "$(date '+%m-%d %H:%M') feat/rl-env push failed; pushed to branch r16u-nosym-results" >> $P \
    || echo "$(date '+%m-%d %H:%M') PUSH FAILED (see above)" >> $P
fi
# power off once the results are on GitHub (user 2026-11-24: "푸쉬까지 성공적으로 되면 pc 전원 종료"); needs the sudoers line
# "lee ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff" (polkit asks for a password from a background service); otherwise the PC stays on
if tail -1 $P | grep -q "pushed to"; then
  echo "$(date '+%m-%d %H:%M') powering off in 2 min" >> $P; sleep 120
  sudo -n /usr/bin/systemctl poweroff >> $P 2>&1 || echo "$(date '+%m-%d %H:%M') power-off not allowed (sudoers line missing): PC left on" >> $P
fi
